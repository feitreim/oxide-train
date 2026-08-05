//! shapes, in one container and on one set of clocks.
//!
//! All of them are bandwidth-shaped, so the number is effective bytes per
//! second and not FLOP/s: each row reports the traffic its kernel is obliged to
//! move — compulsory reads plus writes, no reuse assumed — divided by the
//! measured time. A kernel that moves the same bytes slower has lost, whatever
//! it does to the instruction count, and that is how #70 and #78 decided which
//! families reached tiles: the ones that won are the ones with two rows.
//!
//! Shapes are `gpu/model/src/bin/train.rs`'s: `N = 24576` rows, `D = 3072`
//! model width, `FF = 4096` FFN width, `VP = 50432` padded vocabulary.
//!
//!     modal run modal_app.py --kernel ops --bin bench

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};

// `cargo oxide` embeds the CUDA artifact into the selected binary target, so
// this binary includes the canonical kernel source as a module (as main.rs
// does) instead of importing the library crate.
#[path = "../lib.rs"]
#[allow(dead_code)]
mod device;
use device::{
    CLASSIFIER_THREADS, NORM_THREADS, NORM_TILE_BLOCK_ROWS, NORM_TILE_THREADS,
    SWIGLU_TILE_BLOCK_ROWS, SWIGLU_TILE_THREADS, kernels, rope_table,
};

/// Rows the norms are timed at — the training config's `B * T`.
const N: usize = 24_576;
/// Model width, the norms' row length.
const D: usize = 3_072;
/// Real vocabulary, and the padded row stride the lm-head writes.
const VOCAB: usize = 50_257;
const VP: usize = 50_432;
/// SwiGLU's row length.
const FF: usize = 4_096;

const WARMUP: usize = 3;
const ITERS: usize = 20;

fn row_grid(rows: usize, threads: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn norm_tile_grid(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((rows / NORM_TILE_BLOCK_ROWS) as u32, 1, 1),
        block_dim: (NORM_TILE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn swiglu_tile_grid(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((rows / SWIGLU_TILE_BLOCK_ROWS) as u32, 1, 1),
        block_dim: (SWIGLU_TILE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One family's two timings, printed as the ratio that decides the port.
fn report(family: &str, bytes: f64, shipped_ms: f64, tile_ms: f64) {
    let gbs = |ms: f64| bytes / (ms / 1_000.0) / 1e9;
    println!("{family}  ({:.0} MB of compulsory traffic)", bytes / 1e6);
    println!(
        "  shipped: {shipped_ms:8.4} ms  {:8.1} GB/s",
        gbs(shipped_ms)
    );
    println!("  tile:    {tile_ms:8.4} ms  {:8.1} GB/s", gbs(tile_ms));
    println!(
        "  speedup: {:.3}x  ({})",
        shipped_ms / tile_ms,
        if tile_ms < shipped_ms {
            "tile wins"
        } else {
            "tile loses"
        }
    );
}

/// A kernel with no tile counterpart: the rate it achieves, and nothing to
/// divide it by. Every family #70 measured and left flat is one of these.
fn report_one(family: &str, bytes: f64, ms: f64) {
    println!("{family}  ({:.0} MB of compulsory traffic)", bytes / 1e6);
    println!(
        "  shipped: {ms:8.4} ms  {:8.1} GB/s",
        bytes / (ms / 1_000.0) / 1e9
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    bench_rms_norm(&stream, &module)?;
    bench_classifier(&stream, &module)?;
    bench_swiglu(&stream, &module)?;
    bench_rope(&stream, &module)?;
    Ok(())
}

/// The fused classifier forward, packed-bf16 logits, at the lm-head's shape.
///
/// No tile row, and the `.local` fix that gave the norm forward one below does
/// not change that: a warp-per-band version re-measured on the fixed pin is
/// 0.442x, and 0.563x with its `right_fill` deleted so it carries no frame at
/// all. What it is short of is CTAs — 24576 rows become 384 blocks against the
/// shipped kernel's 24576 — and no tile shape reaches that (#70, #78).
fn bench_classifier(
    stream: &std::sync::Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    let logits = DeviceBuffer::<u32>::zeroed(stream, N * VP / 2)?;
    let targets = DeviceBuffer::from_host(
        stream,
        &(0..N).map(|row| (row % VOCAB) as u32).collect::<Vec<_>>(),
    )?;
    let mut losses = DeviceBuffer::<f32>::zeroed(stream, N)?;

    let shipped = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: the launch shape is the one the kernel documents, over a
        // N x VP packed buffer whose targets are all inside VOCAB.
        unsafe {
            module.fused_classifier_forward_bf16(
                stream,
                row_grid(N, CLASSIFIER_THREADS),
                &logits,
                &targets,
                N as u32,
                VOCAB as u32,
                VP as u32,
                &mut losses,
            )?
        };
        Ok(())
    })?;
    // One pass over the packed logits; the losses are N floats and round off.
    report_one(
        "fused_classifier forward bf16",
        (N * VP) as f64 * 2.0,
        shipped,
    );
    Ok(())
}

/// The SwiGLU family — the measure-first tier's elementwise case, at the dense
/// FFN's shape.
///
/// The bf16 forward is the MoE panel's arm and rounds on the way out, so it
/// moves half the bytes the fp32 one does; the two backwards read three and two
/// operands respectively. All four are the same walk over the same rectangle.
fn bench_swiglu(
    stream: &std::sync::Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    let len = N * FF;
    let element = 4.0;
    let operand = len as f64 * element;
    let gate = DeviceBuffer::from_host(stream, &uniform_vec(len, 5))?;
    let up = DeviceBuffer::from_host(stream, &uniform_vec(len, 6))?;
    let dy = DeviceBuffer::from_host(stream, &uniform_vec(len, 7))?;
    let mut y = DeviceBuffer::<f32>::zeroed(stream, len)?;
    let mut words = DeviceBuffer::<u32>::zeroed(stream, len / 2)?;

    let flat = LaunchConfig::for_num_elems(len as u32);
    let shipped = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: all three buffers are `len` long and the launch covers
        // exactly that many elements.
        unsafe { module.swiglu_forward(stream, flat, &gate, &up, &mut y)? };
        Ok(())
    })?;
    let tile = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: N divides SWIGLU_TILE_BLOCK_ROWS and FF the tile chunk.
        unsafe {
            module.swiglu_forward_tile(
                stream,
                swiglu_tile_grid(N),
                &gate,
                &up,
                FF as u32,
                &mut y,
            )?
        };
        Ok(())
    })?;
    report("swiglu forward", 3.0 * operand, shipped, tile);

    // The bf16 arm has no tile row: the flat kernel already stores a packed
    // pair per thread, and a tile version of it measured 0.70x (#70).
    let shipped = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: the packed output is one word per input pair.
        unsafe {
            module.swiglu_forward_bf16(
                stream,
                LaunchConfig::for_num_elems((len / 2) as u32),
                &gate,
                &up,
                &mut words,
            )?
        };
        Ok(())
    })?;
    report_one("swiglu forward bf16", 2.5 * operand, shipped);

    let shipped = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: all four buffers are `len` long.
        unsafe { module.swiglu_backward_gate(stream, flat, &gate, &up, &dy, &mut y)? };
        Ok(())
    })?;
    let tile = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: as the fp32 forward, with `dy` the third operand.
        unsafe {
            module.swiglu_backward_gate_tile(
                stream,
                swiglu_tile_grid(N),
                &gate,
                &up,
                &dy,
                FF as u32,
                &mut y,
            )?
        };
        Ok(())
    })?;
    report("swiglu backward_gate", 4.0 * operand, shipped, tile);

    let shipped = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: all three buffers are `len` long.
        unsafe { module.swiglu_backward_up(stream, flat, &gate, &dy, &mut y)? };
        Ok(())
    })?;
    let tile = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: as the fp32 forward, over two operands.
        unsafe {
            module.swiglu_backward_up_tile(
                stream,
                swiglu_tile_grid(N),
                &gate,
                &dy,
                FF as u32,
                &mut y,
            )?
        };
        Ok(())
    })?;
    report("swiglu backward_up", 3.0 * operand, shipped, tile);
    Ok(())
}

/// RoPE forward.
///
/// #70 read this row against SwiGLU's — comparable traffic, one `exp` in it —
/// and concluded the `powf`/`sin`/`cos` *per element* and not the memory was
/// what the kernel waited on. Those angles depend only on `(position, pair)`,
/// so they now come from a 1 MiB host-built table and this row is bandwidth
/// again. The traffic denominator counts only the rotated tensor: the table is
/// re-read `N * H / T` times and lives in L2, so charging it as compulsory
/// would flatter the rate.
fn bench_rope(
    stream: &std::sync::Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const T: usize = 2_048;
    const H: usize = 24;
    const HD: usize = 128;

    let x = DeviceBuffer::from_host(stream, &uniform_vec(N * D, 7))?;
    let table = DeviceBuffer::from_host(stream, &rope_table(T, HD))?;
    let mut y = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let shipped = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: both buffers are N x D, D is H * HD, and the table matches.
        unsafe {
            module.rope_forward(
                stream,
                LaunchConfig::for_num_elems((N * D / 2) as u32),
                &x,
                &table,
                T as u32,
                H as u32,
                HD as u32,
                &mut y,
            )?
        };
        Ok(())
    })?;
    report_one("rope forward", 2.0 * (N * D) as f64 * 4.0, shipped);
    Ok(())
}

fn bench_rms_norm(
    stream: &std::sync::Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const EPS: f32 = 1e-5;

    let x = DeviceBuffer::from_host(stream, &uniform_vec(N * D, 1))?;
    let weight = DeviceBuffer::from_host(stream, &uniform_vec(D, 2))?;
    let dy = DeviceBuffer::from_host(stream, &uniform_vec(N * D, 3))?;
    let mut y = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let mut dx = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let mut inv = DeviceBuffer::<f32>::zeroed(stream, N)?;
    let row = (N * D) as f64 * 4.0;

    // Forward reads x once for the statistic, once for the output, and writes y.
    let shipped = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: the launch shape is the one `rms_norm_forward_fast` documents
        // and every buffer is N x D.
        unsafe {
            module.rms_norm_forward_fast(
                stream,
                row_grid(N, NORM_THREADS),
                &x,
                &weight,
                EPS,
                D as u32,
                &mut y,
            )?
        };
        Ok(())
    })?;
    let tile = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: N is a multiple of NORM_TILE_BLOCK_ROWS and D of the chunk.
        unsafe {
            module.rms_norm_forward_tile(
                stream,
                norm_tile_grid(N),
                &x,
                &weight,
                EPS,
                D as u32,
                &mut y,
            )?
        };
        Ok(())
    })?;
    report("rms_norm forward", 3.0 * row, shipped, tile);

    // Backward reads x and dy twice each and writes dx.
    //
    // No tile row, and the rate directly above is why: the forward's tile
    // collects it to 5184 GB/s and stops there, which is where this kernel
    // already is. A tile version measured 0.985-0.997x over four shapes — it
    // reaches the same roof from the same side, and matching is not winning.
    let shipped = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: as the forward launch, plus `inv` being N long.
        unsafe {
            module.rms_norm_backward_x_fast(
                stream,
                row_grid(N, NORM_THREADS),
                &x,
                &weight,
                &dy,
                EPS,
                D as u32,
                &mut dx,
                &mut inv,
            )?
        };
        Ok(())
    })?;
    report_one("rms_norm backward_x", 5.0 * row, shipped);
    Ok(())
}
