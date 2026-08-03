//! Shipped kernel against tile kernel, one container and one set of clocks.
//!
//! Every family here is bandwidth-shaped, so the number that decides a port is
//! effective bytes per second and not FLOP/s: each row reports the traffic its
//! kernel is obliged to move (compulsory reads plus writes, no reuse assumed)
//! divided by the measured time. A tile version that moves the same bytes
//! slower has lost, whatever it does to the instruction count.
//!
//! Shapes are `gpu/model/src/bin/train.rs`'s: `N = 24576` rows of `D = 3072`
//! for the norms.
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
use device::{NORM_THREADS, NORM_TILE_BLOCK_ROWS, NORM_TILE_THREADS, kernels};

/// Rows the norms are timed at — the training config's `B * T`.
const N: usize = 24_576;
/// Model width, the norms' row length.
const D: usize = 3_072;

const WARMUP: usize = 3;
const ITERS: usize = 20;

fn row_grid(rows: usize, threads: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn tile_grid(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((rows / NORM_TILE_BLOCK_ROWS) as u32, 1, 1),
        block_dim: (NORM_TILE_THREADS as u32, 1, 1),
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
        if tile_ms < shipped_ms { "tile wins" } else { "tile loses" }
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    bench_rms_norm(&stream, &module)?;
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

    let element = 4.0;
    let row = (N * D) as f64 * element;

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
                tile_grid(N),
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
    let tile = time_gpu_iters(stream, WARMUP, ITERS, || {
        // SAFETY: as the forward tile launch, plus `inv` being N long.
        unsafe {
            module.rms_norm_backward_x_tile(
                stream,
                tile_grid(N),
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
    report("rms_norm backward_x", 5.0 * row, shipped, tile);
    Ok(())
}
