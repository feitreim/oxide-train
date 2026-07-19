//! Flash attention generations versus the `ops` materialized baseline.
//!
//! The shape matches the training sequence length (T=1024, H=24, HD=64) where
//! the per-row kernels' serial key scan is the measured tail (7e7); `B` is
//! kept small because attention time scales with `B` while its parallelism is
//! already saturated.
//!
//! Run with `./run.sh flash-attn bench`.

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

#[path = "../lib.rs"]
mod flash;
#[path = "../../../ops/src/lib.rs"]
mod naive;

const B: usize = 2;
const T: usize = 1024;
const H: usize = 24;
const HD: usize = 64;
const N: usize = B * T;
const D: usize = H * HD;

fn per_row_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((N * H) as u32, 1, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert!(HD.is_power_of_two() && HD <= flash::MAX_HEAD_DIM);
    assert_eq!(HD, flash::TILE_HD);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let flash_module = flash::kernels::load(&ctx)?;
    let naive_module = naive::kernels::load(&ctx)?;

    let q = DeviceBuffer::from_host(&stream, &uniform_vec(N * D, 81))?;
    let k = DeviceBuffer::from_host(&stream, &uniform_vec(N * D, 82))?;
    let v = DeviceBuffer::from_host(&stream, &uniform_vec(N * D, 83))?;
    let dy = DeviceBuffer::from_host(&stream, &uniform_vec(N * D, 84))?;
    let mut probabilities = DeviceBuffer::<f32>::zeroed(&stream, N * H * T)?;
    let mut y = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut logsumexp = DeviceBuffer::<f32>::zeroed(&stream, N * H)?;
    let mut softmax_dot = DeviceBuffer::<f32>::zeroed(&stream, N * H)?;
    let mut dq = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut dk = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut dv = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;

    let naive_forward_ms = time_gpu_iters(&stream, 1, 5, || {
        // SAFETY: the launch geometry matches `attention_probabilities`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            naive_module.attention_probabilities(
                &stream,
                LaunchConfig::for_num_elems((N * H * T) as u32),
                &q,
                &k,
                T as u32,
                H as u32,
                HD as u32,
                &mut probabilities,
            )
        }?;
        // SAFETY: the launch geometry matches `attention_output`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            naive_module.attention_output(
                &stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                &probabilities,
                &v,
                T as u32,
                H as u32,
                HD as u32,
                &mut y,
            )
        }?;
        Ok(())
    })?;
    let per_row_forward_ms = time_gpu_iters(&stream, 2, 10, || {
        // SAFETY: the launch geometry matches `flash_attention_forward`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            flash_module.flash_attention_forward(
                &stream,
                per_row_config(),
                &q,
                &k,
                &v,
                T as u32,
                H as u32,
                HD as u32,
                &mut y,
                &mut logsumexp,
            )
        }?;
        Ok(())
    })?;
    let tiled_forward_ms = time_gpu_iters(&stream, 5, 20, || {
        // SAFETY: the launch geometry matches `flash_attention_forward_tiled`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            flash_module.flash_attention_forward_tiled(
                &stream,
                flash::tiled_forward_config(B, T, H, HD),
                &q,
                &k,
                &v,
                T as u32,
                H as u32,
                &mut y,
                &mut logsumexp,
            )
        }?;
        Ok(())
    })?;

    let naive_backward_ms = time_gpu_iters(&stream, 1, 5, || {
        // SAFETY: the launch geometry matches `attention_backward_q`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            naive_module.attention_backward_q(
                &stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                &q,
                &k,
                &v,
                &probabilities,
                &dy,
                T as u32,
                H as u32,
                HD as u32,
                &mut dq,
            )
        }?;
        // SAFETY: the launch geometry matches `attention_backward_k`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            naive_module.attention_backward_k(
                &stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                &q,
                &v,
                &probabilities,
                &dy,
                T as u32,
                H as u32,
                HD as u32,
                &mut dk,
            )
        }?;
        // SAFETY: the launch geometry matches `attention_backward_v`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            naive_module.attention_backward_v(
                &stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                &probabilities,
                &dy,
                T as u32,
                H as u32,
                HD as u32,
                &mut dv,
            )
        }?;
        Ok(())
    })?;
    let per_row_backward_ms = time_gpu_iters(&stream, 2, 10, || {
        // SAFETY: the launch geometry matches `flash_attention_backward_q`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            flash_module.flash_attention_backward_q(
                &stream,
                per_row_config(),
                &q,
                &k,
                &v,
                &y,
                &dy,
                &logsumexp,
                T as u32,
                H as u32,
                HD as u32,
                &mut dq,
            )
        }?;
        // SAFETY: the launch geometry matches `flash_attention_backward_kv`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            flash_module.flash_attention_backward_kv(
                &stream,
                per_row_config(),
                &q,
                &k,
                &v,
                &y,
                &dy,
                &logsumexp,
                T as u32,
                H as u32,
                HD as u32,
                &mut dk,
                &mut dv,
            )
        }?;
        Ok(())
    })?;
    let tiled_backward_ms = time_gpu_iters(&stream, 5, 20, || {
        // SAFETY: the launch geometry matches `flash_attention_backward_dot`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            flash_module.flash_attention_backward_dot(
                &stream,
                flash::dot_config(N, H, HD),
                &dy,
                &y,
                HD as u32,
                &mut softmax_dot,
            )
        }?;
        // SAFETY: the launch geometry matches `flash_attention_backward_q_tiled`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            flash_module.flash_attention_backward_q_tiled(
                &stream,
                flash::tiled_backward_q_config(B, T, H, HD),
                &q,
                &k,
                &v,
                &dy,
                &logsumexp,
                &softmax_dot,
                T as u32,
                H as u32,
                &mut dq,
            )
        }?;
        // SAFETY: the launch geometry matches `flash_attention_backward_kv_tiled`'s grid/block
        // contract for this shape, and every buffer was allocated to the
        // extents the kernel indexes.
        unsafe {
            flash_module.flash_attention_backward_kv_tiled(
                &stream,
                flash::tiled_backward_kv_config(B, T, H, HD),
                &q,
                &k,
                &v,
                &dy,
                &logsumexp,
                &softmax_dot,
                T as u32,
                H as u32,
                &mut dk,
                &mut dv,
            )
        }?;
        Ok(())
    })?;

    let probability_mib = (N * H * T * size_of::<f32>()) as f64 / (1024.0 * 1024.0);
    println!("fp32 causal attention [B={B},T={T},H={H},HD={HD}]");
    println!("  materialized probabilities: {probability_mib:.2} MiB");
    println!(
        "  forward  naive={naive_forward_ms:9.3} ms  per-row={per_row_forward_ms:9.3} ms  tiled={tiled_forward_ms:9.3} ms  tiled speedup vs per-row={:.2}x",
        per_row_forward_ms / tiled_forward_ms
    );
    println!(
        "  backward naive={naive_backward_ms:9.3} ms  per-row={per_row_backward_ms:9.3} ms  tiled={tiled_backward_ms:9.3} ms  tiled speedup vs per-row={:.2}x",
        per_row_backward_ms / tiled_backward_ms
    );
    Ok(())
}
