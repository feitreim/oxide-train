//! Flash attention versus the `llama-ops` materialized-probability baseline.
//!
//! Run with `./run.sh flash-attn bench`.

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

#[path = "../lib.rs"]
mod flash;
#[path = "../../../llama-ops/src/lib.rs"]
mod naive;

const B: usize = 2;
const T: usize = 64;
const H: usize = 8;
const HD: usize = 64;
const N: usize = B * T;
const D: usize = H * HD;

fn forward_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((N * H) as u32, 1, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn backward_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((B * H) as u32, 1, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert!(HD.is_power_of_two() && HD <= flash::MAX_HEAD_DIM);

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
    let mut dq = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut dk = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut dv = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;

    let naive_forward_ms = time_gpu_iters(&stream, 2, 10, || {
        naive_module.attention_probabilities(
            &stream,
            LaunchConfig::for_num_elems((N * H * T) as u32),
            &q,
            &k,
            T as u32,
            H as u32,
            HD as u32,
            &mut probabilities,
        )?;
        naive_module.attention_output(
            &stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &probabilities,
            &v,
            T as u32,
            H as u32,
            HD as u32,
            &mut y,
        )?;
        Ok(())
    })?;
    let flash_forward_ms = time_gpu_iters(&stream, 5, 20, || {
        flash_module.flash_attention_forward(
            &stream,
            forward_config(),
            &q,
            &k,
            &v,
            T as u32,
            H as u32,
            HD as u32,
            &mut y,
        )?;
        Ok(())
    })?;

    let naive_backward_ms = time_gpu_iters(&stream, 1, 5, || {
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
        )?;
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
        )?;
        naive_module.attention_backward_v(
            &stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &probabilities,
            &dy,
            T as u32,
            H as u32,
            HD as u32,
            &mut dv,
        )?;
        Ok(())
    })?;
    let flash_backward_ms = time_gpu_iters(&stream, 2, 10, || {
        flash_module.flash_attention_backward(
            &stream,
            backward_config(),
            &q,
            &k,
            &v,
            &dy,
            T as u32,
            H as u32,
            HD as u32,
            &mut dq,
            &mut dk,
            &mut dv,
        )?;
        Ok(())
    })?;

    let probability_mib = (N * H * T * size_of::<f32>()) as f64 / (1024.0 * 1024.0);
    println!("fp32 causal attention [B={B},T={T},H={H},HD={HD}]");
    println!("  materialized probabilities: {probability_mib:.2} MiB");
    println!(
        "  forward  naive={naive_forward_ms:8.3} ms  flash={flash_forward_ms:8.3} ms  speedup={:.2}x",
        naive_forward_ms / flash_forward_ms
    );
    println!(
        "  backward naive={naive_backward_ms:8.3} ms  flash={flash_backward_ms:8.3} ms  speedup={:.2}x",
        naive_backward_ms / flash_backward_ms
    );
    Ok(())
}
