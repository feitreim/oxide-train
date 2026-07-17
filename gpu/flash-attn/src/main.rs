//! Parity checks against `llama-ops`' materialized-probability attention.
//!
//! Both kernel generations are checked: the per-row flash kernels and the
//! FlashAttention-2 style tiled kernels. `T` is deliberately not a multiple of
//! any tile size so partial query/key tiles and the causal diagonal are all
//! exercised.

use bench_util::uniform_vec;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

#[path = "lib.rs"]
mod flash;
#[path = "../../llama-ops/src/lib.rs"]
mod naive;

const B: usize = 2;
const T: usize = 80;
const H: usize = 3;
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

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_error = 0.0f32;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (a - e).abs();
        max_error = max_error.max(error);
        let tolerance = atol + rtol * e.abs();
        assert!(
            error <= tolerance,
            "{name} mismatch at {i}: flash={a}, naive={e}, error={error}, tolerance={tolerance}"
        );
    }
    println!("  {name:<7} max abs error: {max_error:.3e}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert!(HD.is_power_of_two() && HD <= flash::MAX_HEAD_DIM);
    assert_eq!(HD, flash::TILE_HD);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let flash_module = flash::kernels::load(&ctx)?;
    let naive_module = naive::kernels::load(&ctx)?;

    let q = DeviceBuffer::from_host(&stream, &uniform_vec(N * D, 71))?;
    let k = DeviceBuffer::from_host(&stream, &uniform_vec(N * D, 72))?;
    let v = DeviceBuffer::from_host(&stream, &uniform_vec(N * D, 73))?;
    let dy = DeviceBuffer::from_host(&stream, &uniform_vec(N * D, 74))?;

    let mut probabilities = DeviceBuffer::<f32>::zeroed(&stream, N * H * T)?;
    let mut expected_y = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut expected_dq = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut expected_dk = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut expected_dv = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
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
        &mut expected_y,
    )?;
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
        &mut expected_dq,
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
        &mut expected_dk,
    )?;
    naive_module.attention_backward_v(
        &stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        &probabilities,
        &dy,
        T as u32,
        H as u32,
        HD as u32,
        &mut expected_dv,
    )?;
    let expected_y = expected_y.to_host_vec(&stream)?;
    let expected_dq = expected_dq.to_host_vec(&stream)?;
    let expected_dk = expected_dk.to_host_vec(&stream)?;
    let expected_dv = expected_dv.to_host_vec(&stream)?;

    let mut actual_y = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut logsumexp = DeviceBuffer::<f32>::zeroed(&stream, N * H)?;
    let mut actual_dq = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut actual_dk = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut actual_dv = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;

    println!("per-row flash parity against llama-ops [{B},{T},{H},{HD}]");
    flash_module.flash_attention_forward(
        &stream,
        per_row_config(),
        &q,
        &k,
        &v,
        T as u32,
        H as u32,
        HD as u32,
        &mut actual_y,
        &mut logsumexp,
    )?;
    flash_module.flash_attention_backward_q(
        &stream,
        per_row_config(),
        &q,
        &k,
        &v,
        &actual_y,
        &dy,
        &logsumexp,
        T as u32,
        H as u32,
        HD as u32,
        &mut actual_dq,
    )?;
    flash_module.flash_attention_backward_kv(
        &stream,
        per_row_config(),
        &q,
        &k,
        &v,
        &actual_y,
        &dy,
        &logsumexp,
        T as u32,
        H as u32,
        HD as u32,
        &mut actual_dk,
        &mut actual_dv,
    )?;
    assert_close("y", &actual_y.to_host_vec(&stream)?, &expected_y, 5e-5, 5e-5);
    assert_close("dq", &actual_dq.to_host_vec(&stream)?, &expected_dq, 1e-4, 1e-4);
    assert_close("dk", &actual_dk.to_host_vec(&stream)?, &expected_dk, 1e-4, 1e-4);
    assert_close("dv", &actual_dv.to_host_vec(&stream)?, &expected_dv, 1e-4, 1e-4);

    println!("tiled flash parity against llama-ops [{B},{T},{H},{HD}]");
    let mut tiled_y = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut tiled_logsumexp = DeviceBuffer::<f32>::zeroed(&stream, N * H)?;
    let mut softmax_dot = DeviceBuffer::<f32>::zeroed(&stream, N * H)?;
    let mut tiled_dq = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut tiled_dk = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    let mut tiled_dv = DeviceBuffer::<f32>::zeroed(&stream, N * D)?;
    flash_module.flash_attention_forward_tiled(
        &stream,
        flash::tiled_forward_config(B, T, H, HD),
        &q,
        &k,
        &v,
        T as u32,
        H as u32,
        &mut tiled_y,
        &mut tiled_logsumexp,
    )?;
    flash_module.flash_attention_backward_dot(
        &stream,
        flash::dot_config(N, H, HD),
        &dy,
        &tiled_y,
        HD as u32,
        &mut softmax_dot,
    )?;
    flash_module.flash_attention_backward_q_tiled(
        &stream,
        flash::tiled_backward_q_config(B, T, H, HD),
        &q,
        &k,
        &v,
        &dy,
        &tiled_logsumexp,
        &softmax_dot,
        T as u32,
        H as u32,
        &mut tiled_dq,
    )?;
    flash_module.flash_attention_backward_kv_tiled(
        &stream,
        flash::tiled_backward_kv_config(B, T, H, HD),
        &q,
        &k,
        &v,
        &dy,
        &tiled_logsumexp,
        &softmax_dot,
        T as u32,
        H as u32,
        &mut tiled_dk,
        &mut tiled_dv,
    )?;
    assert_close("y", &tiled_y.to_host_vec(&stream)?, &expected_y, 5e-5, 5e-5);
    assert_close(
        "lse",
        &tiled_logsumexp.to_host_vec(&stream)?,
        &logsumexp.to_host_vec(&stream)?,
        5e-5,
        5e-5,
    );
    assert_close("dq", &tiled_dq.to_host_vec(&stream)?, &expected_dq, 1e-4, 1e-4);
    assert_close("dk", &tiled_dk.to_host_vec(&stream)?, &expected_dk, 1e-4, 1e-4);
    assert_close("dv", &tiled_dv.to_host_vec(&stream)?, &expected_dv, 1e-4, 1e-4);

    println!(
        "✓ per-row and tiled parity passed without the {:.2} MiB probability buffer",
        (N * H * T * size_of::<f32>()) as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}
