//! Parity checks against `llama-ops`' materialized-probability attention.
//!
//! Both kernel generations are checked at each shape: the per-row flash
//! kernels and the FlashAttention-2 style tiled kernels. The shapes cover a
//! `T` that is not a multiple of any tile size (partial query/key tiles plus
//! the causal diagonal) and the tiny `T=4` configuration the llama-model
//! overfit gate trains at (a single mostly-padded tile).

use bench_util::uniform_vec;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};

#[path = "lib.rs"]
mod flash;
#[path = "../../llama-ops/src/lib.rs"]
mod naive;

const HD: usize = 64;

fn per_row_config(rows: usize, heads: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((rows * heads) as u32, 1, 1),
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

#[allow(clippy::too_many_arguments)]
fn check_shape(
    stream: &CudaStream,
    flash_module: &flash::kernels::LoadedModule,
    naive_module: &naive::kernels::LoadedModule,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = b * t;
    let d = h * HD;

    let q = DeviceBuffer::from_host(stream, &uniform_vec(n * d, 71))?;
    let k = DeviceBuffer::from_host(stream, &uniform_vec(n * d, 72))?;
    let v = DeviceBuffer::from_host(stream, &uniform_vec(n * d, 73))?;
    let dy = DeviceBuffer::from_host(stream, &uniform_vec(n * d, 74))?;

    let mut probabilities = DeviceBuffer::<f32>::zeroed(stream, n * h * t)?;
    let mut expected_y = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut expected_dq = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut expected_dk = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut expected_dv = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    naive_module.attention_probabilities(
        stream,
        LaunchConfig::for_num_elems((n * h * t) as u32),
        &q,
        &k,
        t as u32,
        h as u32,
        HD as u32,
        &mut probabilities,
    )?;
    naive_module.attention_output(
        stream,
        LaunchConfig::for_num_elems((n * d) as u32),
        &probabilities,
        &v,
        t as u32,
        h as u32,
        HD as u32,
        &mut expected_y,
    )?;
    naive_module.attention_backward_q(
        stream,
        LaunchConfig::for_num_elems((n * d) as u32),
        &q,
        &k,
        &v,
        &probabilities,
        &dy,
        t as u32,
        h as u32,
        HD as u32,
        &mut expected_dq,
    )?;
    naive_module.attention_backward_k(
        stream,
        LaunchConfig::for_num_elems((n * d) as u32),
        &q,
        &v,
        &probabilities,
        &dy,
        t as u32,
        h as u32,
        HD as u32,
        &mut expected_dk,
    )?;
    naive_module.attention_backward_v(
        stream,
        LaunchConfig::for_num_elems((n * d) as u32),
        &probabilities,
        &dy,
        t as u32,
        h as u32,
        HD as u32,
        &mut expected_dv,
    )?;
    let expected_y = expected_y.to_host_vec(stream)?;
    let expected_dq = expected_dq.to_host_vec(stream)?;
    let expected_dk = expected_dk.to_host_vec(stream)?;
    let expected_dv = expected_dv.to_host_vec(stream)?;

    let mut actual_y = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut logsumexp = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
    let mut actual_dq = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut actual_dk = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut actual_dv = DeviceBuffer::<f32>::zeroed(stream, n * d)?;

    println!("per-row flash parity against llama-ops [{b},{t},{h},{HD}]");
    flash_module.flash_attention_forward(
        stream,
        per_row_config(n, h),
        &q,
        &k,
        &v,
        t as u32,
        h as u32,
        HD as u32,
        &mut actual_y,
        &mut logsumexp,
    )?;
    flash_module.flash_attention_backward_q(
        stream,
        per_row_config(n, h),
        &q,
        &k,
        &v,
        &actual_y,
        &dy,
        &logsumexp,
        t as u32,
        h as u32,
        HD as u32,
        &mut actual_dq,
    )?;
    flash_module.flash_attention_backward_kv(
        stream,
        per_row_config(n, h),
        &q,
        &k,
        &v,
        &actual_y,
        &dy,
        &logsumexp,
        t as u32,
        h as u32,
        HD as u32,
        &mut actual_dk,
        &mut actual_dv,
    )?;
    assert_close("y", &actual_y.to_host_vec(stream)?, &expected_y, 5e-5, 5e-5);
    assert_close("dq", &actual_dq.to_host_vec(stream)?, &expected_dq, 1e-4, 1e-4);
    assert_close("dk", &actual_dk.to_host_vec(stream)?, &expected_dk, 1e-4, 1e-4);
    assert_close("dv", &actual_dv.to_host_vec(stream)?, &expected_dv, 1e-4, 1e-4);

    println!("tiled flash parity against llama-ops [{b},{t},{h},{HD}]");
    let mut tiled_y = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut tiled_logsumexp = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
    let mut softmax_dot = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
    let mut tiled_dq = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut tiled_dk = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut tiled_dv = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    flash_module.flash_attention_forward_tiled(
        stream,
        flash::tiled_forward_config(b, t, h, HD),
        &q,
        &k,
        &v,
        t as u32,
        h as u32,
        &mut tiled_y,
        &mut tiled_logsumexp,
    )?;
    flash_module.flash_attention_backward_dot(
        stream,
        flash::dot_config(n, h, HD),
        &dy,
        &tiled_y,
        HD as u32,
        &mut softmax_dot,
    )?;
    flash_module.flash_attention_backward_q_tiled(
        stream,
        flash::tiled_backward_q_config(b, t, h, HD),
        &q,
        &k,
        &v,
        &dy,
        &tiled_logsumexp,
        &softmax_dot,
        t as u32,
        h as u32,
        &mut tiled_dq,
    )?;
    flash_module.flash_attention_backward_kv_tiled(
        stream,
        flash::tiled_backward_kv_config(b, t, h, HD),
        &q,
        &k,
        &v,
        &dy,
        &tiled_logsumexp,
        &softmax_dot,
        t as u32,
        h as u32,
        &mut tiled_dk,
        &mut tiled_dv,
    )?;
    assert_close("y", &tiled_y.to_host_vec(stream)?, &expected_y, 5e-5, 5e-5);
    assert_close(
        "lse",
        &tiled_logsumexp.to_host_vec(stream)?,
        &logsumexp.to_host_vec(stream)?,
        5e-5,
        5e-5,
    );
    assert_close("dq", &tiled_dq.to_host_vec(stream)?, &expected_dq, 1e-4, 1e-4);
    assert_close("dk", &tiled_dk.to_host_vec(stream)?, &expected_dk, 1e-4, 1e-4);
    assert_close("dv", &tiled_dv.to_host_vec(stream)?, &expected_dv, 1e-4, 1e-4);

    // DIAGNOSTIC: training reuses gradient scratch buffers, so any output
    // element the tiled kernels skip writing leaks stale data. Seed every
    // output with a sentinel and rerun; a surviving sentinel means a
    // write-coverage gap. Then loop for bit-stability to expose races.
    let first_y = tiled_y.to_host_vec(stream)?;
    let first_lse = tiled_logsumexp.to_host_vec(stream)?;
    let first_dot = softmax_dot.to_host_vec(stream)?;
    let first_dq = tiled_dq.to_host_vec(stream)?;
    let first_dk = tiled_dk.to_host_vec(stream)?;
    let first_dv = tiled_dv.to_host_vec(stream)?;
    let sentinel_y = vec![1.0e30f32; n * d];
    let sentinel_h = vec![1.0e30f32; n * h];
    for round in 0..200 {
        let mut tiled_y = DeviceBuffer::from_host(stream, &sentinel_y)?;
        let mut tiled_logsumexp = DeviceBuffer::from_host(stream, &sentinel_h)?;
        let mut softmax_dot = DeviceBuffer::from_host(stream, &sentinel_h)?;
        let mut tiled_dq = DeviceBuffer::from_host(stream, &sentinel_y)?;
        let mut tiled_dk = DeviceBuffer::from_host(stream, &sentinel_y)?;
        let mut tiled_dv = DeviceBuffer::from_host(stream, &sentinel_y)?;
        flash_module.flash_attention_forward_tiled(
            stream,
            flash::tiled_forward_config(b, t, h, HD),
            &q,
            &k,
            &v,
            t as u32,
            h as u32,
            &mut tiled_y,
            &mut tiled_logsumexp,
        )?;
        flash_module.flash_attention_backward_dot(
            stream,
            flash::dot_config(n, h, HD),
            &dy,
            &tiled_y,
            HD as u32,
            &mut softmax_dot,
        )?;
        flash_module.flash_attention_backward_q_tiled(
            stream,
            flash::tiled_backward_q_config(b, t, h, HD),
            &q,
            &k,
            &v,
            &dy,
            &tiled_logsumexp,
            &softmax_dot,
            t as u32,
            h as u32,
            &mut tiled_dq,
        )?;
        flash_module.flash_attention_backward_kv_tiled(
            stream,
            flash::tiled_backward_kv_config(b, t, h, HD),
            &q,
            &k,
            &v,
            &dy,
            &tiled_logsumexp,
            &softmax_dot,
            t as u32,
            h as u32,
            &mut tiled_dk,
            &mut tiled_dv,
        )?;
        for (name, buffer, first) in [
            ("y", &tiled_y, &first_y),
            ("lse", &tiled_logsumexp, &first_lse),
            ("dot", &softmax_dot, &first_dot),
            ("dq", &tiled_dq, &first_dq),
            ("dk", &tiled_dk, &first_dk),
            ("dv", &tiled_dv, &first_dv),
        ] {
            let values = buffer.to_host_vec(stream)?;
            for (i, (&a, &b)) in values.iter().zip(first).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "{name} unstable at [{b},{t},{h}] round {round} index {i}: \
                     {a:e} (bits {:#x}) vs first {b:e} — sentinel leak or race",
                    a.to_bits(),
                );
            }
        }
    }
    println!("  sentinel + 200-round bit-stability passed");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert!(HD.is_power_of_two() && HD <= flash::MAX_HEAD_DIM);
    assert_eq!(HD, flash::TILE_HD);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let flash_module = flash::kernels::load(&ctx)?;
    let naive_module = naive::kernels::load(&ctx)?;

    check_shape(&stream, &flash_module, &naive_module, 2, 80, 3)?;
    check_shape(&stream, &flash_module, &naive_module, 1, 4, 2)?;
    println!("✓ per-row and tiled parity passed on both shapes");
    Ok(())
}
