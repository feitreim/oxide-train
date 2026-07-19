//! Parity checks against `ops`' materialized-probability attention.
//!
//! Both kernel generations are checked at each shape: the per-row flash
//! kernels and the FlashAttention-2 style tiled kernels. The shapes cover a
//! `T` that is not a multiple of any tile size (partial query/key tiles plus
//! the causal diagonal) and the tiny `T=4` configuration the model
//! overfit gate trains at (a single mostly-padded tile).
//!
//! Tile-aligned shapes additionally gate the tcgen05 forward (issue #35)
//! against both fp32 oracles at bf16-appropriate tolerances, after checking
//! the device staging kernel bit-exactly against a CPU mirror. Those checks
//! load `flash.ptx`, so `src/bin/flash.rs` must be built first (modal_app.py
//! prepares it, mirroring the model's `gemm.ptx` staging).

use bench_util::uniform_vec;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};

#[path = "lib.rs"]
mod flash;
#[path = "../../ops/src/lib.rs"]
mod naive;

use flash::host::{Tcgen05Flash, create_flash_head_tma_map, flash_forward_config};

const HD: usize = 64;
const LOG2_E: f32 = std::f32::consts::LOG2_E;

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let round = 0x7fffu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// CPU mirror of `stage_attention_heads_bf16`, used to gate the device
/// staging kernel bit-exactly.
fn stage_heads(input: &[f32], b: usize, t: usize, h: usize, scale: f32) -> Vec<u32> {
    let mut staged = vec![0u32; b * h * t * HD / 2];
    for plane in 0..b * h {
        let (batch, head) = (plane / h, plane % h);
        for token in 0..t {
            for pair in 0..HD / 2 {
                let base = ((batch * t + token) * h + head) * HD + pair * 2;
                let low = f32_to_bf16_rne(input[base] * scale) as u32;
                let high = f32_to_bf16_rne(input[base + 1] * scale) as u32;
                staged[(plane * t + token) * HD / 2 + pair] = low | (high << 16);
            }
        }
    }
    staged
}

/// Stage one operand on device and require bit-parity with the CPU mirror.
fn stage_on_device(
    stream: &CudaStream,
    flash_module: &flash::kernels::LoadedModule,
    input: &DeviceBuffer<f32>,
    host_input: &[f32],
    b: usize,
    t: usize,
    h: usize,
    scale: f32,
    name: &str,
) -> Result<DeviceBuffer<u32>, Box<dyn std::error::Error>> {
    let mut staged = DeviceBuffer::<u32>::zeroed(stream, b * h * t * HD / 2)?;
    flash_module.stage_attention_heads_bf16(
        stream,
        flash::stage_heads_config(b * t, h, HD),
        input,
        t as u32,
        h as u32,
        scale,
        &mut staged,
    )?;
    let device_words = staged.to_host_vec(stream)?;
    let host_words = stage_heads(host_input, b, t, h, scale);
    for (i, (&d, &e)) in device_words.iter().zip(&host_words).enumerate() {
        assert_eq!(d, e, "{name} staging word {i}: device {d:#010x} vs cpu {e:#010x}");
    }
    Ok(staged)
}

/// tcgen05 forward vs both fp32 oracles at a tile-aligned shape. Inputs are
/// quantized through the device staging kernel; the oracles run on the
/// original fp32 values, so tolerances are the bf16-appropriate ones (the
/// dominant term is operand quantization, per the 7e9 precedent).
#[allow(clippy::too_many_arguments)]
fn check_tcgen05_shape(
    stream: &CudaStream,
    flash_module: &flash::kernels::LoadedModule,
    naive_module: &naive::kernels::LoadedModule,
    tcgen05: &Tcgen05Flash,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = b * t;
    let d = h * HD;
    let q = uniform_vec(n * d, 171);
    let k = uniform_vec(n * d, 172);
    let v = uniform_vec(n * d, 173);
    let q_device = DeviceBuffer::from_host(stream, &q)?;
    let k_device = DeviceBuffer::from_host(stream, &k)?;
    let v_device = DeviceBuffer::from_host(stream, &v)?;

    let q_scale = LOG2_E / (HD as f32).sqrt();
    let q_staged = stage_on_device(stream, flash_module, &q_device, &q, b, t, h, q_scale, "q")?;
    let k_staged = stage_on_device(stream, flash_module, &k_device, &k, b, t, h, 1.0, "k")?;
    let v_staged = stage_on_device(stream, flash_module, &v_device, &v, b, t, h, 1.0, "v")?;

    // Tier 1 oracle: materialized probabilities from ops.
    let mut probabilities = DeviceBuffer::<f32>::zeroed(stream, n * h * t)?;
    let mut naive_y = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    naive_module.attention_probabilities(
        stream,
        LaunchConfig::for_num_elems((n * h * t) as u32),
        &q_device,
        &k_device,
        t as u32,
        h as u32,
        HD as u32,
        &mut probabilities,
    )?;
    naive_module.attention_output(
        stream,
        LaunchConfig::for_num_elems((n * d) as u32),
        &probabilities,
        &v_device,
        t as u32,
        h as u32,
        HD as u32,
        &mut naive_y,
    )?;

    // Tier 2 oracle: the fp32 tiled forward and its log-sum-exp.
    let mut tiled_y = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut tiled_lse = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
    flash_module.flash_attention_forward_tiled(
        stream,
        flash::tiled_forward_config(b, t, h, HD),
        &q_device,
        &k_device,
        &v_device,
        t as u32,
        h as u32,
        &mut tiled_y,
        &mut tiled_lse,
    )?;

    let q_tma = unsafe { create_flash_head_tma_map(stream, &q_staged, t, b * h)? };
    let k_tma = unsafe { create_flash_head_tma_map(stream, &k_staged, t, b * h)? };
    let v_tma = unsafe { create_flash_head_tma_map(stream, &v_staged, t, b * h)? };
    let mut y = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut lse = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
    unsafe {
        tcgen05.forward(
            stream,
            flash_forward_config(b, t, h),
            q_tma.as_ptr(),
            k_tma.as_ptr(),
            v_tma.as_ptr(),
            t as u32,
            h as u32,
            &mut y,
            &mut lse,
        )?;
    }

    println!("tcgen05 parity against both oracles [{b},{t},{h},{HD}]");
    // Measured maxima vs the fp32 oracles: y 2.4e-3, lse 8.9e-4 (T up to
    // 1024) — dominated by bf16 operand quantization; ~4x headroom.
    let y_host = y.to_host_vec(stream)?;
    assert_close("y/naive", &y_host, &naive_y.to_host_vec(stream)?, 1.0e-2, 1.0e-2);
    assert_close("y/tiled", &y_host, &tiled_y.to_host_vec(stream)?, 1.0e-2, 1.0e-2);
    assert_close(
        "lse",
        &lse.to_host_vec(stream)?,
        &tiled_lse.to_host_vec(stream)?,
        5.0e-3,
        0.0,
    );
    Ok(())
}

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

    println!("per-row flash parity against ops [{b},{t},{h},{HD}]");
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
    assert_close(
        "dq",
        &actual_dq.to_host_vec(stream)?,
        &expected_dq,
        1e-4,
        1e-4,
    );
    assert_close(
        "dk",
        &actual_dk.to_host_vec(stream)?,
        &expected_dk,
        1e-4,
        1e-4,
    );
    assert_close(
        "dv",
        &actual_dv.to_host_vec(stream)?,
        &expected_dv,
        1e-4,
        1e-4,
    );

    println!("tiled flash parity against ops [{b},{t},{h},{HD}]");
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
    assert_close(
        "dq",
        &tiled_dq.to_host_vec(stream)?,
        &expected_dq,
        1e-4,
        1e-4,
    );
    assert_close(
        "dk",
        &tiled_dk.to_host_vec(stream)?,
        &expected_dk,
        1e-4,
        1e-4,
    );
    assert_close(
        "dv",
        &tiled_dv.to_host_vec(stream)?,
        &expected_dv,
        1e-4,
        1e-4,
    );

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

    let tcgen05 = Tcgen05Flash::load_from_ptx(&ctx, "flash.ptx")?;
    check_tcgen05_shape(&stream, &flash_module, &naive_module, &tcgen05, 1, 128, 2)?;
    check_tcgen05_shape(&stream, &flash_module, &naive_module, &tcgen05, 2, 256, 3)?;
    check_tcgen05_shape(&stream, &flash_module, &naive_module, &tcgen05, 1, 1024, 4)?;
    println!("✓ tcgen05 forward parity passed on tile-aligned shapes");
    Ok(())
}
