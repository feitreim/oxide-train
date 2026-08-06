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
//! the device staging kernel bit-exactly against a CPU mirror. The fp32 and
//! tcgen05 kernels load from one embedded pure-PTX artifact.

use bench_util::{function_profile, uniform_vec};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};

#[path = "lib.rs"]
mod flash;
#[path = "../../ops/src/lib.rs"]
mod naive;

use flash::host::{
    Tcgen05Flash, correction_count_len, create_flash_head_tma_map, create_flash_row_major_tma_map,
    device_sm_count, flash_backward_kv_config, flash_backward_q_config, flash_forward_config,
};

const HD: usize = 128;
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

/// Round a `[B*T, H*HD]` operand to bf16 pairs where it lies — the layout a
/// projection GEMM's packed epilogue writes, and the one V is read in through
/// `create_flash_row_major_tma_map` instead of being relaid out.
fn pack_rows(input: &[f32]) -> Vec<u32> {
    input
        .chunks_exact(2)
        .map(|pair| f32_to_bf16_rne(pair[0]) as u32 | ((f32_to_bf16_rne(pair[1]) as u32) << 16))
        .collect()
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
    // SAFETY: the launch uses live buffers sized for its contract, and the
    // subsequent host copy synchronizes the stream before either can drop.
    unsafe {
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
            assert_eq!(
                d, e,
                "{name} staging word {i}: device {d:#010x} vs cpu {e:#010x}"
            );
        }
        Ok(staged)
    }
}

/// tcgen05 forward vs both fp32 oracles at a tile-aligned shape. Inputs are
/// quantized through the device staging kernel; the oracles run on the
/// original fp32 values, so tolerances are the bf16-appropriate ones (the
/// dominant term is operand quantization, per the 7e9 precedent).
#[allow(clippy::too_many_arguments, unused_unsafe)]
fn check_tcgen05_shape(
    stream: &CudaStream,
    flash_module: &flash::kernels::LoadedModule,
    naive_module: &naive::kernels::LoadedModule,
    tcgen05: &Tcgen05Flash,
    sm_count: usize,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: launch shapes match the documented kernel contracts and all
    // buffers/TMA descriptors remain live until their stream work completes.
    unsafe {
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
        let v_rows = DeviceBuffer::from_host(stream, &pack_rows(&v))?;

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
        let v_tma = unsafe { create_flash_row_major_tma_map(stream, &v_rows, n, h)? };
        let naive_y_host = naive_y.to_host_vec(stream)?;
        let tiled_y_host = tiled_y.to_host_vec(stream)?;
        let tiled_lse_host = tiled_lse.to_host_vec(stream)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        let mut lse = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
        let mut corrections = DeviceBuffer::<u32>::zeroed(stream, correction_count_len(b, t, h))?;
        unsafe {
            tcgen05.forward(
                stream,
                flash_forward_config(b, t, h, sm_count),
                q_tma.as_ptr(),
                k_tma.as_ptr(),
                v_tma.as_ptr(),
                t as u32,
                h as u32,
                b as u32,
                &mut y,
                &mut lse,
                &mut corrections,
            )?;
        }

        println!("tcgen05 forward parity against both oracles [{b},{t},{h},{HD}]");
        // Measured maxima vs the fp32 oracles: y 2.4e-3, lse 8.9e-4 (T up
        // to 1024) — dominated by bf16 operand quantization; ~4x headroom.
        let y_host = y.to_host_vec(stream)?;
        assert_close("y/naive", &y_host, &naive_y_host, 1.0e-2, 1.0e-2);
        assert_close("y/tiled", &y_host, &tiled_y_host, 1.0e-2, 1.0e-2);
        assert_close(
            "lse",
            &lse.to_host_vec(stream)?,
            &tiled_lse_host,
            5.0e-3,
            0.0,
        );
        Ok(())
    }
}

/// tcgen05 backward (issue #35 phase 4) vs both fp32 oracles at a
/// tile-aligned shape, wired exactly like the model: the tcgen05 forward
/// produces `y`/LSE from the staged bf16 operands, the fp32 `backward_dot`
/// reduces `Σ dy·y` from that `y`, and the two gradient kernels consume the
/// staged operands plus those statistics. Gradients are compared against the
/// materialized-probability oracle at bf16-appropriate tolerances (operand
/// quantization dominates, per the forward's 7e9/7e10 precedent).
#[allow(clippy::too_many_arguments, unused_unsafe)]
fn check_tcgen05_backward_shape(
    stream: &CudaStream,
    flash_module: &flash::kernels::LoadedModule,
    naive_module: &naive::kernels::LoadedModule,
    tcgen05: &Tcgen05Flash,
    sm_count: usize,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: launch shapes match the documented kernel contracts and all
    // buffers/TMA descriptors remain live until their stream work completes.
    unsafe {
        let n = b * t;
        let d = h * HD;
        let q = uniform_vec(n * d, 271);
        let k = uniform_vec(n * d, 272);
        let v = uniform_vec(n * d, 273);
        let dy = uniform_vec(n * d, 274);
        let q_device = DeviceBuffer::from_host(stream, &q)?;
        let k_device = DeviceBuffer::from_host(stream, &k)?;
        let v_device = DeviceBuffer::from_host(stream, &v)?;
        let dy_device = DeviceBuffer::from_host(stream, &dy)?;

        let q_scale = LOG2_E / (HD as f32).sqrt();
        let q_staged = stage_on_device(stream, flash_module, &q_device, &q, b, t, h, q_scale, "q")?;
        let k_staged = stage_on_device(stream, flash_module, &k_device, &k, b, t, h, 1.0, "k")?;
        let v_rows = DeviceBuffer::from_host(stream, &pack_rows(&v))?;
        let dy_staged = stage_on_device(stream, flash_module, &dy_device, &dy, b, t, h, 1.0, "dy")?;

        // Tier-1 oracle: materialized-probability backward from ops.
        let mut probabilities = DeviceBuffer::<f32>::zeroed(stream, n * h * t)?;
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
        let mut expected_dq = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        let mut expected_dk = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        let mut expected_dv = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        naive_module.attention_backward_q(
            stream,
            LaunchConfig::for_num_elems((n * d) as u32),
            &q_device,
            &k_device,
            &v_device,
            &probabilities,
            &dy_device,
            t as u32,
            h as u32,
            HD as u32,
            &mut expected_dq,
        )?;
        naive_module.attention_backward_k(
            stream,
            LaunchConfig::for_num_elems((n * d) as u32),
            &q_device,
            &v_device,
            &probabilities,
            &dy_device,
            t as u32,
            h as u32,
            HD as u32,
            &mut expected_dk,
        )?;
        naive_module.attention_backward_v(
            stream,
            LaunchConfig::for_num_elems((n * d) as u32),
            &probabilities,
            &dy_device,
            t as u32,
            h as u32,
            HD as u32,
            &mut expected_dv,
        )?;
        let expected_dq = expected_dq.to_host_vec(stream)?;
        let expected_dk = expected_dk.to_host_vec(stream)?;
        let expected_dv = expected_dv.to_host_vec(stream)?;

        // Model data flow: tcgen05 forward y/LSE, then fp32 backward_dot over y.
        let q_tma = unsafe { create_flash_head_tma_map(stream, &q_staged, t, b * h)? };
        let k_tma = unsafe { create_flash_head_tma_map(stream, &k_staged, t, b * h)? };
        let v_tma = unsafe { create_flash_row_major_tma_map(stream, &v_rows, n, h)? };
        let dy_tma = unsafe { create_flash_head_tma_map(stream, &dy_staged, t, b * h)? };
        let mut y = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        let mut lse = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
        let mut corrections = DeviceBuffer::<u32>::zeroed(stream, correction_count_len(b, t, h))?;
        unsafe {
            tcgen05.forward(
                stream,
                flash_forward_config(b, t, h, sm_count),
                q_tma.as_ptr(),
                k_tma.as_ptr(),
                v_tma.as_ptr(),
                t as u32,
                h as u32,
                b as u32,
                &mut y,
                &mut lse,
                &mut corrections,
            )?;
        }
        let mut dot = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
        flash_module.flash_attention_backward_dot(
            stream,
            flash::dot_config(n, h, HD),
            &dy_device,
            &y,
            HD as u32,
            &mut dot,
        )?;

        let mut dq = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        let mut dk = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        let mut dv = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        unsafe {
            tcgen05.backward_q(
                stream,
                flash_backward_q_config(b, t, h, sm_count),
                q_tma.as_ptr(),
                k_tma.as_ptr(),
                v_tma.as_ptr(),
                dy_tma.as_ptr(),
                &lse,
                &dot,
                t as u32,
                h as u32,
                b as u32,
                &mut dq,
            )?;
            tcgen05.backward_kv(
                stream,
                flash_backward_kv_config(b, t, h, sm_count),
                q_tma.as_ptr(),
                k_tma.as_ptr(),
                v_tma.as_ptr(),
                dy_tma.as_ptr(),
                &lse,
                &dot,
                t as u32,
                h as u32,
                b as u32,
                &mut dk,
                &mut dv,
            )?;
        }

        println!("tcgen05 backward parity against ops oracle [{b},{t},{h},{HD}]");
        assert_close("dq", &dq.to_host_vec(stream)?, &expected_dq, 1.0e-2, 1.0e-2);
        assert_close("dk", &dk.to_host_vec(stream)?, &expected_dk, 1.0e-2, 1.0e-2);
        assert_close("dv", &dv.to_host_vec(stream)?, &expected_dv, 1.0e-2, 1.0e-2);
        Ok(())
    }
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
    // SAFETY: launch shapes match the documented kernel contracts and every
    // device buffer remains valid through the corresponding host readback.
    unsafe {
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
}

/// The fused rotate-and-stage pass against the composition it substitutes for.
///
/// `stage_qk_heads_bf16` replaces `split_group3` + two `rope_forward` passes +
/// two `stage_attention_heads_bf16` launches. Everything it drops was fp32
/// storage between two kernels, so the substitution owes exact parity — every
/// staged word, not a tolerance — and that is what is asserted here. The
/// unfused kernels keep their own bit-exact CPU mirror above; this gate is the
/// third tier of the chain (SPEC §11).
fn check_fused_staging(
    stream: &CudaStream,
    flash_module: &flash::kernels::LoadedModule,
    naive_module: &naive::kernels::LoadedModule,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: every launch below is sized from the same (b, t, h) shape as the
    // buffers it reads and writes.
    unsafe {
        let (n, d) = (b * t, h * HD);
        let words = n * d / 2;
        let qk = DeviceBuffer::from_host(stream, &uniform_vec(n * 2 * d, 181))?;
        let table = DeviceBuffer::from_host(stream, &naive::rope_table(t, HD))?;
        let q_scale = LOG2_E / (HD as f32).sqrt();
        let elements = LaunchConfig::for_num_elems((n * d) as u32);
        let pairs = LaunchConfig::for_num_elems(words as u32);
        let staging = flash::stage_heads_config(n, h, HD);

        let mut split_q = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        let mut split_k = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        naive_module.split_group2(stream, elements, &qk, d as u32, &mut split_q, &mut split_k)?;
        let mut rotated_q = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        let mut rotated_k = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        naive_module.rope_forward(
            stream,
            pairs,
            &split_q,
            &table,
            t as u32,
            h as u32,
            HD as u32,
            &mut rotated_q,
        )?;
        naive_module.rope_forward(
            stream,
            pairs,
            &split_k,
            &table,
            t as u32,
            h as u32,
            HD as u32,
            &mut rotated_k,
        )?;

        let mut fused_q = DeviceBuffer::<u32>::zeroed(stream, words)?;
        let mut fused_k = DeviceBuffer::<u32>::zeroed(stream, words)?;
        flash_module.stage_qk_heads_bf16(
            stream,
            staging,
            &qk,
            &table,
            t as u32,
            h as u32,
            q_scale,
            &mut fused_q,
            &mut fused_k,
        )?;

        for (name, operand, scale, fused) in [
            ("q", &rotated_q, q_scale, &fused_q),
            ("k", &rotated_k, 1.0, &fused_k),
        ] {
            let mut expected = DeviceBuffer::<u32>::zeroed(stream, words)?;
            flash_module.stage_attention_heads_bf16(
                stream,
                staging,
                operand,
                t as u32,
                h as u32,
                scale,
                &mut expected,
            )?;
            let (got, want) = (fused.to_host_vec(stream)?, expected.to_host_vec(stream)?);
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g, w,
                    "fused {name} staging word {i} at [{b},{t},{h}]: \
                     {g:#010x} vs split+rope+stage {w:#010x}"
                );
            }
        }
        Ok(())
    }
}

/// What ptxas gave the fused dY pass and the two kernels it replaces.
///
/// Fusion raises liveness, and a kernel that crosses an occupancy step can
/// lose on the clock while winning on traffic. The driver is asked directly,
/// at each kernel's own block width, rather than inferred from the diff.
fn report_dy_dot_residency(
    flash_module: &flash::kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    let module = flash_module.as_cuda_module();
    println!("dY pass residency (registers/thread, spill bytes, blocks/SM)");
    for (name, block_threads) in [
        ("stage_attention_heads_bf16", 256),
        ("flash_attention_backward_dot_bf16", HD as u32),
        ("stage_attention_dy_dot_bf16", 256),
    ] {
        let function = module.load_function(name)?;
        let profile = function_profile(&function)?;
        let blocks = function.max_active_blocks_per_multiprocessor(block_threads, 0)?;
        println!(
            "  {name:<34} {:>4} {:>6} {blocks:>4} at {block_threads} threads",
            profile.registers, profile.spill_bytes,
        );
    }
    Ok(())
}

/// `stage_attention_dy_dot_bf16` against the two kernels it replaces.
///
/// The staged panel is the same arithmetic on the same bytes and the dot's
/// butterfly walks the same pairings as the twin's shared-memory tree, so this
/// substitution owes exact parity in both outputs — every staged word and
/// every dot bit — and that is what is asserted here.
fn check_fused_dy_dot(
    stream: &CudaStream,
    flash_module: &flash::kernels::LoadedModule,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: every launch below is sized from the same (b, t, h) shape as the
    // buffers it reads and writes.
    unsafe {
        let (n, d) = (b * t, h * HD);
        let words = n * d / 2;
        let host_dy = uniform_vec(n * d, 907);
        let host_y = uniform_vec(n * d, 613);
        let packed_y: Vec<u32> = host_y
            .chunks_exact(2)
            .map(|pair| f32_to_bf16_rne(pair[0]) as u32 | ((f32_to_bf16_rne(pair[1]) as u32) << 16))
            .collect();
        let dy = DeviceBuffer::from_host(stream, &host_dy)?;
        let y = DeviceBuffer::from_host(stream, &packed_y)?;

        let mut want_staged = DeviceBuffer::<u32>::zeroed(stream, words)?;
        flash_module.stage_attention_heads_bf16(
            stream,
            flash::stage_heads_config(n, h, HD),
            &dy,
            t as u32,
            h as u32,
            1.0,
            &mut want_staged,
        )?;
        let mut want_dot = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
        flash_module.flash_attention_backward_dot_bf16(
            stream,
            flash::dot_config(n, h, HD),
            &dy,
            &y,
            HD as u32,
            &mut want_dot,
        )?;

        let mut got_staged = DeviceBuffer::<u32>::zeroed(stream, words)?;
        let mut got_dot = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
        flash_module.stage_attention_dy_dot_bf16(
            stream,
            flash::stage_dy_dot_config(n, h, HD),
            &dy,
            &y,
            t as u32,
            h as u32,
            &mut got_staged,
            &mut got_dot,
        )?;

        for (i, (&g, &w)) in got_staged
            .to_host_vec(stream)?
            .iter()
            .zip(&want_staged.to_host_vec(stream)?)
            .enumerate()
        {
            assert_eq!(
                g, w,
                "fused dy staging word {i} at [{b},{t},{h}]: {g:#010x} vs stage {w:#010x}"
            );
        }
        for (i, (&g, &w)) in got_dot
            .to_host_vec(stream)?
            .iter()
            .zip(&want_dot.to_host_vec(stream)?)
            .enumerate()
        {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "fused dy dot {i} at [{b},{t},{h}]: {g} vs backward_dot {w}"
            );
        }
        Ok(())
    }
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

    let tcgen05 = Tcgen05Flash::load(&ctx)?;
    let sm_count = device_sm_count(&ctx)?;
    check_tcgen05_shape(
        &stream,
        &flash_module,
        &naive_module,
        &tcgen05,
        sm_count,
        1,
        128,
        2,
    )?;
    check_tcgen05_shape(
        &stream,
        &flash_module,
        &naive_module,
        &tcgen05,
        sm_count,
        2,
        256,
        3,
    )?;
    check_tcgen05_shape(
        &stream,
        &flash_module,
        &naive_module,
        &tcgen05,
        sm_count,
        1,
        1024,
        4,
    )?;
    println!("✓ tcgen05 forward parity passed on tile-aligned shapes");
    for (b, t, h) in [(1, 128, 2), (2, 256, 3), (1, 1024, 4)] {
        check_fused_staging(&stream, &flash_module, &naive_module, b, t, h)?;
    }
    println!("✓ fused qkv rotate-and-stage matches split + rope + stage exactly");
    for (b, t, h) in [(1, 128, 2), (2, 256, 3), (1, 1024, 4)] {
        check_fused_dy_dot(&stream, &flash_module, b, t, h)?;
    }
    println!("✓ fused dy stage-and-dot matches stage + backward_dot exactly");
    report_dy_dot_residency(&flash_module)?;
    check_tcgen05_backward_shape(
        &stream,
        &flash_module,
        &naive_module,
        &tcgen05,
        sm_count,
        1,
        128,
        2,
    )?;
    check_tcgen05_backward_shape(
        &stream,
        &flash_module,
        &naive_module,
        &tcgen05,
        sm_count,
        2,
        256,
        3,
    )?;
    check_tcgen05_backward_shape(
        &stream,
        &flash_module,
        &naive_module,
        &tcgen05,
        sm_count,
        1,
        1024,
        4,
    )?;
    println!("✓ tcgen05 backward parity passed on tile-aligned shapes");
    Ok(())
}
