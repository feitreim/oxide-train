//! CPU/GPU parity checks for the reference Llama kernels.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use nn::{
    CausalAttention, Embedding, Module, RmsNorm, Rope, SoftmaxCrossEntropy,
    SoftmaxCrossEntropyInput, SwiGlu,
};
use tensor_core::{Rank1, Rank2, Rank3};
use tensor_cpu::CpuTensor;

// `cargo oxide` collects kernels from the selected binary target, not from a
// separately compiled library dependency. Reuse the canonical library source
// as a module so this binary's embedded artifact contains the kernels.
#[path = "lib.rs"]
mod device;
use device::{CLASSIFIER_THREADS, NORM_THREADS, kernels};
use tensor_core::bf16;

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * e.abs();
        assert!(
            (a - e).abs() <= tolerance,
            "{name} mismatch at {i}: gpu={a}, cpu={e}, tolerance={tolerance}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    check_rms_norm(&stream, &module)?;
    check_swiglu(&stream, &module)?;
    check_embedding(&stream, &module)?;
    check_cross_entropy(&stream, &module)?;
    check_classifier_bf16(&stream, &module)?;
    check_rope(&stream, &module)?;
    check_attention(&stream, &module)?;
    check_group_split_join(&stream, &module)?;

    println!("✓ llama-ops forward/backward parity checks passed");
    Ok(())
}

fn check_rope(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 10;
    const T: usize = 5;
    const D: usize = 12;
    const H: usize = 3;
    const HD: usize = 4;
    let x = CpuTensor::<f32, Rank2<N, D>>::uniform(10);
    let dy = CpuTensor::<f32, Rank2<N, D>>::uniform(11);
    let mut cpu = Rope::<N, T, D, H, HD>;
    let (cpu_y, ()) = cpu.forward(x.clone());
    let cpu_dx = cpu.backward((), dy.clone());
    let x_dev = DeviceBuffer::from_host(stream, x.as_slice())?;
    let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let mut dx_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;

    module.rope_forward(
        stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        &x_dev,
        T as u32,
        H as u32,
        HD as u32,
        &mut y_dev,
    )?;
    module.rope_backward(
        stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        &dy_dev,
        T as u32,
        H as u32,
        HD as u32,
        &mut dx_dev,
    )?;
    assert_close(
        "rope y",
        &y_dev.to_host_vec(stream)?,
        cpu_y.as_slice(),
        2e-6,
        2e-6,
    );
    assert_close(
        "rope dx",
        &dx_dev.to_host_vec(stream)?,
        cpu_dx.as_slice(),
        2e-6,
        2e-6,
    );
    Ok(())
}

fn check_attention(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 10;
    const T: usize = 5;
    const D: usize = 12;
    const H: usize = 3;
    const HD: usize = 4;
    let q = CpuTensor::<f32, Rank2<N, D>>::uniform(12);
    let k = CpuTensor::<f32, Rank2<N, D>>::uniform(13);
    let v = CpuTensor::<f32, Rank2<N, D>>::uniform(14);
    let dy = CpuTensor::<f32, Rank2<N, D>>::uniform(15);
    let mut cpu = CausalAttention::<N, T, D, H, HD>;
    let (cpu_y, cpu_ctx) = cpu.forward((q.clone(), k.clone(), v.clone()));
    let (cpu_dq, cpu_dk, cpu_dv) = cpu.backward(cpu_ctx, dy.clone());

    let q_dev = DeviceBuffer::from_host(stream, q.as_slice())?;
    let k_dev = DeviceBuffer::from_host(stream, k.as_slice())?;
    let v_dev = DeviceBuffer::from_host(stream, v.as_slice())?;
    let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
    let mut p_dev = DeviceBuffer::<f32>::zeroed(stream, N * H * T)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let mut dq_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let mut dk_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let mut dv_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    module.attention_probabilities(
        stream,
        LaunchConfig::for_num_elems((N * H * T) as u32),
        &q_dev,
        &k_dev,
        T as u32,
        H as u32,
        HD as u32,
        &mut p_dev,
    )?;
    module.attention_output(
        stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        &p_dev,
        &v_dev,
        T as u32,
        H as u32,
        HD as u32,
        &mut y_dev,
    )?;
    module.attention_backward_q(
        stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        &q_dev,
        &k_dev,
        &v_dev,
        &p_dev,
        &dy_dev,
        T as u32,
        H as u32,
        HD as u32,
        &mut dq_dev,
    )?;
    module.attention_backward_k(
        stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        &q_dev,
        &v_dev,
        &p_dev,
        &dy_dev,
        T as u32,
        H as u32,
        HD as u32,
        &mut dk_dev,
    )?;
    module.attention_backward_v(
        stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        &p_dev,
        &dy_dev,
        T as u32,
        H as u32,
        HD as u32,
        &mut dv_dev,
    )?;

    assert_close(
        "attention y",
        &y_dev.to_host_vec(stream)?,
        cpu_y.as_slice(),
        3e-5,
        3e-5,
    );
    assert_close(
        "attention dq",
        &dq_dev.to_host_vec(stream)?,
        cpu_dq.as_slice(),
        5e-5,
        5e-5,
    );
    assert_close(
        "attention dk",
        &dk_dev.to_host_vec(stream)?,
        cpu_dk.as_slice(),
        5e-5,
        5e-5,
    );
    assert_close(
        "attention dv",
        &dv_dev.to_host_vec(stream)?,
        cpu_dv.as_slice(),
        5e-5,
        5e-5,
    );
    let probabilities = p_dev.to_host_vec(stream)?;
    let probabilities = CpuTensor::<f32, Rank3<N, H, T>>::from_slice(&probabilities);
    for row in 0..N {
        for head in 0..H {
            let start = (row * H + head) * T;
            let sum: f32 = probabilities.as_slice()[start..start + T].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
    Ok(())
}

fn check_rms_norm(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 5;
    const D: usize = 7;
    let x = CpuTensor::<f32, Rank2<N, D>>::uniform(1);
    let weight = CpuTensor::<f32, Rank1<D>>::uniform(2).map(|v| v + 1.25);
    let dy = CpuTensor::<f32, Rank2<N, D>>::uniform(3);
    let mut cpu = RmsNorm::<N, D>::new(weight.clone(), 1e-5);
    let (cpu_y, cpu_ctx) = cpu.forward(x.clone());
    let cpu_dx = cpu.backward(cpu_ctx, dy.clone());

    let x_dev = DeviceBuffer::from_host(stream, x.as_slice())?;
    let weight_dev = DeviceBuffer::from_host(stream, weight.as_slice())?;
    let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let mut dx_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let mut dw_dev = DeviceBuffer::<f32>::zeroed(stream, D)?;

    module.rms_norm_forward(
        stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        &x_dev,
        &weight_dev,
        1e-5,
        D as u32,
        &mut y_dev,
    )?;
    module.rms_norm_backward_x(
        stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        &x_dev,
        &weight_dev,
        &dy_dev,
        1e-5,
        D as u32,
        &mut dx_dev,
    )?;
    module.rms_norm_backward_weight(
        stream,
        LaunchConfig::for_num_elems(D as u32),
        &x_dev,
        &dy_dev,
        1e-5,
        N as u32,
        D as u32,
        &mut dw_dev,
    )?;

    assert_close(
        "rmsnorm y",
        &y_dev.to_host_vec(stream)?,
        cpu_y.as_slice(),
        2e-5,
        2e-5,
    );
    assert_close(
        "rmsnorm dx",
        &dx_dev.to_host_vec(stream)?,
        cpu_dx.as_slice(),
        3e-5,
        3e-5,
    );
    assert_close(
        "rmsnorm dw",
        &dw_dev.to_host_vec(stream)?,
        cpu.dw.as_slice(),
        3e-5,
        3e-5,
    );

    // Fast weight-gradient pair against the naive oracle above.
    let mut inv_dev = DeviceBuffer::<f32>::zeroed(stream, N)?;
    let mut dw_fast_dev = DeviceBuffer::<f32>::zeroed(stream, D)?;
    module.rms_norm_row_inv(
        stream,
        LaunchConfig {
            grid_dim: (N as u32, 1, 1),
            block_dim: (NORM_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        },
        &x_dev,
        1e-5,
        D as u32,
        &mut inv_dev,
    )?;
    module.rms_norm_backward_weight_fast(
        stream,
        LaunchConfig::for_num_elems(D as u32),
        &x_dev,
        &dy_dev,
        &inv_dev,
        N as u32,
        D as u32,
        &mut dw_fast_dev,
    )?;
    assert_close(
        "rmsnorm dw fast vs naive",
        &dw_fast_dev.to_host_vec(stream)?,
        &dw_dev.to_host_vec(stream)?,
        1e-6,
        1e-5,
    );
    Ok(())
}

fn check_classifier_bf16(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // Odd real vocabulary exercises the packed tail; the second case covers a
    // multi-iteration lane stride.
    check_classifier_bf16_case::<5, 13, 16>(stream, module)?;
    check_classifier_bf16_case::<3, 517, 520>(stream, module)
}

fn check_classifier_bf16_case<const N: usize, const C: usize, const CP: usize>(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(CP % 2 == 0 && CP >= C);
    let logits = CpuTensor::<f32, Rank2<N, C>>::uniform(16).scale(5.0);
    // Round to bf16 once; the f32 oracle then sees the exact values the bf16
    // kernels decode, so the only differences are lane order and the bf16
    // rounding of the written gradients.
    let rounded: Vec<f32> = logits
        .as_slice()
        .iter()
        .map(|&value| bf16::from_f32(value).to_f32())
        .collect();
    let mut packed = vec![0u32; N * CP / 2];
    for row in 0..N {
        for col in 0..C {
            let bits = bf16::from_f32(rounded[row * C + col]).to_bits() as u32;
            packed[(row * CP + col) / 2] |= bits << (16 * (col % 2));
        }
    }
    let targets_usize: [usize; N] = std::array::from_fn(|row| (row * 101 + C - 1) % C);
    let targets = targets_usize.map(|v| v as u32);
    let targets_dev = DeviceBuffer::from_host(stream, &targets)?;
    let classifier_config = LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (CLASSIFIER_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // f32 fused oracle on the rounded values.
    let rounded_dev = DeviceBuffer::from_host(stream, &rounded)?;
    let mut oracle_losses = DeviceBuffer::<f32>::zeroed(stream, N)?;
    let mut oracle_dlogits = DeviceBuffer::from_host(stream, &rounded)?;
    module.fused_classifier_forward(
        stream,
        classifier_config,
        &rounded_dev,
        &targets_dev,
        N as u32,
        C as u32,
        &mut oracle_losses,
    )?;
    module.fused_classifier_backward_in_place(
        stream,
        classifier_config,
        &targets_dev,
        1.0,
        N as u32,
        C as u32,
        &mut oracle_dlogits,
    )?;

    let packed_dev = DeviceBuffer::from_host(stream, &packed)?;
    let mut losses = DeviceBuffer::<f32>::zeroed(stream, N)?;
    let mut dlogits = DeviceBuffer::from_host(stream, &packed)?;
    module.fused_classifier_forward_bf16(
        stream,
        classifier_config,
        &packed_dev,
        &targets_dev,
        N as u32,
        C as u32,
        CP as u32,
        &mut losses,
    )?;
    module.fused_classifier_backward_in_place_bf16(
        stream,
        classifier_config,
        &targets_dev,
        1.0,
        N as u32,
        C as u32,
        CP as u32,
        &mut dlogits,
    )?;

    assert_close(
        "bf16 classifier losses vs f32 fused",
        &losses.to_host_vec(stream)?,
        &oracle_losses.to_host_vec(stream)?,
        5e-5,
        2e-5,
    );
    let dlogits = dlogits.to_host_vec(stream)?;
    let oracle = oracle_dlogits.to_host_vec(stream)?;
    for row in 0..N {
        for col in 0..CP {
            let word = dlogits[(row * CP + col) / 2];
            let bits = (word >> (16 * (col % 2))) as u16;
            if col < C {
                let actual = bf16::from_bits(bits).to_f32();
                let expected = oracle[row * C + col];
                let tolerance = 1e-6 + 4e-3 * expected.abs();
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "bf16 classifier dlogits mismatch at [{row},{col}]: \
                     gpu={actual}, oracle={expected}, tolerance={tolerance}"
                );
            } else {
                assert_eq!(bits, 0, "padded dlogits column [{row},{col}] is not zero");
            }
        }
    }
    Ok(())
}

fn check_swiglu(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const LEN: usize = 33;
    let gate = CpuTensor::<f32, Rank2<3, 11>>::uniform(4);
    let up = CpuTensor::<f32, Rank2<3, 11>>::uniform(5);
    let dy = CpuTensor::<f32, Rank2<3, 11>>::uniform(6);
    let mut cpu = SwiGlu::<3, 11>;
    let (cpu_y, cpu_ctx) = cpu.forward((gate.clone(), up.clone()));
    let (cpu_dgate, cpu_dup) = cpu.backward(cpu_ctx, dy.clone());

    let gate_dev = DeviceBuffer::from_host(stream, gate.as_slice())?;
    let up_dev = DeviceBuffer::from_host(stream, up.as_slice())?;
    let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
    let mut dgate_dev = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
    let mut dup_dev = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
    module.swiglu_forward(
        stream,
        LaunchConfig::for_num_elems(LEN as u32),
        &gate_dev,
        &up_dev,
        &mut y_dev,
    )?;
    module.swiglu_backward_gate(
        stream,
        LaunchConfig::for_num_elems(LEN as u32),
        &gate_dev,
        &up_dev,
        &dy_dev,
        &mut dgate_dev,
    )?;
    module.swiglu_backward_up(
        stream,
        LaunchConfig::for_num_elems(LEN as u32),
        &gate_dev,
        &dy_dev,
        &mut dup_dev,
    )?;

    assert_close(
        "swiglu y",
        &y_dev.to_host_vec(stream)?,
        cpu_y.as_slice(),
        1e-6,
        1e-5,
    );
    assert_close(
        "swiglu dgate",
        &dgate_dev.to_host_vec(stream)?,
        cpu_dgate.as_slice(),
        2e-6,
        1e-5,
    );
    assert_close(
        "swiglu dup",
        &dup_dev.to_host_vec(stream)?,
        cpu_dup.as_slice(),
        2e-6,
        1e-5,
    );
    Ok(())
}

fn check_embedding(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 6;
    const V: usize = 9;
    const D: usize = 5;
    let tokens_usize = [2, 7, 2, 0, 7, 4];
    let tokens = tokens_usize.map(|v| v as u32);
    let weight = CpuTensor::<f32, Rank2<V, D>>::uniform(7);
    let dy = CpuTensor::<f32, Rank2<N, D>>::uniform(8);
    let mut cpu = Embedding::<N, V, D>::new(weight.clone());
    let (cpu_y, cpu_ctx) = cpu.forward(tokens_usize);
    cpu.backward(cpu_ctx, dy.clone());

    let weight_dev = DeviceBuffer::from_host(stream, weight.as_slice())?;
    let tokens_dev = DeviceBuffer::from_host(stream, &tokens)?;
    let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
    let mut dw_dev = DeviceBuffer::<f32>::zeroed(stream, V * D)?;
    module.embedding_forward(
        stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        &weight_dev,
        &tokens_dev,
        D as u32,
        &mut y_dev,
    )?;
    module.embedding_backward(
        stream,
        LaunchConfig::for_num_elems((V * D) as u32),
        &tokens_dev,
        &dy_dev,
        N as u32,
        D as u32,
        &mut dw_dev,
    )?;

    assert_close(
        "embedding y",
        &y_dev.to_host_vec(stream)?,
        cpu_y.as_slice(),
        0.0,
        0.0,
    );
    assert_close(
        "embedding dw",
        &dw_dev.to_host_vec(stream)?,
        cpu.dw.as_slice(),
        1e-6,
        1e-6,
    );
    Ok(())
}

/// Round-trips grouped tensors through split and join at a shape that does
/// not divide the 256-thread launch rounding, so block-excess threads are
/// exercised in both kernels.
fn check_group_split_join(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const ROWS: usize = 7;
    const WIDTH: usize = 13;
    let packed3 = CpuTensor::<f32, Rank2<ROWS, 39>>::uniform(11);
    let packed2 = CpuTensor::<f32, Rank2<ROWS, 26>>::uniform(12);
    let part = |packed: &[f32], groups: usize, group: usize| -> Vec<f32> {
        (0..ROWS * WIDTH)
            .map(|i| packed[(i / WIDTH * groups + group) * WIDTH + i % WIDTH])
            .collect()
    };

    let packed3_dev = DeviceBuffer::from_host(stream, packed3.as_slice())?;
    let mut first = DeviceBuffer::<f32>::zeroed(stream, ROWS * WIDTH)?;
    let mut second = DeviceBuffer::<f32>::zeroed(stream, ROWS * WIDTH)?;
    let mut third = DeviceBuffer::<f32>::zeroed(stream, ROWS * WIDTH)?;
    let elems = LaunchConfig::for_num_elems((ROWS * WIDTH) as u32);
    module.split_group3(
        stream,
        elems,
        &packed3_dev,
        WIDTH as u32,
        &mut first,
        &mut second,
        &mut third,
    )?;
    for (name, buffer, group) in [
        ("split_group3 first", &first, 0),
        ("split_group3 second", &second, 1),
        ("split_group3 third", &third, 2),
    ] {
        assert_close(
            name,
            &buffer.to_host_vec(stream)?,
            &part(packed3.as_slice(), 3, group),
            0.0,
            0.0,
        );
    }
    let mut joined3 = DeviceBuffer::<f32>::zeroed(stream, ROWS * 3 * WIDTH)?;
    // SAFETY: the three parts are disjoint [ROWS, WIDTH] tensors and the
    // output holds exactly ROWS * 3 * WIDTH elements.
    unsafe {
        module.join_group3(
            stream,
            elems,
            &first,
            &second,
            &third,
            WIDTH as u32,
            &mut joined3,
        )?;
    }
    assert_close(
        "join_group3",
        &joined3.to_host_vec(stream)?,
        packed3.as_slice(),
        0.0,
        0.0,
    );

    let packed2_dev = DeviceBuffer::from_host(stream, packed2.as_slice())?;
    module.split_group2(
        stream,
        elems,
        &packed2_dev,
        WIDTH as u32,
        &mut first,
        &mut second,
    )?;
    for (name, buffer, group) in [
        ("split_group2 first", &first, 0),
        ("split_group2 second", &second, 1),
    ] {
        assert_close(
            name,
            &buffer.to_host_vec(stream)?,
            &part(packed2.as_slice(), 2, group),
            0.0,
            0.0,
        );
    }
    let mut joined2 = DeviceBuffer::<f32>::zeroed(stream, ROWS * 2 * WIDTH)?;
    // SAFETY: both parts are disjoint [ROWS, WIDTH] tensors and the output
    // holds exactly ROWS * 2 * WIDTH elements.
    unsafe {
        module.join_group2(stream, elems, &first, &second, WIDTH as u32, &mut joined2)?;
    }
    assert_close(
        "join_group2",
        &joined2.to_host_vec(stream)?,
        packed2.as_slice(),
        0.0,
        0.0,
    );
    Ok(())
}

fn check_cross_entropy(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    check_cross_entropy_case::<5, 13>(stream, module)?;
    check_cross_entropy_case::<5, 517>(stream, module)
}

fn check_cross_entropy_case<const N: usize, const C: usize>(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    let logits = CpuTensor::<f32, Rank2<N, C>>::uniform(9).scale(5.0);
    let targets_usize = std::array::from_fn(|row| (row * 101 + C - 1) % C);
    let targets = targets_usize.map(|v| v as u32);
    let mut cpu = SoftmaxCrossEntropy::<N, C>;
    let (cpu_loss, cpu_ctx) = cpu.forward(SoftmaxCrossEntropyInput {
        logits: logits.clone(),
        targets: targets_usize,
    });
    let cpu_dx = cpu.backward(cpu_ctx, CpuTensor::from_slice(&[1.0])).logits;

    let logits_dev = DeviceBuffer::from_host(stream, logits.as_slice())?;
    let targets_dev = DeviceBuffer::from_host(stream, &targets)?;
    let mut probabilities_dev = DeviceBuffer::<f32>::zeroed(stream, N * C)?;
    let mut losses_dev = DeviceBuffer::<f32>::zeroed(stream, N)?;
    let mut dlogits_dev = DeviceBuffer::<f32>::zeroed(stream, N * C)?;
    let mut fused_losses_dev = DeviceBuffer::<f32>::zeroed(stream, N)?;
    let mut fused_dlogits_dev = DeviceBuffer::from_host(stream, logits.as_slice())?;
    module.softmax_forward(
        stream,
        LaunchConfig::for_num_elems((N * C) as u32),
        &logits_dev,
        C as u32,
        &mut probabilities_dev,
    )?;
    module.cross_entropy_loss(
        stream,
        LaunchConfig::for_num_elems(N as u32),
        &logits_dev,
        &targets_dev,
        N as u32,
        C as u32,
        &mut losses_dev,
    )?;
    module.softmax_cross_entropy_backward(
        stream,
        LaunchConfig::for_num_elems((N * C) as u32),
        &probabilities_dev,
        &targets_dev,
        1.0,
        N as u32,
        C as u32,
        &mut dlogits_dev,
    )?;
    let classifier_config = LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (CLASSIFIER_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module.fused_classifier_forward(
        stream,
        classifier_config,
        &logits_dev,
        &targets_dev,
        N as u32,
        C as u32,
        &mut fused_losses_dev,
    )?;
    module.fused_classifier_backward_in_place(
        stream,
        classifier_config,
        &targets_dev,
        1.0,
        N as u32,
        C as u32,
        &mut fused_dlogits_dev,
    )?;

    let losses = losses_dev.to_host_vec(stream)?;
    let fused_losses = fused_losses_dev.to_host_vec(stream)?;
    assert_close(
        "fused classifier losses vs naive",
        &fused_losses,
        &losses,
        5e-5,
        2e-5,
    );
    let gpu_loss = fused_losses.iter().sum::<f32>() / N as f32;
    assert_close(
        "cross entropy loss",
        &[gpu_loss],
        cpu_loss.as_slice(),
        2e-5,
        2e-5,
    );
    assert_close(
        "fused classifier dx vs naive",
        &fused_dlogits_dev.to_host_vec(stream)?,
        &dlogits_dev.to_host_vec(stream)?,
        5e-6,
        2e-5,
    );
    assert_close(
        "fused classifier dx vs CPU",
        &fused_dlogits_dev.to_host_vec(stream)?,
        cpu_dx.as_slice(),
        5e-6,
        2e-5,
    );
    Ok(())
}
