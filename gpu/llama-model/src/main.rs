//! End-to-end forward/backward parity against `nn::Llama`.
//!
//! The network is fp32 except the bf16 tcgen05 lm-head, so quantities
//! downstream of the logits carry bf16 tolerances while the fused-AdamW
//! master-weight comparison stays tight: both optimizers are fed the exact
//! bf16-rounded gradients the GPU produced.
//!
//! Dimensions are the smallest that exercise the real tcgen05 path: `D` and
//! `VP` are one 128 tile, `N` = 8 real token rows inside one padded `NP` =
//! 128 tile, and the odd `VOCAB` = 17 exercises the classifier's packed tail.

use cuda_core::CudaContext;
use nn::Llama;
use optim::{AdamWConfig, LlamaAdamW};
use tensor_core::{Rank2, Rank3, Shape, bf16};
use tensor_cpu::CpuTensor;

#[path = "lib.rs"]
mod model;
use model::{GpuLlama, GpuLlamaAdamW, GpuLlamaWorkspace};

const N: usize = 8;
const NP: usize = 128;
const T: usize = 4;
const VOCAB: usize = 17;
const VP: usize = 128;
const D: usize = 128;
const H: usize = 4;
const HD: usize = 32;
const FF: usize = 19;

/// Loss and gradients that crossed the bf16 head: inputs quantized to bf16,
/// fp32 accumulation, outputs re-rounded to bf16.
const BF16_ATOL: f32 = 2e-3;
const BF16_RTOL: f32 = 3e-2;

fn assert_close<S: Shape>(
    name: &str,
    gpu: &model::tensor_device::GpuTensor<f32, S>,
    cpu: &CpuTensor<f32, S>,
    stream: &cuda_core::CudaStream,
    atol: f32,
    rtol: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = gpu.to_host(stream)?;
    assert_close_slices(name, &actual, cpu.as_slice(), atol, rtol);
    Ok(())
}

fn assert_close_slices(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * e.abs();
        assert!(
            (a - e).abs() <= tolerance,
            "{name} mismatch at {i}: gpu={a}, cpu={e}, tolerance={tolerance}"
        );
    }
}

fn assert_grouped_close<const IN: usize, const GROUPS: usize, const OUT: usize>(
    name: &str,
    gpu: &model::tensor_device::GpuTensor<f32, Rank3<IN, GROUPS, OUT>>,
    expected: [&CpuTensor<f32, Rank2<IN, OUT>>; GROUPS],
    stream: &cuda_core::CudaStream,
    atol: f32,
    rtol: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = gpu.to_host(stream)?;
    for input in 0..IN {
        for (group, expected) in expected.iter().enumerate() {
            for output in 0..OUT {
                let index = (input * GROUPS + group) * OUT + output;
                let expected = expected.as_slice()[input * OUT + output];
                let tolerance = atol + rtol * expected.abs();
                assert!(
                    (actual[index] - expected).abs() <= tolerance,
                    "{name} mismatch at [{input},{group},{output}]: gpu={}, cpu={expected}, tolerance={tolerance}",
                    actual[index],
                );
            }
        }
    }
    Ok(())
}

fn split_grouped<const IN: usize, const GROUPS: usize, const OUT: usize>(
    gpu: &model::tensor_device::GpuTensor<f32, Rank3<IN, GROUPS, OUT>>,
    stream: &cuda_core::CudaStream,
) -> Result<[CpuTensor<f32, Rank2<IN, OUT>>; GROUPS], Box<dyn std::error::Error>> {
    let grouped = gpu.to_host(stream)?;
    Ok(std::array::from_fn(|group| {
        let mut values = vec![0.0; IN * OUT];
        for input in 0..IN {
            let source = (input * GROUPS + group) * OUT;
            values[input * OUT..(input + 1) * OUT].copy_from_slice(&grouped[source..source + OUT]);
        }
        CpuTensor::from_slice(&values)
    }))
}

fn unpack_bf16(words: &[u32]) -> Vec<f32> {
    let mut values = Vec::with_capacity(words.len() * 2);
    for &word in words {
        values.push(bf16::from_bits(word as u16).to_f32());
        values.push(bf16::from_bits((word >> 16) as u16).to_f32());
    }
    values
}

fn pack_bf16(values: &[f32]) -> Vec<u32> {
    values
        .chunks_exact(2)
        .map(|pair| {
            bf16::from_f32(pair[0]).to_bits() as u32
                | ((bf16::from_f32(pair[1]).to_bits() as u32) << 16)
        })
        .collect()
}

/// `[D, VP]` values -> `[D, VOCAB]`, asserting the padded columns are zero.
fn strip_vocab_padding(name: &str, padded: &[f32]) -> Vec<f32> {
    assert_eq!(padded.len(), D * VP);
    let mut stripped = Vec::with_capacity(D * VOCAB);
    for row in 0..D {
        stripped.extend_from_slice(&padded[row * VP..row * VP + VOCAB]);
        for column in VOCAB..VP {
            assert_eq!(
                padded[row * VP + column],
                0.0,
                "{name}: padded column [{row},{column}] is not zero"
            );
        }
    }
    stripped
}

/// The head's packed-bf16 gradient as stripped `[D, VOCAB]` f32 values.
fn head_gradient(
    head: &model::GpuBf16Head<D, VP>,
    stream: &cuda_core::CudaStream,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let words = head.dw_words().to_host_vec(stream)?;
    Ok(strip_vocab_padding("lm_head.dw", &unpack_bf16(&words)))
}

fn check_head_gradients(
    label: &str,
    gpu: &GpuLlama<N, NP, T, VOCAB, VP, D, H, HD, FF>,
    cpu: &Llama<N, T, VOCAB, D, H, HD, FF>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_close_slices(
        label,
        &head_gradient(&gpu.lm_head, stream)?,
        cpu.lm_head.dw.as_slice(),
        BF16_ATOL,
        BF16_RTOL,
    );
    Ok(())
}

/// The compute copies are exact rounded shadows of the master: `w` is
/// bf16(master) bit-for-bit and `w_t` is its element transpose.
fn check_head_compute_copies(
    head: &model::GpuBf16Head<D, VP>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    let master = head.master.to_host(stream)?;
    let expected_w = pack_bf16(&master);
    assert_eq!(
        head.w_words().to_host_vec(stream)?,
        expected_w,
        "lm_head.w is not the rounded master"
    );
    let rounded = unpack_bf16(&expected_w);
    let mut transposed = vec![0.0f32; VP * D];
    for row in 0..D {
        for column in 0..VP {
            transposed[column * D + row] = rounded[row * VP + column];
        }
    }
    assert_eq!(
        head.w_t_words().to_host_vec(stream)?,
        pack_bf16(&transposed),
        "lm_head.w_t is not the transposed rounded master"
    );
    Ok(())
}

fn overfit_tiny_batch(
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    gemm: &model::gemm_kernels::LoadedModule,
    gemm_bf16: &model::Tcgen05Gemm,
    flash: &model::flash_kernels::LoadedModule,
    llama: &model::llama_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    type TinyLlama = Llama<4, 4, 4, 128, 2, 64, 12>;
    let tokens = [0, 1, 2, 3];
    let targets = [1, 2, 3, 0];
    let cpu = TinyLlama::new(100);
    let mut gpu = GpuLlama::<4, 128, 4, 4, 128, 128, 2, 64, 12>::from_cpu(stream, &cpu)?;
    let config = AdamWConfig {
        learning_rate: 0.03,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut optimizer = GpuLlamaAdamW::new(stream, config)?;
    let mut workspace = GpuLlamaWorkspace::<4, 128, 4, 4, 128, 128, 2, 12>::new(stream)?;
    let mut initial_loss = None;

    // More steps than the fp32 gate needed: the bf16 head plateaus for a few
    // hundred steps with two competing logits tied at bf16 resolution until
    // the fp32 master accumulates enough sub-ulp progress to break the tie
    // (reproduced on CPU in crates/optim/examples/overfit_probe.rs, which
    // escapes by step ~300 and reaches ~5e-6 by step 360).
    for _ in 0..600 {
        gpu.zero_grad(stream, tensor)?;
        gpu.forward(
            tokens,
            targets,
            &mut workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            llama,
        )?;
        if initial_loss.is_none() {
            initial_loss = Some(workspace.loss().to_host(stream)?[0]);
        }
        gpu.backward(
            &mut workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            llama,
        )?;
        optimizer.update(&mut gpu, stream, tensor)?;
    }

    gpu.forward(
        tokens,
        targets,
        &mut workspace,
        stream,
        tensor,
        gemm,
        gemm_bf16,
        flash,
        llama,
    )?;
    let final_loss = workspace.loss().to_host(stream)?[0];
    let initial_loss = initial_loss.expect("training loop runs at least once");
    assert!(
        final_loss < 0.05,
        "GPU tiny batch did not overfit: initial={initial_loss}, final={final_loss}"
    );
    assert!(
        final_loss < initial_loss * 0.05,
        "GPU loss did not fall enough: initial={initial_loss}, final={final_loss}"
    );
    println!("✓ fused GPU AdamW overfits a tiny batch ({initial_loss:.6} -> {final_loss:.6})");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let tensor = model::tensor_kernels::load(&ctx)?;
    let gemm = model::gemm_kernels::load(&ctx)?;
    let gemm_bf16 = model::Tcgen05Gemm::load_from_ptx(&ctx, "gemm.ptx")?;
    let flash = model::flash_kernels::load(&ctx)?;
    let llama = model::llama_kernels::load(&ctx)?;

    let mut cpu = Llama::<N, T, VOCAB, D, H, HD, FF>::new(42);
    let mut gpu = GpuLlama::<N, NP, T, VOCAB, VP, D, H, HD, FF>::from_cpu(&stream, &cpu)?;
    let mut workspace = GpuLlamaWorkspace::<N, NP, T, VOCAB, VP, D, H, FF>::new(&stream)?;
    let tokens = [1, 5, 5, 2, 9, 3, 16, 0];
    let targets = [5, 5, 2, 7, 3, 16, 0, 4];

    let (cpu_loss, cpu_ctx) = cpu.forward(tokens, targets);
    gpu.forward(
        tokens,
        targets,
        &mut workspace,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &llama,
    )?;
    assert_close(
        "loss",
        workspace.loss(),
        &cpu_loss,
        &stream,
        BF16_ATOL,
        BF16_RTOL,
    )?;

    cpu.backward(cpu_ctx);
    gpu.backward(
        &mut workspace,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &llama,
    )?;

    macro_rules! grad {
        ($field:ident, $label:expr) => {
            assert_close(
                $label,
                &gpu.$field.dw,
                &cpu.$field.dw,
                &stream,
                BF16_ATOL,
                BF16_RTOL,
            )?;
        };
    }
    macro_rules! grads {
        ($suffix:expr) => {
            grad!(embedding, concat!("embedding.dw", $suffix));
            grad!(attention_norm, concat!("attention_norm.dw", $suffix));
            assert_grouped_close(
                concat!("qkv_proj.dw", $suffix),
                &gpu.qkv_proj.dw,
                [&cpu.q_proj.dw, &cpu.k_proj.dw, &cpu.v_proj.dw],
                &stream,
                BF16_ATOL,
                BF16_RTOL,
            )?;
            grad!(o_proj, concat!("o_proj.dw", $suffix));
            grad!(ffn_norm, concat!("ffn_norm.dw", $suffix));
            assert_grouped_close(
                concat!("gate_up_proj.dw", $suffix),
                &gpu.gate_up_proj.dw,
                [&cpu.gate_proj.dw, &cpu.up_proj.dw],
                &stream,
                BF16_ATOL,
                BF16_RTOL,
            )?;
            grad!(down_proj, concat!("down_proj.dw", $suffix));
            grad!(final_norm, concat!("final_norm.dw", $suffix));
            check_head_gradients(concat!("lm_head.dw", $suffix), &gpu, &cpu, &stream)?;
        };
    }
    grads!("");

    // Second pass through the same workspace: identical weights and inputs
    // must reproduce identical loss and gradients. Catches state leaking
    // between steps via reused buffers (including the padded rows of the
    // packed head buffers), which single-pass parity cannot.
    gpu.zero_grad(&stream, &tensor)?;
    gpu.forward(
        tokens,
        targets,
        &mut workspace,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &llama,
    )?;
    assert_close(
        "loss (pass 2)",
        workspace.loss(),
        &cpu_loss,
        &stream,
        BF16_ATOL,
        BF16_RTOL,
    )?;
    gpu.backward(
        &mut workspace,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &llama,
    )?;
    grads!(" (pass 2)");

    // Feed the exact GPU gradients to both optimizers so this comparison
    // isolates the fused update kernels from forward/backward rounding. The
    // lm-head grads are the bf16-rounded values the GPU kernel consumes.
    macro_rules! copy_grad {
        ($field:ident) => {
            cpu.$field.dw = gpu.$field.dw.to_cpu(&stream)?;
        };
    }
    copy_grad!(embedding);
    copy_grad!(attention_norm);
    let [q_grad, k_grad, v_grad] = split_grouped(&gpu.qkv_proj.dw, &stream)?;
    cpu.q_proj.dw = q_grad;
    cpu.k_proj.dw = k_grad;
    cpu.v_proj.dw = v_grad;
    copy_grad!(o_proj);
    copy_grad!(ffn_norm);
    let [gate_grad, up_grad] = split_grouped(&gpu.gate_up_proj.dw, &stream)?;
    cpu.gate_proj.dw = gate_grad;
    cpu.up_proj.dw = up_grad;
    copy_grad!(down_proj);
    copy_grad!(final_norm);
    cpu.lm_head.dw = CpuTensor::from_slice(&head_gradient(&gpu.lm_head, &stream)?);

    let config = AdamWConfig {
        learning_rate: 0.01,
        weight_decay: 0.1,
        ..AdamWConfig::default()
    };
    let mut cpu_optimizer = LlamaAdamW::new(config);
    let mut gpu_optimizer = GpuLlamaAdamW::new(&stream, config)?;
    cpu_optimizer.update(&mut cpu);
    gpu_optimizer.update(&mut gpu, &stream, &tensor)?;

    macro_rules! weight {
        ($field:ident) => {
            assert_close(
                concat!(stringify!($field), ".w after AdamW"),
                &gpu.$field.w,
                &cpu.$field.w,
                &stream,
                2e-6,
                2e-6,
            )?;
        };
    }
    weight!(embedding);
    weight!(attention_norm);
    assert_grouped_close(
        "qkv_proj.w after AdamW",
        &gpu.qkv_proj.w,
        [&cpu.q_proj.w, &cpu.k_proj.w, &cpu.v_proj.w],
        &stream,
        2e-6,
        2e-6,
    )?;
    weight!(o_proj);
    weight!(ffn_norm);
    assert_grouped_close(
        "gate_up_proj.w after AdamW",
        &gpu.gate_up_proj.w,
        [&cpu.gate_proj.w, &cpu.up_proj.w],
        &stream,
        2e-6,
        2e-6,
    )?;
    weight!(down_proj);
    weight!(final_norm);
    let master = gpu.lm_head.master.to_host(&stream)?;
    assert_close_slices(
        "lm_head master after AdamW",
        &strip_vocab_padding("lm_head.master", &master),
        cpu.lm_head.w.as_slice(),
        2e-6,
        2e-6,
    );
    check_head_compute_copies(&gpu.lm_head, &stream)?;
    gpu.zero_grad(&stream, &tensor)?;

    println!("✓ full GPU Llama forward/backward and AdamW (bf16 lm-head) match CPU");
    overfit_tiny_batch(&stream, &tensor, &gemm, &gemm_bf16, &flash, &llama)?;
    Ok(())
}
