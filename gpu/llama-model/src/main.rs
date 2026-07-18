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
//!
//! These shapes are deliberately non-tile-aligned, so the block linears take
//! their fp32 fallback; [`aligned_tcgen05_linears`] runs a second, fully
//! 128-aligned configuration that is the end-to-end gate for the bf16
//! tcgen05 block-linear path (7e9).

use cuda_core::{CudaContext, DeviceBuffer};
use nn::Llama;
use optim::{AdamWConfig, LlamaAdamW, LlamaMuon, MuonConfig, zeroth_power_via_newton_schulz};
use tensor_core::{Rank2, Rank3, Shape, bf16, rng::uniform_vec};
use tensor_cpu::CpuTensor;

#[path = "lib.rs"]
mod model;
use model::{GpuLlama, GpuLlamaAdamW, GpuLlamaMuon, GpuLlamaWorkspace, GpuMuonScratch};

const N: usize = 8;
const NP: usize = 128;
const T: usize = 4;
const VOCAB: usize = 17;
const VP: usize = 128;
// `HD` must match the tiled flash kernels' compile-time head width (7e7).
const D: usize = 128;
const H: usize = 2;
const HD: usize = 64;
const FF: usize = 19;

/// Loss and gradients that crossed the bf16 head: inputs quantized to bf16,
/// fp32 accumulation, outputs re-rounded to bf16.
const BF16_ATOL: f32 = 2e-3;
const BF16_RTOL: f32 = 3e-2;

/// Newton–Schulz runs in fp32 on both sides, but five quintic iterations of
/// GEMMs amplify summation-order differences between the CPU loops and the
/// register-tiled GPU kernels (the map's slope reaches ~3.4 per iteration on
/// the smallest singular directions). Real defects — a wrong coefficient or
/// a transposed operand — sit orders of magnitude above this.
const NS_ATOL: f32 = 5e-4;
const NS_RTOL: f32 = 5e-3;

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

/// Direct parity of the GPU zeroth power against the CPU reference on
/// square, wide, and tall matrices; tall exercises the transpose-free
/// `X = aX + XB` iteration.
fn newton_schulz_parity(
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    gemm: &model::gemm_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    fn check<const R: usize, const C: usize>(
        name: &str,
        seed: u64,
        stream: &cuda_core::CudaStream,
        tensor: &model::tensor_kernels::LoadedModule,
        gemm: &model::gemm_kernels::LoadedModule,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let values: Vec<f32> = uniform_vec(R * C, seed)
            .iter()
            .map(|&value| value - 0.5)
            .collect();
        let expected =
            zeroth_power_via_newton_schulz::<R, C>(&CpuTensor::from_slice(&values), 5);
        let input = DeviceBuffer::from_host(stream, &values)?;
        let mut scratch = GpuMuonScratch::new(stream, R * C, R * C, R.min(C))?;
        let actual = scratch.zeroth_power(&input, R, C, 5, stream, tensor, gemm)?;
        assert_close_slices(name, &actual, expected.as_slice(), NS_ATOL, NS_RTOL);
        Ok(())
    }

    check::<D, D>("newton_schulz square", 11, stream, tensor, gemm)?;
    check::<FF, D>("newton_schulz wide", 12, stream, tensor, gemm)?;
    check::<D, FF>("newton_schulz tall", 13, stream, tensor, gemm)?;
    println!("✓ GPU Newton–Schulz zeroth power matches CPU (square/wide/tall)");
    Ok(())
}

fn muon_overfit_tiny_batch(
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
    // AdamW stays at the 0.02 the plain-AdamW gate settled on (0.03 is
    // knife-edge on this batch); Muon's default 0.02 handles the hidden
    // matrices.
    let mut optimizer = GpuLlamaMuon::new(
        stream,
        MuonConfig::default(),
        AdamWConfig {
            learning_rate: 0.02,
            weight_decay: 0.0,
            ..AdamWConfig::default()
        },
    )?;
    let mut workspace = GpuLlamaWorkspace::<4, 128, 4, 4, 128, 128, 2, 12>::new(stream)?;
    let mut initial_loss = None;

    for _ in 0..600 {
        gpu.zero_grad(stream, tensor)?;
        gpu.forward(
            &tokens,
            &targets,
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
        optimizer.update(&mut gpu, stream, tensor, gemm)?;
    }

    gpu.forward(
        &tokens,
        &targets,
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
        "GPU tiny batch did not overfit with Muon: initial={initial_loss}, final={final_loss}"
    );
    assert!(
        final_loss < initial_loss * 0.05,
        "GPU Muon loss did not fall enough: initial={initial_loss}, final={final_loss}"
    );
    println!("✓ GPU Muon overfits a tiny batch ({initial_loss:.6} -> {final_loss:.6})");
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
    // 0.03 is knife-edge on this batch: the bf16 two-logit tie's escape is
    // violently sensitive there, and a CPU sweep injecting +/-1-ulp-scale
    // noise (modelling kernel summation-order differences, 7e7) left ~1 in 8
    // realizations parked on the tie past step 900. At 0.02 every sampled
    // realization converges by ~step 60; the plateau-escape mechanism itself
    // stays observable in crates/optim/examples/overfit_probe.rs at 0.03.
    let config = AdamWConfig {
        learning_rate: 0.02,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut optimizer = GpuLlamaAdamW::new(stream, config)?;
    let mut workspace = GpuLlamaWorkspace::<4, 128, 4, 4, 128, 128, 2, 12>::new(stream)?;
    let mut initial_loss = None;

    // At this learning rate the CPU probe converges by ~step 60 across all
    // sampled sub-ulp noise realizations; 600 steps keeps a wide margin.
    for _ in 0..600 {
        gpu.zero_grad(stream, tensor)?;
        gpu.forward(
            &tokens,
            &targets,
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
        &tokens,
        &targets,
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

/// End-to-end gate for the tcgen05 block-linear path (7e9).
///
/// Every shape here is a multiple of the 128 tile, so the block linears run
/// the integrated bf16 path: activation quantize, the two weight-gradient
/// transposes, the prefix TMA maps, the fp32-output epilogues, and the
/// post-AdamW compute-weight refresh. One forward/backward is compared
/// against the CPU reference under the bf16 tolerances, then the same model
/// must overfit a deterministic token mapping — which fails if any operand
/// orientation, map selection, or master/compute sync is wrong.
fn aligned_tcgen05_linears(
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    gemm: &model::gemm_kernels::LoadedModule,
    gemm_bf16: &model::Tcgen05Gemm,
    flash: &model::flash_kernels::LoadedModule,
    llama: &model::llama_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const NA: usize = 128;
    const TA: usize = 4;
    const VA: usize = 17;
    const VPA: usize = 128;
    const DA: usize = 128;
    const HA: usize = 2;
    const FFA: usize = 128;

    let mut cpu = Llama::<NA, TA, VA, DA, HA, HD, FFA>::new(7);
    let mut gpu = GpuLlama::<NA, NA, TA, VA, VPA, DA, HA, HD, FFA>::from_cpu(stream, &cpu)?;
    let mut workspace = GpuLlamaWorkspace::<NA, NA, TA, VA, VPA, DA, HA, FFA>::new(stream)?;
    assert!(
        workspace.tcgen05_linears_active(),
        "aligned gate is not exercising the tcgen05 block-linear path"
    );

    // Target is a fixed function of the current token: exact for parity and
    // learnable to ~zero loss by the overfit loop below.
    let tokens: [usize; NA] = std::array::from_fn(|i| (i * 7 + 3) % VA);
    let targets: [usize; NA] = std::array::from_fn(|i| (tokens[i] + 1) % VA);

    let (cpu_loss, cpu_ctx) = cpu.forward(tokens, targets);
    gpu.forward(
        &tokens, &targets, &mut workspace, stream, tensor, gemm, gemm_bf16, flash, llama,
    )?;
    assert_close(
        "aligned loss",
        workspace.loss(),
        &cpu_loss,
        stream,
        BF16_ATOL,
        BF16_RTOL,
    )?;

    cpu.backward(cpu_ctx);
    gpu.backward(&mut workspace, stream, tensor, gemm, gemm_bf16, flash, llama)?;

    macro_rules! grad {
        ($field:ident) => {
            assert_close(
                concat!("aligned ", stringify!($field), ".dw"),
                &gpu.$field.dw,
                &cpu.$field.dw,
                stream,
                BF16_ATOL,
                BF16_RTOL,
            )?;
        };
    }
    grad!(embedding);
    grad!(attention_norm);
    assert_grouped_close(
        "aligned qkv_proj.dw",
        &gpu.qkv_proj.dw,
        [&cpu.q_proj.dw, &cpu.k_proj.dw, &cpu.v_proj.dw],
        stream,
        BF16_ATOL,
        BF16_RTOL,
    )?;
    grad!(o_proj);
    grad!(ffn_norm);
    assert_grouped_close(
        "aligned gate_up_proj.dw",
        &gpu.gate_up_proj.dw,
        [&cpu.gate_proj.dw, &cpu.up_proj.dw],
        stream,
        BF16_ATOL,
        BF16_RTOL,
    )?;
    grad!(down_proj);
    grad!(final_norm);
    println!("✓ aligned tcgen05 block linears match CPU forward/backward");

    let config = AdamWConfig {
        learning_rate: 0.02,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut optimizer = GpuLlamaAdamW::new(stream, config)?;
    let mut initial_loss = None;
    for _ in 0..600 {
        gpu.zero_grad(stream, tensor)?;
        gpu.forward(
            &tokens, &targets, &mut workspace, stream, tensor, gemm, gemm_bf16, flash, llama,
        )?;
        if initial_loss.is_none() {
            initial_loss = Some(workspace.loss().to_host(stream)?[0]);
        }
        gpu.backward(&mut workspace, stream, tensor, gemm, gemm_bf16, flash, llama)?;
        optimizer.update(&mut gpu, stream, tensor)?;
    }
    gpu.forward(
        &tokens, &targets, &mut workspace, stream, tensor, gemm, gemm_bf16, flash, llama,
    )?;
    let final_loss = workspace.loss().to_host(stream)?[0];
    let initial_loss = initial_loss.expect("training loop runs at least once");
    assert!(
        final_loss < 0.05,
        "aligned tcgen05 model did not overfit: initial={initial_loss}, final={final_loss}"
    );
    assert!(
        final_loss < initial_loss * 0.05,
        "aligned tcgen05 loss did not fall enough: initial={initial_loss}, final={final_loss}"
    );
    println!(
        "✓ aligned tcgen05 block linears overfit ({initial_loss:.6} -> {final_loss:.6})"
    );
    Ok(())
}

/// Muon counterpart of the aligned overfit gate: the same tile-aligned
/// shapes route forward/backward through the bf16 tcgen05 block linears
/// while the optimizer orthogonalizes with fp32 register GEMMs, so this
/// fails if the Muon update or the post-Muon master→compute sync is wrong.
fn aligned_muon_overfit(
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    gemm: &model::gemm_kernels::LoadedModule,
    gemm_bf16: &model::Tcgen05Gemm,
    flash: &model::flash_kernels::LoadedModule,
    llama: &model::llama_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const NA: usize = 128;
    const TA: usize = 4;
    const VA: usize = 17;
    const VPA: usize = 128;
    const DA: usize = 128;
    const HA: usize = 2;
    const FFA: usize = 128;

    let cpu = Llama::<NA, TA, VA, DA, HA, HD, FFA>::new(7);
    let mut gpu = GpuLlama::<NA, NA, TA, VA, VPA, DA, HA, HD, FFA>::from_cpu(stream, &cpu)?;
    let mut workspace = GpuLlamaWorkspace::<NA, NA, TA, VA, VPA, DA, HA, FFA>::new(stream)?;
    assert!(
        workspace.tcgen05_linears_active(),
        "aligned Muon gate is not exercising the tcgen05 block-linear path"
    );
    let tokens: [usize; NA] = std::array::from_fn(|i| (i * 7 + 3) % VA);
    let targets: [usize; NA] = std::array::from_fn(|i| (tokens[i] + 1) % VA);

    let mut optimizer = GpuLlamaMuon::new(
        stream,
        MuonConfig::default(),
        AdamWConfig {
            learning_rate: 0.02,
            weight_decay: 0.0,
            ..AdamWConfig::default()
        },
    )?;
    let mut initial_loss = None;
    for _ in 0..600 {
        gpu.zero_grad(stream, tensor)?;
        gpu.forward(
            &tokens, &targets, &mut workspace, stream, tensor, gemm, gemm_bf16, flash, llama,
        )?;
        if initial_loss.is_none() {
            initial_loss = Some(workspace.loss().to_host(stream)?[0]);
        }
        gpu.backward(&mut workspace, stream, tensor, gemm, gemm_bf16, flash, llama)?;
        optimizer.update(&mut gpu, stream, tensor, gemm)?;
    }
    gpu.forward(
        &tokens, &targets, &mut workspace, stream, tensor, gemm, gemm_bf16, flash, llama,
    )?;
    let final_loss = workspace.loss().to_host(stream)?[0];
    let initial_loss = initial_loss.expect("training loop runs at least once");
    assert!(
        final_loss < 0.05,
        "aligned tcgen05 model did not overfit with Muon: initial={initial_loss}, final={final_loss}"
    );
    assert!(
        final_loss < initial_loss * 0.05,
        "aligned tcgen05 Muon loss did not fall enough: initial={initial_loss}, final={final_loss}"
    );
    println!("✓ aligned tcgen05 block linears overfit with Muon ({initial_loss:.6} -> {final_loss:.6})");
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
        &tokens,
        &targets,
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
        &tokens,
        &targets,
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

    println!("✓ full GPU Llama forward/backward and AdamW (bf16 lm-head) match CPU");

    // Muon parity on the exact same GPU gradients: the hidden matrices take
    // the Newton–Schulz update while everything else takes a second AdamW
    // step. Both sides enter within the fp32 AdamW tolerance of each other,
    // so hidden weights carry Newton–Schulz tolerances and the elementwise
    // paths stay tight.
    let muon_config = MuonConfig {
        learning_rate: 0.05,
        weight_decay: 0.1,
        ..MuonConfig::default()
    };
    let mut cpu_muon = LlamaMuon::new(muon_config, config);
    let mut gpu_muon = GpuLlamaMuon::new(&stream, muon_config, config)?;
    cpu_muon.update(&mut cpu);
    gpu_muon.update(&mut gpu, &stream, &tensor, &gemm)?;

    assert_grouped_close(
        "qkv_proj momentum after Muon",
        &gpu_muon.qkv_proj.momentum,
        [
            &cpu_muon.q_proj.momentum,
            &cpu_muon.k_proj.momentum,
            &cpu_muon.v_proj.momentum,
        ],
        &stream,
        1e-6,
        1e-6,
    )?;
    assert_close(
        "down_proj momentum after Muon",
        &gpu_muon.down_proj.momentum,
        &cpu_muon.down_proj.momentum,
        &stream,
        1e-6,
        1e-6,
    )?;

    macro_rules! muon_weight {
        ($field:ident) => {
            assert_close(
                concat!(stringify!($field), ".w after Muon"),
                &gpu.$field.w,
                &cpu.$field.w,
                &stream,
                NS_ATOL,
                NS_RTOL,
            )?;
        };
    }
    assert_grouped_close(
        "qkv_proj.w after Muon",
        &gpu.qkv_proj.w,
        [&cpu.q_proj.w, &cpu.k_proj.w, &cpu.v_proj.w],
        &stream,
        NS_ATOL,
        NS_RTOL,
    )?;
    muon_weight!(o_proj);
    assert_grouped_close(
        "gate_up_proj.w after Muon",
        &gpu.gate_up_proj.w,
        [&cpu.gate_proj.w, &cpu.up_proj.w],
        &stream,
        NS_ATOL,
        NS_RTOL,
    )?;
    muon_weight!(down_proj);
    macro_rules! muon_aux_weight {
        ($field:ident) => {
            assert_close(
                concat!(stringify!($field), ".w after Muon's AdamW"),
                &gpu.$field.w,
                &cpu.$field.w,
                &stream,
                1e-5,
                1e-5,
            )?;
        };
    }
    muon_aux_weight!(embedding);
    muon_aux_weight!(attention_norm);
    muon_aux_weight!(ffn_norm);
    muon_aux_weight!(final_norm);
    let master = gpu.lm_head.master.to_host(&stream)?;
    assert_close_slices(
        "lm_head master after Muon's AdamW",
        &strip_vocab_padding("lm_head.master", &master),
        cpu.lm_head.w.as_slice(),
        1e-5,
        1e-5,
    );
    check_head_compute_copies(&gpu.lm_head, &stream)?;
    gpu.zero_grad(&stream, &tensor)?;
    println!("✓ full GPU Muon update (Newton–Schulz hidden matrices) matches CPU");

    newton_schulz_parity(&stream, &tensor, &gemm)?;
    overfit_tiny_batch(&stream, &tensor, &gemm, &gemm_bf16, &flash, &llama)?;
    muon_overfit_tiny_batch(&stream, &tensor, &gemm, &gemm_bf16, &flash, &llama)?;
    aligned_tcgen05_linears(&stream, &tensor, &gemm, &gemm_bf16, &flash, &llama)?;
    aligned_muon_overfit(&stream, &tensor, &gemm, &gemm_bf16, &flash, &llama)?;
    Ok(())
}
