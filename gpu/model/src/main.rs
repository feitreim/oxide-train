//! End-to-end forward/backward parity against `nn::Dense`.
//!
//! Saved activations are packed bf16 — the stream the projections read where
//! it lies — and gradients are fp32, so every quantity a forward pass carries
//! and everything downstream of it takes bf16 tolerances. The sums and the
//! statistics inside a kernel are still fp32, so the error is one rounding per
//! layer boundary, not one per operand. Master weights are bf16 on both sides (#57): the CPU
//! reference is snapped onto the bf16 grid at construction and after every
//! update, and the post-update comparison is made *on that grid* — see
//! [`assert_master_close`].
//!
//! The base dimensions exercise the 256-tiled lm-head and attention linears:
//! `N` = 8 real token rows inside padded `NP=256`, with `D=VP=256`. The odd
//! `VOCAB=17` exercises the classifier tail and `FF=19` retains an fp32
//! fallback case.
//!
//! [`aligned_tcgen05_linears`] runs a second, fully 256-aligned configuration
//! as the end-to-end gate for every bf16 tcgen05 block-linear path (7e9).

use cuda_core::{CudaContext, DeviceBuffer};
use nn::{Dense, ExpertFfn, Module, MoeDense};
use optim::{
    AdamWConfig, AuxLossSchedule, Bf16MasterInit, DenseAdamW, DenseMuon, MuonConfig,
    VisitCpuParameters, round_to_bf16_master, zeroth_power_via_newton_schulz,
};
use tensor_core::{Rank2, Rank3, Rank4, Shape, bf16, rng::uniform_vec};
use tensor_cpu::CpuTensor;

#[path = "lib.rs"]
mod model;
use model::{
    GpuDense, GpuDenseAdamW, GpuDenseDense, GpuDenseDenseAdamW, GpuDenseMuon, GpuDenseWorkspace,
    GpuExpertAdamW, GpuExpertFfn, GpuExpertWorkspace, GpuMoeWorkspace, GpuMuonScratch,
};

const N: usize = 8;
const NP: usize = 256;
const T: usize = 4;
const VOCAB: usize = 17;
const VP: usize = 256;
// `HD` must match the tiled flash kernels' compile-time head width (7e7); at
// HD=128 four heads (D=512) keep the tiny overfit gates converging.
const D: usize = 512;
const H: usize = 4;
const HD: usize = 128;
const FF: usize = 19;

/// Loss and gradients that crossed the bf16 head: inputs quantized to bf16,
/// fp32 accumulation, outputs re-rounded to bf16.
const BF16_ATOL: f32 = 3e-3;
const BF16_RTOL: f32 = 3e-2;

/// Post-update masters are compared on the bf16 grid, where one ulp is one
/// integer step. Both sides compute the update in fp32 from the same
/// gradients, so they agree to fp32 rounding — but that is enough to land on
/// either side of a bf16 rounding boundary, which moves the stored weight by
/// exactly one ulp. Anything larger is a real defect and still fails.
const MASTER_ULPS: i32 = 1;

/// Absolute floor carried over unchanged from the fp32-master gate. Kept
/// because the ulp budget is a relative statement and a near-cancelling update
/// (`w - lr·update` landing within 1e-8 of zero from an O(0.01) weight) has
/// unbounded relative error on both sides.
const MASTER_ATOL: f32 = 2e-6;

/// Budget for a master whose *input* was already a rounding apart: the second
/// update in a sequence inherits the first's ulp and can add its own.
const MASTER_ULPS_CHAINED: i32 = 2;

/// Newton–Schulz runs in fp32 on both sides, but five quintic iterations of
/// GEMMs amplify summation-order differences between the CPU loops and the
/// register-tiled GPU kernels (the map's slope reaches ~3.4 per iteration on
/// the smallest singular directions). Real defects — a wrong coefficient or
/// a transposed operand — sit orders of magnitude above this.
const NS_ATOL: f32 = 5e-4;
const NS_RTOL: f32 = 5e-3;

/// One bf16 ulp, relative: bf16 keeps seven stored mantissa bits.
const BF16_ULP_RTOL: f32 = 1.0 / 128.0;

/// Muon's hidden matrices land in a bf16 master, so the Newton–Schulz
/// divergence above and the master's own rounding stack. The iteration's
/// spread is far wider than an ulp here, which is why this path keeps a float
/// tolerance instead of the grid comparison the AdamW masters use.
const MUON_MASTER_RTOL: f32 = NS_RTOL + BF16_ULP_RTOL;

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

/// Total order over bf16 values, so "one ulp apart" is one integer step.
fn bf16_order(value: f32) -> i32 {
    let bits = bf16::from_f32(value).to_bits();
    if bits & 0x8000 != 0 {
        -((bits & 0x7fff) as i32)
    } else {
        bits as i32
    }
}

/// Compare two bf16 masters on the bf16 grid instead of with a relative float
/// tolerance.
///
/// Both sides store these parameters as bf16 (#57) and take the same
/// gradients, so their fp32 updates agree to fp32 rounding — but a difference
/// that small can straddle a bf16 rounding boundary and move the *stored*
/// weight a whole ulp (2^-8 to 2^-7 relative). A relative tolerance wide enough
/// to allow that would be wide enough to hide a real defect; the ulp distance
/// is the honest statement, and it fails loudly on anything larger.
///
/// [`MASTER_ATOL`] still applies beside it, unchanged from the fp32-master
/// gate: where a step's update nearly cancels the weight, the result's relative
/// error is unbounded while its absolute error is not, and an ulp count says
/// nothing useful about it.
fn assert_master_close(name: &str, actual: &[f32], expected: &[f32], max_ulps: i32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let ulps = (bf16_order(a) - bf16_order(e)).abs();
        assert!(
            ulps <= max_ulps || (a - e).abs() <= MASTER_ATOL,
            "{name} mismatch at {i}: gpu={a}, cpu={e}, {ulps} bf16 ulps apart \
             (max {max_ulps}, atol {MASTER_ATOL})"
        );
    }
}

fn assert_master_tensor_close<S: Shape>(
    name: &str,
    gpu: &model::tensor_device::GpuBf16Tensor<S>,
    cpu: &CpuTensor<f32, S>,
    stream: &cuda_core::CudaStream,
    max_ulps: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_master_close(name, &gpu.to_f32_host(stream)?, cpu.as_slice(), max_ulps);
    Ok(())
}

/// Interleave `GROUPS` separate `[IN, OUT]` reference matrices into the GPU's
/// `[IN, GROUPS, OUT]` layout.
fn interleave_groups<const IN: usize, const GROUPS: usize, const OUT: usize>(
    expected: [&CpuTensor<f32, Rank2<IN, OUT>>; GROUPS],
) -> Vec<f32> {
    let mut interleaved = Vec::with_capacity(IN * GROUPS * OUT);
    for input in 0..IN {
        for expected in expected {
            interleaved.extend_from_slice(&expected.as_slice()[input * OUT..(input + 1) * OUT]);
        }
    }
    interleaved
}

fn assert_grouped_master_close<const IN: usize, const GROUPS: usize, const OUT: usize>(
    name: &str,
    gpu: &model::tensor_device::GpuBf16Tensor<Rank3<IN, GROUPS, OUT>>,
    expected: [&CpuTensor<f32, Rank2<IN, OUT>>; GROUPS],
    stream: &cuda_core::CudaStream,
    max_ulps: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_master_close(
        name,
        &gpu.to_f32_host(stream)?,
        &interleave_groups(expected),
        max_ulps,
    );
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

/// Every gradient of a dense model, copied off the device.
///
/// An AdamW step clears each gradient as it consumes it, so the Muon parity
/// below — which is deliberately run over *the exact same* gradients the AdamW
/// step just saw — has to put them back first.
struct DenseGradients {
    embedding: Vec<f32>,
    attention_norm: Vec<f32>,
    qkv_proj: Vec<f32>,
    o_proj: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate_up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    final_norm: Vec<f32>,
    lm_head: Vec<u32>,
}

impl DenseGradients {
    fn save<
        const N: usize,
        const NP: usize,
        const T: usize,
        const VOCAB: usize,
        const VP: usize,
        const D: usize,
        const H: usize,
        const HD: usize,
        const FF: usize,
    >(
        gpu: &model::GpuDenseDense<N, NP, T, VOCAB, VP, D, H, HD, FF>,
        stream: &cuda_core::CudaStream,
    ) -> Result<Self, cuda_core::DriverError> {
        Ok(Self {
            embedding: gpu.embedding.dw.to_host(stream)?,
            attention_norm: gpu.attention_norm.dw.to_host(stream)?,
            qkv_proj: gpu.qkv_proj.dw.to_host(stream)?,
            o_proj: gpu.o_proj.dw.to_host(stream)?,
            ffn_norm: gpu.ffn_norm.dw.to_host(stream)?,
            gate_up_proj: gpu.gate_up_proj.dw.to_host(stream)?,
            down_proj: gpu.down_proj.dw.to_host(stream)?,
            final_norm: gpu.final_norm.dw.to_host(stream)?,
            lm_head: gpu.lm_head.dw_words().to_host_vec(stream)?,
        })
    }

    fn restore<
        const N: usize,
        const NP: usize,
        const T: usize,
        const VOCAB: usize,
        const VP: usize,
        const D: usize,
        const H: usize,
        const HD: usize,
        const FF: usize,
    >(
        &self,
        gpu: &mut model::GpuDenseDense<N, NP, T, VOCAB, VP, D, H, HD, FF>,
        stream: &cuda_core::CudaStream,
    ) -> Result<(), cuda_core::DriverError> {
        macro_rules! restore {
            ($field:ident) => {
                gpu.$field
                    .dw
                    .as_device_buffer_mut()
                    .copy_from_host(stream, &self.$field)?;
            };
        }
        restore!(embedding);
        restore!(attention_norm);
        restore!(qkv_proj);
        restore!(o_proj);
        restore!(ffn_norm);
        restore!(gate_up_proj);
        restore!(down_proj);
        restore!(final_norm);
        gpu.lm_head
            .dw_words_mut()
            .copy_from_host(stream, &self.lm_head)
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

fn stacked_expert_gradients<const E: usize, const C: usize, const D: usize, const FF: usize>(
    experts: &[ExpertFfn<C, D, FF>; E],
) -> (
    CpuTensor<f32, Rank4<E, D, 2, FF>>,
    CpuTensor<f32, Rank3<E, FF, D>>,
) {
    let mut gate_up = vec![0.0; E * D * 2 * FF];
    let mut down = vec![0.0; E * FF * D];
    for (expert, source) in experts.iter().enumerate() {
        for input in 0..D {
            let destination = (expert * D + input) * 2 * FF;
            gate_up[destination..destination + FF]
                .copy_from_slice(&source.gate_proj.dw.as_slice()[input * FF..(input + 1) * FF]);
            gate_up[destination + FF..destination + 2 * FF]
                .copy_from_slice(&source.up_proj.dw.as_slice()[input * FF..(input + 1) * FF]);
        }
        down[expert * FF * D..(expert + 1) * FF * D]
            .copy_from_slice(source.down_proj.dw.as_slice());
    }
    (
        CpuTensor::from_slice(&gate_up),
        CpuTensor::from_slice(&down),
    )
}

fn expected_global_transpose_words(values: &[f32], rows: usize, columns: usize) -> Vec<u32> {
    let rounded = unpack_bf16(&pack_bf16(values));
    let mut transposed = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            transposed[column * rows + row] = rounded[row * columns + column];
        }
    }
    pack_bf16(&transposed)
}

fn check_expert_compute_copies<const E: usize, const D: usize, const FF: usize>(
    experts: &GpuExpertFfn<E, D, FF>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    let gate_up_master = experts.gate_up.to_f32_host(stream)?;
    let gate_up_t = experts
        .gate_up_transposed_words()
        .expect("aligned experts must own gate/up compute weights");
    assert_eq!(
        gate_up_t.to_host_vec(stream)?,
        expected_global_transpose_words(&gate_up_master, E * D, 2 * FF),
        "expert gate/up transposed compute copy is stale"
    );

    let down_master = experts.down.to_f32_host(stream)?;
    let down_t = experts
        .down_transposed_words()
        .expect("aligned experts must own down compute weights");
    assert_eq!(
        down_t.to_host_vec(stream)?,
        expected_global_transpose_words(&down_master, E * FF, D),
        "expert down transposed compute copy is stale"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn expert_compute_parity<const E: usize, const C: usize, const D: usize, const FF: usize>(
    label: &str,
    aligned: bool,
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    gemm: &model::gemm_kernels::LoadedModule,
    gemm_bf16: &model::Tcgen05Gemm,
    dense: &model::dense_kernels::LoadedModule,
    reference: &model::dense_reference_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut cpu: [ExpertFfn<C, D, FF>; E] = std::array::from_fn(|expert| {
        let mut expert = ExpertFfn::initialized(700 + 3 * expert as u64);
        // The GPU rounds these on upload; the reference has to start there too.
        round_to_bf16_master(&mut expert.gate_proj.w);
        round_to_bf16_master(&mut expert.up_proj.w);
        round_to_bf16_master(&mut expert.down_proj.w);
        expert
    });
    let mut gpu = GpuExpertFfn::from_cpu(stream, &cpu)?;
    let mut workspace = GpuExpertWorkspace::<E, C, D, FF>::new(stream)?;
    assert_eq!(
        workspace.tcgen05_active(),
        aligned,
        "{label}: workspace selected the wrong expert GEMM path"
    );
    gpu.zero_grad(stream, tensor)?;

    let live_slots = C.saturating_sub(2).max(1);
    let mut bins: Vec<f32> = uniform_vec(E * C * D, 801)
        .into_iter()
        .map(|value| value - 0.5)
        .collect();
    for expert in 0..E {
        for slot in live_slots..C {
            bins[(expert * C + slot) * D..(expert * C + slot + 1) * D].fill(0.0);
        }
    }

    let forward_pairs: [(CpuTensor<f32, Rank2<C, D>>, nn::ExpertFfnCtx<C, D, FF>); E] =
        std::array::from_fn(|expert| {
            let start = expert * C * D;
            cpu[expert].forward(CpuTensor::from_slice(&bins[start..start + C * D]))
        });
    let mut expected_output = vec![0.0; E * C * D];
    for expert in 0..E {
        expected_output[expert * C * D..(expert + 1) * C * D]
            .copy_from_slice(forward_pairs[expert].0.as_slice());
    }
    let contexts = forward_pairs.map(|(_, context)| context);

    workspace.upload_bins(&bins, stream)?;
    gpu.forward(&mut workspace, stream, tensor, gemm, gemm_bf16, dense)?;
    let sparse_output = workspace.acts.bin_output.to_host(stream)?;
    let (atol, rtol) = if aligned {
        (BF16_ATOL, BF16_RTOL)
    } else {
        (1e-5, 1e-5)
    };
    assert_close_slices(
        &format!("{label} forward"),
        &sparse_output,
        &expected_output,
        atol,
        rtol,
    );

    for expert in 0..E {
        for slot in live_slots..C {
            for feature in 0..D {
                let index = (expert * C + slot) * D + feature;
                assert_eq!(
                    sparse_output[index].to_bits(),
                    0.0f32.to_bits(),
                    "{label}: dead bin row produced a non-zero output at {index}"
                );
            }
        }
    }

    // Filling the formerly dead rows makes every expert exactly full. Since
    // rows do not interact, all original live outputs must remain bit-identical.
    let mut full_bins = bins.clone();
    let fill: Vec<f32> = uniform_vec(E * C * D, 802)
        .into_iter()
        .map(|value| value - 0.5)
        .collect();
    for expert in 0..E {
        for slot in live_slots..C {
            let start = (expert * C + slot) * D;
            full_bins[start..start + D].copy_from_slice(&fill[start..start + D]);
        }
    }
    workspace.upload_bins(&full_bins, stream)?;
    gpu.forward(&mut workspace, stream, tensor, gemm, gemm_bf16, dense)?;
    let full_output = workspace.acts.bin_output.to_host(stream)?;
    for expert in 0..E {
        for slot in 0..live_slots {
            for feature in 0..D {
                let index = (expert * C + slot) * D + feature;
                assert_eq!(
                    sparse_output[index].to_bits(),
                    full_output[index].to_bits(),
                    "{label}: live output changed when dead rows were filled at {index}"
                );
            }
        }
    }

    // Restore the sparse forward state before backward.
    workspace.upload_bins(&bins, stream)?;
    gpu.forward(&mut workspace, stream, tensor, gemm, gemm_bf16, dense)?;
    let output_gradient: Vec<f32> = uniform_vec(E * C * D, 803)
        .into_iter()
        .map(|value| (value - 0.5) * 0.05)
        .collect();
    workspace.upload_output_gradient(&output_gradient, stream)?;

    let mut contexts = contexts.into_iter();
    let expected_input_gradients: [_; E] = std::array::from_fn(|expert| {
        let start = expert * C * D;
        cpu[expert].backward(
            contexts.next().unwrap(),
            CpuTensor::from_slice(&output_gradient[start..start + C * D]),
        )
    });
    let mut expected_input_gradient = vec![0.0; E * C * D];
    for expert in 0..E {
        expected_input_gradient[expert * C * D..(expert + 1) * C * D]
            .copy_from_slice(expected_input_gradients[expert].as_slice());
    }
    gpu.backward(&mut workspace, stream, tensor, gemm, gemm_bf16, dense)?;
    assert_close_slices(
        &format!("{label} input gradient"),
        &workspace.scratch.d_bin_input.to_host(stream)?,
        &expected_input_gradient,
        atol,
        rtol,
    );
    let (expected_gate_up, expected_down) = stacked_expert_gradients(&cpu);
    assert_close(
        &format!("{label} gate/up gradient"),
        &gpu.d_gate_up,
        &expected_gate_up,
        stream,
        atol,
        rtol,
    )?;
    assert_close(
        &format!("{label} down gradient"),
        &gpu.d_down,
        &expected_down,
        stream,
        atol,
        rtol,
    )?;

    // A second backward without zero_grad must accumulate into both stacked
    // gradient entries, matching the CPU module's += contract.
    let second_pairs: [(CpuTensor<f32, Rank2<C, D>>, nn::ExpertFfnCtx<C, D, FF>); E] =
        std::array::from_fn(|expert| {
            let start = expert * C * D;
            cpu[expert].forward(CpuTensor::from_slice(&bins[start..start + C * D]))
        });
    let second_contexts = second_pairs.map(|(_, context)| context);
    let mut second_contexts = second_contexts.into_iter();
    for expert in 0..E {
        let start = expert * C * D;
        cpu[expert].backward(
            second_contexts.next().unwrap(),
            CpuTensor::from_slice(&output_gradient[start..start + C * D]),
        );
    }
    gpu.forward(&mut workspace, stream, tensor, gemm, gemm_bf16, dense)?;
    gpu.backward(&mut workspace, stream, tensor, gemm, gemm_bf16, dense)?;
    let (expected_gate_up, expected_down) = stacked_expert_gradients(&cpu);
    assert_close(
        &format!("{label} accumulated gate/up gradient"),
        &gpu.d_gate_up,
        &expected_gate_up,
        stream,
        2.0 * atol,
        rtol,
    )?;
    assert_close(
        &format!("{label} accumulated down gradient"),
        &gpu.d_down,
        &expected_down,
        stream,
        2.0 * atol,
        rtol,
    )?;

    if aligned {
        let mut optimizer = GpuExpertAdamW::new(
            stream,
            AdamWConfig {
                learning_rate: 1e-3,
                ..AdamWConfig::default()
            },
        )?;
        optimizer.update(&mut gpu, stream, tensor)?;
        check_expert_compute_copies(&gpu, stream)?;
    }

    println!("✓ {label} expert forward/backward, zero rows, accumulation, and compute sync");
    Ok(())
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
    gpu: &GpuDenseDense<N, NP, T, VOCAB, VP, D, H, HD, FF>,
    cpu: &Dense<N, T, VOCAB, D, H, HD, FF>,
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

/// `w_t` is the element transpose of the rounded master. The untransposed
/// operand needs no check since #58: it is the master's own words.
fn check_head_compute_copies(
    head: &model::GpuBf16Head<D, VP>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    let master = head.master.to_f32_host(stream)?;
    let rounded = unpack_bf16(&pack_bf16(&master));
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
        let expected = zeroth_power_via_newton_schulz::<R, C>(&CpuTensor::from_slice(&values), 5);
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
    flash_bf16: &model::Tcgen05Flash,
    dense: &model::dense_kernels::LoadedModule,
    reference: &model::dense_reference_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    type TinyDense = Dense<4, 4, 4, 512, 4, 128, 12>;
    let tokens = [0, 1, 2, 3];
    let targets = [1, 2, 3, 0];
    let cpu = TinyDense::new(100);
    let mut gpu = GpuDenseDense::<4, 256, 4, 4, 256, 512, 4, 128, 12>::from_cpu(stream, &cpu)?;
    // AdamW stays at the 0.02 the plain-AdamW gate settled on (0.03 is
    // knife-edge on this batch); Muon's default 0.02 handles the hidden
    // matrices.
    let mut optimizer = GpuDenseMuon::new(
        stream,
        MuonConfig::default(),
        AdamWConfig {
            learning_rate: 0.02,
            weight_decay: 0.0,
            ..AdamWConfig::default()
        },
    )?;
    let mut workspace = GpuDenseWorkspace::<4, 256, 4, 4, 256, 512, 4, 12>::new(stream)?;
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
            flash_bf16,
            dense,
            reference,
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
            flash_bf16,
            dense,
            reference,
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
        flash_bf16,
        dense,
        reference,
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
    flash_bf16: &model::Tcgen05Flash,
    dense: &model::dense_kernels::LoadedModule,
    reference: &model::dense_reference_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    type TinyDense = Dense<4, 4, 4, 512, 4, 128, 12>;
    let tokens = [0, 1, 2, 3];
    let targets = [1, 2, 3, 0];
    let cpu = TinyDense::new(100);
    let mut gpu = GpuDenseDense::<4, 256, 4, 4, 256, 512, 4, 128, 12>::from_cpu(stream, &cpu)?;
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
    let mut optimizer = GpuDenseDenseAdamW::new(stream, config)?;
    let mut workspace = GpuDenseWorkspace::<4, 256, 4, 4, 256, 512, 4, 12>::new(stream)?;
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
            flash_bf16,
            dense,
            reference,
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
            flash_bf16,
            dense,
            reference,
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
        flash_bf16,
        dense,
        reference,
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
/// Every GEMM dimension here is a multiple of the 256 tile, so the block linears run
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
    flash_bf16: &model::Tcgen05Flash,
    dense: &model::dense_kernels::LoadedModule,
    reference: &model::dense_reference_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const NA: usize = 256;
    const TA: usize = 4;
    const VA: usize = 17;
    const VPA: usize = 256;
    const DA: usize = 512;
    const HA: usize = 4;
    const FFA: usize = 256;

    let mut cpu = Dense::<NA, TA, VA, DA, HA, HD, FFA>::new(7);
    cpu.visit_cpu_parameters(&mut Bf16MasterInit);
    let mut gpu = GpuDenseDense::<NA, NA, TA, VA, VPA, DA, HA, HD, FFA>::from_cpu(stream, &cpu)?;
    let mut workspace = GpuDenseWorkspace::<NA, NA, TA, VA, VPA, DA, HA, FFA>::new(stream)?;
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
        &tokens,
        &targets,
        &mut workspace,
        stream,
        tensor,
        gemm,
        gemm_bf16,
        flash,
        flash_bf16,
        dense,
        reference,
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
    gpu.backward(
        &mut workspace,
        stream,
        tensor,
        gemm,
        gemm_bf16,
        flash,
        flash_bf16,
        dense,
        reference,
    )?;

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
    let mut optimizer = GpuDenseDenseAdamW::new(stream, config)?;
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
            flash_bf16,
            dense,
            reference,
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
            flash_bf16,
            dense,
            reference,
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
        flash_bf16,
        dense,
        reference,
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
    println!("✓ aligned tcgen05 block linears overfit ({initial_loss:.6} -> {final_loss:.6})");
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
    flash_bf16: &model::Tcgen05Flash,
    dense: &model::dense_kernels::LoadedModule,
    reference: &model::dense_reference_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const NA: usize = 256;
    const TA: usize = 4;
    const VA: usize = 17;
    const VPA: usize = 256;
    const DA: usize = 512;
    const HA: usize = 4;
    const FFA: usize = 256;

    let cpu = Dense::<NA, TA, VA, DA, HA, HD, FFA>::new(7);
    let mut gpu = GpuDenseDense::<NA, NA, TA, VA, VPA, DA, HA, HD, FFA>::from_cpu(stream, &cpu)?;
    let mut workspace = GpuDenseWorkspace::<NA, NA, TA, VA, VPA, DA, HA, FFA>::new(stream)?;
    assert!(
        workspace.tcgen05_linears_active(),
        "aligned Muon gate is not exercising the tcgen05 block-linear path"
    );
    let tokens: [usize; NA] = std::array::from_fn(|i| (i * 7 + 3) % VA);
    let targets: [usize; NA] = std::array::from_fn(|i| (tokens[i] + 1) % VA);

    let mut optimizer = GpuDenseMuon::new(
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
            &tokens,
            &targets,
            &mut workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
            reference,
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
            flash_bf16,
            dense,
            reference,
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
        flash_bf16,
        dense,
        reference,
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
    println!(
        "✓ aligned tcgen05 block linears overfit with Muon ({initial_loss:.6} -> {final_loss:.6})"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn moe_model_parity<
    const MN: usize,
    const MNP: usize,
    const MT: usize,
    const MFF: usize,
    const ME: usize,
    const MK: usize,
    const MC: usize,
    const ML: usize,
>(
    label: &str,
    expect_tcgen05: bool,
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    gemm: &model::gemm_kernels::LoadedModule,
    gemm_bf16: &model::Tcgen05Gemm,
    flash: &model::flash_kernels::LoadedModule,
    flash_bf16: &model::Tcgen05Flash,
    dense: &model::dense_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const AUX: f32 = 0.02;
    let mut cpu = MoeDense::<MN, MT, VOCAB, D, H, HD, MFF, ME, MK, MC, ML>::new(71, AUX);
    cpu.visit_cpu_parameters(&mut Bf16MasterInit);
    // Deterministic ties force experts 0/1 over capacity while all remaining
    // experts stay underfull, covering both dispatch edge cases.
    for block in &mut cpu.blocks {
        block.ffn.router.w.as_mut_slice().fill(0.0);
    }
    let mut gpu =
        GpuDense::<MN, MNP, MT, VOCAB, VP, D, H, HD, MFF, ME, MK, MC, ML>::from_cpu(stream, &cpu)?;
    let mut workspace =
        GpuMoeWorkspace::<MN, MNP, MT, VOCAB, VP, D, H, MFF, ME, MK, MC, ML>::new(stream)?;
    assert_eq!(workspace.tcgen05_linears_active(), expect_tcgen05);
    assert_eq!(workspace.tcgen05_experts_active(), expect_tcgen05);
    let tokens: [usize; MN] = std::array::from_fn(|i| (i * 7 + 3) % VOCAB);
    let targets: [usize; MN] = std::array::from_fn(|i| (tokens[i] + 1) % VOCAB);

    let (cpu_loss, cpu_ctx) = cpu.forward(tokens, targets);
    gpu.forward(
        &tokens,
        &targets,
        AUX,
        &mut workspace,
        stream,
        tensor,
        gemm,
        gemm_bf16,
        flash,
        flash_bf16,
        dense,
    )?;
    assert_close(
        &format!("{label} loss"),
        workspace.loss(),
        &cpu_loss,
        stream,
        BF16_ATOL,
        BF16_RTOL,
    )?;
    cpu.backward(cpu_ctx);
    gpu.backward(
        AUX,
        &mut workspace,
        stream,
        tensor,
        gemm,
        gemm_bf16,
        flash,
        flash_bf16,
        dense,
    )?;

    macro_rules! common_grad {
        ($field:ident) => {
            assert_close(
                concat!(stringify!($field), ".dw"),
                &gpu.$field.dw,
                &cpu.$field.dw,
                stream,
                BF16_ATOL,
                BF16_RTOL,
            )?;
        };
    }
    common_grad!(embedding);
    for (index, (gpu_block, cpu_block)) in gpu.blocks.iter().zip(cpu.blocks.iter()).enumerate() {
        assert_close(
            &format!("MoE block{index} attention_norm.dw"),
            &gpu_block.attention_norm.dw,
            &cpu_block.attention_norm.dw,
            stream,
            BF16_ATOL,
            BF16_RTOL,
        )?;
        assert_grouped_close(
            &format!("MoE block{index} qkv_proj.dw"),
            &gpu_block.qkv_proj.dw,
            [
                &cpu_block.q_proj.dw,
                &cpu_block.k_proj.dw,
                &cpu_block.v_proj.dw,
            ],
            stream,
            BF16_ATOL,
            BF16_RTOL,
        )?;
        assert_close(
            &format!("MoE block{index} o_proj.dw"),
            &gpu_block.o_proj.dw,
            &cpu_block.o_proj.dw,
            stream,
            BF16_ATOL,
            BF16_RTOL,
        )?;
        assert_close(
            &format!("MoE block{index} ffn_norm.dw"),
            &gpu_block.ffn_norm.dw,
            &cpu_block.ffn_norm.dw,
            stream,
            BF16_ATOL,
            BF16_RTOL,
        )?;
        assert_close(
            &format!("MoE block{index} router.dw"),
            &gpu_block.d_router,
            &cpu_block.ffn.router.dw,
            stream,
            BF16_ATOL,
            BF16_RTOL,
        )?;
        let mut expected_gate_up = Vec::with_capacity(ME * D * 2 * MFF);
        let mut expected_down = Vec::with_capacity(ME * MFF * D);
        for expert in &cpu_block.ffn.experts {
            for input in 0..D {
                expected_gate_up.extend_from_slice(
                    &expert.gate_proj.dw.as_slice()[input * MFF..(input + 1) * MFF],
                );
                expected_gate_up.extend_from_slice(
                    &expert.up_proj.dw.as_slice()[input * MFF..(input + 1) * MFF],
                );
            }
            expected_down.extend_from_slice(expert.down_proj.dw.as_slice());
        }
        assert_close_slices(
            &format!("MoE block{index} expert gate/up gradients"),
            &gpu_block.experts.d_gate_up.to_host(stream)?,
            &expected_gate_up,
            BF16_ATOL,
            BF16_RTOL,
        );
        assert_close_slices(
            &format!("MoE block{index} expert down gradients"),
            &gpu_block.experts.d_down.to_host(stream)?,
            &expected_down,
            BF16_ATOL,
            BF16_RTOL,
        );
    }
    common_grad!(final_norm);
    assert_close_slices(
        "MoE lm_head.dw",
        &head_gradient(&gpu.lm_head, stream)?,
        cpu.lm_head.dw.as_slice(),
        BF16_ATOL,
        BF16_RTOL,
    );
    println!("✓ {label} substituted MoE model matches CPU with drops and underfull experts");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn aligned_moe_overfit(
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    gemm: &model::gemm_kernels::LoadedModule,
    gemm_bf16: &model::Tcgen05Gemm,
    flash: &model::flash_kernels::LoadedModule,
    flash_bf16: &model::Tcgen05Flash,
    dense: &model::dense_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const ON: usize = 256;
    const OT: usize = 4;
    const OFF: usize = 256;
    const OE: usize = 2;
    const OK: usize = 1;
    const OC: usize = 256;
    let schedule = AuxLossSchedule {
        base_coefficient: 0.01,
        decay_horizon: 1_200.0,
    };
    let cpu =
        MoeDense::<ON, OT, VOCAB, D, H, HD, OFF, OE, OK, OC>::new(97, schedule.base_coefficient);
    let mut gpu =
        GpuDense::<ON, ON, OT, VOCAB, VP, D, H, HD, OFF, OE, OK, OC>::from_cpu(stream, &cpu)?;
    let mut workspace =
        GpuMoeWorkspace::<ON, ON, OT, VOCAB, VP, D, H, OFF, OE, OK, OC>::new(stream)?;
    let config = AdamWConfig {
        learning_rate: 0.02,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut optimizer = GpuDenseAdamW::new(stream, config, schedule, 1)?;
    let tokens: [usize; ON] = std::array::from_fn(|i| (i * 7 + 3) % VOCAB);
    let targets: [usize; ON] = std::array::from_fn(|i| (tokens[i] + 1) % VOCAB);
    let mut initial_loss = None;
    for _ in 0..1_200 {
        let coefficient = optimizer.aux_coefficient();
        gpu.zero_grad(stream, tensor)?;
        gpu.forward(
            &tokens,
            &targets,
            coefficient,
            &mut workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
        )?;
        if initial_loss.is_none() {
            initial_loss = Some(workspace.loss().to_host(stream)?[0]);
        }
        gpu.backward(
            coefficient,
            &mut workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
        )?;
        optimizer.update(&mut gpu, stream, tensor)?;
    }
    gpu.forward(
        &tokens,
        &targets,
        optimizer.aux_coefficient(),
        &mut workspace,
        stream,
        tensor,
        gemm,
        gemm_bf16,
        flash,
        flash_bf16,
        dense,
    )?;
    let initial_loss = initial_loss.expect("training loop runs");
    let final_loss = workspace.loss().to_host(stream)?[0];
    assert!(
        final_loss < 0.05 && final_loss < initial_loss * 0.05,
        "aligned MoE did not overfit: initial={initial_loss}, final={final_loss}"
    );
    println!(
        "✓ aligned tcgen05 MoE overfits with scheduled aux loss ({initial_loss:.6} -> {final_loss:.6})"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn moe_checkpoint_gate(
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    gemm: &model::gemm_kernels::LoadedModule,
    gemm_bf16: &model::Tcgen05Gemm,
    flash: &model::flash_kernels::LoadedModule,
    flash_bf16: &model::Tcgen05Flash,
    dense: &model::dense_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const CN: usize = 4;
    const CT: usize = 4;
    const CFF: usize = 19;
    const CE: usize = 3;
    const CK: usize = 2;
    const CC: usize = 3;
    let schedule = AuxLossSchedule {
        base_coefficient: 0.02,
        decay_horizon: 100.0,
    };
    let config = AdamWConfig {
        learning_rate: 0.01,
        weight_decay: 0.1,
        ..AdamWConfig::default()
    };
    const CL: usize = 2;
    let mut cpu = MoeDense::<CN, CT, VOCAB, D, H, HD, CFF, CE, CK, CC, CL>::new(
        123,
        schedule.base_coefficient,
    );
    cpu.visit_cpu_parameters(&mut Bf16MasterInit);
    let mut gpu =
        GpuDense::<CN, NP, CT, VOCAB, VP, D, H, HD, CFF, CE, CK, CC, CL>::from_cpu(stream, &cpu)?;
    let mut optimizer = GpuDenseAdamW::new(stream, config, schedule, CL)?;
    let mut workspace =
        GpuMoeWorkspace::<CN, NP, CT, VOCAB, VP, D, H, CFF, CE, CK, CC, CL>::new(stream)?;
    let tokens = [1, 5, 5, 2];
    let targets = [5, 5, 2, 7];

    let base = std::env::temp_dir().join(format!("oxide-train-{}", std::process::id()));
    let checkpoint_path = base.with_extension("ckpt");
    let continued_path = base.with_extension("continued-a");
    let resumed_path = base.with_extension("continued-b");
    let initial_path = base.with_extension("step0");
    let initial_continued_path = base.with_extension("step0-continued");
    let tampered_path = base.with_extension("tampered");
    let legacy_path = base.with_extension("v4");

    // Step zero, saved and resumed before anything has trained. This is the
    // state the init-checkpoint cache distributes -- a measurement harness
    // resumes it in place of re-deriving the parameters -- and it is the one
    // corner the resume below cannot speak for: there the optimizer step is
    // one and its moments are populated, here both are zero, and `next_batch`
    // is zero rather than mid-shard. The claim to check is not that the file
    // round-trips but that the *trajectory* does: the first training step
    // taken from a resumed step zero has to be the first training step taken
    // from fresh initialization, which is readable as bytes.
    model::checkpoint::save(&initial_path, &gpu, &optimizer, 0, stream)?;
    let initial = model::checkpoint::load::<CN, NP, CT, VOCAB, VP, D, H, HD, CFF, CE, CK, CC, CL>(
        &initial_path,
        stream,
        tensor,
    )?;
    assert_eq!(initial.next_batch, 0);
    assert_eq!(initial.optimizer.step(), 0);
    let mut initial_gpu = initial.model;
    let mut initial_optimizer = initial.optimizer;
    let mut initial_workspace =
        GpuMoeWorkspace::<CN, NP, CT, VOCAB, VP, D, H, CFF, CE, CK, CC, CL>::new(stream)?;

    for (candidate, candidate_optimizer, candidate_workspace) in [
        (&mut gpu, &mut optimizer, &mut workspace),
        (
            &mut initial_gpu,
            &mut initial_optimizer,
            &mut initial_workspace,
        ),
    ] {
        let coefficient = candidate_optimizer.aux_coefficient();
        candidate.zero_grad(stream, tensor)?;
        candidate.forward(
            &tokens,
            &targets,
            coefficient,
            candidate_workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
        )?;
        candidate.backward(
            coefficient,
            candidate_workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
        )?;
        candidate_optimizer.update(candidate, stream, tensor)?;
    }
    model::checkpoint::save(&checkpoint_path, &gpu, &optimizer, 7, stream)?;
    model::checkpoint::save(
        &initial_continued_path,
        &initial_gpu,
        &initial_optimizer,
        7,
        stream,
    )?;
    assert_eq!(
        std::fs::read(&checkpoint_path)?,
        std::fs::read(&initial_continued_path)?,
        "a step taken from a resumed step-0 checkpoint diverged from fresh initialization"
    );
    drop((initial_gpu, initial_optimizer, initial_workspace));
    let loaded = model::checkpoint::load::<CN, NP, CT, VOCAB, VP, D, H, HD, CFF, CE, CK, CC, CL>(
        &checkpoint_path,
        stream,
        tensor,
    )?;
    assert_eq!(loaded.next_batch, 7);
    assert_eq!(loaded.optimizer.aux_schedule(), schedule);
    let mut resumed_gpu = loaded.model;
    let mut resumed_optimizer = loaded.optimizer;
    let mut resumed_workspace =
        GpuMoeWorkspace::<CN, NP, CT, VOCAB, VP, D, H, CFF, CE, CK, CC, CL>::new(stream)?;

    for (candidate, candidate_optimizer, candidate_workspace) in [
        (&mut gpu, &mut optimizer, &mut workspace),
        (
            &mut resumed_gpu,
            &mut resumed_optimizer,
            &mut resumed_workspace,
        ),
    ] {
        let coefficient = candidate_optimizer.aux_coefficient();
        candidate.zero_grad(stream, tensor)?;
        candidate.forward(
            &tokens,
            &targets,
            coefficient,
            candidate_workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
        )?;
        candidate.backward(
            coefficient,
            candidate_workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
        )?;
        candidate_optimizer.update(candidate, stream, tensor)?;
    }
    model::checkpoint::save(&continued_path, &gpu, &optimizer, 8, stream)?;
    model::checkpoint::save(&resumed_path, &resumed_gpu, &resumed_optimizer, 8, stream)?;
    assert_eq!(
        std::fs::read(&continued_path)?,
        std::fs::read(&resumed_path)?,
        "checkpoint resume changed the continued trajectory"
    );

    assert!(
        model::checkpoint::load::<CN, NP, CT, VOCAB, VP, D, H, HD, CFF, 4, CK, CC, CL>(
            &checkpoint_path,
            stream,
            tensor
        )
        .is_err()
    );
    assert!(
        model::checkpoint::load::<CN, NP, CT, VOCAB, VP, D, H, HD, CFF, CE, 1, CC, CL>(
            &checkpoint_path,
            stream,
            tensor
        )
        .is_err()
    );
    assert!(
        model::checkpoint::load::<CN, NP, CT, VOCAB, VP, D, H, HD, CFF, CE, CK, 4, CL>(
            &checkpoint_path,
            stream,
            tensor
        )
        .is_err()
    );
    assert!(
        model::checkpoint::load::<CN, NP, CT, VOCAB, VP, D, H, HD, CFF, CE, CK, CC, 1>(
            &checkpoint_path,
            stream,
            tensor
        )
        .is_err()
    );

    let mut tampered = std::fs::read(&checkpoint_path)?;
    const AUX_BASE_OFFSET: usize = 8 + 4 + 11 * 8 + 2 * 8 + 5 * 4;
    tampered[AUX_BASE_OFFSET..AUX_BASE_OFFSET + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(&tampered_path, tampered)?;
    assert!(
        model::checkpoint::load::<CN, NP, CT, VOCAB, VP, D, H, HD, CFF, CE, CK, CC, CL>(
            &tampered_path,
            stream,
            tensor
        )
        .is_err(),
        "tampered aux-loss schedule was accepted"
    );

    // v4 carried fp32 masters. Its parameter payload is twice the size and
    // cannot be reinterpreted, so the version check has to refuse it outright
    // rather than let a resume read garbage.
    let mut legacy = std::fs::read(&checkpoint_path)?;
    legacy[8..12].copy_from_slice(&4u32.to_le_bytes());
    std::fs::write(&legacy_path, legacy)?;
    assert!(
        model::checkpoint::load::<CN, NP, CT, VOCAB, VP, D, H, HD, CFF, CE, CK, CC, CL>(
            &legacy_path,
            stream,
            tensor
        )
        .is_err(),
        "fp32-master checkpoint v4 was accepted by a v5 build"
    );

    for path in [
        checkpoint_path,
        continued_path,
        resumed_path,
        initial_path,
        initial_continued_path,
        tampered_path,
        legacy_path,
    ] {
        let _ = std::fs::remove_file(path);
    }
    println!(
        "✓ checkpoint v5 (bf16 masters) resumes bit-identically from step 0 and mid-run, rejects v4, and rejects E/K/C/L or schedule mismatches"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let tensor = model::tensor_kernels::load(&ctx)?;
    let gemm = model::gemm_kernels::load(&ctx)?;
    let gemm_bf16 = model::Tcgen05Gemm::load(&ctx)?;
    let flash_bf16 = model::Tcgen05Flash::load(&ctx)?;
    let flash = model::flash_kernels::load(&ctx)?;
    let dense = model::dense_kernels::load(&ctx)?;
    let reference = model::dense_reference_kernels::load(&ctx)?;

    let mut cpu = Dense::<N, T, VOCAB, D, H, HD, FF>::new(42);
    // Both sides keep bf16 masters (#57); the GPU rounds on upload, so the
    // reference starts on the same grid instead of one rounding away.
    cpu.visit_cpu_parameters(&mut Bf16MasterInit);
    let mut gpu = GpuDenseDense::<N, NP, T, VOCAB, VP, D, H, HD, FF>::from_cpu(&stream, &cpu)?;
    let mut workspace = GpuDenseWorkspace::<N, NP, T, VOCAB, VP, D, H, FF>::new(&stream)?;
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
        &flash_bf16,
        &dense,
        &reference,
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
        &flash_bf16,
        &dense,
        &reference,
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
        &flash_bf16,
        &dense,
        &reference,
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
        &flash_bf16,
        &dense,
        &reference,
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
    let mut cpu_optimizer = DenseAdamW::new(config);
    let mut gpu_optimizer = GpuDenseDenseAdamW::new(&stream, config)?;
    cpu_optimizer.update(&mut cpu);
    // The AdamW step below clears every gradient it consumes; the Muon parity
    // that follows runs over the same ones, so keep a copy.
    let gradients = DenseGradients::save(&gpu, &stream)?;
    gpu_optimizer.update(&mut gpu, &stream, &tensor)?;

    // Norms keep fp32 storage, so they keep the tight fp32 tolerance; every
    // bf16 master is compared on the bf16 grid instead.
    macro_rules! master_weight {
        ($field:ident) => {
            assert_master_tensor_close(
                concat!(stringify!($field), ".w after AdamW"),
                &gpu.$field.w,
                &cpu.$field.w,
                &stream,
                MASTER_ULPS,
            )?;
        };
    }
    macro_rules! norm_weight {
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
    master_weight!(embedding);
    norm_weight!(attention_norm);
    assert_grouped_master_close(
        "qkv_proj.w after AdamW",
        &gpu.qkv_proj.w,
        [&cpu.q_proj.w, &cpu.k_proj.w, &cpu.v_proj.w],
        &stream,
        MASTER_ULPS,
    )?;
    master_weight!(o_proj);
    norm_weight!(ffn_norm);
    assert_grouped_master_close(
        "gate_up_proj.w after AdamW",
        &gpu.gate_up_proj.w,
        [&cpu.gate_proj.w, &cpu.up_proj.w],
        &stream,
        MASTER_ULPS,
    )?;
    master_weight!(down_proj);
    norm_weight!(final_norm);
    let master = gpu.lm_head.master.to_f32_host(&stream)?;
    assert_master_close(
        "lm_head master after AdamW",
        &strip_vocab_padding("lm_head.master", &master),
        cpu.lm_head.w.as_slice(),
        MASTER_ULPS,
    );
    check_head_compute_copies(&gpu.lm_head, &stream)?;

    println!("✓ full GPU Dense forward/backward and AdamW (bf16 lm-head) match CPU");

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
    let mut cpu_muon = DenseMuon::new(muon_config, config);
    let mut gpu_muon = GpuDenseMuon::new(&stream, muon_config, config)?;
    cpu_muon.update(&mut cpu);
    gradients.restore(&mut gpu, &stream)?;
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
            assert_close_slices(
                concat!(stringify!($field), ".w after Muon"),
                &gpu.$field.w.to_f32_host(&stream)?,
                cpu.$field.w.as_slice(),
                NS_ATOL,
                MUON_MASTER_RTOL,
            );
        };
    }
    assert_close_slices(
        "qkv_proj.w after Muon",
        &gpu.qkv_proj.w.to_f32_host(&stream)?,
        &interleave_groups([&cpu.q_proj.w, &cpu.k_proj.w, &cpu.v_proj.w]),
        NS_ATOL,
        MUON_MASTER_RTOL,
    );
    muon_weight!(o_proj);
    assert_close_slices(
        "gate_up_proj.w after Muon",
        &gpu.gate_up_proj.w.to_f32_host(&stream)?,
        &interleave_groups([&cpu.gate_proj.w, &cpu.up_proj.w]),
        NS_ATOL,
        MUON_MASTER_RTOL,
    );
    muon_weight!(down_proj);
    macro_rules! muon_norm_weight {
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
    assert_master_tensor_close(
        "embedding.w after Muon's AdamW",
        &gpu.embedding.w,
        &cpu.embedding.w,
        &stream,
        MASTER_ULPS_CHAINED,
    )?;
    muon_norm_weight!(attention_norm);
    muon_norm_weight!(ffn_norm);
    muon_norm_weight!(final_norm);
    let master = gpu.lm_head.master.to_f32_host(&stream)?;
    assert_master_close(
        "lm_head master after Muon's AdamW",
        &strip_vocab_padding("lm_head.master", &master),
        cpu.lm_head.w.as_slice(),
        MASTER_ULPS_CHAINED,
    );
    check_head_compute_copies(&gpu.lm_head, &stream)?;
    gpu.zero_grad(&stream, &tensor)?;
    println!("✓ full GPU Muon update (Newton–Schulz hidden matrices) matches CPU");

    newton_schulz_parity(&stream, &tensor, &gemm)?;
    expert_compute_parity::<3, 5, 128, 19>(
        "fp32 oracle",
        false,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &dense,
        &reference,
    )?;
    expert_compute_parity::<2, 256, 256, 256>(
        "aligned tcgen05",
        true,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &dense,
        &reference,
    )?;
    moe_model_parity::<8, 256, 4, 19, 3, 2, 3, 2>(
        "fp32-oracle",
        false,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &flash_bf16,
        &dense,
    )?;
    moe_model_parity::<256, 256, 256, 256, 3, 2, 256, 2>(
        "aligned tcgen05",
        true,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &flash_bf16,
        &dense,
    )?;
    aligned_moe_overfit(
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &flash_bf16,
        &dense,
    )?;
    moe_checkpoint_gate(
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &flash_bf16,
        &dense,
    )?;
    // The parity helpers own temporary expert workspaces. Their device frees
    // are stream-ordered; complete those frees before the independent overfit
    // gates begin allocating models and workspaces of their own.
    stream.synchronize()?;
    overfit_tiny_batch(
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &flash_bf16,
        &dense,
        &reference,
    )?;
    muon_overfit_tiny_batch(
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &flash_bf16,
        &dense,
        &reference,
    )?;
    aligned_tcgen05_linears(
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &flash_bf16,
        &dense,
        &reference,
    )?;
    aligned_muon_overfit(
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &flash_bf16,
        &dense,
        &reference,
    )?;
    Ok(())
}
