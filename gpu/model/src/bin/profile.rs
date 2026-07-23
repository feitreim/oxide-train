//! CUDA-event breakdown of one full, realistically-sized GPU training step.
//!
//! This profiles the current integrated path only. Same-container 7.x
//! before/after claims come from `BASELINE_REF=<git-ref> ./run.sh model
//! profile`, which builds and runs a retained baseline binary and this one
//! back-to-back on the same GPU (SPEC §10.1).

use bench_util::StepProfiler;
use cuda_core::CudaContext;
use nn::MoeDense;
use optim::{AdamWConfig, AuxLossSchedule};

#[path = "../lib.rs"]
mod model;
use model::{GpuDense, GpuDenseAdamW, GpuDenseWorkspace};

const B: usize = 32;
const T: usize = 1_024;
const N: usize = B * T;
const NP: usize = 32_768;
const VOCAB: usize = 50_257;
const VP: usize = 50_432;
const D: usize = 1_536;
const H: usize = 24;
const HD: usize = 64;
const FF: usize = 2_048;
const E: usize = 8;
const K: usize = 2;
const C: usize = 8_192;
const WARMUP_STEPS: usize = 2;

const fn parameter_count() -> usize {
    // Untied embedding/lm-head, four attention projections, router, all expert
    // projections, and the three RMSNorm weights.
    // Padded lm-head vocabulary columns are frozen zeros, not parameters.
    2 * VOCAB * D + 4 * D * D + D * E + E * 3 * D * FF + 3 * D
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(N, B * T);
    assert_eq!(D, H * HD);
    println!(
        "config: params={} ({:.1}M) B={} T={} vocab={} (padded {}) D={} H={} HD={} E={} K={} FF_expert={} C={} active_FF={}",
        parameter_count(),
        parameter_count() as f64 / 1_000_000.0,
        B,
        T,
        VOCAB,
        VP,
        D,
        H,
        HD,
        E,
        K,
        FF,
        C,
        K * FF,
    );

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let tensor = model::tensor_kernels::load(&ctx)?;
    let gemm = model::gemm_kernels::load(&ctx)?;
    let gemm_bf16 = model::Tcgen05Gemm::load_from_ptx(&ctx, "gemm.ptx")?;
    let flash_bf16 = model::Tcgen05Flash::load_from_ptx(&ctx, "flash.ptx")?;
    let flash = model::flash_kernels::load(&ctx)?;
    let dense = model::dense_kernels::load(&ctx)?;

    let aux_schedule = AuxLossSchedule::default();
    eprintln!("profile setup: initializing CPU parameters");
    let cpu = MoeDense::<N, T, VOCAB, D, H, HD, FF, E, K, C>::new(42, aux_schedule.coefficient(0));
    eprintln!("profile setup: uploading parameters");
    let mut gpu = GpuDense::<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C>::from_cpu(&stream, &cpu)?;
    drop(cpu);
    eprintln!("profile setup: allocating optimizer and workspace");
    let mut optimizer = GpuDenseAdamW::new(&stream, AdamWConfig::default(), aux_schedule)?;
    let mut workspace = GpuDenseWorkspace::<N, NP, T, VOCAB, VP, D, H, FF, E, K, C>::new(&stream)?;
    let tokens: Vec<usize> = (0..N).map(|i| (i * 7919 + 17) % VOCAB).collect();
    let targets: Vec<usize> = (0..N).map(|i| tokens[(i + 1) % N]).collect();
    let tokens: &[usize; N] = tokens.as_slice().try_into().expect("length N");
    let targets: &[usize; N] = targets.as_slice().try_into().expect("length N");

    for step in 0..WARMUP_STEPS {
        eprintln!("warmup {}/{}: zero_grad", step + 1, WARMUP_STEPS);
        let aux_coefficient = optimizer.aux_coefficient();
        gpu.zero_grad(&stream, &tensor)?;
        stream.synchronize()?;
        eprintln!("warmup {}/{}: forward", step + 1, WARMUP_STEPS);
        gpu.forward(
            tokens,
            targets,
            aux_coefficient,
            &mut workspace,
            &stream,
            &tensor,
            &gemm,
            &gemm_bf16,
            &flash,
            &flash_bf16,
            &dense,
        )?;
        stream.synchronize()?;
        eprintln!("warmup {}/{}: backward", step + 1, WARMUP_STEPS);
        gpu.backward(
            aux_coefficient,
            &mut workspace,
            &stream,
            &tensor,
            &gemm,
            &gemm_bf16,
            &flash,
            &flash_bf16,
            &dense,
        )?;
        stream.synchronize()?;
        eprintln!("warmup {}/{}: optimizer", step + 1, WARMUP_STEPS);
        optimizer.update(&mut gpu, &stream, &tensor)?;
        stream.synchronize()?;
    }
    eprintln!("profile: measuring step");

    let mut profiler = StepProfiler::start(&stream)?;
    // Gradient fills are named kernel spans. The pinned input H2D copies remain
    // deliberately inside the full-step interval and appear as unattributed
    // device time rather than being mislabeled as kernels.
    gpu.zero_grad_profiled(&stream, &tensor, &mut profiler)?;
    let aux_coefficient = optimizer.aux_coefficient();
    gpu.forward_profiled(
        tokens,
        targets,
        aux_coefficient,
        &mut workspace,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &flash_bf16,
        &dense,
        &mut profiler,
    )?;
    gpu.backward_profiled(
        aux_coefficient,
        &mut workspace,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &flash_bf16,
        &dense,
        &mut profiler,
    )?;
    optimizer.update_profiled(&mut gpu, &stream, &tensor, &mut profiler)?;
    let profile = profiler.finish(&stream)?;

    println!("fp32 top-k routing + bf16 tcgen05 experts/block linears/lm-head");
    println!("{profile}");
    println!();
    println!("scope: in-place zero_grad + forward + backward + AdamW");
    Ok(())
}
