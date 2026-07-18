//! CUDA-event breakdown of one full, realistically-sized GPU training step.
//!
//! This profiles the current integrated path only. Same-container 7.x
//! before/after claims come from `BASELINE_REF=<git-ref> ./run.sh llama-model
//! profile`, which builds and runs a retained baseline binary and this one
//! back-to-back on the same GPU (SPEC §10.1).

use bench_util::StepProfiler;
use cuda_core::CudaContext;
use nn::Llama;
use optim::AdamWConfig;

#[path = "../lib.rs"]
mod model;
use model::{GpuLlama, GpuLlamaAdamW, GpuLlamaWorkspace};

const B: usize = 32;
const T: usize = 1_024;
const N: usize = B * T;
const NP: usize = 32_768;
const VOCAB: usize = 50_257;
const VP: usize = 50_304;
const D: usize = 1_536;
const H: usize = 24;
const HD: usize = 64;
const FF: usize = 4_096;
const WARMUP_STEPS: usize = 2;

const fn parameter_count() -> usize {
    // Untied embedding/lm-head, four attention projections, three FFN
    // projections, and the three RMSNorm weights in this reference model.
    // Padded lm-head vocabulary columns are frozen zeros, not parameters.
    2 * VOCAB * D + 4 * D * D + 3 * D * FF + 3 * D
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(N, B * T);
    assert_eq!(D, H * HD);
    println!(
        "config: params={} ({:.1}M) B={} T={} vocab={} (padded {}) D={} H={} HD={} FF={}",
        parameter_count(),
        parameter_count() as f64 / 1_000_000.0,
        B,
        T,
        VOCAB,
        VP,
        D,
        H,
        HD,
        FF,
    );

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let tensor = model::tensor_kernels::load(&ctx)?;
    let gemm = model::gemm_kernels::load(&ctx)?;
    let gemm_bf16 = model::Tcgen05Gemm::load_from_ptx(&ctx, "gemm.ptx")?;
    let flash = model::flash_kernels::load(&ctx)?;
    let llama = model::llama_kernels::load(&ctx)?;

    let cpu = Llama::<N, T, VOCAB, D, H, HD, FF>::new(42);
    let mut gpu = GpuLlama::<N, NP, T, VOCAB, VP, D, H, HD, FF>::from_cpu(&stream, &cpu)?;
    drop(cpu);
    let mut optimizer = GpuLlamaAdamW::new(&stream, AdamWConfig::default())?;
    let mut workspace = GpuLlamaWorkspace::<N, NP, T, VOCAB, VP, D, H, FF>::new(&stream)?;
    let tokens: Vec<usize> = (0..N).map(|i| (i * 7919 + 17) % VOCAB).collect();
    let targets: Vec<usize> = (0..N).map(|i| tokens[(i + 1) % N]).collect();
    let tokens: &[usize; N] = tokens.as_slice().try_into().expect("length N");
    let targets: &[usize; N] = targets.as_slice().try_into().expect("length N");

    for _ in 0..WARMUP_STEPS {
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
        gpu.backward(
            &mut workspace,
            &stream,
            &tensor,
            &gemm,
            &gemm_bf16,
            &flash,
            &llama,
        )?;
        optimizer.update(&mut gpu, &stream, &tensor)?;
    }
    stream.synchronize()?;

    let mut profiler = StepProfiler::start(&stream)?;
    // Gradient fills are named kernel spans. The pinned input H2D copies remain
    // deliberately inside the full-step interval and appear as unattributed
    // device time rather than being mislabeled as kernels.
    gpu.zero_grad_profiled(&stream, &tensor, &mut profiler)?;
    gpu.forward_profiled(
        tokens,
        targets,
        &mut workspace,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &llama,
        &mut profiler,
    )?;
    gpu.backward_profiled(
        &mut workspace,
        &stream,
        &tensor,
        &gemm,
        &gemm_bf16,
        &flash,
        &llama,
        &mut profiler,
    )?;
    optimizer.update_profiled(&mut gpu, &stream, &tensor, &mut profiler)?;
    let profile = profiler.finish(&stream)?;

    println!("bf16 tcgen05 block linears/lm-head + fast norm and atomic embedding backward");
    println!("{profile}");
    println!();
    println!("scope: in-place zero_grad + forward + backward + AdamW");
    Ok(())
}
