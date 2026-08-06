//! CUDA-event breakdown of one full, realistically-sized GPU training step.
//!
//! This profiles the current integrated path only. Same-container 7.x
//! before/after claims come from `BASELINE_REF=<git-ref> ./run.sh model
//! profile`, which builds and runs a retained baseline binary and this one
//! back-to-back on the same GPU (SPEC §10.1).

use bench_util::{StepProfiler, function_profile};
use cuda_core::CudaContext;
use optim::{AdamWConfig, AuxLossSchedule};

#[path = "../lib.rs"]
mod model;
use model::{
    GEMM_SHARED_BYTES, GEMM_THREADS, GpuDense, GpuDenseAdamW, GpuMoeWorkspace, NORM_THREADS,
};

const B: usize = 12;
const T: usize = 2_048;
const N: usize = B * T;
const NP: usize = 24_576;
const VOCAB: usize = 50_257;
const VP: usize = 50_432;
const D: usize = 3_072;
const H: usize = 24;
const HD: usize = 128;
const FF: usize = 4_096;
const E: usize = 8;
const K: usize = 2;
const C: usize = 6_144;
const L: usize = 12;
const WARMUP_STEPS: usize = 2;

const fn parameter_count() -> usize {
    // Untied embedding/lm-head, per block: four attention projections, router,
    // all expert projections, and two RMSNorm weights; plus the final norm.
    // Padded lm-head vocabulary columns are frozen zeros, not parameters.
    2 * VOCAB * D + L * (4 * D * D + D * E + E * 3 * D * FF + 2 * D) + D
}

fn print_vram(label: &str) {
    let mut free = 0usize;
    let mut total = 0usize;
    let rc = unsafe { cuda_bindings::cuMemGetInfo_v2(&mut free, &mut total) };
    if rc == 0 {
        println!(
            "vram {label}: used={:.1}GiB free={:.1}GiB total={:.1}GiB",
            (total - free) as f64 / (1 << 30) as f64,
            free as f64 / (1 << 30) as f64,
            total as f64 / (1 << 30) as f64,
        );
    }
}

/// What ptxas gave the block-per-row RMSNorm kernels and how many blocks of
/// one fit on an SM. Fusing raises liveness, and a fused kernel that crosses
/// an occupancy step can lose despite the traffic it saves, so the numbers
/// belong beside the timings rather than in a one-off run.
fn report_norm_kernels(
    dense: &model::dense_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "rmsnorm kernels (registers/thread, spill bytes, blocks/SM at {NORM_THREADS} threads)"
    );
    for name in [
        "rms_norm_forward_fast_bf16",
        "rms_norm_forward_residual_bf16",
        "embedding_forward_norm_bf16",
        "rms_norm_backward_fused_bf16",
        "rms_norm_backward_x_fast_bf16",
        "rms_norm_backward_weight_fast_bf16",
    ] {
        let function = dense.as_cuda_module().load_function(name)?;
        let profile = function_profile(&function)?;
        let blocks = function.max_active_blocks_per_multiprocessor(NORM_THREADS as u32, 0)?;
        println!(
            "  {name:<38} {:>3} regs, {:>4} spill bytes, {blocks} blocks/SM",
            profile.registers, profile.spill_bytes
        );
    }
    Ok(())
}

/// The same for the three tcgen05 GEMMs, which `gpu/gemm`'s own gate also
/// prints — and prints *different numbers* for.
///
/// Same source, other crate: what ptxas allocates depends on what else it is
/// compiling, so the counts the GEMM budget pins are the counts in `gpu/gemm`
/// and not necessarily the ones the training step launches. That matters more
/// than it reads: a launch whose block a thread's registers do not divide into
/// the file is refused outright (`701`), not merely made slow, so the number to
/// read here is `blocks/SM` — a **zero** is a launch the driver will not take.
fn report_gemm_kernels(gemm: &model::Tcgen05Gemm) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "tcgen05 GEMMs (registers/thread, spill bytes, CTAs/SM at {GEMM_THREADS} threads \
         and {GEMM_SHARED_BYTES} B)"
    );
    for (name, function) in gemm.kernels() {
        let profile = function_profile(function)?;
        let blocks = function
            .max_active_blocks_per_multiprocessor(GEMM_THREADS, GEMM_SHARED_BYTES as u32)?;
        println!(
            "  {name:<38} {:>3} regs, {:>4} spill bytes, {blocks} CTAs/SM",
            profile.registers, profile.spill_bytes
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(N, B * T);
    assert_eq!(D, H * HD);
    println!(
        "config: params={} ({:.1}M) L={} B={} T={} vocab={} (padded {}) D={} H={} HD={} E={} K={} FF_expert={} C={} active_FF={}",
        parameter_count(),
        parameter_count() as f64 / 1_000_000.0,
        L,
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
    let gemm_bf16 = model::Tcgen05Gemm::load(&ctx)?;
    let flash_bf16 = model::Tcgen05Flash::load(&ctx)?;
    let flash = model::flash_kernels::load(&ctx)?;
    let dense = model::dense_kernels::load(&ctx)?;
    report_norm_kernels(&dense)?;
    report_gemm_kernels(&gemm_bf16)?;

    let aux_schedule = AuxLossSchedule::default();
    eprintln!("profile setup: initializing parameters block by block");
    let mut gpu = GpuDense::<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>::initialized(
        &stream,
        42,
        aux_schedule.coefficient(0),
    )?;
    eprintln!("profile setup: allocating optimizer and workspace");
    let mut optimizer = GpuDenseAdamW::new(&stream, AdamWConfig::default(), aux_schedule, L)?;
    let mut workspace = GpuMoeWorkspace::<N, NP, T, VOCAB, VP, D, H, FF, E, K, C, L>::new(&stream)?;
    print_vram("after model/optimizer/workspace allocation");
    let tokens: Vec<usize> = (0..N).map(|i| (i * 7919 + 17) % VOCAB).collect();
    let targets: Vec<usize> = (0..N).map(|i| tokens[(i + 1) % N]).collect();
    let tokens: &[usize; N] = tokens.as_slice().try_into().expect("length N");
    let targets: &[usize; N] = targets.as_slice().try_into().expect("length N");

    for step in 0..WARMUP_STEPS {
        let aux_coefficient = optimizer.aux_coefficient();
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
    // No gradient fills: the AdamW write-back clears each gradient as it
    // consumes it. The pinned input H2D copies remain deliberately inside the
    // full-step interval and appear as unattributed device time rather than
    // being mislabeled as kernels.
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
    println!("scope: forward + backward + AdamW (which clears its own gradients)");
    Ok(())
}
