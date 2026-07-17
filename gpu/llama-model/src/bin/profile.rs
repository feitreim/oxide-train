//! CUDA-event breakdown of one full, realistically-sized GPU training step.

use bench_util::{StepProfile, StepProfiler};
use cuda_core::CudaContext;
use nn::Llama;
use optim::AdamWConfig;

#[path = "../lib.rs"]
mod model;
use model::{GpuLlama, GpuLlamaAdamW, QkvMode};

const B: usize = 1;
const T: usize = 64;
const N: usize = B * T;
const VOCAB: usize = 50_257;
const D: usize = 1_536;
const H: usize = 24;
const HD: usize = 64;
const FF: usize = 4_096;
const WARMUP_STEPS: usize = 2;
const MEASURE_STEPS: usize = 10;

type ProfileModel = GpuLlama<N, T, VOCAB, D, H, HD, FF>;
type ProfileOptimizer = GpuLlamaAdamW<VOCAB, D, FF>;

const fn parameter_count() -> usize {
    // Untied embedding/lm-head, four attention projections, three FFN
    // projections, and the three RMSNorm weights in this reference model.
    2 * VOCAB * D + 4 * D * D + 3 * D * FF + 3 * D
}

#[allow(clippy::too_many_arguments)]
fn run_step(
    gpu: &mut ProfileModel,
    optimizer: &mut ProfileOptimizer,
    tokens: [usize; N],
    targets: [usize; N],
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    llama: &model::llama_kernels::LoadedModule,
    fusion: &model::fusion_kernels::LoadedModule,
) -> Result<(), cuda_core::DriverError> {
    gpu.zero_grad(stream)?;
    let (_, step_ctx) = gpu.forward(tokens, targets, stream, tensor, llama, fusion)?;
    gpu.backward(step_ctx, stream, tensor, llama, fusion)?;
    optimizer.update(gpu, stream, tensor)
}

#[allow(clippy::too_many_arguments)]
fn profile_step(
    gpu: &mut ProfileModel,
    optimizer: &mut ProfileOptimizer,
    tokens: [usize; N],
    targets: [usize; N],
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    llama: &model::llama_kernels::LoadedModule,
    fusion: &model::fusion_kernels::LoadedModule,
) -> Result<StepProfile, cuda_core::DriverError> {
    let mut profiler = StepProfiler::start(stream)?;
    // Gradient-buffer zero fills and input H2D copies are deliberately inside
    // the full-step events. They appear as unattributed device time rather than
    // being mislabeled as kernels.
    gpu.zero_grad(stream)?;
    let (_, step_ctx) = gpu.forward_profiled(
        tokens,
        targets,
        stream,
        tensor,
        llama,
        fusion,
        &mut profiler,
    )?;
    gpu.backward_profiled(step_ctx, stream, tensor, llama, fusion, &mut profiler)?;
    optimizer.update_profiled(gpu, stream, tensor, &mut profiler)?;
    profiler.finish(stream)
}

#[allow(clippy::too_many_arguments)]
fn time_full_steps(
    gpu: &mut ProfileModel,
    optimizer: &mut ProfileOptimizer,
    tokens: [usize; N],
    targets: [usize; N],
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    llama: &model::llama_kernels::LoadedModule,
    fusion: &model::fusion_kernels::LoadedModule,
) -> Result<f64, cuda_core::DriverError> {
    stream.synchronize()?;
    let flags = cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
    let start = stream.record_event(Some(flags))?;
    for _ in 0..MEASURE_STEPS {
        run_step(
            gpu, optimizer, tokens, targets, stream, tensor, llama, fusion,
        )?;
    }
    let end = stream.record_event(Some(flags))?;
    Ok(start.elapsed_ms(&end)? as f64 / MEASURE_STEPS as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(N, B * T);
    assert_eq!(D, H * HD);
    println!(
        "config: params={} ({:.1}M) B={} T={} vocab={} D={} H={} HD={} FF={}",
        parameter_count(),
        parameter_count() as f64 / 1_000_000.0,
        B,
        T,
        VOCAB,
        D,
        H,
        HD,
        FF,
    );

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let tensor = model::tensor_kernels::load(&ctx)?;
    let llama = model::llama_kernels::load(&ctx)?;
    let fusion = model::fusion_kernels::load(&ctx)?;

    let cpu = Llama::<N, T, VOCAB, D, H, HD, FF>::new(42);
    let mut baseline = GpuLlama::from_cpu_with_qkv_mode(&stream, &cpu, QkvMode::Unfused)?;
    drop(cpu);
    let cpu = Llama::<N, T, VOCAB, D, H, HD, FF>::new(42);
    let mut candidate = GpuLlama::from_cpu_with_qkv_mode(&stream, &cpu, QkvMode::Fused)?;
    drop(cpu);
    let mut baseline_optimizer =
        GpuLlamaAdamW::new_with_qkv_mode(&stream, AdamWConfig::default(), QkvMode::Unfused)?;
    let mut candidate_optimizer =
        GpuLlamaAdamW::new_with_qkv_mode(&stream, AdamWConfig::default(), QkvMode::Fused)?;
    let tokens = std::array::from_fn(|i| (i * 7919 + 17) % VOCAB);
    let targets = std::array::from_fn(|i| tokens[(i + 1) % N]);

    for _ in 0..WARMUP_STEPS {
        run_step(
            &mut baseline,
            &mut baseline_optimizer,
            tokens,
            targets,
            &stream,
            &tensor,
            &llama,
            &fusion,
        )?;
    }
    for _ in 0..WARMUP_STEPS {
        run_step(
            &mut candidate,
            &mut candidate_optimizer,
            tokens,
            targets,
            &stream,
            &tensor,
            &llama,
            &fusion,
        )?;
    }
    stream.synchronize()?;

    let baseline_step_ms = time_full_steps(
        &mut baseline,
        &mut baseline_optimizer,
        tokens,
        targets,
        &stream,
        &tensor,
        &llama,
        &fusion,
    )?;
    let candidate_step_ms = time_full_steps(
        &mut candidate,
        &mut candidate_optimizer,
        tokens,
        targets,
        &stream,
        &tensor,
        &llama,
        &fusion,
    )?;

    let baseline_profile = profile_step(
        &mut baseline,
        &mut baseline_optimizer,
        tokens,
        targets,
        &stream,
        &tensor,
        &llama,
        &fusion,
    )?;
    let candidate_profile = profile_step(
        &mut candidate,
        &mut candidate_optimizer,
        tokens,
        targets,
        &stream,
        &tensor,
        &llama,
        &fusion,
    )?;

    println!("=== baseline: separate Q/K/V projections ===");
    println!("{baseline_profile}");
    println!();
    println!("=== candidate: packed horizontal QKV projection ===");
    println!("{candidate_profile}");
    println!();
    let saved = baseline_step_ms - candidate_step_ms;
    let percent = 100.0 * saved / baseline_step_ms;
    println!(
        "unprofiled full-step mean ({MEASURE_STEPS} steps/path): baseline={baseline_step_ms:.4} ms candidate={candidate_step_ms:.4} ms"
    );
    println!(
        "QKV fusion full-step delta: {:+.4} ms ({:+.2}%), speedup {:.3}x",
        -saved,
        -percent,
        baseline_step_ms / candidate_step_ms,
    );
    println!();
    println!("scope: zero_grad + forward + backward + AdamW");
    Ok(())
}
