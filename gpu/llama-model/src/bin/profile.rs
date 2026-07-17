//! CUDA-event breakdown of one full, realistically-sized GPU training step.

use bench_util::StepProfiler;
use cuda_core::CudaContext;
use nn::Llama;
use optim::AdamWConfig;

#[path = "../lib.rs"]
mod model;
use model::{GpuLlama, GpuLlamaAdamW, GpuLlamaWorkspace};

const B: usize = 1;
const T: usize = 64;
const N: usize = B * T;
const VOCAB: usize = 50_257;
const D: usize = 1_536;
const H: usize = 24;
const HD: usize = 64;
const FF: usize = 4_096;
const WARMUP_STEPS: usize = 2;

const fn parameter_count() -> usize {
    // Untied embedding/lm-head, four attention projections, three FFN
    // projections, and the three RMSNorm weights in this reference model.
    2 * VOCAB * D + 4 * D * D + 3 * D * FF + 3 * D
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

    let cpu = Llama::<N, T, VOCAB, D, H, HD, FF>::new(42);
    let mut gpu = GpuLlama::from_cpu(&stream, &cpu)?;
    drop(cpu);
    let mut optimizer = GpuLlamaAdamW::new(&stream, AdamWConfig::default())?;
    let mut workspace = GpuLlamaWorkspace::<N, T, VOCAB, D, H, FF>::new(&stream)?;
    let tokens = std::array::from_fn(|i| (i * 7919 + 17) % VOCAB);
    let targets = std::array::from_fn(|i| tokens[(i + 1) % N]);

    for _ in 0..WARMUP_STEPS {
        gpu.zero_grad(&stream, &tensor)?;
        gpu.forward(tokens, targets, &mut workspace, &stream, &tensor, &llama)?;
        gpu.backward(&mut workspace, &stream, &tensor, &llama)?;
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
        &llama,
        &mut profiler,
    )?;
    gpu.backward_profiled(&mut workspace, &stream, &tensor, &llama, &mut profiler)?;
    optimizer.update_profiled(&mut gpu, &stream, &tensor, &mut profiler)?;
    let profile = profiler.finish(&stream)?;

    println!("{profile}");
    println!();
    println!("scope: in-place zero_grad + forward + backward + AdamW");
    Ok(())
}
