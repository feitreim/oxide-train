//! CUDA-event breakdown of one full GPU forward/backward step.

use bench_util::StepProfiler;
use cuda_core::CudaContext;
use nn::Llama;

#[path = "../lib.rs"]
mod model;
use model::GpuLlama;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 8;
    const T: usize = 4;
    const VOCAB: usize = 17;
    const D: usize = 12;
    const H: usize = 3;
    const HD: usize = 4;
    const FF: usize = 19;
    const WARMUP_STEPS: usize = 3;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let tensor = model::tensor_kernels::load(&ctx)?;
    let llama = model::llama_kernels::load(&ctx)?;

    let cpu = Llama::<N, T, VOCAB, D, H, HD, FF>::new(42);
    let mut gpu = GpuLlama::from_cpu(&stream, &cpu)?;
    let tokens = [1, 5, 5, 2, 9, 3, 16, 0];
    let targets = [5, 5, 2, 7, 3, 16, 0, 4];

    for _ in 0..WARMUP_STEPS {
        let (_, step_ctx) = gpu.forward(tokens, targets, &stream, &tensor, &llama)?;
        gpu.backward(step_ctx, &stream, &tensor, &llama)?;
        gpu.zero_grad(&stream)?;
    }
    stream.synchronize()?;

    let mut profiler = StepProfiler::start(&stream)?;
    let (_, step_ctx) =
        gpu.forward_profiled(tokens, targets, &stream, &tensor, &llama, &mut profiler)?;
    gpu.backward_profiled(step_ctx, &stream, &tensor, &llama, &mut profiler)?;
    let profile = profiler.finish(&stream)?;

    println!("{profile}");
    println!();
    println!("scope: forward + backward (optimizer lands with milestone 6)");
    Ok(())
}
