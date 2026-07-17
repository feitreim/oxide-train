//! Naive versus shared-memory tiled fp32 GEMM throughput.
//!
//! Run with `./run.sh tensor-gpu bench`.

use bench_util::time_gpu_iters;
use cuda_core::CudaContext;
use tensor_core::Rank2;
use tensor_cpu::CpuTensor;

// `cargo oxide` embeds the CUDA artifact into the selected binary target, so
// this binary includes the canonical kernel source as a module (the same
// pattern as llama-ops) instead of importing the library crate.
#[path = "../lib.rs"]
#[allow(dead_code)]
mod device;
use device::{GpuTensor, kernels};

const M: usize = 1024;
const K: usize = 1024;
const N: usize = 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // Embedded-artifact loader: libdevice math in this module means the
    // backend emits NVVM IR, not a standalone .ptx file.
    let module = kernels::load(&ctx)?;
    let a = GpuTensor::from_cpu(&stream, &CpuTensor::<f32, Rank2<M, K>>::uniform(11))?;
    let b = GpuTensor::from_cpu(&stream, &CpuTensor::<f32, Rank2<K, N>>::uniform(12))?;

    let naive_ms = time_gpu_iters(&stream, 2, 10, || {
        let _ = a.matmul_naive(&b, &stream, &module)?;
        Ok(())
    })?;
    let tiled_ms = time_gpu_iters(&stream, 5, 20, || {
        let _ = a.matmul(&b, &stream, &module)?;
        Ok(())
    })?;

    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let tflops = |ms: f64| flops / (ms / 1_000.0) / 1e12;
    println!("fp32 GEMM [{M},{K}] x [{K},{N}]");
    println!(
        "  naive: {naive_ms:8.3} ms  {:7.3} TFLOP/s",
        tflops(naive_ms)
    );
    println!(
        "  tiled: {tiled_ms:8.3} ms  {:7.3} TFLOP/s",
        tflops(tiled_ms)
    );
    println!("  speedup: {:.2}x", naive_ms / tiled_ms);
    Ok(())
}
