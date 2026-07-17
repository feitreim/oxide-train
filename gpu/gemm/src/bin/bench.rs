//! CUDA-event throughput benchmark for the GEMM ladder.
//!
//! Run one configuration with `./run.sh gemm bench`, or tune the fp32 rung in
//! one B200 container with, for example:
//!
//! `SWEEP="BM=64 BN=64 BK=16 TM=4 TN=4,BM=128 BN=64 BK=16 TM=8 TN=4" ./run.sh gemm`

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer};
use gemm::{
    BK, BM, BN, TM, TN, create_bf16_tma_map, fp32, fp32_launch_config, kernels,
    tcgen05_launch_config,
};
use half::bf16;

const FP32_M: usize = 2048;
const FP32_N: usize = 2048;
const FP32_K: usize = 2048;
const BF16_M: usize = 4096;
const BF16_N: usize = 4096;
const BF16_K: usize = 4096;

fn tflops(m: usize, n: usize, k: usize, milliseconds: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / (milliseconds / 1_000.0) / 1.0e12
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let fp32_module = fp32::kernels::from_module(context.load_module_from_file("gemm.ptx")?)?;
    let module = kernels::from_module(context.load_module_from_file("gemm.ptx")?)?;

    let fp32_a = DeviceBuffer::from_host(&stream, &uniform_vec(FP32_M * FP32_K, 11))?;
    let fp32_b = DeviceBuffer::from_host(&stream, &uniform_vec(FP32_K * FP32_N, 12))?;
    let mut fp32_c = DeviceBuffer::<f32>::zeroed(&stream, FP32_M * FP32_N)?;
    let fp32_config = fp32_launch_config(FP32_M, FP32_N);

    let fp32_store_ms = time_gpu_iters(&stream, 3, 10, || {
        unsafe {
            fp32_module.register_gemm_store(
                &stream,
                fp32_config,
                FP32_M,
                FP32_N,
                FP32_K,
                &fp32_a,
                &fp32_b,
                &mut fp32_c,
            )
        }
        .map_err(Into::into)
    })?;
    let fp32_accumulate_ms = time_gpu_iters(&stream, 3, 10, || {
        unsafe {
            fp32_module.register_gemm_accumulate(
                &stream,
                fp32_config,
                FP32_M,
                FP32_N,
                FP32_K,
                &fp32_a,
                &fp32_b,
                &mut fp32_c,
            )
        }
        .map_err(Into::into)
    })?;

    println!("fp32 register tile BM={BM} BN={BN} BK={BK} TM={TM} TN={TN}");
    println!(
        "  store      [{FP32_M},{FP32_K}]x[{FP32_K},{FP32_N}]: \
         {fp32_store_ms:8.3} ms  {:8.2} TFLOP/s",
        tflops(FP32_M, FP32_N, FP32_K, fp32_store_ms)
    );
    println!(
        "  accumulate [{FP32_M},{FP32_K}]x[{FP32_K},{FP32_N}]: \
         {fp32_accumulate_ms:8.3} ms  {:8.2} TFLOP/s",
        tflops(FP32_M, FP32_N, FP32_K, fp32_accumulate_ms)
    );

    let bf16_a: Vec<u16> = uniform_vec(BF16_M * BF16_K, 13)
        .into_iter()
        .map(|value| bf16::from_f32(value).to_bits())
        .collect();
    let bf16_b: Vec<u16> = uniform_vec(BF16_N * BF16_K, 14)
        .into_iter()
        .map(|value| bf16::from_f32(value).to_bits())
        .collect();
    let bf16_a = DeviceBuffer::from_host(&stream, &bf16_a)?;
    let bf16_b = DeviceBuffer::from_host(&stream, &bf16_b)?;
    let a_tma = create_bf16_tma_map(&stream, &bf16_a, BF16_K, BF16_M)?;
    let b_tma = create_bf16_tma_map(&stream, &bf16_b, BF16_K, BF16_N)?;
    let mut bf16_c = DeviceBuffer::<u32>::zeroed(&stream, BF16_M * BF16_N / 2)?;
    let bf16_config = tcgen05_launch_config(BF16_M, BF16_N, BF16_K);

    let bf16_store_ms = time_gpu_iters(&stream, 5, 20, || {
        unsafe {
            module.gemm_tcgen05_bf16_store(
                &stream,
                bf16_config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut bf16_c,
                BF16_N as u32,
                BF16_K as u32,
            )
        }
        .map_err(Into::into)
    })?;
    let bf16_accumulate_ms = time_gpu_iters(&stream, 5, 20, || {
        unsafe {
            module.gemm_tcgen05_bf16_accumulate(
                &stream,
                bf16_config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut bf16_c,
                BF16_N as u32,
                BF16_K as u32,
            )
        }
        .map_err(Into::into)
    })?;
    let mut bf16_f32_c = DeviceBuffer::<f32>::zeroed(&stream, BF16_M * BF16_N)?;
    let bf16_f32_store_ms = time_gpu_iters(&stream, 5, 20, || {
        unsafe {
            module.gemm_tcgen05_bf16_f32_store(
                &stream,
                bf16_config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut bf16_f32_c,
                BF16_N as u32,
                BF16_K as u32,
            )
        }
        .map_err(Into::into)
    })?;
    let bf16_f32_accumulate_ms = time_gpu_iters(&stream, 5, 20, || {
        unsafe {
            module.gemm_tcgen05_bf16_f32_accumulate(
                &stream,
                bf16_config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut bf16_f32_c,
                BF16_N as u32,
                BF16_K as u32,
            )
        }
        .map_err(Into::into)
    })?;

    println!("bf16 tcgen05 128x128x64, fp32 TMEM accumulate");
    println!(
        "  store      [{BF16_M},{BF16_K}]x[{BF16_N},{BF16_K}]^T: \
         {bf16_store_ms:8.3} ms  {:8.2} TFLOP/s",
        tflops(BF16_M, BF16_N, BF16_K, bf16_store_ms)
    );
    println!(
        "  accumulate [{BF16_M},{BF16_K}]x[{BF16_N},{BF16_K}]^T: \
         {bf16_accumulate_ms:8.3} ms  {:8.2} TFLOP/s",
        tflops(BF16_M, BF16_N, BF16_K, bf16_accumulate_ms)
    );
    println!(
        "  f32 store  [{BF16_M},{BF16_K}]x[{BF16_N},{BF16_K}]^T: \
         {bf16_f32_store_ms:8.3} ms  {:8.2} TFLOP/s",
        tflops(BF16_M, BF16_N, BF16_K, bf16_f32_store_ms)
    );
    println!(
        "  f32 accum  [{BF16_M},{BF16_K}]x[{BF16_N},{BF16_K}]^T: \
         {bf16_f32_accumulate_ms:8.3} ms  {:8.2} TFLOP/s",
        tflops(BF16_M, BF16_N, BF16_K, bf16_f32_accumulate_ms)
    );
    Ok(())
}
