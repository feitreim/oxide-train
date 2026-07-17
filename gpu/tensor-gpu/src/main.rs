//! CPU/GPU parity tests for typed storage and all foundation kernels.
//!
//! Run on a GPU with `./run.sh tensor-gpu`.

use cuda_core::CudaContext;
use tensor_core::{Rank1, Rank2, Rank3};
use tensor_cpu::CpuTensor;
use tensor_gpu::{GpuTensor, kernels};

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * e.abs();
        assert!(
            (a - e).abs() <= tolerance,
            "{name} mismatch at {i}: gpu={a}, cpu={e}, tolerance={tolerance}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::from_module(ctx.load_module_from_file("tensor_gpu.ptx")?)?;

    check_storage(&stream)?;
    check_elementwise_and_reductions(&stream, &module)?;
    check_gemm(&stream, &module)?;

    println!("✓ tensor-gpu storage, elementwise, reduction, and GEMM parity passed");
    Ok(())
}

fn check_storage(stream: &cuda_core::CudaStream) -> Result<(), Box<dyn std::error::Error>> {
    let cpu = CpuTensor::<u16, Rank3<2, 3, 5>>::from_fn(|i| (i * 7) as u16);
    let gpu = GpuTensor::from_cpu(stream, &cpu)?;
    assert_eq!(gpu.as_device_buffer().len(), 30);
    assert_eq!(gpu.to_cpu(stream)?, cpu);
    Ok(())
}

fn check_elementwise_and_reductions(
    stream: &cuda_core::CudaStream,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // Deliberately larger than one block and not divisible by 256.
    type Shape = Rank1<1009>;
    let a = CpuTensor::<f32, Shape>::uniform(1);
    let b = CpuTensor::<f32, Shape>::uniform(2);
    let ga = GpuTensor::from_cpu(stream, &a)?;
    let gb = GpuTensor::from_cpu(stream, &b)?;

    assert_close(
        "add",
        &ga.add(&gb, stream, module)?.to_host(stream)?,
        a.add(&b).as_slice(),
        0.0,
        0.0,
    );
    assert_close(
        "mul",
        &ga.mul(&gb, stream, module)?.to_host(stream)?,
        a.mul(&b).as_slice(),
        0.0,
        0.0,
    );
    assert_close(
        "scale",
        &ga.scale(-0.75, stream, module)?.to_host(stream)?,
        a.scale(-0.75).as_slice(),
        0.0,
        0.0,
    );

    let mut gpu_acc = GpuTensor::from_cpu(stream, &a)?;
    gpu_acc.add_scaled_assign(0.25, &gb, stream, module)?;
    let mut cpu_acc = a.clone();
    cpu_acc.add_scaled_assign(0.25, &b);
    assert_close(
        "add_scaled_assign",
        &gpu_acc.to_host(stream)?,
        cpu_acc.as_slice(),
        0.0,
        0.0,
    );

    assert_close(
        "sum",
        &ga.sum(stream, module)?.to_host(stream)?,
        &[a.sum()],
        2e-5,
        2e-5,
    );
    assert_close(
        "dot",
        &ga.dot(&gb, stream, module)?.to_host(stream)?,
        &[a.dot(&b)],
        3e-5,
        3e-5,
    );
    Ok(())
}

fn check_gemm(
    stream: &cuda_core::CudaStream,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // Rectangular and non-tile-aligned in every dimension, exercising all
    // boundary paths instead of only the easy square case.
    const M: usize = 19;
    const K: usize = 23;
    const N: usize = 29;
    let a = CpuTensor::<f32, Rank2<M, K>>::uniform(3);
    let b = CpuTensor::<f32, Rank2<K, N>>::uniform(4);
    let expected = a.matmul(&b);
    let ga = GpuTensor::from_cpu(stream, &a)?;
    let gb = GpuTensor::from_cpu(stream, &b)?;

    assert_close(
        "gemm naive",
        &ga.matmul_naive(&gb, stream, module)?.to_host(stream)?,
        expected.as_slice(),
        2e-5,
        2e-5,
    );
    assert_close(
        "gemm tiled",
        &ga.matmul(&gb, stream, module)?.to_host(stream)?,
        expected.as_slice(),
        2e-5,
        2e-5,
    );
    Ok(())
}
