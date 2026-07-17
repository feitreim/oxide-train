//! CPU/GPU parity for all GEMM ladder rungs and epilogues.
//!
//! Run on B200 with `./run.sh gemm`.

use bench_util::uniform_vec;
use cuda_core::{CudaContext, DeviceBuffer};
use gemm::{create_bf16_tma_map, fp32_launch_config, kernels, tcgen05_launch_config};
use half::bf16;

fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0.0f64;
            for inner in 0..k {
                sum += a[row * k + inner] as f64 * b[inner * n + column] as f64;
            }
            output[row * n + column] = sum as f32;
        }
    }
    output
}

fn matmul_transposed_b(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0.0f64;
            for inner in 0..k {
                sum += a[row * k + inner] as f64 * b[column * k + inner] as f64;
            }
            output[row * n + column] = sum as f32;
        }
    }
    output
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name} mismatch at {index}: gpu={actual}, cpu={expected}, tolerance={tolerance}"
        );
    }
}

fn quantize_bf16(values: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = values
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect();
    let rounded = bits
        .iter()
        .map(|&bits| bf16::from_bits(bits).to_f32())
        .collect();
    (bits, rounded)
}

fn unpack_bf16(values: &[u32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(values.len() * 2);
    for &pair in values {
        output.push(bf16::from_bits(pair as u16).to_f32());
        output.push(bf16::from_bits((pair >> 16) as u16).to_f32());
    }
    output
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::from_module(context.load_module_from_file("gemm.ptx")?)?;

    check_fp32(&stream, &module)?;
    check_tcgen05_bf16(&stream, &module)?;
    println!("✓ fp32 and tcgen05 bf16 GEMM store/accumulate parity passed");
    Ok(())
}

fn check_fp32(
    stream: &cuda_core::CudaStream,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // Exercise every boundary path, not only aligned training shapes.
    const M: usize = 73;
    const N: usize = 91;
    const K: usize = 47;
    let a = uniform_vec(M * K, 1);
    let b = uniform_vec(K * N, 2);
    let initial = uniform_vec(M * N, 3);
    let expected = matmul(&a, &b, M, N, K);
    let expected_accumulate: Vec<f32> = initial
        .iter()
        .zip(&expected)
        .map(|(initial, product)| initial + product)
        .collect();
    let device_a = DeviceBuffer::from_host(stream, &a)?;
    let device_b = DeviceBuffer::from_host(stream, &b)?;

    let mut store = DeviceBuffer::<f32>::zeroed(stream, M * N)?;
    unsafe {
        module.gemm_fp32_store(
            stream,
            fp32_launch_config(M, N),
            M,
            N,
            K,
            &device_a,
            &device_b,
            &mut store,
        )
    }?;
    assert_close(
        "fp32 store",
        &store.to_host_vec(stream)?,
        &expected,
        2e-5,
        2e-5,
    );

    let mut accumulate = DeviceBuffer::from_host(stream, &initial)?;
    unsafe {
        module.gemm_fp32_accumulate(
            stream,
            fp32_launch_config(M, N),
            M,
            N,
            K,
            &device_a,
            &device_b,
            &mut accumulate,
        )
    }?;
    assert_close(
        "fp32 accumulate",
        &accumulate.to_host_vec(stream)?,
        &expected_accumulate,
        2e-5,
        2e-5,
    );
    Ok(())
}

fn check_tcgen05_bf16(
    stream: &cuda_core::CudaStream,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const M: usize = 128;
    const N: usize = 128;
    const K: usize = 64;
    let (a_bits, a) = quantize_bf16(&uniform_vec(M * K, 4));
    // tcgen05 consumes B in transposed [N,K] storage so K remains contiguous.
    let (b_bits, b) = quantize_bf16(&uniform_vec(N * K, 5));
    let expected = matmul_transposed_b(&a, &b, M, N, K);
    let (_, initial) = quantize_bf16(&uniform_vec(M * N, 6));
    let expected_accumulate: Vec<f32> = initial
        .iter()
        .zip(&expected)
        .map(|(initial, product)| initial + product)
        .collect();

    let device_a = DeviceBuffer::from_host(stream, &a_bits)?;
    let device_b = DeviceBuffer::from_host(stream, &b_bits)?;
    let a_tma = create_bf16_tma_map(stream, &device_a, K, M)?;
    let b_tma = create_bf16_tma_map(stream, &device_b, K, N)?;
    let config = tcgen05_launch_config(M, N, K);

    let mut store = DeviceBuffer::<u32>::zeroed(stream, M * N / 2)?;
    unsafe {
        module.gemm_tcgen05_bf16_store(
            stream,
            config,
            a_tma.as_ptr(),
            b_tma.as_ptr(),
            &mut store,
            N as u32,
            K as u32,
        )
    }?;
    assert_close(
        "tcgen05 bf16 store",
        &unpack_bf16(&store.to_host_vec(stream)?),
        &expected,
        0.03,
        0.01,
    );

    let mut accumulate = DeviceBuffer::from_host(stream, &pack_bf16(&initial))?;
    unsafe {
        module.gemm_tcgen05_bf16_accumulate(
            stream,
            config,
            a_tma.as_ptr(),
            b_tma.as_ptr(),
            &mut accumulate,
            N as u32,
            K as u32,
        )
    }?;
    assert_close(
        "tcgen05 bf16 accumulate",
        &unpack_bf16(&accumulate.to_host_vec(stream)?),
        &expected_accumulate,
        0.04,
        0.015,
    );
    Ok(())
}
