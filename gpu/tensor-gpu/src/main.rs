//! CPU/GPU parity tests for typed storage and all foundation kernels.
//!
//! Run on a GPU with `./run.sh tensor-gpu`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use tensor_core::{Rank1, Rank2, Rank3, bf16};
use tensor_cpu::CpuTensor;

// `cargo oxide` embeds the CUDA artifact into the selected binary target, so
// this binary includes the canonical kernel source as a module (the same
// pattern as ops) instead of importing the library crate.
#[path = "lib.rs"]
mod device;
use device::{GpuTensor, kernels, master_transpose_config, transpose_pairs_config};

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
    // The embedded-artifact loader, not `load_module_from_file`: kernels that
    // touch libdevice math (`sqrt` in the AdamW updates) make the backend
    // emit NVVM IR instead of a standalone .ptx file.
    let module = kernels::load(&ctx)?;

    check_storage(&stream)?;
    check_elementwise_and_reductions(&stream, &module)?;
    check_gemm(&stream, &module)?;
    check_bf16_pairs(&stream, &module)?;
    check_adamw_master(&stream, &module)?;
    check_adamw_master_transposed(&stream, &module)?;

    println!("✓ tensor-gpu storage, elementwise, reduction, GEMM, and bf16 parity passed");
    Ok(())
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

fn unpack_bf16(words: &[u32]) -> Vec<f32> {
    let mut values = Vec::with_capacity(words.len() * 2);
    for &word in words {
        values.push(bf16::from_bits(word as u16).to_f32());
        values.push(bf16::from_bits((word >> 16) as u16).to_f32());
    }
    values
}

fn check_bf16_pairs(
    stream: &cuda_core::CudaStream,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const ROWS: usize = 128;
    const COLS: usize = 192;
    let source = CpuTensor::<f32, Rank2<ROWS, COLS>>::uniform(7);

    // fill_u32 overwrites stale packed contents in place.
    let mut filled = DeviceBuffer::<u32>::from_host(stream, &vec![u32::MAX; 96])?;
    // SAFETY: the launch covers exactly the 96 allocated words.
    unsafe { module.fill_u32(stream, LaunchConfig::for_num_elems(96), 0, &mut filled) }?;
    assert!(filled.to_host_vec(stream)?.iter().all(|&word| word == 0));

    // f32 -> packed pairs matches host round-to-nearest-even bit-for-bit and
    // leaves padding words beyond the input untouched.
    let device_source = DeviceBuffer::from_host(stream, source.as_slice())?;
    let mut packed = DeviceBuffer::<u32>::zeroed(stream, ROWS * COLS / 2 + 64)?;
    // SAFETY: source has two f32 values per launched pair and packed is larger
    // than the launched output range.
    unsafe {
        module.convert_f32_to_bf16_pairs(
            stream,
            LaunchConfig::for_num_elems((ROWS * COLS / 2) as u32),
            &device_source,
            &mut packed,
        )
    }?;
    let packed_host = packed.to_host_vec(stream)?;
    assert_eq!(
        &packed_host[..ROWS * COLS / 2],
        pack_bf16(source.as_slice())
    );
    assert!(packed_host[ROWS * COLS / 2..].iter().all(|&word| word == 0));

    // packed pairs -> f32 round-trips the rounded values exactly.
    let mut widened = DeviceBuffer::<f32>::zeroed(stream, ROWS * COLS)?;
    // SAFETY: packed contains one word per two output values, and widened has
    // exactly ROWS * COLS elements.
    unsafe {
        module.convert_bf16_pairs_to_f32(
            stream,
            LaunchConfig::for_num_elems((ROWS * COLS) as u32),
            &packed,
            &mut widened,
        )
    }?;
    assert_close(
        "convert_bf16_pairs_to_f32",
        &widened.to_host_vec(stream)?,
        &unpack_bf16(&packed_host[..ROWS * COLS / 2]),
        0.0,
        0.0,
    );

    // Element-level transpose, checked bit-exactly against a host transpose.
    let matrix = DeviceBuffer::from_host(stream, &packed_host[..ROWS * COLS / 2])?;
    let mut transposed = DeviceBuffer::<u32>::zeroed(stream, ROWS * COLS / 2)?;
    unsafe {
        module.transpose_bf16_pairs(
            stream,
            transpose_pairs_config(ROWS, COLS),
            &matrix,
            ROWS as u32,
            COLS as u32,
            &mut transposed,
        )?;
    }
    let elements = unpack_bf16(&packed_host[..ROWS * COLS / 2]);
    let mut expected = vec![0.0f32; ROWS * COLS];
    for row in 0..ROWS {
        for col in 0..COLS {
            expected[col * ROWS + row] = elements[row * COLS + col];
        }
    }
    assert_eq!(transposed.to_host_vec(stream)?, pack_bf16(&expected));

    // Fused quantize-and-transpose reads the fp32 source directly and must
    // match the separate convert-then-transpose bit-for-bit.
    let mut fused = DeviceBuffer::<u32>::zeroed(stream, ROWS * COLS / 2)?;
    unsafe {
        module.convert_f32_transpose_bf16_pairs(
            stream,
            transpose_pairs_config(ROWS, COLS),
            &device_source,
            ROWS as u32,
            COLS as u32,
            &mut fused,
        )?;
    }
    assert_eq!(fused.to_host_vec(stream)?, pack_bf16(&expected));

    // Fused convert-and-transpose emits both the row-major packed matrix and
    // its transpose from one fp32 read; both must be bit-exact.
    let mut tee_normal = DeviceBuffer::<u32>::zeroed(stream, ROWS * COLS / 2)?;
    let mut tee_transposed = DeviceBuffer::<u32>::zeroed(stream, ROWS * COLS / 2)?;
    unsafe {
        module.convert_f32_to_bf16_pairs_and_transpose(
            stream,
            transpose_pairs_config(ROWS, COLS),
            &device_source,
            ROWS as u32,
            COLS as u32,
            &mut tee_normal,
            &mut tee_transposed,
        )?;
    }
    assert_eq!(
        tee_normal.to_host_vec(stream)?,
        pack_bf16(source.as_slice())
    );
    assert_eq!(tee_transposed.to_host_vec(stream)?, pack_bf16(&expected));
    Ok(())
}

/// The fused bf16-master AdamW kernels: fp32 math and moments, one rounded
/// bf16 write-back.
///
/// Round-to-nearest is checked bit-exactly against the host reference, which
/// pins both the arithmetic and the rounding. Stochastic rounding is checked
/// for the two properties it must have: reproducibility (its stream is keyed
/// only on the step, parameter, and element) and staying within one bf16 ulp
/// of nearest.
fn check_adamw_master(
    stream: &cuda_core::CudaStream,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const LEN: usize = 1030;
    const SEED: u64 = tensor_core::rng::stream_seed(17, 3);
    let initial = CpuTensor::<f32, Rank1<LEN>>::uniform(8);
    let master_words = pack_bf16(initial.as_slice());
    let weights = unpack_bf16(&master_words);
    let gradient = CpuTensor::<f32, Rank1<LEN>>::uniform(9);
    let (learning_rate, beta1, beta2, epsilon, weight_decay) = (0.01, 0.9, 0.999, 1e-8, 0.1);
    let (first_correction, second_correction) = (1.0 / (1.0 - beta1), 1.0 / (1.0 - beta2));

    let update = |weight: f32, g: f32| {
        let first_hat = (1.0 - beta1) * g * first_correction;
        let second_hat = (1.0 - beta2) * g * g * second_correction;
        let step = first_hat / (second_hat.sqrt() + epsilon) + weight_decay * weight;
        weight - learning_rate * step
    };

    let mut run =
        |packed_gradient: bool, rounding: u32| -> Result<Vec<u32>, cuda_core::DriverError> {
            let mut master = DeviceBuffer::from_host(stream, &master_words)?;
            let mut first = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
            let mut second = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
            let config = LaunchConfig::for_num_elems((LEN / 2) as u32);
            // SAFETY: master and the packed gradient hold LEN / 2 words, both
            // moments hold LEN floats, and the launch covers exactly LEN / 2 pairs.
            unsafe {
                if packed_gradient {
                    let mut device_gradient =
                        DeviceBuffer::from_host(stream, &pack_bf16(gradient.as_slice()))?;
                    module.adamw_bf16_master_packed_grad(
                        stream,
                        config,
                        &mut device_gradient,
                        learning_rate,
                        beta1,
                        beta2,
                        epsilon,
                        weight_decay,
                        first_correction,
                        second_correction,
                        rounding,
                        SEED,
                        &mut master,
                        &mut first,
                        &mut second,
                    )?;
                    assert!(
                        device_gradient.to_host_vec(stream)?.iter().all(|&w| w == 0),
                        "adamw_bf16_master_packed_grad left the gradient it consumed non-zero"
                    );
                } else {
                    let mut device_gradient = DeviceBuffer::from_host(stream, gradient.as_slice())?;
                    module.adamw_bf16_master(
                        stream,
                        config,
                        &mut device_gradient,
                        learning_rate,
                        beta1,
                        beta2,
                        epsilon,
                        weight_decay,
                        first_correction,
                        second_correction,
                        rounding,
                        SEED,
                        &mut master,
                        &mut first,
                        &mut second,
                    )?;
                    assert!(
                        device_gradient
                            .to_host_vec(stream)?
                            .iter()
                            .all(|&g| g == 0.0),
                        "adamw_bf16_master left the gradient it consumed non-zero"
                    );
                }
            }
            master.to_host_vec(stream)
        };

    for (packed_gradient, label) in [(false, "fp32 gradient"), (true, "packed gradient")] {
        let seen = if packed_gradient {
            unpack_bf16(&pack_bf16(gradient.as_slice()))
        } else {
            gradient.as_slice().to_vec()
        };
        let expected: Vec<f32> = weights
            .iter()
            .zip(&seen)
            .map(|(&weight, &g)| update(weight, g))
            .collect();

        let nearest = run(packed_gradient, device::MASTER_ROUNDING_NEAREST)?;
        assert_eq!(
            nearest,
            pack_bf16(&expected),
            "adamw_bf16_master ({label}) round-to-nearest write-back"
        );

        let stochastic = run(packed_gradient, device::MASTER_ROUNDING_STOCHASTIC)?;
        assert_eq!(
            stochastic,
            run(packed_gradient, device::MASTER_ROUNDING_STOCHASTIC)?,
            "stochastic rounding ({label}) is not reproducible"
        );
        assert_ne!(
            stochastic, nearest,
            "stochastic rounding ({label}) never differed from nearest"
        );
        let nearest_values = unpack_bf16(&nearest);
        for (i, (&drawn, &near)) in unpack_bf16(&stochastic)
            .iter()
            .zip(&nearest_values)
            .enumerate()
        {
            let ulp = (near.abs() / 128.0).max(f32::MIN_POSITIVE);
            assert!(
                (drawn - near).abs() <= ulp,
                "stochastic rounding ({label}) moved element {i} more than one bf16 ulp: \
                 {drawn} vs {near}"
            );
        }
    }
    Ok(())
}

/// The fused write-back that also emits the master's transpose must produce
/// exactly what the flat write-back produces, bit for bit.
///
/// Nothing here is a tolerance: the tile walk reorders which thread owns which
/// element but not the arithmetic, and the stochastic-rounding stream is keyed
/// on the global element index either way, so master words and both moments
/// have to compare equal as words. On top of that the transpose must match the
/// master it was emitted from, and the gradient must come back zeroed.
fn check_adamw_master_transposed(
    stream: &cuda_core::CudaStream,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // Two tiles down by two across, so the walk crosses both block dimensions.
    const ROWS: usize = 128;
    const COLS: usize = 128;
    const LEN: usize = ROWS * COLS;
    const SEED: u64 = tensor_core::rng::stream_seed(5, 11);
    let initial = CpuTensor::<f32, Rank1<LEN>>::uniform(21);
    let master_words = pack_bf16(initial.as_slice());
    let gradient = CpuTensor::<f32, Rank1<LEN>>::uniform(22);
    let moment = CpuTensor::<f32, Rank1<LEN>>::uniform(23);
    let (learning_rate, beta1, beta2, epsilon, weight_decay) = (0.01, 0.9, 0.999, 1e-8, 0.1);
    let (first_correction, second_correction) = (1.0 / (1.0 - beta1), 1.0 / (1.0 - beta2));

    struct Outcome {
        master: Vec<u32>,
        first: Vec<f32>,
        second: Vec<f32>,
        gradient: Vec<f32>,
        transposed: Vec<u32>,
    }

    let mut run = |tiled: bool, rounding: u32| -> Result<Outcome, cuda_core::DriverError> {
        let mut master = DeviceBuffer::from_host(stream, &master_words)?;
        let mut first = DeviceBuffer::from_host(stream, moment.as_slice())?;
        let mut second = DeviceBuffer::from_host(stream, moment.as_slice())?;
        let mut wide = DeviceBuffer::from_host(stream, gradient.as_slice())?;
        let mut transposed = DeviceBuffer::<u32>::zeroed(stream, LEN / 2)?;
        // SAFETY: every buffer describes the same [ROWS, COLS] matrix in the
        // dtype its parameter names, and both launches cover it exactly.
        unsafe {
            if tiled {
                module.adamw_bf16_master_transposed(
                    stream,
                    master_transpose_config(ROWS, COLS),
                    &mut wide,
                    ROWS as u32,
                    COLS as u32,
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                    first_correction,
                    second_correction,
                    rounding,
                    SEED,
                    &mut master,
                    &mut first,
                    &mut second,
                    &mut transposed,
                )?;
            } else {
                module.adamw_bf16_master(
                    stream,
                    LaunchConfig::for_num_elems((LEN / 2) as u32),
                    &mut wide,
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                    first_correction,
                    second_correction,
                    rounding,
                    SEED,
                    &mut master,
                    &mut first,
                    &mut second,
                )?;
            }
        }
        Ok(Outcome {
            master: master.to_host_vec(stream)?,
            first: first.to_host_vec(stream)?,
            second: second.to_host_vec(stream)?,
            gradient: wide.to_host_vec(stream)?,
            transposed: transposed.to_host_vec(stream)?,
        })
    };

    for (rounding, mode) in [
        (device::MASTER_ROUNDING_NEAREST, "nearest"),
        (device::MASTER_ROUNDING_STOCHASTIC, "stochastic"),
    ] {
        let flat = run(false, rounding)?;
        let tiled = run(true, rounding)?;
        assert_eq!(
            flat.master, tiled.master,
            "fused transpose write-back ({mode}) is not bit-identical"
        );
        assert_eq!(
            bits(&flat.first),
            bits(&tiled.first),
            "fused transpose write-back ({mode}) moved the first moment"
        );
        assert_eq!(
            bits(&flat.second),
            bits(&tiled.second),
            "fused transpose write-back ({mode}) moved the second moment"
        );
        for outcome in [&flat, &tiled] {
            assert!(
                outcome.gradient.iter().all(|&g| g == 0.0),
                "an AdamW write-back ({mode}) left the gradient it consumed non-zero"
            );
        }

        let updated = unpack_bf16(&tiled.master);
        let mut expected = vec![0.0f32; LEN];
        for row in 0..ROWS {
            for column in 0..COLS {
                expected[column * ROWS + row] = updated[row * COLS + column];
            }
        }
        assert_eq!(
            tiled.transposed,
            pack_bf16(&expected),
            "fused write-back transpose ({mode}) is not the master's transpose"
        );
    }
    Ok(())
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
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

    let rhs_tn = CpuTensor::<f32, Rank2<M, N>>::uniform(5);
    let gpu_rhs_tn = GpuTensor::from_cpu(stream, &rhs_tn)?;
    assert_close(
        "gemm tn",
        &ga.matmul_tn(&gpu_rhs_tn, stream, module)?.to_host(stream)?,
        a.matmul_tn(&rhs_tn).as_slice(),
        2e-5,
        2e-5,
    );

    let rhs_nt = CpuTensor::<f32, Rank2<N, K>>::uniform(6);
    let gpu_rhs_nt = GpuTensor::from_cpu(stream, &rhs_nt)?;
    assert_close(
        "gemm nt",
        &ga.matmul_nt(&gpu_rhs_nt, stream, module)?.to_host(stream)?,
        a.matmul_nt(&rhs_nt).as_slice(),
        2e-5,
        2e-5,
    );
    Ok(())
}
