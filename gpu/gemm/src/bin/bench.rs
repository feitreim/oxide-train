//! CUDA-event throughput benchmark for the GEMM ladder, against cuBLASLt.
//!
//! `./run.sh gemm bench` runs the kernels alone. `FEATURES=cublas ./run.sh gemm
//! bench` adds the denominator: cuBLASLt in the same process, on the same
//! stream, through the same clock, reading byte-identical operands — so the
//! only thing left different between the two figures is the code that runs
//! between the events.
//!
//! The ratios are all against **cuBLASLt's store**, `C = A·Bᵀ` with a bf16 `C`,
//! because that is the one signature both sides have. The accumulating and fp32
//! rows are divided by it too, and they write or move more than it does; read
//! those as "what the fold costs against a library that is not doing it"
//! rather than as a like-for-like defeat.
//!
//! Sizes go to 16384³ deliberately. The persistent grid's whole argument is
//! that it stops paying cluster start-up per wave, and 16384³ is 37 waves of
//! it — the shape where the kernel this replaced was furthest behind.

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer};
use gemm::{
    BK, BM, BN, TM, TN, Tcgen05Gemm, TmaLayout, create_bf16_tma_map, fp32, fp32_launch_config,
    tcgen05_launch_config,
};
use half::bf16;

const FP32_M: usize = 2048;
const FP32_N: usize = 2048;
const FP32_K: usize = 2048;

/// Square shapes the tcgen05 rung is reported at. 4096³ is the size every
/// figure in this repo's history is quoted at, and the two above it are where
/// the schedule change is supposed to show.
const SHAPES: [usize; 3] = [4096, 8192, 16384];

/// Warm-ups discarded and launches timed, the same on both sides of every
/// ratio.
const WARMUP: usize = 5;
const ITERS: usize = 20;

/// Elements of `C` the baseline's correctness check will pull to the host.
#[cfg(feature = "cublas")]
const VERIFY_ELEMENTS: usize = 4096 * 4096;

fn tflops(m: usize, n: usize, k: usize, milliseconds: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / (milliseconds / 1_000.0) / 1.0e12
}

/// A packed row-major bf16 operand, generated straight into its 16-bit storage.
///
/// Not `uniform_vec(..).map(..)`: at 16384² the fp32 intermediate is a gigabyte
/// of host memory that exists only to be narrowed. The values are arbitrary —
/// nothing here checks numerics — but both sides of every comparison read the
/// *same* buffer, which is the only property that matters.
fn operand(elements: usize, seed: u64) -> Vec<u16> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..elements)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // A small signed value, exact in bf16 and far from overflow at
            // K = 16384.
            let value = ((state >> 33) % 9) as f32 / 4.0 - 1.0;
            bf16::from_f32(value).to_bits()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = Tcgen05Gemm::load(&context)?;

    bench_fp32(&context, &stream)?;

    println!(
        "bf16 tcgen05 cta_group::2, [256, 256] tile, K=64 x {} stages, persistent \
         over {} clusters, grouped at {}, {} B plan",
        gemm::STAGES,
        gemm::MAX_CLUSTERS,
        gemm::GROUP,
        gemm::SHARED_BYTES
    );
    #[cfg(feature = "cublas")]
    println!("  denominator: {}", gemm::cublaslt::about());
    #[cfg(not(feature = "cublas"))]
    println!("  no denominator: built without `--features cublas`");

    for size in SHAPES {
        bench_tcgen05(&stream, &module, size)?;
    }
    Ok(())
}

fn bench_fp32(
    context: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    let module = fp32::kernels::load(context)?;
    let a = DeviceBuffer::from_host(stream, &uniform_vec(FP32_M * FP32_K, 11))?;
    let b = DeviceBuffer::from_host(stream, &uniform_vec(FP32_K * FP32_N, 12))?;
    let mut c = DeviceBuffer::<f32>::zeroed(stream, FP32_M * FP32_N)?;
    let config = fp32_launch_config(FP32_M, FP32_N);

    let store = time_gpu_iters(stream, 3, 10, || {
        unsafe {
            module.register_gemm_store(stream, config, FP32_M, FP32_N, FP32_K, &a, &b, &mut c)
        }
        .map_err(Into::into)
    })?;
    let accumulate = time_gpu_iters(stream, 3, 10, || {
        unsafe {
            module.register_gemm_accumulate(stream, config, FP32_M, FP32_N, FP32_K, &a, &b, &mut c)
        }
        .map_err(Into::into)
    })?;

    println!("fp32 register tile BM={BM} BN={BN} BK={BK} TM={TM} TN={TN}");
    println!(
        "  store      [{FP32_M},{FP32_K}]x[{FP32_K},{FP32_N}]: {store:8.3} ms  {:8.2} TFLOP/s",
        tflops(FP32_M, FP32_N, FP32_K, store)
    );
    println!(
        "  accumulate [{FP32_M},{FP32_K}]x[{FP32_K},{FP32_N}]: \
         {accumulate:8.3} ms  {:8.2} TFLOP/s",
        tflops(FP32_M, FP32_N, FP32_K, accumulate)
    );
    Ok(())
}

fn bench_tcgen05(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &Tcgen05Gemm,
    size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let (m, n, k) = (size, size, size);
    let a = DeviceBuffer::from_host(stream, &operand(m * k, 13))?;
    let b = DeviceBuffer::from_host(stream, &operand(n * k, 14))?;
    let a_tma = create_bf16_tma_map(stream, &a, k, m, TmaLayout::KMajor)?;
    let b_tma = create_bf16_tma_map(stream, &b, k, n, TmaLayout::KMajor)?;
    let config = tcgen05_launch_config(m, n, k);

    let mut packed = DeviceBuffer::<u32>::zeroed(stream, m * n / 2)?;
    let store = time_gpu_iters(stream, WARMUP, ITERS, || {
        unsafe {
            module.store(
                stream,
                config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut packed,
                n as u32,
                k as u32,
            )
        }
        .map_err(Into::into)
    })?;
    // The baseline goes here, while our own `C` is still the store's: it is
    // compared against ours before any time is reported.
    let baseline = baseline_ms(stream, &a, &b, &packed, m, n, k)?;

    let accumulate = time_gpu_iters(stream, WARMUP, ITERS, || {
        unsafe {
            module.accumulate(
                stream,
                config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut packed,
                n as u32,
                k as u32,
            )
        }
        .map_err(Into::into)
    })?;
    drop(packed);

    let mut wide = DeviceBuffer::<f32>::zeroed(stream, m * n)?;
    let f32_store = time_gpu_iters(stream, WARMUP, ITERS, || {
        unsafe {
            module.f32_store(
                stream,
                config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut wide,
                n as u32,
                k as u32,
            )
        }
        .map_err(Into::into)
    })?;
    let f32_accumulate = time_gpu_iters(stream, WARMUP, ITERS, || {
        unsafe {
            module.f32_accumulate(
                stream,
                config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut wide,
                n as u32,
                k as u32,
            )
        }
        .map_err(Into::into)
    })?;
    drop(wide);

    println!("{m}x{n}x{k}");
    if let Some((milliseconds, algorithm)) = &baseline {
        println!(
            "  cuBLASLt        {milliseconds:8.3} ms  {:8.2} TFLOP/s   {algorithm}",
            tflops(m, n, k, *milliseconds)
        );
    }
    for (name, milliseconds) in [
        ("bf16 store     ", store),
        ("bf16 accumulate", accumulate),
        ("f32 store      ", f32_store),
        ("f32 accumulate ", f32_accumulate),
    ] {
        let ratio = match &baseline {
            Some((reference, _)) => format!("   {:.3} of cuBLASLt", reference / milliseconds),
            None => String::new(),
        };
        println!(
            "  {name} {milliseconds:8.3} ms  {:8.2} TFLOP/s{ratio}",
            tflops(m, n, k, milliseconds)
        );
    }
    Ok(())
}

/// The baseline's time and the algorithm it chose, or `None` when the binary
/// was built without the feature that can link it.
///
/// Its `C` is compared against `ours` — the store's output, still in place —
/// before any time is reported. The plausible bug in a column-major baseline is
/// a transposed operand, which is fast and wrong rather than slow and obvious,
/// and would be a denominator with nothing visibly the matter with it.
#[cfg(feature = "cublas")]
fn baseline_ms(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    a: &DeviceBuffer<u16>,
    b: &DeviceBuffer<u16>,
    ours: &DeviceBuffer<u32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Option<(f64, String)>, Box<dyn std::error::Error>> {
    let baseline = gemm::cublaslt::Baseline::new(stream, m, n, k)?;
    let c = DeviceBuffer::<u16>::zeroed(stream, m * n)?;
    unsafe { baseline.launch(stream, a, b, &c) }?;
    stream.synchronize()?;
    // At the smallest shape only. The comparison is host-side and a 16384²
    // pair is a gigabyte and a half of it; what it is checking — that the
    // operands are not transposed and the layouts line up — is a property of
    // the configuration and does not vary with the size.
    if m * n <= VERIFY_ELEMENTS {
        agrees(&c.to_host_vec(stream)?, &ours.to_host_vec(stream)?, m, n)?;
    }

    let milliseconds = time_gpu_iters(stream, WARMUP, ITERS, || unsafe {
        baseline.launch(stream, a, b, &c)
    })?;
    Ok(Some((milliseconds, baseline.algorithm())))
}

/// Do the two `C`s agree to within one bf16 ulp of each other?
///
/// Not `==`: both sides accumulate in fp32 and round once, but they reduce K in
/// different orders, so the fp32 values reaching the two epilogues differ in the
/// last bits and can round to adjacent bf16. What this catches is the failure
/// worth catching — a transposed operand, a dropped tile, a wrong leading
/// dimension — all of which are wrong by whole values and not by an ulp.
#[cfg(feature = "cublas")]
fn agrees(
    theirs: &[u16],
    ours: &[u32],
    m: usize,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let widen = |bits: u16| f32::from_bits((bits as u32) << 16);
    let mut worst = (0.0f32, 0usize);
    let mut wrong = 0usize;
    for index in 0..m * n {
        let mine = widen((ours[index / 2] >> (16 * (index % 2))) as u16);
        let (theirs, scale) = (widen(theirs[index]), widen(theirs[index]).abs().max(1.0));
        let error = (mine - theirs).abs() / scale;
        if error > worst.0 {
            worst = (error, index);
        }
        // One bf16 ulp is 2^-8 of the value; allow two.
        if error > 2.0 / 256.0 {
            wrong += 1;
        }
    }
    if wrong > 0 {
        let (error, index) = worst;
        return Err(format!(
            "cuBLASLt and the kernel disagree on {wrong} of {} elements at {m}x{n}: \
             worst |rel| {error:.3e} at ({}, {})",
            m * n,
            index / n,
            index % n
        )
        .into());
    }
    Ok(())
}

#[cfg(not(feature = "cublas"))]
fn baseline_ms(
    _stream: &std::sync::Arc<cuda_core::CudaStream>,
    _a: &DeviceBuffer<u16>,
    _b: &DeviceBuffer<u16>,
    _ours: &DeviceBuffer<u32>,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<Option<(f64, String)>, Box<dyn std::error::Error>> {
    Ok(None)
}
