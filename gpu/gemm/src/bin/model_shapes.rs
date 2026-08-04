//! Ours-vs-cuBLASLt at the training step's exact GEMM shapes and modes.
//!
//! `bench` prices the kernel at square shapes, in one mode pairing per element.
//! The training step does not run square shapes: it runs the fourteen
//! rectangles below, in three of the kernel's four modes, and the profile's
//! per-row rates spread 0.82–1.55 PF/s across them. This binary puts the same
//! denominator under each row the square bench has — cuBLASLt in-process, on
//! the same stream, through the same clock, reading byte-identical operands,
//! **configured for the same product**: the weight-gradient rows divide by a
//! cuBLASLt `dW += Aᵀ·B` with `β = 1` over the same native `[K, MN]` panels,
//! not by the store they are not.
//!
//! `FEATURES=cublas ./run.sh gemm model_shapes` is the intended spelling;
//! without the feature the binary reports the kernel alone.
//!
//! Shapes are written `MxKxN` — `C[M, N] = A[M, K]·B[K, N]` — matching the
//! step profile's rows. Per-expert shapes use the profile's per-expert token
//! count; the step launches eight of each.

use bench_util::time_gpu_iters;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gemm::{Tcgen05Gemm, TmaLayout, create_bf16_tma_map, tcgen05_launch_config};
use half::bf16;
use std::sync::Arc;

/// Warm-ups discarded and launches timed, the same on both sides of every
/// ratio, and the same as `bench`'s.
const WARMUP: usize = 5;
const ITERS: usize = 20;

/// Largest `C` the host-side agreement check will pull back. Every mode/element
/// combination has at least one shape under it, so each cuBLASLt configuration
/// — where the plausible bug is a transposed operand or a wrong leading
/// dimension, fast and wrong rather than slow and obvious — is checked against
/// the kernel at a rectangle before its time is believed. Above the cap only
/// the shape changes, not the configuration.
#[cfg(feature = "cublas")]
const VERIFY_ELEMENTS: usize = 160_000_000;

/// The kernel mode a row exercises, named by output element and fold.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// `gemm_tcgen05_f32_optimized` mode 2: K-major operands, fp32 `C`
    /// overwritten. Forward projections and backward input gradients.
    F32Store,
    /// `gemm_tcgen05_f32_optimized` mode 3, MN-major: `dW += Aᵀ·B` over native
    /// `[K, M]` / `[K, N]` panels into fp32 gradients.
    F32AccumulateT,
    /// `gemm_tcgen05_bf16_optimized` mode 0: the lm-head forward and its input
    /// gradient.
    Bf16Store,
    /// `gemm_tcgen05_bf16_optimized` mode 1, MN-major: the lm-head weight
    /// gradient, bf16 out.
    Bf16AccumulateT,
}

impl Mode {
    fn transposed(self) -> bool {
        matches!(self, Mode::F32AccumulateT | Mode::Bf16AccumulateT)
    }

    fn label(self) -> &'static str {
        match self {
            Mode::F32Store => "f32 store",
            Mode::F32AccumulateT => "f32 acc-T",
            Mode::Bf16Store => "bf16 store",
            Mode::Bf16AccumulateT => "bf16 acc-T",
        }
    }
}

struct Case {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    mode: Mode,
}

const fn case(name: &'static str, m: usize, k: usize, n: usize, mode: Mode) -> Case {
    Case {
        name,
        m,
        k,
        n,
        mode,
    }
}

/// Every tcgen05 GEMM the training step launches, at its exact shape and mode.
const CASES: &[Case] = &[
    case("qkv fwd", 24576, 3072, 9216, Mode::F32Store),
    case("o_proj fwd", 24576, 3072, 3072, Mode::F32Store),
    case("gate_up fwd /expert", 6144, 3072, 8192, Mode::F32Store),
    case("down fwd /expert", 6144, 4096, 3072, Mode::F32Store),
    case("bwd qkv dx", 24576, 9216, 3072, Mode::F32Store),
    case("bwd gate_up dx /expert", 6144, 8192, 3072, Mode::F32Store),
    case("bwd down dx /expert", 6144, 3072, 4096, Mode::F32Store),
    case("bwd qkv dW", 3072, 24576, 9216, Mode::F32AccumulateT),
    case("bwd o_proj dW", 3072, 24576, 3072, Mode::F32AccumulateT),
    case(
        "bwd gate_up dW /expert",
        3072,
        6144,
        8192,
        Mode::F32AccumulateT,
    ),
    case(
        "bwd down dW /expert",
        4096,
        6144,
        3072,
        Mode::F32AccumulateT,
    ),
    case("lm_head fwd", 24576, 3072, 50432, Mode::Bf16Store),
    case("bwd lm_head dx", 24576, 50432, 3072, Mode::Bf16Store),
    case("bwd lm_head dW", 3072, 24576, 50432, Mode::Bf16AccumulateT),
];

fn tflops(m: usize, n: usize, k: usize, milliseconds: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / (milliseconds / 1_000.0) / 1.0e12
}

/// A packed row-major bf16 operand — `bench`'s generator, byte for byte.
fn operand(elements: usize, seed: u64) -> Vec<u16> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..elements)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let value = ((state >> 33) % 9) as f32 / 4.0 - 1.0;
            bf16::from_f32(value).to_bits()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = Tcgen05Gemm::load(&context)?;

    println!(
        "model-shape GEMMs, MxKxN, {} clusters persistent (tiles = M/256 * N/256)",
        gemm::MAX_CLUSTERS
    );
    #[cfg(feature = "cublas")]
    println!("  denominator: {}", gemm::cublaslt::about());
    #[cfg(not(feature = "cublas"))]
    println!("  no denominator: built without `--features cublas`");

    let mut summary = Vec::new();
    for case in CASES {
        run_case(&stream, &module, case, &mut summary)?;
    }

    println!();
    println!(
        "{:<24} {:<11} {:>7} {:>26} {:>26} {:>6}",
        "shape (MxKxN)", "mode", "tiles", "ours", "cuBLASLt", "ratio"
    );
    for row in &summary {
        println!("{row}");
    }

    println!();
    println!(
        "walk and fold priced apart (ours only): store is mode 2 K-major, \
         store-T mode 2 MN-major, acc-T mode 3 MN-major"
    );
    for case in CASES.iter().filter(|c| c.mode == Mode::F32AccumulateT) {
        decompose(&stream, &module, case)?;
    }
    Ok(())
}

/// Three timings at one weight-gradient shape, one variable apart each:
/// `store → store-T` is the MN-major operand walk, `store-T → acc-T` is the
/// fold's extra read-modify of `C`. Checked first: from a zeroed `C` the fold
/// adds exact zeros, so store-T and one acc-T launch must agree bitwise.
fn decompose(
    stream: &Arc<CudaStream>,
    module: &Tcgen05Gemm,
    case: &Case,
) -> Result<(), Box<dyn std::error::Error>> {
    let (m, k, n) = (case.m, case.k, case.n);
    let a = DeviceBuffer::from_host(stream, &operand(m * k, 13))?;
    let b = DeviceBuffer::from_host(stream, &operand(n * k, 14))?;
    let a_k = create_bf16_tma_map(stream, &a, k, m, TmaLayout::KMajor)?;
    let b_k = create_bf16_tma_map(stream, &b, k, n, TmaLayout::KMajor)?;
    let a_mn = create_bf16_tma_map(stream, &a, m, k, TmaLayout::MnMajor)?;
    let b_mn = create_bf16_tma_map(stream, &b, n, k, TmaLayout::MnMajor)?;
    let config = tcgen05_launch_config(m, n, k);
    let mut c = DeviceBuffer::<f32>::zeroed(stream, m * n)?;

    unsafe {
        module.f32_accumulate_transposed(
            stream,
            config,
            a_mn.as_ptr(),
            b_mn.as_ptr(),
            &mut c,
            n as u32,
            k as u32,
        )
    }?;
    let folded_once = c.to_host_vec(stream)?;
    unsafe {
        module.f32_store_transposed(
            stream,
            config,
            a_mn.as_ptr(),
            b_mn.as_ptr(),
            &mut c,
            n as u32,
            k as u32,
        )
    }?;
    let stored = c.to_host_vec(stream)?;
    if let Some(at) = (0..m * n).find(|&at| folded_once[at].to_bits() != stored[at].to_bits()) {
        return Err(format!(
            "store-T and a from-zero acc-T disagree at {} ({}, {}): {} vs {}",
            case.name,
            at / n,
            at % n,
            stored[at],
            folded_once[at]
        )
        .into());
    }

    let store = time_gpu_iters(stream, WARMUP, ITERS, || {
        unsafe {
            module.f32_store(
                stream,
                config,
                a_k.as_ptr(),
                b_k.as_ptr(),
                &mut c,
                n as u32,
                k as u32,
            )
        }
        .map_err(Into::into)
    })?;
    let store_t = time_gpu_iters(stream, WARMUP, ITERS, || {
        unsafe {
            module.f32_store_transposed(
                stream,
                config,
                a_mn.as_ptr(),
                b_mn.as_ptr(),
                &mut c,
                n as u32,
                k as u32,
            )
        }
        .map_err(Into::into)
    })?;
    let acc_t = time_gpu_iters(stream, WARMUP, ITERS, || {
        unsafe {
            module.f32_accumulate_transposed(
                stream,
                config,
                a_mn.as_ptr(),
                b_mn.as_ptr(),
                &mut c,
                n as u32,
                k as u32,
            )
        }
        .map_err(Into::into)
    })?;
    println!(
        "{}: {m}x{k}x{n}  store {store:7.3} ms {:7.2} TF/s | store-T {store_t:7.3} ms {:7.2} TF/s \
         (walk {:+5.1}%) | acc-T {acc_t:7.3} ms {:7.2} TF/s (fold {:+5.1}%)",
        case.name,
        tflops(m, n, k, store),
        tflops(m, n, k, store_t),
        100.0 * (store_t - store) / store,
        tflops(m, n, k, acc_t),
        100.0 * (acc_t - store_t) / store_t,
    );
    Ok(())
}

fn run_case(
    stream: &Arc<CudaStream>,
    module: &Tcgen05Gemm,
    case: &Case,
    summary: &mut Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (m, k, n) = (case.m, case.k, case.n);
    // The operands' *storage* is `[M, K]` and `[K, N]`-shaped bytes either way;
    // what the mode changes is which axis the tensor map calls its width.
    let a = DeviceBuffer::from_host(stream, &operand(m * k, 13))?;
    let b = DeviceBuffer::from_host(stream, &operand(n * k, 14))?;
    let (a_tma, b_tma) = if case.mode.transposed() {
        (
            create_bf16_tma_map(stream, &a, m, k, TmaLayout::MnMajor)?,
            create_bf16_tma_map(stream, &b, n, k, TmaLayout::MnMajor)?,
        )
    } else {
        (
            create_bf16_tma_map(stream, &a, k, m, TmaLayout::KMajor)?,
            create_bf16_tma_map(stream, &b, k, n, TmaLayout::KMajor)?,
        )
    };
    let config = tcgen05_launch_config(m, n, k);
    let tiles = (m / 256) * (n / 256);

    // One clean launch into a zeroed `C` before anything is timed: the
    // accumulating modes fold, so the agreement check has to know the fold
    // started from zero. The timed launches keep folding into the same buffer,
    // which changes its values and none of its addresses.
    let (ours, baseline) = match case.mode {
        Mode::F32Store | Mode::F32AccumulateT => {
            let mut c = DeviceBuffer::<f32>::zeroed(stream, m * n)?;
            let launch = |c: &mut DeviceBuffer<f32>| -> Result<(), Box<dyn std::error::Error>> {
                let (a_tma, b_tma) = (a_tma.as_ptr(), b_tma.as_ptr());
                match case.mode {
                    Mode::F32Store => unsafe {
                        module.f32_store(stream, config, a_tma, b_tma, c, n as u32, k as u32)
                    },
                    _ => unsafe {
                        module.f32_accumulate_transposed(
                            stream, config, a_tma, b_tma, c, n as u32, k as u32,
                        )
                    },
                }
                .map_err(Into::into)
            };
            launch(&mut c)?;
            let baseline = baseline_f32(stream, &a, &b, &c, case)?;
            let ours = time_gpu_iters(stream, WARMUP, ITERS, || launch(&mut c))?;
            (ours, baseline)
        }
        Mode::Bf16Store | Mode::Bf16AccumulateT => {
            let mut c = DeviceBuffer::<u32>::zeroed(stream, m * n / 2)?;
            let launch = |c: &mut DeviceBuffer<u32>| -> Result<(), Box<dyn std::error::Error>> {
                let (a_tma, b_tma) = (a_tma.as_ptr(), b_tma.as_ptr());
                match case.mode {
                    Mode::Bf16Store => unsafe {
                        module.store(stream, config, a_tma, b_tma, c, n as u32, k as u32)
                    },
                    _ => unsafe {
                        module.accumulate_transposed(
                            stream, config, a_tma, b_tma, c, n as u32, k as u32,
                        )
                    },
                }
                .map_err(Into::into)
            };
            launch(&mut c)?;
            let baseline = baseline_bf16(stream, &a, &b, &c, case)?;
            let ours = time_gpu_iters(stream, WARMUP, ITERS, || launch(&mut c))?;
            (ours, baseline)
        }
    };

    println!(
        "{}: {m}x{k}x{n} {} ({tiles} tiles)",
        case.name,
        case.mode.label()
    );
    let row = if let Some((theirs, algorithm)) = &baseline {
        println!(
            "  cuBLASLt {theirs:8.3} ms  {:8.2} TFLOP/s   {algorithm}",
            tflops(m, n, k, *theirs)
        );
        let ratio = theirs / ours;
        let flag = if ratio < 0.90 {
            "  ** below 0.90 **"
        } else {
            ""
        };
        println!(
            "  ours     {ours:8.3} ms  {:8.2} TFLOP/s   {ratio:.3} of cuBLASLt{flag}",
            tflops(m, n, k, ours)
        );
        format!(
            "{:>13.3} ms {:8.2} TF/s {:>13.3} ms {:8.2} TF/s {ratio:>6.3}{flag}",
            ours,
            tflops(m, n, k, ours),
            theirs,
            tflops(m, n, k, *theirs)
        )
    } else {
        println!(
            "  ours     {ours:8.3} ms  {:8.2} TFLOP/s",
            tflops(m, n, k, ours)
        );
        format!("{:>13.3} ms {:8.2} TF/s", ours, tflops(m, n, k, ours))
    };
    summary.push(format!(
        "{:<24} {:<11} {tiles:>7} {row}",
        format!("{m}x{k}x{n} {}", case.name),
        case.mode.label()
    ));
    Ok(())
}

#[cfg(feature = "cublas")]
fn form(case: &Case) -> gemm::cublaslt::Form {
    use gemm::cublaslt::Form;
    if case.mode.transposed() {
        Form::AccumulateTransposed
    } else {
        Form::Store
    }
}

/// The baseline's time and chosen algorithm at this case's form, fp32 out —
/// checked against the kernel's clean run before any time is reported.
#[cfg(feature = "cublas")]
fn baseline_f32(
    stream: &Arc<CudaStream>,
    a: &DeviceBuffer<u16>,
    b: &DeviceBuffer<u16>,
    ours: &DeviceBuffer<f32>,
    case: &Case,
) -> Result<Option<(f64, String)>, Box<dyn std::error::Error>> {
    use gemm::cublaslt::{Baseline, OutElement};
    let baseline =
        Baseline::with_form(stream, case.m, case.n, case.k, form(case), OutElement::F32)?;
    let c = DeviceBuffer::<f32>::zeroed(stream, case.m * case.n)?;
    unsafe {
        baseline.launch_devptrs(stream, a.cu_deviceptr(), b.cu_deviceptr(), c.cu_deviceptr())
    }?;
    stream.synchronize()?;
    if case.m * case.n <= VERIFY_ELEMENTS {
        // Both sides accumulate fp32 and reduce K in different orders, so the
        // values differ by reassociation; what the check is for — a transposed
        // operand, a wrong leading dimension — is wrong by whole values.
        agree_f32(&c.to_host_vec(stream)?, &ours.to_host_vec(stream)?, case)?;
    }
    let ms = time_gpu_iters(stream, WARMUP, ITERS, || unsafe {
        baseline.launch_devptrs(stream, a.cu_deviceptr(), b.cu_deviceptr(), c.cu_deviceptr())
    })?;
    Ok(Some((ms, baseline.algorithm())))
}

/// [`baseline_f32`] with the kernel's packed-bf16 `C` on the other side.
#[cfg(feature = "cublas")]
fn baseline_bf16(
    stream: &Arc<CudaStream>,
    a: &DeviceBuffer<u16>,
    b: &DeviceBuffer<u16>,
    ours: &DeviceBuffer<u32>,
    case: &Case,
) -> Result<Option<(f64, String)>, Box<dyn std::error::Error>> {
    use gemm::cublaslt::{Baseline, OutElement};
    let baseline =
        Baseline::with_form(stream, case.m, case.n, case.k, form(case), OutElement::Bf16)?;
    let c = DeviceBuffer::<u16>::zeroed(stream, case.m * case.n)?;
    unsafe {
        baseline.launch_devptrs(stream, a.cu_deviceptr(), b.cu_deviceptr(), c.cu_deviceptr())
    }?;
    stream.synchronize()?;
    if case.m * case.n <= VERIFY_ELEMENTS {
        agree_bf16(&c.to_host_vec(stream)?, &ours.to_host_vec(stream)?, case)?;
    }
    let ms = time_gpu_iters(stream, WARMUP, ITERS, || unsafe {
        baseline.launch_devptrs(stream, a.cu_deviceptr(), b.cu_deviceptr(), c.cu_deviceptr())
    })?;
    Ok(Some((ms, baseline.algorithm())))
}

/// Do the two fp32 `C`s agree to within reassociation error?
///
/// The tolerance is deliberately loose — 1/32 of the value or of 1.0,
/// whichever scale is larger — because K runs to 50432 and the two sides sum
/// it in different orders. Configuration bugs are wrong by whole values.
#[cfg(feature = "cublas")]
fn agree_f32(theirs: &[f32], ours: &[f32], case: &Case) -> Result<(), Box<dyn std::error::Error>> {
    let mut worst = (0.0f32, 0usize);
    let mut wrong = 0usize;
    for (index, (mine, theirs)) in ours.iter().zip(theirs).enumerate() {
        let error = (mine - theirs).abs() / theirs.abs().max(1.0);
        if error > worst.0 {
            worst = (error, index);
        }
        if error > 1.0 / 32.0 {
            wrong += 1;
        }
    }
    disagreement(wrong, worst, case)
}

/// `bench`'s packed-bf16 agreement check, at twice its tolerance: K here runs
/// to 50432, and reassociation that deep can reach a couple of bf16 ulp.
#[cfg(feature = "cublas")]
fn agree_bf16(theirs: &[u16], ours: &[u32], case: &Case) -> Result<(), Box<dyn std::error::Error>> {
    let widen = |bits: u16| f32::from_bits((bits as u32) << 16);
    let mut worst = (0.0f32, 0usize);
    let mut wrong = 0usize;
    for (index, theirs) in theirs.iter().enumerate() {
        let mine = widen((ours[index / 2] >> (16 * (index % 2))) as u16);
        let theirs = widen(*theirs);
        let error = (mine - theirs).abs() / theirs.abs().max(1.0);
        if error > worst.0 {
            worst = (error, index);
        }
        if error > 4.0 / 256.0 {
            wrong += 1;
        }
    }
    disagreement(wrong, worst, case)
}

#[cfg(feature = "cublas")]
fn disagreement(
    wrong: usize,
    worst: (f32, usize),
    case: &Case,
) -> Result<(), Box<dyn std::error::Error>> {
    if wrong == 0 {
        return Ok(());
    }
    let (error, index) = worst;
    Err(format!(
        "cuBLASLt and the kernel disagree on {wrong} of {} elements at {} \
         ({}x{}x{} {}): worst |rel| {error:.3e} at ({}, {})",
        case.m * case.n,
        case.name,
        case.m,
        case.k,
        case.n,
        case.mode.label(),
        index / case.n,
        index % case.n
    )
    .into())
}

#[cfg(not(feature = "cublas"))]
fn baseline_f32(
    _stream: &Arc<CudaStream>,
    _a: &DeviceBuffer<u16>,
    _b: &DeviceBuffer<u16>,
    _ours: &DeviceBuffer<f32>,
    _case: &Case,
) -> Result<Option<(f64, String)>, Box<dyn std::error::Error>> {
    Ok(None)
}

#[cfg(not(feature = "cublas"))]
fn baseline_bf16(
    _stream: &Arc<CudaStream>,
    _a: &DeviceBuffer<u16>,
    _b: &DeviceBuffer<u16>,
    _ours: &DeviceBuffer<u32>,
    _case: &Case,
) -> Result<Option<(f64, String)>, Box<dyn std::error::Error>> {
    Ok(None)
}
