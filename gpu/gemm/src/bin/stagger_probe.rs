//! The anti-phase probe — oxide-train#80 remedy 2, asked as a schedule
//! question before it is asked as an architecture one.
//!
//! ferro-kittens #114 measured this kernel's epilogue fully exposed (1.01×
//! with the MMA beside it and without), and ferro #188 measured a lone
//! `M256_N256` chain at ~0.90 of two chains' aggregate — together those say an
//! SM's two resident CTAs *could* cover most of either one's drain with the
//! other's MMA and currently never do, because the static schedule starts
//! every cluster together on identically-priced items and they stay in phase.
//! `stagger_start` delays half the clusters' first item once per launch;
//! identical item periods then preserve the offset. This binary sweeps the
//! delay and both co-residency guesses (upper-half vs odd cluster indices) at
//! the worst shallow-K `model_shapes` rows, with cuBLASLt as the denominator
//! and a deep-K row as the do-not-regress guard.
//!
//! `FEATURES=cublas ./run.sh gemm stagger_probe` is the intended spelling.

use bench_util::time_gpu_iters;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gemm::{Tcgen05Gemm, TmaLayout, create_bf16_tma_map, tcgen05_launch_config};
use half::bf16;
use std::sync::Arc;

const WARMUP: usize = 5;
const ITERS: usize = 20;

/// Delays swept, in microseconds. The drain the stagger exists to hide is
/// ~6–7 µs (bf16) / ~10–13 µs (f32) per tile, and the delay is paid once per
/// launch — so past ~2× the drain the sweep is pricing the delay, not the
/// phase.
const MICROSECONDS: [u32; 6] = [0, 4, 8, 12, 16, 20];

/// Who is late: `0` the upper half of cluster indices (co-residency by launch
/// wave), `1` the odd ones (co-residency by adjacency).
const RULES: [u32; 2] = [0, 1];

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    F32Store,
    F32AccumulateT,
    Bf16Store,
}

struct Case {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    mode: Mode,
}

/// The three worst f32-store rows, the worst acc-T row, the bf16 row that
/// shares their K, and one deep-K guard.
const CASES: &[Case] = &[
    Case {
        name: "gate_up fwd",
        m: 6144,
        k: 3072,
        n: 8192,
        mode: Mode::F32Store,
    },
    Case {
        name: "qkv fwd",
        m: 24576,
        k: 3072,
        n: 9216,
        mode: Mode::F32Store,
    },
    Case {
        name: "bwd down dx",
        m: 6144,
        k: 3072,
        n: 4096,
        mode: Mode::F32Store,
    },
    Case {
        name: "bwd down dW",
        m: 4096,
        k: 6144,
        n: 3072,
        mode: Mode::F32AccumulateT,
    },
    Case {
        name: "lm_head fwd",
        m: 24576,
        k: 3072,
        n: 50432,
        mode: Mode::Bf16Store,
    },
    Case {
        name: "bwd gate_up dx (guard)",
        m: 6144,
        k: 8192,
        n: 3072,
        mode: Mode::F32Store,
    },
];

fn tflops(m: usize, n: usize, k: usize, milliseconds: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / (milliseconds / 1_000.0) / 1.0e12
}

/// `bench`'s packed row-major bf16 operand generator, byte for byte.
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

fn encode(microseconds: u32, rule: u32) -> u32 {
    if microseconds == 0 {
        0
    } else {
        (rule << 24) | (microseconds * 1_000)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = Tcgen05Gemm::load(&context)?;

    println!(
        "stagger probe (oxide-train#80 remedy 2): one launch-time delay on half the\n\
         clusters, swept over {MICROSECONDS:?} us at rules {{0: upper half, 1: odd}},\n\
         {} clusters persistent",
        gemm::MAX_CLUSTERS
    );
    #[cfg(feature = "cublas")]
    println!("  denominator: {}", gemm::cublaslt::about());
    #[cfg(not(feature = "cublas"))]
    println!("  no denominator: built without `--features cublas`");

    for case in CASES {
        run_case(&stream, &module, case)?;
    }
    Ok(())
}

fn run_case(
    stream: &Arc<CudaStream>,
    module: &Tcgen05Gemm,
    case: &Case,
) -> Result<(), Box<dyn std::error::Error>> {
    let (m, k, n) = (case.m, case.k, case.n);
    let a = DeviceBuffer::from_host(stream, &operand(m * k, 13))?;
    let b = DeviceBuffer::from_host(stream, &operand(n * k, 14))?;
    let (a_tma, b_tma) = if case.mode == Mode::F32AccumulateT {
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
    let mode = match case.mode {
        Mode::F32Store => "f32 store",
        Mode::F32AccumulateT => "f32 acc-T",
        Mode::Bf16Store => "bf16 store",
    };
    println!(
        "\n{}: {m}x{k}x{n} {mode} ({} tiles)",
        case.name,
        (m / 256) * (n / 256)
    );

    let mut c_f32 = DeviceBuffer::<f32>::zeroed(stream, m * n)?;
    let mut c_bf16 = DeviceBuffer::<u32>::zeroed(stream, m * n / 2)?;
    let launch = |c_f32: &mut DeviceBuffer<f32>,
                  c_bf16: &mut DeviceBuffer<u32>,
                  stagger: u32|
     -> Result<(), Box<dyn std::error::Error>> {
        let (a_tma, b_tma) = (a_tma.as_ptr(), b_tma.as_ptr());
        match case.mode {
            Mode::F32Store => unsafe {
                module.f32_store_staggered(
                    stream, config, a_tma, b_tma, c_f32, n as u32, k as u32, stagger,
                )
            },
            Mode::F32AccumulateT => unsafe {
                module.f32_accumulate_transposed_staggered(
                    stream, config, a_tma, b_tma, c_f32, n as u32, k as u32, stagger,
                )
            },
            Mode::Bf16Store => unsafe {
                module.store_staggered(
                    stream, config, a_tma, b_tma, c_bf16, n as u32, k as u32, stagger,
                )
            },
        }
        .map_err(Into::into)
    };

    // The stagger is a delay and nothing else, so a staggered f32 store must
    // reproduce the unstaggered one bitwise — the check that the parameter
    // reached the kernel without perturbing anything it should not.
    if case.mode == Mode::F32Store {
        launch(&mut c_f32, &mut c_bf16, 0)?;
        let plain = c_f32.to_host_vec(stream)?;
        launch(&mut c_f32, &mut c_bf16, encode(12, 0))?;
        let staggered = c_f32.to_host_vec(stream)?;
        if let Some(at) = (0..m * n).find(|&at| plain[at].to_bits() != staggered[at].to_bits()) {
            return Err(format!(
                "staggered launch disagrees with plain at {} ({}, {}): {} vs {}",
                case.name,
                at / n,
                at % n,
                plain[at],
                staggered[at]
            )
            .into());
        }
    }

    let plain = time_gpu_iters(stream, WARMUP, ITERS, || {
        launch(&mut c_f32, &mut c_bf16, 0)
    })?;
    println!(
        "  stagger 0            {plain:8.3} ms  {:8.2} TF/s",
        tflops(m, n, k, plain)
    );
    let mut best = (plain, 0u32, 0u32);
    for rule in RULES {
        for microseconds in MICROSECONDS {
            if microseconds == 0 {
                continue;
            }
            let stagger = encode(microseconds, rule);
            let ours = time_gpu_iters(stream, WARMUP, ITERS, || {
                launch(&mut c_f32, &mut c_bf16, stagger)
            })?;
            println!(
                "  stagger {microseconds:>2} us rule {rule} {ours:8.3} ms  {:8.2} TF/s  {:+6.1}% vs plain",
                tflops(m, n, k, ours),
                100.0 * (plain - ours) / plain
            );
            if ours < best.0 {
                best = (ours, microseconds, rule);
            }
        }
    }

    let baseline = baseline(stream, &a, &b, case)?;
    if let Some((theirs, algorithm)) = baseline {
        println!(
            "  cuBLASLt             {theirs:8.3} ms  {:8.2} TF/s   {algorithm}",
            tflops(m, n, k, theirs)
        );
        println!(
            "  ratio: plain {:.3}, best {:.3} (stagger {} us rule {})",
            theirs / plain,
            theirs / best.0,
            best.1,
            best.2
        );
    }
    Ok(())
}

/// The library's time at this case's form — no agreement check here: the
/// kernels are `model_shapes`-verified, and the stagger is checked against the
/// plain launch bitwise above.
#[cfg(feature = "cublas")]
fn baseline(
    stream: &Arc<CudaStream>,
    a: &DeviceBuffer<u16>,
    b: &DeviceBuffer<u16>,
    case: &Case,
) -> Result<Option<(f64, String)>, Box<dyn std::error::Error>> {
    use gemm::cublaslt::{Baseline, Form, OutElement};
    let form = if case.mode == Mode::F32AccumulateT {
        Form::AccumulateTransposed
    } else {
        Form::Store
    };
    let out = if case.mode == Mode::Bf16Store {
        OutElement::Bf16
    } else {
        OutElement::F32
    };
    let baseline = Baseline::with_form(stream, case.m, case.n, case.k, form, out)?;
    let elements = case.m * case.n;
    let (c_f32, c_bf16);
    let c = match out {
        OutElement::F32 => {
            c_f32 = DeviceBuffer::<f32>::zeroed(stream, elements)?;
            c_f32.cu_deviceptr()
        }
        OutElement::Bf16 => {
            c_bf16 = DeviceBuffer::<u16>::zeroed(stream, elements)?;
            c_bf16.cu_deviceptr()
        }
    };
    let ms = time_gpu_iters(stream, WARMUP, ITERS, || unsafe {
        baseline.launch_devptrs(stream, a.cu_deviceptr(), b.cu_deviceptr(), c)
    })?;
    Ok(Some((ms, baseline.algorithm())))
}

#[cfg(not(feature = "cublas"))]
fn baseline(
    _stream: &Arc<CudaStream>,
    _a: &DeviceBuffer<u16>,
    _b: &DeviceBuffer<u16>,
    _case: &Case,
) -> Result<Option<(f64, String)>, Box<dyn std::error::Error>> {
    Ok(None)
}
