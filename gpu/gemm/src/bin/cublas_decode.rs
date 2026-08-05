//! One launch of each side at three shallow-K shapes, for a profiler to read
//! (oxide-train#80, anomaly 2).
//!
//! `model_shapes` says cuBLASLt runs `tile=23 cluster=3 stages=35` at every
//! shallow depth and reaches 1850–1970 TF/s with a tile of **half our area**.
//! #80 first read `cluster=3` as a 4-CTA cluster with a multicast `A`; the
//! 4-CTA kernel that reading licensed lost by two, so the reading is doubtful
//! and the library's structure is unexplained. Nothing in this repo can decode
//! it — it is a closed binary, and `cublasLtMatmulAlgoConfigGetAttribute`'s
//! `uint16_t`s are opaque indices — so the decode has to come from outside the
//! library.
//!
//! This binary exists to give a profiler exactly six kernels to look at instead
//! of `model_shapes`' several hundred: one cuBLASLt launch and one of ours at
//! each of three shapes, all after the allocations and the heuristic search,
//! with no warm-ups and no timing loop. Every launch is announced on stdout
//! first, so a profiler's kernel list lines up with the shapes by position.
//!
//! `FEATURES=cublas ./run.sh gemm cublas_decode`, and under Nsight Compute
//! `NCU=1 FEATURES=cublas ./run.sh gemm cublas_decode`. Nsight's counter
//! library does **not** initialise on the Modal container ("LibraryNotLoaded",
//! which is the driver's profiling support and not something an image can
//! install), so what actually decoded #80's anomaly 2 is
//! `baselines/cublas_decode.cu` — the same three shapes with CUPTI's callback
//! API on them, which reads a launch configuration without asking for a single
//! performance counter.

use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use gemm::{Tcgen05Gemm, TmaLayout, create_bf16_tma_map, tcgen05_launch_config};
use half::bf16;

/// Shapes whose ratio sits under 0.90 and whose `K` is shallow — the regime the
/// whole anomaly lives in. `lm_head fwd` is here because it is the one row #80
/// measured cuBLASLt saturating the fill path on, so its counters are the
/// calibration for the other two.
struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "qkv fwd     24576x3072x9216",
        m: 24576,
        k: 3072,
        n: 9216,
    },
    Shape {
        name: "gate_up fwd  6144x3072x8192",
        m: 6144,
        k: 3072,
        n: 8192,
    },
    Shape {
        name: "lm_head fwd 24576x3072x50432",
        m: 24576,
        k: 3072,
        n: 50432,
    },
];

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

#[cfg(not(feature = "cublas"))]
fn main() {
    println!("cublas_decode needs `--features cublas`: it exists to launch the library");
}

#[cfg(feature = "cublas")]
fn main() -> Result<(), Box<dyn Error>> {
    use gemm::cublaslt::{Baseline, Form, OutElement};

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = Tcgen05Gemm::load(&context)?;
    println!("{}", gemm::cublaslt::about());
    println!("one launch a side, three shapes, in this order:");

    for shape in SHAPES {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let a = DeviceBuffer::from_host(&stream, &operand(m * k, 13))?;
        let b = DeviceBuffer::from_host(&stream, &operand(n * k, 14))?;
        let c = DeviceBuffer::<u16>::zeroed(&stream, m * n)?;

        let baseline = Baseline::with_form(&stream, m, n, k, Form::Store, OutElement::Bf16)?;
        println!("  {}  cuBLASLt  {}", shape.name, baseline.algorithm());
        // SAFETY: the three buffers are the packed bf16 shapes this baseline
        // was configured for.
        unsafe { baseline.launch(&stream, &a, &b, &c) }?;
        stream.synchronize()?;

        let a_tma = create_bf16_tma_map(&stream, &a, k, m, TmaLayout::KMajor)?;
        let b_tma = create_bf16_tma_map(&stream, &b, k, n, TmaLayout::KMajor)?;
        let mut ours = DeviceBuffer::<u32>::zeroed(&stream, m * n / 2)?;
        let config = tcgen05_launch_config(m, n, k);
        println!(
            "  {}  ours      {} clusters, {} B plan",
            shape.name,
            config.grid_dim.0 / 2,
            gemm::SHARED_BYTES
        );
        // SAFETY: the maps describe the live operands at this shape and `ours`
        // holds `m * n / 2` packed pairs.
        unsafe {
            module.store(
                &stream,
                config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut ours,
                n as u32,
                k as u32,
            )
        }?;
        stream.synchronize()?;
    }
    Ok(())
}
