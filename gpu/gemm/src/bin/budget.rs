//! Where a shallow-K item's time goes, and what cuBLASLt's does (#80 forensics).
//!
//! `model_shapes` reports ratios; this reports a **budget**. At a fixed output
//! geometry — so the tile count, the wave quantization and the
//! `pipeline::grouped` walk are all held still — `K` is swept and both sides
//! are timed. Per output tile the model is `t = fixed + per_block · K / 64`,
//! so a least-squares fit over the sweep separates what an item pays **once**
//! from what it pays **per K block**. Run for ours and for cuBLASLt at the same
//! points, that is the whole shallow-vs-deep question in two numbers a side: if
//! the intercepts differ and the slopes do not, the item's fixed cost is the
//! story; if the slopes differ, the feed is.
//!
//! The companion instrument — this kernel with a `clock64` on each phase of an
//! item, which is what attributed the intercept to the item boundary and the
//! accumulator handoff rather than to the pipeline fill or the feed — is on
//! branch `80-phase-probe`. A stopwatch does not belong in the shipped module,
//! and the fit above needs none.
//!
//! `FEATURES=cublas ./run.sh gemm budget`. Without the feature the sweep
//! reports the kernel alone.

use std::error::Error;
use std::sync::Arc;

use bench_util::time_gpu_iters;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gemm::{Tcgen05Gemm, TmaLayout, create_bf16_tma_map, tcgen05_launch_config};
use half::bf16;

const WARMUP: usize = 5;
const ITERS: usize = 20;

/// One output geometry, swept in K. `tiles` and the wave efficiency are
/// properties of `(m, n)` alone, so every point of a sweep shares them.
struct Geometry {
    name: &'static str,
    m: usize,
    n: usize,
    depths: &'static [usize],
}

const GEOMETRIES: &[Geometry] = &[
    Geometry {
        name: "qkv fwd    24576 x K x 9216",
        m: 24576,
        n: 9216,
        depths: &[1024, 2048, 3072, 4096, 6144, 9216, 12288],
    },
    Geometry {
        name: "gate_up fwd 6144 x K x 8192",
        m: 6144,
        n: 8192,
        depths: &[1024, 2048, 3072, 4096, 6144, 8192, 12288],
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

/// Least squares of `y = intercept + slope * x` — the only fit in this binary,
/// and it is two sums.
fn fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let covariance: f64 = points
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let variance: f64 = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
    let slope = covariance / variance;
    (mean_y - slope * mean_x, slope)
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = Tcgen05Gemm::load(&context)?;

    println!(
        "#80 forensics: per-item budget at fixed geometry, {} clusters",
        gemm::MAX_CLUSTERS
    );
    #[cfg(feature = "cublas")]
    println!("  denominator: {}", gemm::cublaslt::about());
    #[cfg(not(feature = "cublas"))]
    println!("  no denominator: built without `--features cublas`");
    println!();

    for geometry in GEOMETRIES {
        sweep(&stream, &module, geometry)?;
        println!();
    }

    Ok(())
}

/// Time both sides over `geometry.depths` and fit each one's per-item cost.
fn sweep(
    stream: &Arc<CudaStream>,
    module: &Tcgen05Gemm,
    geometry: &Geometry,
) -> Result<(), Box<dyn Error>> {
    let (m, n) = (geometry.m, geometry.n);
    let tiles = (m / 256) * (n / 256);
    let waves = tiles as f64 / gemm::MAX_CLUSTERS as f64;
    println!(
        "{}  {tiles} tiles = {waves:.2} waves over {} clusters (last-wave efficiency {:.1}%)",
        geometry.name,
        gemm::MAX_CLUSTERS,
        100.0 * waves / waves.ceil()
    );
    println!(
        "{:>7} {:>7} {:>10} {:>10} {:>8} {:>11} {:>11}",
        "K", "blocks", "ours ms", "theirs ms", "ratio", "ours us/tile", "theirs us/tile"
    );

    let mut ours_points = Vec::new();
    let mut theirs_points = Vec::new();
    // Per *cluster* items, since a cluster is what serializes them.
    let per_item = |ms: f64| ms * 1_000.0 / (tiles as f64 / gemm::MAX_CLUSTERS as f64);

    for &k in geometry.depths {
        let a = DeviceBuffer::from_host(stream, &operand(m * k, 13))?;
        let b = DeviceBuffer::from_host(stream, &operand(n * k, 14))?;
        let a_tma = create_bf16_tma_map(stream, &a, k, m, TmaLayout::KMajor)?;
        let b_tma = create_bf16_tma_map(stream, &b, k, n, TmaLayout::KMajor)?;
        let config = tcgen05_launch_config(m, n, k);
        let mut c = DeviceBuffer::<f32>::zeroed(stream, m * n)?;
        let ours = time_gpu_iters(stream, WARMUP, ITERS, || {
            unsafe {
                module.f32_store(
                    stream,
                    config,
                    a_tma.as_ptr(),
                    b_tma.as_ptr(),
                    &mut c,
                    n as u32,
                    k as u32,
                )
            }
            .map_err(Into::into)
        })?;
        let theirs = baseline(stream, &a, &b, m, n, k)?;
        let blocks = (k / 64) as f64;
        ours_points.push((blocks, per_item(ours)));
        match theirs {
            Some((ms, ref algorithm)) => {
                theirs_points.push((blocks, per_item(ms)));
                println!(
                    "{k:>7} {:>7} {ours:>10.4} {ms:>10.4} {:>8.3} {:>11.3} {:>11.3}   {algorithm}",
                    blocks as usize,
                    ms / ours,
                    per_item(ours),
                    per_item(ms)
                );
            }
            None => println!(
                "{k:>7} {:>7} {ours:>10.4} {:>10} {:>8} {:>11.3} {:>11}",
                blocks as usize,
                "-",
                "-",
                per_item(ours),
                "-"
            ),
        }
    }

    let (fixed, block) = fit(&ours_points);
    println!(
        "  ours     per item: {fixed:7.3} us fixed + {block:6.4} us per K block  \
         (K=3072 -> {:.3} us, fixed is {:.1}%)",
        fixed + block * 48.0,
        100.0 * fixed / (fixed + block * 48.0)
    );
    if !theirs_points.is_empty() {
        let (fixed_theirs, block_theirs) = fit(&theirs_points);
        println!(
            "  cuBLASLt per item: {fixed_theirs:7.3} us fixed + {block_theirs:6.4} us per K block  \
             (K=3072 -> {:.3} us, fixed is {:.1}%)",
            fixed_theirs + block_theirs * 48.0,
            100.0 * fixed_theirs / (fixed_theirs + block_theirs * 48.0)
        );
        println!(
            "  delta            : {:+7.3} us fixed + {:+6.4} us per K block",
            fixed - fixed_theirs,
            block - block_theirs
        );
    }
    Ok(())
}

/// cuBLASLt's time and chosen algorithm at this shape, fp32 out, store form.
#[cfg(feature = "cublas")]
fn baseline(
    stream: &Arc<CudaStream>,
    a: &DeviceBuffer<u16>,
    b: &DeviceBuffer<u16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Option<(f64, String)>, Box<dyn Error>> {
    use gemm::cublaslt::{Baseline, Form, OutElement};
    let baseline = Baseline::with_form(stream, m, n, k, Form::Store, OutElement::F32)?;
    let c = DeviceBuffer::<f32>::zeroed(stream, m * n)?;
    let ms = time_gpu_iters(stream, WARMUP, ITERS, || unsafe {
        baseline.launch_devptrs(stream, a.cu_deviceptr(), b.cu_deviceptr(), c.cu_deviceptr())
    })?;
    Ok(Some((ms, baseline.algorithm())))
}

#[cfg(not(feature = "cublas"))]
fn baseline(
    _stream: &Arc<CudaStream>,
    _a: &DeviceBuffer<u16>,
    _b: &DeviceBuffer<u16>,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<Option<(f64, String)>, Box<dyn Error>> {
    Ok(None)
}
