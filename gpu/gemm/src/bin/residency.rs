//! Does the persistent grid actually fit on the device? (oxide-train#80)
//!
//! `MAX_CLUSTERS = 148` launches 296 CTAs and the kernel's whole wave
//! arithmetic — the `grouped` walk's reuse, every "last-wave efficiency" figure
//! on #80 — assumes they run at once, two an SM. `cuOccupancyMaxActiveClusters`
//! says **74**. If the driver is right, main has been running two sequential
//! waves of its single persistent wave since it was written, and #80's
//! multicast negative (4-CTA clusters at 33 resident) was comparing a half-wide
//! machine against a full one without knowing it.
//!
//! Three instruments, in the order that decides the question:
//!
//! 1. **The model.** Every occupancy query the driver will answer, swept: the
//!    non-cluster `cuOccupancyMaxActiveBlocksPerMultiprocessor` (which knows
//!    nothing about clusters, so it isolates the *per-CTA* resource clip),
//!    `cuOccupancyMaxActiveClusters` at each cluster width, and both against a
//!    falling shared-memory plan so a number that does not move is a shape the
//!    device will not pack rather than a budget this kernel overspent.
//!
//! 2. **The measurement.** [`gemm::residency_probe`] on the shipped launch
//!    shape: every CTA holds its SM for a fixed wall-clock interval and reports
//!    `%smid` and the interval. A grid the device runs in one wave takes the
//!    hold; one it runs in two takes twice the hold. The packing and the true
//!    peak overlap come out of the same launch.
//!
//! 3. **The shipped kernel.** The probe cannot carry 168 registers, so the last
//!    word is the real kernel timed across grid widths. The kernel reads its
//!    own grid, so any width is correct and only the items per cluster change:
//!    if 148 clusters run in one wave, 74 clusters must take about twice as
//!    long; if 148 was already two waves, 74 costs nothing.
//!
//! `./run.sh gemm residency` — no `cublas` feature needed, since every number
//! here is ours against ours.

use std::error::Error;
use std::sync::Arc;

use bench_util::{function_profile, time_gpu_iters};
use cuda_core::{CudaContext, CudaFunction, CudaStream, DeviceBuffer};
use gemm::optimized::kernels;
use gemm::residency_probe::SLOTS;
use gemm::{Tcgen05Gemm, TmaLayout, create_bf16_tma_map, tcgen05_launch_config_over};
use half::bf16;

const WARMUP: usize = 3;
const ITERS: usize = 10;

/// Wall-clock nanoseconds each probe CTA holds its SM. Long enough that launch
/// skew (a few µs across 296 CTAs) cannot fake an extra wave, short enough that
/// a sweep is free.
const HOLD_NS: u64 = 2_000_000;

/// Tensor-memory columns a probe CTA takes, which is the accumulator's
/// `BLOCK_N` — and the resource `MAX_CLUSTERS` is derived from.
const COLUMNS: u32 = 256;

/// The floor of every shared-memory sweep. Not zero: the probe attaches a
/// [`kittens::plan::SharedPlan`] for its tensor-memory staging word, and a
/// launch that declares no dynamic shared memory at all gives that word an
/// address outside the CTA's window — which a `cta_group::2` allocation, whose
/// write crosses to the peer, reports as a **CGA out-of-range address** and not
/// as a null dereference.
const MINIMUM_BYTES: u32 = 1_024;

/// The shapes the grid-width sweep runs, chosen for tile counts that are whole
/// multiples of neither candidate wave width, so a wave-quantization artifact
/// cannot be mistaken for a residency one.
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
        name: "bwd qkv dx  24576x9216x3072",
        m: 24576,
        k: 9216,
        n: 3072,
    },
    Shape {
        name: "square            8192x8192",
        m: 8192,
        k: 8192,
        n: 8192,
    },
];

/// Grid widths swept, in clusters. `MAX_CLUSTERS` and its half are the two
/// hypotheses; the quarter and the three-quarter point say whether the curve
/// between them is a step or a slope.
const WIDTHS: &[u32] = &[148, 111, 74, 37];

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

/// `host.rs`'s private helper, which the probe entry points need too.
fn opt_in_dynamic_smem(function: &CudaFunction, bytes: u32) -> Result<(), Box<dyn Error>> {
    use cuda_core::sys::{
        CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        cuFuncSetAttribute, cudaError_enum_CUDA_SUCCESS,
    };
    // SAFETY: `function` is a live entry point of a loaded module.
    let status = unsafe {
        cuFuncSetAttribute(
            function.cu_function(),
            CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )
    };
    if status != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuFuncSetAttribute(dynamic smem {bytes}) failed: {status:?}").into());
    }
    Ok(())
}

/// The percentage of an SM's L1/shared memory the driver configures as shared
/// for a function. `-1` is "the driver decides", and what it decides for a
/// kernel that opted into a large plan is the question.
const CARVEOUT: u32 =
    cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;

fn function_attribute(function: &CudaFunction, attribute: u32) -> Result<i32, Box<dyn Error>> {
    use cuda_core::sys::{cuFuncGetAttribute, cudaError_enum_CUDA_SUCCESS};
    let mut value = 0i32;
    // SAFETY: `function` is a live entry point of a loaded module.
    let status = unsafe { cuFuncGetAttribute(&mut value, attribute, function.cu_function()) };
    if status != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuFuncGetAttribute({attribute}) failed: {status:?}").into());
    }
    Ok(value)
}

fn set_function_attribute(
    function: &CudaFunction,
    attribute: u32,
    value: i32,
) -> Result<(), Box<dyn Error>> {
    use cuda_core::sys::{cuFuncSetAttribute, cudaError_enum_CUDA_SUCCESS};
    // SAFETY: `function` is a live entry point of a loaded module.
    let status = unsafe { cuFuncSetAttribute(function.cu_function(), attribute, value) };
    if status != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuFuncSetAttribute({attribute}, {value}) failed: {status:?}").into());
    }
    Ok(())
}

fn device_attribute(context: &CudaContext, attribute: u32) -> Result<i32, Box<dyn Error>> {
    use cuda_core::sys::{cuDeviceGetAttribute, cudaError_enum_CUDA_SUCCESS};
    let mut value = 0i32;
    // SAFETY: a pure query against this context's own device handle.
    let status = unsafe { cuDeviceGetAttribute(&mut value, attribute, context.cu_device()) };
    if status != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuDeviceGetAttribute({attribute}) failed: {status:?}").into());
    }
    Ok(value)
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = Tcgen05Gemm::load(&context)?;
    let probe = kernels::load(&context)?;
    for name in ["residency_probe_cg2", "residency_probe_cg4"] {
        let function = probe.as_cuda_module().load_function(name)?;
        opt_in_dynamic_smem(&function, gemm::SHARED_BYTES as u32)?;
    }

    println!("#80: what the persistent grid actually gets");
    println!("  {}", context.device_name()?);
    device_facts(&context)?;
    println!();
    the_model(&context, &module, &probe)?;
    println!();
    the_measurement(&stream, &probe)?;
    println!();
    // `the_model` left every kernel asking for the whole carveout. If that is
    // what was clipping residency, the shipped kernel is faster with it than
    // without, at the same source — so both settings are timed, default first.
    for carveout in [-1, 100] {
        for &(_, function) in module.kernels().iter() {
            set_function_attribute(function, CARVEOUT, carveout)?;
        }
        println!("carveout = {carveout}");
        the_shipped_kernel(&stream, &module)?;
    }
    Ok(())
}

/// The per-SM numbers `MAX_CLUSTERS`'s derivation is an arithmetic over, read
/// off the device rather than assumed.
fn device_facts(context: &CudaContext) -> Result<(), Box<dyn Error>> {
    use cuda_core::sys::{
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR as MAX_BLOCKS,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK as REGS_BLOCK,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR as REGS_SM,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN as SMEM_OPTIN,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR as SMEM_SM,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR as THREADS_SM,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as SM_COUNT,
    };
    for (name, attribute) in [
        ("SMs", SM_COUNT),
        ("shared memory / SM", SMEM_SM),
        ("shared memory / block (opt-in)", SMEM_OPTIN),
        ("registers / SM", REGS_SM),
        ("registers / block", REGS_BLOCK),
        ("threads / SM", THREADS_SM),
        ("blocks / SM", MAX_BLOCKS),
    ] {
        println!("  {name:<32} {}", device_attribute(context, attribute)?);
    }
    println!(
        "  {:<32} {} B, {} threads, {} tensor-memory columns a CTA",
        "this kernel plans",
        gemm::SHARED_BYTES,
        gemm::optimized::THREADS,
        COLUMNS
    );
    Ok(())
}

/// Instrument 1: every occupancy answer the driver will give, swept.
fn the_model(
    context: &CudaContext,
    module: &Tcgen05Gemm,
    probe: &kernels::LoadedModule,
) -> Result<(), Box<dyn Error>> {
    let sms = context.multiprocessor_count()?;
    let block = (gemm::optimized::THREADS, 1, 1);
    let plan = gemm::SHARED_BYTES as u32;

    println!("the model — what the driver says, per kernel");
    println!(
        "  {:<34} {:>4} {:>6} {:>10} {:>12} {:>12}",
        "kernel", "regs", "spill", "blocks/SM", "cg2 clusters", "cg4 clusters"
    );
    let cg2_grid = (2 * gemm::MAX_CLUSTERS, 1, 1);
    let cg4_grid = (4 * gemm::MAX_CLUSTERS, 1, 1);
    let named: Vec<(&str, &CudaFunction)> = module.kernels().to_vec();
    for &(name, function) in &named {
        let profile = function_profile(function)?;
        let blocks = function.max_active_blocks_per_multiprocessor(block.0, plan)?;
        let cg2 = function.max_active_clusters(cg2_grid, block, plan, (2, 1, 1))?;
        println!(
            "  {name:<34} {:>4} {:>6} {blocks:>10} {:>12} {:>12}",
            profile.registers,
            profile.spill_bytes,
            format!("{cg2} = {} CTAs", 2 * cg2),
            "-",
        );
    }
    for name in ["residency_probe_cg2", "residency_probe_cg4"] {
        let function = probe.as_cuda_module().load_function(name)?;
        let profile = function_profile(&function)?;
        let blocks = function.max_active_blocks_per_multiprocessor(block.0, plan)?;
        let (ranks, grid) = match name.ends_with("cg4") {
            true => (4, cg4_grid),
            false => (2, cg2_grid),
        };
        let clusters = function.max_active_clusters(grid, block, plan, (ranks, 1, 1))?;
        println!(
            "  {name:<34} {:>4} {:>6} {blocks:>10} {:>12} {:>12}",
            profile.registers,
            profile.spill_bytes,
            if ranks == 2 {
                format!("{clusters} = {} CTAs", 2 * clusters)
            } else {
                "-".into()
            },
            if ranks == 4 {
                format!("{clusters} = {} CTAs", 4 * clusters)
            } else {
                "-".into()
            },
        );
    }

    // Which resource clips it? The non-cluster query knows nothing about
    // clusters, so a `1` there is a **per-CTA** clip and the cluster shape is
    // not on trial at all. Sweeping the plan then says whether that clip is
    // shared memory, and — since the answer is a step function — exactly where
    // its edge is. A kernel is only ever one byte away from the wrong side of
    // it, and nothing in the source says where the byte is.
    let (name, function) = named[1];
    println!();
    println!("  {name} against a falling plan, {sms} SMs:");
    println!(
        "    {:>9}  {:>10}  {:>16}",
        "bytes", "blocks/SM", "cg2 clusters"
    );
    // Never above the opted-in maximum: the query rejects a plan the function
    // was never raised to, and an error there is not an occupancy answer.
    for bytes in [plan, 112_000, 96_000, 64_000, 32_000, MINIMUM_BYTES] {
        let blocks = function.max_active_blocks_per_multiprocessor(block.0, bytes)?;
        let cg2 = function.max_active_clusters(cg2_grid, block, bytes, (2, 1, 1))?;
        println!(
            "    {bytes:>9}  {blocks:>10}  {:>16}",
            format!("{cg2} = {} CTAs", 2 * cg2),
        );
    }

    // The edge itself, to the byte, for every kernel and for both queries. A
    // plan that ends up one allocation unit over the line costs half the
    // machine and reports nothing; this is the number `SHARED_BYTES` has to be
    // asserted against, in place of the 116 736 the source guesses.
    println!();
    println!("  the byte where two CTAs stop fitting:");
    for &(name, function) in &named {
        let blocks = |bytes: u32| {
            function
                .max_active_blocks_per_multiprocessor(block.0, bytes)
                .unwrap_or(0)
        };
        let clusters = |bytes: u32| {
            function
                .max_active_clusters(cg2_grid, block, bytes, (2, 1, 1))
                .unwrap_or(0)
        };
        let (mut fits, mut over) = (0u32, plan);
        while over - fits > 1 {
            let middle = fits + (over - fits) / 2;
            match blocks(middle) >= 2 {
                true => fits = middle,
                false => over = middle,
            }
        }
        println!(
            "    {name:<34} 2 CTAs up to {fits} B (plan is {plan}, {:+} B), \
             cg2 there {} clusters",
            fits as i64 - plan as i64,
            clusters(fits),
        );
    }

    // Does the cluster query even depend on the grid it is handed? If the
    // answer tracks `grid / cluster`, it is reporting the launch and not the
    // device.
    println!();
    println!("  {name} at cg2 against the grid it is asked about:");
    for clusters in [37u32, 74, 148, 296, 592] {
        let value = function.max_active_clusters((2 * clusters, 1, 1), block, plan, (2, 1, 1))?;
        println!("    grid {:>4} clusters -> {value:>4} resident", clusters);
    }

    // An SM's shared memory is not simply *there*: the L1/shared split is a
    // carveout the driver picks, and opting a kernel into 114 816 B only
    // obliges it to configure enough for **one** CTA. If that is the clip, the
    // fix is a host-side attribute rather than a byte of the plan — so the
    // preference is read, then raised, then the same two queries are asked
    // again. A number that moves here is the whole anomaly.
    println!();
    println!("  the L1/shared carveout, before and after asking for all of it:");
    for &(name, function) in &named {
        let carveout = function_attribute(function, CARVEOUT)?;
        let blocks = function.max_active_blocks_per_multiprocessor(block.0, plan)?;
        let clusters = function.max_active_clusters(cg2_grid, block, plan, (2, 1, 1))?;
        set_function_attribute(function, CARVEOUT, 100)?;
        println!(
            "    {name:<34} carveout {carveout:>4} -> {:>4}, blocks/SM {blocks} -> {}, \
             cg2 {clusters} -> {} clusters",
            function_attribute(function, CARVEOUT)?,
            function.max_active_blocks_per_multiprocessor(block.0, plan)?,
            function.max_active_clusters(cg2_grid, block, plan, (2, 1, 1))?,
        );
    }

    // The 4-CTA shape asked of the kernel compiled for it — the 2-CTA entry
    // points carry a required cluster dimension, and asking one of those about
    // a width it was not compiled for is `CUDA_ERROR_INVALID_CLUSTER_SIZE`
    // rather than an occupancy answer.
    let cg4 = probe
        .as_cuda_module()
        .load_function("residency_probe_cg4")?;
    println!();
    println!("  residency_probe_cg4 against a falling plan:");
    for bytes in [plan, 96_000, 64_000, 32_000, MINIMUM_BYTES] {
        let blocks = cg4.max_active_blocks_per_multiprocessor(block.0, bytes)?;
        let clusters = cg4.max_active_clusters(cg4_grid, block, bytes, (4, 1, 1))?;
        println!(
            "    {bytes:>9}  {blocks:>10} blocks/SM  {clusters:>4} clusters = {} CTAs",
            4 * clusters
        );
    }
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}

/// Instrument 2: hold every SM and see how many holds overlap.
fn the_measurement(
    stream: &Arc<CudaStream>,
    probe: &kernels::LoadedModule,
) -> Result<(), Box<dyn Error>> {
    println!(
        "the measurement — {:.2} ms of hold per CTA",
        HOLD_NS as f64 / 1e6
    );
    println!(
        "  {:<38} {:>6} {:>9} {:>7} {:>6} {:>7} {:>9}",
        "launch", "CTAs", "wall ms", "waves", "SMs", "max/SM", "overlap"
    );
    // The two cluster widths at the CTA counts each one's own arithmetic
    // claims, plus each one's half — a launch that fits shows one wave at the
    // full width, and a launch that does not shows two. The plan is swept with
    // them, because the probe needs four bytes of it and the rest is there
    // purely to reproduce what the shipped kernel reserves: if the shipped plan
    // runs two waves and a smaller one runs one, the plan *is* the residency.
    let plan = gemm::SHARED_BYTES as u32;
    for (ranks, clusters, bytes, columns) in [
        (2u32, 148u32, plan, COLUMNS),
        (2, 148, 112_000, COLUMNS),
        (2, 148, 96_000, COLUMNS),
        (2, 148, MINIMUM_BYTES, COLUMNS),
        (2, 74, plan, COLUMNS),
        // The four-CTA width, at its own arithmetic's 74 clusters and then
        // walked down. 74 x 4 = 296 CTAs is the same count the pair width
        // places two an SM across all 148, but a four-CTA cluster has to fit
        // inside a GPC, so the device reaches fewer SMs and stacks three CTAs
        // on some of them — and three CTAs cannot each hold 256 of an SM's 512
        // tensor-memory columns. The knee is where the third CTA stops
        // appearing, and #80's multicast kernel launched well above it.
        (4, 74, plan, COLUMNS),
        (4, 73, plan, COLUMNS),
        (4, 72, plan, COLUMNS),
        (4, 71, plan, COLUMNS),
        (4, 70, plan, COLUMNS),
        (4, 66, plan, COLUMNS),
        (4, 74, MINIMUM_BYTES, COLUMNS),
        // Half the columns, so three CTAs of an SM can each hold their own:
        // if the second wave is tensor memory rather than placement, it is
        // this arm that runs in one.
        (4, 74, plan, 128),
        (4, 33, plan, COLUMNS),
    ] {
        let ctas = ranks * clusters;
        let mut out = DeviceBuffer::<u64>::zeroed(stream, SLOTS * ctas as usize)?;
        let config = cuda_core::LaunchConfig {
            grid_dim: (ctas, 1, 1),
            block_dim: (gemm::optimized::THREADS, 1, 1),
            shared_mem_bytes: bytes,
        };
        let start = std::time::Instant::now();
        // SAFETY: `out` holds `SLOTS` u64 per CTA of this grid, the launch
        // declares the plan the probe attaches to, and `COLUMNS` is the
        // accumulator's own charge against an SM's 512.
        unsafe {
            match ranks {
                2 => probe.residency_probe_cg2(stream, config, &mut out, HOLD_NS, columns),
                _ => probe.residency_probe_cg4(stream, config, &mut out, HOLD_NS, columns),
            }
        }?;
        stream.synchronize()?;
        let wall = start.elapsed().as_secs_f64() * 1e3;
        let report = out.to_host_vec(stream)?;
        let (sms, max_per_sm, overlap) = analyze(&report);
        println!(
            "  {:<38} {ctas:>6} {wall:>9.3} {:>7.2} {sms:>6} {max_per_sm:>7} {overlap:>9}",
            format!("cg{ranks} x {clusters} cl, {bytes} B, {columns} col"),
            wall / (HOLD_NS as f64 / 1e6),
        );
        // A fault in a later arm aborts the process, and a pipe holds what a
        // terminal would have shown. Every row this instrument prints is a
        // result on its own, so none of them waits on the next one.
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!(
        "  waves ~1 and overlap = CTAs is one resident wave; waves ~2 and \
         overlap ~CTAs/2 is two."
    );
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}

/// Distinct SMs, the most CTAs any one of them hosted, and the largest number
/// of holds that were live at one instant.
///
/// The overlap is a sweep over interval endpoints rather than a pairwise test:
/// every CTA's entry is +1 and its exit is −1, and the running maximum over the
/// sorted events is the peak residency the launch actually reached.
fn analyze(report: &[u64]) -> (usize, usize, i32) {
    let ctas = report.len() / SLOTS;
    let mut per_sm = std::collections::HashMap::new();
    let mut events = Vec::with_capacity(2 * ctas);
    for cta in 0..ctas {
        let at = SLOTS * cta;
        *per_sm.entry(report[at]).or_insert(0usize) += 1;
        events.push((report[at + 2], 1i32));
        events.push((report[at + 3], -1i32));
    }
    events.sort_unstable();
    let (mut live, mut peak) = (0, 0);
    for (_, delta) in events {
        live += delta;
        peak = peak.max(live);
    }
    (
        per_sm.len(),
        per_sm.values().copied().max().unwrap_or(0),
        peak,
    )
}

/// Instrument 3: the shipped kernel across grid widths, at its real registers.
fn the_shipped_kernel(
    stream: &Arc<CudaStream>,
    module: &Tcgen05Gemm,
) -> Result<(), Box<dyn Error>> {
    println!("the shipped kernel — fp32 store timed across grid widths");
    println!(
        "  a width the device runs in one wave costs time proportionally when \
         halved; one it already ran in two costs nothing"
    );
    print!("  {:<30} {:>7}", "shape", "tiles");
    for width in WIDTHS {
        print!("{:>12}", format!("{width} cl"));
    }
    println!("{:>10}", "74/148");
    for shape in SHAPES {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let a = DeviceBuffer::from_host(stream, &operand(m * k, 13))?;
        let b = DeviceBuffer::from_host(stream, &operand(n * k, 14))?;
        let a_tma = create_bf16_tma_map(stream, &a, k, m, TmaLayout::KMajor)?;
        let b_tma = create_bf16_tma_map(stream, &b, k, n, TmaLayout::KMajor)?;
        let mut c = DeviceBuffer::<f32>::zeroed(stream, m * n)?;
        print!("  {:<30} {:>7}", shape.name, (m / 256) * (n / 256));
        let mut times = Vec::new();
        for &width in WIDTHS {
            let config = tcgen05_launch_config_over(m, n, k, width);
            let time = time_gpu_iters(stream, WARMUP, ITERS, || {
                // SAFETY: the maps describe live operands at this shape, `c`
                // holds `m * n`, and any whole number of clusters is a legal
                // grid for a kernel that reads its own width.
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
            print!("{time:>12.4}");
            times.push(time);
        }
        println!("{:>10.3}", times[2] / times[0]);
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    Ok(())
}
