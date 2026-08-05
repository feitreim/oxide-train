//! CPU/GPU parity for all GEMM ladder rungs and epilogues.
//!
//! Run on B200 with `./run.sh gemm`.

use bench_util::{KernelBudget, enforce_kernel_budgets, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer};
use gemm::{
    Tcgen05Gemm, TmaLayout, create_bf16_tma_map, fp32, fp32_launch_config, tcgen05_launch_config,
};
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

fn matmul_transposed_a(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0; k * n];
    for row in 0..k {
        for column in 0..n {
            let mut sum = 0.0f64;
            for inner in 0..m {
                sum += a[inner * k + row] as f64 * b[inner * n + column] as f64;
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

/// ptxas allocation pins for the production kernels (SPEC §13, 7e17): hard
/// ceiling on regression, ratchet hint on improvement. Re-pinned at the ferro
/// `c648c67` bump's gated measurement on a B200, after ferro #180 unrolled the
/// mover walks and the `.local` frames they homed disappeared.
///
/// They went **up** from the 48 the exact-cover kernel held, and that is the
/// rewrite working rather than regressing. That kernel's four epilogue warps
/// read TMEM one `[16, 16]` fragment at a time — an `.x1` LDTM whose registers
/// are the load's return value, so the compiler could fuse each fragment
/// straight through to its store and never hold a band. The `.x8` drain lands
/// 32 fp32 at once and keeps four blocks live, which is what ferro-kittens #117
/// measured at **+23.1% / +8.8% / +5.1%** and priced at +52 registers. The
/// ceiling is 255: two CTAs of 128 threads admit the whole file, so the
/// headroom is what matters and there is a great deal of it.
///
/// The fp32 store entry point is higher because its drain has no staging tile
/// to hand values off to (GAPS.md §2.6): its `store_rows` walk holds the band.
/// It no longer accumulates — the fold moved to `gemm_tcgen05_f32_accumulate`,
/// whose drain never reads `C` at all (the reduction store, ferro #42 /
/// oxide-train#80 remedy 1): each `[16, 64]` half-band is scattered into a
/// staging tile and added into `C` by the copy engine, so it holds half the
/// band the store path holds and no `C` values beside it.
///
/// `max_spill_bytes` here is the **local-memory frame**, not spill stores —
/// `bench_util` reads `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`. The bf16 frame is
/// zero since ferro #180: the 256 B it used to carry was the `Job`'s state
/// homed behind a rolled mover walk, and unrolling the walk bought it back for
/// two registers. The fp32 store entry point's 512 B frame (ferro #174's "band
/// plus `C` beside it") went to zero when the fold left with the accumulate
/// split, so every shipped GEMM kernel now carries no depot at all.
// The deferred-drain epilogue (oxide-train#80 remedy 2) holds registers past
// its last `tcgen05.ld` — that hold is the overlap, since the accumulator is
// released to the next item's MMA while the held bands' stores still issue —
// so every ceiling moved up: bf16 82 → 137 and the reduce 80 → 127 (each a
// one-pass hoist), the fp32 store 102 → 118 (no hoist; its two-band form
// measured 181, past the ~170 the register file grants 12 warps an SM, and
// paid the 2 → 1 CTA cliff — model_shapes f32 rows at 0.46–0.53). A priced
// trade, not a leak: at these counts residency stays tensor-memory-bound at
// 2 CTAs/SM.
const KERNEL_BUDGETS: [KernelBudget; 3] = [
    KernelBudget {
        name: "gemm_tcgen05_bf16_optimized",
        max_registers: 133,
        max_spill_bytes: 0,
    },
    KernelBudget {
        name: "gemm_tcgen05_f32_optimized",
        max_registers: 109,
        max_spill_bytes: 0,
    },
    KernelBudget {
        name: "gemm_tcgen05_f32_accumulate",
        max_registers: 128,
        max_spill_bytes: 0,
    },
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let fp32_module = fp32::kernels::load(&context)?;
    let module = Tcgen05Gemm::load(&context)?;
    enforce_kernel_budgets(&module.kernels(), &KERNEL_BUDGETS)?;

    check_fp32(&stream, &fp32_module)?;
    check_tcgen05_bf16(&stream, &module)?;
    check_tcgen05_bf16_transposed(&stream, &module)?;
    check_tcgen05_many_items(&stream, &module)?;
    println!("✓ fp32 and tcgen05 bf16 GEMM store/accumulate parity passed");
    Ok(())
}

/// Does a **cluster taking a second work item** still compute the right `C`?
///
/// Every failure mode the persistent grid has that the exact-cover kernel did
/// not is at that boundary, and no shape above reaches it: the two 256³ gates
/// launch one cluster with one item each. What has to survive a second item is
/// the whole barrier lifecycle — `inval` then `init` per item, the pair of
/// cluster boundaries around `work`, the accumulator starting fresh on the new
/// item's first K block rather than adding to the last item's — and every one
/// of those is silent when it is wrong, in the sense that the phase simply
/// never completes or a tile comes back doubled.
///
/// Both shapes are past [`gemm::MAX_CLUSTERS`] tiles so that some clusters take
/// two. `4096x2560` divides the [`gemm::GROUP`] traversal width exactly;
/// `2816x3584` does not — 11 tile-rows against a group of 8 — which is what
/// executes `pipeline::grouped`'s short last group, whose failure is an
/// *aliased* tile rather than a missing one.
///
/// The operands make the whole of `C` a 7x21 table: `A`'s values depend on the
/// row only through `row % 7` and `B`'s on the column only through
/// `column % 21`, so the host reference is `7 * 21 * K` products instead of
/// `M * N * K`, and a 10-megaelement `C` is checkable at all. They are small
/// integers, so every product and every partial sum is exact in fp32 whatever
/// order the hardware reduces in — which is what lets the comparison be `==` on
/// the bf16 words, with no tolerance to hide a misplaced tile behind.
fn check_tcgen05_many_items(
    stream: &cuda_core::CudaStream,
    module: &Tcgen05Gemm,
) -> Result<(), Box<dyn std::error::Error>> {
    const K: usize = 256;
    const SHAPES: [(usize, usize); 2] = [(4096, 2560), (2816, 3584)];

    for (m, n) in SHAPES {
        let a_bits = periodic_operand(m, K, a_value);
        let b_bits = periodic_operand(n, K, b_value);
        let device_a = DeviceBuffer::from_host(stream, &a_bits)?;
        let device_b = DeviceBuffer::from_host(stream, &b_bits)?;
        let a_tma = create_bf16_tma_map(stream, &device_a, K, m, TmaLayout::KMajor)?;
        let b_tma = create_bf16_tma_map(stream, &device_b, K, n, TmaLayout::KMajor)?;
        let config = tcgen05_launch_config(m, n, K);

        let mut store = DeviceBuffer::<u32>::zeroed(stream, m * n / 2)?;
        unsafe {
            module.store(
                stream,
                config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut store,
                n as u32,
                K as u32,
            )
        }?;
        compare_periodic(
            &format!("tcgen05 bf16 store {m}x{n}x{K}"),
            &unpack_bf16(&store.to_host_vec(stream)?),
            m,
            n,
            K,
        )?;

        // The same tiles again, folded into a `C` that already holds them: an
        // item boundary that leaked would double one tile and not another.
        unsafe {
            module.accumulate(
                stream,
                config,
                a_tma.as_ptr(),
                b_tma.as_ptr(),
                &mut store,
                n as u32,
                K as u32,
            )
        }?;
        let folded = unpack_bf16(&store.to_host_vec(stream)?);
        let halved: Vec<f32> = folded.iter().map(|value| value / 2.0).collect();
        compare_periodic(
            &format!("tcgen05 bf16 accumulate {m}x{n}x{K}"),
            &halved,
            m,
            n,
            K,
        )?;
    }
    Ok(())
}

/// `A[row, depth]`: integers exact in bf16, periodic in `row` with period 7.
fn a_value(row: usize, depth: usize) -> f32 {
    ((row * 5 + depth * 3) % 7) as f32 - 3.0
}

/// `B[column, depth]`: period 21, which shares a factor of 7 with `A`'s so that
/// no K walk can make two different `(row, column)` cells collide by accident.
fn b_value(column: usize, depth: usize) -> f32 {
    ((column * 4 + depth * 5) % 21) as f32 - 10.0
}

/// A `[lines, depth]` K-major bf16 operand, as the `u16` words a device buffer
/// holds.
fn periodic_operand(lines: usize, depth: usize, value: impl Fn(usize, usize) -> f32) -> Vec<u16> {
    let mut staged = Vec::with_capacity(lines * depth);
    for line in 0..lines {
        for step in 0..depth {
            staged.push(bf16::from_f32(value(line, step)).to_bits());
        }
    }
    staged
}

/// Compare an observed `C` against the 7x21 table the operands make it.
///
/// `==` on the bf16 words: the reference carries the same single rounding the
/// kernel does, from an fp32 sum that is exact on both sides, so a tolerance
/// would only be somewhere for a wrong tile to hide.
fn compare_periodic(
    name: &str,
    observed: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let exact: Vec<f32> = (0..7 * 21)
        .map(|cell| {
            (0..k)
                .map(|depth| a_value(cell / 21, depth) * b_value(cell % 21, depth))
                .sum()
        })
        .collect();
    let reference: Vec<f32> = exact
        .iter()
        .map(|&value| bf16::from_f32(value).to_f32())
        .collect();
    let mut wrong = 0usize;
    let mut sample = Vec::new();
    for row in 0..m {
        for column in 0..n {
            let want = reference[(row % 7) * 21 + column % 21];
            let got = observed[row * n + column];
            if got != want {
                wrong += 1;
                if sample.len() < 8 {
                    sample.push(format!("C[{row}, {column}] = {got}, want {want}"));
                }
            }
        }
    }
    if wrong > 0 {
        return Err(format!(
            "{name}: {wrong} of {} elements wrong: {}",
            m * n,
            sample.join("; ")
        )
        .into());
    }
    Ok(())
}

/// Weight-gradient orientation (#53): `dW += Aᵀ·B` with both operands read
/// MN-major out of their native `[K, M]` / `[K, N]` row-major panels through
/// the descriptor's `transpose_a`/`transpose_b` bits — no transposed staging
/// buffers anywhere. The operand geometry is the one
/// `src/bin/transpose_probe.rs` pinned down on a single tile; this gate is the
/// same geometry inside the real M256xN256 cta_group::2 pair-UMMA pipeline.
fn check_tcgen05_bf16_transposed(
    stream: &cuda_core::CudaStream,
    module: &Tcgen05Gemm,
) -> Result<(), Box<dyn std::error::Error>> {
    const M: usize = 256;
    const N: usize = 256;
    const K: usize = 256;
    // Native panels: `a[k, m]` and `b[k, n]`, exactly how a backward pass has
    // its activations and output gradients lying in memory.
    let (a_bits, a) = quantize_bf16(&uniform_vec(K * M, 7));
    let (b_bits, b) = quantize_bf16(&uniform_vec(K * N, 8));
    let (_, initial) = quantize_bf16(&uniform_vec(M * N, 9));

    let mut expected = initial.clone();
    for row in 0..M {
        for column in 0..N {
            let mut sum = 0.0f64;
            for inner in 0..K {
                sum += a[inner * M + row] as f64 * b[inner * N + column] as f64;
            }
            expected[row * N + column] += sum as f32;
        }
    }

    let device_a = DeviceBuffer::from_host(stream, &a_bits)?;
    let device_b = DeviceBuffer::from_host(stream, &b_bits)?;
    let a_tma = create_bf16_tma_map(stream, &device_a, M, K, TmaLayout::MnMajor)?;
    let b_tma = create_bf16_tma_map(stream, &device_b, N, K, TmaLayout::MnMajor)?;
    let mut accumulate = DeviceBuffer::from_host(stream, &initial)?;
    unsafe {
        module.f32_accumulate_transposed(
            stream,
            tcgen05_launch_config(M, N, K),
            a_tma.as_ptr(),
            b_tma.as_ptr(),
            &mut accumulate,
            N as u32,
            K as u32,
        )
    }?;
    assert_close(
        "tcgen05 bf16 transposed f32 accumulate",
        &accumulate.to_host_vec(stream)?,
        &expected,
        0.03,
        0.01,
    );
    Ok(())
}

fn check_fp32(
    stream: &cuda_core::CudaStream,
    module: &fp32::kernels::LoadedModule,
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
        module.register_gemm_store(
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
        module.register_gemm_accumulate(
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

    let mut b_transposed = vec![0.0; N * K];
    for column in 0..N {
        for row in 0..K {
            b_transposed[column * K + row] = b[row * N + column];
        }
    }
    let device_b_transposed = DeviceBuffer::from_host(stream, &b_transposed)?;
    let mut nt_store = DeviceBuffer::<f32>::zeroed(stream, M * N)?;
    unsafe {
        module.register_gemm_nt_store(
            stream,
            fp32_launch_config(M, N),
            M,
            N,
            K,
            &device_a,
            &device_b_transposed,
            &mut nt_store,
        )
    }?;
    assert_close(
        "fp32 nt store",
        &nt_store.to_host_vec(stream)?,
        &matmul_transposed_b(&a, &b_transposed, M, N, K),
        2e-5,
        2e-5,
    );

    let tn_b = uniform_vec(M * N, 4);
    let tn_initial = uniform_vec(K * N, 5);
    let tn_product = matmul_transposed_a(&a, &tn_b, M, N, K);
    let tn_expected: Vec<f32> = tn_initial
        .iter()
        .zip(&tn_product)
        .map(|(initial, product)| initial + product)
        .collect();
    let device_tn_b = DeviceBuffer::from_host(stream, &tn_b)?;
    let mut tn_accumulate = DeviceBuffer::from_host(stream, &tn_initial)?;
    unsafe {
        module.register_gemm_tn_accumulate(
            stream,
            fp32_launch_config(K, N),
            K,
            N,
            M,
            &device_a,
            &device_tn_b,
            &mut tn_accumulate,
        )
    }?;
    assert_close(
        "fp32 tn accumulate",
        &tn_accumulate.to_host_vec(stream)?,
        &tn_expected,
        2e-5,
        2e-5,
    );
    Ok(())
}

fn check_tcgen05_bf16(
    stream: &cuda_core::CudaStream,
    module: &Tcgen05Gemm,
) -> Result<(), Box<dyn std::error::Error>> {
    // One full four-stage K pipeline cycle exercises every TMA/MMA stage.
    const M: usize = 256;
    const N: usize = 256;
    const K: usize = 256;
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
    let a_tma = create_bf16_tma_map(stream, &device_a, K, M, TmaLayout::KMajor)?;
    let b_tma = create_bf16_tma_map(stream, &device_b, K, N, TmaLayout::KMajor)?;
    let config = tcgen05_launch_config(M, N, K);

    let mut store = DeviceBuffer::<u32>::zeroed(stream, M * N / 2)?;
    unsafe {
        module.store(
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
        module.accumulate(
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

    let mut f32_store = DeviceBuffer::<f32>::zeroed(stream, M * N)?;
    unsafe {
        module.f32_store(
            stream,
            config,
            a_tma.as_ptr(),
            b_tma.as_ptr(),
            &mut f32_store,
            N as u32,
            K as u32,
        )
    }?;
    assert_close(
        "tcgen05 bf16 f32 store",
        &f32_store.to_host_vec(stream)?,
        &expected,
        0.03,
        0.01,
    );

    let mut f32_accumulate = DeviceBuffer::from_host(stream, &initial)?;
    unsafe {
        module.f32_accumulate(
            stream,
            config,
            a_tma.as_ptr(),
            b_tma.as_ptr(),
            &mut f32_accumulate,
            N as u32,
            K as u32,
        )
    }?;
    assert_close(
        "tcgen05 bf16 f32 accumulate",
        &f32_accumulate.to_host_vec(stream)?,
        &expected_accumulate,
        0.03,
        0.01,
    );
    Ok(())
}
