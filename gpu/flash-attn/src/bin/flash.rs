//! Standalone tcgen05 attention parity and performance harness.
//!
//! Run on B200 with `./run.sh flash-attn flash`.

use std::sync::Arc;

use bench_util::{KernelBudget, enforce_kernel_budgets, time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

#[path = "../host.rs"]
mod host;
#[path = "../tcgen05.rs"]
#[allow(dead_code)]
mod tcgen05_device;

use tcgen05_device as tcgen05;

use host::{
    FLASH_HD, FLASH_QUERIES, FLASH_SUBTILE_HD, FLASH_TILE, Tcgen05Flash, correction_count_len,
    create_flash_head_tma_map, device_sm_count, flash_backward_kv_config, flash_backward_q_config,
    flash_forward_config,
};

/// Pinned ptxas ceilings (issue #61 phase 0): the B200-measured budgets from
/// SPEC 13.7e16 (forwards; 592 B is a fixed local frame, not a spill) and the
/// 2026-07-25 phase-1 gate run (backwards). Any tile-library port that pushes
/// a kernel past its pin is a library bug; ratchet a pin down when a run
/// reports lower numbers.
///
/// Phase-2/3 re-pins (2026-07-26): the kittens ports leave every kernel's
/// instruction stream equal-or-smaller (normalized PTX diff vs main; local
/// frames unchanged to the byte), but ptxas allocates by schedule, and the
/// counts moved with each formulation: persistent 206→238→239, pipelined
/// 224→219→179, sync 48→40. Occupancy is TMEM-pinned at one CTA/SM
/// throughout, and the bench holds or improves where it matters —
/// persistent 234.7 TFLOP/s / pipelined 187.6 (vs 230.3 / 182.0 at main);
/// the sync oracle dipped 126→115 with its 8-reg drop, tolerated because
/// it is a correctness reference, not a production path. The pins track
/// the measured allocation rather than fighting ordering sensitivity.
///
/// Phase-4 re-pins (2026-07-26): the pipeline.rs port (forward kernels
/// fully on kittens semaphores/streams, persistent scaffold extracted to
/// `pipeline::run`) lands at pipelined 180 / persistent 240 — +1 each with
/// opcode-identical-or-smaller PTX (persistent net −1 instruction), stable
/// across two formulations; a third formulation that reordered barrier
/// init after the TMEM alloc cost pipelined +3 more and was reverted, so
/// the init-before-alloc ordering is deliberate. Sync stays 40, backwards
/// untouched. No occupancy boundary anywhere near (TMEM pins 1 CTA/SM).
///
/// Phase-5 re-pins (2026-07-26): the backward/probe port (`Fragment` drains
/// and swizzled stores, `PhasedSemaphore`, `tma_load_at`, `alloc_block`)
/// ratchets backward q 62→52 and kv 64→60 — the first port in this series
/// where the allocator gave registers back. PTX is +9/+8 instructions, all
/// of it the fall-through `bra.uni` block splits `alloc_block`'s guard
/// introduces; against that the kv register pass lost five `or.pred`s
/// (looped causal masking shares the `masked_all` term eight hand-written
/// `keep_*` expressions each re-derived) and three `cvta.shared`s. Local
/// frames unchanged to the byte, forwards byte-identical, bench flat.
///
/// Forward re-pins for the unified P-store (2026-07-26): `softmax_tile`'s
/// pass-2 write moved onto `store_fragment_bf16`, making the library the only
/// fragment-store path in the crate. Statically the change is strictly
/// cleaner — local frames unchanged at 592 B, `ld/st.local` and the
/// FMA/mul/select histogram identical to the byte, 13–28 *fewer* instructions
/// per forward kernel (the fragment is built as a literal over an
/// `#[inline(always)]` `masked_exp2`; filling it through an indexed loop
/// instead does NOT scalarize, and puts 64 B of frame and 76 `st.local` into
/// the softmax inner loop — issue #61's risk 1, worth remembering). ptxas
/// re-rolls the schedule against it: sync 40→32, pipelined 180→216,
/// persistent 240→236, reproducible across runs. Occupancy is unaffected
/// (TMEM pins 1 CTA/SM; 216×128 threads ≈ 27.6K of the 64K register file).
///
/// `backward q pipelined` (2026-07-26) was the first kernel written
/// kittens-first rather than ported, and cost **one** register over the
/// synchronous kernel for a whole pipeline. #69 retired both: the two backward
/// kernels below are the only ones now, and their pins are re-set to what
/// ptxas gives a 128-thread `.maxntid` rather than a 192- or 1024-thread one.
const KERNEL_BUDGETS: [KernelBudget; 3] = [
    KernelBudget {
        // The unified forward (#68): 244 regs / 1136 B frame, against the
        // persistent kernel's 236 / 592.
        //
        // The 1136 B is `out_acc`, the 128-register output accumulator. Both
        // things that write it — `online_rescale` and `merge_output_tile` —
        // take it by `&mut`, so it is address-taken and lands in the LLVM
        // local depot whole. It is not hot: the only reader inside the key
        // stream is the correction path, which fires on **0.00%** of visits at
        // every gated shape.
        //
        // The band beside it *was* hot, and that is what `SCORE_CHUNK`
        // records: holding the whole `[32, 64]` score band as a value put it
        // in the depot too (1328 B, 3546 local accesses) and cost **2.635 ms**
        // at the profile shape; walking it 16 columns at a time is 2.125.
        //
        // Getting the accumulator out of the depot means not keeping it in
        // registers at all — `tcgen05.st` (absent when this correction scheme
        // was designed, present in the library now) lets the rescale happen in
        // the O segment. See the PR.
        name: "forward",
        max_registers: 246,
        max_spill_bytes: 1072,
    },
    KernelBudget {
        // The unified query-parallel backward (#69): 244 regs / 528 B frame,
        // against the warp-specialized kernel's 52 / 512 and the synchronous
        // one's 53 / 512.
        //
        // **The jump is `.maxntid`, not the code.** Those kernels declared 192
        // and 1024 threads; this one declares 128, so ptxas' per-thread budget
        // went from 65536/1024 to 65536/128 and it spent what it was given.
        // 244 × 128 threads is 31.2K of the 64K register file — two blocks'
        // worth — and residency is pinned at one CTA an SM by tensor memory
        // regardless, so there is nothing here to reclaim.
        //
        // The frame is 528 B and flat: dQ lives in tensor memory for the whole
        // key stream, so unlike the forward there is no 128-register
        // accumulator taken by `&mut` to land in the LLVM local depot. What is
        // there is the `pipeline::Job` struct itself, which `pipeline::run`
        // takes by `&mut` for the same reason.
        name: "backward q",
        max_registers: 244,
        max_spill_bytes: 528,
    },
    KernelBudget {
        // The unified key-parallel backward, at exactly kernel A's 244 / 528
        // despite carrying a second gradient accumulator and holding `Pᵀ` live
        // across forming `dSᵀ` from it — the two kernels' pressure is the same
        // `.maxntid` ceiling and neither is near what it would need to spill.
        // Its predecessors differed (59/1024 sync, 70/1024 pipelined) because
        // they were near theirs.
        name: "backward kv",
        max_registers: 244,
        max_spill_bytes: 528,
    },
];

/// Launch the forward. A free function where there used to be an enum over
/// three kernels — #68 left one, and its operand and output contract is the
/// three generations'.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_forward(
    flash: &Tcgen05Flash,
    stream: &Arc<CudaStream>,
    b: usize,
    t: usize,
    h: usize,
    ctas: usize,
    q_tma: &host::FlashHeadTmaMap,
    k_tma: &host::FlashHeadTmaMap,
    v_tma: &host::FlashHeadTmaMap,
    y: &mut DeviceBuffer<f32>,
    lse: &mut DeviceBuffer<f32>,
    corrections: &mut DeviceBuffer<u32>,
) -> Result<(), cuda_core::DriverError> {
    unsafe {
        flash.forward(
            stream,
            flash_forward_config(b, t, h, ctas),
            q_tma.as_ptr(),
            k_tma.as_ptr(),
            v_tma.as_ptr(),
            t as u32,
            h as u32,
            b as u32,
            y,
            lse,
            corrections,
        )
    }
}

const LOG2_E: f32 = std::f32::consts::LOG2_E;
const LN_2: f64 = std::f64::consts::LN_2;

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let round = 0x7fffu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// CPU mirror of `stage_attention_heads_bf16`: fp32 `[B*T, H*64]` into packed
/// bf16 head panels `[B*H, T, 64]` with `scale` folded into the rounding.
fn stage_heads(input: &[f32], b: usize, t: usize, h: usize, scale: f32) -> Vec<u32> {
    let mut staged = vec![0u32; b * h * t * FLASH_HD / 2];
    for plane in 0..b * h {
        let (batch, head) = (plane / h, plane % h);
        for token in 0..t {
            for pair in 0..FLASH_HD / 2 {
                let base = ((batch * t + token) * h + head) * FLASH_HD + pair * 2;
                let low = f32_to_bf16_rne(input[base] * scale) as u32;
                let high = f32_to_bf16_rne(input[base + 1] * scale) as u32;
                staged[(plane * t + token) * FLASH_HD / 2 + pair] = low | (high << 16);
            }
        }
    }
    staged
}

fn staged_value(staged: &[u32], t: usize, plane: usize, token: usize, feature: usize) -> f32 {
    let word = staged[(plane * t + token) * FLASH_HD / 2 + feature / 2];
    let bits = if feature % 2 == 0 { word } else { word >> 16 } as u16;
    bf16_to_f32(bits)
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_error = 0.0f32;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (a - e).abs();
        max_error = max_error.max(error);
        let tolerance = atol + rtol * e.abs();
        assert!(
            error <= tolerance,
            "{name} mismatch at {i}: tcgen05={a}, reference={e}, error={error}, \
             tolerance={tolerance}"
        );
    }
    println!("  {name:<9} max abs error: {max_error:.3e}");
}

/// Software exp2/log2 accuracy against the host libm oracles. Gates the
/// polynomial paths standalone before any attention math depends on them.
fn check_math(
    stream: &Arc<CudaStream>,
    flash: &Tcgen05Flash,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut exp_inputs: Vec<f32> = Vec::new();
    let mut i = -1300i32;
    while i <= 300 {
        exp_inputs.push(i as f32 * 0.1 + 0.0173);
        i += 1;
    }
    exp_inputs.extend_from_slice(&[0.0, -0.5, 0.5, -1.0e30, -126.0, 24.999]);
    let input = DeviceBuffer::from_host(stream, &exp_inputs)?;
    let mut output = DeviceBuffer::<f32>::zeroed(stream, exp_inputs.len())?;
    flash.software_exp2(stream, &input, &mut output)?;
    let actual = output.to_host_vec(stream)?;
    let mut max_rel = 0.0f64;
    for (&x, &a) in exp_inputs.iter().zip(&actual) {
        let reference = (x as f64).exp2();
        if x <= -125.0 {
            // Clamped flush region: anything at subnormal scale is "zero".
            assert!(
                a >= 0.0 && (a as f64) < 1.0e-35,
                "exp2({x}) flush failed: {a}"
            );
            continue;
        }
        let rel = ((a as f64) - reference).abs() / reference;
        max_rel = max_rel.max(rel);
        assert!(rel < 2.0e-4, "exp2({x}): {a} vs {reference}, rel {rel:.3e}");
    }
    println!("  exp2      max rel error: {max_rel:.3e}");

    let mut log_inputs: Vec<f32> = Vec::new();
    let mut u = -2000i32;
    while u <= 2000 {
        log_inputs.push((u as f32 * 0.01 + 0.00417).exp2());
        u += 1;
    }
    log_inputs.extend_from_slice(&[1.0, 2.0, 0.5, 1.5, 128.0]);
    let input = DeviceBuffer::from_host(stream, &log_inputs)?;
    let mut output = DeviceBuffer::<f32>::zeroed(stream, log_inputs.len())?;
    flash.software_log2(stream, &input, &mut output)?;
    let actual = output.to_host_vec(stream)?;
    let mut max_abs = 0.0f64;
    for (&x, &a) in log_inputs.iter().zip(&actual) {
        let reference = (x as f64).log2();
        let abs = ((a as f64) - reference).abs();
        max_abs = max_abs.max(abs);
        assert!(abs < 1.0e-5, "log2({x}): {a} vs {reference}, abs {abs:.3e}");
    }
    println!("  log2      max abs error: {max_abs:.3e}");
    Ok(())
}

/// Empirically verify the SWIZZLE_128B placement the P-write path assumes on
/// one 64-wide `[TILE, 64]` subtile: fill a full `[TILE, 128]` panel with
/// sequential word indices, TMA its first (columns 0..64) subtile into shared
/// memory, dump the words linearly, and check each landing position against
/// `smem(r, c, k) = r*32 + (c ^ ((r + phase) & 7))*4 + k` against the source
/// word `r*64 + c*4 + k` — the 16-byte chunk XORed with the subtile's
/// *absolute* 128-byte row phase, which the kernel reports as a trailing word
/// (the swizzle acts on physical address bits [9:7], not tile-relative rows).
/// On mismatch, print enough of the observed permutation to derive the real
/// formula.
fn check_swizzle_layout(
    stream: &Arc<CudaStream>,
    flash: &Tcgen05Flash,
) -> Result<(), Box<dyn std::error::Error>> {
    let panel_words = FLASH_HD * 2 / 4; // 64 source u32 words per full-panel row
    let sub_words = FLASH_SUBTILE_HD * 2 / 4; // 32 smem words per subtile row
    let chunks = sub_words / 4; // 8 sixteen-byte chunks per subtile row
    let source_total = FLASH_TILE * panel_words;
    let smem_total = FLASH_TILE * sub_words;
    let pattern: Vec<u32> = (0..source_total as u32).collect();
    let source = DeviceBuffer::from_host(stream, &pattern)?;
    let tma = unsafe { create_flash_head_tma_map(stream, &source, FLASH_TILE, 1)? };
    let mut output = DeviceBuffer::<u32>::zeroed(stream, smem_total + 1)?;
    unsafe { flash.swizzle_probe(stream, tma.as_ptr(), &mut output)? };
    let dump = output.to_host_vec(stream)?;
    let phase = dump[smem_total] as usize;

    let mut mismatches = 0usize;
    for row in 0..FLASH_TILE {
        for chunk in 0..chunks {
            for sub in 0..4 {
                let source_word = row * panel_words + chunk * 4 + sub;
                let landing = row * sub_words + (chunk ^ ((row + phase) & 7)) * 4 + sub;
                if dump[landing] != source_word as u32 {
                    if mismatches < 24 {
                        println!(
                            "  word {source_word} (row {row} chunk {chunk}+{sub}) expected at \
                             {landing}, smem[{landing}] = {}",
                            dump[landing]
                        );
                    }
                    mismatches += 1;
                }
            }
        }
    }
    if mismatches == 0 {
        println!("  swizzle   TMA 128B placement matches chunk ^ ((row + {phase}) & 7)");
    } else {
        println!("  swizzle   {mismatches}/{smem_total} words off (phase {phase}); rows 0..4:");
        for row in 0..4 {
            let observed: Vec<u32> = (0..chunks)
                .map(|c| dump[row * sub_words + c * 4] % (panel_words as u32) / 4)
                .collect();
            println!("    row {row}: source chunk order in smem = {observed:?}");
        }
        return Err("TMA swizzle layout differs from the P-write assumption".into());
    }
    Ok(())
}

/// Transposed-B operand validation: `C[64,64] = A[64,64]·B[64,64]` with `B`
/// stored `[K, N]` row-major, the V-subtile orientation of the `O = P·V` MMA.
/// A and B are each staged as one head panel whose first 64 columns hold the
/// operand, so the probe reuses the exact head-panel TMA subtile machinery of
/// the forward.
fn check_transpose_probe(
    stream: &Arc<CudaStream>,
    flash: &Tcgen05Flash,
) -> Result<(), Box<dyn std::error::Error>> {
    // score_mma path (2 HD subtiles, K=128, M128 shape over 64-row tiles):
    // C[m,n] = sum_k A[m,k]·B[n,k] (the Q·Kᵀ Gram).
    let a = uniform_vec(FLASH_TILE * FLASH_HD, 91);
    let b = uniform_vec(FLASH_TILE * FLASH_HD, 92);
    let a_staged = stage_heads(&a, 1, FLASH_TILE, 1, 1.0);
    let b_staged = stage_heads(&b, 1, FLASH_TILE, 1, 1.0);

    let mut expected = vec![0.0f32; FLASH_TILE * FLASH_SUBTILE_HD];
    for m in 0..FLASH_TILE {
        for n in 0..FLASH_SUBTILE_HD {
            let mut sum = 0.0f64;
            for k in 0..FLASH_HD {
                let a_value = staged_value(&a_staged, FLASH_TILE, 0, m, k);
                let b_value = staged_value(&b_staged, FLASH_TILE, 0, n, k);
                sum += a_value as f64 * b_value as f64;
            }
            expected[m * FLASH_SUBTILE_HD + n] = sum as f32;
        }
    }

    let a_device = DeviceBuffer::from_host(stream, &a_staged)?;
    let b_device = DeviceBuffer::from_host(stream, &b_staged)?;
    let a_tma = unsafe { create_flash_head_tma_map(stream, &a_device, FLASH_TILE, 1)? };
    let b_tma = unsafe { create_flash_head_tma_map(stream, &b_device, FLASH_TILE, 1)? };
    let mut output = DeviceBuffer::<f32>::zeroed(stream, FLASH_TILE * FLASH_SUBTILE_HD)?;
    unsafe { flash.transpose_probe(stream, a_tma.as_ptr(), b_tma.as_ptr(), &mut output)? };
    assert_close(
        "score_mma probe",
        &output.to_host_vec(stream)?,
        &expected,
        5.0e-2,
        5.0e-2,
    );
    Ok(())
}

/// Forward parity for every kernel against one CPU reference computed from
/// the same staged bf16 operands (exact exp2, f64 accumulation), so the
/// tolerance covers only the device-side differences: fp32 tensor-core
/// accumulation, the exp2 polynomial, the bf16 rounding of P, and the
/// conditional-segment rescale points. Also reports each kernel's measured
/// mid-stream O-segment correction rate.
fn check_forward(
    stream: &Arc<CudaStream>,
    flash: &Tcgen05Flash,
    sm_count: usize,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let d = h * FLASH_HD;
    let n = b * t;
    let q = uniform_vec(n * d, 71);
    let k = uniform_vec(n * d, 72);
    let v = uniform_vec(n * d, 73);
    let q_scale = LOG2_E / (FLASH_HD as f32).sqrt();
    let q_staged = stage_heads(&q, b, t, h, q_scale);
    let k_staged = stage_heads(&k, b, t, h, 1.0);
    let v_staged = stage_heads(&v, b, t, h, 1.0);

    let mut expected_y = vec![0.0f32; n * d];
    let mut expected_lse = vec![0.0f32; n * h];
    for plane in 0..b * h {
        let (batch, head) = (plane / h, plane % h);
        for token in 0..t {
            let mut scores = vec![0.0f64; token + 1];
            let mut row_max = f64::NEG_INFINITY;
            for key in 0..=token {
                let mut sum = 0.0f64;
                for feature in 0..FLASH_HD {
                    sum += staged_value(&q_staged, t, plane, token, feature) as f64
                        * staged_value(&k_staged, t, plane, key, feature) as f64;
                }
                scores[key] = sum;
                row_max = row_max.max(sum);
            }
            let mut denominator = 0.0f64;
            for key in 0..=token {
                denominator += (scores[key] - row_max).exp2();
            }
            let row = batch * t + token;
            for feature in 0..FLASH_HD {
                let mut acc = 0.0f64;
                for key in 0..=token {
                    acc += (scores[key] - row_max).exp2()
                        * staged_value(&v_staged, t, plane, key, feature) as f64;
                }
                expected_y[row * d + head * FLASH_HD + feature] = (acc / denominator) as f32;
            }
            expected_lse[row * h + head] = (LN_2 * (row_max + denominator.log2())) as f32;
        }
    }

    let q_device = DeviceBuffer::from_host(stream, &q_staged)?;
    let k_device = DeviceBuffer::from_host(stream, &k_staged)?;
    let v_device = DeviceBuffer::from_host(stream, &v_staged)?;
    let q_tma = unsafe { create_flash_head_tma_map(stream, &q_device, t, b * h)? };
    let k_tma = unsafe { create_flash_head_tma_map(stream, &k_device, t, b * h)? };
    let v_tma = unsafe { create_flash_head_tma_map(stream, &v_device, t, b * h)? };
    let mut y = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut lse = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
    let mut corrections = DeviceBuffer::<u32>::zeroed(stream, correction_count_len(b, t, h))?;
    unsafe {
        launch_forward(
            flash,
            stream,
            b,
            t,
            h,
            sm_count,
            &q_tma,
            &k_tma,
            &v_tma,
            &mut y,
            &mut lse,
            &mut corrections,
        )?;
    }
    println!("tcgen05 forward parity against staged-bf16 CPU reference [{b},{t},{h},{FLASH_HD}]");
    let y_host = y.to_host_vec(stream)?;
    let lse_host = lse.to_host_vec(stream)?;
    // Diagnostic slice (batch 0, head 0): one y feature and the LSE per
    // sampled row. LSE encodes the kernel's internal row max + sum, so a
    // matching LSE with a broken y isolates the P·V path, and vice versa.
    for row in [0usize, 1, 2, 7, 8, 15, 16, 31, 32, 64, 127] {
        if row < t {
            println!(
                "  row {row:>3}: y0 gpu {:>12.6} ref {:>12.6} | lse gpu {:>12.6} ref {:>12.6}",
                y_host[row * d],
                expected_y[row * d],
                lse_host[row * h],
                expected_lse[row * h],
            );
        }
    }
    // Measured maxima: y 1.4e-3, lse 1.4e-4 (T=128..512); ~3x headroom.
    assert_close("y", &y_host, &expected_y, 5.0e-3, 5.0e-3);
    assert_close("lse", &lse_host, &expected_lse, 1.0e-3, 0.0);
    print_correction_rate(stream, &corrections, b, t, h)?;
    Ok(())
}

/// Backward parity for the two tcgen05 gradient kernels against a CPU
/// reference computed from the same staged bf16 operands (exact exp2, f64
/// accumulation), mirroring the exact operations each kernel issues so the
/// tolerance covers only device-side arithmetic: fp32 tensor-core
/// accumulation, the exp2 polynomial, and the bf16 rounding of the P/dS
/// operands. The saved LSE (natural log) and softmax dot are computed on the
/// CPU and fed to the kernels exactly as the model's `backward_dot` would.
fn check_backward(
    stream: &Arc<CudaStream>,
    flash: &Tcgen05Flash,
    sm_count: usize,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let d = h * FLASH_HD;
    let n = b * t;
    let q = uniform_vec(n * d, 61);
    let k = uniform_vec(n * d, 62);
    let v = uniform_vec(n * d, 63);
    let dy = uniform_vec(n * d, 64);
    let q_scale = LOG2_E / (FLASH_HD as f32).sqrt();
    let q_staged = stage_heads(&q, b, t, h, q_scale);
    let k_staged = stage_heads(&k, b, t, h, 1.0);
    let v_staged = stage_heads(&v, b, t, h, 1.0);
    let dy_staged = stage_heads(&dy, b, t, h, 1.0);

    let scale = 1.0 / (FLASH_HD as f64).sqrt();
    let mut lse = vec![0.0f32; n * h];
    let mut dot = vec![0.0f32; n * h];
    let mut expected_dq = vec![0.0f32; n * d];
    let mut expected_dk = vec![0.0f32; n * d];
    let mut expected_dv = vec![0.0f32; n * d];
    for plane in 0..b * h {
        let (batch, head) = (plane / h, plane % h);
        // Per query row: base-2 scores, row max, denominator, and dot.
        let mut probabilities = vec![vec![0.0f64; t]; t];
        let mut dp = vec![vec![0.0f64; t]; t];
        let mut row_dot = vec![0.0f64; t];
        for query in 0..t {
            let mut row_max = f64::NEG_INFINITY;
            for key in 0..=query {
                let mut score = 0.0f64;
                for feature in 0..FLASH_HD {
                    score += staged_value(&q_staged, t, plane, query, feature) as f64
                        * staged_value(&k_staged, t, plane, key, feature) as f64;
                }
                probabilities[query][key] = score;
                row_max = row_max.max(score);
            }
            let mut denominator = 0.0f64;
            for key in 0..=query {
                let value = (probabilities[query][key] - row_max).exp2();
                probabilities[query][key] = value;
                denominator += value;
            }
            for key in 0..=query {
                probabilities[query][key] /= denominator;
            }
            let mut acc_dot = 0.0f64;
            for feature in 0..FLASH_HD {
                let mut y = 0.0f64;
                for key in 0..=query {
                    y += probabilities[query][key]
                        * staged_value(&v_staged, t, plane, key, feature) as f64;
                }
                acc_dot += staged_value(&dy_staged, t, plane, query, feature) as f64 * y;
            }
            row_dot[query] = acc_dot;
            for key in 0..=query {
                let mut value = 0.0f64;
                for feature in 0..FLASH_HD {
                    value += staged_value(&dy_staged, t, plane, query, feature) as f64
                        * staged_value(&v_staged, t, plane, key, feature) as f64;
                }
                dp[query][key] = value;
            }
            let row = batch * t + query;
            lse[row * h + head] = (LN_2 * (row_max + denominator.log2())) as f32;
            dot[row * h + head] = acc_dot as f32;
        }
        // dQ = Σ_k dS·K (scale folded), dK = Σ_q dSᵀ·Q_staged (ln2 folded so
        // ln2·scale·log2e = scale), dV = Σ_q Pᵀ·dY.
        for query in 0..t {
            let out = (batch * t + query) * d + head * FLASH_HD;
            for feature in 0..FLASH_HD {
                let mut value = 0.0f64;
                for key in 0..=query {
                    let dscore =
                        probabilities[query][key] * (dp[query][key] - row_dot[query]) * scale;
                    value += dscore * staged_value(&k_staged, t, plane, key, feature) as f64;
                }
                expected_dq[out + feature] = value as f32;
            }
        }
        for key in 0..t {
            let out = (batch * t + key) * d + head * FLASH_HD;
            for feature in 0..FLASH_HD {
                let mut dk_value = 0.0f64;
                let mut dv_value = 0.0f64;
                for query in key..t {
                    let probability = probabilities[query][key];
                    let dscore = probability * (dp[query][key] - row_dot[query]) * LN_2;
                    dk_value += dscore * staged_value(&q_staged, t, plane, query, feature) as f64;
                    dv_value +=
                        probability * staged_value(&dy_staged, t, plane, query, feature) as f64;
                }
                expected_dk[out + feature] = dk_value as f32;
                expected_dv[out + feature] = dv_value as f32;
            }
        }
    }

    let q_device = DeviceBuffer::from_host(stream, &q_staged)?;
    let k_device = DeviceBuffer::from_host(stream, &k_staged)?;
    let v_device = DeviceBuffer::from_host(stream, &v_staged)?;
    let dy_device = DeviceBuffer::from_host(stream, &dy_staged)?;
    let q_tma = unsafe { create_flash_head_tma_map(stream, &q_device, t, b * h)? };
    let k_tma = unsafe { create_flash_head_tma_map(stream, &k_device, t, b * h)? };
    let v_tma = unsafe { create_flash_head_tma_map(stream, &v_device, t, b * h)? };
    let dy_tma = unsafe { create_flash_head_tma_map(stream, &dy_device, t, b * h)? };
    let lse_device = DeviceBuffer::from_host(stream, &lse)?;
    let dot_device = DeviceBuffer::from_host(stream, &dot)?;

    let mut dq = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut dk = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut dv = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    unsafe {
        flash.backward_q(
            stream,
            flash_backward_q_config(b, t, h, sm_count),
            q_tma.as_ptr(),
            k_tma.as_ptr(),
            v_tma.as_ptr(),
            dy_tma.as_ptr(),
            &lse_device,
            &dot_device,
            t as u32,
            h as u32,
            b as u32,
            &mut dq,
        )?;
        flash.backward_kv(
            stream,
            flash_backward_kv_config(b, t, h, sm_count),
            q_tma.as_ptr(),
            k_tma.as_ptr(),
            v_tma.as_ptr(),
            dy_tma.as_ptr(),
            &lse_device,
            &dot_device,
            t as u32,
            h as u32,
            b as u32,
            &mut dk,
            &mut dv,
        )?;
    }
    println!("tcgen05 backward parity against staged-bf16 CPU reference [{b},{t},{h},{FLASH_HD}]");
    assert_close("dq", &dq.to_host_vec(stream)?, &expected_dq, 5.0e-3, 5.0e-3);
    assert_close("dk", &dk.to_host_vec(stream)?, &expected_dk, 5.0e-3, 5.0e-3);
    assert_close("dv", &dv.to_host_vec(stream)?, &expected_dv, 5.0e-3, 5.0e-3);
    Ok(())
}

/// Sum the per-workstream correction counts and report them against the
/// number of key-tile visits that could have corrected (everything past
/// each stream's first tile).
fn print_correction_rate(
    stream: &Arc<CudaStream>,
    corrections: &DeviceBuffer<u32>,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let counts = corrections.to_host_vec(stream)?;
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    // A query block of `FLASH_QUERIES` rows streams `2 * (block + 1)` key
    // tiles and cannot correct on its first, so the eligible visits are the
    // key-tile visits less one per work item.
    let blocks = t / FLASH_QUERIES;
    let visits = blocks * (blocks + 1);
    let eligible = (b * h * (visits - blocks)) as u64;
    let rate = if eligible == 0 {
        0.0
    } else {
        total as f64 / eligible as f64 * 100.0
    };
    println!("  corrections: {total} of {eligible} eligible tile visits ({rate:.2}%)");
    Ok(())
}

/// Kernel-only TFLOP/s at the post-7e9 profile shape (B=32, T=1024, H=24)
/// for all three forwards in the same container — the deltas are the direct
/// measure of what each phase bought — plus the profile-shape correction
/// rate (the phase-3 conditional-rescale checklist number).
fn bench(
    stream: &Arc<CudaStream>,
    flash: &Tcgen05Flash,
    sm_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let (b, t, h) = (32usize, 1024usize, 24usize);
    let d = h * FLASH_HD;
    let n = b * t;
    let q_scale = LOG2_E / (FLASH_HD as f32).sqrt();
    let q_staged = stage_heads(&uniform_vec(n * d, 81), b, t, h, q_scale);
    let k_staged = stage_heads(&uniform_vec(n * d, 82), b, t, h, 1.0);
    let v_staged = stage_heads(&uniform_vec(n * d, 83), b, t, h, 1.0);
    let q_device = DeviceBuffer::from_host(stream, &q_staged)?;
    let k_device = DeviceBuffer::from_host(stream, &k_staged)?;
    let v_device = DeviceBuffer::from_host(stream, &v_staged)?;
    let q_tma = unsafe { create_flash_head_tma_map(stream, &q_device, t, b * h)? };
    let k_tma = unsafe { create_flash_head_tma_map(stream, &k_device, t, b * h)? };
    let v_tma = unsafe { create_flash_head_tma_map(stream, &v_device, t, b * h)? };
    let mut y = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut lse = DeviceBuffer::<f32>::zeroed(stream, n * h)?;
    let mut corrections = DeviceBuffer::<u32>::zeroed(stream, correction_count_len(b, t, h))?;

    // ALGORITHMIC work, which is the number that survived #68: one
    // `[64, 64]` score block and its `[64, 128]` output contribution per
    // causal `(query tile, key tile)` pair, `nt*(nt+1)/2` pairs per (b, h).
    //
    // Through #35 this counted *issued* MMA work instead, and issued was twice
    // algorithmic because a 64-query tile filled half of every `M128`. The
    // unified forward issues no phantom rows, so the two counts have converged
    // and the old TFLOP/s figures are 2x this scale — compare the milliseconds.
    let tiles = t / FLASH_TILE;
    let pairs = (b * h * tiles * (tiles + 1) / 2) as f64;
    let flop = pairs * 2.0 * (2.0 * 64.0 * 64.0 * 128.0);
    let milliseconds = time_gpu_iters(stream, 3, 20, || {
        unsafe {
            launch_forward(
                flash,
                stream,
                b,
                t,
                h,
                sm_count,
                &q_tma,
                &k_tma,
                &v_tma,
                &mut y,
                &mut lse,
                &mut corrections,
            )?;
        }
        Ok(())
    })?;
    println!(
        "tcgen05 forward [{b},{t},{h},{FLASH_HD}]: {milliseconds:.3} ms, {:.1} TFLOP/s",
        flop / (milliseconds * 1.0e-3) / 1.0e12
    );
    print_correction_rate(stream, &corrections, b, t, h)?;

    // Backward kernels at the same shape. Per key-tile visit each kernel
    // issues its two 128x128x64 score GEMMs plus a 128x64x128 gradient GEMM
    // (kernel A: dQ; kernel B: dV and dK — three gradient GEMMs there), so the
    // combined backward MMA work is 5 GEMMs per visit against the forward's 2.
    let dy_staged = stage_heads(&uniform_vec(n * d, 84), b, t, h, 1.0);
    let dy_device = DeviceBuffer::from_host(stream, &dy_staged)?;
    let dy_tma = unsafe { create_flash_head_tma_map(stream, &dy_device, t, b * h)? };
    let lse_in = DeviceBuffer::from_host(stream, &vec![0.0f32; n * h])?;
    let dot_in = DeviceBuffer::from_host(stream, &vec![0.0f32; n * h])?;
    let mut dq = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut dk = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    let mut dv = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
    // Both kernels together: five 128x128x64-equivalent GEMMs per causal
    // (query tile, key tile) pair, which is the price of the split — a fused
    // backward would issue three and pay for the difference in gradient
    // traffic instead. See the PR.
    let backward_flop = pairs * 5.0 * (2.0 * 128.0 * 128.0 * 64.0);

    // Per key-tile visit kernel A issues two 128x128x64 score GEMMs plus one
    // 128x64x128 gradient GEMM; kernel B two score GEMMs plus two gradient
    // ones. Read the milliseconds — the baselines these replace are recorded
    // in the PR, and a TFLOP/s ratio between two kernels that issue different
    // work is not one.
    let q_flop = pairs * 3.0 * (2.0 * 128.0 * 128.0 * 64.0);
    let kv_flop = pairs * 4.0 * (2.0 * 128.0 * 128.0 * 64.0);
    let milliseconds = time_gpu_iters(stream, 3, 20, || {
        unsafe {
            flash.backward_kv(
                stream,
                flash_backward_kv_config(b, t, h, sm_count),
                q_tma.as_ptr(),
                k_tma.as_ptr(),
                v_tma.as_ptr(),
                dy_tma.as_ptr(),
                &lse_in,
                &dot_in,
                t as u32,
                h as u32,
                b as u32,
                &mut dk,
                &mut dv,
            )?;
        }
        Ok(())
    })?;
    println!(
        "tcgen05 backward kv [{b},{t},{h},{FLASH_HD}]: {milliseconds:.3} ms, {:.1} TFLOP/s",
        kv_flop / (milliseconds * 1.0e-3) / 1.0e12
    );
    let milliseconds = time_gpu_iters(stream, 3, 20, || {
        unsafe {
            flash.backward_q(
                stream,
                flash_backward_q_config(b, t, h, sm_count),
                q_tma.as_ptr(),
                k_tma.as_ptr(),
                v_tma.as_ptr(),
                dy_tma.as_ptr(),
                &lse_in,
                &dot_in,
                t as u32,
                h as u32,
                b as u32,
                &mut dq,
            )?;
        }
        Ok(())
    })?;
    println!(
        "tcgen05 backward q [{b},{t},{h},{FLASH_HD}]: {milliseconds:.3} ms, {:.1} TFLOP/s",
        q_flop / (milliseconds * 1.0e-3) / 1.0e12
    );

    let milliseconds = time_gpu_iters(stream, 3, 20, || {
        unsafe {
            flash.backward_q(
                stream,
                flash_backward_q_config(b, t, h, sm_count),
                q_tma.as_ptr(),
                k_tma.as_ptr(),
                v_tma.as_ptr(),
                dy_tma.as_ptr(),
                &lse_in,
                &dot_in,
                t as u32,
                h as u32,
                b as u32,
                &mut dq,
            )?;
            flash.backward_kv(
                stream,
                flash_backward_kv_config(b, t, h, sm_count),
                q_tma.as_ptr(),
                k_tma.as_ptr(),
                v_tma.as_ptr(),
                dy_tma.as_ptr(),
                &lse_in,
                &dot_in,
                t as u32,
                h as u32,
                b as u32,
                &mut dk,
                &mut dv,
            )?;
        }
        Ok(())
    })?;
    println!(
        "tcgen05 backward, both kernels [{b},{t},{h},{FLASH_HD}]: {milliseconds:.3} ms, {:.1} TFLOP/s",
        backward_flop / (milliseconds * 1.0e-3) / 1.0e12
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        host::FLASH_FORWARD_SMEM_BYTES as usize,
        tcgen05_device::FLASH_FORWARD_SMEM,
        "the launch must request the kernel's own plan, or residency is the request's"
    );
    println!(
        "forward plan: {} B, {} CTA/SM",
        tcgen05_device::FLASH_FORWARD_SMEM,
        host::FLASH_FORWARD_CTAS_PER_SM
    );
    assert_eq!(
        host::FLASH_FORWARD_BLOCK_THREADS as usize,
        tcgen05_device::FLASH_FORWARD_BLOCK,
        "host.rs and tcgen05.rs disagree on the forward block width"
    );
    assert_eq!(
        host::FLASH_BACKWARD_Q_SMEM_BYTES as usize,
        tcgen05_device::FLASH_BACKWARD_Q_SMEM,
        "host.rs and tcgen05.rs disagree on the kernel-A shared plan"
    );
    assert_eq!(
        host::FLASH_BACKWARD_KV_SMEM_BYTES as usize,
        tcgen05_device::FLASH_BACKWARD_KV_SMEM,
        "host.rs and tcgen05.rs disagree on the kernel-B shared plan"
    );
    assert_eq!(
        host::FLASH_BACKWARD_BLOCK_THREADS as usize,
        tcgen05_device::FLASH_BACKWARD_BLOCK,
        "host.rs and tcgen05.rs disagree on the backward block width"
    );
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let flash = Tcgen05Flash::load(&ctx)?;
    let sm_count = device_sm_count(&ctx)?;
    println!("persistent grid: min(work items, {sm_count} SMs)");

    // What ptxas gave each kernel, gated against the pinned ceilings.
    // `.maxntid` (from `#[launch_bounds]`) is the register budget's divisor,
    // so a stale one shows up here as a low register count and a nonzero
    // spill frame — never in the PTX, which ptxas consumes.
    enforce_kernel_budgets(&flash.kernels(), &KERNEL_BUDGETS)?;

    println!("software math parity");
    check_math(&stream, &flash)?;
    println!("TMA swizzle layout probe");
    check_swizzle_layout(&stream, &flash)?;
    println!("transpose_b operand probe");
    check_transpose_probe(&stream, &flash)?;
    // T=384 exercises the persistent kernel's inactive-stream-B pair;
    // (4, 256, 38) puts 152 work items over the SM count so CTAs loop.
    check_forward(&stream, &flash, sm_count, 2, 128, 3)?;
    check_forward(&stream, &flash, sm_count, 1, 256, 2)?;
    check_forward(&stream, &flash, sm_count, 1, 384, 2)?;
    check_forward(&stream, &flash, sm_count, 1, 512, 2)?;
    check_forward(&stream, &flash, sm_count, 4, 256, 38)?;
    check_backward(&stream, &flash, sm_count, 2, 128, 3)?;
    check_backward(&stream, &flash, sm_count, 1, 256, 2)?;
    check_backward(&stream, &flash, sm_count, 1, 1024, 4)?;
    println!("✓ tcgen05 parity passed");
    bench(&stream, &flash, sm_count)?;
    Ok(())
}
