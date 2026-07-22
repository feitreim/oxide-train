//! FA4-shaped tcgen05 attention forward (issue #35, phases 1–3) and backward
//! (phase 4).
//!
//! This module compiles ONLY into `src/bin/flash.rs`, whose device artifact
//! takes the pure-PTX path and ships as `flash.ptx`. It must never be reached
//! from `lib.rs`: the oracle kernels there use libdevice math, which forces
//! their artifact through libNVVM, and libNVVM rejects tcgen05 constructs.
//! For the same reason nothing here may call `f32::exp/ln/sqrt/floor` — the
//! softmax runs on a software `exp2` and the LSE epilogue on a software
//! `log2`, which is also FA4's SFU-offload optimization.
//!
//! Kernel shape contract (the host launchers in `host.rs` enforce it):
//! - operands are packed-bf16 staging buffers `[B*H, T, HD]`, one contiguous
//!   `[T, HD]` panel per head, produced by tensor-gpu's
//!   `stage_attention_heads_bf16` (Q arrives pre-scaled by
//!   `softmax_scale * log2(e)`, so scores are base-2 native);
//! - `T` is a multiple of the 128-row tile; `HD == 64`; non-aligned shapes
//!   stay on the fp32 tiled kernels in `lib.rs`;
//! - outputs keep the existing contract: fp32 `y[B*T, H*HD]` and fp32
//!   `logsumexp[B*T, H]` in natural-log units.
//!
//! One CTA workstream owns a 128-query tile of one `(batch, head)` and
//! streams 128-key tiles: TMA loads Q/K/V into swizzled shared tiles,
//! `S = Q·Kᵀ` accumulates in fp32 TMEM, a register softmax (mask → row max →
//! software exp2 → running sum) packs bf16 probabilities back to shared
//! memory with swizzled `stmatrix` stores, and `O += P·V` accumulates in a
//! TMEM *segment* under a fixed per-row max reference (`enable_d` across
//! tiles). FA4's conditional correction, adapted for the missing
//! `tcgen05.st`: only when some row's tile max climbs more than
//! `CORRECTION_THRESHOLD` above the reference does the warpgroup drain the
//! segment into per-thread registers, rescale, and restart it — otherwise
//! the segment just keeps accumulating and the softmax warpgroup never
//! touches O TMEM.
//!
//! Three kernels share the per-tile math (`softmax_tile` /
//! `merge_output_tile` / `store_outputs`):
//! - `flash_forward_tcgen05` — the phase-1 synchronous form, one stage of
//!   everything, block-wide `sync_threads` between stages. Kept as the
//!   in-artifact oracle and the debug fallback for pipeline hangs.
//! - `flash_forward_pipelined` — the phase-2 warp-specialized form: a TMA
//!   load warp runs a `PIPELINE_STAGES`-deep K/V ring, an MMA warp issues
//!   tcgen05 into a double-buffered S so `S-MMA(i+1)` runs while the
//!   128-thread softmax warpgroup works tile `i`, and every handoff is an
//!   mbarrier phase-parity spin (no named barriers in v0.2.1; FA4 does the
//!   same). Correction and epilogue stay fused into the softmax warpgroup:
//!   the output accumulator lives in its registers and TMEM lane ownership
//!   pins the drains to warps 0–3 — a separate correction warpgroup only
//!   becomes possible with `tcgen05.st` (#34).
//! - `flash_forward_persistent` — the phase-3 form: two softmax warpgroups
//!   ping-pong adjacent query tiles over one shared K/V ring and one MMA
//!   warp, and CTAs run a static persistent work-item loop. See the kernel
//!   doc for the scheduling and barrier story.
//!
//! Phase 4 adds two synchronous backward kernels sharing the same idioms —
//! the swizzle-aware bf16 fragment writes, the transposed-B gradient MMA
//! shape, and the fp32 TMEM accumulators. `flash_backward_q_tcgen05`
//! (query-parallel) recomputes `S`/`dP` per key tile and accumulates
//! `dQ += dS·K`; `flash_backward_kv_tcgen05` (key-parallel) recomputes the
//! transposed `Sᵀ`/`dPᵀ` per query tile and accumulates `dV += Pᵀ·dY` and
//! `dK += dSᵀ·Q`. Probabilities are recomputed base-2 from the saved LSE
//! (`P = exp2(s − lse·log2e)`, no running-max machinery); gradient writes are
//! disjoint by tile so there are no atomics. The three-kernel split
//! (`backward_dot` stays fp32 in `lib.rs`, then dQ, then dK/dV) keeps the
//! gradient outputs disjoint. Both take the packed-bf16 Q/K/V/dY staging
//! panels plus the read-only `logsumexp` (natural log) and `dot` (`Σ dy·y`)
//! device slices, and write fp32 `dq`/`dk`/`dv`.

use cuda_device::DisjointSlice;
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
    mbarrier_init, mbarrier_inval, mbarrier_try_wait_parity,
};
use cuda_device::shared::{DynamicSharedArray, SharedArray};
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
    cvt_f32x2_bf16x2, stmatrix_m8n8_x2, tcgen05_alloc, tcgen05_commit_shared_cluster,
    tcgen05_dealloc, tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync,
    tcgen05_ld_16x256b_pure, tcgen05_load_wait, tcgen05_mma_f16,
};
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_3d_g2s};
use cuda_device::{cuda_module, kernel, launch_bounds, thread, warp};

// Tile contract; `host.rs` mirrors these as FLASH_TILE / FLASH_HD (kept
// non-pub here so SWEEP's one-definition rule never sees two copies).
const TILE: usize = 128;
const HD: usize = 64;

/// Bytes of one swizzled bf16 `[128, 64]` shared tile.
const TILE_BYTES: usize = TILE * HD * 2;
/// Dynamic shared plan of the synchronous kernel: Q, K, V tiles plus the two
/// stacked P subtiles.
pub const FLASH_DYNAMIC_SMEM: usize = 5 * TILE_BYTES;

/// K/V ring depth of the pipelined kernel (SWEEP knob). Two is the floor:
/// the staggered issue order (`S-MMA(i)` before `O-MMA(i-1)`) needs one
/// stage of load-ahead to make progress. Four is the ceiling the host-side
/// launch allocation (`host::FLASH_PIPELINE_SMEM_BYTES`) is sized for.
pub const PIPELINE_STAGES: usize = 3;
const _: () = assert!(2 <= PIPELINE_STAGES && PIPELINE_STAGES <= 4);
/// Dynamic shared plan of the pipelined kernel: Q, the K and V rings, and
/// the two stacked P subtiles.
pub const FLASH_PIPELINE_SMEM: usize = (3 + 2 * PIPELINE_STAGES) * TILE_BYTES;
/// Threads of the pipelined kernel: the softmax/correction/epilogue
/// warpgroup plus the TMA-load warp and the MMA-issue warp.
pub const FLASH_PIPELINE_BLOCK: usize = TILE + 64;

/// Dynamic shared plan of the synchronous query-parallel backward (kernel A):
/// the resident Q and dY tiles, the streamed K and V tiles, and the two
/// stacked dS subtiles.
pub const FLASH_BACKWARD_Q_SMEM: usize = 6 * TILE_BYTES;
/// Dynamic shared plan of the synchronous key-parallel backward (kernel B):
/// the resident K and V tiles, the streamed Q and dY tiles, and the two
/// stacked subtile pairs (Pᵀ and dSᵀ).
pub const FLASH_BACKWARD_KV_SMEM: usize = 8 * TILE_BYTES;

/// Base-2 slack a tile's row max may climb above the O segment's reference
/// before the warpgroup forces a correction (SWEEP knob). P values reach at
/// most `2^CORRECTION_THRESHOLD`, comfortably inside bf16 range and the fp32
/// accumulation headroom of a full key stream.
pub const CORRECTION_THRESHOLD: f32 = 8.0;

/// K/V ring depth of the persistent kernel: `PIPELINE_STAGES` capped at 3 so
/// the doubled Q/P footprint of two workstreams stays inside the ~227 KiB
/// shared-memory budget.
pub const PERSISTENT_STAGES: usize = if PIPELINE_STAGES < 3 { PIPELINE_STAGES } else { 3 };
/// Dynamic shared plan of the persistent kernel: two Q tiles, the K and V
/// rings, and one two-subtile P buffer per workstream.
pub const FLASH_PERSISTENT_SMEM: usize = (2 + 2 * PERSISTENT_STAGES + 4) * TILE_BYTES;
/// Threads of the persistent kernel: two softmax warpgroups plus the
/// TMA-load warp and the MMA-issue warp.
pub const FLASH_PERSISTENT_BLOCK: usize = 2 * TILE + 64;

/// Finite stand-in for "masked" in the base-2 score domain; far enough below
/// any real score that `exp2` flushes it to a subnormal-scale value while the
/// running-max recurrence stays NaN-free.
const MASKED_SCORE: f32 = -1.0e30;

#[cuda_module]
pub mod kernels {
    use super::*;

    const LN2: f32 = 0.693_147_18;
    /// Softmax scale for `HD == 64` (`1/sqrt(64)`), written as a literal
    /// because `1.0/(HD as f32).sqrt()` would lower to libdevice `sqrtf`.
    const SCALE: f32 = 0.125;
    /// `log2(e)`, converting the saved natural-log LSE into the base-2 domain
    /// the recomputed probabilities live in.
    const LOG2E: f32 = 1.442_695_04;

    /// Same encoding as gemm's operand descriptors: SWIZZLE_128B tiles with a
    /// 16-byte leading offset and 1024-byte stride.
    #[inline(always)]
    fn smem_descriptor(smem_address: u64) -> u64 {
        const LEADING_BYTES: u32 = 16;
        const STRIDE_BYTES: u32 = 1024;
        const SWIZZLE_128B: u8 = 2;
        let address = (smem_address >> 4) & 0x3fff;
        let leading = ((LEADING_BYTES >> 4) & 0x3fff) as u64;
        let stride = ((STRIDE_BYTES >> 4) & 0x3fff) as u64;
        address | (leading << 16) | (stride << 32) | (1u64 << 46) | ((SWIZZLE_128B as u64) << 61)
    }

    /// NaN-free float max/min. `f32::max`/`f32::min` lower to libdevice
    /// (`__nv_fmaxf`), which would silently force this artifact off the
    /// pure-PTX path; comparison + select stays native. All scores here are
    /// finite by construction (`MASKED_SCORE` is finite).
    #[inline(always)]
    fn fmax(a: f32, b: f32) -> f32 {
        if a > b { a } else { b }
    }

    #[inline(always)]
    fn fmin(a: f32, b: f32) -> f32 {
        if a < b { a } else { b }
    }

    /// `2^x` on FMA units: round-to-nearest split via the 1.5·2²³ shift trick,
    /// exponent-bit insertion for the integer part, and a degree-3 minimax
    /// polynomial (max relative error 7.5e-5 on the reduced range) for the
    /// fraction. The clamp keeps the exponent field in the normal range and
    /// flushes `MASKED_SCORE` inputs to a harmless ~2^-125.
    #[inline(always)]
    fn exp2_approx(x: f32) -> f32 {
        const SHIFT: f32 = 12582912.0; // 1.5 * 2^23
        const C0: f32 = 0.999_928_07;
        const C1: f32 = 0.693_260_99;
        const C2: f32 = 0.242_611_12;
        const C3: f32 = 0.055_171_67;
        let x = fmin(fmax(x, -125.0), 125.0);
        let shifted = x + SHIFT;
        let integer = (shifted.to_bits() as i32).wrapping_sub(0x4b40_0000);
        let fraction = x - (shifted - SHIFT);
        let poly = C0 + fraction * (C1 + fraction * (C2 + fraction * C3));
        f32::from_bits((poly.to_bits() as i32).wrapping_add(integer << 23) as u32)
    }

    /// `log2(x)` for positive normal `x`: exponent extraction, mantissa
    /// renormalized to `[√½, √2]`, then the atanh series in
    /// `t = (m-1)/(m+1)` (four terms; |error| < 5e-8 on the reduced range).
    #[inline(always)]
    fn log2_approx(x: f32) -> f32 {
        const C0: f32 = 2.885_390_1;
        const C1: f32 = 0.961_796_7;
        const C2: f32 = 0.577_078_02;
        const C3: f32 = 0.412_198_58;
        let bits = x.to_bits();
        let mut exponent = ((bits >> 23) as i32) - 127;
        let mut mantissa = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
        if mantissa > 1.414_213_6 {
            mantissa *= 0.5;
            exponent += 1;
        }
        let t = (mantissa - 1.0) / (mantissa + 1.0);
        let t2 = t * t;
        exponent as f32 + t * (C0 + t2 * (C1 + t2 * (C2 + t2 * C3)))
    }

    #[inline(always)]
    fn quad_max(value: f32) -> f32 {
        let value = fmax(value, warp::shuffle_xor_f32(value, 1));
        fmax(value, warp::shuffle_xor_f32(value, 2))
    }

    #[inline(always)]
    fn quad_sum(value: f32) -> f32 {
        let value = value + warp::shuffle_xor_f32(value, 1);
        value + warp::shuffle_xor_f32(value, 2)
    }

    /// One key tile of register softmax, shared by the three forward
    /// kernels. `warp_id` and every row coordinate are warpgroup-local: the
    /// persistent kernel's second warpgroup passes `warp::warp_id() - 4`.
    ///
    /// Drains `S[128, 128]` from `s_tmem` twice — pass 1 for masked row
    /// maxima, pass 2 to exponentiate against the O segment's per-row max
    /// reference `m_ref` — and stores bf16 probabilities into the two
    /// stacked SWIZZLE_128B P subtiles via `stmatrix` (the per-row
    /// addresses apply the 16-byte-chunk XOR the TMA swizzle would have
    /// produced, folding in the tile base's absolute 128-byte row phase, so
    /// the O-MMA descriptors read P exactly like a TMA-loaded operand).
    ///
    /// Between the passes sits FA4's conditional correction, adapted to the
    /// missing `tcgen05.st`: the O TMEM segment keeps accumulating under
    /// `m_ref` (the caller issues `O = P·V` with `enable_d` set) until some
    /// row's tile max exceeds `m_ref + CORRECTION_THRESHOLD`. A segment
    /// restart is collective — one row over the line forces every row — so
    /// the decision is a per-warp `vote.any` published in
    /// `votes[(tile & 1) * 4 + warp]` and made warpgroup-wide by one
    /// 128-count phase of `vote_barrier`. On a correction the warpgroup
    /// drains the segment (tiles `.. tile`, complete because the caller
    /// waited out the previous tile's O MMA) into `out_acc`, rescales
    /// `out_acc`/`running_sum` to the new reference, and returns `true` so
    /// the caller restarts the segment on this tile's O MMA. Tile 0 always
    /// votes yes — it starts the first segment — but skips the drain.
    ///
    /// Per-thread fragment ownership (base-LDTM 16x256b): for each 16-row
    /// half-warp block this thread owns rows `lane/4` and `lane/4 + 8`,
    /// columns `2*(lane%4)` and `+1` of each 8-column half. Row statistics
    /// live once per owned row, replicated across the 4 lanes of a quad by
    /// shuffle reductions.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn softmax_tile(
        s_tmem: u32,
        o_tmem: u32,
        tile: u32,
        diagonal: bool,
        warp_id: u32,
        lane: u32,
        p_smem: *mut u8,
        p_phase: usize,
        votes: *mut u32,
        vote_barrier: *const Barrier,
        m_ref: &mut [f32; 4],
        running_sum: &mut [f32; 4],
        out_acc: &mut [[f32; 16]; 4],
    ) -> bool {
        unsafe {
            let quad = (lane % 4) as usize;
            let row_in_16 = (lane / 4) as usize;

            // Pass 1: tile row maxima (masked, base-2 domain).
            let mut tile_max = [MASKED_SCORE; 4];
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 8 {
                    let column = column_block * 16;
                    let low = tcgen05_ld_16x256b_pure(s_tmem + (tmem_row << 16) + column);
                    tcgen05_load_wait();
                    let high = tcgen05_ld_16x256b_pure(s_tmem + (tmem_row << 16) + column + 8);
                    tcgen05_load_wait();
                    let slot_a = (row_block * 2) as usize;
                    let slot_b = slot_a + 1;
                    if diagonal {
                        let row_a = (tmem_row as usize + row_in_16) as u32;
                        let row_b = row_a + 8;
                        let col = column + 2 * quad as u32;
                        let mut max_a = tile_max[slot_a];
                        max_a = fmax(max_a, if col <= row_a { low[0] } else { MASKED_SCORE });
                        max_a =
                            fmax(max_a, if col + 1 <= row_a { low[1] } else { MASKED_SCORE });
                        max_a =
                            fmax(max_a, if col + 8 <= row_a { high[0] } else { MASKED_SCORE });
                        max_a =
                            fmax(max_a, if col + 9 <= row_a { high[1] } else { MASKED_SCORE });
                        tile_max[slot_a] = max_a;
                        let mut max_b = tile_max[slot_b];
                        max_b = fmax(max_b, if col <= row_b { low[2] } else { MASKED_SCORE });
                        max_b =
                            fmax(max_b, if col + 1 <= row_b { low[3] } else { MASKED_SCORE });
                        max_b =
                            fmax(max_b, if col + 8 <= row_b { high[2] } else { MASKED_SCORE });
                        max_b =
                            fmax(max_b, if col + 9 <= row_b { high[3] } else { MASKED_SCORE });
                        tile_max[slot_b] = max_b;
                    } else {
                        tile_max[slot_a] = fmax(
                            fmax(fmax(tile_max[slot_a], low[0]), fmax(low[1], high[0])),
                            high[1],
                        );
                        tile_max[slot_b] = fmax(
                            fmax(fmax(tile_max[slot_b], low[2]), fmax(low[3], high[2])),
                            high[3],
                        );
                    }
                    column_block += 1;
                }
                row_block += 1;
            }
            let mut row_max = [0.0f32; 4];
            let mut exceed = false;
            let mut slot = 0usize;
            while slot < 4 {
                row_max[slot] = quad_max(tile_max[slot]);
                exceed = exceed || row_max[slot] > m_ref[slot] + CORRECTION_THRESHOLD;
                slot += 1;
            }

            // Collective correction vote (tile 0 always trips it: m_ref
            // still sits at MASKED_SCORE). One word per warp, one barrier
            // phase per tile.
            let parity = tile & 1;
            let warp_vote = warp::any(exceed);
            if lane == 0 {
                *votes.add((parity * 4 + warp_id) as usize) = warp_vote as u32;
            }
            mbarrier_arrive(vote_barrier);
            while !mbarrier_try_wait_parity(vote_barrier, parity) {}
            let base = (parity * 4) as usize;
            let correction = (*votes.add(base)
                | *votes.add(base + 1)
                | *votes.add(base + 2)
                | *votes.add(base + 3))
                != 0;

            if correction {
                if tile > 0 {
                    merge_output_tile(o_tmem, warp_id, out_acc);
                }
                let mut slot = 0usize;
                while slot < 4 {
                    let next = fmax(m_ref[slot], row_max[slot]);
                    let factor = exp2_approx(m_ref[slot] - next);
                    m_ref[slot] = next;
                    running_sum[slot] *= factor;
                    let mut value = 0usize;
                    while value < 16 {
                        out_acc[slot][value] *= factor;
                        value += 1;
                    }
                    slot += 1;
                }
            }

            // Pass 2: probabilities — re-drain S, exponentiate against the
            // segment reference, accumulate row sums, and store bf16 P
            // through the swizzle-aware stmatrix addresses.
            let mut tile_sum = [0.0f32; 4];
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 8 {
                    let column = column_block * 16;
                    let low = tcgen05_ld_16x256b_pure(s_tmem + (tmem_row << 16) + column);
                    tcgen05_load_wait();
                    let high = tcgen05_ld_16x256b_pure(s_tmem + (tmem_row << 16) + column + 8);
                    tcgen05_load_wait();
                    let slot_a = (row_block * 2) as usize;
                    let slot_b = slot_a + 1;
                    let row_a = (tmem_row as usize + row_in_16) as u32;
                    let row_b = row_a + 8;
                    let col = column + 2 * quad as u32;

                    let s_a0 = if !diagonal || col <= row_a { low[0] } else { MASKED_SCORE };
                    let s_a1 = if !diagonal || col + 1 <= row_a { low[1] } else { MASKED_SCORE };
                    let s_a8 = if !diagonal || col + 8 <= row_a { high[0] } else { MASKED_SCORE };
                    let s_a9 = if !diagonal || col + 9 <= row_a { high[1] } else { MASKED_SCORE };
                    let s_b0 = if !diagonal || col <= row_b { low[2] } else { MASKED_SCORE };
                    let s_b1 = if !diagonal || col + 1 <= row_b { low[3] } else { MASKED_SCORE };
                    let s_b8 = if !diagonal || col + 8 <= row_b { high[2] } else { MASKED_SCORE };
                    let s_b9 = if !diagonal || col + 9 <= row_b { high[3] } else { MASKED_SCORE };
                    let p_a0 = exp2_approx(s_a0 - m_ref[slot_a]);
                    let p_a1 = exp2_approx(s_a1 - m_ref[slot_a]);
                    let p_a8 = exp2_approx(s_a8 - m_ref[slot_a]);
                    let p_a9 = exp2_approx(s_a9 - m_ref[slot_a]);
                    let p_b0 = exp2_approx(s_b0 - m_ref[slot_b]);
                    let p_b1 = exp2_approx(s_b1 - m_ref[slot_b]);
                    let p_b8 = exp2_approx(s_b8 - m_ref[slot_b]);
                    let p_b9 = exp2_approx(s_b9 - m_ref[slot_b]);
                    tile_sum[slot_a] += p_a0 + p_a1 + p_a8 + p_a9;
                    tile_sum[slot_b] += p_b0 + p_b1 + p_b8 + p_b9;

                    let subtile = (column_block / 4) as usize * TILE_BYTES;
                    let chunk_low = ((column_block % 4) * 2) as usize;
                    let chunk = if (8..16).contains(&lane) { chunk_low + 1 } else { chunk_low };
                    let row_low = tmem_row as usize + (lane % 8) as usize;
                    let row_high = row_low + 8;
                    let address_low = p_smem
                        .add(subtile + row_low * 128 + (chunk ^ ((row_low + p_phase) & 7)) * 16);
                    let address_high = p_smem.add(
                        subtile + row_high * 128 + (chunk ^ ((row_high + p_phase) & 7)) * 16,
                    );
                    stmatrix_m8n8_x2(
                        address_low,
                        cvt_f32x2_bf16x2(p_a0, p_a1),
                        cvt_f32x2_bf16x2(p_a8, p_a9),
                    );
                    stmatrix_m8n8_x2(
                        address_high,
                        cvt_f32x2_bf16x2(p_b0, p_b1),
                        cvt_f32x2_bf16x2(p_b8, p_b9),
                    );
                    column_block += 1;
                }
                row_block += 1;
            }
            let mut slot = 0usize;
            while slot < 4 {
                running_sum[slot] += quad_sum(tile_sum[slot]);
                slot += 1;
            }
            correction
        }
    }

    /// Drain the `O = P·V` TMEM segment and add it into the per-thread
    /// output accumulator. Called by `softmax_tile` on a correction (which
    /// then rescales the merged accumulator to the new reference) and by
    /// the epilogue for the final segment. `warp_id` is warpgroup-local.
    #[inline(always)]
    unsafe fn merge_output_tile(o_tmem: u32, warp_id: u32, out_acc: &mut [[f32; 16]; 4]) {
        unsafe {
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 4 {
                    let column = column_block * 16;
                    let low = tcgen05_ld_16x256b_pure(o_tmem + (tmem_row << 16) + column);
                    tcgen05_load_wait();
                    let high = tcgen05_ld_16x256b_pure(o_tmem + (tmem_row << 16) + column + 8);
                    tcgen05_load_wait();
                    let slot_a = (row_block * 2) as usize;
                    let slot_b = slot_a + 1;
                    let base = (column_block * 4) as usize;
                    out_acc[slot_a][base] += low[0];
                    out_acc[slot_a][base + 1] += low[1];
                    out_acc[slot_a][base + 2] += high[0];
                    out_acc[slot_a][base + 3] += high[1];
                    out_acc[slot_b][base] += low[2];
                    out_acc[slot_b][base + 1] += low[3];
                    out_acc[slot_b][base + 2] += high[2];
                    out_acc[slot_b][base + 3] += high[3];
                    column_block += 1;
                }
                row_block += 1;
            }
        }
    }

    /// Normalize the accumulated output and store fp32 `y` plus the
    /// natural-log LSE straight to global memory through the fragment map.
    /// `max_ref` is whatever per-row reference `running_sum` is relative to
    /// (the final segment's `m_ref`), so the LSE is exact even when the
    /// reference trails the true row max by up to the correction threshold.
    /// `warp_id` is warpgroup-local.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn store_outputs(
        batch: u32,
        t: u32,
        h: u32,
        head: u32,
        query_tile: u32,
        warp_id: u32,
        lane: u32,
        max_ref: &[f32; 4],
        running_sum: &[f32; 4],
        out_acc: &[[f32; 16]; 4],
        output: &mut DisjointSlice<f32>,
        logsumexp: &mut DisjointSlice<f32>,
    ) {
        unsafe {
            let quad = (lane % 4) as usize;
            let row_in_16 = (lane / 4) as usize;
            let d_model = (h as usize) * HD;
            let mut slot = 0usize;
            while slot < 4 {
                let local_row =
                    warp_id as usize * 32 + (slot / 2) * 16 + (slot % 2) * 8 + row_in_16;
                let global_row =
                    (batch * t) as usize + query_tile as usize * TILE + local_row;
                let inverse = 1.0 / running_sum[slot];
                let out_base = global_row * d_model + head as usize * HD;
                let mut column_block = 0usize;
                while column_block < 4 {
                    let column = column_block * 16 + 2 * quad;
                    let base = column_block * 4;
                    *output.get_unchecked_mut(out_base + column) = out_acc[slot][base] * inverse;
                    *output.get_unchecked_mut(out_base + column + 1) =
                        out_acc[slot][base + 1] * inverse;
                    *output.get_unchecked_mut(out_base + column + 8) =
                        out_acc[slot][base + 2] * inverse;
                    *output.get_unchecked_mut(out_base + column + 9) =
                        out_acc[slot][base + 3] * inverse;
                    column_block += 1;
                }
                if quad == 0 {
                    *logsumexp.get_unchecked_mut(global_row * h as usize + head as usize) =
                        LN2 * (max_ref[slot] + log2_approx(running_sum[slot]));
                }
                slot += 1;
            }
        }
    }

    /// Issue one `S = Q·Kᵀ` tile (M128_N128, four chained K=16 MMAs) from
    /// the current leader thread; the caller owns the commit.
    #[inline(always)]
    unsafe fn score_mma(s_tmem: u32, q_smem: *mut u8, k_smem: *mut u8, s_instruction: u32) {
        unsafe {
            let mut chunk = 0u64;
            while chunk < 4 {
                let a_descriptor = smem_descriptor(q_smem as u64 + chunk * 32);
                let b_descriptor = smem_descriptor(k_smem as u64 + chunk * 32);
                tcgen05_mma_f16(s_tmem, a_descriptor, b_descriptor, s_instruction, chunk > 0);
                chunk += 1;
            }
        }
    }

    /// Store one thread's eight-value bf16 fragment into a stacked
    /// SWIZZLE_128B `[128, 128]` subtile pair, exactly like `softmax_tile`'s
    /// pass-2 P-write: the two `stmatrix` targets carry the 16-byte-chunk XOR
    /// the TMA swizzle would produce, folding in the tile base's absolute
    /// 128-byte row phase, so the accumulating MMA reads the operand like a
    /// TMA-loaded tile. Shared by both backward kernels (dS for kernel A, Pᵀ
    /// and dSᵀ for kernel B).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn write_bf16_fragment(
        smem: *mut u8,
        phase: usize,
        tmem_row: u32,
        column_block: u32,
        lane: u32,
        a0: f32,
        a1: f32,
        a8: f32,
        a9: f32,
        b0: f32,
        b1: f32,
        b8: f32,
        b9: f32,
    ) {
        unsafe {
            let subtile = (column_block / 4) as usize * TILE_BYTES;
            let chunk_low = ((column_block % 4) * 2) as usize;
            let chunk = if (8..16).contains(&lane) { chunk_low + 1 } else { chunk_low };
            let row_low = tmem_row as usize + (lane % 8) as usize;
            let row_high = row_low + 8;
            let address_low =
                smem.add(subtile + row_low * 128 + (chunk ^ ((row_low + phase) & 7)) * 16);
            let address_high =
                smem.add(subtile + row_high * 128 + (chunk ^ ((row_high + phase) & 7)) * 16);
            stmatrix_m8n8_x2(address_low, cvt_f32x2_bf16x2(a0, a1), cvt_f32x2_bf16x2(a8, a9));
            stmatrix_m8n8_x2(address_high, cvt_f32x2_bf16x2(b0, b1), cvt_f32x2_bf16x2(b8, b9));
        }
    }

    /// True `dS = P·(dP − D)·scale` for one score element, base-2 domain.
    /// `s` is the staged pre-scaled score (`scale·log2e·(q·k)`) so the
    /// probability is `exp2(s − lse2)`; `keep` is false only on masked
    /// diagonal positions, where the gradient is a literal zero rather than
    /// `exp2` of `MASKED_SCORE`. `factor` folds the operand scaling the MMA
    /// leaves for the caller (`scale` for dQ against unscaled K; `ln2` for dK
    /// against the pre-scaled Q, since `ln2·scale·log2e = scale`).
    #[inline(always)]
    fn backward_dscore(s: f32, dp: f32, lse2: f32, dot: f32, factor: f32, keep: bool) -> f32 {
        if keep {
            exp2_approx(s - lse2) * (dp - dot) * factor
        } else {
            0.0
        }
    }

    /// One key tile of the query-parallel backward register pass (kernel A):
    /// drain `S` and `dP` from TMEM through the 16x256b fragment map,
    /// recompute `P = exp2(s − lse2[row])`, form `dS = P·(dP − dot[row])·scale`
    /// (diagonal positions past the causal edge are zero), and store the bf16
    /// dS into the two stacked SWIZZLE_128B subtiles. `lse2`/`dot` are the
    /// query tile's per-row statistics staged in shared memory; rows are query
    /// rows, so both index by the fragment's row coordinate.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn backward_q_tile(
        s_tmem: u32,
        dp_tmem: u32,
        diagonal: bool,
        warp_id: u32,
        lane: u32,
        lse2: *const f32,
        dot: *const f32,
        ds_smem: *mut u8,
        ds_phase: usize,
    ) {
        unsafe {
            let quad = (lane % 4) as usize;
            let row_in_16 = (lane / 4) as usize;
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 8 {
                    let column = column_block * 16;
                    let s_low = tcgen05_ld_16x256b_pure(s_tmem + (tmem_row << 16) + column);
                    tcgen05_load_wait();
                    let s_high = tcgen05_ld_16x256b_pure(s_tmem + (tmem_row << 16) + column + 8);
                    tcgen05_load_wait();
                    let dp_low = tcgen05_ld_16x256b_pure(dp_tmem + (tmem_row << 16) + column);
                    tcgen05_load_wait();
                    let dp_high = tcgen05_ld_16x256b_pure(dp_tmem + (tmem_row << 16) + column + 8);
                    tcgen05_load_wait();
                    let row_a = tmem_row + row_in_16 as u32;
                    let row_b = row_a + 8;
                    let col = column + 2 * quad as u32;
                    let lse_a = *lse2.add(row_a as usize);
                    let dot_a = *dot.add(row_a as usize);
                    let lse_b = *lse2.add(row_b as usize);
                    let dot_b = *dot.add(row_b as usize);

                    let ds_a0 =
                        backward_dscore(s_low[0], dp_low[0], lse_a, dot_a, SCALE, !diagonal || col <= row_a);
                    let ds_a1 = backward_dscore(
                        s_low[1], dp_low[1], lse_a, dot_a, SCALE, !diagonal || col + 1 <= row_a,
                    );
                    let ds_a8 = backward_dscore(
                        s_high[0], dp_high[0], lse_a, dot_a, SCALE, !diagonal || col + 8 <= row_a,
                    );
                    let ds_a9 = backward_dscore(
                        s_high[1], dp_high[1], lse_a, dot_a, SCALE, !diagonal || col + 9 <= row_a,
                    );
                    let ds_b0 =
                        backward_dscore(s_low[2], dp_low[2], lse_b, dot_b, SCALE, !diagonal || col <= row_b);
                    let ds_b1 = backward_dscore(
                        s_low[3], dp_low[3], lse_b, dot_b, SCALE, !diagonal || col + 1 <= row_b,
                    );
                    let ds_b8 = backward_dscore(
                        s_high[2], dp_high[2], lse_b, dot_b, SCALE, !diagonal || col + 8 <= row_b,
                    );
                    let ds_b9 = backward_dscore(
                        s_high[3], dp_high[3], lse_b, dot_b, SCALE, !diagonal || col + 9 <= row_b,
                    );
                    write_bf16_fragment(
                        ds_smem, ds_phase, tmem_row, column_block, lane, ds_a0, ds_a1, ds_a8,
                        ds_a9, ds_b0, ds_b1, ds_b8, ds_b9,
                    );
                    column_block += 1;
                }
                row_block += 1;
            }
        }
    }

    /// One query tile of the key-parallel backward register pass (kernel B):
    /// drain the transposed `Sᵀ` and `dPᵀ` from TMEM, recompute the
    /// transposed probabilities `Pᵀ = exp2(sᵀ − lse2[col])` and
    /// `dSᵀ = Pᵀ·(dPᵀ − dot[col])·ln2`, and store both into their own stacked
    /// subtile pairs. Rows are key rows and columns are query rows, so the
    /// statistics index by the fragment's *column* coordinate and the causal
    /// mask zeroes positions where the key row leads the query column.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn backward_kv_tile(
        st_tmem: u32,
        dpt_tmem: u32,
        diagonal: bool,
        warp_id: u32,
        lane: u32,
        lse2: *const f32,
        dot: *const f32,
        p_smem: *mut u8,
        p_phase: usize,
        ds_smem: *mut u8,
        ds_phase: usize,
    ) {
        unsafe {
            let quad = (lane % 4) as usize;
            let row_in_16 = (lane / 4) as usize;
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 8 {
                    let column = column_block * 16;
                    let s_low = tcgen05_ld_16x256b_pure(st_tmem + (tmem_row << 16) + column);
                    tcgen05_load_wait();
                    let s_high = tcgen05_ld_16x256b_pure(st_tmem + (tmem_row << 16) + column + 8);
                    tcgen05_load_wait();
                    let dp_low = tcgen05_ld_16x256b_pure(dpt_tmem + (tmem_row << 16) + column);
                    tcgen05_load_wait();
                    let dp_high = tcgen05_ld_16x256b_pure(dpt_tmem + (tmem_row << 16) + column + 8);
                    tcgen05_load_wait();
                    let row_a = tmem_row + row_in_16 as u32;
                    let row_b = row_a + 8;
                    let col = column + 2 * quad as u32;
                    let lse_c0 = *lse2.add(col as usize);
                    let dot_c0 = *dot.add(col as usize);
                    let lse_c1 = *lse2.add(col as usize + 1);
                    let dot_c1 = *dot.add(col as usize + 1);
                    let lse_c8 = *lse2.add(col as usize + 8);
                    let dot_c8 = *dot.add(col as usize + 8);
                    let lse_c9 = *lse2.add(col as usize + 9);
                    let dot_c9 = *dot.add(col as usize + 9);

                    let p_a0 = if !diagonal || row_a <= col { exp2_approx(s_low[0] - lse_c0) } else { 0.0 };
                    let p_a1 =
                        if !diagonal || row_a <= col + 1 { exp2_approx(s_low[1] - lse_c1) } else { 0.0 };
                    let p_a8 =
                        if !diagonal || row_a <= col + 8 { exp2_approx(s_high[0] - lse_c8) } else { 0.0 };
                    let p_a9 =
                        if !diagonal || row_a <= col + 9 { exp2_approx(s_high[1] - lse_c9) } else { 0.0 };
                    let p_b0 = if !diagonal || row_b <= col { exp2_approx(s_low[2] - lse_c0) } else { 0.0 };
                    let p_b1 =
                        if !diagonal || row_b <= col + 1 { exp2_approx(s_low[3] - lse_c1) } else { 0.0 };
                    let p_b8 =
                        if !diagonal || row_b <= col + 8 { exp2_approx(s_high[2] - lse_c8) } else { 0.0 };
                    let p_b9 =
                        if !diagonal || row_b <= col + 9 { exp2_approx(s_high[3] - lse_c9) } else { 0.0 };

                    let ds_a0 = p_a0 * (dp_low[0] - dot_c0) * LN2;
                    let ds_a1 = p_a1 * (dp_low[1] - dot_c1) * LN2;
                    let ds_a8 = p_a8 * (dp_high[0] - dot_c8) * LN2;
                    let ds_a9 = p_a9 * (dp_high[1] - dot_c9) * LN2;
                    let ds_b0 = p_b0 * (dp_low[2] - dot_c0) * LN2;
                    let ds_b1 = p_b1 * (dp_low[3] - dot_c1) * LN2;
                    let ds_b8 = p_b8 * (dp_high[2] - dot_c8) * LN2;
                    let ds_b9 = p_b9 * (dp_high[3] - dot_c9) * LN2;

                    write_bf16_fragment(
                        p_smem, p_phase, tmem_row, column_block, lane, p_a0, p_a1, p_a8, p_a9,
                        p_b0, p_b1, p_b8, p_b9,
                    );
                    write_bf16_fragment(
                        ds_smem, ds_phase, tmem_row, column_block, lane, ds_a0, ds_a1, ds_a8,
                        ds_a9, ds_b0, ds_b1, ds_b8, ds_b9,
                    );
                    column_block += 1;
                }
                row_block += 1;
            }
        }
    }

    /// Issue one `dQ/dK/dV += A·B` gradient tile from the leader thread: the
    /// forward O-MMA shape (M128_N64, transpose_b, eight chained K=16 chunks)
    /// with `A` walking two stacked bf16 subtiles and `B` walking a key/query
    /// operand by 2048-byte rows. `fresh` starts a new TMEM accumulator (the
    /// block's first visited tile); otherwise it accumulates with `enable_d`.
    #[inline(always)]
    unsafe fn grad_mma(acc_tmem: u32, a_smem: *mut u8, b_smem: *mut u8, instruction: u32, fresh: bool) {
        unsafe {
            let mut chunk = 0u64;
            while chunk < 8 {
                let a_descriptor =
                    smem_descriptor(a_smem as u64 + (chunk / 4) * TILE_BYTES as u64 + (chunk % 4) * 32);
                let b_descriptor = smem_descriptor(b_smem as u64 + chunk * 2048);
                tcgen05_mma_f16(acc_tmem, a_descriptor, b_descriptor, instruction, chunk > 0 || !fresh);
                chunk += 1;
            }
        }
    }

    /// Drain a 64-column gradient accumulator and store fp32 straight to
    /// global memory through the fragment map, at the block's `tile` rows.
    /// Like `store_outputs` minus the `1/sum` scale and the LSE write — the
    /// gradients are already complete sums.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn store_grad_tile(
        batch: u32,
        t: u32,
        h: u32,
        head: u32,
        tile: u32,
        warp_id: u32,
        lane: u32,
        grad_acc: &[[f32; 16]; 4],
        output: &mut DisjointSlice<f32>,
    ) {
        unsafe {
            let quad = (lane % 4) as usize;
            let row_in_16 = (lane / 4) as usize;
            let d_model = (h as usize) * HD;
            let mut slot = 0usize;
            while slot < 4 {
                let local_row =
                    warp_id as usize * 32 + (slot / 2) * 16 + (slot % 2) * 8 + row_in_16;
                let global_row = (batch * t) as usize + tile as usize * TILE + local_row;
                let out_base = global_row * d_model + head as usize * HD;
                let mut column_block = 0usize;
                while column_block < 4 {
                    let column = column_block * 16 + 2 * quad;
                    let base = column_block * 4;
                    *output.get_unchecked_mut(out_base + column) = grad_acc[slot][base];
                    *output.get_unchecked_mut(out_base + column + 1) = grad_acc[slot][base + 1];
                    *output.get_unchecked_mut(out_base + column + 8) = grad_acc[slot][base + 2];
                    *output.get_unchecked_mut(out_base + column + 9) = grad_acc[slot][base + 3];
                    column_block += 1;
                }
                slot += 1;
            }
        }
    }

    /// Elementwise `2^x` accuracy oracle for the standalone parity gate.
    #[kernel]
    pub fn software_exp2(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot = exp2_approx(input[i]);
        }
    }

    /// Elementwise `log2(x)` accuracy oracle for the standalone parity gate.
    #[kernel]
    pub fn software_log2(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot = log2_approx(input[i]);
        }
    }

    /// Dumps the raw shared-memory word layout of one TMA-loaded `[128, 64]`
    /// bf16 tile, plus the tile's absolute 128-byte row phase as a trailing
    /// word. The P-write path mirrors TMA's SWIZZLE_128B placement — which
    /// XORs *absolute* address bits [9:7] — with manual address XORs; the
    /// host fills the staging tile with sequential word indices and verifies
    /// the exact permutation from this dump.
    #[kernel]
    pub unsafe fn swizzle_probe(src_tma: *const TmaDescriptor, mut output: DisjointSlice<u32>) {
        unsafe {
            static mut TMA_BARRIER: Barrier = Barrier::UNINIT;

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tid = thread::threadIdx_x();
            if tid == 0 {
                mbarrier_init(&raw mut TMA_BARRIER, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if tid == 0 {
                cp_async_bulk_tensor_3d_g2s(smem, src_tma, 0, 0, 0, &raw mut TMA_BARRIER);
                mbarrier_arrive_expect_tx(&raw const TMA_BARRIER, 1, TILE_BYTES as u32);
            }
            while !mbarrier_try_wait_parity(&raw const TMA_BARRIER, 0) {}
            thread::sync_threads();

            let words = smem as *const u32;
            let mut index = tid as usize;
            while index < TILE_BYTES / 4 {
                *output.get_unchecked_mut(index) = *words.add(index);
                index += TILE;
            }
            if tid == 0 {
                *output.get_unchecked_mut(TILE_BYTES / 4) = ((smem as usize >> 7) & 7) as u32;
            }
            thread::sync_threads();
            if tid == 0 {
                mbarrier_inval(&raw mut TMA_BARRIER);
            }
        }
    }

    /// Validation kernel for the transposed-B operand path the `O = P·V` MMA
    /// depends on: one CTA computes `C[128, 64] = A[128, 128] · B[128, 64]`
    /// with `B` stored row-major `[K, N]` — the natural V-tile orientation —
    /// consumed through `transpose_b` instruction-descriptor bit plus
    /// 16-row (2048-byte) descriptor advances per K chunk.
    ///
    /// The epilogue stores each thread's fragment straight to global memory
    /// through the decoded (row, column) ownership map, so a failure here
    /// distinguishes descriptor problems (values transposed/permuted in
    /// blocks) from fragment-map problems (fine-grained scrambling).
    #[kernel]
    pub unsafe fn transpose_b_probe(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut output: DisjointSlice<f32>,
    ) {
        unsafe {
            static mut TMEM_ADDRESS: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut MMA_BARRIER: Barrier = Barrier::UNINIT;

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let a_smem = smem;
            let b_smem = smem.add(2 * TILE_BYTES);

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let is_leader = tid == 0;

            if is_leader {
                mbarrier_init(&raw mut TMA_BARRIER, 1);
                mbarrier_init(&raw mut MMA_BARRIER, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDRESS as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem = *(&raw const TMEM_ADDRESS as *const u32);

            if is_leader {
                cp_async_bulk_tensor_3d_g2s(a_smem, a_tma, 0, 0, 0, &raw mut TMA_BARRIER);
                cp_async_bulk_tensor_3d_g2s(
                    a_smem.add(TILE_BYTES),
                    a_tma,
                    0,
                    0,
                    1,
                    &raw mut TMA_BARRIER,
                );
                cp_async_bulk_tensor_3d_g2s(b_smem, b_tma, 0, 0, 0, &raw mut TMA_BARRIER);
                mbarrier_arrive_expect_tx(&raw const TMA_BARRIER, 1, (3 * TILE_BYTES) as u32);
            }
            while !mbarrier_try_wait_parity(&raw const TMA_BARRIER, 0) {}
            thread::sync_threads();

            let instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N64)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .transpose_b(true)
                .build()
                .raw();
            if is_leader {
                let mut chunk = 0u32;
                while chunk < 8 {
                    let a_descriptor = smem_descriptor(
                        a_smem as u64 + (chunk / 4) as u64 * TILE_BYTES as u64
                            + (chunk % 4) as u64 * 32,
                    );
                    let b_descriptor = smem_descriptor(b_smem as u64 + chunk as u64 * 2048);
                    tcgen05_mma_f16(tmem, a_descriptor, b_descriptor, instruction, chunk > 0);
                    chunk += 1;
                }
                tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
            }
            while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, 0) {}
            thread::sync_threads();

            let quad = (lane % 4) as usize;
            let row_in_16 = (lane / 4) as usize;
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 4 {
                    let column = (column_block * 16) as usize;
                    let low = tcgen05_ld_16x256b_pure(tmem + (tmem_row << 16) + column as u32);
                    tcgen05_load_wait();
                    let high =
                        tcgen05_ld_16x256b_pure(tmem + (tmem_row << 16) + column as u32 + 8);
                    tcgen05_load_wait();
                    let row_a = tmem_row as usize + row_in_16;
                    let row_b = row_a + 8;
                    let col = column + 2 * quad;
                    *output.get_unchecked_mut(row_a * HD + col) = low[0];
                    *output.get_unchecked_mut(row_a * HD + col + 1) = low[1];
                    *output.get_unchecked_mut(row_b * HD + col) = low[2];
                    *output.get_unchecked_mut(row_b * HD + col + 1) = low[3];
                    *output.get_unchecked_mut(row_a * HD + col + 8) = high[0];
                    *output.get_unchecked_mut(row_a * HD + col + 9) = high[1];
                    *output.get_unchecked_mut(row_b * HD + col + 8) = high[2];
                    *output.get_unchecked_mut(row_b * HD + col + 9) = high[3];
                    column_block += 1;
                }
                row_block += 1;
            }

            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem, 512);
            }
            if is_leader {
                mbarrier_inval(&raw mut TMA_BARRIER);
                mbarrier_inval(&raw mut MMA_BARRIER);
            }
        }
    }

    /// Synchronous tcgen05 causal attention forward. Launch with
    /// `host::flash_forward_config`: grid `(T/128, H, B)`, 128 threads,
    /// `FLASH_DYNAMIC_SMEM` dynamic shared bytes (opted in by the loader).
    /// `correction_counts` gets one word per CTA (`plane * tiles +
    /// query_tile`): how many mid-stream O-segment corrections the key
    /// stream triggered.
    #[kernel]
    pub unsafe fn flash_forward_tcgen05(
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        sequence_length: u32,
        heads: u32,
        mut output: DisjointSlice<f32>,
        mut logsumexp: DisjointSlice<f32>,
        mut correction_counts: DisjointSlice<u32>,
    ) {
        unsafe {
            static mut TMEM_ADDRESS: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut MMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut VOTE_BARRIER: Barrier = Barrier::UNINIT;
            static mut VOTES: SharedArray<u32, 8, 4> = SharedArray::UNINIT;

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let q_smem = smem;
            let k_smem = smem.add(TILE_BYTES);
            let v_smem = smem.add(2 * TILE_BYTES);
            let p_smem = smem.add(3 * TILE_BYTES);

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != TILE {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let is_leader = tid == 0;

            let query_tile = thread::blockIdx_x();
            let head = thread::blockIdx_y();
            let batch = thread::blockIdx_z();
            let t = sequence_length;
            let h = heads;
            let plane = (batch * h + head) as i32;

            if is_leader {
                mbarrier_init(&raw mut TMA_BARRIER, 1);
                mbarrier_init(&raw mut MMA_BARRIER, 1);
                mbarrier_init(&raw mut VOTE_BARRIER, TILE as u32);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDRESS as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem = *(&raw const TMEM_ADDRESS as *const u32);
            let s_tmem = tmem;
            let o_tmem = tmem + 256;

            let s_instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let o_instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N64)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .transpose_b(true)
                .build()
                .raw();

            // Q stays operand-A resident for the whole key stream.
            if is_leader {
                cp_async_bulk_tensor_3d_g2s(
                    q_smem,
                    q_tma,
                    0,
                    (query_tile * TILE as u32) as i32,
                    plane,
                    &raw mut TMA_BARRIER,
                );
                mbarrier_arrive_expect_tx(&raw const TMA_BARRIER, 1, TILE_BYTES as u32);
            }
            while !mbarrier_try_wait_parity(&raw const TMA_BARRIER, 0) {}
            thread::sync_threads();

            let mut m_ref = [MASKED_SCORE; 4];
            let mut running_sum = [0.0f32; 4];
            let mut out_acc = [[0.0f32; 16]; 4];
            let mut corrections = 0u32;

            // The 128B swizzle XORs *absolute* shared-address bits [9:7], not
            // tile-relative rows. Dynamic shared memory starts just past the
            // static barrier words, so the P tile's base row phase is
            // nonzero; fold it into the manual stmatrix swizzle (the P
            // subtiles are a whole number of 8-row groups apart, so one
            // phase serves both).
            let p_phase = (p_smem as usize >> 7) & 7;

            let mut tma_phase = 1u32;
            let mut mma_phase = 0u32;
            let mut key_tile = 0u32;
            while key_tile <= query_tile {
                if is_leader {
                    let key_row = (key_tile * TILE as u32) as i32;
                    cp_async_bulk_tensor_3d_g2s(
                        k_smem,
                        k_tma,
                        0,
                        key_row,
                        plane,
                        &raw mut TMA_BARRIER,
                    );
                    cp_async_bulk_tensor_3d_g2s(
                        v_smem,
                        v_tma,
                        0,
                        key_row,
                        plane,
                        &raw mut TMA_BARRIER,
                    );
                    mbarrier_arrive_expect_tx(&raw const TMA_BARRIER, 1, 2 * TILE_BYTES as u32);
                }
                while !mbarrier_try_wait_parity(&raw const TMA_BARRIER, tma_phase & 1) {}
                tma_phase += 1;
                thread::sync_threads();

                // S = Q·Kᵀ, fresh accumulation each tile.
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    score_mma(s_tmem, q_smem, k_smem, s_instruction);
                    tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
                }
                while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, mma_phase & 1) {}
                mma_phase += 1;
                thread::sync_threads();

                let correction = softmax_tile(
                    s_tmem,
                    o_tmem,
                    key_tile,
                    key_tile == query_tile,
                    warp_id,
                    lane,
                    p_smem,
                    p_phase,
                    &raw mut VOTES as *mut u32,
                    &raw const VOTE_BARRIER,
                    &mut m_ref,
                    &mut running_sum,
                    &mut out_acc,
                );
                if key_tile > 0 && correction {
                    corrections += 1;
                }

                // P was written through the generic proxy; fence before the
                // async-proxy MMA consumes it.
                fence_proxy_async_shared_cta();
                tcgen05_fence_before_thread_sync();
                thread::sync_threads();

                // O = P·V: continue the TMEM segment, or restart it when
                // this tile's vote drained it (`correction` is uniform
                // across the block, so the leader's copy is the vote).
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    let mut chunk = 0u64;
                    while chunk < 8 {
                        let a_descriptor = smem_descriptor(
                            p_smem as u64 + (chunk / 4) * TILE_BYTES as u64 + (chunk % 4) * 32,
                        );
                        let b_descriptor = smem_descriptor(v_smem as u64 + chunk * 2048);
                        tcgen05_mma_f16(
                            o_tmem,
                            a_descriptor,
                            b_descriptor,
                            o_instruction,
                            chunk > 0 || !correction,
                        );
                        chunk += 1;
                    }
                    tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
                }
                while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, mma_phase & 1) {}
                mma_phase += 1;

                tcgen05_fence_before_thread_sync();
                thread::sync_threads();
                key_tile += 1;
            }

            merge_output_tile(o_tmem, warp_id, &mut out_acc);

            store_outputs(
                batch,
                t,
                h,
                head,
                query_tile,
                warp_id,
                lane,
                &m_ref,
                &running_sum,
                &out_acc,
                &mut output,
                &mut logsumexp,
            );
            if is_leader {
                let tiles = t as usize / TILE;
                *correction_counts
                    .get_unchecked_mut(plane as usize * tiles + query_tile as usize) =
                    corrections;
            }

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem, 512);
            }
            if is_leader {
                mbarrier_inval(&raw mut TMA_BARRIER);
                mbarrier_inval(&raw mut MMA_BARRIER);
                mbarrier_inval(&raw mut VOTE_BARRIER);
            }
        }
    }

    /// Synchronous tcgen05 query-parallel backward (issue #35, phase 4,
    /// kernel A). Launch with `host::flash_backward_q_config`: grid
    /// `(T/128, H, B)`, 128 threads, `FLASH_BACKWARD_Q_SMEM` dynamic shared
    /// bytes (opted in by the loader).
    ///
    /// One CTA owns a 128-query tile of one `(batch, head)` and streams the
    /// causal key tiles `0..=query_tile`. Q and dY stay resident; per key
    /// tile it recomputes `S = Q·Kᵀ` and `dP = dY·Vᵀ` into fp32 TMEM, forms
    /// the true `dS = P·(dP − D)·scale` in registers (§ `backward_q_tile`),
    /// stores bf16 dS through the swizzle-aware path, and accumulates
    /// `dQ += dS·K` in a TMEM segment via the transposed-B O-MMA shape (K
    /// staged unscaled, so `scale` is folded into the bf16 dS). The saved LSE
    /// is the normalizer — no running-max machinery — and `dot` is the
    /// pre-staged `Σ dy·y` per query row. `dq` is written directly from the
    /// register accumulator; blocks own disjoint query tiles, so no atomics.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub unsafe fn flash_backward_q_tcgen05(
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        logsumexp: &[f32],
        dot: &[f32],
        sequence_length: u32,
        heads: u32,
        mut dq: DisjointSlice<f32>,
    ) {
        unsafe {
            static mut TMEM_ADDRESS: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut MMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut LSE2: SharedArray<f32, 128> = SharedArray::UNINIT;
            static mut DOTS: SharedArray<f32, 128> = SharedArray::UNINIT;

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let q_smem = smem;
            let dy_smem = smem.add(TILE_BYTES);
            let k_smem = smem.add(2 * TILE_BYTES);
            let v_smem = smem.add(3 * TILE_BYTES);
            let ds_smem = smem.add(4 * TILE_BYTES);

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != TILE {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let is_leader = tid == 0;

            let query_tile = thread::blockIdx_x();
            let head = thread::blockIdx_y();
            let batch = thread::blockIdx_z();
            let t = sequence_length;
            let h = heads;
            let plane = (batch * h + head) as i32;

            if is_leader {
                mbarrier_init(&raw mut TMA_BARRIER, 1);
                mbarrier_init(&raw mut MMA_BARRIER, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDRESS as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem = *(&raw const TMEM_ADDRESS as *const u32);
            let s_tmem = tmem;
            let dp_tmem = tmem + 128;
            let dq_tmem = tmem + 256;

            let s_instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let grad_instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N64)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .transpose_b(true)
                .build()
                .raw();

            // Q and dY stay operand-A resident for the whole key stream.
            if is_leader {
                cp_async_bulk_tensor_3d_g2s(
                    q_smem,
                    q_tma,
                    0,
                    (query_tile * TILE as u32) as i32,
                    plane,
                    &raw mut TMA_BARRIER,
                );
                cp_async_bulk_tensor_3d_g2s(
                    dy_smem,
                    dy_tma,
                    0,
                    (query_tile * TILE as u32) as i32,
                    plane,
                    &raw mut TMA_BARRIER,
                );
                mbarrier_arrive_expect_tx(&raw const TMA_BARRIER, 1, 2 * TILE_BYTES as u32);
            }
            while !mbarrier_try_wait_parity(&raw const TMA_BARRIER, 0) {}

            // The 128 query rows' LSE (converted to base-2) and softmax dot,
            // staged once so the register pass reads them from shared memory.
            let query_row = (batch * t) as usize + query_tile as usize * TILE + tid as usize;
            let stat_index = query_row * h as usize + head as usize;
            (*(&raw mut LSE2 as *mut f32).add(tid as usize)) = logsumexp[stat_index] * LOG2E;
            (*(&raw mut DOTS as *mut f32).add(tid as usize)) = dot[stat_index];
            thread::sync_threads();

            let ds_phase = (ds_smem as usize >> 7) & 7;
            let mut dq_acc = [[0.0f32; 16]; 4];

            let mut tma_phase = 1u32;
            let mut mma_phase = 0u32;
            let mut key_tile = 0u32;
            while key_tile <= query_tile {
                if is_leader {
                    let key_row = (key_tile * TILE as u32) as i32;
                    cp_async_bulk_tensor_3d_g2s(
                        k_smem,
                        k_tma,
                        0,
                        key_row,
                        plane,
                        &raw mut TMA_BARRIER,
                    );
                    cp_async_bulk_tensor_3d_g2s(
                        v_smem,
                        v_tma,
                        0,
                        key_row,
                        plane,
                        &raw mut TMA_BARRIER,
                    );
                    mbarrier_arrive_expect_tx(&raw const TMA_BARRIER, 1, 2 * TILE_BYTES as u32);
                }
                while !mbarrier_try_wait_parity(&raw const TMA_BARRIER, tma_phase & 1) {}
                tma_phase += 1;
                thread::sync_threads();

                // S = Q·Kᵀ and dP = dY·Vᵀ, both fresh into their own TMEM.
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    score_mma(s_tmem, q_smem, k_smem, s_instruction);
                    score_mma(dp_tmem, dy_smem, v_smem, s_instruction);
                    tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
                }
                while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, mma_phase & 1) {}
                mma_phase += 1;
                thread::sync_threads();

                backward_q_tile(
                    s_tmem,
                    dp_tmem,
                    key_tile == query_tile,
                    warp_id,
                    lane,
                    &raw const LSE2 as *const f32,
                    &raw const DOTS as *const f32,
                    ds_smem,
                    ds_phase,
                );

                fence_proxy_async_shared_cta();
                tcgen05_fence_before_thread_sync();
                thread::sync_threads();

                // dQ += dS·K, continuing the accumulator (fresh on key 0).
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    grad_mma(dq_tmem, ds_smem, k_smem, grad_instruction, key_tile == 0);
                    tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
                }
                while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, mma_phase & 1) {}
                mma_phase += 1;

                tcgen05_fence_before_thread_sync();
                thread::sync_threads();
                key_tile += 1;
            }

            merge_output_tile(dq_tmem, warp_id, &mut dq_acc);
            store_grad_tile(batch, t, h, head, query_tile, warp_id, lane, &dq_acc, &mut dq);

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem, 512);
            }
            if is_leader {
                mbarrier_inval(&raw mut TMA_BARRIER);
                mbarrier_inval(&raw mut MMA_BARRIER);
            }
        }
    }

    /// Synchronous tcgen05 key-parallel backward (issue #35, phase 4,
    /// kernel B). Launch with `host::flash_backward_kv_config`: grid
    /// `(T/128, H, B)`, 128 threads, `FLASH_BACKWARD_KV_SMEM` dynamic shared
    /// bytes (opted in by the loader).
    ///
    /// One CTA owns a 128-key tile of one `(batch, head)` and streams the
    /// causal query tiles `key_tile..T/128`. K and V stay resident; per query
    /// tile it recomputes the transposed `Sᵀ = K·Qᵀ` and `dPᵀ = V·dYᵀ` into
    /// fp32 TMEM, forms the transposed probabilities and `dSᵀ·ln2` in
    /// registers (§ `backward_kv_tile`), and accumulates `dV += Pᵀ·dY` and
    /// `dK += dSᵀ·Q` in two TMEM segments. The staged Q carries `scale·log2e`,
    /// so folding `ln2` into dSᵀ lands `scale` on dK (`ln2·scale·log2e =
    /// scale`) and dV needs no factor. `dk`/`dv` are written directly from the
    /// register accumulators; blocks own disjoint key tiles, so no atomics.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub unsafe fn flash_backward_kv_tcgen05(
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        logsumexp: &[f32],
        dot: &[f32],
        sequence_length: u32,
        heads: u32,
        mut dk: DisjointSlice<f32>,
        mut dv: DisjointSlice<f32>,
    ) {
        unsafe {
            static mut TMEM_ADDRESS: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut MMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut LSE2: SharedArray<f32, 128> = SharedArray::UNINIT;
            static mut DOTS: SharedArray<f32, 128> = SharedArray::UNINIT;

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let k_smem = smem;
            let v_smem = smem.add(TILE_BYTES);
            let q_smem = smem.add(2 * TILE_BYTES);
            let dy_smem = smem.add(3 * TILE_BYTES);
            let p_smem = smem.add(4 * TILE_BYTES);
            let ds_smem = smem.add(6 * TILE_BYTES);

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != TILE {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let is_leader = tid == 0;

            let key_tile = thread::blockIdx_x();
            let head = thread::blockIdx_y();
            let batch = thread::blockIdx_z();
            let t = sequence_length;
            let h = heads;
            let plane = (batch * h + head) as i32;
            let tiles = t / TILE as u32;

            if is_leader {
                mbarrier_init(&raw mut TMA_BARRIER, 1);
                mbarrier_init(&raw mut MMA_BARRIER, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDRESS as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem = *(&raw const TMEM_ADDRESS as *const u32);
            let st_tmem = tmem;
            let dpt_tmem = tmem + 128;
            let dv_tmem = tmem + 256;
            let dk_tmem = tmem + 320;

            let s_instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let grad_instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N64)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .transpose_b(true)
                .build()
                .raw();

            // K and V stay operand-A resident for the whole query stream.
            if is_leader {
                let key_row = (key_tile * TILE as u32) as i32;
                cp_async_bulk_tensor_3d_g2s(k_smem, k_tma, 0, key_row, plane, &raw mut TMA_BARRIER);
                cp_async_bulk_tensor_3d_g2s(v_smem, v_tma, 0, key_row, plane, &raw mut TMA_BARRIER);
                mbarrier_arrive_expect_tx(&raw const TMA_BARRIER, 1, 2 * TILE_BYTES as u32);
            }
            while !mbarrier_try_wait_parity(&raw const TMA_BARRIER, 0) {}

            let p_phase = (p_smem as usize >> 7) & 7;
            let ds_phase = (ds_smem as usize >> 7) & 7;
            let mut dv_acc = [[0.0f32; 16]; 4];
            let mut dk_acc = [[0.0f32; 16]; 4];

            let mut tma_phase = 1u32;
            let mut mma_phase = 0u32;
            let mut query_tile = key_tile;
            while query_tile < tiles {
                if is_leader {
                    let query_row = (query_tile * TILE as u32) as i32;
                    cp_async_bulk_tensor_3d_g2s(
                        q_smem,
                        q_tma,
                        0,
                        query_row,
                        plane,
                        &raw mut TMA_BARRIER,
                    );
                    cp_async_bulk_tensor_3d_g2s(
                        dy_smem,
                        dy_tma,
                        0,
                        query_row,
                        plane,
                        &raw mut TMA_BARRIER,
                    );
                    mbarrier_arrive_expect_tx(&raw const TMA_BARRIER, 1, 2 * TILE_BYTES as u32);
                }
                while !mbarrier_try_wait_parity(&raw const TMA_BARRIER, tma_phase & 1) {}
                tma_phase += 1;
                thread::sync_threads();

                // Stage this query tile's 128 rows' base-2 LSE and dot.
                let global_row =
                    (batch * t) as usize + query_tile as usize * TILE + tid as usize;
                let stat_index = global_row * h as usize + head as usize;
                (*(&raw mut LSE2 as *mut f32).add(tid as usize)) = logsumexp[stat_index] * LOG2E;
                (*(&raw mut DOTS as *mut f32).add(tid as usize)) = dot[stat_index];
                thread::sync_threads();

                // Sᵀ = K·Qᵀ and dPᵀ = V·dYᵀ, both fresh into their own TMEM.
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    score_mma(st_tmem, k_smem, q_smem, s_instruction);
                    score_mma(dpt_tmem, v_smem, dy_smem, s_instruction);
                    tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
                }
                while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, mma_phase & 1) {}
                mma_phase += 1;
                thread::sync_threads();

                backward_kv_tile(
                    st_tmem,
                    dpt_tmem,
                    query_tile == key_tile,
                    warp_id,
                    lane,
                    &raw const LSE2 as *const f32,
                    &raw const DOTS as *const f32,
                    p_smem,
                    p_phase,
                    ds_smem,
                    ds_phase,
                );

                fence_proxy_async_shared_cta();
                tcgen05_fence_before_thread_sync();
                thread::sync_threads();

                // dV += Pᵀ·dY and dK += dSᵀ·Q, fresh on the first query tile.
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    let fresh = query_tile == key_tile;
                    grad_mma(dv_tmem, p_smem, dy_smem, grad_instruction, fresh);
                    grad_mma(dk_tmem, ds_smem, q_smem, grad_instruction, fresh);
                    tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
                }
                while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, mma_phase & 1) {}
                mma_phase += 1;

                tcgen05_fence_before_thread_sync();
                thread::sync_threads();
                query_tile += 1;
            }

            merge_output_tile(dv_tmem, warp_id, &mut dv_acc);
            merge_output_tile(dk_tmem, warp_id, &mut dk_acc);
            store_grad_tile(batch, t, h, head, key_tile, warp_id, lane, &dv_acc, &mut dv);
            store_grad_tile(batch, t, h, head, key_tile, warp_id, lane, &dk_acc, &mut dk);

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem, 512);
            }
            if is_leader {
                mbarrier_inval(&raw mut TMA_BARRIER);
                mbarrier_inval(&raw mut MMA_BARRIER);
            }
        }
    }

    /// Issue one tile's `O = P·V` from the MMA warp: wait for the softmax
    /// warpgroup to publish P, chain the eight K=16 MMAs — continuing the
    /// TMEM segment, or starting a fresh one when the warpgroup's restart
    /// flag for this tile says its vote drained it — and commit completion
    /// into `o_full`. The flag read is ordered by the `p_full` wait (the
    /// warpgroup writes it before arriving).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn output_mma(
        tile: u32,
        stages: usize,
        o_tmem: u32,
        p_smem: *mut u8,
        v_ring: *mut u8,
        o_instruction: u32,
        p_full: *const Barrier,
        o_full: *mut Barrier,
        restart: *const u32,
    ) {
        unsafe {
            while !mbarrier_try_wait_parity(p_full, tile & 1) {}
            let fresh = *restart.add((tile & 1) as usize) != 0;
            let v_smem = v_ring.add((tile as usize % stages) * TILE_BYTES);
            let mut chunk = 0u64;
            while chunk < 8 {
                let a_descriptor = smem_descriptor(
                    p_smem as u64 + (chunk / 4) * TILE_BYTES as u64 + (chunk % 4) * 32,
                );
                let b_descriptor = smem_descriptor(v_smem as u64 + chunk * 2048);
                tcgen05_mma_f16(
                    o_tmem,
                    a_descriptor,
                    b_descriptor,
                    o_instruction,
                    chunk > 0 || !fresh,
                );
                chunk += 1;
            }
            tcgen05_commit_shared_cluster(o_full as *mut u64);
        }
    }

    /// One softmax-warpgroup tile step, shared by the pipelined and
    /// persistent forwards: wait for S, run `softmax_tile` (with the
    /// correction vote), publish the segment-restart flag and P, then wait
    /// out this tile's O MMA and recycle the K/V stage. Callers pass
    /// `diagonal` as a literal so the mask logic folds out of the full-tile
    /// loop, and `double_s` as a literal to select the pipelined kernel's
    /// double-buffered S phase arithmetic. `tid_in_group`/`warp_id` are
    /// warpgroup-local.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn warpgroup_tile(
        i: u32,
        diagonal: bool,
        stages: usize,
        double_s: bool,
        tid_in_group: u32,
        warp_id: u32,
        lane: u32,
        s_tmem: u32,
        o_tmem: u32,
        p_smem: *mut u8,
        p_phase: usize,
        votes: *mut u32,
        vote_barrier: *const Barrier,
        restart: *mut u32,
        s_full: *mut Barrier,
        s_free: *mut Barrier,
        p_full: *const Barrier,
        o_full: *const Barrier,
        kv_free: *mut Barrier,
        m_ref: &mut [f32; 4],
        running_sum: &mut [f32; 4],
        out_acc: &mut [[f32; 16]; 4],
        corrections: &mut u32,
    ) {
        unsafe {
            let (buffer, s_parity) = if double_s {
                ((i & 1) as usize, (i / 2) & 1)
            } else {
                (0usize, i & 1)
            };
            while !mbarrier_try_wait_parity(s_full.add(buffer), s_parity) {}
            let correction = softmax_tile(
                s_tmem + (buffer as u32) * 128,
                o_tmem,
                i,
                diagonal,
                warp_id,
                lane,
                p_smem,
                p_phase,
                votes,
                vote_barrier,
                m_ref,
                running_sum,
                out_acc,
            );
            if i > 0 && correction {
                *corrections += 1;
            }
            if tid_in_group == 0 {
                *restart.add((i & 1) as usize) = correction as u32;
            }
            // Both S passes are drained and P is fenced into the async
            // proxy: release the S buffer and publish P (which also
            // releases the restart flag to the MMA warp).
            fence_proxy_async_shared_cta();
            mbarrier_arrive(s_free.add(buffer));
            mbarrier_arrive(p_full);
            while !mbarrier_try_wait_parity(o_full, i & 1) {}
            if tid_in_group == 0 {
                mbarrier_arrive(kv_free.add(i as usize % stages));
            }
        }
    }

    /// Warp-specialized pipelined causal forward (issue #35, phase 2).
    /// Launch with `host::flash_pipelined_config`: grid `(T/128, H, B)`,
    /// `FLASH_PIPELINE_BLOCK` threads, `host::FLASH_PIPELINE_SMEM_BYTES`
    /// dynamic shared bytes (opted in by the loader). Same operand and
    /// output contract as `flash_forward_tcgen05`.
    ///
    /// Roles, connected only by mbarrier phase-parity spins:
    /// - warp 4's leader streams TMA — Q once, then the K/V ring running
    ///   `PIPELINE_STAGES` tiles ahead (`kv_full[stage]` expect_tx,
    ///   recycling on `kv_free[stage]`);
    /// - warp 5's leader issues MMAs, staggered so `S-MMA(i)` (S buffer
    ///   `i % 2`, guarded by `s_free`) reaches the tensor core before
    ///   `O-MMA(i-1)` (guarded by `p_full`): while the warpgroup runs
    ///   softmax(i), the core is already producing `S(i+1)` — the phase-2
    ///   overlap. The stagger is also why `PIPELINE_STAGES >= 2`: with one
    ///   stage, `S-MMA(i)`'s wait for `kv_full` would precede the
    ///   `O-MMA(i-1)` whose completion recycles the stage.
    /// - warps 0–3 are the softmax warpgroup: wait `s_full`, run
    ///   `softmax_tile` (which only drains the O segment on a correction
    ///   vote), release `s_free` and `p_full`, wait `o_full`, recycle the
    ///   K/V stage. Correction and epilogue stay fused here — the output
    ///   accumulator lives in these registers and TMEM lane ownership pins
    ///   every drain to warps 0–3.
    ///
    /// P stays single-buffered: the warpgroup's `o_full(i)` wait proves
    /// O-MMA(i) has finished reading P before pass 2 of tile `i+1`
    /// overwrites it. The same wait is where the warpgroup — not a second
    /// tcgen05 commit, whose "prior operations" scope is ambiguous —
    /// releases `kv_free`. Phase-parity arithmetic is sound because every
    /// barrier's completions lead their waiter by at most one phase: each
    /// producer's next completion transitively requires the previous
    /// consumer wait.
    #[kernel]
    #[launch_bounds(192, 1)]
    pub unsafe fn flash_forward_pipelined(
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        sequence_length: u32,
        heads: u32,
        mut output: DisjointSlice<f32>,
        mut logsumexp: DisjointSlice<f32>,
        mut correction_counts: DisjointSlice<u32>,
    ) {
        unsafe {
            static mut TMEM_ADDRESS: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            // Barrier rings as raw u64 words (a Barrier is one 64-bit state
            // word); pointer-cast so the lengths can follow PIPELINE_STAGES.
            static mut KV_FULL: SharedArray<u64, PIPELINE_STAGES, 8> = SharedArray::UNINIT;
            static mut KV_FREE: SharedArray<u64, PIPELINE_STAGES, 8> = SharedArray::UNINIT;
            static mut S_FULL: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut S_FREE: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut P_FULL: Barrier = Barrier::UNINIT;
            static mut O_FULL: Barrier = Barrier::UNINIT;
            static mut VOTE_BARRIER: Barrier = Barrier::UNINIT;
            static mut VOTES: SharedArray<u32, 8, 4> = SharedArray::UNINIT;
            static mut RESTART: SharedArray<u32, 2, 4> = SharedArray::UNINIT;

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let q_smem = smem;
            let k_ring = smem.add(TILE_BYTES);
            let v_ring = smem.add((1 + PIPELINE_STAGES) * TILE_BYTES);
            let p_smem = smem.add((1 + 2 * PIPELINE_STAGES) * TILE_BYTES);

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != FLASH_PIPELINE_BLOCK {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();

            let kv_full = &raw mut KV_FULL as *mut Barrier;
            let kv_free = &raw mut KV_FREE as *mut Barrier;
            let s_full = &raw mut S_FULL as *mut Barrier;
            let s_free = &raw mut S_FREE as *mut Barrier;

            let query_tile = thread::blockIdx_x();
            let head = thread::blockIdx_y();
            let batch = thread::blockIdx_z();
            let t = sequence_length;
            let h = heads;
            let plane = (batch * h + head) as i32;
            let key_tiles = query_tile + 1;

            if tid == 0 {
                let mut stage = 0usize;
                while stage < PIPELINE_STAGES {
                    mbarrier_init(kv_full.add(stage), 1);
                    mbarrier_init(kv_free.add(stage), 1);
                    stage += 1;
                }
                mbarrier_init(s_full, 1);
                mbarrier_init(s_full.add(1), 1);
                mbarrier_init(s_free, TILE as u32);
                mbarrier_init(s_free.add(1), TILE as u32);
                mbarrier_init(&raw mut P_FULL, TILE as u32);
                mbarrier_init(&raw mut O_FULL, 1);
                mbarrier_init(&raw mut VOTE_BARRIER, TILE as u32);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDRESS as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem = *(&raw const TMEM_ADDRESS as *const u32);
            let o_tmem = tmem + 256;

            if tid < TILE as u32 {
                // Softmax / correction / epilogue warpgroup. The key loop
                // is split so the diagonal tile is the only one paying for
                // mask logic (the full-tile calls fold `diagonal = false`).
                let p_phase = (p_smem as usize >> 7) & 7;
                let votes = &raw mut VOTES as *mut u32;
                let restart = &raw mut RESTART as *mut u32;
                let mut m_ref = [MASKED_SCORE; 4];
                let mut running_sum = [0.0f32; 4];
                let mut out_acc = [[0.0f32; 16]; 4];
                let mut corrections = 0u32;
                let mut i = 0u32;
                while i + 1 < key_tiles {
                    warpgroup_tile(
                        i,
                        false,
                        PIPELINE_STAGES,
                        true,
                        tid,
                        warp_id,
                        lane,
                        tmem,
                        o_tmem,
                        p_smem,
                        p_phase,
                        votes,
                        &raw const VOTE_BARRIER,
                        restart,
                        s_full,
                        s_free,
                        &raw const P_FULL,
                        &raw const O_FULL,
                        kv_free,
                        &mut m_ref,
                        &mut running_sum,
                        &mut out_acc,
                        &mut corrections,
                    );
                    i += 1;
                }
                warpgroup_tile(
                    i,
                    true,
                    PIPELINE_STAGES,
                    true,
                    tid,
                    warp_id,
                    lane,
                    tmem,
                    o_tmem,
                    p_smem,
                    p_phase,
                    votes,
                    &raw const VOTE_BARRIER,
                    restart,
                    s_full,
                    s_free,
                    &raw const P_FULL,
                    &raw const O_FULL,
                    kv_free,
                    &mut m_ref,
                    &mut running_sum,
                    &mut out_acc,
                    &mut corrections,
                );
                merge_output_tile(o_tmem, warp_id, &mut out_acc);
                store_outputs(
                    batch,
                    t,
                    h,
                    head,
                    query_tile,
                    warp_id,
                    lane,
                    &m_ref,
                    &running_sum,
                    &out_acc,
                    &mut output,
                    &mut logsumexp,
                );
                if tid == 0 {
                    let tiles = t as usize / TILE;
                    *correction_counts.get_unchecked_mut(
                        (batch * h + head) as usize * tiles + query_tile as usize,
                    ) = corrections;
                }
            } else if tid == TILE as u32 {
                // TMA load warp leader: Q once, then the K/V ring.
                let mut i = 0u32;
                while i < key_tiles {
                    let stage = (i as usize) % PIPELINE_STAGES;
                    if i as usize >= PIPELINE_STAGES {
                        let parity = ((i as usize / PIPELINE_STAGES - 1) & 1) as u32;
                        while !mbarrier_try_wait_parity(kv_free.add(stage), parity) {}
                    }
                    let key_row = (i * TILE as u32) as i32;
                    cp_async_bulk_tensor_3d_g2s(
                        k_ring.add(stage * TILE_BYTES),
                        k_tma,
                        0,
                        key_row,
                        plane,
                        kv_full.add(stage),
                    );
                    cp_async_bulk_tensor_3d_g2s(
                        v_ring.add(stage * TILE_BYTES),
                        v_tma,
                        0,
                        key_row,
                        plane,
                        kv_full.add(stage),
                    );
                    if i == 0 {
                        cp_async_bulk_tensor_3d_g2s(
                            q_smem,
                            q_tma,
                            0,
                            (query_tile * TILE as u32) as i32,
                            plane,
                            kv_full.add(stage),
                        );
                        mbarrier_arrive_expect_tx(kv_full.add(stage), 1, 3 * TILE_BYTES as u32);
                    } else {
                        mbarrier_arrive_expect_tx(kv_full.add(stage), 1, 2 * TILE_BYTES as u32);
                    }
                    i += 1;
                }
            } else if tid == (TILE + 32) as u32 {
                // MMA warp leader.
                let s_instruction = Tcgen05InstructionDescriptor::builder()
                    .shape(Tcgen05MmaShape::M128_N128)
                    .element_type(Tcgen05ElementType::BF16)
                    .accumulator_type(Tcgen05AccumulatorType::F32)
                    .build()
                    .raw();
                let o_instruction = Tcgen05InstructionDescriptor::builder()
                    .shape(Tcgen05MmaShape::M128_N64)
                    .element_type(Tcgen05ElementType::BF16)
                    .accumulator_type(Tcgen05AccumulatorType::F32)
                    .transpose_b(true)
                    .build()
                    .raw();
                tcgen05_fence_after_thread_sync();
                let mut i = 0u32;
                while i < key_tiles {
                    let stage = (i as usize) % PIPELINE_STAGES;
                    while !mbarrier_try_wait_parity(
                        kv_full.add(stage),
                        ((i as usize / PIPELINE_STAGES) & 1) as u32,
                    ) {}
                    let buffer = (i & 1) as usize;
                    if i >= 2 {
                        while !mbarrier_try_wait_parity(s_free.add(buffer), (i / 2 - 1) & 1) {}
                    }
                    score_mma(
                        tmem + (buffer as u32) * 128,
                        q_smem,
                        k_ring.add(stage * TILE_BYTES),
                        s_instruction,
                    );
                    tcgen05_commit_shared_cluster(s_full.add(buffer) as *mut u64);
                    if i > 0 {
                        output_mma(
                            i - 1,
                            PIPELINE_STAGES,
                            o_tmem,
                            p_smem,
                            v_ring,
                            o_instruction,
                            &raw const P_FULL,
                            &raw mut O_FULL,
                            &raw const RESTART as *const u32,
                        );
                    }
                    i += 1;
                }
                output_mma(
                    key_tiles - 1,
                    PIPELINE_STAGES,
                    o_tmem,
                    p_smem,
                    v_ring,
                    o_instruction,
                    &raw const P_FULL,
                    &raw mut O_FULL,
                    &raw const RESTART as *const u32,
                );
            }

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem, 512);
            }
            if tid == 0 {
                let mut stage = 0usize;
                while stage < PIPELINE_STAGES {
                    mbarrier_inval(kv_full.add(stage));
                    mbarrier_inval(kv_free.add(stage));
                    stage += 1;
                }
                mbarrier_inval(s_full);
                mbarrier_inval(s_full.add(1));
                mbarrier_inval(s_free);
                mbarrier_inval(s_free.add(1));
                mbarrier_inval(&raw mut P_FULL);
                mbarrier_inval(&raw mut O_FULL);
                mbarrier_inval(&raw mut VOTE_BARRIER);
            }
        }
    }

    /// One full workstream of the persistent forward, run by one softmax
    /// warpgroup: the split full-tile/diagonal key loop, the final segment
    /// drain, and the outputs. All indices (`tid_in_group`, `warp_id`) are
    /// warpgroup-local; the barrier pointers are this stream's own set
    /// except `kv_free`, which both streams share (each arrives once per
    /// tile it consumed).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn persistent_stream(
        query_tile: u32,
        batch: u32,
        t: u32,
        h: u32,
        head: u32,
        tid_in_group: u32,
        warp_id: u32,
        lane: u32,
        s_tmem: u32,
        o_tmem: u32,
        p_smem: *mut u8,
        votes: *mut u32,
        vote_barrier: *const Barrier,
        restart: *mut u32,
        s_full: *mut Barrier,
        s_free: *mut Barrier,
        p_full: *const Barrier,
        o_full: *const Barrier,
        kv_free: *mut Barrier,
        output: &mut DisjointSlice<f32>,
        logsumexp: &mut DisjointSlice<f32>,
        correction_counts: &mut DisjointSlice<u32>,
    ) {
        unsafe {
            let p_phase = (p_smem as usize >> 7) & 7;
            let key_tiles = query_tile + 1;
            let mut m_ref = [MASKED_SCORE; 4];
            let mut running_sum = [0.0f32; 4];
            let mut out_acc = [[0.0f32; 16]; 4];
            let mut corrections = 0u32;
            let mut i = 0u32;
            while i + 1 < key_tiles {
                warpgroup_tile(
                    i,
                    false,
                    PERSISTENT_STAGES,
                    false,
                    tid_in_group,
                    warp_id,
                    lane,
                    s_tmem,
                    o_tmem,
                    p_smem,
                    p_phase,
                    votes,
                    vote_barrier,
                    restart,
                    s_full,
                    s_free,
                    p_full,
                    o_full,
                    kv_free,
                    &mut m_ref,
                    &mut running_sum,
                    &mut out_acc,
                    &mut corrections,
                );
                i += 1;
            }
            warpgroup_tile(
                i,
                true,
                PERSISTENT_STAGES,
                false,
                tid_in_group,
                warp_id,
                lane,
                s_tmem,
                o_tmem,
                p_smem,
                p_phase,
                votes,
                vote_barrier,
                restart,
                s_full,
                s_free,
                p_full,
                o_full,
                kv_free,
                &mut m_ref,
                &mut running_sum,
                &mut out_acc,
                &mut corrections,
            );
            merge_output_tile(o_tmem, warp_id, &mut out_acc);
            store_outputs(
                batch,
                t,
                h,
                head,
                query_tile,
                warp_id,
                lane,
                &m_ref,
                &running_sum,
                &out_acc,
                output,
                logsumexp,
            );
            if tid_in_group == 0 {
                let tiles = t as usize / TILE;
                *correction_counts.get_unchecked_mut(
                    (batch * h + head) as usize * tiles + query_tile as usize,
                ) = corrections;
            }
        }
    }

    /// (Re)initialize the persistent kernel's barrier set for one work
    /// item. `kv_free` counts one arrival per active consumer stream.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn persistent_barriers_init(
        kv_full: *mut Barrier,
        kv_free: *mut Barrier,
        s_full: *mut Barrier,
        s_free: *mut Barrier,
        p_full: *mut Barrier,
        o_full: *mut Barrier,
        vote_barrier: *mut Barrier,
        kv_consumers: u32,
    ) {
        unsafe {
            let mut stage = 0usize;
            while stage < PERSISTENT_STAGES {
                mbarrier_init(kv_full.add(stage), 1);
                mbarrier_init(kv_free.add(stage), kv_consumers);
                stage += 1;
            }
            let mut stream = 0usize;
            while stream < 2 {
                mbarrier_init(s_full.add(stream), 1);
                mbarrier_init(s_free.add(stream), TILE as u32);
                mbarrier_init(p_full.add(stream), TILE as u32);
                mbarrier_init(o_full.add(stream), 1);
                mbarrier_init(vote_barrier.add(stream), TILE as u32);
                stream += 1;
            }
        }
    }

    /// Invalidate the persistent kernel's barrier set between work items
    /// (and at exit), wiping whatever unbalanced arrivals the item left.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn persistent_barriers_inval(
        kv_full: *mut Barrier,
        kv_free: *mut Barrier,
        s_full: *mut Barrier,
        s_free: *mut Barrier,
        p_full: *mut Barrier,
        o_full: *mut Barrier,
        vote_barrier: *mut Barrier,
    ) {
        unsafe {
            let mut stage = 0usize;
            while stage < PERSISTENT_STAGES {
                mbarrier_inval(kv_full.add(stage));
                mbarrier_inval(kv_free.add(stage));
                stage += 1;
            }
            let mut stream = 0usize;
            while stream < 2 {
                mbarrier_inval(s_full.add(stream));
                mbarrier_inval(s_free.add(stream));
                mbarrier_inval(p_full.add(stream));
                mbarrier_inval(o_full.add(stream));
                mbarrier_inval(vote_barrier.add(stream));
                stream += 1;
            }
        }
    }

    /// Persistent two-Q-tile ping-pong forward (issue #35, phase 3).
    /// Launch with `host::flash_persistent_config`: a 1-D grid of at most
    /// `ceil(tiles/2) * H * B` CTAs, `FLASH_PERSISTENT_BLOCK` threads,
    /// `host::FLASH_PERSISTENT_SMEM_BYTES` dynamic shared bytes. Operand
    /// and output contracts match the other tcgen05 forwards.
    ///
    /// Each work item is a (query-tile *pair*, head, batch): warpgroup A
    /// (warps 0–3) owns query tile `2p`, warpgroup B (warps 4–7) owns
    /// `2p+1`, and both share one K/V ring, one TMA-load warp (warp 8) and
    /// one MMA warp (warp 9). TMEM holds a single-buffered S plus an O
    /// segment per stream (384 of the 512 columns): while one warpgroup
    /// runs softmax, the MMA warp feeds the other stream — the ping-pong
    /// a single 128-thread warpgroup could not reach, and the main use of
    /// the SM's issue slots now that TMEM pins occupancy to one CTA.
    ///
    /// CTAs run a static strided work-item loop (`blockIdx.x`, stepping by
    /// `gridDim.x`) with items ordered by *descending* pair index, so the
    /// causally long pairs are dealt out before the cheap ones. Launching
    /// with grid = the full item count degenerates to one item per CTA —
    /// the non-persistent config kept for hang debugging. Every mbarrier is
    /// re-initialized per item behind a block sync, so each item's phase
    /// arithmetic starts from zero and unbalanced arrivals (stream A never
    /// arrives for the shared stream's extra diagonal tile; an inactive
    /// stream B never arrives at all) are wiped, not threaded through
    /// parity math. `kv_free`'s arrival count is likewise chosen per item:
    /// two consumer streams normally, one when the last odd pair leaves
    /// stream B inactive.
    #[kernel]
    #[launch_bounds(320, 1)]
    pub unsafe fn flash_forward_persistent(
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        sequence_length: u32,
        heads: u32,
        batches: u32,
        mut output: DisjointSlice<f32>,
        mut logsumexp: DisjointSlice<f32>,
        mut correction_counts: DisjointSlice<u32>,
    ) {
        unsafe {
            static mut TMEM_ADDRESS: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut KV_FULL: SharedArray<u64, PERSISTENT_STAGES, 8> = SharedArray::UNINIT;
            static mut KV_FREE: SharedArray<u64, PERSISTENT_STAGES, 8> = SharedArray::UNINIT;
            // Two-stream barrier pairs, indexed A = 0, B = 1.
            static mut S_FULL: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut S_FREE: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut P_FULL: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut O_FULL: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut VOTE: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut VOTES: SharedArray<u32, 16, 4> = SharedArray::UNINIT;
            static mut RESTART: SharedArray<u32, 4, 4> = SharedArray::UNINIT;

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let q_a = smem;
            let q_b = smem.add(TILE_BYTES);
            let k_ring = smem.add(2 * TILE_BYTES);
            let v_ring = smem.add((2 + PERSISTENT_STAGES) * TILE_BYTES);
            let p_a = smem.add((2 + 2 * PERSISTENT_STAGES) * TILE_BYTES);
            let p_b = smem.add((4 + 2 * PERSISTENT_STAGES) * TILE_BYTES);

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != FLASH_PERSISTENT_BLOCK {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();

            let kv_full = &raw mut KV_FULL as *mut Barrier;
            let kv_free = &raw mut KV_FREE as *mut Barrier;
            let s_full = &raw mut S_FULL as *mut Barrier;
            let s_free = &raw mut S_FREE as *mut Barrier;
            let p_full = &raw mut P_FULL as *mut Barrier;
            let o_full = &raw mut O_FULL as *mut Barrier;
            let vote_barrier = &raw mut VOTE as *mut Barrier;
            let votes = &raw mut VOTES as *mut u32;
            let restart = &raw mut RESTART as *mut u32;

            let t = sequence_length;
            let h = heads;
            let tiles = t / TILE as u32;
            let pairs = tiles.div_ceil(2);
            let plane_count = h * batches;
            let work_items = pairs * plane_count;

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDRESS as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem = *(&raw const TMEM_ADDRESS as *const u32);
            // TMEM columns: S_A 0..128, S_B 128..256, O_A 256..320,
            // O_B 320..384.

            let mut initialized = false;
            let mut item = thread::blockIdx_x();
            while item < work_items {
                let pair = pairs - 1 - item / plane_count;
                let plane = item % plane_count;
                let batch = plane / h;
                let head = plane % h;
                let tile_a = pair * 2;
                let tile_b = tile_a + 1;
                let b_active = tile_b < tiles;
                let tiles_a = tile_a + 1;
                let tiles_b = tile_b + 1;
                let stream_tiles = if b_active { tiles_b } else { tiles_a };

                if tid == 0 {
                    if initialized {
                        persistent_barriers_inval(
                            kv_full, kv_free, s_full, s_free, p_full, o_full, vote_barrier,
                        );
                    }
                    persistent_barriers_init(
                        kv_full,
                        kv_free,
                        s_full,
                        s_free,
                        p_full,
                        o_full,
                        vote_barrier,
                        1 + b_active as u32,
                    );
                    fence_proxy_async_shared_cta();
                }
                initialized = true;
                thread::sync_threads();

                if tid < TILE as u32 {
                    persistent_stream(
                        tile_a,
                        batch,
                        t,
                        h,
                        head,
                        tid,
                        warp_id,
                        lane,
                        tmem,
                        tmem + 256,
                        p_a,
                        votes,
                        vote_barrier,
                        restart,
                        s_full,
                        s_free,
                        p_full,
                        o_full,
                        kv_free,
                        &mut output,
                        &mut logsumexp,
                        &mut correction_counts,
                    );
                } else if tid < 2 * TILE as u32 {
                    if b_active {
                        persistent_stream(
                            tile_b,
                            batch,
                            t,
                            h,
                            head,
                            tid - TILE as u32,
                            warp_id - 4,
                            lane,
                            tmem + 128,
                            tmem + 320,
                            p_b,
                            votes.add(8),
                            vote_barrier.add(1),
                            restart.add(2),
                            s_full.add(1),
                            s_free.add(1),
                            p_full.add(1),
                            o_full.add(1),
                            kv_free,
                            &mut output,
                            &mut logsumexp,
                            &mut correction_counts,
                        );
                    }
                } else if tid == (2 * TILE) as u32 {
                    // TMA load warp leader: both Q tiles once, then the
                    // shared K/V ring over the longer stream.
                    let plane_index = plane as i32;
                    let mut i = 0u32;
                    while i < stream_tiles {
                        let stage = (i as usize) % PERSISTENT_STAGES;
                        if i as usize >= PERSISTENT_STAGES {
                            let parity = ((i as usize / PERSISTENT_STAGES - 1) & 1) as u32;
                            while !mbarrier_try_wait_parity(kv_free.add(stage), parity) {}
                        }
                        let key_row = (i * TILE as u32) as i32;
                        cp_async_bulk_tensor_3d_g2s(
                            k_ring.add(stage * TILE_BYTES),
                            k_tma,
                            0,
                            key_row,
                            plane_index,
                            kv_full.add(stage),
                        );
                        cp_async_bulk_tensor_3d_g2s(
                            v_ring.add(stage * TILE_BYTES),
                            v_tma,
                            0,
                            key_row,
                            plane_index,
                            kv_full.add(stage),
                        );
                        if i == 0 {
                            cp_async_bulk_tensor_3d_g2s(
                                q_a,
                                q_tma,
                                0,
                                (tile_a * TILE as u32) as i32,
                                plane_index,
                                kv_full.add(stage),
                            );
                            if b_active {
                                cp_async_bulk_tensor_3d_g2s(
                                    q_b,
                                    q_tma,
                                    0,
                                    (tile_b * TILE as u32) as i32,
                                    plane_index,
                                    kv_full.add(stage),
                                );
                            }
                            let q_tiles = 1 + b_active as u32;
                            mbarrier_arrive_expect_tx(
                                kv_full.add(stage),
                                1,
                                (2 + q_tiles) * TILE_BYTES as u32,
                            );
                        } else {
                            mbarrier_arrive_expect_tx(kv_full.add(stage), 1, 2 * TILE_BYTES as u32);
                        }
                        i += 1;
                    }
                } else if tid == (2 * TILE + 32) as u32 {
                    // MMA warp leader: per shared tile, S-MMAs for both
                    // streams (each gated by its own single-buffered S
                    // being free), then the previous tile's O-MMAs — the
                    // same stagger as the pipelined kernel, per stream.
                    let s_instruction = Tcgen05InstructionDescriptor::builder()
                        .shape(Tcgen05MmaShape::M128_N128)
                        .element_type(Tcgen05ElementType::BF16)
                        .accumulator_type(Tcgen05AccumulatorType::F32)
                        .build()
                        .raw();
                    let o_instruction = Tcgen05InstructionDescriptor::builder()
                        .shape(Tcgen05MmaShape::M128_N64)
                        .element_type(Tcgen05ElementType::BF16)
                        .accumulator_type(Tcgen05AccumulatorType::F32)
                        .transpose_b(true)
                        .build()
                        .raw();
                    tcgen05_fence_after_thread_sync();
                    let mut i = 0u32;
                    while i < stream_tiles {
                        let stage = (i as usize) % PERSISTENT_STAGES;
                        while !mbarrier_try_wait_parity(
                            kv_full.add(stage),
                            ((i as usize / PERSISTENT_STAGES) & 1) as u32,
                        ) {}
                        let k_smem = k_ring.add(stage * TILE_BYTES);
                        if i < tiles_a {
                            if i >= 1 {
                                while !mbarrier_try_wait_parity(s_free, (i - 1) & 1) {}
                            }
                            score_mma(tmem, q_a, k_smem, s_instruction);
                            tcgen05_commit_shared_cluster(s_full as *mut u64);
                        }
                        if b_active {
                            if i >= 1 {
                                while !mbarrier_try_wait_parity(s_free.add(1), (i - 1) & 1) {}
                            }
                            score_mma(tmem + 128, q_b, k_smem, s_instruction);
                            tcgen05_commit_shared_cluster(s_full.add(1) as *mut u64);
                        }
                        if i > 0 {
                            output_mma(
                                i - 1,
                                PERSISTENT_STAGES,
                                tmem + 256,
                                p_a,
                                v_ring,
                                o_instruction,
                                p_full,
                                o_full,
                                restart,
                            );
                            if b_active {
                                output_mma(
                                    i - 1,
                                    PERSISTENT_STAGES,
                                    tmem + 320,
                                    p_b,
                                    v_ring,
                                    o_instruction,
                                    p_full.add(1),
                                    o_full.add(1),
                                    restart.add(2),
                                );
                            }
                        }
                        i += 1;
                    }
                    if b_active {
                        output_mma(
                            tiles_b - 1,
                            PERSISTENT_STAGES,
                            tmem + 320,
                            p_b,
                            v_ring,
                            o_instruction,
                            p_full.add(1),
                            o_full.add(1),
                            restart.add(2),
                        );
                    } else {
                        output_mma(
                            tiles_a - 1,
                            PERSISTENT_STAGES,
                            tmem + 256,
                            p_a,
                            v_ring,
                            o_instruction,
                            p_full,
                            o_full,
                            restart,
                        );
                    }
                }

                tcgen05_fence_before_thread_sync();
                thread::sync_threads();
                item += thread::gridDim_x();
            }

            if warp_id == 0 {
                tcgen05_dealloc(tmem, 512);
            }
            if tid == 0 && initialized {
                persistent_barriers_inval(
                    kv_full, kv_free, s_full, s_free, p_full, o_full, vote_barrier,
                );
            }
        }
    }
}
