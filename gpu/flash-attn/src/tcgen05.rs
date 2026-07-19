//! FA4-shaped tcgen05 attention forward (issue #35, phases 1–2).
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
//! One CTA owns a 128-query tile of one `(batch, head)` and streams 128-key
//! tiles: TMA loads Q/K/V into swizzled shared tiles, `S = Q·Kᵀ` accumulates
//! in fp32 TMEM, a register softmax (mask → running max → software exp2 →
//! running sum) packs bf16 probabilities back to shared memory with swizzled
//! `stmatrix` stores, and `O += P·V` accumulates in a second TMEM region that
//! is drained and rescale-merged into per-thread registers every tile (the
//! "always drain" form of FA4's conditional correction; conditional segments
//! land in phase 3).
//!
//! Two kernels share that per-tile math (`softmax_tile` / `merge_output_tile`
//! / `store_outputs`):
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

/// Finite stand-in for "masked" in the base-2 score domain; far enough below
/// any real score that `exp2` flushes it to a subnormal-scale value while the
/// running-max recurrence stays NaN-free.
const MASKED_SCORE: f32 = -1.0e30;

#[cuda_module]
pub mod kernels {
    use super::*;

    const LN2: f32 = 0.693_147_18;

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

    /// One key tile of register softmax, shared by both forward kernels.
    ///
    /// Drains `S[128, 128]` from `s_tmem` twice — pass 1 for masked row
    /// maxima, pass 2 to exponentiate against the merged running max — and
    /// stores bf16 probabilities into the two stacked SWIZZLE_128B P
    /// subtiles via `stmatrix` (the per-row addresses apply the
    /// 16-byte-chunk XOR the TMA swizzle would have produced, folding in
    /// the tile base's absolute 128-byte row phase, so the O-MMA
    /// descriptors read P exactly like a TMA-loaded operand). On return the
    /// row statistics are merged and `out_acc` is rescaled, ready for the
    /// O-segment drain; the caller still owns proxy fencing and whatever
    /// synchronization hands P to the MMA.
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
        diagonal: bool,
        warp_id: u32,
        lane: u32,
        p_smem: *mut u8,
        p_phase: usize,
        running_max: &mut [f32; 4],
        running_sum: &mut [f32; 4],
        out_acc: &mut [[f32; 16]; 4],
    ) {
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
            let mut factor = [0.0f32; 4];
            let mut slot = 0usize;
            while slot < 4 {
                let tile = quad_max(tile_max[slot]);
                let next = fmax(running_max[slot], tile);
                factor[slot] = exp2_approx(running_max[slot] - next);
                running_max[slot] = next;
                slot += 1;
            }

            // Pass 2: probabilities — re-drain S, exponentiate, accumulate
            // row sums, and store bf16 P through the swizzle-aware stmatrix
            // addresses.
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
                    let p_a0 = exp2_approx(s_a0 - running_max[slot_a]);
                    let p_a1 = exp2_approx(s_a1 - running_max[slot_a]);
                    let p_a8 = exp2_approx(s_a8 - running_max[slot_a]);
                    let p_a9 = exp2_approx(s_a9 - running_max[slot_a]);
                    let p_b0 = exp2_approx(s_b0 - running_max[slot_b]);
                    let p_b1 = exp2_approx(s_b1 - running_max[slot_b]);
                    let p_b8 = exp2_approx(s_b8 - running_max[slot_b]);
                    let p_b9 = exp2_approx(s_b9 - running_max[slot_b]);
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
                let sum = quad_sum(tile_sum[slot]);
                running_sum[slot] = running_sum[slot] * factor[slot] + sum;
                let mut value = 0usize;
                while value < 16 {
                    out_acc[slot][value] *= factor[slot];
                    value += 1;
                }
                slot += 1;
            }
        }
    }

    /// Drain one `O = P·V` TMEM segment and add it into the per-thread
    /// output accumulator (which `softmax_tile` already rescaled).
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
        running_max: &[f32; 4],
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
                        LN2 * (running_max[slot] + log2_approx(running_sum[slot]));
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
    #[kernel]
    pub unsafe fn flash_forward_tcgen05(
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        sequence_length: u32,
        heads: u32,
        mut output: DisjointSlice<f32>,
        mut logsumexp: DisjointSlice<f32>,
    ) {
        unsafe {
            static mut TMEM_ADDRESS: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut MMA_BARRIER: Barrier = Barrier::UNINIT;

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

            let mut running_max = [MASKED_SCORE; 4];
            let mut running_sum = [0.0f32; 4];
            let mut out_acc = [[0.0f32; 16]; 4];

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
                    let mut chunk = 0u64;
                    while chunk < 4 {
                        let a_descriptor = smem_descriptor(q_smem as u64 + chunk * 32);
                        let b_descriptor = smem_descriptor(k_smem as u64 + chunk * 32);
                        tcgen05_mma_f16(s_tmem, a_descriptor, b_descriptor, s_instruction, chunk > 0);
                        chunk += 1;
                    }
                    tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
                }
                while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, mma_phase & 1) {}
                mma_phase += 1;
                thread::sync_threads();

                softmax_tile(
                    s_tmem,
                    key_tile == query_tile,
                    warp_id,
                    lane,
                    p_smem,
                    p_phase,
                    &mut running_max,
                    &mut running_sum,
                    &mut out_acc,
                );

                // P was written through the generic proxy; fence before the
                // async-proxy MMA consumes it.
                fence_proxy_async_shared_cta();
                tcgen05_fence_before_thread_sync();
                thread::sync_threads();

                // O = P·V for this tile (fresh segment; merged in registers).
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    let mut chunk = 0u64;
                    while chunk < 8 {
                        let a_descriptor = smem_descriptor(
                            p_smem as u64 + (chunk / 4) * TILE_BYTES as u64 + (chunk % 4) * 32,
                        );
                        let b_descriptor = smem_descriptor(v_smem as u64 + chunk * 2048);
                        tcgen05_mma_f16(o_tmem, a_descriptor, b_descriptor, o_instruction, chunk > 0);
                        chunk += 1;
                    }
                    tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
                }
                while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, mma_phase & 1) {}
                mma_phase += 1;
                thread::sync_threads();

                merge_output_tile(o_tmem, warp_id, &mut out_acc);

                tcgen05_fence_before_thread_sync();
                thread::sync_threads();
                key_tile += 1;
            }

            store_outputs(
                batch,
                t,
                h,
                head,
                query_tile,
                warp_id,
                lane,
                &running_max,
                &running_sum,
                &out_acc,
                &mut output,
                &mut logsumexp,
            );

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
    /// warpgroup to publish P, chain the eight K=16 MMAs as a fresh segment,
    /// and commit completion into `o_full`.
    #[inline(always)]
    unsafe fn output_mma(
        tile: u32,
        o_tmem: u32,
        p_smem: *mut u8,
        v_ring: *mut u8,
        o_instruction: u32,
        p_full: *const Barrier,
        o_full: *mut Barrier,
    ) {
        unsafe {
            while !mbarrier_try_wait_parity(p_full, tile & 1) {}
            let v_smem = v_ring.add((tile as usize % PIPELINE_STAGES) * TILE_BYTES);
            let mut chunk = 0u64;
            while chunk < 8 {
                let a_descriptor = smem_descriptor(
                    p_smem as u64 + (chunk / 4) * TILE_BYTES as u64 + (chunk % 4) * 32,
                );
                let b_descriptor = smem_descriptor(v_smem as u64 + chunk * 2048);
                tcgen05_mma_f16(o_tmem, a_descriptor, b_descriptor, o_instruction, chunk > 0);
                chunk += 1;
            }
            tcgen05_commit_shared_cluster(o_full as *mut u64);
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
    ///   `softmax_tile`, release `s_free` and `p_full`, wait `o_full`,
    ///   merge the O segment. Correction and epilogue stay fused here — the
    ///   output accumulator lives in these registers and TMEM lane
    ///   ownership pins every drain to warps 0–3.
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
                // Softmax / correction / epilogue warpgroup.
                let p_phase = (p_smem as usize >> 7) & 7;
                let mut running_max = [MASKED_SCORE; 4];
                let mut running_sum = [0.0f32; 4];
                let mut out_acc = [[0.0f32; 16]; 4];
                let mut i = 0u32;
                while i < key_tiles {
                    let buffer = (i & 1) as usize;
                    while !mbarrier_try_wait_parity(s_full.add(buffer), (i / 2) & 1) {}
                    softmax_tile(
                        tmem + (buffer as u32) * 128,
                        i == query_tile,
                        warp_id,
                        lane,
                        p_smem,
                        p_phase,
                        &mut running_max,
                        &mut running_sum,
                        &mut out_acc,
                    );
                    // Both S passes are drained and P is fenced into the
                    // async proxy: release the S buffer and publish P.
                    fence_proxy_async_shared_cta();
                    mbarrier_arrive(s_free.add(buffer));
                    mbarrier_arrive(&raw const P_FULL);
                    while !mbarrier_try_wait_parity(&raw const O_FULL, i & 1) {}
                    if tid == 0 {
                        mbarrier_arrive(kv_free.add((i as usize) % PIPELINE_STAGES));
                    }
                    merge_output_tile(o_tmem, warp_id, &mut out_acc);
                    i += 1;
                }
                store_outputs(
                    batch,
                    t,
                    h,
                    head,
                    query_tile,
                    warp_id,
                    lane,
                    &running_max,
                    &running_sum,
                    &out_acc,
                    &mut output,
                    &mut logsumexp,
                );
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
                    let s_tmem = tmem + (buffer as u32) * 128;
                    let k_smem = k_ring.add(stage * TILE_BYTES);
                    let mut chunk = 0u64;
                    while chunk < 4 {
                        let a_descriptor = smem_descriptor(q_smem as u64 + chunk * 32);
                        let b_descriptor = smem_descriptor(k_smem as u64 + chunk * 32);
                        tcgen05_mma_f16(s_tmem, a_descriptor, b_descriptor, s_instruction, chunk > 0);
                        chunk += 1;
                    }
                    tcgen05_commit_shared_cluster(s_full.add(buffer) as *mut u64);
                    if i > 0 {
                        output_mma(
                            i - 1,
                            o_tmem,
                            p_smem,
                            v_ring,
                            o_instruction,
                            &raw const P_FULL,
                            &raw mut O_FULL,
                        );
                    }
                    i += 1;
                }
                output_mma(
                    key_tiles - 1,
                    o_tmem,
                    p_smem,
                    v_ring,
                    o_instruction,
                    &raw const P_FULL,
                    &raw mut O_FULL,
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
            }
        }
    }
}
