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
//! - `T` is a multiple of the 64-row tile; `HD == 128`; non-aligned shapes
//!   stay on the fp32 tiled kernels in `lib.rs`;
//! - outputs keep the existing contract: fp32 `y[B*T, H*HD]` and fp32
//!   `logsumexp[B*T, H]` in natural-log units.
//!
//! One CTA workstream owns a 64-query tile of one `(batch, head)` and
//! streams 64-key tiles: TMA loads Q/K/V into swizzled shared tiles (each
//! 128-wide head panel is two stacked 64-wide SWIZZLE_128B subtiles),
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
//! Two synchronous backward kernels share the same idioms —
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
use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::{DynamicSharedArray, SharedArray};
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
    cvt_f32x2_bf16x2, tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync,
};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, launch_bounds, thread, warp};
use kittens::ldst::{stmatrix_m8n8_x2, store_fragment_bf16};
use kittens::mma::{self, mma_ab, mma_abt};
use kittens::pipeline;
use kittens::reg::{
    Fragment, RegTile, RegVec, exp2_approx, fmax, log2_approx, online_rescale, value_column,
};
use kittens::shared::{Bf16, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{PhasedSemaphore, Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_block, dealloc_block};

// Tile contract; `host.rs` mirrors these as FLASH_TILE / FLASH_HD (kept
// non-pub here so SWEEP's one-definition rule never sees two copies).
//
// The head width is 128, but every SWIZZLE_128B shared tile is stored as one
// or two 64-wide (128-byte-row) subtiles so the swizzle phase still equals the
// row index inside each subtile — the coincidence HD=64 gave for free. A
// 128-wide operand (Q/K/V) is two stacked `[TILE, 64]` subtiles a
// `SUBTILE_BYTES` apart; the 64-wide P/dS operands are a single subtile. HD-deep
// MMAs (`S = Q·Kᵀ`) walk 8 K=16 chunks across the two subtiles; HD-wide MMAs
// (`O = P·V`, the gradient MMAs) split into two N=64 accumulations, one per V/K
// subtile.
const TILE: usize = 64;
const HD: usize = 128;

/// One `[TILE, HD]` bf16 operand panel as a kittens tile — two stacked
/// SWIZZLE_128B subtiles, the layout described above. Phase 1 of issue #61
/// moves the pipelined kernel's loader warp onto these; later phases absorb
/// the rest of the raw swizzle/barrier machinery.
type Panel = SharedTile<Bf16, TILE, HD, Swizzle128B>;
/// An `N`-deep ring of panels (the K and V streams); the pipelined and
/// persistent kernels pick their own stage depths.
type PanelRingN<const N: usize> = SharedTileRing<Bf16, TILE, HD, Swizzle128B, N>;
type PanelRing = PanelRingN<PIPELINE_STAGES>;
type PersistentPanelRing = PanelRingN<PERSISTENT_STAGES>;
/// The single-subtile `[TILE, TILE]` bf16 probability tile the softmax
/// warpgroup writes with swizzled `stmatrix` stores. Its `swizzled_chunk`
/// folds in the tile base's absolute 128-byte row phase — the fact the
/// per-kernel `p_phase` variables used to carry by hand.
type PTile = SharedTile<Bf16, TILE, TILE, Swizzle128B>;
/// 7e15's paired K-major operand: two adjacent 64-row tiles stacked into
/// `[2·TILE, HD]`, so each of its two HD subtiles is 128 rows and the
/// K-major MMA walk strides `TILE_BYTES` between them (Q for backward dQ,
/// K/V for backward dK/dV).
type PairedPanel = SharedTile<Bf16, { 2 * TILE }, HD, Swizzle128B>;
/// The paired single-subtile `[2·TILE, TILE]` bf16 operand the backward
/// kernels store through `store_fragment_bf16` (dS, Pᵀ, dSᵀ): 128 swizzled
/// 128-byte rows feeding the gradient MMAs' A side.
type PairedPTile = SharedTile<Bf16, { 2 * TILE }, TILE, Swizzle128B>;
const _: () = assert!(Panel::BYTES == TILE_BYTES && Panel::SUBTILE_BYTES == SUBTILE_BYTES);
const _: () = assert!(PTile::BYTES == SUBTILE_BYTES);
const _: () = assert!(PairedPanel::SUBTILE_BYTES == TILE_BYTES);
const _: () = assert!(PairedPTile::BYTES == 2 * SUBTILE_BYTES && PairedPTile::SUBTILES == 1);

/// The `S = Q·Kᵀ` (and `dP = dY·Vᵀ`) score accumulator segment. `M128_N64`
/// writes 128 TMEM rows; the forward kernels drain the real 64, the paired
/// backwards all 128.
type STmem = TmemTile<{ 2 * TILE }, TILE>;
/// An HD-wide output/gradient accumulator segment (`O`, `dQ`, `dK`, `dV`) —
/// two 64-column MMA bands side by side.
type AccTmem = TmemTile<{ 2 * TILE }, HD>;

/// Bytes of one full-width bf16 `[TILE, HD]` operand (two stacked subtiles).
const TILE_BYTES: usize = TILE * HD * 2;
/// Bytes of one 64-wide `[TILE, 64]` SWIZZLE_128B subtile — half a `TILE_BYTES`
/// panel, and the footprint of a `[TILE, TILE]` P/dS probability operand.
const SUBTILE_BYTES: usize = TILE_BYTES / 2;
/// Phantom-read pad, used by the three FORWARD kernels (sync, pipelined,
/// persistent) — all still on the `M128`-over-64-row scheme. Each of their MMAs
/// reads 128 operand rows but only fills a 64-row tile, so the unused rows
/// 64..128 stream a `TILE_BYTES` tail past each operand's base; one extra
/// `TILE_BYTES` at the end of the plan keeps those reads in bounds (the garbage
/// lands in accumulator rows 64..128, never drained). The BACKWARD kernels
/// (Design B, #47 item 2) instead PAIR two adjacent 64-row tiles into every
/// M128 MMA — all 128 rows are real — so their plans carry no `PHANTOM_PAD`.
///
/// The natural `M64_N64` shape is NOT an option: its shared-memory A operand
/// reads only 32 distinct rows. Measured with `transpose_b_probe` (`A[m,k]=m`,
/// `B=1`), accumulator row `m` reads A-row `16*(m>>5) + (m&15)` — rows `m` and
/// `m+16` alias and A-rows 32..63 are never read at all, i.e. the read drops
/// datapath-lane bits [5:4]. It is a property of the `M64` SS-operand→datapath
/// mapping, not of the descriptor: sweeping the A stride-byte-offset
/// (512/1024/2048) moves *which* 32 rows are read and never reaches 64. The
/// forward pairing that would recover the wasted half is correct but measured
/// SLOWER (226 → 170 TFLOP/s) — it fuses the two query tiles into one
/// warpgroup and so destroys the persistent kernel's ping-pong.
const PHANTOM_PAD: usize = TILE_BYTES;
/// Dynamic shared plan of the synchronous kernel: Q, K, V panels plus the
/// single P subtile.
pub const FLASH_DYNAMIC_SMEM: usize = 3 * TILE_BYTES + SUBTILE_BYTES + PHANTOM_PAD;

/// K/V ring depth of the pipelined kernel (SWEEP knob). Two is the floor:
/// the staggered issue order (`S-MMA(i)` before `O-MMA(i-1)`) needs one
/// stage of load-ahead to make progress. Four is the ceiling the host-side
/// launch allocation (`host::FLASH_PIPELINE_SMEM_BYTES`) is sized for.
pub const PIPELINE_STAGES: usize = 3;
const _: () = assert!(2 <= PIPELINE_STAGES && PIPELINE_STAGES <= 4);
/// Dynamic shared plan of the pipelined kernel: Q, the K and V rings, and
/// the single P subtile.
pub const FLASH_PIPELINE_SMEM: usize =
    (1 + 2 * PIPELINE_STAGES) * TILE_BYTES + SUBTILE_BYTES + PHANTOM_PAD;
/// Threads of the pipelined kernel: the softmax/correction/epilogue
/// warpgroup plus the TMA-load warp and the MMA-issue warp.
pub const FLASH_PIPELINE_BLOCK: usize = TILE + 64;
/// `#[launch_bounds]` only accepts integer literals, so the kernel's
/// `.maxntid` is spelled out; keep it equal to the block width. Declaring more
/// threads than are launched makes ptxas budget registers for a block that
/// never exists (`65536 / maxntid` per thread), which is how the HD=128
/// conversion's stale 128-row-era values silently squeezed the register-hungry
/// softmax warpgroup.
const _: () = assert!(FLASH_PIPELINE_BLOCK == 128);

/// Dynamic shared plan of the PAIRED query-parallel backward (kernel A, Design
/// B): the resident stacked `[Q_A;Q_B]` and `[dY_A;dY_B]` operands
/// (`2 * TILE_BYTES` each), the streamed K and V panels, and the single stacked
/// `[128, 64]` dS tile (`TILE_BYTES`). No `PHANTOM_PAD` — every paired MMA fills
/// 128 real query rows.
pub const FLASH_BACKWARD_Q_SMEM: usize = 7 * TILE_BYTES;
/// Dynamic shared plan of the PAIRED key-parallel backward (kernel B, Design B):
/// the resident stacked `[K_A;K_B]` and `[V_A;V_B]` operands (`2 * TILE_BYTES`
/// each), the streamed Q and dY panels, and the stacked `[128, 64]` Pᵀ and dSᵀ
/// tiles (`TILE_BYTES` each). No `PHANTOM_PAD`.
pub const FLASH_BACKWARD_KV_SMEM: usize = 8 * TILE_BYTES;

/// K/V ring depth of the warp-specialized backward (SWEEP knob). Two is the
/// floor for the same reason as `PIPELINE_STAGES`: the staggered issue order
/// (`S/dP-MMA(i)` before `dQ-MMA(i-1)`) needs a stage of load-ahead, and the
/// K stage of tile `i` cannot recycle until `dQ-MMA(i)` — which reads it a
/// second time — has been observed.
pub const BACKWARD_STAGES: usize = 3;
const _: () = assert!(2 <= BACKWARD_STAGES && BACKWARD_STAGES <= 4);
/// Dynamic shared plan of the PIPELINED query-parallel backward: the resident
/// stacked Q and dY pairs (`2 * TILE_BYTES` each), the K and V rings, and the
/// single stacked `[128, 64]` dS tile. dS stays single-buffered for the same
/// reason P does in the forward — the warpgroup's `dq_done(i)` wait proves the
/// gradient MMA finished reading it before tile `i+1` overwrites it.
pub const FLASH_BACKWARD_Q_PIPELINED_SMEM: usize = (5 + 2 * BACKWARD_STAGES) * TILE_BYTES;
/// Threads of the pipelined backward: the 128-thread gradient warpgroup (the
/// paired `S` is 128 real rows, so it takes four warps, not the forward's two)
/// plus the TMA-load warp and the MMA-issue warp.
pub const FLASH_BACKWARD_Q_PIPELINED_BLOCK: usize = 2 * TILE + 64;
/// Mirrors the `FLASH_PIPELINE_BLOCK` `.maxntid` note.
const _: () = assert!(FLASH_BACKWARD_Q_PIPELINED_BLOCK == 192);
/// Dynamic shared plan of the PIPELINED key-parallel backward: the resident
/// stacked K and V pairs, the Q and dY rings, and the stacked Pᵀ and dSᵀ
/// tiles (both single-buffered, released by the same `grad_done` wait that
/// recycles the ring).
pub const FLASH_BACKWARD_KV_PIPELINED_SMEM: usize = (6 + 2 * BACKWARD_STAGES) * TILE_BYTES;
/// Threads of the pipelined key-parallel backward; same three roles.
pub const FLASH_BACKWARD_KV_PIPELINED_BLOCK: usize = 2 * TILE + 64;

/// Base-2 slack a tile's row max may climb above the O segment's reference
/// before the warpgroup forces a correction (SWEEP knob). P values reach at
/// most `2^CORRECTION_THRESHOLD`, comfortably inside bf16 range and the fp32
/// accumulation headroom of a full key stream.
pub const CORRECTION_THRESHOLD: f32 = 8.0;

/// K/V ring depth of the persistent kernel: `PIPELINE_STAGES` capped at 3 so
/// the doubled Q/P footprint of two workstreams stays inside the ~227 KiB
/// shared-memory budget.
pub const PERSISTENT_STAGES: usize = if PIPELINE_STAGES < 3 {
    PIPELINE_STAGES
} else {
    3
};
/// Dynamic shared plan of the persistent kernel: two Q panels, the K and V
/// rings, and one P subtile per workstream.
pub const FLASH_PERSISTENT_SMEM: usize =
    (2 + 2 * PERSISTENT_STAGES) * TILE_BYTES + 2 * SUBTILE_BYTES + PHANTOM_PAD;
/// Threads of the persistent kernel: two softmax warpgroups plus the
/// TMA-load warp and the MMA-issue warp.
pub const FLASH_PERSISTENT_BLOCK: usize = 2 * TILE + 64;
/// Mirrors the `FLASH_PIPELINE_BLOCK` `.maxntid` note.
const _: () = assert!(FLASH_PERSISTENT_BLOCK == 192);

/// Finite stand-in for "masked" in the base-2 score domain; far enough below
/// any real score that `exp2` flushes it to a subnormal-scale value while the
/// running-max recurrence stays NaN-free.
const MASKED_SCORE: f32 = -1.0e30;

#[cuda_module]
pub mod kernels {
    use super::*;

    const LN2: f32 = 0.693_147_18;
    /// Softmax scale for `HD == 128` (`1/sqrt(128)`), written as a literal
    /// because `1.0/(HD as f32).sqrt()` would lower to libdevice `sqrtf`.
    const SCALE: f32 = 0.088_388_35;
    /// `log2(e)`, converting the saved natural-log LSE into the base-2 domain
    /// the recomputed probabilities live in.
    const LOG2E: f32 = 1.442_695_04;

    // The gemm-encoded operand descriptors and the chained K=16 MMA walks
    // (score/gradient, plain and 7e15-paired) now live in kittens::shared
    // (`operand_descriptor`) and kittens::mma (`mma_abt`/`mma_ab`) — same
    // bits, same issue order.

    // The pure-PTX-safe scalar maps (fmax/fmin/exp2_approx/log2_approx) and
    // the quad reductions now live in kittens::reg; the swizzled stmatrix
    // mover in kittens::ldst. Same code, same bits — see their docs for the
    // libdevice discipline they encode.

    /// One key tile of register softmax, shared by the three forward
    /// kernels. `warp_id` and every row coordinate are warpgroup-local: the
    /// persistent kernel's second warpgroup passes `warp::warp_id() - 4`.
    ///
    /// Drains `S[128, 128]` from `s_tmem` twice — pass 1 for masked row
    /// maxima, pass 2 to exponentiate against the O segment's per-row max
    /// reference `m_ref` — and stores bf16 probabilities into the P tile via
    /// `stmatrix` at `PTile::swizzled_chunk` addresses (the 16-byte-chunk
    /// XOR the TMA swizzle would have produced, absolute base phase folded
    /// in, so the O-MMA descriptors read P exactly like a TMA-loaded
    /// operand).
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
        s_tmem: STmem,
        o_tmem: AccTmem,
        tile: u32,
        diagonal: bool,
        warp_id: u32,
        lane: u32,
        p: PTile,
        votes: *mut u32,
        vote: Semaphore,
        m_ref: &mut RegVec<4>,
        running_sum: &mut RegVec<4>,
        out_acc: &mut RegTile<4, 32>,
    ) -> bool {
        unsafe {
            let quad = (lane % 4) as usize;
            let row_in_16 = (lane / 4) as usize;

            // Pass 1: tile row maxima (masked, base-2 domain).
            let mut tile_max = RegVec::<4>::splat(MASKED_SCORE);
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 4 {
                    let column = column_block * 16;
                    let (low, high) = s_tmem.fragment(tmem_row, column);
                    let slot_a = (row_block * 2) as usize;
                    let slot_b = slot_a + 1;
                    if diagonal {
                        let row_a = (tmem_row as usize + row_in_16) as u32;
                        let row_b = row_a + 8;
                        let col = column + 2 * quad as u32;
                        let mut max_a = tile_max.0[slot_a];
                        max_a = fmax(max_a, if col <= row_a { low[0] } else { MASKED_SCORE });
                        max_a = fmax(
                            max_a,
                            if col + 1 <= row_a {
                                low[1]
                            } else {
                                MASKED_SCORE
                            },
                        );
                        max_a = fmax(
                            max_a,
                            if col + 8 <= row_a {
                                high[0]
                            } else {
                                MASKED_SCORE
                            },
                        );
                        max_a = fmax(
                            max_a,
                            if col + 9 <= row_a {
                                high[1]
                            } else {
                                MASKED_SCORE
                            },
                        );
                        tile_max.0[slot_a] = max_a;
                        let mut max_b = tile_max.0[slot_b];
                        max_b = fmax(max_b, if col <= row_b { low[2] } else { MASKED_SCORE });
                        max_b = fmax(
                            max_b,
                            if col + 1 <= row_b {
                                low[3]
                            } else {
                                MASKED_SCORE
                            },
                        );
                        max_b = fmax(
                            max_b,
                            if col + 8 <= row_b {
                                high[2]
                            } else {
                                MASKED_SCORE
                            },
                        );
                        max_b = fmax(
                            max_b,
                            if col + 9 <= row_b {
                                high[3]
                            } else {
                                MASKED_SCORE
                            },
                        );
                        tile_max.0[slot_b] = max_b;
                    } else {
                        tile_max.0[slot_a] = fmax(
                            fmax(fmax(tile_max.0[slot_a], low[0]), fmax(low[1], high[0])),
                            high[1],
                        );
                        tile_max.0[slot_b] = fmax(
                            fmax(fmax(tile_max.0[slot_b], low[2]), fmax(low[3], high[2])),
                            high[3],
                        );
                    }
                    column_block += 1;
                }
                row_block += 1;
            }
            let row_max = tile_max.quad_max();
            let exceed = row_max.any_exceeds(*m_ref, CORRECTION_THRESHOLD);

            // Collective correction vote (tile 0 always trips it: m_ref
            // still sits at MASKED_SCORE). One word per warp, one barrier
            // phase per tile.
            let parity = tile & 1;
            let warp_vote = warp::any(exceed);
            if lane == 0 {
                *votes.add((parity * 4 + warp_id) as usize) = warp_vote as u32;
            }
            vote.arrive();
            vote.wait(parity);
            // Only two warps make up the softmax warpgroup at TILE=64, so the
            // warpgroup-wide OR spans two vote words.
            let base = (parity * 4) as usize;
            let correction = (*votes.add(base) | *votes.add(base + 1)) != 0;

            if correction {
                if tile > 0 {
                    merge_output_tile(o_tmem, warp_id, out_acc);
                }
                online_rescale(m_ref, row_max, running_sum, out_acc);
            }

            // Pass 2: probabilities — re-drain S, exponentiate against the
            // segment reference, accumulate row sums, and store bf16 P
            // through the swizzle-aware stmatrix addresses (base phase
            // hoisted once, like the old `p_phase`).
            let p_chunks = p.chunk_writer();
            let mut tile_sum = RegVec::<4>::splat(0.0);
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 4 {
                    let column = column_block * 16;
                    let (low, high) = s_tmem.fragment(tmem_row, column);
                    let slot_a = (row_block * 2) as usize;
                    let slot_b = slot_a + 1;
                    let row_a = (tmem_row as usize + row_in_16) as u32;
                    let row_b = row_a + 8;
                    let col = column + 2 * quad as u32;

                    let s_a0 = if !diagonal || col <= row_a {
                        low[0]
                    } else {
                        MASKED_SCORE
                    };
                    let s_a1 = if !diagonal || col + 1 <= row_a {
                        low[1]
                    } else {
                        MASKED_SCORE
                    };
                    let s_a8 = if !diagonal || col + 8 <= row_a {
                        high[0]
                    } else {
                        MASKED_SCORE
                    };
                    let s_a9 = if !diagonal || col + 9 <= row_a {
                        high[1]
                    } else {
                        MASKED_SCORE
                    };
                    let s_b0 = if !diagonal || col <= row_b {
                        low[2]
                    } else {
                        MASKED_SCORE
                    };
                    let s_b1 = if !diagonal || col + 1 <= row_b {
                        low[3]
                    } else {
                        MASKED_SCORE
                    };
                    let s_b8 = if !diagonal || col + 8 <= row_b {
                        high[2]
                    } else {
                        MASKED_SCORE
                    };
                    let s_b9 = if !diagonal || col + 9 <= row_b {
                        high[3]
                    } else {
                        MASKED_SCORE
                    };
                    let p_a0 = exp2_approx(s_a0 - m_ref.0[slot_a]);
                    let p_a1 = exp2_approx(s_a1 - m_ref.0[slot_a]);
                    let p_a8 = exp2_approx(s_a8 - m_ref.0[slot_a]);
                    let p_a9 = exp2_approx(s_a9 - m_ref.0[slot_a]);
                    let p_b0 = exp2_approx(s_b0 - m_ref.0[slot_b]);
                    let p_b1 = exp2_approx(s_b1 - m_ref.0[slot_b]);
                    let p_b8 = exp2_approx(s_b8 - m_ref.0[slot_b]);
                    let p_b9 = exp2_approx(s_b9 - m_ref.0[slot_b]);
                    tile_sum.0[slot_a] += p_a0 + p_a1 + p_a8 + p_a9;
                    tile_sum.0[slot_b] += p_b0 + p_b1 + p_b8 + p_b9;

                    // P is a single 64-wide subtile (128-byte rows), so the
                    // eight 16-byte chunks index the whole row directly.
                    let chunk_low = (column_block * 2) as usize;
                    let chunk = if (8..16).contains(&lane) {
                        chunk_low + 1
                    } else {
                        chunk_low
                    };
                    let row_low = tmem_row as usize + (lane % 8) as usize;
                    let row_high = row_low + 8;
                    stmatrix_m8n8_x2(
                        p_chunks.at(row_low, chunk),
                        cvt_f32x2_bf16x2(p_a0, p_a1),
                        cvt_f32x2_bf16x2(p_a8, p_a9),
                    );
                    stmatrix_m8n8_x2(
                        p_chunks.at(row_high, chunk),
                        cvt_f32x2_bf16x2(p_b0, p_b1),
                        cvt_f32x2_bf16x2(p_b8, p_b9),
                    );
                    column_block += 1;
                }
                row_block += 1;
            }
            running_sum.add_assign(tile_sum.quad_sum());
            correction
        }
    }

    /// Drain the `O = P·V` TMEM segment and add it into the per-thread
    /// output accumulator. Called by `softmax_tile` on a correction (which
    /// then rescales the merged accumulator to the new reference) and by
    /// the epilogue for the final segment. `warp_id` is warpgroup-local.
    #[inline(always)]
    unsafe fn merge_output_tile(o_tmem: AccTmem, warp_id: u32, out_acc: &mut RegTile<4, 32>) {
        unsafe {
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 8 {
                    let column = column_block * 16;
                    let (low, high) = o_tmem.fragment(tmem_row, column);
                    let slot_a = (row_block * 2) as usize;
                    let slot_b = slot_a + 1;
                    let base = (column_block * 4) as usize;
                    out_acc.0[slot_a][base] += low[0];
                    out_acc.0[slot_a][base + 1] += low[1];
                    out_acc.0[slot_a][base + 2] += high[0];
                    out_acc.0[slot_a][base + 3] += high[1];
                    out_acc.0[slot_b][base] += low[2];
                    out_acc.0[slot_b][base + 1] += low[3];
                    out_acc.0[slot_b][base + 2] += high[2];
                    out_acc.0[slot_b][base + 3] += high[3];
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
        max_ref: &RegVec<4>,
        running_sum: &RegVec<4>,
        out_acc: &RegTile<4, 32>,
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
                let global_row = (batch * t) as usize + query_tile as usize * TILE + local_row;
                let inverse = 1.0 / running_sum.0[slot];
                let out_base = global_row * d_model + head as usize * HD;
                let mut column_block = 0usize;
                while column_block < 8 {
                    let column = column_block * 16 + 2 * quad;
                    let base = column_block * 4;
                    *output.get_unchecked_mut(out_base + column) = out_acc.0[slot][base] * inverse;
                    *output.get_unchecked_mut(out_base + column + 1) =
                        out_acc.0[slot][base + 1] * inverse;
                    *output.get_unchecked_mut(out_base + column + 8) =
                        out_acc.0[slot][base + 2] * inverse;
                    *output.get_unchecked_mut(out_base + column + 9) =
                        out_acc.0[slot][base + 3] * inverse;
                    column_block += 1;
                }
                if quad == 0 {
                    *logsumexp.get_unchecked_mut(global_row * h as usize + head as usize) =
                        LN2 * (max_ref.0[slot] + log2_approx(running_sum.0[slot]));
                }
                slot += 1;
            }
        }
    }

    /// Issue one `S = Q·Kᵀ` tile (M128_N64 over the 64-row tile, eight chained
    /// K=16 MMAs walking the two stacked HD subtiles of both operands) from the
    /// current leader thread; the caller owns the commit. The `M128` shape's
    /// phantom rows 64..128 read the operand's `TILE_BYTES` tail and land in
    /// accumulator rows the drain never touches.
    #[inline(always)]
    unsafe fn score_mma(s_tmem: u32, q: Panel, k: Panel, s_instruction: u32) {
        unsafe { mma_abt(s_tmem, q, k, s_instruction, false) }
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

    /// PAIRED query-parallel backward register pass (Design B, #47 item 2): the
    /// 128-thread warpgroup drains ONE stacked `S`/`dP` — warps 0–1 own
    /// accumulator rows 0..63 (query tile A), warps 2–3 own 64..127 (query tile
    /// B) — and forms `dS = P·(dP − dot)·scale` for all 128 rows into ONE
    /// stacked `[128, 64]` dS tile. `warp_id` is the ACTUAL block warp (0..3).
    /// The paired tiles are adjacent, so `lse2`/`dot` are the pair's 128
    /// contiguous query rows and index by the fragment's ROW; the causal edge
    /// compares against the query row WITHIN the tile (`row & 63`). A future
    /// tile for the lower stream (`key > tile_b` → `key == tile_a`'s partner) is
    /// fully masked to `dS = 0`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn backward_q_tile_paired(
        s_tmem: STmem,
        dp_tmem: STmem,
        key: u32,
        tile_a: u32,
        tile_b: u32,
        warp_id: u32,
        lane: u32,
        lse2: *const f32,
        dot: *const f32,
        ds: PairedPTile,
    ) {
        unsafe {
            let ds_chunks = ds.chunk_writer();
            let lane_column = 2 * (lane % 4);
            let row_in_16 = lane / 4;
            let my_tile = if warp_id < 2 { tile_a } else { tile_b };
            let masked_all = key > my_tile;
            let diagonal = key == my_tile;
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column = 0u32;
                while column < TILE as u32 {
                    let s = s_tmem.fragment_tile(tmem_row, column);
                    let dp = dp_tmem.fragment_tile(tmem_row, column);
                    let mut ds_tile = Fragment::zero();
                    let mut slot = 0usize;
                    while slot < 2 {
                        let row = tmem_row + row_in_16 + 8 * slot as u32;
                        let query_row = row & 63;
                        let lse_row = *lse2.add(row as usize);
                        let dot_row = *dot.add(row as usize);
                        let mut value = 0usize;
                        while value < 4 {
                            let key_column = column + lane_column + value_column(value) as u32;
                            ds_tile.0[slot][value] = backward_dscore(
                                s.0[slot][value],
                                dp.0[slot][value],
                                lse_row,
                                dot_row,
                                SCALE,
                                !masked_all && (!diagonal || key_column <= query_row),
                            );
                            value += 1;
                        }
                        slot += 1;
                    }
                    store_fragment_bf16(ds_chunks, tmem_row, column, lane, ds_tile);
                    column += 16;
                }
                row_block += 1;
            }
        }
    }

    /// PAIRED key-parallel backward register pass (Design B, #47 item 2): the
    /// 128-thread warpgroup drains ONE stacked transposed `Sᵀ`/`dPᵀ` — warps
    /// 0–1 own accumulator rows 0..63 (key tile A), warps 2–3 own 64..127 (key
    /// tile B) — and forms `Pᵀ`/`dSᵀ` for all 128 key rows into ONE stacked
    /// `[128, 64]` tile each. Rows are key rows and columns are query rows, so
    /// this is the transpose of kernel A's indexing in both directions: the
    /// per-row statistics become a per-VALUE gather over the streamed query
    /// tile's 64 rows, and the causal edge compares the key row within the tile
    /// (`row & 63`) against the query column. A query tile before the higher
    /// key stream (`query < key_b` → the shared diagonal query `key_a`) fully
    /// masks that stream to zero.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn backward_kv_tile_paired(
        st_tmem: STmem,
        dpt_tmem: STmem,
        query: u32,
        key_a: u32,
        key_b: u32,
        warp_id: u32,
        lane: u32,
        lse2: *const f32,
        dot: *const f32,
        p: PairedPTile,
        ds: PairedPTile,
    ) {
        unsafe {
            let p_chunks = p.chunk_writer();
            let ds_chunks = ds.chunk_writer();
            let lane_column = 2 * (lane % 4);
            let row_in_16 = lane / 4;
            let my_key = if warp_id < 2 { key_a } else { key_b };
            let masked_all = query < my_key;
            let diagonal = query == my_key;
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column = 0u32;
                while column < TILE as u32 {
                    let st = st_tmem.fragment_tile(tmem_row, column);
                    let dpt = dpt_tmem.fragment_tile(tmem_row, column);
                    let mut lse_column = [0.0f32; 4];
                    let mut dot_column = [0.0f32; 4];
                    let mut value = 0usize;
                    while value < 4 {
                        let query_row = (column + lane_column) as usize + value_column(value);
                        lse_column[value] = *lse2.add(query_row);
                        dot_column[value] = *dot.add(query_row);
                        value += 1;
                    }
                    let mut p_tile = Fragment::zero();
                    let mut ds_tile = Fragment::zero();
                    let mut slot = 0usize;
                    while slot < 2 {
                        let key_row = (tmem_row + row_in_16 + 8 * slot as u32) & 63;
                        let mut value = 0usize;
                        while value < 4 {
                            let query_column = column + lane_column + value_column(value) as u32;
                            let keep = !masked_all && (!diagonal || key_row <= query_column);
                            let probability = if keep {
                                exp2_approx(st.0[slot][value] - lse_column[value])
                            } else {
                                0.0
                            };
                            p_tile.0[slot][value] = probability;
                            ds_tile.0[slot][value] =
                                probability * (dpt.0[slot][value] - dot_column[value]) * LN2;
                            value += 1;
                        }
                        slot += 1;
                    }
                    store_fragment_bf16(p_chunks, tmem_row, column, lane, p_tile);
                    store_fragment_bf16(ds_chunks, tmem_row, column, lane, ds_tile);
                    column += 16;
                }
                row_block += 1;
            }
        }
    }

    /// Drain a 128-column gradient accumulator and store fp32 straight to
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
        grad_acc: &RegTile<4, 32>,
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
                while column_block < 8 {
                    let column = column_block * 16 + 2 * quad;
                    let base = column_block * 4;
                    *output.get_unchecked_mut(out_base + column) = grad_acc.0[slot][base];
                    *output.get_unchecked_mut(out_base + column + 1) = grad_acc.0[slot][base + 1];
                    *output.get_unchecked_mut(out_base + column + 8) = grad_acc.0[slot][base + 2];
                    *output.get_unchecked_mut(out_base + column + 9) = grad_acc.0[slot][base + 3];
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

    /// Dumps the raw shared-memory word layout of one TMA-loaded `[TILE, 64]`
    /// bf16 subtile, plus the subtile's absolute 128-byte row phase as a
    /// trailing word. The P-write path mirrors TMA's SWIZZLE_128B placement —
    /// which XORs *absolute* address bits [9:7] — with manual address XORs; the
    /// host fills the staging tile with sequential word indices and verifies
    /// the exact permutation from this dump.
    #[kernel]
    pub unsafe fn swizzle_probe(src_tma: *const TmaDescriptor, mut output: DisjointSlice<u32>) {
        unsafe {
            static mut TMA_BARRIER: Barrier = Barrier::UNINIT;

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let tile = PTile::from_raw(smem);
            let tma = Semaphore::attach(&raw mut TMA_BARRIER);
            let tid = thread::threadIdx_x();
            if tid == 0 {
                tma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if tid == 0 {
                tile.tma_load(src_tma, 0, 0, tma);
                tma.expect_tx(PTile::BYTES as u32);
            }
            tma.wait(0);
            thread::sync_threads();

            let words = smem as *const u32;
            let mut index = tid as usize;
            while index < SUBTILE_BYTES / 4 {
                *output.get_unchecked_mut(index) = *words.add(index);
                index += TILE;
            }
            if tid == 0 {
                *output.get_unchecked_mut(SUBTILE_BYTES / 4) = tile.swizzle_phase() as u32;
            }
            thread::sync_threads();
            if tid == 0 {
                tma.inval();
            }
        }
    }

    /// Validation kernel for the `S = Q·Kᵀ` operand path: one CTA computes
    /// `C[64, 64] = A[64, 64] · B[64, 64]` over the full 128-wide HD (two
    /// stacked subtiles, K=128) through the real `score_mma` walk and the
    /// `M128`-over-64-row accumulator, then drains rows 0..63 through the
    /// decoded (row, column) fragment map. A failure here isolates the
    /// operand descriptor / fragment map from the softmax and epilogue.
    ///
    /// It is also the harness that settled why `M64_N64` is unusable (see
    /// `PHANTOM_PAD`): seed A and B host-side with structured encodings and
    /// solve the raw dump offline. `A[m,k]=m, B=1` reads back the A-row each
    /// accumulator row actually consumed — the M-broadcast probe; `A[m,k]=k`
    /// against one-hot `B` reads back the K-index pairing (identity, so there
    /// is no K-permutation); `A=1, B[n,k]=2^(k/16)` reads back the per-chunk
    /// histogram. Only dense random `A·B` fails, because every other encoding
    /// is invariant under the row aliasing.
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
            let a = Panel::from_raw(smem);
            let b = Panel::from_raw(smem.add(TILE_BYTES));

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let is_leader = tid == 0;

            let tma = Semaphore::attach(&raw mut TMA_BARRIER);
            let mma = Semaphore::attach(&raw mut MMA_BARRIER);
            if is_leader {
                tma.init(1);
                mma.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            let tmem = alloc_block(&raw mut TMEM_ADDRESS as *mut u32, 512);
            let c_tmem = STmem::from_raw(tmem);

            if is_leader {
                a.tma_load(a_tma, 0, 0, tma);
                b.tma_load(b_tma, 0, 0, tma);
                tma.expect_tx((2 * TILE_BYTES) as u32);
            }
            tma.wait(0);
            thread::sync_threads();

            let instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N64)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            if is_leader {
                score_mma(c_tmem.raw(), a, b, instruction);
                mma::commit(mma);
            }
            mma.wait(0);
            thread::sync_threads();

            let lane_column = 2 * (lane % 4) as usize;
            let row_in_16 = (lane / 4) as usize;
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column = 0u32;
                while column < TILE as u32 {
                    let c = c_tmem.fragment_tile(tmem_row, column);
                    let mut slot = 0usize;
                    while slot < 2 {
                        let row = tmem_row as usize + row_in_16 + 8 * slot;
                        let mut value = 0usize;
                        while value < 4 {
                            let out_column = column as usize + lane_column + value_column(value);
                            *output.get_unchecked_mut(row * TILE + out_column) = c.0[slot][value];
                            value += 1;
                        }
                        slot += 1;
                    }
                    column += 16;
                }
                row_block += 1;
            }

            thread::sync_threads();
            dealloc_block(tmem, 512);
            if is_leader {
                tma.inval();
                mma.inval();
            }
        }
    }

    /// Synchronous tcgen05 causal attention forward. Launch with
    /// `host::flash_forward_config`: grid `(T/64, H, B)`, 64 threads,
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
            let q = Panel::from_raw(smem);
            let k = Panel::from_raw(smem.add(TILE_BYTES));
            let v = Panel::from_raw(smem.add(2 * TILE_BYTES));
            // The 128B swizzle XORs *absolute* shared-address bits [9:7], not
            // tile-relative rows; `PTile::swizzled_chunk` folds the base's
            // row phase in.
            let p = PTile::from_raw(smem.add(3 * TILE_BYTES));

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != TILE {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let is_leader = tid == 0;

            let tma_sem = Semaphore::attach(&raw mut TMA_BARRIER);
            let mma_sem = Semaphore::attach(&raw mut MMA_BARRIER);
            let vote = Semaphore::attach(&raw mut VOTE_BARRIER);

            let query_tile = thread::blockIdx_x();
            let head = thread::blockIdx_y();
            let batch = thread::blockIdx_z();
            let t = sequence_length;
            let h = heads;
            let plane = (batch * h + head) as i32;

            if is_leader {
                tma_sem.init(1);
                mma_sem.init(1);
                vote.init(TILE as u32);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            let tmem = alloc_block(&raw mut TMEM_ADDRESS as *mut u32, 512);
            let s_tmem = STmem::from_raw(tmem);
            let o_tmem = AccTmem::from_raw(tmem + 256);

            let s_instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N64)
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
                q.tma_load(q_tma, (query_tile * TILE as u32) as i32, plane, tma_sem);
                tma_sem.expect_tx(TILE_BYTES as u32);
            }
            tma_sem.wait(0);
            thread::sync_threads();

            let mut m_ref = RegVec::<4>::splat(MASKED_SCORE);
            let mut running_sum = RegVec::<4>::splat(0.0);
            let mut out_acc = RegTile::<4, 32>::zero();
            let mut corrections = 0u32;

            let mut tma_phase = 1u32;
            let mut mma_phase = 0u32;
            let mut key_tile = 0u32;
            while key_tile <= query_tile {
                if is_leader {
                    let key_row = (key_tile * TILE as u32) as i32;
                    k.tma_load(k_tma, key_row, plane, tma_sem);
                    v.tma_load(v_tma, key_row, plane, tma_sem);
                    tma_sem.expect_tx(2 * TILE_BYTES as u32);
                }
                tma_sem.wait(tma_phase & 1);
                tma_phase += 1;
                thread::sync_threads();

                // S = Q·Kᵀ, fresh accumulation each tile.
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    score_mma(s_tmem.raw(), q, k, s_instruction);
                    mma::commit(mma_sem);
                }
                mma_sem.wait(mma_phase & 1);
                mma_phase += 1;
                thread::sync_threads();

                let correction = softmax_tile(
                    s_tmem,
                    o_tmem,
                    key_tile,
                    key_tile == query_tile,
                    warp_id,
                    lane,
                    p,
                    &raw mut VOTES as *mut u32,
                    vote,
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
                    mma_ab(o_tmem.raw(), p, v, o_instruction, !correction);
                    mma::commit(mma_sem);
                }
                mma_sem.wait(mma_phase & 1);
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
                    .get_unchecked_mut(plane as usize * tiles + query_tile as usize) = corrections;
            }

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            dealloc_block(tmem, 512);
            if is_leader {
                tma_sem.inval();
                mma_sem.inval();
                vote.inval();
            }
        }
    }

    /// PAIRED tcgen05 query-parallel backward (Design B for #47 item 2). Launch
    /// with `host::flash_backward_q_config`: grid `(T/128, H, B)`, 128 threads,
    /// `FLASH_BACKWARD_Q_SMEM` dynamic shared bytes (opted in by the loader).
    ///
    /// One CTA owns a query-tile PAIR `(2p, 2p+1)` of one `(batch, head)` and
    /// streams the causal key tiles `0..=tile_b`. The two query tiles are
    /// STACKED into every MMA: `S = [Q_A;Q_B]·Kᵀ` and `dP = [dY_A;dY_B]·Vᵀ`
    /// fill all 128 accumulator rows, `dS` is formed for all 128 rows into one
    /// stacked tile, and `dQ += dS·K` drains 128 real rows (rows 0..63 = tile
    /// A's dQ, 64..127 = tile B's). Q/dY stay resident; the 128-thread
    /// warpgroup's warps 0–1 drain rows 0..63 and warps 2–3 drain 64..127.
    /// Because the paired tiles are adjacent their 128 query rows are
    /// contiguous, so LSE/dot stage directly and `store_grad_tile` writes the
    /// pair in one shot. Requires `T % 128 == 0` (host eligibility).
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
            let q = PairedPanel::from_raw(smem);
            let dy = PairedPanel::from_raw(smem.add(2 * TILE_BYTES));
            let k = Panel::from_raw(smem.add(4 * TILE_BYTES));
            let v = Panel::from_raw(smem.add(5 * TILE_BYTES));
            let ds = PairedPTile::from_raw(smem.add(6 * TILE_BYTES));

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != 2 * TILE {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let is_leader = tid == 0;

            let pair = thread::blockIdx_x();
            let head = thread::blockIdx_y();
            let batch = thread::blockIdx_z();
            let t = sequence_length;
            let h = heads;
            let plane = (batch * h + head) as i32;
            let tile_a = pair * 2;
            let tile_b = tile_a + 1;

            let mut tma = PhasedSemaphore::attach(&raw mut TMA_BARRIER);
            let mut mma = PhasedSemaphore::attach(&raw mut MMA_BARRIER);
            if is_leader {
                tma.sem().init(1);
                mma.sem().init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            let tmem = alloc_block(&raw mut TMEM_ADDRESS as *mut u32, 512);
            let s_tmem = STmem::from_raw(tmem);
            let dp_tmem = STmem::from_raw(tmem + 128);
            let dq_tmem = AccTmem::from_raw(tmem + 256);

            let s_instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N64)
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

            // The stacked Q/dY pairs stay operand-A resident for the whole key
            // stream: tile A's rows land at the operand's row 0, tile B's at
            // row TILE, one box per HD subtile each.
            if is_leader {
                q.tma_load_at(q_tma, 0, (tile_a * TILE as u32) as i32, plane, tma.sem());
                q.tma_load_at(q_tma, TILE, (tile_b * TILE as u32) as i32, plane, tma.sem());
                dy.tma_load_at(dy_tma, 0, (tile_a * TILE as u32) as i32, plane, tma.sem());
                dy.tma_load_at(dy_tma, TILE, (tile_b * TILE as u32) as i32, plane, tma.sem());
                tma.sem().expect_tx(4 * TILE_BYTES as u32);
            }
            tma.wait_next();

            // The pair's 128 contiguous query rows' base-2 LSE and softmax dot.
            let query_row = (batch * t) as usize + tile_a as usize * TILE + tid as usize;
            let stat_index = query_row * h as usize + head as usize;
            (*(&raw mut LSE2 as *mut f32).add(tid as usize)) = logsumexp[stat_index] * LOG2E;
            (*(&raw mut DOTS as *mut f32).add(tid as usize)) = dot[stat_index];
            thread::sync_threads();

            let mut dq_acc = RegTile::<4, 32>::zero();

            let mut key_tile = 0u32;
            while key_tile <= tile_b {
                if is_leader {
                    let key_row = (key_tile * TILE as u32) as i32;
                    k.tma_load(k_tma, key_row, plane, tma.sem());
                    v.tma_load(v_tma, key_row, plane, tma.sem());
                    tma.sem().expect_tx(2 * TILE_BYTES as u32);
                }
                tma.wait_next();
                thread::sync_threads();

                // S = [Q_A;Q_B]·Kᵀ and dP = [dY_A;dY_B]·Vᵀ, both fresh.
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    mma_abt(s_tmem.raw(), q, k, s_instruction, false);
                    mma_abt(dp_tmem.raw(), dy, v, s_instruction, false);
                    mma::commit(mma.sem());
                }
                mma.wait_next();
                thread::sync_threads();

                backward_q_tile_paired(
                    s_tmem,
                    dp_tmem,
                    key_tile,
                    tile_a,
                    tile_b,
                    warp_id,
                    lane,
                    &raw const LSE2 as *const f32,
                    &raw const DOTS as *const f32,
                    ds,
                );

                fence_proxy_async_shared_cta();
                tcgen05_fence_before_thread_sync();
                thread::sync_threads();

                // dQ += dS·K, continuing the accumulator (fresh on key 0).
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    mma_ab(dq_tmem.raw(), ds, k, grad_instruction, key_tile != 0);
                    mma::commit(mma.sem());
                }
                mma.wait_next();

                tcgen05_fence_before_thread_sync();
                thread::sync_threads();
                key_tile += 1;
            }

            merge_output_tile(dq_tmem, warp_id, &mut dq_acc);
            // The pair's rows are contiguous, so the block's base tile is
            // `tile_a` and the warp-derived local row (0..127) lands correctly.
            store_grad_tile(batch, t, h, head, tile_a, warp_id, lane, &dq_acc, &mut dq);

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            dealloc_block(tmem, 512);
            if is_leader {
                tma.sem().inval();
                mma.sem().inval();
            }
        }
    }

    /// PAIRED tcgen05 key-parallel backward (Design B for #47 item 2). Launch
    /// with `host::flash_backward_kv_config`: grid `(T/128, H, B)`, 128 threads,
    /// `FLASH_BACKWARD_KV_SMEM` dynamic shared bytes (opted in by the loader).
    ///
    /// One CTA owns a key-tile PAIR `(2p, 2p+1)` and streams the causal query
    /// tiles `key_a..T/64`. The two key tiles are STACKED into every MMA: the
    /// transposed `Sᵀ = [K_A;K_B]·Qᵀ` and `dPᵀ = [V_A;V_B]·dYᵀ` fill all 128
    /// accumulator rows (rows 0..63 = key A, 64..127 = key B), and `dV += Pᵀ·dY`
    /// / `dK += dSᵀ·Q` drain 128 real key rows. K/V stay resident; the streamed
    /// Q/dY are a single 64-row tile. The paired key rows are contiguous, so the
    /// gradients store in one shot at base tile `key_a`. The staged Q carries
    /// `scale·log2e`, so folding `ln2` into dSᵀ lands `scale` on dK. Requires
    /// `T % 128 == 0` (host eligibility).
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
            let k = PairedPanel::from_raw(smem);
            let v = PairedPanel::from_raw(smem.add(2 * TILE_BYTES));
            let q = Panel::from_raw(smem.add(4 * TILE_BYTES));
            let dy = Panel::from_raw(smem.add(5 * TILE_BYTES));
            let p = PairedPTile::from_raw(smem.add(6 * TILE_BYTES));
            let ds = PairedPTile::from_raw(smem.add(7 * TILE_BYTES));

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != 2 * TILE {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let is_leader = tid == 0;

            let pair = thread::blockIdx_x();
            let head = thread::blockIdx_y();
            let batch = thread::blockIdx_z();
            let t = sequence_length;
            let h = heads;
            let plane = (batch * h + head) as i32;
            let tiles = t / TILE as u32;
            let key_a = pair * 2;
            let key_b = key_a + 1;

            let mut tma = PhasedSemaphore::attach(&raw mut TMA_BARRIER);
            let mut mma = PhasedSemaphore::attach(&raw mut MMA_BARRIER);
            if is_leader {
                tma.sem().init(1);
                mma.sem().init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            let tmem = alloc_block(&raw mut TMEM_ADDRESS as *mut u32, 512);
            // Sᵀ/dPᵀ are 64-column segments; the dV/dK gradients are 128
            // columns, following at tmem+128 and tmem+256.
            let st_tmem = STmem::from_raw(tmem);
            let dpt_tmem = STmem::from_raw(tmem + 64);
            let dv_tmem = AccTmem::from_raw(tmem + 128);
            let dk_tmem = AccTmem::from_raw(tmem + 256);

            let s_instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N64)
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

            // The stacked K/V pairs stay operand-A resident for the whole query
            // stream.
            if is_leader {
                k.tma_load_at(k_tma, 0, (key_a * TILE as u32) as i32, plane, tma.sem());
                k.tma_load_at(k_tma, TILE, (key_b * TILE as u32) as i32, plane, tma.sem());
                v.tma_load_at(v_tma, 0, (key_a * TILE as u32) as i32, plane, tma.sem());
                v.tma_load_at(v_tma, TILE, (key_b * TILE as u32) as i32, plane, tma.sem());
                tma.sem().expect_tx(4 * TILE_BYTES as u32);
            }
            tma.wait_next();

            let mut dv_acc = RegTile::<4, 32>::zero();
            let mut dk_acc = RegTile::<4, 32>::zero();

            let mut query_tile = key_a;
            while query_tile < tiles {
                if is_leader {
                    let query_row = (query_tile * TILE as u32) as i32;
                    q.tma_load(q_tma, query_row, plane, tma.sem());
                    dy.tma_load(dy_tma, query_row, plane, tma.sem());
                    tma.sem().expect_tx(2 * TILE_BYTES as u32);
                }
                tma.wait_next();
                thread::sync_threads();

                // Stage this query tile's 64 rows' base-2 LSE and dot (indexed
                // by query column in the transposed register pass).
                if tid < TILE as u32 {
                    let global_row =
                        (batch * t) as usize + query_tile as usize * TILE + tid as usize;
                    let stat_index = global_row * h as usize + head as usize;
                    (*(&raw mut LSE2 as *mut f32).add(tid as usize)) =
                        logsumexp[stat_index] * LOG2E;
                    (*(&raw mut DOTS as *mut f32).add(tid as usize)) = dot[stat_index];
                }
                thread::sync_threads();

                // Sᵀ = [K_A;K_B]·Qᵀ and dPᵀ = [V_A;V_B]·dYᵀ, both fresh.
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    mma_abt(st_tmem.raw(), k, q, s_instruction, false);
                    mma_abt(dpt_tmem.raw(), v, dy, s_instruction, false);
                    mma::commit(mma.sem());
                }
                mma.wait_next();
                thread::sync_threads();

                backward_kv_tile_paired(
                    st_tmem,
                    dpt_tmem,
                    query_tile,
                    key_a,
                    key_b,
                    warp_id,
                    lane,
                    &raw const LSE2 as *const f32,
                    &raw const DOTS as *const f32,
                    p,
                    ds,
                );

                fence_proxy_async_shared_cta();
                tcgen05_fence_before_thread_sync();
                thread::sync_threads();

                // dV += Pᵀ·dY and dK += dSᵀ·Q, fresh on the first query tile.
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    let accumulate = query_tile != key_a;
                    mma_ab(dv_tmem.raw(), p, dy, grad_instruction, accumulate);
                    mma_ab(dk_tmem.raw(), ds, q, grad_instruction, accumulate);
                    mma::commit(mma.sem());
                }
                mma.wait_next();

                tcgen05_fence_before_thread_sync();
                thread::sync_threads();
                query_tile += 1;
            }

            merge_output_tile(dv_tmem, warp_id, &mut dv_acc);
            merge_output_tile(dk_tmem, warp_id, &mut dk_acc);
            store_grad_tile(batch, t, h, head, key_a, warp_id, lane, &dv_acc, &mut dv);
            store_grad_tile(batch, t, h, head, key_a, warp_id, lane, &dk_acc, &mut dk);

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            dealloc_block(tmem, 512);
            if is_leader {
                tma.sem().inval();
                mma.sem().inval();
            }
        }
    }

    /// Issue one tile's `dQ += dS·K` from the MMA warp: wait for the gradient
    /// warpgroup to publish dS, chain the four K=16 MMAs across K's two HD
    /// subtiles into the dQ segment (fresh on the first key tile, accumulating
    /// after), and commit completion into `dq_done`. That commit is what
    /// releases both the dS tile and the K ring stage — the K panel of tile `i`
    /// is read twice, once as the score MMA's B operand and once here.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn gradient_mma<const STAGES: usize>(
        i: u32,
        ds: PairedPTile,
        k: PanelRingN<STAGES>,
        dq_tmem: AccTmem,
        ds_full: Semaphore,
        dq_done: Semaphore,
        instruction: u32,
    ) {
        unsafe {
            ds_full.wait(i & 1);
            mma_ab(dq_tmem.raw(), ds, k.tile(i), instruction, i != 0);
            mma::commit(dq_done);
        }
    }

    /// Warp-specialized pipelined query-parallel backward (issue #61 phase 5:
    /// the first kernel written kittens-first). Launch with
    /// `host::flash_backward_q_pipelined_config`: grid `(T/128, H, B)`,
    /// `FLASH_BACKWARD_Q_PIPELINED_BLOCK` threads,
    /// `host::FLASH_BACKWARD_Q_PIPELINED_SMEM_BYTES` dynamic shared bytes.
    /// Identical operand, statistic, and output contract to
    /// `flash_backward_q_tcgen05` — same PAIRED Design-B math, same causal
    /// masking, same fragment map — with the synchronous kernel's four
    /// block-wide `sync_threads` per key tile replaced by the forward's
    /// three-role warp specialization:
    ///
    /// - warp 4's leader streams TMA: the resident Q/dY pairs once (charged
    ///   onto the first K/V stage), then the K/V ring running `BACKWARD_STAGES`
    ///   tiles ahead;
    /// - warp 5's leader issues MMAs, staggered exactly like the forward so
    ///   `S/dP-MMA(i)` reaches the tensor core before `dQ-MMA(i-1)`: while the
    ///   warpgroup forms `dS(i)`, the core is already producing `S(i+1)`;
    /// - warps 0–3 are the gradient warpgroup: wait `s_full`, run
    ///   `backward_q_tile_paired`, release `s_free`, publish `ds_full`, wait
    ///   `dq_done`, recycle the K/V stage. The dQ accumulator lives in TMEM
    ///   for the whole key stream and is drained once at the end, so unlike
    ///   the forward there is no per-tile correction to fuse in here.
    ///
    /// Where the synchronous kernel exposed every stage — TMA, then MMA, then
    /// the register pass, then the gradient MMA, each behind its own block
    /// sync — this overlaps all four. The parity argument is the forward's:
    /// every barrier's completions lead their waiter by at most one phase,
    /// because each producer's next completion transitively requires the
    /// previous consumer wait.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(192, 1)]
    pub unsafe fn flash_backward_q_pipelined(
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
            static mut KV_FULL: SharedArray<u64, BACKWARD_STAGES, 8> = SharedArray::UNINIT;
            static mut KV_FREE: SharedArray<u64, BACKWARD_STAGES, 8> = SharedArray::UNINIT;
            static mut S_FULL: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut S_FREE: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut DS_FULL: Barrier = Barrier::UNINIT;
            static mut DQ_DONE: Barrier = Barrier::UNINIT;
            static mut LSE2: SharedArray<f32, 128> = SharedArray::UNINIT;
            static mut DOTS: SharedArray<f32, 128> = SharedArray::UNINIT;

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let q = PairedPanel::from_raw(smem);
            let dy = PairedPanel::from_raw(smem.add(2 * TILE_BYTES));
            let k = PanelRingN::<BACKWARD_STAGES>::attach(smem.add(4 * TILE_BYTES));
            let v = PanelRingN::<BACKWARD_STAGES>::attach(
                smem.add((4 + BACKWARD_STAGES) * TILE_BYTES),
            );
            let ds = PairedPTile::from_raw(smem.add((4 + 2 * BACKWARD_STAGES) * TILE_BYTES));

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != FLASH_BACKWARD_Q_PIPELINED_BLOCK {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let group = 2 * TILE as u32;

            let pair = thread::blockIdx_x();
            let head = thread::blockIdx_y();
            let batch = thread::blockIdx_z();
            let t = sequence_length;
            let h = heads;
            let plane = (batch * h + head) as i32;
            let tile_a = pair * 2;
            let tile_b = tile_a + 1;
            let key_tiles = tile_b + 1;

            let kv_full =
                SemaphoreRing::<BACKWARD_STAGES>::attach(&raw mut KV_FULL as *mut Barrier);
            let kv_free =
                SemaphoreRing::<BACKWARD_STAGES>::attach(&raw mut KV_FREE as *mut Barrier);
            let s_full = SemaphoreRing::<2>::attach(&raw mut S_FULL as *mut Barrier);
            let s_free = SemaphoreRing::<2>::attach(&raw mut S_FREE as *mut Barrier);
            let ds_full = Semaphore::attach(&raw mut DS_FULL);
            let dq_done = Semaphore::attach(&raw mut DQ_DONE);

            // Barrier init precedes the TMEM allocation (the validated
            // ordering — see the ptxas pins in flash.rs).
            if tid == 0 {
                kv_full.init_all(1);
                kv_free.init_all(1);
                s_full.init_all(1);
                s_free.init_all(group);
                ds_full.init(group);
                dq_done.init(1);
                fence_proxy_async_shared_cta();
            }
            // The pair's 128 contiguous query rows' base-2 LSE and softmax dot,
            // staged while every warp is still running the prologue together —
            // after the split there is no block-wide sync left to hide behind.
            if tid < group {
                let query_row = (batch * t) as usize + tile_a as usize * TILE + tid as usize;
                let stat_index = query_row * h as usize + head as usize;
                (*(&raw mut LSE2 as *mut f32).add(tid as usize)) = logsumexp[stat_index] * LOG2E;
                (*(&raw mut DOTS as *mut f32).add(tid as usize)) = dot[stat_index];
            }
            thread::sync_threads();

            // Two 64-wide S buffers at columns 0..128, two dP buffers at
            // 128..256, the 128-column dQ accumulator at 256.
            let tmem = alloc_block(&raw mut TMEM_ADDRESS as *mut u32, 512);
            let s_tmem = STmem::from_raw(tmem);
            let dp_tmem = STmem::from_raw(tmem + 128);
            let dq_tmem = AccTmem::from_raw(tmem + 256);

            if tid < group {
                // Gradient warpgroup.
                let mut i = 0u32;
                while i < key_tiles {
                    s_full.wait(i);
                    backward_q_tile_paired(
                        s_tmem.columns_right((i & 1) * 64),
                        dp_tmem.columns_right((i & 1) * 64),
                        i,
                        tile_a,
                        tile_b,
                        warp_id,
                        lane,
                        &raw const LSE2 as *const f32,
                        &raw const DOTS as *const f32,
                        ds,
                    );
                    // dS is fenced into the async proxy before it is published;
                    // both S buffers are drained, so the score MMA may reuse
                    // this one.
                    fence_proxy_async_shared_cta();
                    s_free.sem(i).arrive();
                    ds_full.arrive();
                    dq_done.wait(i & 1);
                    if tid == 0 {
                        kv_free.sem(i).arrive();
                    }
                    i += 1;
                }
                let mut dq_acc = RegTile::<4, 32>::zero();
                merge_output_tile(dq_tmem, warp_id, &mut dq_acc);
                // The pair's rows are contiguous, so the block's base tile is
                // `tile_a` and the warp-derived local row (0..127) lands
                // correctly.
                store_grad_tile(batch, t, h, head, tile_a, warp_id, lane, &dq_acc, &mut dq);
            } else if tid == group {
                // TMA load warp leader. Q/dY ride the first stage's expected
                // bytes exactly like the forward's Q does.
                let mut i = 0u32;
                while i < key_tiles {
                    kv_free.wait_recycled(i);
                    let full = kv_full.sem(i);
                    let key_row = (i * TILE as u32) as i32;
                    k.tile(i).tma_load(k_tma, key_row, plane, full);
                    v.tile(i).tma_load(v_tma, key_row, plane, full);
                    if i == 0 {
                        let row_a = (tile_a * TILE as u32) as i32;
                        let row_b = (tile_b * TILE as u32) as i32;
                        q.tma_load_at(q_tma, 0, row_a, plane, full);
                        q.tma_load_at(q_tma, TILE, row_b, plane, full);
                        dy.tma_load_at(dy_tma, 0, row_a, plane, full);
                        dy.tma_load_at(dy_tma, TILE, row_b, plane, full);
                        full.expect_tx((2 * Panel::BYTES + 2 * PairedPanel::BYTES) as u32);
                    } else {
                        full.expect_tx(2 * Panel::BYTES as u32);
                    }
                    i += 1;
                }
            } else if tid == group + 32 {
                // MMA warp leader.
                let s_instruction = Tcgen05InstructionDescriptor::builder()
                    .shape(Tcgen05MmaShape::M128_N64)
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
                tcgen05_fence_after_thread_sync();
                let mut i = 0u32;
                while i < key_tiles {
                    kv_full.wait(i);
                    s_free.wait_recycled(i);
                    let buffer = (i & 1) * 64;
                    mma_abt(
                        s_tmem.columns_right(buffer).raw(),
                        q,
                        k.tile(i),
                        s_instruction,
                        false,
                    );
                    mma_abt(
                        dp_tmem.columns_right(buffer).raw(),
                        dy,
                        v.tile(i),
                        s_instruction,
                        false,
                    );
                    mma::commit(s_full.sem(i));
                    if i > 0 {
                        gradient_mma(i - 1, ds, k, dq_tmem, ds_full, dq_done, grad_instruction);
                    }
                    i += 1;
                }
                gradient_mma(
                    key_tiles - 1,
                    ds,
                    k,
                    dq_tmem,
                    ds_full,
                    dq_done,
                    grad_instruction,
                );
            }

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            dealloc_block(tmem, 512);
            if tid == 0 {
                kv_full.inval_all();
                kv_free.inval_all();
                s_full.inval_all();
                s_free.inval_all();
                ds_full.inval();
                dq_done.inval();
            }
        }
    }

    /// Issue one query tile's `dV += Pᵀ·dY` and `dK += dSᵀ·Q` from the MMA
    /// warp: wait for the gradient warpgroup to publish both operand tiles,
    /// chain each gradient MMA across the streamed panel's two HD subtiles,
    /// and commit into `grad_done`. As in kernel A that commit is what
    /// releases the operand tiles AND the ring stage — Q and dY are each read
    /// twice, once by a score MMA and once here.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn gradient_mma_kv<const STAGES: usize>(
        step: u32,
        p: PairedPTile,
        ds: PairedPTile,
        q: PanelRingN<STAGES>,
        dy: PanelRingN<STAGES>,
        dv_tmem: AccTmem,
        dk_tmem: AccTmem,
        pds_full: Semaphore,
        grad_done: Semaphore,
        instruction: u32,
    ) {
        unsafe {
            pds_full.wait(step & 1);
            let accumulate = step != 0;
            mma_ab(dv_tmem.raw(), p, dy.tile(step), instruction, accumulate);
            mma_ab(dk_tmem.raw(), ds, q.tile(step), instruction, accumulate);
            mma::commit(grad_done);
        }
    }

    /// Warp-specialized pipelined key-parallel backward — kernel B's half of
    /// the phase-5 kittens-first pair. Launch with
    /// `host::flash_backward_kv_pipelined_config`. Identical operand,
    /// statistic, and output contract to `flash_backward_kv_tcgen05`, with the
    /// same three roles as `flash_backward_q_pipelined`.
    ///
    /// Two differences from kernel A, both forced by the transposed math:
    /// - **the loop is relative.** Kernel B streams query tiles `key_a..T/64`,
    ///   so every ring index and phase parity runs off `step = query - key_a`
    ///   while the data addressing keeps the absolute tile. A `SemaphoreRing`
    ///   derives its parity from the visit count, which is the step, not the
    ///   tile.
    /// - **the per-tile statistics ride the ring.** Rows are key rows and
    ///   columns are query rows here, so `lse2`/`dot` index by the *streamed*
    ///   tile's 64 rows and must be re-staged every step — which the
    ///   synchronous kernel did behind a block sync that no longer exists.
    ///   The TMA warp stages them into a `BACKWARD_STAGES`-deep window and
    ///   publishes them on the stage's own `qdy_full`: a `warp::sync_mask`
    ///   orders the 32 lanes' stores ahead of the leader's `expect_tx`, and
    ///   the mbarrier's release makes them visible to the warpgroup exactly
    ///   like the forward's `restart` flag.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(192, 1)]
    pub unsafe fn flash_backward_kv_pipelined(
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
            static mut QDY_FULL: SharedArray<u64, BACKWARD_STAGES, 8> = SharedArray::UNINIT;
            static mut QDY_FREE: SharedArray<u64, BACKWARD_STAGES, 8> = SharedArray::UNINIT;
            static mut S_FULL: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut S_FREE: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut PDS_FULL: Barrier = Barrier::UNINIT;
            static mut GRAD_DONE: Barrier = Barrier::UNINIT;
            static mut LSE2: SharedArray<f32, { TILE * BACKWARD_STAGES }> = SharedArray::UNINIT;
            static mut DOTS: SharedArray<f32, { TILE * BACKWARD_STAGES }> = SharedArray::UNINIT;

            let smem = DynamicSharedArray::<u8, 128>::get_raw();
            let k = PairedPanel::from_raw(smem);
            let v = PairedPanel::from_raw(smem.add(2 * TILE_BYTES));
            let q = PanelRingN::<BACKWARD_STAGES>::attach(smem.add(4 * TILE_BYTES));
            let dy = PanelRingN::<BACKWARD_STAGES>::attach(
                smem.add((4 + BACKWARD_STAGES) * TILE_BYTES),
            );
            let p = PairedPTile::from_raw(smem.add((4 + 2 * BACKWARD_STAGES) * TILE_BYTES));
            let ds = PairedPTile::from_raw(smem.add((5 + 2 * BACKWARD_STAGES) * TILE_BYTES));

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != FLASH_BACKWARD_KV_PIPELINED_BLOCK {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let group = 2 * TILE as u32;

            let pair = thread::blockIdx_x();
            let head = thread::blockIdx_y();
            let batch = thread::blockIdx_z();
            let t = sequence_length;
            let h = heads;
            let plane = (batch * h + head) as i32;
            let tiles = t / TILE as u32;
            let key_a = pair * 2;
            let key_b = key_a + 1;
            let steps = tiles - key_a;

            let qdy_full =
                SemaphoreRing::<BACKWARD_STAGES>::attach(&raw mut QDY_FULL as *mut Barrier);
            let qdy_free =
                SemaphoreRing::<BACKWARD_STAGES>::attach(&raw mut QDY_FREE as *mut Barrier);
            let s_full = SemaphoreRing::<2>::attach(&raw mut S_FULL as *mut Barrier);
            let s_free = SemaphoreRing::<2>::attach(&raw mut S_FREE as *mut Barrier);
            let pds_full = Semaphore::attach(&raw mut PDS_FULL);
            let grad_done = Semaphore::attach(&raw mut GRAD_DONE);
            let lse_base = &raw mut LSE2 as *mut f32;
            let dot_base = &raw mut DOTS as *mut f32;

            if tid == 0 {
                qdy_full.init_all(1);
                qdy_free.init_all(1);
                s_full.init_all(1);
                s_free.init_all(group);
                pds_full.init(group);
                grad_done.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            // Sᵀ/dPᵀ take two 64-column buffers each (columns 0..256); the two
            // 128-column gradient accumulators fill the rest of the 512.
            let tmem = alloc_block(&raw mut TMEM_ADDRESS as *mut u32, 512);
            let st_tmem = STmem::from_raw(tmem);
            let dpt_tmem = STmem::from_raw(tmem + 128);
            let dv_tmem = AccTmem::from_raw(tmem + 256);
            let dk_tmem = AccTmem::from_raw(tmem + 384);

            if tid < group {
                // Gradient warpgroup.
                let mut step = 0u32;
                while step < steps {
                    let stage = (step as usize % BACKWARD_STAGES) * TILE;
                    s_full.wait(step);
                    backward_kv_tile_paired(
                        st_tmem.columns_right((step & 1) * 64),
                        dpt_tmem.columns_right((step & 1) * 64),
                        key_a + step,
                        key_a,
                        key_b,
                        warp_id,
                        lane,
                        lse_base.add(stage) as *const f32,
                        dot_base.add(stage) as *const f32,
                        p,
                        ds,
                    );
                    fence_proxy_async_shared_cta();
                    s_free.sem(step).arrive();
                    pds_full.arrive();
                    grad_done.wait(step & 1);
                    if tid == 0 {
                        qdy_free.sem(step).arrive();
                    }
                    step += 1;
                }
                let mut dv_acc = RegTile::<4, 32>::zero();
                let mut dk_acc = RegTile::<4, 32>::zero();
                merge_output_tile(dv_tmem, warp_id, &mut dv_acc);
                merge_output_tile(dk_tmem, warp_id, &mut dk_acc);
                store_grad_tile(batch, t, h, head, key_a, warp_id, lane, &dv_acc, &mut dv);
                store_grad_tile(batch, t, h, head, key_a, warp_id, lane, &dk_acc, &mut dk);
            } else if warp_id == group / 32 {
                // TMA load warp: every lane stages two of the streamed tile's
                // 64 per-query-row statistics, then the leader issues the
                // boxes and charges the stage's expected bytes.
                let mut step = 0u32;
                while step < steps {
                    let query_tile = key_a + step;
                    qdy_free.wait_recycled(step);
                    let stage = (step as usize % BACKWARD_STAGES) * TILE;
                    let mut row = lane;
                    while row < TILE as u32 {
                        let global_row =
                            (batch * t) as usize + query_tile as usize * TILE + row as usize;
                        let stat_index = global_row * h as usize + head as usize;
                        *lse_base.add(stage + row as usize) = logsumexp[stat_index] * LOG2E;
                        *dot_base.add(stage + row as usize) = dot[stat_index];
                        row += 32;
                    }
                    warp::sync_mask(u32::MAX);
                    if lane == 0 {
                        let full = qdy_full.sem(step);
                        let query_row = (query_tile * TILE as u32) as i32;
                        q.tile(step).tma_load(q_tma, query_row, plane, full);
                        dy.tile(step).tma_load(dy_tma, query_row, plane, full);
                        if step == 0 {
                            let row_a = (key_a * TILE as u32) as i32;
                            let row_b = (key_b * TILE as u32) as i32;
                            k.tma_load_at(k_tma, 0, row_a, plane, full);
                            k.tma_load_at(k_tma, TILE, row_b, plane, full);
                            v.tma_load_at(v_tma, 0, row_a, plane, full);
                            v.tma_load_at(v_tma, TILE, row_b, plane, full);
                            full.expect_tx((2 * Panel::BYTES + 2 * PairedPanel::BYTES) as u32);
                        } else {
                            full.expect_tx(2 * Panel::BYTES as u32);
                        }
                    }
                    step += 1;
                }
            } else if tid == group + 32 {
                // MMA warp leader.
                let s_instruction = Tcgen05InstructionDescriptor::builder()
                    .shape(Tcgen05MmaShape::M128_N64)
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
                tcgen05_fence_after_thread_sync();
                let mut step = 0u32;
                while step < steps {
                    qdy_full.wait(step);
                    s_free.wait_recycled(step);
                    let buffer = (step & 1) * 64;
                    mma_abt(
                        st_tmem.columns_right(buffer).raw(),
                        k,
                        q.tile(step),
                        s_instruction,
                        false,
                    );
                    mma_abt(
                        dpt_tmem.columns_right(buffer).raw(),
                        v,
                        dy.tile(step),
                        s_instruction,
                        false,
                    );
                    mma::commit(s_full.sem(step));
                    if step > 0 {
                        gradient_mma_kv(
                            step - 1,
                            p,
                            ds,
                            q,
                            dy,
                            dv_tmem,
                            dk_tmem,
                            pds_full,
                            grad_done,
                            grad_instruction,
                        );
                    }
                    step += 1;
                }
                gradient_mma_kv(
                    steps - 1,
                    p,
                    ds,
                    q,
                    dy,
                    dv_tmem,
                    dk_tmem,
                    pds_full,
                    grad_done,
                    grad_instruction,
                );
            }

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            dealloc_block(tmem, 512);
            if tid == 0 {
                qdy_full.inval_all();
                qdy_free.inval_all();
                s_full.inval_all();
                s_free.inval_all();
                pds_full.inval();
                grad_done.inval();
            }
        }
    }

    /// Issue one tile's `O = P·V` from the MMA warp: wait for the softmax
    /// warpgroup to publish P, chain the eight K=16 MMAs — continuing the
    /// TMEM segment, or starting a fresh one when the warpgroup's restart
    /// flag for this tile says its vote drained it — and commit completion
    /// into `o_full`. The flag read is ordered by the `p_full` wait (the
    /// warpgroup writes it before arriving).
    #[inline(always)]
    unsafe fn output_mma<const STAGES: usize, const S_BUFFERS: usize>(
        tile: u32,
        stream: Stream<S_BUFFERS>,
        v: PanelRingN<STAGES>,
        o_instruction: u32,
    ) {
        unsafe {
            stream.p_full.wait(tile & 1);
            let fresh = *stream.restart.add((tile & 1) as usize) != 0;
            mma_ab(
                stream.o_tmem.raw(),
                stream.p,
                v.tile(tile),
                o_instruction,
                !fresh,
            );
            mma::commit(stream.o_full);
        }
    }

    /// One softmax-warpgroup tile step, shared by the pipelined and
    /// persistent forwards: wait for S, run `softmax_tile` (with the
    /// correction vote), publish the segment-restart flag and P, then wait
    /// out this tile's O MMA and recycle the K/V stage. Callers pass
    /// `diagonal` as a literal so the mask logic folds out of the full-tile
    /// loop; `S_BUFFERS` picks the S ring depth (2 for the pipelined
    /// kernel's double-buffered S, 1 for the persistent kernel's
    /// single-buffered per-stream S — the rings own both kernels' phase
    /// arithmetic). `tid_in_group`/`warp_id` are warpgroup-local.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn warpgroup_tile<const STAGES: usize, const S_BUFFERS: usize>(
        i: u32,
        diagonal: bool,
        tid_in_group: u32,
        warp_id: u32,
        lane: u32,
        stream: Stream<S_BUFFERS>,
        kv_free: SemaphoreRing<STAGES>,
        m_ref: &mut RegVec<4>,
        running_sum: &mut RegVec<4>,
        out_acc: &mut RegTile<4, 32>,
        corrections: &mut u32,
    ) {
        unsafe {
            stream.s_full.wait(i);
            let correction = softmax_tile(
                // Each S buffer is 64 columns wide.
                stream.s_tmem.columns_right((i % S_BUFFERS as u32) * 64),
                stream.o_tmem,
                i,
                diagonal,
                warp_id,
                lane,
                stream.p,
                stream.votes,
                stream.vote,
                m_ref,
                running_sum,
                out_acc,
            );
            if i > 0 && correction {
                *corrections += 1;
            }
            if tid_in_group == 0 {
                *stream.restart.add((i & 1) as usize) = correction as u32;
            }
            // Both S passes are drained and P is fenced into the async
            // proxy: release the S buffer and publish P (which also
            // releases the restart flag to the MMA warp).
            fence_proxy_async_shared_cta();
            stream.s_free.sem(i).arrive();
            stream.p_full.arrive();
            stream.o_full.wait(i & 1);
            if tid_in_group == 0 {
                kv_free.sem(i).arrive();
            }
        }
    }

    /// Warp-specialized pipelined causal forward (issue #35, phase 2).
    /// Launch with `host::flash_pipelined_config`: grid `(T/64, H, B)`,
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
    #[launch_bounds(128, 1)]
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
            let q = Panel::from_raw(smem);
            let k = PanelRing::attach(smem.add(TILE_BYTES));
            let v = PanelRing::attach(smem.add((1 + PIPELINE_STAGES) * TILE_BYTES));

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != FLASH_PIPELINE_BLOCK {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();

            let kv_full =
                SemaphoreRing::<PIPELINE_STAGES>::attach(&raw mut KV_FULL as *mut Barrier);
            let kv_free =
                SemaphoreRing::<PIPELINE_STAGES>::attach(&raw mut KV_FREE as *mut Barrier);

            let query_tile = thread::blockIdx_x();
            let head = thread::blockIdx_y();
            let batch = thread::blockIdx_z();
            let t = sequence_length;
            let h = heads;
            let plane = (batch * h + head) as i32;
            let key_tiles = query_tile + 1;

            // Barrier init precedes the TMEM allocation (the validated
            // ordering); the stream handle needs `tmem`, so its barriers are
            // initialized through ring/semaphore views of the same words.
            if tid == 0 {
                kv_full.init_all(1);
                kv_free.init_all(1);
                SemaphoreRing::<2>::attach(&raw mut S_FULL as *mut Barrier).init_all(1);
                SemaphoreRing::<2>::attach(&raw mut S_FREE as *mut Barrier).init_all(TILE as u32);
                Semaphore::attach(&raw mut P_FULL).init(TILE as u32);
                Semaphore::attach(&raw mut O_FULL).init(1);
                Semaphore::attach(&raw mut VOTE_BARRIER).init(TILE as u32);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            // Two 64-wide S buffers occupy TMEM columns 0..128; O follows at
            // 128. The P subtile's `PTile` folds the tile base's absolute
            // 128-byte row phase into its swizzled `stmatrix` addresses.
            let tmem = alloc_block(&raw mut TMEM_ADDRESS as *mut u32, 512);
            let stream = Stream::<2>::attach(
                0,
                tmem,
                smem.add((1 + 2 * PIPELINE_STAGES) * TILE_BYTES),
                &raw mut VOTES as *mut u32,
                &raw mut RESTART as *mut u32,
                &raw mut VOTE_BARRIER,
                &raw mut S_FULL as *mut Barrier,
                &raw mut S_FREE as *mut Barrier,
                &raw mut P_FULL,
                &raw mut O_FULL,
            );

            if tid < TILE as u32 {
                // Softmax / correction / epilogue warpgroup. The key loop
                // is split so the diagonal tile is the only one paying for
                // mask logic (the full-tile calls fold `diagonal = false`).
                let mut m_ref = RegVec::<4>::splat(MASKED_SCORE);
                let mut running_sum = RegVec::<4>::splat(0.0);
                let mut out_acc = RegTile::<4, 32>::zero();
                let mut corrections = 0u32;
                let mut i = 0u32;
                while i + 1 < key_tiles {
                    warpgroup_tile::<PIPELINE_STAGES, 2>(
                        i,
                        false,
                        tid,
                        warp_id,
                        lane,
                        stream,
                        kv_free,
                        &mut m_ref,
                        &mut running_sum,
                        &mut out_acc,
                        &mut corrections,
                    );
                    i += 1;
                }
                warpgroup_tile::<PIPELINE_STAGES, 2>(
                    i,
                    true,
                    tid,
                    warp_id,
                    lane,
                    stream,
                    kv_free,
                    &mut m_ref,
                    &mut running_sum,
                    &mut out_acc,
                    &mut corrections,
                );
                merge_output_tile(stream.o_tmem, warp_id, &mut out_acc);
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
                // TMA load warp leader: Q once, then the K/V ring, on kittens
                // tiles — the ring types own the stage selection and free-slot
                // parity, the panel type the two-subtile box issue.
                let mut i = 0u32;
                while i < key_tiles {
                    kv_free.wait_recycled(i);
                    let full = kv_full.sem(i);
                    let key_row = (i * TILE as u32) as i32;
                    k.tile(i).tma_load(k_tma, key_row, plane, full);
                    v.tile(i).tma_load(v_tma, key_row, plane, full);
                    if i == 0 {
                        q.tma_load(q_tma, (query_tile * TILE as u32) as i32, plane, full);
                        full.expect_tx(3 * Panel::BYTES as u32);
                    } else {
                        full.expect_tx(2 * Panel::BYTES as u32);
                    }
                    i += 1;
                }
            } else if tid == (TILE + 32) as u32 {
                // MMA warp leader.
                let s_instruction = Tcgen05InstructionDescriptor::builder()
                    .shape(Tcgen05MmaShape::M128_N64)
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
                    kv_full.wait(i);
                    stream.s_free.wait_recycled(i);
                    score_mma(
                        stream.s_tmem.columns_right((i & 1) * 64).raw(),
                        q,
                        k.tile(i),
                        s_instruction,
                    );
                    mma::commit(stream.s_full.sem(i));
                    if i > 0 {
                        output_mma(i - 1, stream, v, o_instruction);
                    }
                    i += 1;
                }
                output_mma(key_tiles - 1, stream, v, o_instruction);
            }

            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            dealloc_block(tmem, 512);
            if tid == 0 {
                kv_full.inval_all();
                kv_free.inval_all();
                stream.inval();
            }
        }
    }

    /// One softmax workstream's resources, shared by the pipelined kernel
    /// (a single stream with a double-buffered S, `S_BUFFERS = 2`) and the
    /// persistent kernel (two ping-pong streams, each with a single-buffered
    /// S): the stream's S/O TMEM segments, P tile, vote/restart words, and
    /// the barrier set the softmax warpgroup handshakes with the MMA warp.
    /// The S rings own both kernels' phase arithmetic — double-buffered
    /// `(i & 1, (i / 2) & 1)` and single-buffered `(0, i & 1)` are the same
    /// `SemaphoreRing` walk at different depths.
    #[derive(Clone, Copy)]
    struct Stream<const S_BUFFERS: usize> {
        s_tmem: STmem,
        o_tmem: AccTmem,
        p: PTile,
        votes: *mut u32,
        restart: *mut u32,
        vote: Semaphore,
        s_full: SemaphoreRing<S_BUFFERS>,
        s_free: SemaphoreRing<S_BUFFERS>,
        p_full: Semaphore,
        o_full: Semaphore,
    }

    impl<const S_BUFFERS: usize> Stream<S_BUFFERS> {
        /// Stream `index` (A = 0, B = 1) of the kernel's resource plan: S at
        /// TMEM column `64·index`, O at `128 + 128·index`, the `index`-th P
        /// subtile, and slot `index` of every per-stream barrier pair and
        /// vote/restart array. The pipelined kernel is the one-stream case.
        #[allow(clippy::too_many_arguments)]
        #[inline(always)]
        unsafe fn attach(
            index: usize,
            tmem: u32,
            p: *mut u8,
            votes: *mut u32,
            restart: *mut u32,
            vote: *mut Barrier,
            s_full: *mut Barrier,
            s_free: *mut Barrier,
            p_full: *mut Barrier,
            o_full: *mut Barrier,
        ) -> Self {
            unsafe {
                Self {
                    s_tmem: STmem::from_raw(tmem + 64 * index as u32),
                    o_tmem: AccTmem::from_raw(tmem + 128 + 128 * index as u32),
                    p: PTile::from_raw(p.add(index * SUBTILE_BYTES)),
                    votes: votes.add(8 * index),
                    restart: restart.add(2 * index),
                    vote: Semaphore::attach(vote.add(index)),
                    s_full: SemaphoreRing::attach(s_full.add(index)),
                    s_free: SemaphoreRing::attach(s_free.add(index)),
                    p_full: Semaphore::attach(p_full.add(index)),
                    o_full: Semaphore::attach(o_full.add(index)),
                }
            }
        }

        /// (Re)initialize the stream's barriers for one work item.
        #[inline(always)]
        unsafe fn init(self) {
            unsafe {
                self.s_full.init_all(1);
                self.s_free.init_all(TILE as u32);
                self.p_full.init(TILE as u32);
                self.o_full.init(1);
                self.vote.init(TILE as u32);
            }
        }

        /// Invalidate the stream's barriers between work items.
        #[inline(always)]
        unsafe fn inval(self) {
            unsafe {
                self.s_full.inval_all();
                self.s_free.inval_all();
                self.p_full.inval();
                self.o_full.inval();
                self.vote.inval();
            }
        }
    }

    /// One full workstream of the persistent forward, run by one softmax
    /// warpgroup: the split full-tile/diagonal key loop, the final segment
    /// drain, and the outputs. All indices (`tid_in_group`, `warp_id`) are
    /// warpgroup-local; the barriers are this stream's own set except
    /// `kv_free`, which both streams share (each arrives once per tile it
    /// consumed).
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
        stream: Stream<1>,
        kv_free: SemaphoreRing<PERSISTENT_STAGES>,
        output: &mut DisjointSlice<f32>,
        logsumexp: &mut DisjointSlice<f32>,
        correction_counts: &mut DisjointSlice<u32>,
    ) {
        unsafe {
            let key_tiles = query_tile + 1;
            let mut m_ref = RegVec::<4>::splat(MASKED_SCORE);
            let mut running_sum = RegVec::<4>::splat(0.0);
            let mut out_acc = RegTile::<4, 32>::zero();
            let mut corrections = 0u32;
            let mut i = 0u32;
            while i + 1 < key_tiles {
                warpgroup_tile::<PERSISTENT_STAGES, 1>(
                    i,
                    false,
                    tid_in_group,
                    warp_id,
                    lane,
                    stream,
                    kv_free,
                    &mut m_ref,
                    &mut running_sum,
                    &mut out_acc,
                    &mut corrections,
                );
                i += 1;
            }
            warpgroup_tile::<PERSISTENT_STAGES, 1>(
                i,
                true,
                tid_in_group,
                warp_id,
                lane,
                stream,
                kv_free,
                &mut m_ref,
                &mut running_sum,
                &mut out_acc,
                &mut corrections,
            );
            merge_output_tile(stream.o_tmem, warp_id, &mut out_acc);
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
                *correction_counts
                    .get_unchecked_mut((batch * h + head) as usize * tiles + query_tile as usize) =
                    corrections;
            }
        }
    }

    /// The persistent forward as a [`pipeline::Job`]: tile and semaphore
    /// handles built once at launch, work items decoded per iteration, and
    /// the whole barrier set re-initialized per item by the harness —
    /// `kv_free`'s arrival count is re-chosen each time (one consumer
    /// stream, or two when the pair's odd tail leaves stream B active).
    struct PersistentForward<'k, 'o, 'l, 'c> {
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        t: u32,
        h: u32,
        tiles: u32,
        pairs: u32,
        plane_count: u32,
        tid: u32,
        warp_id: u32,
        lane: u32,
        q_a: Panel,
        q_b: Panel,
        k: PersistentPanelRing,
        v: PersistentPanelRing,
        kv_full: SemaphoreRing<PERSISTENT_STAGES>,
        kv_free: SemaphoreRing<PERSISTENT_STAGES>,
        a: Stream<1>,
        b: Stream<1>,
        output: &'k mut DisjointSlice<'o, f32>,
        logsumexp: &'k mut DisjointSlice<'l, f32>,
        correction_counts: &'k mut DisjointSlice<'c, u32>,
    }

    impl PersistentForward<'_, '_, '_, '_> {
        /// Decode a work item into its (pair, plane) coordinates. Items are
        /// ordered by *descending* pair index, so the causally long pairs
        /// are dealt out before the cheap ones.
        #[inline(always)]
        fn item_pair(&self, item: u32) -> (u32, u32) {
            (
                self.pairs - 1 - item / self.plane_count,
                item % self.plane_count,
            )
        }
    }

    impl pipeline::Job for PersistentForward<'_, '_, '_, '_> {
        #[inline(always)]
        unsafe fn init(&self, item: u32) {
            unsafe {
                let (pair, _) = self.item_pair(item);
                let b_active = pair * 2 + 1 < self.tiles;
                self.kv_full.init_all(1);
                // One arrival per active consumer stream per tile consumed.
                self.kv_free.init_all(1 + b_active as u32);
                self.a.init();
                self.b.init();
            }
        }

        #[inline(always)]
        unsafe fn inval(&self) {
            unsafe {
                self.kv_full.inval_all();
                self.kv_free.inval_all();
                self.a.inval();
                self.b.inval();
            }
        }

        #[inline(always)]
        unsafe fn work(&mut self, item: u32) {
            unsafe {
                let (pair, plane) = self.item_pair(item);
                let batch = plane / self.h;
                let head = plane % self.h;
                let tile_a = pair * 2;
                let tile_b = tile_a + 1;
                let b_active = tile_b < self.tiles;
                let tiles_a = tile_a + 1;
                let tiles_b = tile_b + 1;
                let stream_tiles = if b_active { tiles_b } else { tiles_a };

                if self.tid < TILE as u32 {
                    persistent_stream(
                        tile_a,
                        batch,
                        self.t,
                        self.h,
                        head,
                        self.tid,
                        self.warp_id,
                        self.lane,
                        self.a,
                        self.kv_free,
                        self.output,
                        self.logsumexp,
                        self.correction_counts,
                    );
                } else if self.tid >= 2 * TILE as u32 {
                    // Stream B runs on warps 4-5 (warpgroup-1 positions 0-1),
                    // NOT warps 2-3: a `tcgen05.ld` warp can only reach TMEM
                    // lanes `(warp % 4) * 32 .. +32`, and a 64-row tile's M128
                    // accumulator keeps its real rows in lanes 0..63. Warps
                    // 2-3 (positions 2-3) would read lanes 64..127 — the
                    // undrained phantom rows — so stream B must sit at a
                    // warpgroup boundary (positions 0-1) to reach lanes 0..63.
                    if b_active {
                        persistent_stream(
                            tile_b,
                            batch,
                            self.t,
                            self.h,
                            head,
                            self.tid - 2 * TILE as u32,
                            self.warp_id - (2 * TILE / 32) as u32,
                            self.lane,
                            self.b,
                            self.kv_free,
                            self.output,
                            self.logsumexp,
                            self.correction_counts,
                        );
                    }
                } else if self.tid == TILE as u32 {
                    // TMA load warp leader (warp 2): both Q tiles once, then
                    // the shared K/V ring over the longer stream.
                    let plane = plane as i32;
                    let mut i = 0u32;
                    while i < stream_tiles {
                        self.kv_free.wait_recycled(i);
                        let full = self.kv_full.sem(i);
                        let key_row = (i * TILE as u32) as i32;
                        self.k.tile(i).tma_load(self.k_tma, key_row, plane, full);
                        self.v.tile(i).tma_load(self.v_tma, key_row, plane, full);
                        if i == 0 {
                            self.q_a.tma_load(
                                self.q_tma,
                                (tile_a * TILE as u32) as i32,
                                plane,
                                full,
                            );
                            if b_active {
                                self.q_b.tma_load(
                                    self.q_tma,
                                    (tile_b * TILE as u32) as i32,
                                    plane,
                                    full,
                                );
                            }
                            let q_tiles = 1 + b_active as u32;
                            full.expect_tx((2 + q_tiles) * TILE_BYTES as u32);
                        } else {
                            full.expect_tx(2 * TILE_BYTES as u32);
                        }
                        i += 1;
                    }
                } else if self.tid == (TILE + 32) as u32 {
                    // MMA warp leader (warp 3): per shared tile, S-MMAs for
                    // both streams (each gated by its own single-buffered S
                    // being free), then the previous tile's O-MMAs — the
                    // same stagger as the pipelined kernel, per stream.
                    let s_instruction = Tcgen05InstructionDescriptor::builder()
                        .shape(Tcgen05MmaShape::M128_N64)
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
                        self.kv_full.wait(i);
                        let k_tile = self.k.tile(i);
                        if i < tiles_a {
                            self.a.s_free.wait_recycled(i);
                            score_mma(self.a.s_tmem.raw(), self.q_a, k_tile, s_instruction);
                            mma::commit(self.a.s_full.sem(i));
                        }
                        if b_active {
                            self.b.s_free.wait_recycled(i);
                            score_mma(self.b.s_tmem.raw(), self.q_b, k_tile, s_instruction);
                            mma::commit(self.b.s_full.sem(i));
                        }
                        if i > 0 {
                            output_mma(i - 1, self.a, self.v, o_instruction);
                            if b_active {
                                output_mma(i - 1, self.b, self.v, o_instruction);
                            }
                        }
                        i += 1;
                    }
                    if b_active {
                        output_mma(tiles_b - 1, self.b, self.v, o_instruction);
                    } else {
                        output_mma(tiles_a - 1, self.a, self.v, o_instruction);
                    }
                }
            }
        }
    }

    /// Persistent two-Q-tile ping-pong forward (issue #35, phase 3).
    /// Launch with `host::flash_persistent_config`: a 1-D grid of at most
    /// `ceil(tiles/2) * H * B` CTAs, `FLASH_PERSISTENT_BLOCK` threads,
    /// `host::FLASH_PERSISTENT_SMEM_BYTES` dynamic shared bytes. Operand
    /// and output contracts match the other tcgen05 forwards.
    ///
    /// Each work item is a (query-tile *pair*, head, batch): stream A
    /// (warps 0–1) owns query tile `2p`, stream B (warps 4–5) owns `2p+1`,
    /// and both share one K/V ring, one TMA-load warp (warp 2) and one MMA
    /// warp (warp 3). A stream is only two warps because a 64-row tile fills
    /// TMEM lanes 0..63; stream B sits at warpgroup-1 positions 0–1 (warps
    /// 4–5) — NOT the adjacent warps 2–3 — because a `tcgen05.ld` warp
    /// reaches only lanes `(warp % 4) * 32 .. +32`, so only positions 0–1 of
    /// a warpgroup can drain the accumulator's real rows 0..63. TMEM holds a
    /// single-buffered S plus an O segment per stream (384 of the 512
    /// columns): while one stream runs softmax, the MMA warp feeds the
    /// other — the ping-pong a single warpgroup could not reach, and the
    /// main use of the SM's issue slots now that TMEM pins occupancy to one
    /// CTA.
    ///
    /// The strided work-item loop and its per-item barrier lifecycle are
    /// `kittens::pipeline::run` (extracted from this kernel); the
    /// [`PersistentForward`] job orders items by *descending* pair index, so
    /// the causally long pairs are dealt out before the cheap ones.
    /// Launching with grid = the full item count degenerates to one item
    /// per CTA — the non-persistent config kept for hang debugging. The
    /// per-item re-initialization means each item's phase arithmetic starts
    /// from zero and unbalanced arrivals (stream A never arrives for the
    /// shared stream's extra diagonal tile; an inactive stream B never
    /// arrives at all) are wiped, not threaded through parity math;
    /// `kv_free`'s arrival count is likewise chosen per item — two consumer
    /// streams normally, one when the last odd pair leaves stream B
    /// inactive.
    #[kernel]
    #[launch_bounds(192, 1)]
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
            let q_a = Panel::from_raw(smem);
            let q_b = Panel::from_raw(smem.add(TILE_BYTES));
            let k = PersistentPanelRing::attach(smem.add(2 * TILE_BYTES));
            let v = PersistentPanelRing::attach(smem.add((2 + PERSISTENT_STAGES) * TILE_BYTES));
            let p_a = smem.add((2 + 2 * PERSISTENT_STAGES) * TILE_BYTES);

            let tid = thread::threadIdx_x();
            if thread::blockDim_x() as usize != FLASH_PERSISTENT_BLOCK {
                return;
            }
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();

            let s_full = &raw mut S_FULL as *mut Barrier;
            let s_free = &raw mut S_FREE as *mut Barrier;
            let p_full = &raw mut P_FULL as *mut Barrier;
            let o_full = &raw mut O_FULL as *mut Barrier;
            let vote = &raw mut VOTE as *mut Barrier;
            let votes = &raw mut VOTES as *mut u32;
            let restart = &raw mut RESTART as *mut u32;

            let t = sequence_length;
            let h = heads;
            let tiles = t / TILE as u32;
            let pairs = tiles.div_ceil(2);
            let plane_count = h * batches;
            let work_items = pairs * plane_count;

            let tmem = alloc_block(&raw mut TMEM_ADDRESS as *mut u32, 512);
            // TMEM columns: S_A 0..64, S_B 64..128, O_A 128..256,
            // O_B 256..384.
            let a = Stream::<1>::attach(
                0, tmem, p_a, votes, restart, vote, s_full, s_free, p_full, o_full,
            );
            let b = Stream::<1>::attach(
                1, tmem, p_a, votes, restart, vote, s_full, s_free, p_full, o_full,
            );

            let mut job = PersistentForward {
                q_tma,
                k_tma,
                v_tma,
                t,
                h,
                tiles,
                pairs,
                plane_count,
                tid,
                warp_id,
                lane,
                q_a,
                q_b,
                k,
                v,
                kv_full: SemaphoreRing::<PERSISTENT_STAGES>::attach(
                    &raw mut KV_FULL as *mut Barrier,
                ),
                kv_free: SemaphoreRing::<PERSISTENT_STAGES>::attach(
                    &raw mut KV_FREE as *mut Barrier,
                ),
                a,
                b,
                output: &mut output,
                logsumexp: &mut logsumexp,
                correction_counts: &mut correction_counts,
            };
            pipeline::run(&mut job, work_items);
            dealloc_block(tmem, 512);
        }
    }
}
