//! FA4-shaped tcgen05 attention forward (issue #68) and backward (issue #35
//! phase 4).
//!
//! At cuda-oxide b099f64 this module shares a pure-PTX artifact with the
//! libdevice-backed oracle kernels. The softmax's software `exp2` and LSE
//! epilogue's software `log2` remain deliberate FA4 SFU-offload optimizations,
//! not artifact-path workarounds.
//!
//! Kernel shape contract (the host launchers in `host.rs` enforce it):
//! - operands are packed-bf16 staging buffers `[B*H, T, HD]`, one contiguous
//!   `[T, HD]` panel per head, produced by tensor-gpu's
//!   `stage_attention_heads_bf16` (Q arrives pre-scaled by
//!   `softmax_scale * log2(e)`, so scores are base-2 native);
//! - `T` is a multiple of the 128-row query block; `HD == 128`; non-aligned
//!   shapes stay on the fp32 tiled kernels in `lib.rs`;
//! - outputs keep the existing contract: fp32 `y[B*T, H*HD]` and fp32
//!   `logsumexp[B*T, H]` in natural-log units.
//!
//! `flash_forward` is one kernel where there were three (#68). A CTA owns a
//! `QUERIES = 128` query block of one `(batch, head)` and streams the causal
//! `TILE`-key tiles beneath it: TMA loads Q/K/V into swizzled shared tiles
//! (each 128-wide head panel is two stacked 64-wide SWIZZLE_128B subtiles),
//! `S = Q·Kᵀ` accumulates in a double-buffered fp32 TMEM segment, a register
//! softmax (mask → row max → software exp2 → running sum) packs bf16
//! probabilities back to shared memory with swizzled `stmatrix` stores, and
//! `O += P·V` accumulates in a TMEM *segment* under a fixed per-row max
//! reference (`enable_d` across tiles). FA4's conditional correction, adapted
//! for the missing `tcgen05.st`: only when some row's tile max climbs more than
//! `CORRECTION_THRESHOLD` above the reference does the warpgroup drain the
//! segment into per-thread registers, rescale, and restart it — otherwise the
//! segment just keeps accumulating and the warpgroup never touches O TMEM.
//!
//! The three generations it replaces were all shaped by a 64-query tile against
//! an `M128` accumulator: every MMA filled 64 real rows and 64 phantom ones,
//! and the phase-2 and phase-3 kernels bought that half back with structure — a
//! dedicated TMA warp and MMA warp, then two softmax warpgroups ping-ponging
//! adjacent query tiles. At 128 query rows the MMA is whole, the softmax
//! warpgroup is naturally the accumulator's four warps, and one
//! double-buffered score segment is the whole of the overlap. What is left is
//! the library's: `SharedPlan` for the plan, `pipeline::run` for the persistent
//! work-item loop, `make_causal_at` for the coordinate-origin mask,
//! `online_rescale` for the flash rescale, and `block_reduce` for the
//! correction vote all three kernels used to open-code.
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
use cuda_device::tcgen05::{tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, launch_bounds, thread, warp};
use kittens::global::{GlobalRows, store_row_vec, store_rows};
use kittens::ldst::{store_fragment, store_tile};
use kittens::mma::{self, MmaShape, mma_ab, mma_abt};
use kittens::pipeline;
use kittens::plan::SharedPlan;
use kittens::reg::{
    BaseLdtm, Exp2Approx, Fragment, Max, RegTile, RegVec, exp2_approx, log2_approx, online_rescale,
    warp_reduce,
};
use kittens::shared::{Bf16, F32, SharedTile, SharedTileRing, SharedVec, Swizzle128B};
use kittens::sync::{PhasedSemaphore, Semaphore, SemaphoreRing, block_reduce};
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
/// Query rows one forward CTA owns, and the `M` of every MMA in this module:
/// two stacked `TILE` blocks, so an `M128` accumulator's 128 rows are all real.
/// The 64-query forward that filled half of one is what #68 removed.
const QUERIES: usize = 2 * TILE;

/// Every MMA here is one warpgroup's `M128_N64`. The element type, the fp32
/// accumulator and the transpose flags are the operand tiles' and the entry
/// point's own now, so a call states only the shape.
const MMA_SHAPE: MmaShape = MmaShape::M128_N64;

/// One `[TILE, HD]` bf16 operand panel as a kittens tile — two stacked
/// SWIZZLE_128B subtiles, the layout described above. Phase 1 of issue #61
/// moves the pipelined kernel's loader warp onto these; later phases absorb
/// the rest of the raw swizzle/barrier machinery.
type Panel = SharedTile<Bf16, TILE, HD, Swizzle128B>;
/// An `N`-deep ring of panels — the K and V streams of every kernel here; each
/// picks its own stage depth.
type PanelRingN<const N: usize> = SharedTileRing<Bf16, TILE, HD, Swizzle128B, N>;
/// The forward's K/V ring.
type ForwardPanelRing = PanelRingN<FORWARD_STAGES>;
/// The single-subtile `[TILE, TILE]` bf16 tile the swizzle probe writes. Its
/// `swizzled_chunk` folds in the tile base's absolute 128-byte row phase — the
/// fact the per-kernel `p_phase` variables used to carry by hand.
type PTile = SharedTile<Bf16, TILE, TILE, Swizzle128B>;
/// The paired K-major operand: two adjacent 64-row tiles stacked into
/// `[QUERIES, HD]`, so each of its two HD subtiles is 128 rows and the K-major
/// MMA walk strides `TILE_BYTES` between them. The forward's resident Q, and
/// the backward's Q (dQ) or K/V (dK/dV).
type PairedPanel = SharedTile<Bf16, QUERIES, HD, Swizzle128B>;
/// The paired single-subtile `[QUERIES, TILE]` bf16 probability/gradient
/// operand: 128 swizzled 128-byte rows feeding an MMA's A side. The forward's
/// P, and the backward's dS / Pᵀ / dSᵀ.
type PairedPTile = SharedTile<Bf16, QUERIES, TILE, Swizzle128B>;
/// The forward's two-deep P ring: tile `i` writes the slot the output MMA of
/// tile `i - 2` finished reading.
type ProbabilityRing = SharedTileRing<Bf16, QUERIES, TILE, Swizzle128B, 2>;
const _: () = assert!(Panel::BYTES == TILE_BYTES && Panel::SUBTILE_BYTES == SUBTILE_BYTES);
const _: () = assert!(PTile::BYTES == SUBTILE_BYTES);
const _: () = assert!(PairedPanel::SUBTILE_BYTES == TILE_BYTES);
const _: () = assert!(PairedPTile::BYTES == 2 * SUBTILE_BYTES && PairedPTile::SUBTILES == 1);

/// The `S = Q·Kᵀ` (and `dP = dY·Vᵀ`) score accumulator segment. Every kernel
/// here fills all 128 of its `M128` rows.
type STmem = TmemTile<QUERIES, TILE>;
/// An HD-wide output/gradient accumulator segment (`O`, `dQ`, `dK`, `dV`) —
/// two 64-column MMA bands side by side.
type AccTmem = TmemTile<QUERIES, HD>;

/// Columns of a warp's score band that one pass of the softmax holds.
///
/// **Not the whole `[32, TILE]` band.** The output accumulator is 128
/// registers and resident for the whole key stream; a 64-wide score band
/// beside it is 64 more, and at that width the register ops' generic
/// `SLOTS x VALUES` loops stop scalarizing — the band lands in the LLVM local
/// depot and the drain, the mask, the reduction and the `exp2` all become
/// `ld.local`/`st.local` in the one loop this kernel cannot afford traffic in.
/// Measured: 1328 B of frame and 3546 local accesses at 64 columns, and
/// **2.635 ms** at the profile shape against the persistent kernel's 1.937.
/// At 16 the chunk is 16 registers, every pass stays in them, and the price is
/// a second drain of the segment — which is what `softmax_tile` paid too.
const SCORE_CHUNK: usize = 16;
const _: () = assert!(TILE.is_multiple_of(SCORE_CHUNK));

/// One `SCORE_CHUNK`-wide slice of a warp's score band, and the whole of its
/// output accumulator: the four warps of an `M128` drain own 32 TMEM lanes
/// each.
type ScoreChunk = RegTile<32, SCORE_CHUNK, BaseLdtm>;
type OutBand = RegTile<32, HD, BaseLdtm>;
/// A per-row statistic of one of those bands — the running max, the running
/// sum, the LSE.
type Rows = RegVec<32, BaseLdtm>;

/// Bytes of one full-width bf16 `[TILE, HD]` operand (two stacked subtiles).
const TILE_BYTES: usize = TILE * HD * 2;
/// Bytes of one 64-wide `[TILE, 64]` SWIZZLE_128B subtile — half a `TILE_BYTES`
/// panel.
const SUBTILE_BYTES: usize = TILE_BYTES / 2;

/// K/V ring depth of the forward (SWEEP knob). Two is the floor: the staggered
/// issue order (`S-MMA(i+1)` before `O-MMA(i)`) needs a stage of load-ahead to
/// make progress. Four is the ceiling the host-side launch allocation
/// (`host::FLASH_FORWARD_SMEM_BYTES`) is sized for.
pub const FORWARD_STAGES: usize = 3;
const _: () = assert!(2 <= FORWARD_STAGES && FORWARD_STAGES <= 4);
/// Depth of the `O = P·V` completion ring. Three rather than two because the
/// deepest wait on it reaches *two* tiles back — the probability slot tile `i`
/// overwrites was last read by the output MMA of tile `i - 2` — and a ring
/// cannot be as shallow as its own deepest wait.
const OUTPUT_LAG: usize = 3;
/// Warps of the forward: exactly the four an `M128` accumulator's 128 TMEM
/// lanes are drained by, which is why there is no separate MMA or TMA warp.
const FORWARD_WARPS: usize = QUERIES / 32;
/// Threads of the forward.
pub const FLASH_FORWARD_BLOCK: usize = QUERIES;
/// `#[launch_bounds]` only accepts integer literals, so the kernel's
/// `.maxntid` is spelled out; keep it equal to the block width. Declaring more
/// threads than are launched makes ptxas budget registers for a block that
/// never exists (`65536 / maxntid` per thread), which is how the HD=128
/// conversion's stale 128-row-era values silently squeezed the register-hungry
/// softmax warpgroup.
const _: () = assert!(FLASH_FORWARD_BLOCK == 128);
/// Dynamic shared plan of the forward, as the plan itself computes it.
pub const FLASH_FORWARD_SMEM: usize = forward_plan(SharedPlan::sizing()).plan.bytes();
/// Columns of tensor memory the forward allocates: two `[QUERIES, TILE]` score
/// segments plus the `[QUERIES, HD]` output beside them. `tcgen05.alloc` takes a
/// power of two in `[32, 512]`, and 256 is exactly the sum.
const FORWARD_TMEM_COLUMNS: u32 = 256;
const _: () = assert!(
    FORWARD_TMEM_COLUMNS as usize == 2 * TILE + HD
        && FORWARD_TMEM_COLUMNS.is_power_of_two()
        && FORWARD_TMEM_COLUMNS >= 32
        && FORWARD_TMEM_COLUMNS <= 512,
    "tcgen05.alloc takes a power of two in [32, 512] that covers the scores and the output"
);

/// Dynamic shared plan of the PAIRED query-parallel backward (kernel A, Design
/// B): the resident stacked `[Q_A;Q_B]` and `[dY_A;dY_B]` operands
/// (`2 * TILE_BYTES` each), the streamed K and V panels, and the single stacked
/// `[128, 64]` dS tile (`TILE_BYTES`). Every paired MMA fills 128 real query
/// rows, so there is no phantom-read pad in any plan here any more — the
/// forward's 64-query tile was the last thing that needed one.
pub const FLASH_BACKWARD_Q_SMEM: usize = 7 * TILE_BYTES;
/// Dynamic shared plan of the PAIRED key-parallel backward (kernel B, Design B):
/// the resident stacked `[K_A;K_B]` and `[V_A;V_B]` operands (`2 * TILE_BYTES`
/// each), the streamed Q and dY panels, and the stacked `[128, 64]` Pᵀ and dSᵀ
/// tiles (`TILE_BYTES` each).
pub const FLASH_BACKWARD_KV_SMEM: usize = 8 * TILE_BYTES;

/// K/V ring depth of the warp-specialized backward (SWEEP knob). Two is the
/// floor for the same reason as `FORWARD_STAGES`: the staggered issue order
/// (`S/dP-MMA(i)` before `dQ-MMA(i-1)`) needs a stage of load-ahead, and the
/// K stage of tile `i` cannot recycle until `dQ-MMA(i)` — which reads it a
/// second time — has been observed.
pub const BACKWARD_STAGES: usize = 3;
const _: () = assert!(2 <= BACKWARD_STAGES && BACKWARD_STAGES <= 4);
/// Dynamic shared plan of the PIPELINED query-parallel backward: the resident
/// stacked Q and dY pairs (`2 * TILE_BYTES` each), the K and V rings, and the
/// single stacked `[128, 64]` dS tile. dS stays single-buffered because the
/// warpgroup's `dq_done(i)` wait proves the gradient MMA finished reading it
/// before tile `i+1` overwrites it.
pub const FLASH_BACKWARD_Q_PIPELINED_SMEM: usize = (5 + 2 * BACKWARD_STAGES) * TILE_BYTES;
/// Threads of the pipelined backward: the 128-thread gradient warpgroup plus
/// the TMA-load warp and the MMA-issue warp.
pub const FLASH_BACKWARD_Q_PIPELINED_BLOCK: usize = QUERIES + 64;
/// Mirrors the `FLASH_FORWARD_BLOCK` `.maxntid` note.
const _: () = assert!(FLASH_BACKWARD_Q_PIPELINED_BLOCK == 192);
/// Dynamic shared plan of the PIPELINED key-parallel backward: the resident
/// stacked K and V pairs, the Q and dY rings, and the stacked Pᵀ and dSᵀ
/// tiles (both single-buffered, released by the same `grad_done` wait that
/// recycles the ring).
pub const FLASH_BACKWARD_KV_PIPELINED_SMEM: usize = (6 + 2 * BACKWARD_STAGES) * TILE_BYTES;
/// Threads of the pipelined key-parallel backward; same three roles.
pub const FLASH_BACKWARD_KV_PIPELINED_BLOCK: usize = QUERIES + 64;

/// Base-2 slack a tile's row max may climb above the O segment's reference
/// before the warpgroup forces a correction (SWEEP knob). P values reach at
/// most `2^CORRECTION_THRESHOLD`, comfortably inside bf16 range and the fp32
/// accumulation headroom of a full key stream.
pub const CORRECTION_THRESHOLD: f32 = 8.0;

/// The forward's shared plan, as a [`SharedPlan`] carve rather than a
/// column of `smem.add(n * TILE_BYTES)`: the resident `[2·TILE, HD]` query
/// pair, the K and V rings, the two-deep probability ring, every mbarrier,
/// the correction vote's scratch and the TMEM staging word.
struct Forward {
    q: PairedPanel,
    k: ForwardPanelRing,
    v: ForwardPanelRing,
    p: ProbabilityRing,
    /// One arrival per K/V stage, filled by the TMA engine.
    kv_loaded: SemaphoreRing<FORWARD_STAGES>,
    /// `S = Q·Kᵀ` completion, one per double-buffered score segment.
    scored: SemaphoreRing<2>,
    /// `O = P·V` completion. [`OUTPUT_LAG`] deep, not two: the deepest wait
    /// on it reaches two tiles back.
    accumulated: SemaphoreRing<OUTPUT_LAG>,
    q_loaded: Semaphore,
    /// `block_reduce`'s per-warp partials — the correction vote.
    votes: SharedVec<F32, FORWARD_WARPS>,
    tmem_slot: *mut u32,
    plan: SharedPlan,
}

#[inline(always)]
const fn forward_plan(at: SharedPlan) -> Forward {
    let (q, at) = at.tile::<Bf16, QUERIES, HD, Swizzle128B>();
    let (k, at) = at.tile_ring::<Bf16, TILE, HD, Swizzle128B, FORWARD_STAGES>();
    let (v, at) = at.tile_ring::<Bf16, TILE, HD, Swizzle128B, FORWARD_STAGES>();
    let (p, at) = at.tile_ring::<Bf16, QUERIES, TILE, Swizzle128B, 2>();
    let (kv_loaded, at) = at.semaphores::<FORWARD_STAGES>();
    let (scored, at) = at.semaphores::<2>();
    let (accumulated, at) = at.semaphores::<OUTPUT_LAG>();
    let (q_loaded, at) = at.semaphore();
    let (votes, at) = at.vec::<F32, FORWARD_WARPS>();
    let (tmem_slot, at) = at.tmem_slot();
    Forward {
        q,
        k,
        v,
        p,
        kv_loaded,
        scored,
        accumulated,
        q_loaded,
        votes,
        tmem_slot,
        plan: at,
    }
}

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

    /// Drain an accumulator segment and add it into the per-thread output
    /// accumulator: the forward's `O = P·V` on a correction and at the
    /// epilogue, and the backward's `dQ`/`dK`/`dV` at theirs. `warp_id` is
    /// warpgroup-local, and its four values cover the segment's 128 rows.
    #[inline(always)]
    unsafe fn merge_output_tile(
        o_tmem: AccTmem,
        warp_id: u32,
        out_acc: &mut RegTile<32, 128, BaseLdtm>,
    ) {
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

    /// Issue one `S = Q·Kᵀ` tile from the current leader thread — `M128_N64`,
    /// eight chained K=16 MMAs walking the two stacked HD subtiles of both
    /// operands; the caller owns the commit. `AR` is the A operand's row count,
    /// `QUERIES` for the forward's paired query block and `TILE` for the
    /// operand probe that only fills half the accumulator on purpose.
    #[inline(always)]
    unsafe fn score_mma<const AR: usize>(
        s_tmem: u32,
        q: SharedTile<Bf16, AR, HD, Swizzle128B>,
        k: Panel,
    ) {
        unsafe { mma_abt(s_tmem, q, k, MMA_SHAPE, false) }
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
                            let key_column = column + BaseLdtm::column(lane, value);
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
                    store_fragment(ds_chunks, tmem_row, column, lane, ds_tile);
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
                        let query_row = (column + BaseLdtm::column(lane, value)) as usize;
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
                            let query_column = column + BaseLdtm::column(lane, value);
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
                    store_fragment(p_chunks, tmem_row, column, lane, p_tile);
                    store_fragment(ds_chunks, tmem_row, column, lane, ds_tile);
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
        grad_acc: &RegTile<32, 128, BaseLdtm>,
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
                let charge = tile.tma_load(src_tma, 0, 0, tma);
                tma.expect_tx(charge);
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
    /// It is also the harness that settled why `M64_N64` is unusable, which is
    /// why every MMA in this module is `M128`: its shared-memory A operand
    /// reads only 32 distinct rows. Accumulator row `m` reads A-row
    /// `16*(m>>5) + (m&15)` — rows `m` and `m+16` alias and A-rows 32..63 are
    /// never read at all, i.e. the read drops datapath-lane bits [5:4]. It is a
    /// property of the `M64` SS-operand→datapath mapping, not of the
    /// descriptor: sweeping the A stride-byte-offset (512/1024/2048) moves
    /// *which* 32 rows are read and never reaches 64. Seed A and B host-side
    /// with structured encodings and
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
                let charge = a.tma_load(a_tma, 0, 0, tma) + b.tma_load(b_tma, 0, 0, tma);
                tma.expect_tx(charge);
            }
            tma.wait(0);
            thread::sync_threads();

            if is_leader {
                score_mma(c_tmem.raw(), a, b);
                mma::commit(mma);
            }
            mma.wait(0);
            thread::sync_threads();

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
                            let out_column =
                                column as usize + BaseLdtm::column(lane, value) as usize;
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

            // The stacked Q/dY pairs stay operand-A resident for the whole key
            // stream: tile A's rows land at the operand's row 0, tile B's at
            // row TILE, one box per HD subtile each.
            if is_leader {
                let row_a = (tile_a * TILE as u32) as i32;
                let row_b = (tile_b * TILE as u32) as i32;
                let charge = q.tma_load_at::<TILE>(q_tma, 0, row_a, plane, tma.sem())
                    + q.tma_load_at::<TILE>(q_tma, TILE, row_b, plane, tma.sem())
                    + dy.tma_load_at::<TILE>(dy_tma, 0, row_a, plane, tma.sem())
                    + dy.tma_load_at::<TILE>(dy_tma, TILE, row_b, plane, tma.sem());
                tma.sem().expect_tx(charge);
            }
            tma.wait_next();

            // The pair's 128 contiguous query rows' base-2 LSE and softmax dot.
            let query_row = (batch * t) as usize + tile_a as usize * TILE + tid as usize;
            let stat_index = query_row * h as usize + head as usize;
            (*(&raw mut LSE2 as *mut f32).add(tid as usize)) = logsumexp[stat_index] * LOG2E;
            (*(&raw mut DOTS as *mut f32).add(tid as usize)) = dot[stat_index];
            thread::sync_threads();

            let mut dq_acc = RegTile::<32, 128, BaseLdtm>::zero();

            let mut key_tile = 0u32;
            while key_tile <= tile_b {
                if is_leader {
                    let key_row = (key_tile * TILE as u32) as i32;
                    let charge = k.tma_load(k_tma, key_row, plane, tma.sem())
                        + v.tma_load(v_tma, key_row, plane, tma.sem());
                    tma.sem().expect_tx(charge);
                }
                tma.wait_next();
                thread::sync_threads();

                // S = [Q_A;Q_B]·Kᵀ and dP = [dY_A;dY_B]·Vᵀ, both fresh.
                if is_leader {
                    tcgen05_fence_after_thread_sync();
                    mma_abt(s_tmem.raw(), q, k, MMA_SHAPE, false);
                    mma_abt(dp_tmem.raw(), dy, v, MMA_SHAPE, false);
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
                    mma_ab(dq_tmem.raw(), ds, k, MMA_SHAPE, key_tile != 0);
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

            // The stacked K/V pairs stay operand-A resident for the whole query
            // stream.
            if is_leader {
                let row_a = (key_a * TILE as u32) as i32;
                let row_b = (key_b * TILE as u32) as i32;
                let charge = k.tma_load_at::<TILE>(k_tma, 0, row_a, plane, tma.sem())
                    + k.tma_load_at::<TILE>(k_tma, TILE, row_b, plane, tma.sem())
                    + v.tma_load_at::<TILE>(v_tma, 0, row_a, plane, tma.sem())
                    + v.tma_load_at::<TILE>(v_tma, TILE, row_b, plane, tma.sem());
                tma.sem().expect_tx(charge);
            }
            tma.wait_next();

            let mut dv_acc = RegTile::<32, 128, BaseLdtm>::zero();
            let mut dk_acc = RegTile::<32, 128, BaseLdtm>::zero();

            let mut query_tile = key_a;
            while query_tile < tiles {
                if is_leader {
                    let query_row = (query_tile * TILE as u32) as i32;
                    let charge = q.tma_load(q_tma, query_row, plane, tma.sem())
                        + dy.tma_load(dy_tma, query_row, plane, tma.sem());
                    tma.sem().expect_tx(charge);
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
                    mma_abt(st_tmem.raw(), k, q, MMA_SHAPE, false);
                    mma_abt(dpt_tmem.raw(), v, dy, MMA_SHAPE, false);
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
                    mma_ab(dv_tmem.raw(), p, dy, MMA_SHAPE, accumulate);
                    mma_ab(dk_tmem.raw(), ds, q, MMA_SHAPE, accumulate);
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
    ) {
        unsafe {
            ds_full.wait(i & 1);
            mma_ab(dq_tmem.raw(), ds, k.tile(i), MMA_SHAPE, i != 0);
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
            let v =
                PanelRingN::<BACKWARD_STAGES>::attach(smem.add((4 + BACKWARD_STAGES) * TILE_BYTES));
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
                let mut dq_acc = RegTile::<32, 128, BaseLdtm>::zero();
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
                    let mut charge = k.tile(i).tma_load(k_tma, key_row, plane, full)
                        + v.tile(i).tma_load(v_tma, key_row, plane, full);
                    if i == 0 {
                        let row_a = (tile_a * TILE as u32) as i32;
                        let row_b = (tile_b * TILE as u32) as i32;
                        charge = charge
                            + q.tma_load_at::<TILE>(q_tma, 0, row_a, plane, full)
                            + q.tma_load_at::<TILE>(q_tma, TILE, row_b, plane, full)
                            + dy.tma_load_at::<TILE>(dy_tma, 0, row_a, plane, full)
                            + dy.tma_load_at::<TILE>(dy_tma, TILE, row_b, plane, full);
                    }
                    full.expect_tx(charge);
                    i += 1;
                }
            } else if tid == group + 32 {
                // MMA warp leader.
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
                        MMA_SHAPE,
                        false,
                    );
                    mma_abt(
                        dp_tmem.columns_right(buffer).raw(),
                        dy,
                        v.tile(i),
                        MMA_SHAPE,
                        false,
                    );
                    mma::commit(s_full.sem(i));
                    if i > 0 {
                        gradient_mma(i - 1, ds, k, dq_tmem, ds_full, dq_done);
                    }
                    i += 1;
                }
                gradient_mma(key_tiles - 1, ds, k, dq_tmem, ds_full, dq_done);
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
    ) {
        unsafe {
            pds_full.wait(step & 1);
            let accumulate = step != 0;
            mma_ab(dv_tmem.raw(), p, dy.tile(step), MMA_SHAPE, accumulate);
            mma_ab(dk_tmem.raw(), ds, q.tile(step), MMA_SHAPE, accumulate);
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
            let dy =
                PanelRingN::<BACKWARD_STAGES>::attach(smem.add((4 + BACKWARD_STAGES) * TILE_BYTES));
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
                let mut dv_acc = RegTile::<32, 128, BaseLdtm>::zero();
                let mut dk_acc = RegTile::<32, 128, BaseLdtm>::zero();
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
                        let mut charge = q.tile(step).tma_load(q_tma, query_row, plane, full)
                            + dy.tile(step).tma_load(dy_tma, query_row, plane, full);
                        if step == 0 {
                            let row_a = (key_a * TILE as u32) as i32;
                            let row_b = (key_b * TILE as u32) as i32;
                            charge = charge
                                + k.tma_load_at::<TILE>(k_tma, 0, row_a, plane, full)
                                + k.tma_load_at::<TILE>(k_tma, TILE, row_b, plane, full)
                                + v.tma_load_at::<TILE>(v_tma, 0, row_a, plane, full)
                                + v.tma_load_at::<TILE>(v_tma, TILE, row_b, plane, full);
                        }
                        full.expect_tx(charge);
                    }
                    step += 1;
                }
            } else if tid == group + 32 {
                // MMA warp leader.
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
                        MMA_SHAPE,
                        false,
                    );
                    mma_abt(
                        dpt_tmem.columns_right(buffer).raw(),
                        v,
                        dy.tile(step),
                        MMA_SHAPE,
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

    /// The forward as a [`pipeline::Job`]: handles built once at launch, work
    /// items decoded per iteration, and the whole barrier set re-initialized
    /// per item by the harness — so each item's phase arithmetic starts from
    /// zero and a short key stream's unbalanced arrivals are wiped rather than
    /// threaded through parity math.
    struct ForwardStream<'k, 'c> {
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        t: u32,
        h: u32,
        tiles: u32,
        planes: u32,
        leader: bool,
        warp_id: u32,
        lane: u32,
        shared: Forward,
        scores: STmem,
        accumulator: AccTmem,
        y: GlobalRows<F32>,
        lse: GlobalRows<F32>,
        corrections: &'k mut DisjointSlice<'c, u32>,
    }

    impl ForwardStream<'_, '_> {
        /// One `SCORE_CHUNK`-wide slice of this warp's score band.
        #[inline(always)]
        unsafe fn chunk(&self, segment: STmem, band: u32, column: u32) -> ScoreChunk {
            unsafe { segment.tile(band, column) }
        }

        /// The double-buffered score segment tile `key_tile` lands in.
        #[inline(always)]
        fn score_segment(&self, key_tile: u32) -> STmem {
            if key_tile & 1 == 0 {
                self.scores
            } else {
                self.scores.columns_right(TILE as u32)
            }
        }

        /// Stage `key_tile`'s K and V panels, charging one barrier for both.
        #[inline(always)]
        unsafe fn load_kv(&self, key_tile: u32, plane: i32) {
            unsafe {
                let row = (key_tile * TILE as u32) as i32;
                let full = self.shared.kv_loaded.sem(key_tile);
                let charge = self
                    .shared
                    .k
                    .tile(key_tile)
                    .tma_load(self.k_tma, row, plane, full)
                    + self
                        .shared
                        .v
                        .tile(key_tile)
                        .tma_load(self.v_tma, row, plane, full);
                full.expect_tx(charge);
            }
        }
    }

    impl pipeline::Job for ForwardStream<'_, '_> {
        #[inline(always)]
        unsafe fn init(&self, _item: u32) {
            unsafe {
                self.shared.kv_loaded.init_all(1);
                self.shared.scored.init_all(1);
                self.shared.accumulated.init_all(1);
                self.shared.q_loaded.init(1);
            }
        }

        #[inline(always)]
        unsafe fn inval(&self) {
            unsafe {
                self.shared.kv_loaded.inval_all();
                self.shared.scored.inval_all();
                self.shared.accumulated.inval_all();
                self.shared.q_loaded.inval();
            }
        }

        #[inline(always)]
        unsafe fn work(&mut self, item: u32) {
            unsafe {
                // Descending query tile: the causal cost of tile `i` is `i + 1`
                // query-tile-heights of keys, so dealing the long streams first
                // is what keeps the tail of a persistent grid short.
                let query_tile = self.tiles - 1 - item / self.planes;
                let plane = item % self.planes;
                let head = plane % self.h;
                let batch = plane / self.h;
                let query_base = query_tile * QUERIES as u32;
                // A query block is two key tiles tall, and its last row attends
                // to its own key: `2 * query_tile + 2` tiles, the last two of
                // them crossing the diagonal.
                let key_tiles = 2 * query_tile + 2;
                let first_masked = 2 * query_tile;

                let (leader, warp_id, lane) = (self.leader, self.warp_id, self.lane);
                let band = 32 * warp_id;

                if leader {
                    tcgen05_fence_after_thread_sync();
                    // The query block is twice the map's box, so it arrives as
                    // two stacked `TILE`-row loads. `tma_load` would issue the
                    // right two boxes and charge the barrier for `QUERIES` rows
                    // of each, which the engine never delivers — a hang, not a
                    // wrong answer.
                    let charge = self.shared.q.tma_load_at::<TILE>(
                        self.q_tma,
                        0,
                        query_base as i32,
                        plane as i32,
                        self.shared.q_loaded,
                    ) + self.shared.q.tma_load_at::<TILE>(
                        self.q_tma,
                        TILE,
                        (query_base + TILE as u32) as i32,
                        plane as i32,
                        self.shared.q_loaded,
                    );
                    self.shared.q_loaded.expect_tx(charge);
                    let mut stage = 0u32;
                    while stage < FORWARD_STAGES as u32 && stage < key_tiles {
                        self.load_kv(stage, plane as i32);
                        stage += 1;
                    }
                    // Only the MMA reads Q and K, so only the issuing thread
                    // waits for them.
                    self.shared.q_loaded.wait(0);
                    self.shared.kv_loaded.wait(0);
                    score_mma(
                        self.score_segment(0).raw(),
                        self.shared.q,
                        self.shared.k.tile(0),
                    );
                    mma::commit(self.shared.scored.sem(0));
                }

                let mut m_ref = Rows::splat(MASKED_SCORE);
                let mut running_sum = Rows::splat(0.0);
                let mut out_acc = OutBand::zero();
                let mut corrections = 0u32;

                let mut key_tile = 0u32;
                while key_tile < key_tiles {
                    if leader {
                        // The next tile's scores are issued before this tile's
                        // softmax runs, so the tensor core and the warpgroup
                        // overlap. Its segment is the one drained last tile,
                        // which the item's `sync_threads` proved free.
                        if key_tile + 1 < key_tiles {
                            self.shared.kv_loaded.wait(key_tile + 1);
                            score_mma(
                                self.score_segment(key_tile + 1).raw(),
                                self.shared.q,
                                self.shared.k.tile(key_tile + 1),
                            );
                            mma::commit(self.shared.scored.sem(key_tile + 1));
                        }
                        // A K/V stage is free once the output MMA that read its
                        // V has completed.
                        let refill = key_tile + FORWARD_STAGES as u32 - 1;
                        if key_tile > 0 && refill < key_tiles {
                            self.shared.accumulated.wait(key_tile - 1);
                            self.load_kv(refill, plane as i32);
                        }
                    }

                    self.shared.scored.wait(key_tile);
                    let segment = self.score_segment(key_tile);
                    let key_base = key_tile * TILE as u32;
                    let masked = key_tile >= first_masked;

                    // Pass 1: this tile's row maxima. The band is walked a
                    // `SCORE_CHUNK` at a time and never held whole — see
                    // `SCORE_CHUNK` for what holding it whole costs.
                    let mut row_max = Rows::splat(MASKED_SCORE);
                    let mut column = 0u32;
                    while column < TILE as u32 {
                        let mut chunk = self.chunk(segment, band, column);
                        if masked {
                            // Both origins go in rather than their difference:
                            // that is negative for a band above the diagonal,
                            // and a `u32` subtraction there would wrap and mask
                            // nothing.
                            chunk.make_causal_at(
                                lane,
                                query_base + band,
                                key_base + column,
                                MASKED_SCORE,
                            );
                        }
                        row_max.max_assign(chunk.row_max());
                        column += SCORE_CHUNK as u32;
                    }

                    // FA4's conditional correction. The output segment keeps
                    // accumulating under `m_ref` until some row's tile max
                    // climbs more than `CORRECTION_THRESHOLD` above it; a
                    // restart is collective, since `enable_d` is one flag for
                    // the whole MMA, so the per-thread test folds to a warp
                    // value and then to a block one. Tile 0 always trips it —
                    // `m_ref` is still the sentinel — and starts the first
                    // segment without a drain.
                    let exceeded = row_max.any_exceeds(m_ref, CORRECTION_THRESHOLD);
                    let restart = block_reduce::<Max, FORWARD_WARPS>(
                        self.shared.votes,
                        warp_reduce::<Max>(if exceeded { 1.0 } else { 0.0 }),
                    ) != 0.0;
                    if restart {
                        if key_tile > 0 {
                            self.shared.accumulated.wait(key_tile - 1);
                            merge_output_tile(self.accumulator, warp_id, &mut out_acc);
                            corrections += 1;
                        }
                        online_rescale(&mut m_ref, row_max, &mut running_sum, &mut out_acc);
                    }

                    // The probability ring is two deep, so this slot was last
                    // read by the output MMA two tiles back.
                    if key_tile >= 2 {
                        self.shared.accumulated.wait(key_tile - 2);
                    }

                    // Pass 2: probabilities against the segment reference,
                    // re-draining the same chunks and storing each through
                    // `stmatrix` before the next is read.
                    let probabilities = self.shared.p.tile(key_tile).chunk_writer();
                    let mut tile_sum = Rows::splat(0.0);
                    let mut column = 0u32;
                    while column < TILE as u32 {
                        let mut chunk = self.chunk(segment, band, column);
                        if masked {
                            chunk.make_causal_at(
                                lane,
                                query_base + band,
                                key_base + column,
                                MASKED_SCORE,
                            );
                        }
                        chunk.sub_row_assign(m_ref);
                        chunk.unary_map_assign::<Exp2Approx>();
                        tile_sum.add_assign(chunk.row_sum());
                        store_tile(probabilities, band, column, lane, chunk);
                        column += SCORE_CHUNK as u32;
                    }
                    running_sum.add_assign(tile_sum);

                    // P was written through the generic proxy; fence before the
                    // async-proxy MMA reads it.
                    fence_proxy_async_shared_cta();
                    tcgen05_fence_before_thread_sync();
                    thread::sync_threads();
                    if leader {
                        tcgen05_fence_after_thread_sync();
                        mma_ab(
                            self.accumulator.raw(),
                            self.shared.p.tile(key_tile),
                            self.shared.v.tile(key_tile),
                            MMA_SHAPE,
                            !restart,
                        );
                        mma::commit(self.shared.accumulated.sem(key_tile));
                    }
                    key_tile += 1;
                }

                self.shared.accumulated.wait(key_tiles - 1);
                merge_output_tile(self.accumulator, warp_id, &mut out_acc);
                out_acc.scale_rows(running_sum.recip());

                let row = batch * self.t + query_base + band;
                store_rows(self.y, row, head * HD as u32, lane, out_acc);
                // The reference the sum is relative to trails the true row max
                // by at most the correction threshold, and adding it back is
                // what makes the LSE exact anyway.
                let mut lse = running_sum.log2();
                lse.add_assign(m_ref);
                lse.scale_assign(LN2);
                store_row_vec(self.lse, row, head, lane, lse);

                if leader {
                    *self
                        .corrections
                        .get_unchecked_mut((plane * self.tiles + query_tile) as usize) =
                        corrections;
                }
            }
        }
    }

    /// tcgen05 causal attention forward — one kernel, replacing the
    /// synchronous, pipelined and persistent generations of issue #35.
    ///
    /// Launch with `host::flash_forward_config`: a 1-D grid of at most
    /// `(T / QUERIES) * H * B` CTAs, `FLASH_FORWARD_BLOCK` threads,
    /// `host::FLASH_FORWARD_SMEM_BYTES` dynamic shared bytes.
    ///
    /// One work item is a (query tile, head, batch). A CTA owns `QUERIES = 128`
    /// query rows — **a whole `M128` accumulator, not half of one** — and
    /// streams the causal `TILE`-key tiles beneath them. That is the single
    /// change that collapses the three generations: at 64 query rows every MMA
    /// filled 64 real accumulator rows and 64 phantom ones, so half the tensor
    /// core was thrown away and the kernels bought it back with structure — two
    /// warpgroups ping-ponging two query tiles (persistent), a separate MMA warp
    /// and TMA warp to keep either one fed (pipelined). At 128 the warpgroup is
    /// four warps because the accumulator is 128 lanes, one softmax pass covers
    /// twice the keys, and the overlap that took three roles is a
    /// double-buffered score segment: the leader issues `S(i+1)` before the
    /// warpgroup runs the softmax of tile `i`, so the tensor core is running
    /// `S(i+1)` and `O(i-1)` throughout it.
    ///
    /// The rest is the library's. `SharedPlan` carves the plan,
    /// `pipeline::run` is the persistent work-item loop (items dealt by
    /// descending query tile, longest key streams first), `make_causal_at`
    /// masks against the *pair* of block origins, `online_rescale` is the flash
    /// rescale, and `block_reduce` is the correction vote three kernels used to
    /// open-code as a votes array and a barrier phase.
    ///
    /// Operand and output contracts are the extraction's: packed-bf16
    /// `[B*H, T, HD]` staging panels with Q pre-scaled by
    /// `softmax_scale * log2(e)`, fp32 `y[B*T, H*HD]`, fp32 `logsumexp[B*T, H]`
    /// in natural-log units, and one `correction_counts` word per work item
    /// (`plane * tiles + query_tile`) counting mid-stream segment restarts.
    #[kernel]
    #[launch_bounds(128, 1)]
    pub unsafe fn flash_forward(
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
            if thread::blockDim_x() as usize != FLASH_FORWARD_BLOCK {
                return;
            }
            let shared = forward_plan(SharedPlan::attach());
            let tmem = alloc_block(shared.tmem_slot, FORWARD_TMEM_COLUMNS);
            // Two `[QUERIES, TILE]` score segments, then the `[QUERIES, HD]`
            // output beside them: 64 + 64 + 128 of the 256 columns.
            let scores = STmem::from_raw(tmem);
            let accumulator: AccTmem = scores.columns_right(TILE as u32).split_columns();

            let tiles = sequence_length / QUERIES as u32;
            let planes = heads * batches;
            let mut job = ForwardStream {
                q_tma,
                k_tma,
                v_tma,
                t: sequence_length,
                h: heads,
                tiles,
                planes,
                leader: thread::threadIdx_x() == 0,
                warp_id: warp::warp_id(),
                lane: warp::lane_id(),
                shared,
                scores,
                accumulator,
                y: GlobalRows::<F32>::from_slice(&mut output, heads as usize * HD),
                lse: GlobalRows::<F32>::from_slice(&mut logsumexp, heads as usize),
                corrections: &mut correction_counts,
            };
            pipeline::run(&mut job, tiles * planes);
            dealloc_block(tmem, FORWARD_TMEM_COLUMNS);
        }
    }
}
