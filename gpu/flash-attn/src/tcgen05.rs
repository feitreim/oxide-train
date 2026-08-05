//! FA4-shaped tcgen05 attention forward (issue #68) and backward (issue #35
//! phase 4).
//!
//! At cuda-oxide b099f64 this module shares a pure-PTX artifact with the
//! libdevice-backed oracle kernels. The LSE epilogue's software `log2` is a
//! deliberate FA4 SFU-offload optimization, not an artifact-path workaround;
//! the forward softmax's `exp2` is the SFU one (#68), because a four-warp
//! warpgroup that is not tensor-core bound has nothing to offload to.
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
//! `O += P·V` accumulates in a TMEM segment under a fixed per-row max
//! reference (`enable_d` across every tile but the first). FA4's conditional
//! correction: only when some row's tile max climbs more than
//! `CORRECTION_THRESHOLD` above the reference does the warpgroup drain the
//! segment, rescale it, and `tcgen05.st` it back — otherwise the segment just
//! keeps accumulating and the warpgroup never touches O TMEM. **O never
//! reaches a resident register band**, which is what keeps this kernel's frame
//! at zero; the earlier scheme, written when `tcgen05.st` was missing from the
//! library, restarted the segment and carried the running output in a
//! 128-register accumulator that the LLVM local depot took whole.
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
//! `TmemTile::tile_x8`/`store_tile` for the rescale's round trip, and
//! `block_reduce` for the correction vote all three kernels used to open-code.
//!
//! `flash_backward_q` and `flash_backward_kv` are one kernel each where there
//! were two (#69), and they are the forward's own shape: a CTA owns `QUERIES`
//! rows of one axis, the four warps of the `M128` accumulator are the whole
//! block, and the leader issues the next tile's score MMAs before the
//! warpgroup runs this tile's register pass. `flash_backward_q`
//! (query-parallel) holds Q and dY resident, streams the causal key tiles,
//! recomputes `S`/`dP` per tile and accumulates `dQ += dS·K` in TMEM;
//! `flash_backward_kv` (key-parallel) holds K and V resident, streams the
//! query tiles at and after them, recomputes the transposed `Sᵀ`/`dPᵀ` and
//! accumulates `dV += Pᵀ·dY` and `dK += dSᵀ·Q`. The transpose is the MMA's,
//! not a register one: `mma_abt(K, Q)` produces the key-major band directly,
//! so nothing here wants the second `FragmentLayout` the library does not
//! have.
//!
//! Probabilities are recomputed base-2 from the saved LSE
//! (`P = exp2(s − lse·log2e)`, no running-max machinery). **The gradient
//! accumulators never leave TMEM until the item ends**: a query block owns
//! every key its rows attend to and a key block owns every query that attends
//! to it, so each output tile has exactly one writer and the epilogue is a
//! plain `store_rows`. That is what keeps atomics — and TMA reduction stores,
//! ferro #42 — out of this module; the price is that `S` is computed twice
//! across the two kernels, which a fused form would not pay and could not
//! avoid paying for in gradient traffic instead. The three-kernel split
//! (`backward_dot` stays fp32 in `lib.rs`, then dQ, then dK/dV) is unchanged.
//! Both take the packed-bf16 Q/K/V/dY staging panels plus the read-only
//! `logsumexp` (natural log) and `dot` (`Σ dy·y`) device slices, and write
//! fp32 `dq`/`dk`/`dv`.

use cuda_device::DisjointSlice;
use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta};
use cuda_device::shared::{DynamicSharedArray, SharedArray};
use cuda_device::tcgen05::{tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cuda_module, kernel, launch_bounds, thread, warp};
use kittens::global::{GlobalRows, load_col_vec, load_row_vec, store_row_vec, store_rows};
use kittens::ldst::store_tile;
use kittens::mma::{self, MmaShape, mma_ab, mma_abt};
use kittens::pipeline;
use kittens::plan::SharedPlan;
use kittens::reg::{
    BaseLdtm, ColVec, Exp2Hw, Max, RegTile, RegVec, exp2_approx, log2_approx, warp_reduce,
};
use kittens::shared::{Bf16, F32, SharedTile, SharedTileRing, SharedVec, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing, block_reduce};
use kittens::tmem::{TmemTile, alloc_block, dealloc_block, store_wait};

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
/// The forward's P ring: tile `i` writes the slot the output MMA of tile
/// `i - PROBABILITY_STAGES` finished reading.
type ProbabilityRing = SharedTileRing<Bf16, QUERIES, TILE, Swizzle128B, PROBABILITY_STAGES>;
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
/// **Not the whole `[32, TILE]` band.** At 64 columns the register ops'
/// generic `SLOTS x VALUES` loops stop scalarizing — the band lands in the LLVM
/// local depot and the drain, the mask, the reduction and the `exp2` all become
/// `ld.local`/`st.local` in the one loop this kernel cannot afford traffic in.
/// Measured: 1328 B of frame and 3546 local accesses at 64 columns, and
/// **2.635 ms** at the profile shape against the persistent kernel's 1.937.
/// At 16 the chunk is 16 registers, every pass stays in them, and the price is
/// a second drain of the segment — which is what `softmax_tile` paid too.
///
/// This is [`OutHalf`]'s rule on the other axis, and the two are the whole of
/// what keeps this kernel's frame at zero: no fp32 value here is wider than the
/// registers that hold it, and everything wider lives in tensor memory.
const SCORE_CHUNK: usize = 16;
const _: () = assert!(TILE.is_multiple_of(SCORE_CHUNK));

/// One `SCORE_CHUNK`-wide slice of a warp's score band: the four warps of an
/// `M128` drain own 32 TMEM lanes each.
type ScoreChunk = RegTile<32, SCORE_CHUNK, BaseLdtm>;
/// A warp's whole `[32, HD]` output band. The backward kernels' drain, where
/// the segment goes straight out and the band is one value with one use.
type OutBand = RegTile<32, HD, BaseLdtm>;
/// One 64-column group of that band — the `.x8` drain's own width, and the
/// widest fp32 value the *forward* holds.
///
/// **Not the whole band**, for `SCORE_CHUNK`'s reason one axis over: a band
/// that is transformed on the way out is two 128-register values at once (the
/// drain and the rescaled result) or one with its address taken, and either is
/// a band that does not fit (ferro #181). Measured at the profile shape: the
/// resident `&mut` accumulator this replaced was 560 B of frame and
/// **1.212 ms**; draining the segment whole into a short-lived `&mut` band was
/// 1072 B and 0.996; the by-value 128 was 253 registers, over the ptxas pin.
/// At 64 the drain, the rescale and the store all stay in registers.
type OutHalf = RegTile<32, TILE, BaseLdtm>;
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
pub const FORWARD_STAGES: usize = 2;
const _: () = assert!(2 <= FORWARD_STAGES && FORWARD_STAGES <= 4);
/// Buffers of the probability tile. One, which is the floor: the output MMA of
/// tile `i - 1` has to have finished reading P before tile `i` overwrites it,
/// and at one buffer that wait is a tile closer than at two. It is one because
/// the whole plan has to fit under half an SM's shared memory to get a second
/// CTA on it, and this tile is 16 KiB of the 114688 that does.
pub const PROBABILITY_STAGES: usize = 1;
const _: () = assert!(PROBABILITY_STAGES >= 1 && PROBABILITY_STAGES < OUTPUT_LAG);
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
pub const FORWARD_TMEM_COLUMNS: u32 = 256;
const _: () = assert!(
    FORWARD_TMEM_COLUMNS as usize == 2 * TILE + HD
        && FORWARD_TMEM_COLUMNS.is_power_of_two()
        && FORWARD_TMEM_COLUMNS >= 32
        && FORWARD_TMEM_COLUMNS <= 512,
    "tcgen05.alloc takes a power of two in [32, 512] that covers the scores and the output"
);

/// Ring depth of each backward kernel's streamed operand — K/V in the
/// query-parallel kernel, Q/dY in the key-parallel one (SWEEP knob).
///
/// Three where the forward now takes two, and for a reason that is the
/// forward's own turned around: the forward dropped to two to fit two CTAs on
/// an SM, and the backward cannot have two at any plan size — both kernels
/// hold their score segments and their gradient accumulators in tensor memory
/// and land on `tcgen05.alloc`'s 512 columns, which is the whole of an SM's.
/// With residency fixed there is nothing to buy by starving the ring, and at
/// three the stage a score MMA waits for was loaded a whole iteration before
/// it, rather than by the refill immediately above it.
pub const BACKWARD_STAGES: usize = 3;
const _: () = assert!(2 <= BACKWARD_STAGES && BACKWARD_STAGES <= 4);
/// Buffers of the bf16 gradient operand a backward kernel hands its gradient
/// MMA — `dS` in kernel A, `Pᵀ` and `dSᵀ` in kernel B.
///
/// Two, unlike the forward's probability tile, because nothing here is
/// competing for a second CTA's worth of shared memory and the slack is worth
/// having: at two, the register pass of tile `i` writes the slot the gradient
/// MMA of tile `i - 2` finished reading, and at one it would open by waiting
/// on the MMA the previous pass had just issued.
const GRADIENT_STAGES: usize = 2;
/// Depth of the gradient-MMA completion ring, which is one more than the
/// operand ring it recycles: its deepest wait reaches `GRADIENT_STAGES` tiles
/// back, and a ring cannot be as shallow as its own deepest wait.
const GRADIENT_LAG: usize = GRADIENT_STAGES + 1;
const _: () = assert!(GRADIENT_STAGES >= 1);

/// A backward kernel's streamed-operand ring.
type BackwardPanelRing = PanelRingN<BACKWARD_STAGES>;
/// The bf16 gradient operand ring both backward kernels hand their gradient
/// MMAs: `[QUERIES, TILE]` tiles, `GRADIENT_STAGES` deep.
type GradientRing = SharedTileRing<Bf16, QUERIES, TILE, Swizzle128B, GRADIENT_STAGES>;
/// A per-column statistic of a `SCORE_CHUNK`-wide slice — the key-parallel
/// backward's LSE and dot, which its transposed band indexes by column where
/// the query-parallel one indexes the same numbers by row.
type Cols = ColVec<SCORE_CHUNK, BaseLdtm>;

/// Tensor-memory columns a backward kernel allocates. Kernel A wants 384 (two
/// score segments, two gradient segments, one `[QUERIES, HD]` accumulator) and
/// kernel B wants all 512 (a second accumulator); `tcgen05.alloc` takes a
/// power of two, so both round to the same number and both pin an SM to one
/// CTA. That is why neither plan below is written against a shared-memory
/// budget the way the forward's is.
pub const BACKWARD_TMEM_COLUMNS: u32 = 512;

/// The query-parallel backward's shared plan: the resident `[QUERIES, HD]`
/// query and output-gradient blocks, the streamed K and V rings, the dS ring,
/// every mbarrier and the TMEM staging word.
struct BackwardQ {
    q: PairedPanel,
    dy: PairedPanel,
    k: BackwardPanelRing,
    v: BackwardPanelRing,
    ds: GradientRing,
    /// One arrival per K/V stage, filled by the TMA engine.
    kv_loaded: SemaphoreRing<BACKWARD_STAGES>,
    /// `S = Q·Kᵀ` and `dP = dY·Vᵀ` completion — one arrival for the pair, one
    /// per double-buffered score segment.
    scored: SemaphoreRing<2>,
    /// `dQ += dS·K` completion. It recycles three things at three different
    /// depths: the dS slot two tiles back, the K/V stage
    /// `BACKWARD_STAGES - 1` ahead, and the accumulator itself at the end.
    accumulated: SemaphoreRing<GRADIENT_LAG>,
    qdy_loaded: Semaphore,
    tmem_slot: *mut u32,
    plan: SharedPlan,
}

#[inline(always)]
const fn backward_q_plan(at: SharedPlan) -> BackwardQ {
    let (q, at) = at.tile::<Bf16, QUERIES, HD, Swizzle128B>();
    let (dy, at) = at.tile::<Bf16, QUERIES, HD, Swizzle128B>();
    let (k, at) = at.tile_ring::<Bf16, TILE, HD, Swizzle128B, BACKWARD_STAGES>();
    let (v, at) = at.tile_ring::<Bf16, TILE, HD, Swizzle128B, BACKWARD_STAGES>();
    let (ds, at) = at.tile_ring::<Bf16, QUERIES, TILE, Swizzle128B, GRADIENT_STAGES>();
    let (kv_loaded, at) = at.semaphores::<BACKWARD_STAGES>();
    let (scored, at) = at.semaphores::<2>();
    let (accumulated, at) = at.semaphores::<GRADIENT_LAG>();
    let (qdy_loaded, at) = at.semaphore();
    let (tmem_slot, at) = at.tmem_slot();
    BackwardQ {
        q,
        dy,
        k,
        v,
        ds,
        kv_loaded,
        scored,
        accumulated,
        qdy_loaded,
        tmem_slot,
        plan: at,
    }
}

/// The key-parallel backward's shared plan: the resident K and V blocks, the
/// streamed Q and dY rings, and **two** gradient operand rings, since its
/// gradient MMA pair reads `Pᵀ` and `dSᵀ` where kernel A's reads only `dS`.
struct BackwardKv {
    k: PairedPanel,
    v: PairedPanel,
    q: BackwardPanelRing,
    dy: BackwardPanelRing,
    p: GradientRing,
    ds: GradientRing,
    qdy_loaded: SemaphoreRing<BACKWARD_STAGES>,
    scored: SemaphoreRing<2>,
    /// `dV += Pᵀ·dY` and `dK += dSᵀ·Q` completion, one arrival for the pair.
    accumulated: SemaphoreRing<GRADIENT_LAG>,
    kv_loaded: Semaphore,
    tmem_slot: *mut u32,
    plan: SharedPlan,
}

#[inline(always)]
const fn backward_kv_plan(at: SharedPlan) -> BackwardKv {
    let (k, at) = at.tile::<Bf16, QUERIES, HD, Swizzle128B>();
    let (v, at) = at.tile::<Bf16, QUERIES, HD, Swizzle128B>();
    let (q, at) = at.tile_ring::<Bf16, TILE, HD, Swizzle128B, BACKWARD_STAGES>();
    let (dy, at) = at.tile_ring::<Bf16, TILE, HD, Swizzle128B, BACKWARD_STAGES>();
    let (p, at) = at.tile_ring::<Bf16, QUERIES, TILE, Swizzle128B, GRADIENT_STAGES>();
    let (ds, at) = at.tile_ring::<Bf16, QUERIES, TILE, Swizzle128B, GRADIENT_STAGES>();
    let (qdy_loaded, at) = at.semaphores::<BACKWARD_STAGES>();
    let (scored, at) = at.semaphores::<2>();
    let (accumulated, at) = at.semaphores::<GRADIENT_LAG>();
    let (kv_loaded, at) = at.semaphore();
    let (tmem_slot, at) = at.tmem_slot();
    BackwardKv {
        k,
        v,
        q,
        dy,
        p,
        ds,
        qdy_loaded,
        scored,
        accumulated,
        kv_loaded,
        tmem_slot,
        plan: at,
    }
}

/// Dynamic shared plan of the query-parallel backward, as the plan computes it.
pub const FLASH_BACKWARD_Q_SMEM: usize = backward_q_plan(SharedPlan::sizing()).plan.bytes();
/// Dynamic shared plan of the key-parallel backward.
pub const FLASH_BACKWARD_KV_SMEM: usize = backward_kv_plan(SharedPlan::sizing()).plan.bytes();
/// Dynamic shared memory one CTA may opt into on a B200: the SM's 228 KiB less
/// the KiB the driver keeps. Kernel B's plan is the largest in the repo and
/// this is the only thing bounding it, since its residency is already one.
const MAX_DYNAMIC_SMEM: usize = 232_448;
const _: () = assert!(FLASH_BACKWARD_Q_SMEM <= MAX_DYNAMIC_SMEM);
const _: () = assert!(FLASH_BACKWARD_KV_SMEM <= MAX_DYNAMIC_SMEM);
/// Warps of either backward kernel that run the register pass: the four an
/// `M128` accumulator's 128 TMEM lanes are drained by.
const PASS_WARPS: u32 = QUERIES as u32 / 32;
/// The warp that issues every TMA and every MMA, and runs no pass.
///
/// **It is a fifth warp rather than one of the four**, which is the whole of
/// issue #94's first remedy. The leader used to be thread 0 of the pass
/// warpgroup, so its MMA issue — measured at 2 108 / 2 628 ticks, 42–49% of a
/// tile visit — ran *before* its own register pass and the other three warps
/// waited it out at the block barrier. The phase budget put 99% of a visit in
/// that one serial chain. Split across two warps the tile costs
/// `max(issue, pass)` instead of their sum.
///
/// This is not the warp specialization #75 deleted. That was 192 threads with
/// a TMA warp *and* an MMA warp, six barrier sets, and a per-query statistic
/// re-staged through a shared ring because specialization had removed the
/// block sync it hid behind. Here every ring, every barrier and the one
/// `sync_threads` per tile are exactly as #75 left them; the only change is
/// which warp calls `mma_*`.
const ISSUE_WARP: u32 = PASS_WARPS;
/// Threads of either backward kernel: the accumulator's four drain warps plus
/// the issue warp.
pub const FLASH_BACKWARD_BLOCK: usize = QUERIES + 32;
/// Mirrors the `FLASH_FORWARD_BLOCK` `.maxntid` note.
const _: () = assert!(FLASH_BACKWARD_BLOCK == 160);

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
    let (p, at) = at.tile_ring::<Bf16, QUERIES, TILE, Swizzle128B, PROBABILITY_STAGES>();
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

/// The two backward kernels with a stopwatch on every phase — a child module so
/// that it sees the plans and the tile contract without either becoming crate
/// surface. Reached only by the `probe` bin, through the entry points below.
#[path = "phase_probe.rs"]
pub mod phase_probe;

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

    /// The query-parallel backward as a [`pipeline::Job`], the forward's own
    /// shape: one work item is a (query block, head, batch), the barrier set is
    /// re-initialized per item by the harness, and the four warps that drain an
    /// `M128` accumulator are the whole block.
    struct BackwardQStream {
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        t: u32,
        h: u32,
        tiles: u32,
        planes: u32,
        leader: bool,
        warp_id: u32,
        lane: u32,
        shared: BackwardQ,
        /// Two `[QUERIES, TILE]` score segments, side by side.
        scores: STmem,
        /// Two more for `dP`.
        gradients: STmem,
        /// `dQ`, resident for the whole key stream.
        accumulator: AccTmem,
        lse: GlobalRows<F32>,
        dot: GlobalRows<F32>,
        dq: GlobalRows<F32>,
    }

    impl BackwardQStream {
        /// The double-buffered half of `segment` that key tile `key_tile` lands
        /// in — the one the register pass drained two tiles ago.
        #[inline(always)]
        fn buffer(&self, segment: STmem, key_tile: u32) -> STmem {
            if key_tile & 1 == 0 {
                segment
            } else {
                segment.columns_right(TILE as u32)
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

        /// Issue `S = Q·Kᵀ` and `dP = dY·Vᵀ` for one key tile into its score
        /// segment, and publish both on one arrival: the register pass reads
        /// them together and there is nothing it could do with one alone.
        ///
        /// `MMA_SHAPE` is the *band* the accumulator gets, which for a
        /// `[TILE, HD]` B operand read transposed is its `TILE` rows — the one
        /// argument no operand can supply, and the one ferro #175 found two
        /// callers reading as the tile's.
        #[inline(always)]
        unsafe fn score_mmas(&self, key_tile: u32) {
            unsafe {
                mma_abt(
                    self.buffer(self.scores, key_tile).raw(),
                    self.shared.q,
                    self.shared.k.tile(key_tile),
                    MMA_SHAPE,
                    false,
                );
                mma_abt(
                    self.buffer(self.gradients, key_tile).raw(),
                    self.shared.dy,
                    self.shared.v.tile(key_tile),
                    MMA_SHAPE,
                    false,
                );
                mma::commit(self.shared.scored.sem(key_tile));
            }
        }
    }

    impl pipeline::Job for BackwardQStream {
        #[inline(always)]
        unsafe fn init(&self, _item: u32) {
            unsafe {
                self.shared.kv_loaded.init_all(1);
                self.shared.scored.init_all(1);
                self.shared.accumulated.init_all(1);
                self.shared.qdy_loaded.init(1);
            }
        }

        #[inline(always)]
        unsafe fn inval(&self) {
            unsafe {
                self.shared.kv_loaded.inval_all();
                self.shared.scored.inval_all();
                self.shared.accumulated.inval_all();
                self.shared.qdy_loaded.inval();
            }
        }

        #[inline(always)]
        unsafe fn work(&mut self, item: u32) {
            unsafe {
                // Descending query block, for the forward's reason: block `i`
                // streams `2i + 2` key tiles, so dealing the long streams first
                // keeps the tail of a persistent grid short. The imbalance is
                // steeper here than in the forward only in that there is more
                // of it per tile.
                let query_block = self.tiles - 1 - item / self.planes;
                let plane = item % self.planes;
                let head = plane % self.h;
                let batch = plane / self.h;
                let query_base = query_block * QUERIES as u32;
                let key_tiles = 2 * query_block + 2;
                let first_masked = 2 * query_block;

                let (leader, warp_id, lane) = (self.leader, self.warp_id, self.lane);
                let band = 32 * warp_id;
                let issuing = warp_id == ISSUE_WARP;

                if leader {
                    tcgen05_fence_after_thread_sync();
                    // Q and dY are operand-A resident for the whole key stream.
                    // Each block is twice the map's box, so it arrives as two
                    // stacked `TILE`-row loads — `tma_load` would charge the
                    // barrier for `QUERIES` rows the engine never delivers.
                    let full = self.shared.qdy_loaded;
                    let second = (query_base + TILE as u32) as i32;
                    let charge = self.shared.q.tma_load_at::<TILE>(
                        self.q_tma,
                        0,
                        query_base as i32,
                        plane as i32,
                        full,
                    ) + self.shared.q.tma_load_at::<TILE>(
                        self.q_tma,
                        TILE,
                        second,
                        plane as i32,
                        full,
                    ) + self.shared.dy.tma_load_at::<TILE>(
                        self.dy_tma,
                        0,
                        query_base as i32,
                        plane as i32,
                        full,
                    ) + self.shared.dy.tma_load_at::<TILE>(
                        self.dy_tma,
                        TILE,
                        second,
                        plane as i32,
                        full,
                    );
                    full.expect_tx(charge);
                    let mut stage = 0u32;
                    while stage < BACKWARD_STAGES as u32 && stage < key_tiles {
                        self.load_kv(stage, plane as i32);
                        stage += 1;
                    }
                    // Only the MMA reads these operands, so only the issuing
                    // thread waits for them.
                    full.wait(0);
                    self.shared.kv_loaded.wait(0);
                    self.score_mmas(0);
                }

                // Both roles run `key_tiles` iterations and meet at the one
                // `sync_threads` that publishes dS, so the barrier counts match
                // whichever branch a warp is in. The issue warp's iteration is
                // the refill, the next tile's score MMAs and — after the
                // barrier — this tile's gradient MMA; the pass warps' iteration
                // is the register pass. They are the two halves the phase
                // budget found serialized in warp 0, and the tile now costs the
                // longer of them (issue #94).
                if issuing {
                    let mut key_tile = 0u32;
                    while key_tile < key_tiles {
                        if lane == 0 {
                            // Refill before issuing, for the forward's reason: at
                            // the ring's floor the stage the score MMA below waits
                            // for is the one this refill loads, and issuing first
                            // deadlocks the issuer against a load it has not made
                            // yet. A K/V stage is free once the gradient MMA that
                            // read its K completed — K is read twice, by the score
                            // MMA and again by `dQ += dS·K`, and the second is the
                            // later.
                            let refill = key_tile + BACKWARD_STAGES as u32 - 1;
                            if key_tile > 0 && refill < key_tiles {
                                self.shared.accumulated.wait(key_tile - 1);
                                self.load_kv(refill, plane as i32);
                            }
                            // The next tile's scores are issued before this tile's
                            // register pass ends, so the tensor core is producing
                            // `S(i+1)`/`dP(i+1)` while the pass warps form `dS(i)`
                            // and `dQ(i-1)` accumulates behind it.
                            if key_tile + 1 < key_tiles {
                                self.shared.kv_loaded.wait(key_tile + 1);
                                self.score_mmas(key_tile + 1);
                            }
                        }
                        tcgen05_fence_before_thread_sync();
                        thread::sync_threads();
                        if lane == 0 {
                            tcgen05_fence_after_thread_sync();
                            mma_ab(
                                self.accumulator.raw(),
                                self.shared.ds.tile(key_tile),
                                self.shared.k.tile(key_tile),
                                MMA_SHAPE,
                                key_tile != 0,
                            );
                            mma::commit(self.shared.accumulated.sem(key_tile));
                        }
                        key_tile += 1;
                    }
                } else {
                    // The block's 128 query rows are contiguous and each carries
                    // one saved f32 in the head's own column of `[rows, heads]`,
                    // which is a `RegVec`'s shape exactly — the statistic reaches
                    // registers without a staging tile, a scatter or a barrier
                    // (ferro #170).
                    let row = batch * self.t + query_base + band;
                    let mut lse2: Rows = load_row_vec(self.lse, row, head, lane);
                    lse2.scale_assign(LOG2E);
                    let dots: Rows = load_row_vec(self.dot, row, head, lane);

                    let mut key_tile = 0u32;
                    while key_tile < key_tiles {
                        self.shared.scored.wait(key_tile);
                        // The dS slot this tile writes was last read by the
                        // gradient MMA `GRADIENT_STAGES` tiles back.
                        if key_tile >= GRADIENT_STAGES as u32 {
                            self.shared
                                .accumulated
                                .wait(key_tile - GRADIENT_STAGES as u32);
                        }

                        let scores = self.buffer(self.scores, key_tile);
                        let gradients = self.buffer(self.gradients, key_tile);
                        let key_base = key_tile * TILE as u32;
                        let masked = key_tile >= first_masked;
                        let ds = self.shared.ds.tile(key_tile).chunk_writer();

                        // `dS = exp2(s - lse2)·(dP - D)·scale`, a `SCORE_CHUNK` of
                        // columns at a time. Never the whole `[32, TILE]` band as
                        // a value: that is what put the forward's softmax in the
                        // LLVM local depot and cost it 19.9%, and the two segments
                        // read here would be twice the band.
                        let mut column = 0u32;
                        while column < TILE as u32 {
                            let mut dscore: ScoreChunk = scores.tile(band, column);
                            if masked {
                                // Both origins go in rather than their difference,
                                // which is negative for a band above the diagonal.
                                // Masking the *score* rather than the gradient is
                                // what lets one call replace the `keep` predicate
                                // the four kernels this replaces each rebuilt: a
                                // masked score exp2s to a subnormal and multiplies
                                // the finite `dP - D` to nothing.
                                dscore.make_causal_at(
                                    lane,
                                    query_base + band,
                                    key_base + column,
                                    MASKED_SCORE,
                                );
                            }
                            dscore.sub_row_assign(lse2);
                            dscore.unary_map_assign::<Exp2Hw>();
                            let mut dp: ScoreChunk = gradients.tile(band, column);
                            dp.sub_row_assign(dots);
                            dscore.mul_assign(dp);
                            // The MMA above reads an unscaled K, so `scale` lands
                            // here; the staged Q already carries `scale·log2e`.
                            dscore.scale_assign(SCALE);
                            store_tile(ds, band, column, lane, dscore);
                            column += SCORE_CHUNK as u32;
                        }

                        // dS was written through the generic proxy; fence before
                        // the async-proxy MMA reads it.
                        fence_proxy_async_shared_cta();
                        tcgen05_fence_before_thread_sync();
                        thread::sync_threads();
                        key_tile += 1;
                    }

                    self.shared.accumulated.wait(key_tiles - 1);
                    // One drain, at the end of the stream: dQ is a complete sum,
                    // so there is no `1/sum` and no correction path, and the band
                    // goes straight out. `.x8` because nothing is added to it on
                    // the way.
                    let dq: OutBand = self.accumulator.tile_x8(band, 0);
                    store_rows(self.dq, row, head * HD as u32, lane, dq);
                }
            }
        }
    }

    /// tcgen05 causal attention backward, query-parallel half — one kernel,
    /// replacing the synchronous and warp-specialized generations of issue #47.
    ///
    /// Launch with `host::flash_backward_q_config`: a 1-D grid of at most
    /// `(T / QUERIES) * H * B` CTAs, `FLASH_BACKWARD_BLOCK` threads,
    /// `host::FLASH_BACKWARD_Q_SMEM_BYTES` dynamic shared bytes.
    ///
    /// One work item is a (query block, head, batch). A CTA owns `QUERIES = 128`
    /// query rows, holds their Q and dY resident, and streams the causal
    /// `TILE`-key tiles beneath them: `S = Q·Kᵀ` and `dP = dY·Vᵀ` land in a
    /// double-buffered fp32 TMEM segment pair, the warpgroup forms
    /// `dS = P·(dP − D)·scale` in registers a `SCORE_CHUNK` at a time and packs
    /// it back to shared memory, and `dQ += dS·K` accumulates in TMEM for the
    /// whole stream. **dQ never leaves tensor memory until the item ends**: a
    /// query block owns every key its rows attend to, so its output tile has one
    /// writer and the epilogue is a `store_rows`.
    ///
    /// What made this two kernels was the same thing that made the forward
    /// three — a synchronous form that exposed TMA, MMA, the register pass and
    /// the gradient MMA behind four block barriers each key tile, and a
    /// warp-specialized form that bought them back with a dedicated TMA warp and
    /// MMA warp over 192 threads. Neither is needed once the leader issues
    /// `S(i+1)` before the pass of tile `i`: the tensor core is running the next
    /// tile's scores and the previous tile's `dQ` throughout it, at 128 threads
    /// and **one** block barrier per key tile, the one that publishes dS.
    ///
    /// Operand and output contracts are the extraction's: packed-bf16
    /// `[B*H, T, HD]` staging panels with Q pre-scaled by
    /// `softmax_scale * log2(e)` and dY staged raw, `logsumexp[B*T, H]` and
    /// `dot[B*T, H]` fp32 read-only, fp32 `dq[B*T, H*HD]` written.
    #[kernel]
    #[launch_bounds(160, 1)]
    pub unsafe fn flash_backward_q(
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        logsumexp: &[f32],
        dot: &[f32],
        sequence_length: u32,
        heads: u32,
        batches: u32,
        mut dq: DisjointSlice<f32>,
    ) {
        unsafe {
            if thread::blockDim_x() as usize != FLASH_BACKWARD_BLOCK {
                return;
            }
            let shared = backward_q_plan(SharedPlan::attach());
            let tmem = alloc_block(shared.tmem_slot, BACKWARD_TMEM_COLUMNS);
            // Two score segments at columns 0..128, two dP segments at
            // 128..256, the 128-column dQ accumulator at 256.
            let scores = STmem::from_raw(tmem);
            let gradients = STmem::from_raw(tmem + 2 * TILE as u32);
            let accumulator = AccTmem::from_raw(tmem + 4 * TILE as u32);

            let tiles = sequence_length / QUERIES as u32;
            let planes = heads * batches;
            let stats = heads as usize;
            let mut job = BackwardQStream {
                q_tma,
                k_tma,
                v_tma,
                dy_tma,
                t: sequence_length,
                h: heads,
                tiles,
                planes,
                // The issue warp's lane 0, not thread 0: every `mma_*` and
                // every `tma_load` of the item belongs to the fifth warp
                // (issue #94).
                leader: thread::threadIdx_x() == ISSUE_WARP * 32,
                warp_id: warp::warp_id(),
                lane: warp::lane_id(),
                shared,
                scores,
                gradients,
                accumulator,
                lse: GlobalRows::<F32>::from_raw(logsumexp.as_ptr().cast_mut().cast(), stats),
                dot: GlobalRows::<F32>::from_raw(dot.as_ptr().cast_mut().cast(), stats),
                dq: GlobalRows::<F32>::from_slice(&mut dq, heads as usize * HD),
            };
            pipeline::run(&mut job, tiles * planes);
            dealloc_block(tmem, BACKWARD_TMEM_COLUMNS);
        }
    }

    /// [`flash_backward_q`] with a stopwatch on every phase of a key-tile
    /// visit, whose body is [`super::phase_probe`] and whose only caller is
    /// `src/bin/probe.rs`.
    ///
    /// # Safety
    ///
    /// As [`flash_backward_q`], plus `clocks` holding
    /// [`super::phase_probe::COUNTERS`] zeroed `u64` per CTA of the grid.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(160, 1)]
    pub unsafe fn flash_backward_q_probe(
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        logsumexp: &[f32],
        dot: &[f32],
        sequence_length: u32,
        heads: u32,
        batches: u32,
        mut dq: DisjointSlice<f32>,
        mut clocks: DisjointSlice<u64>,
    ) {
        unsafe {
            phase_probe::backward_q(
                q_tma,
                k_tma,
                v_tma,
                dy_tma,
                logsumexp,
                dot,
                sequence_length,
                heads,
                batches,
                &mut dq,
                &mut clocks,
            )
        }
    }

    /// [`flash_backward_kv`] with the same stopwatch.
    ///
    /// # Safety
    ///
    /// As [`flash_backward_kv`], plus `clocks` as in [`flash_backward_q_probe`].
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(160, 1)]
    pub unsafe fn flash_backward_kv_probe(
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        logsumexp: &[f32],
        dot: &[f32],
        sequence_length: u32,
        heads: u32,
        batches: u32,
        mut dk: DisjointSlice<f32>,
        mut dv: DisjointSlice<f32>,
        mut clocks: DisjointSlice<u64>,
    ) {
        unsafe {
            phase_probe::backward_kv(
                q_tma,
                k_tma,
                v_tma,
                dy_tma,
                logsumexp,
                dot,
                sequence_length,
                heads,
                batches,
                &mut dk,
                &mut dv,
                &mut clocks,
            )
        }
    }

    /// The key-parallel backward as a [`pipeline::Job`]: one work item is a
    /// (key block, head, batch), and everything the query-parallel stream does
    /// by row this does by column.
    struct BackwardKvStream {
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        t: u32,
        h: u32,
        tiles: u32,
        planes: u32,
        leader: bool,
        warp_id: u32,
        lane: u32,
        shared: BackwardKv,
        /// Two `[QUERIES, TILE]` transposed score segments.
        scores: STmem,
        /// Two more for `dPᵀ`.
        gradients: STmem,
        dv_acc: AccTmem,
        dk_acc: AccTmem,
        lse: GlobalRows<F32>,
        dot: GlobalRows<F32>,
        dk: GlobalRows<F32>,
        dv: GlobalRows<F32>,
    }

    impl BackwardKvStream {
        #[inline(always)]
        fn buffer(&self, segment: STmem, step: u32) -> STmem {
            if step & 1 == 0 {
                segment
            } else {
                segment.columns_right(TILE as u32)
            }
        }

        /// Stage the query tile of `step` — Q and dY, one barrier for both.
        #[inline(always)]
        unsafe fn load_qdy(&self, step: u32, query_base: u32, plane: i32) {
            unsafe {
                let row = query_base as i32;
                let full = self.shared.qdy_loaded.sem(step);
                let charge = self
                    .shared
                    .q
                    .tile(step)
                    .tma_load(self.q_tma, row, plane, full)
                    + self
                        .shared
                        .dy
                        .tile(step)
                        .tma_load(self.dy_tma, row, plane, full);
                full.expect_tx(charge);
            }
        }

        /// Issue `Sᵀ = K·Qᵀ` and `dPᵀ = V·dYᵀ` for one streamed query tile.
        ///
        /// **The transpose is the MMA's.** `mma_abt` with the resident K as A
        /// and the streamed Q as B produces the key-major band directly, so
        /// there is no register transpose here and no second `FragmentLayout`
        /// wanted for one — the operand-order square covers this site, which is
        /// the question GAPS §1.3 asks of every backward.
        #[inline(always)]
        unsafe fn score_mmas(&self, step: u32) {
            unsafe {
                mma_abt(
                    self.buffer(self.scores, step).raw(),
                    self.shared.k,
                    self.shared.q.tile(step),
                    MMA_SHAPE,
                    false,
                );
                mma_abt(
                    self.buffer(self.gradients, step).raw(),
                    self.shared.v,
                    self.shared.dy.tile(step),
                    MMA_SHAPE,
                    false,
                );
                mma::commit(self.shared.scored.sem(step));
            }
        }
    }

    impl pipeline::Job for BackwardKvStream {
        #[inline(always)]
        unsafe fn init(&self, _item: u32) {
            unsafe {
                self.shared.qdy_loaded.init_all(1);
                self.shared.scored.init_all(1);
                self.shared.accumulated.init_all(1);
                self.shared.kv_loaded.init(1);
            }
        }

        #[inline(always)]
        unsafe fn inval(&self) {
            unsafe {
                self.shared.qdy_loaded.inval_all();
                self.shared.scored.inval_all();
                self.shared.accumulated.inval_all();
                self.shared.kv_loaded.inval();
            }
        }

        #[inline(always)]
        unsafe fn work(&mut self, item: u32) {
            unsafe {
                // ASCENDING key block, which is the same rule as the forward's
                // descending query tile: key block `i` streams every query at or
                // after it, so block 0 is the longest stream and goes first.
                let key_block = item / self.planes;
                let plane = item % self.planes;
                let head = plane % self.h;
                let batch = plane / self.h;
                let key_base = key_block * QUERIES as u32;
                let steps = 2 * (self.tiles - key_block);

                let (leader, warp_id, lane) = (self.leader, self.warp_id, self.lane);
                let band = 32 * warp_id;
                let issuing = warp_id == ISSUE_WARP;

                if leader {
                    tcgen05_fence_after_thread_sync();
                    let full = self.shared.kv_loaded;
                    let second = (key_base + TILE as u32) as i32;
                    let charge = self.shared.k.tma_load_at::<TILE>(
                        self.k_tma,
                        0,
                        key_base as i32,
                        plane as i32,
                        full,
                    ) + self.shared.k.tma_load_at::<TILE>(
                        self.k_tma,
                        TILE,
                        second,
                        plane as i32,
                        full,
                    ) + self.shared.v.tma_load_at::<TILE>(
                        self.v_tma,
                        0,
                        key_base as i32,
                        plane as i32,
                        full,
                    ) + self.shared.v.tma_load_at::<TILE>(
                        self.v_tma,
                        TILE,
                        second,
                        plane as i32,
                        full,
                    );
                    full.expect_tx(charge);
                    let mut stage = 0u32;
                    while stage < BACKWARD_STAGES as u32 && stage < steps {
                        self.load_qdy(stage, key_base + stage * TILE as u32, plane as i32);
                        stage += 1;
                    }
                    full.wait(0);
                    self.shared.qdy_loaded.wait(0);
                    self.score_mmas(0);
                }

                // The two roles, and the one barrier they meet at, exactly as
                // in [`flash_backward_q`] — this kernel's issue warp simply has
                // two gradient MMAs to post rather than one.
                if issuing {
                    let mut step = 0u32;
                    while step < steps {
                        if lane == 0 {
                            let refill = step + BACKWARD_STAGES as u32 - 1;
                            if step > 0 && refill < steps {
                                self.shared.accumulated.wait(step - 1);
                                self.load_qdy(
                                    refill,
                                    key_base + refill * TILE as u32,
                                    plane as i32,
                                );
                            }
                            if step + 1 < steps {
                                self.shared.qdy_loaded.wait(step + 1);
                                self.score_mmas(step + 1);
                            }
                        }
                        tcgen05_fence_before_thread_sync();
                        thread::sync_threads();
                        if lane == 0 {
                            tcgen05_fence_after_thread_sync();
                            let accumulate = step != 0;
                            mma_ab(
                                self.dv_acc.raw(),
                                self.shared.p.tile(step),
                                self.shared.dy.tile(step),
                                MMA_SHAPE,
                                accumulate,
                            );
                            mma_ab(
                                self.dk_acc.raw(),
                                self.shared.ds.tile(step),
                                self.shared.q.tile(step),
                                MMA_SHAPE,
                                accumulate,
                            );
                            mma::commit(self.shared.accumulated.sem(step));
                        }
                        step += 1;
                    }
                } else {
                    let row = batch * self.t + key_base + band;
                    let mut step = 0u32;
                    while step < steps {
                        // The streamed query tile's rows, which are this band's
                        // COLUMNS. Everything below indexes by column where the
                        // query-parallel kernel indexes by row.
                        let query_base = key_base + step * TILE as u32;
                        self.shared.scored.wait(step);
                        if step >= GRADIENT_STAGES as u32 {
                            self.shared.accumulated.wait(step - GRADIENT_STAGES as u32);
                        }

                        let scores = self.buffer(self.scores, step);
                        let gradients = self.buffer(self.gradients, step);
                        // A 128-key band clears the diagonal once the streamed
                        // query tile is wholly after its last key, which is two
                        // 64-row tiles in.
                        let masked = step < 2;
                        let p = self.shared.p.tile(step).chunk_writer();
                        let ds = self.shared.ds.tile(step).chunk_writer();

                        let mut column = 0u32;
                        while column < TILE as u32 {
                            let query = batch * self.t + query_base + column;
                            // The saved statistic, read onto the axis this band
                            // indexes it by — one element per column, down the
                            // head's own column of `[rows, heads]` (ferro #178).
                            let mut lse2: Cols = load_col_vec(self.lse, query, head, lane);
                            lse2.scale_assign(LOG2E);
                            let dots: Cols = load_col_vec(self.dot, query, head, lane);

                            let mut probability: ScoreChunk = scores.tile(band, column);
                            if masked {
                                // The transposed band's mask: rows are keys and
                                // columns are queries, so this is the other
                                // diagonal and `key - query` is negative on
                                // every step but the first.
                                probability.make_causal_t_at(
                                    lane,
                                    key_base + band,
                                    query_base + column,
                                    MASKED_SCORE,
                                );
                            }
                            probability.sub_col_assign(lse2);
                            probability.unary_map_assign::<Exp2Hw>();
                            store_tile(p, band, column, lane, probability);

                            let mut dpt: ScoreChunk = gradients.tile(band, column);
                            dpt.sub_col_assign(dots);
                            probability.mul_assign(dpt);
                            // `dK += dSᵀ·Q` runs against the *pre-scaled* Q, so
                            // the factor here is `ln2` and not `scale`:
                            // `ln2·scale·log2e = scale`.
                            probability.scale_assign(LN2);
                            store_tile(ds, band, column, lane, probability);
                            column += SCORE_CHUNK as u32;
                        }

                        fence_proxy_async_shared_cta();
                        tcgen05_fence_before_thread_sync();
                        thread::sync_threads();
                        step += 1;
                    }

                    self.shared.accumulated.wait(steps - 1);
                    let dv: OutBand = self.dv_acc.tile_x8(band, 0);
                    store_rows(self.dv, row, head * HD as u32, lane, dv);
                    let dk: OutBand = self.dk_acc.tile_x8(band, 0);
                    store_rows(self.dk, row, head * HD as u32, lane, dk);
                }
            }
        }
    }

    /// tcgen05 causal attention backward, key-parallel half — one kernel where
    /// there were two, the mirror of [`flash_backward_q`].
    ///
    /// Launch with `host::flash_backward_kv_config`. One work item is a
    /// (key block, head, batch). A CTA owns `QUERIES = 128` key rows, holds
    /// their K and V resident, and streams the `TILE`-query tiles at and after
    /// them: `Sᵀ = K·Qᵀ` and `dPᵀ = V·dYᵀ` fill a double-buffered TMEM segment
    /// pair, the warpgroup forms `Pᵀ` and `dSᵀ = Pᵀ·(dPᵀ − D)·ln2`, and
    /// `dV += Pᵀ·dY` / `dK += dSᵀ·Q` accumulate in TMEM for the whole stream.
    /// Both gradient tiles have one writer, for the reason kernel A's does.
    ///
    /// **Everything kernel A indexes by row this indexes by column**, and that
    /// is the whole of the difference: the band's rows are keys, so the causal
    /// mask is the other diagonal (`make_causal_t_at`) and the per-query LSE
    /// and dot are a `ColVec` (`load_col_vec`) rather than a `RegVec`. Neither
    /// existed in the library before this kernel asked for them
    /// (ferro-kittens #178); what did *not* need adding is a register
    /// transpose, because `mma_abt(K, Q)` produces the key-major band itself.
    ///
    /// The two accumulators are why this kernel takes all 512 tensor-memory
    /// columns where kernel A takes 384 — and why neither can reach two CTAs an
    /// SM at any shared-memory plan.
    #[kernel]
    #[launch_bounds(160, 1)]
    pub unsafe fn flash_backward_kv(
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        logsumexp: &[f32],
        dot: &[f32],
        sequence_length: u32,
        heads: u32,
        batches: u32,
        mut dk: DisjointSlice<f32>,
        mut dv: DisjointSlice<f32>,
    ) {
        unsafe {
            if thread::blockDim_x() as usize != FLASH_BACKWARD_BLOCK {
                return;
            }
            let shared = backward_kv_plan(SharedPlan::attach());
            let tmem = alloc_block(shared.tmem_slot, BACKWARD_TMEM_COLUMNS);
            // Two `Sᵀ` segments at 0..128, two `dPᵀ` at 128..256, then the two
            // 128-column gradient accumulators — all 512 columns.
            let scores = STmem::from_raw(tmem);
            let gradients = STmem::from_raw(tmem + 2 * TILE as u32);
            let dv_acc = AccTmem::from_raw(tmem + 4 * TILE as u32);
            let dk_acc = AccTmem::from_raw(tmem + 4 * TILE as u32 + HD as u32);

            let tiles = sequence_length / QUERIES as u32;
            let planes = heads * batches;
            let stats = heads as usize;
            let mut job = BackwardKvStream {
                q_tma,
                k_tma,
                v_tma,
                dy_tma,
                t: sequence_length,
                h: heads,
                tiles,
                planes,
                // The issue warp's lane 0; see [`flash_backward_q`].
                leader: thread::threadIdx_x() == ISSUE_WARP * 32,
                warp_id: warp::warp_id(),
                lane: warp::lane_id(),
                shared,
                scores,
                gradients,
                dv_acc,
                dk_acc,
                lse: GlobalRows::<F32>::from_raw(logsumexp.as_ptr().cast_mut().cast(), stats),
                dot: GlobalRows::<F32>::from_raw(dot.as_ptr().cast_mut().cast(), stats),
                dk: GlobalRows::<F32>::from_slice(&mut dk, heads as usize * HD),
                dv: GlobalRows::<F32>::from_slice(&mut dv, heads as usize * HD),
            };
            pipeline::run(&mut job, tiles * planes);
            dealloc_block(tmem, BACKWARD_TMEM_COLUMNS);
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

        /// Multiply one 64-column half of this warp's rows of the O segment by
        /// `factor`, in place — the correction's whole effect on the output.
        ///
        /// The store is left outstanding; the caller retires both halves with
        /// one `store_wait`.
        #[inline(always)]
        unsafe fn rescale_half(&self, band: u32, column: u32, factor: Rows) {
            unsafe {
                let half: OutHalf = self.accumulator.tile_x8(band, column);
                self.accumulator
                    .store_tile(band, column, half.mul_row(factor));
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
                let mut corrections = 0u32;

                let mut key_tile = 0u32;
                while key_tile < key_tiles {
                    if leader {
                        // Refill before issuing, not after. A K/V stage is
                        // free once the output MMA that read its V completed,
                        // so the refill is `FORWARD_STAGES - 1` tiles ahead —
                        // and at the ring's floor of two stages that is
                        // `key_tile + 1`, exactly the tile the score MMA below
                        // is about to wait for. Issuing first deadlocks the
                        // leader against a load it has not made yet, which is
                        // invisible at three stages and immediate at two.
                        let refill = key_tile + FORWARD_STAGES as u32 - 1;
                        if key_tile > 0 && refill < key_tiles {
                            self.shared.accumulated.wait(key_tile - 1);
                            self.load_kv(refill, plane as i32);
                        }
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
                    // climbs more than `CORRECTION_THRESHOLD` above it; the
                    // vote is collective because a rescale rewrites tensor
                    // memory four warps share. Tile 0 always trips it —
                    // `m_ref` is still the sentinel.
                    let exceeded = row_max.any_exceeds(m_ref, CORRECTION_THRESHOLD);
                    let rescale = block_reduce::<Max, FORWARD_WARPS>(
                        self.shared.votes,
                        warp_reduce::<Max>(if exceeded { 1.0 } else { 0.0 }),
                    ) != 0.0;
                    if rescale {
                        // `online_rescale`'s two scalar halves, open-coded,
                        // because its third argument is an `&mut RegTile` and
                        // this kernel must not hand a 128-register band an
                        // address: **the rescale happens in the O segment**.
                        // The scheme this replaces restarted the segment and
                        // carried the running output in registers instead, and
                        // that band — address-taken, live from the first key
                        // tile to the epilogue — was the 560 B frame and 405
                        // `ld/st.local` ferro #180 left behind, and 1.212 ms at
                        // the profile shape against this form's 0.832.
                        // `tcgen05.st` was absent when the scheme was designed
                        // and is `TmemTile::store_tile` now.
                        let next = m_ref.max(row_max);
                        let factor = m_ref.sub(next).exp2();
                        m_ref = next;
                        running_sum.mul_assign(factor);
                        // Tile 0 has no segment to rescale — its MMA is the one
                        // that writes the segment rather than accumulating into
                        // it, and the sum it scales is still zero.
                        if key_tile > 0 {
                            self.shared.accumulated.wait(key_tile - 1);
                            self.rescale_half(band, 0, factor);
                            self.rescale_half(band, TILE as u32, factor);
                            store_wait();
                            corrections += 1;
                        }
                    }

                    // This slot was last read by the output MMA
                    // `PROBABILITY_STAGES` tiles back.
                    if key_tile >= PROBABILITY_STAGES as u32 {
                        self.shared
                            .accumulated
                            .wait(key_tile - PROBABILITY_STAGES as u32);
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
                        // The SFU `exp2`, not the software polynomial the
                        // extracted kernels used. The polynomial was an
                        // FA4-shaped SFU-offload choice made when the softmax
                        // shared an SM with two other warp roles; this kernel
                        // is four warps and is not tensor-core bound, so the
                        // offload has nothing to offload to. It is also the
                        // *more* accurate of the two here — `ex2.approx.f32`
                        // is ~2 ULP against the polynomial's measured 7.5e-5
                        // relative — and `log2` in the LSE epilogue stays
                        // software, where it is once per row rather than 64
                        // times per thread per key tile.
                        chunk.unary_map_assign::<Exp2Hw>();
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
                            key_tile != 0,
                        );
                        mma::commit(self.shared.accumulated.sem(key_tile));
                    }
                    key_tile += 1;
                }

                self.shared.accumulated.wait(key_tiles - 1);
                // The drain, at the end of the stream: the segment holds the
                // whole sum, because every correction was applied to it rather
                // than to a register copy. A 64-column half at a time — see
                // `OutHalf` — and the normalization rides the way out, so the
                // segment is read once and nothing here has an address.
                let row = batch * self.t + query_base + band;
                let inverse = running_sum.recip();
                let low: OutHalf = self.accumulator.tile_x8(band, 0);
                store_rows(self.y, row, head * HD as u32, lane, low.mul_row(inverse));
                let high: OutHalf = self.accumulator.tile_x8(band, TILE as u32);
                store_rows(
                    self.y,
                    row,
                    head * HD as u32 + TILE as u32,
                    lane,
                    high.mul_row(inverse),
                );
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
    /// masks against the *pair* of block origins, `TmemTile::tile_x8` and
    /// `store_tile` are the rescale's round trip through the O segment, and
    /// `block_reduce` is the correction vote three kernels used to open-code as
    /// a votes array and a barrier phase.
    ///
    /// Operand and output contracts are the extraction's: packed-bf16
    /// `[B*H, T, HD]` staging panels with Q pre-scaled by
    /// `softmax_scale * log2(e)`, fp32 `y[B*T, H*HD]`, fp32 `logsumexp[B*T, H]`
    /// in natural-log units, and one `correction_counts` word per work item
    /// (`plane * tiles + query_tile`) counting mid-stream segment rescales.
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
