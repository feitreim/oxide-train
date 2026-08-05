//! # The tcgen05 GEMMs — `C = A·Bᵀ` and `dW += Aᵀ·B` on the `cta_group::2` path
//!
//! bf16 operands, an fp32 TMEM accumulator, and a `C` that is either packed
//! bf16 or fp32, either overwritten or accumulated into. The training variants
//! share one compute pipeline: the operand layout is a runtime flag on the
//! *walk* and the epilogue is a type parameter, so neither is a second MMA
//! chain.
//!
//! A cluster of **four** CTAs runs **two** `M256_N128` UMMAs — one per
//! `cta_group::2` pair — on adjacent `[2·BLOCK_M, BLOCK_N]` tiles of the same
//! tile-row. Within a pair each rank stages its own [`BLOCK_M`] rows of `A` and
//! its own [`HALF_N`] columns of `B` at the same shared offsets, the
//! instruction reads both ranks' shared memory, and each rank drains its own
//! `[BLOCK_M, BLOCK_N]` band. The pair's even rank owns its MMA, its
//! accumulator and its stage barriers, and every barrier a peer arrives at is
//! that leader's, so the launch synchronizes at `barrier.cluster` scope —
//! twice, at [`Tile::arm`] and [`Tile::retire`], and nowhere else.
//!
//! The grid is **persistent** — [`MAX_CLUSTERS`] clusters, two CTAs an SM — and
//! one work item is one `[2·BLOCK_M, PAIRS·BLOCK_N]` **region**, which the two
//! pairs split a tile each. Each of the three roles walks the static strided
//! schedule for itself ([`Walk`]) and [`pipeline::grouped`] maps an item to a
//! region in blocks of [`GROUP`] tile-rows. K is [`STAGES`] deep over a pair of
//! tile rings; the TMA fills `load` and the MMA's own commit releases `free`,
//! so what proves an operand has been read is the accumulator instruction and
//! not a thread.
//!
//! ## `A` is fetched once and delivered twice
//!
//! Adjacent tiles of one tile-row need **the same `A`**, which is the entire
//! reason the cluster is four CTAs rather than two. Each pair leader issues one
//! `A` half as a replicating TMA multicast to the two CTAs that owe the MMA
//! those rows — one in each pair — so per region `A` is fetched [`PAIRS`] times
//! and `B` [`CLUSTER_RANKS`] times:
//!
//! | | shared-memory fill, bytes/FLOP |
//! |---|---:|
//! | `[256, 256]`, one `M256_N256` | 0.00781 |
//! | `[256, 128]` alone | 0.01172 (×1.50) |
//! | **`[256, 128]` × 2, multicast `A`** | **0.00781** |
//!
//! That is the point of the whole structure. oxide-train#80 measured the narrow
//! tile fixing wave quantization exactly as predicted — `bwd down dW` 64.9% →
//! 86.5%, cuBLASLt's own figure — and *losing thirteen of fourteen rows anyway*,
//! by 5–28 points, because the ×1.50 was not affordable: the fill path
//! saturates in the high teens of TB/s and the kernel already ran at 10–15. The
//! multicast keeps the narrow tile's quantization at the wide tile's bytes.
//!
//! What makes it expressible is one fact about the instruction, measured by
//! `src/bin/mcast_probe.rs` because ferro-kittens' own documentation records it
//! as unestablished: a replicating `cp.async.bulk.tensor` completes on the
//! barrier at the given **offset**, in the CTA of each destination's own
//! `cta_group::2` pair picked by the supplied address's rank parity. Handing it
//! an *even* rank therefore charges every destination's pair leader — so one
//! instruction feeds and accounts for both pairs, and this kernel keeps the
//! single-barrier-per-pair structure of the two-CTA one exactly, with no
//! peer-progress signal anywhere.
//!
//! ## There is no item boundary
//!
//! This kernel used to run on [`pipeline::run`], which re-arms every mbarrier
//! on one thread and takes two `cluster_sync`s per item. oxide-train#80's
//! forensics priced that at **~6 300 SM ticks an item** — 3.6 µs, K-independent,
//! and 10% of a K = 3072 item — against a cuBLASLt whose own depth fit has *no*
//! per-item term at all. So the barriers are armed **once per launch**, the
//! operand ring is indexed by the **global** K block rather than the item's own,
//! and the item stream is two semaphore rings deep ([`ITEMS`]): `full` says an
//! item's MMA chain is complete, `empty` says an accumulator slot has been read.
//! Nothing in the loop is cluster-wide, and the producer crosses an item
//! boundary with [`STAGES`] blocks still in flight.
//!
//! The same forensics is why nothing else changed: the steady state measured
//! **1001 ticks a K block — 2182 TFLOP/s aggregate against the B200's 2250
//! dense bf16, and 8.5% ahead of cuBLASLt's own depth slope** — and the
//! pipeline fill measured 160 ticks. Depth, tile and feed were never the
//! deficit.
//!
//! ## What replaced what
//!
//! This is a rewrite of the 2026-07-26 extraction-point kernel, not a port of
//! it. That one launched **one cluster per output tile** with a hand-unrolled
//! four-stage pipeline over eight `SharedArray` statics, six warps (a TMA warp,
//! an MMA warp and four epilogue warps), and an inert two-slot TMEM ping-pong
//! left over from a CLC schedule it had removed. All of that is gone: the ring
//! is a [`SharedTileRing`] over a [`SharedPlan`], the schedule is [`Walk`] and
//! the two item rings, and the epilogue warps are the CTA's four band warps —
//! joined, since oxide-train#80 remedy 2, by a producer warp and an MMA warp so
//! the band warps never block the schedule.
//!
//! ## The tile is half as wide, and the allocation is not
//!
//! The tile was `[256, 256]`, and every attempt to overlap the drain with the
//! next item's MMA died on the same fact: a second accumulator needs 512 more
//! tensor-memory columns and an SM has 512. So this splits the allocation it
//! already owns. [`ACCUM_COLS`] is still 256 columns and residency is still the
//! two CTAs an SM that pins (`512 / 256`), but the tile is [`BLOCK_N`] = 128
//! wide and the 256 columns are [`SLOTS`] of them: item `i` multiplies into one
//! while item `i − 1`'s drain still reads the other, and the release moves from
//! `empty(i + 1)` to `empty(i + 2)`. The handoff #80's probe measured at
//! **8 310 ticks — 14% of a K = 3072 item, and after #84 the only thing in an
//! item besides the multiply** — comes off the critical path entirely, because
//! what the next item waits for is no longer this item's drain but the one
//! before it, which has had a whole item to finish.
//!
//! Halving the tile is also what answers the *other* residual term. #80 priced
//! wave quantization at `gate_up fwd` 86.5% against cuBLASLt's 94.4% and
//! `bwd down dW` **64.9% against 86.4%** — and cuBLASLt reaches those numbers by
//! running a tile of exactly half our area (`tile=23`, `waves=46.70` against our
//! 23.35). At `[256, 128]` both rows land on cuBLASLt's own figures: 1536 tiles
//! over 148 clusters is 94.3%, and 384 is 86.5%.
//!
//! It is not free. A tile half as wide reads the same `A` for half the output,
//! so this kernel moves **1.5× the operand bytes an output element** the
//! `[256, 256]` tile did. That is the arithmetic-intensity trade ferro #87's
//! sweep charged `[256, 128]` for — measured there at `STAGES = 2` and 4
//! CTAs/SM, neither of which this is — and the reason to believe it is payable
//! is that cuBLASLt pays it on this hardware at 1850–1970 TFLOP/s.
//!
//! ## The epilogue is the kernel
//!
//! ferro-kittens #114 priced this kernel's phases against each other and found
//! the drain **exposed, not hidden** (epilogue with the MMA over without =
//! 1.01) and worth more than the entire remaining distance to cuBLASLt.
//! Nothing outside the cluster will cover it: a second CTA's MMA could absorb
//! a draining neighbour's tensor core (a lone chain sustains ~0.90 of two,
//! ferro #188) but the in-phase schedule is an attractor — a launch-time
//! stagger decays within an item (oxide-train#80's probe) — and the 1 CTA/SM
//! double-buffer loses more to its solo feed than the drain costs (ferro
//! #188's floor). So the cover comes from **inside** the item stream: the
//! epilogue is deferred one item, each drain releases its accumulator slot
//! through [`Release`] the moment its last `tcgen05.ld` retires, and — since
//! the slot it releases is not the one the next item needs — the entire store
//! tail runs beside the next MMA rather than a fraction of it.
//!
//! What #80's own probe added is *where the drain's time goes*: it is the
//! **wait per `.x8` issue**, not the store issue, so the release is worth what
//! the batched lift ([`TmemTile::tile_x8_batched`]) leaves behind it. See
//! [`Wide`].
//!
//! Within a pass the drain keeps the shape #116/#117 measured: TMEM → registers
//! (`tcgen05.ld.16x256b.x8`) → `stmatrix.m8n8.x4` into a per-warp
//! `[32, STAGE_N]` shared tile → 16-byte stores, two passes a band. Not a TMA
//! store: #123 measured that route *losing* by 1.0–1.7% at warp scope on this
//! tile.
//!
//! An **accumulating bf16** `C` leaves the same way through
//! [`accumulate_shared_rows`], which is that store with one `ld.global.v4` in
//! front of it. The kernel this replaced read-modify-wrote `C` a 32-bit word at
//! a time and paid more than the whole rest of the kernel for it: 1113.8
//! TFLOP/s storing against 536.2 accumulating, at 4096³ on a B200.
//!
//! An **fp32** `C` cannot take the `stmatrix` route at all — it moves b16
//! matrices, so nothing can fill an fp32 staging tile that way — and the
//! *store* mode drains through [`store_rows`] a value at a time instead
//! (GAPS.md §2.6 carries the staged-fp32 gap).
//!
//! The **accumulating fp32** `C` — the weight-gradient fold that oxide-train#80
//! measured at +25–29% over its own store at K = 6144 — no longer reads `C` at
//! all. Its drain scatters each fp32 band into a `[16, STAGE_N]` staging tile
//! ([`scatter_tile`], byte-for-byte the bf16 drain's `[32, STAGE_N]` buffer)
//! and hands it to the copy engine as a *reduction store*
//! (`cp.reduce.async.bulk.tensor.add`, ferro #42): the engine adds the tile
//! into `C` in fp32, so the fold's `ld.global` disappears and the sum stays at
//! the accumulator's own precision. One CTA owns each output tile and each
//! element is reduced exactly once per launch, so element order — and with it
//! SPEC decision #20's fp32 weight-gradient accumulation — is unchanged.

use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster, cluster_launch, kernel, thread, warp};
use cuda_host::cuda_module;

use kittens::epilogue::{StoreRing, Warp};
use kittens::global::{GlobalRows, accumulate_shared_rows, store_rows, store_shared_rows};
use kittens::ldst::{scatter_tile, store_tile_x4};
use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline;
use kittens::plan::SharedPlan;
use kittens::shared::{Bf16, F32, SharedTile, SharedTileRing, Swizzle128B, publish_to_async_proxy};
use kittens::sync::{ClusterSemaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};
use kittens::{BaseLdtm, RegTile, lane, warp_id};

/// Rows of the pair tile one rank stages and drains.
pub const BLOCK_M: usize = 128;
/// Columns of one output tile — **half** the widest `cta_group::2` MMA shape,
/// so one tensor-memory allocation holds [`SLOTS`] of them.
pub const BLOCK_N: usize = 128;
/// Columns of `B` one rank stages: the tile's columns, split down the middle.
const HALF_N: usize = BLOCK_N / 2;
/// Accumulator segments one allocation carries, and the depth of the ping-pong
/// the item stream runs over them.
const SLOTS: u32 = 2;
/// fp32 tensor-memory columns a cluster allocates — unchanged from the
/// `M256_N256` kernel this replaces, which is the whole point: the overlap is
/// bought by *splitting* an allocation nothing else could afford to double.
const ACCUM_COLS: u32 = SLOTS * BLOCK_N as u32;
/// One 128-byte swizzle atom of bf16, and the only width
/// [`SharedTile::k_walk`] accepts — so `BLOCK_K` and [`STAGES`] are a
/// factorization of the shared budget rather than two free axes.
pub const BLOCK_K: usize = 64;
/// K=16 chunks one stage chains, both layouts alike.
const CHUNKS: usize = BLOCK_K / 16;
/// Ring depth. 3 → 2 is −11.8% / −7.3% at unchanged residency (ferro #87), and
/// 4 is what a [`HALF_N`]-wide `B` stage leaves room for: halving the tile
/// halved the `B` ring's bytes, and this kernel reads 1.5× the operand bytes an
/// output element that the `[256, 256]` tile did, so the freed 24 KiB go back
/// into depth rather than to the plan's slack.
const STAGES: usize = 4;
/// One warp per 32 accumulator rows — the band warps, every one of which
/// drains — plus the producer warp and the MMA warp, which never do.
pub const THREADS: u32 = (BLOCK_M / 32) as u32 * 32 + 64;
/// Warps that own a 32-row band of the accumulator.
const DRAIN_WARPS: usize = BLOCK_M / 32;
/// The warp whose lane 0 issues the item's TMA loads.
const PRODUCER: u32 = DRAIN_WARPS as u32;
/// The warp whose lane 0 (leader rank only) waits the accumulator free and
/// issues the MMA chain. A warp of its own, not a diverged lane of the
/// producer's: both roles are single-lane loops spinning on barriers, and the
/// first cut of this design ran them as two lanes of one warp — every
/// model_shapes row lost 5–20%, because divergent spin loops share the warp's
/// issue and the MMA chain stalled behind the producer's polling. The two
/// kernels this repo has had before (the extraction-point kernel and ferro's
/// `gemm_ws`) both kept these roles on warps of their own; now this one does
/// too.
const ISSUER: u32 = PRODUCER + 1;
/// The staged drain's band: the narrowest bf16 tile `Swizzle128B` admits *and*
/// the widest four of fit beside the operand rings.
const STAGE_N: usize = 64;
/// CTAs that share one accumulator and one barrier set: the `cta_group::2`
/// pair, which is a property of the instruction and not of the launch.
const PAIR_RANKS: u32 = 2;
/// Pairs in one cluster, and so the output tiles a cluster works at once.
const PAIRS: u32 = 2;
/// CTAs of the cluster. Must agree with `#[cluster_launch]`, which spells it
/// again because an attribute cannot read a constant.
const CLUSTER_RANKS: u32 = PAIRS * PAIR_RANKS;

/// This CTA's pair, and its rank inside it. The `cta_group::2` pairing is the
/// hardware's — ranks `2p` and `2p + 1` — so both fall out of one bit.
#[inline(always)]
const fn pair_of(rank: u32) -> u32 {
    rank / PAIR_RANKS
}
#[inline(always)]
const fn rank_in_pair(rank: u32) -> u32 {
    rank % PAIR_RANKS
}
/// The even rank of `pair`: the CTA that owns its MMA, its accumulator and its
/// barriers.
#[inline(always)]
const fn leader_of(pair: u32) -> u32 {
    pair * PAIR_RANKS
}
/// Every CTA of the cluster: who a `free` release has to reach, because a
/// stage's `A` is written by one pair's producer into *both* pairs' rings.
const CLUSTER_MASK: u16 = ((1u32 << CLUSTER_RANKS) - 1) as u16;

/// The `cta_group::2` multicast mask for `pair` — the two CTAs a `tcgen05`
/// commit has to reach, which is no longer the whole cluster.
#[inline(always)]
const fn pair_mask(pair: u32) -> u16 {
    (((1u32 << PAIR_RANKS) - 1) << leader_of(pair)) as u16
}

/// The TMA multicast mask for the `A` half that `pair`'s leader fetches: one
/// bit per pair, each at the rank that owes the MMA those rows.
///
/// The two masks tile the cluster — `0b0101` and `0b1010` — which is the whole
/// arithmetic-intensity claim in one line: between them the cluster's four CTAs
/// are filled with `A` by [`PAIRS`] fetches instead of [`CLUSTER_RANKS`].
#[inline(always)]
const fn a_mask(pair: u32) -> u16 {
    let mut mask = 0u32;
    let mut p = 0u32;
    while p < PAIRS {
        mask |= 1 << (leader_of(p) + pair);
        p += 1;
    }
    mask as u16
}

const _: () = {
    assert!(PAIRS == 2 && PAIR_RANKS == 2 && CLUSTER_RANKS == 4);
    // Spelled out once against the shifts above, since every barrier in the
    // kernel is addressed through them.
    assert!(pair_mask(0) == 0b0011 && pair_mask(1) == 0b1100);
    assert!(a_mask(0) == 0b0101 && a_mask(1) == 0b1010);
    assert!(leader_of(0) == 0 && leader_of(1) == 2);
};

/// A K-major `[BLOCK_M, BLOCK_K]` stage of `A` — this rank's rows of the pair.
type AStage = SharedTile<Bf16, BLOCK_M, BLOCK_K, Swizzle128B>;
/// A K-major `[HALF_N, BLOCK_K]` stage of `B` — this rank's columns of the
/// tile. Half the rows of an [`AStage`], because an `M256_N128` reads `M / 2`
/// rows of `A` and `N / 2` columns of `B` from each rank of the pair.
type BStage = SharedTile<Bf16, HALF_N, BLOCK_K, Swizzle128B>;
/// The same bytes read MN-major: K along the rows. A 128-byte swizzle caps a
/// TMA box at 128 bytes, so a rank's 128 MN values of `A` arrive as two boxes
/// while its 64 of `B` arrive as one.
type MnAStage = SharedTile<Bf16, BLOCK_K, BLOCK_M, Swizzle128B>;
type MnBStage = SharedTile<Bf16, BLOCK_K, HALF_N, Swizzle128B>;
type ARing = SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing = SharedTileRing<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
type StageTile = SharedTile<Bf16, 32, STAGE_N, Swizzle128B>;
type StageRun = SharedTileRing<Bf16, 32, STAGE_N, Swizzle128B, DRAIN_WARPS>;
type Band = RegTile<32, STAGE_N, BaseLdtm>;
/// `.x8` issues one [`Band`] takes: two 16-row blocks of one 64-column group.
const BAND_ISSUES: usize = (32 / 16) * (STAGE_N / 64);
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;

const _: () = assert!(
    AStage::BYTES == MnAStage::BYTES && BStage::BYTES == MnBStage::BYTES,
    "one ring apiece serves both operand walks"
);

/// SMs on a B200, and the CTAs of one an [`ACCUM_COLS`] accumulator admits:
/// `512 / ACCUM_COLS` tensor-memory columns, which binds before shared memory
/// does.
const SMS: u32 = 148;
const CTAS_PER_SM: u32 = 512 / ACCUM_COLS;
pub const MAX_CLUSTERS: u32 = SMS * CTAS_PER_SM / CLUSTER_RANKS;

/// [`pipeline::grouped`]'s width, swept at this tile shape by ferro-kittens
/// #89 — a tile change is a reason to re-run that sweep.
pub const GROUP: u32 = 8;

/// Items the accumulator handoff holds at once.
///
/// The release moved from `empty(i + 1)` to `empty(i + 2)` — a drain hands back
/// the columns *its own* item used, and the next item is already writing the
/// other slot — so three indices are live at once (`i` waiting, `i + 1`
/// running, `i + 2` released) and the ring is the next power of two up, which
/// keeps the modulo a mask. Four is a ceiling rather than a depth: the issuer
/// cannot run more than [`SLOTS`] items ahead of the epilogue, because item
/// `i + 2` waits on item `i`'s drain.
const ITEMS: usize = 4;

const _: () = assert!(
    ITEMS as u32 >= SLOTS + 1,
    "the released index must not alias the one still being waited on"
);

struct Shared {
    a_ring: ARing,
    b_ring: BRing,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    full: SemaphoreRing<ITEMS>,
    empty: SemaphoreRing<ITEMS>,
    tmem_slot: *mut u32,
    plan: SharedPlan,
}

#[inline(always)]
const fn shared(at: SharedPlan) -> Shared {
    let (a_ring, at) = at.tile_ring::<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>();
    let (b_ring, at) = at.tile_ring::<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>();
    let (load, at) = at.semaphores::<STAGES>();
    let (free, at) = at.semaphores::<STAGES>();
    let (full, at) = at.semaphores::<ITEMS>();
    let (empty, at) = at.semaphores::<ITEMS>();
    let (tmem_slot, at) = at.tmem_slot();
    Shared {
        a_ring,
        b_ring,
        load,
        free,
        full,
        empty,
        tmem_slot,
        plan: at,
    }
}

/// One `[32, STAGE_N]` staging tile per warp, past the operand plan.
#[inline(always)]
const fn staged(at: SharedPlan) -> (StageRun, SharedPlan) {
    at.tile_ring::<Bf16, 32, STAGE_N, Swizzle128B, DRAIN_WARPS>()
}

/// Dynamic shared memory both entry points declare. It has to stay at or under
/// the 116 736 B an SM's 233 472 leaves a CTA at [`CTAS_PER_SM`].
///
/// The fp32 epilogue never touches the staging run and declares it anyway,
/// which costs nothing: tensor memory already pins residency at two, so those
/// 16 424 B buy the bf16 drain its shape and take nothing from the other.
pub const SHARED_BYTES: usize = staged(shared(SharedPlan::sizing()).plan).1.bytes();

const _: () = {
    assert!(THREADS == 192 && MAX_CLUSTERS == 74);
    // The item rings cost the plan nothing: their sixty-four bytes land in the
    // 128-byte alignment padding in front of the staging tiles.
    assert!(SHARED_BYTES == 114_944 && SHARED_BYTES <= 116_736);
    assert!(
        BLOCK_N == 2 * STAGE_N,
        "the staged drains spell their two passes out to hoist the loads ahead of the stores"
    );
};

/// The moment a drain has read everything it will read from tensor memory.
///
/// The deferred epilogue's whole point (oxide-train#80 remedy 2): the next
/// item's MMA needs the accumulator's *columns*, not its values in `C`, so a
/// drain that has retired its last `tcgen05.ld` hands the columns back through
/// the leader's `empty` barrier for the *next* item and issues the stores it
/// still owes behind the release. Between the fence and the arrival every lane
/// of the warp synchronizes, so one lane's arrival covers thirty-two lanes'
/// loads.
#[derive(Clone, Copy)]
struct Release {
    sem: ClusterSemaphore,
}

impl Release {
    /// # Safety
    ///
    /// Every lane of the warp calls this together, with the warp's last
    /// `tcgen05.ld` of the accumulator already waited out.
    #[inline(always)]
    unsafe fn now(self) {
        unsafe {
            tcgen05_fence_before_thread_sync();
            warp::sync_mask(u32::MAX);
            if lane() == 0 {
                self.sem.arrive();
            }
        }
    }
}

/// Where a band of the accumulator goes, and how.
///
/// The one thing that differs between the two entry points. Everything above it
/// — the ring, the walks, the barriers, the schedule — is shared, so a `C` in a
/// new element or under a new fold is an impl here and nothing else.
///
/// Every impl owes `release` the same discipline: it holds as many bands live
/// as the register budgets in `main.rs` admit — four half-bands where a whole
/// one used to be — and calls [`Release::now`] the instant its last load has
/// its registers, so that *half* the store work, and every store's completion,
/// runs beside the next item's MMA rather than in front of it. The fraction is
/// the whole design: held registers over the accumulator's 256 a lane.
trait Drain: Copy {
    /// Push this warp's whole `[32, BLOCK_N]` band out to `C` at
    /// `(row, column)`, releasing the accumulator on the way.
    ///
    /// # Safety
    ///
    /// - Every lane of the warp calls this together, with the accumulator
    ///   complete and fenced and nothing that will overwrite it in flight
    ///   until `release` is arrived at.
    /// - The band's rectangle lies inside `C`.
    unsafe fn drain(
        self,
        accumulator: Accumulator,
        stage: StageTile,
        row: u32,
        column: u32,
        release: Release,
    );
}

/// Packed-bf16 `C`, through the staging tile: `stmatrix.m8n8.x4` in, 16-byte
/// accesses out, [`STAGE_N`] columns a pass.
///
/// The write-after-read each pass owes itself is at *warp* scope because the
/// tile is this warp's alone — `bar.warp.sync` orders memory among the lanes it
/// synchronizes, so the next pass's `stmatrix` cannot overtake a lane still
/// reading this one.
#[derive(Clone, Copy)]
struct Packed {
    c: GlobalRows<Bf16>,
    fold: bool,
}

impl Packed {
    /// One `[32, STAGE_N]` pass: `stmatrix` the band in, copy or fold it out,
    /// and the `bar.warp.sync` that keeps the next pass's `stmatrix` off a
    /// lane still reading this one.
    #[inline(always)]
    unsafe fn pass(self, stage: StageTile, band: Band, row: u32, column: u32) {
        unsafe {
            let lane = lane();
            store_tile_x4(stage.chunk_writer(), 0, 0, lane, band);
            if self.fold {
                accumulate_shared_rows::<Bf16, 32, STAGE_N, Swizzle128B, 32>(
                    self.c, row, column, lane, stage,
                );
            } else {
                store_shared_rows::<Bf16, 32, STAGE_N, Swizzle128B, 32>(
                    self.c, row, column, lane, stage,
                );
            }
            warp::sync_mask(u32::MAX);
        }
    }
}

impl Drain for Packed {
    #[inline(always)]
    unsafe fn drain(
        self,
        accumulator: Accumulator,
        stage: StageTile,
        row: u32,
        column: u32,
        release: Release,
    ) {
        unsafe {
            let band_row = 32 * warp_id();
            let n = STAGE_N as u32;
            // Both issues of a band behind one wait: the staging tile is
            // `[32, STAGE_N]` and nothing slices a wider batch into it, so a
            // pass is two issues and one wait. At [`BLOCK_N`] a band is two
            // passes, so the whole band is read before the release and **every**
            // store this warp owes runs beside the next item's MMA.
            let lift =
                |column| accumulator.tile_x8_batched::<32, STAGE_N, BAND_ISSUES>(band_row, column);
            let first: Band = lift(0);
            let second: Band = lift(n);
            release.now();
            self.pass(stage, first, row, column);
            self.pass(stage, second, row, column + n);
        }
    }
}

/// Overwriting fp32 `C`, straight out of the registers.
///
/// No staging tile, and not for want of asking: [`store_shared_rows`] is
/// generic over its element and would take an fp32 tile, but `stmatrix` moves
/// b16 matrices and nothing bf16-shaped can *fill* one (GAPS.md §2.6). So
/// this is the scattered per-value drain the bf16 path stopped using — one
/// `st.global.v2.f32` per pair of columns against the other's contiguous 16
/// bytes — and it is why the fp32 store rows of the benchmark trail the bf16
/// ones. The *accumulating* fp32 drain is [`Reduce`], which owes nothing to
/// this shape.
///
/// It drains in **one batch behind one wait**, which is what halving the tile
/// bought. Two facts set the shape:
///
/// - The drain is `tcgen05.ld` *latency*, not store issue. #80's probe measured
///   the accumulator handoff at 9 902 ticks against a 9 977-tick drain — the
///   release was buying nothing — and moving store passes behind the release
///   moved the number by 1%. What costs is the **wait per issue**: every
///   `[M, N]` band [`TmemTile::tile_x8`] lifts pays `(M / 16) · (N / 64)` of
///   them serially.
/// - A thread holds 128 fp32 of accumulator and no more, so at
///   `BLOCK_N = 256` no drain could hold a whole band and the fraction of store
///   work behind the release was held registers over 256. At `BLOCK_N = 128` a
///   band *is* 128 a lane, and the fraction is one.
///
/// [`TmemTile::tile_x8_batched`] answers both at once: `[32, BLOCK_N]` is
/// [`kittens::tmem::ISSUE_LIMIT`] issues in flight behind **one** wait, so the
/// band leaves tensor memory in a single latency and the release lands with
/// every store still owed.
#[derive(Clone, Copy)]
struct Wide {
    c: GlobalRows<F32>,
}

/// A warp's whole band, whole across the tile's columns: the widest
/// `tcgen05.ld.16x256b.x8` batch there is (`ISSUE_LIMIT` issues, one wait) and
/// — now that a tile is [`BLOCK_N`] wide — the *entire* band rather than half
/// of it, at the same 128 registers a thread can hold.
type WideBand = RegTile<32, BLOCK_N, BaseLdtm>;
const WIDE_BAND_ISSUES: usize = (32 / 16) * (BLOCK_N / 64);
const _: () = assert!(WIDE_BAND_ISSUES == kittens::tmem::ISSUE_LIMIT);

/// The half-band the reduction drain still lifts one issue at a time: its
/// staging tile is `[16, STAGE_N]` and nothing slices a wider batch back into
/// that shape.
type HalfBand = RegTile<16, STAGE_N, BaseLdtm>;

impl Drain for Wide {
    #[inline(always)]
    unsafe fn drain(
        self,
        accumulator: Accumulator,
        _stage: StageTile,
        row: u32,
        column: u32,
        release: Release,
    ) {
        unsafe {
            let (lane, band_row) = (lane(), 32 * warp_id());
            // **One** wait, and it is the release's cue. A `[32, BLOCK_N]` band
            // is exactly `ISSUE_LIMIT` issues and exactly the 128 registers a
            // thread can hold, so this warp's entire read of tensor memory is
            // one batch and nothing at all stays in front of the release: every
            // store, and every store's completion, runs beside the next item's
            // MMA.
            let band: WideBand =
                accumulator.tile_x8_batched::<32, BLOCK_N, WIDE_BAND_ISSUES>(band_row, 0);
            release.now();
            store_rows(self.c, row, column, lane, band);
        }
    }
}

/// The accumulating drain's staging ring: one fp32 `[16, STAGE_N]` buffer per
/// warp, byte-for-byte this warp's bf16 `[32, STAGE_N]` staging tile — so the
/// fp32 path borrows the bf16 drain's shared plan without costing the plan a
/// byte, and `SHARED_BYTES` does not move.
type ReduceRing = StoreRing<F32, 16, STAGE_N, Swizzle128B, 0, Warp>;

const _: () = assert!(
    ReduceRing::BYTES == StageTile::BYTES,
    "the reduce ring reinterprets one bf16 staging tile exactly"
);

/// Accumulating fp32 `C` that never reads `C`: `dW += Aᵀ·B` with the fold done
/// by the copy engine (`cp.reduce.async.bulk.tensor.add`, ferro #42 —
/// oxide-train#80 remedy 1).
///
/// Each `[16, STAGE_N]` half-band leaves TMEM on the same
/// `tcgen05.ld.16x256b.x8` issue shape as every other drain, is scattered into
/// the warp's fp32 staging tile ([`scatter_tile`]), and is *reduce-stored*:
/// the engine adds it into `C` in fp32, keeping SPEC decision #20's precision
/// with nothing rounded on the way through. The fold that read `C` back at
/// +25–29% over the store is gone.
///
/// Determinism: one CTA owns each output tile, one warp owns each 32-row band,
/// and each element is reduced exactly once per launch — the add order per
/// element is one engine-side `old + new`, the same as the register fold's.
///
/// The ring is depth 1 ([`StoreRing::acquire`] fully drains the engine's read
/// of the previous pass before the next scatter), at warp scope because the
/// buffer is warp-private — the [`kittens::epilogue::Warp`] contract.
#[derive(Clone, Copy)]
struct Reduce {
    c_map: *const TmaDescriptor,
}

impl Reduce {
    /// One `[16, STAGE_N]` pass: acquire the staging tile back from the
    /// engine, scatter the band in, and hand it off as a reduction store.
    #[inline(always)]
    unsafe fn emit(self, ring: &mut ReduceRing, lane: u32, band: HalfBand, row: u32, column: u32) {
        unsafe {
            let staging = ring.acquire();
            scatter_tile(staging.chunk_writer(), 0, 0, lane, band);
            ring.commit_add_2d(self.c_map, column as i32, row as i32);
        }
    }
}

impl Drain for Reduce {
    #[inline(always)]
    unsafe fn drain(
        self,
        accumulator: Accumulator,
        stage: StageTile,
        row: u32,
        column: u32,
        release: Release,
    ) {
        unsafe {
            let (lane, band_row) = (lane(), 32 * warp_id());
            let mut ring = ReduceRing::attach(stage.base());
            // Four passes spelled out rather than looped — the same argument
            // `kittens::tmem`'s batching section makes: a loop-carried band
            // wants a runtime index and lands in local memory. **Three**
            // half-bands live, which at `BLOCK_N = 128` is the whole band but
            // one: this drain carries the ring's cursor and the reduction map
            // beside its bands, and a fourth live band measured 177 registers —
            // past the 168 that twelve warps an SM leave a thread, which is the
            // 2 → 1 CTA cliff by another name. So the release still lands after
            // the last `tcgen05.ld`, with three of the four scatters and all
            // three of their engine round-trips owed.
            let n = STAGE_N as u32;
            let (top, bottom) = (band_row, band_row + 16);
            let b0: HalfBand = accumulator.tile_x8(top, 0);
            let b1: HalfBand = accumulator.tile_x8(top, n);
            let b2: HalfBand = accumulator.tile_x8(bottom, 0);
            self.emit(&mut ring, lane, b0, row, column);
            let b3: HalfBand = accumulator.tile_x8(bottom, n);
            release.now();
            self.emit(&mut ring, lane, b1, row, column + n);
            self.emit(&mut ring, lane, b2, row + 16, column);
            self.emit(&mut ring, lane, b3, row + 16, column + n);
            ring.drain();
        }
    }
}

#[derive(Clone, Copy)]
struct Tile<D: Drain> {
    a_ring: ARing,
    b_ring: BRing,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    accumulator: Accumulator,
    stage: StageTile,
    out: D,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    transposed: bool,
    /// This CTA's place in the cluster, derived once because every barrier,
    /// mask and operand line in the kernel is addressed through it.
    rank: u32,
    pair: u32,
    rank_in_pair: u32,
    leader: u32,
    pair_mask: u16,
    a_mask: u16,
    full: SemaphoreRing<ITEMS>,
    empty: SemaphoreRing<ITEMS>,
}

/// The items one cluster owns, in order: the static strided schedule
/// [`pipeline::run`] used to hand out, now walked by each role for itself.
///
/// Every role has to agree on the sequence without a rendezvous, which it does
/// because the walk is a closed form of the cluster's index and nothing about
/// it is dynamic. That is the whole reason this kernel needs no item mailbox:
/// `gemm_sol`'s `SharedCellRing` exists to publish tiles a CLC query hands out,
/// and these are not handed out.
#[derive(Clone, Copy)]
struct Walk {
    item: u32,
    items: u32,
    stride: u32,
}

impl Walk {
    #[inline(always)]
    fn open(items: u32) -> Self {
        Self {
            item: cluster::cluster_idx(),
            items,
            stride: cluster::num_clusters(),
        }
    }

    /// This cluster's next item, and `None` when it has run out.
    #[inline(always)]
    fn next(&mut self) -> Option<u32> {
        if self.item >= self.items {
            return None;
        }
        let item = self.item;
        self.item += self.stride;
        Some(item)
    }
}

impl<D: Drain> Tile<D> {
    /// Every TMA this cluster will issue, in one uninterrupted loop.
    ///
    /// The ring index is the **global** K block — `sequence * k_blocks + k`,
    /// carried rather than recomputed — so the only thing between the last
    /// stage of an item and the first stage of the next is `free`, and the
    /// producer is [`STAGES`] blocks ahead of the MMA across the item boundary
    /// exactly as it is inside one. That is the boundary oxide-train#80's
    /// forensics priced at ~6 300 SM ticks an item: there is nothing left of it
    /// to pay.
    ///
    /// # Safety
    /// One thread of the CTA, once per launch.
    #[inline(always)]
    unsafe fn produce(&self, mut walk: Walk) {
        unsafe {
            let mut stage_index = 0u32;
            // `A` is fetched by the two pair leaders and replicated; `B` is
            // fetched by everyone. So a leader issues one of each and a peer
            // issues only `B`, and a leader's barrier is charged for exactly
            // twice what a leader issued — the other pair leader's `A` and its
            // own peer's `B` being the same two sizes over again.
            let leads = self.rank == self.leader;
            while let Some(item) = walk.next() {
                let (tile_m, tile_n) = self.locate(item);
                // The `A` half a leader fetches is the one its *pair index*
                // names, not its rank's: rank 0 brings rows `0..BLOCK_M` to
                // ranks {0, 2} and rank 2 brings `BLOCK_M..` to ranks {1, 3},
                // so every CTA ends up holding the half its rank in the pair
                // owes the MMA.
                let a_line =
                    (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.pair) as i32;
                let b_line =
                    (BLOCK_N as u32 * tile_n + HALF_N as u32 * self.rank_in_pair) as i32;
                let mut k = 0u32;
                while k < self.k_blocks {
                    self.free.wait_recycled(stage_index);
                    // Every load of the pair completes on the *pair leader's*
                    // copy of the stage barrier, and only the leader charges it:
                    // `expect_tx` is `.shared::cta`, so a peer could not charge
                    // that barrier even holding its address.
                    //
                    // For the replicated `A` that works because of what the
                    // multicast does with this address, which
                    // `src/mcast_probe.rs` measured: the copy landing in CTA `d`
                    // completes on the barrier at this *offset* in the CTA of
                    // `d`'s own `cta_group::2` pair picked by the address's rank
                    // parity. `at_rank` of an even rank is therefore "each
                    // destination's leader", which is why one instruction feeds
                    // and accounts for both pairs at once.
                    let stage = self.load.sem(stage_index).at_rank(self.leader);
                    let depth = (BLOCK_K as u32 * k) as i32;
                    let (a, b) = (self.a_ring.tile(stage_index), self.b_ring.tile(stage_index));
                    // The map's fast axis dictates the coordinate order: a
                    // K-major operand is one box at `(k, mn)`, an MN-major one a
                    // box per 64-wide subtile at `(mn, k)`. Same transaction
                    // bytes either way, so the charge does not depend on the
                    // branch.
                    let bytes = if self.transposed {
                        let b_bytes = MnBStage::from_raw(b.base())
                            .tma_load_2d_arriving_at(self.b_map, b_line, depth, stage);
                        if leads {
                            MnAStage::from_raw(a.base()).tma_load_2d_multicast_cg2(
                                self.a_map,
                                a_line,
                                depth,
                                stage,
                                self.a_mask,
                            ) + b_bytes
                        } else {
                            b_bytes
                        }
                    } else {
                        let b_bytes = b.tma_load_2d_arriving_at(self.b_map, depth, b_line, stage);
                        if leads {
                            a.tma_load_2d_multicast_cg2(
                                self.a_map,
                                depth,
                                a_line,
                                stage,
                                self.a_mask,
                            ) + b_bytes
                        } else {
                            b_bytes
                        }
                    };
                    if leads {
                        self.load
                            .sem(stage_index)
                            .expect_tx(bytes.across_ranks(PAIR_RANKS));
                    }
                    k += 1;
                    stage_index += 1;
                }
            }
        }
    }

    /// Every MMA this cluster will issue. An item opens by waiting the
    /// accumulator `empty` — which the previous item's drain arrives at the
    /// moment its last `tcgen05.ld` retires — and closes by committing `full`,
    /// which is what lets the band warps read it.
    ///
    /// # Safety
    /// One thread of the leader rank, once per launch.
    #[inline(always)]
    unsafe fn multiply(&self, mut walk: Walk) {
        unsafe {
            let mut sequence = 0u32;
            let mut stage_index = 0u32;
            while walk.next().is_some() {
                self.empty.wait(sequence);
                let target = self.slot(sequence);
                let mut k = 0u32;
                while k < self.k_blocks {
                    self.load.wait(stage_index);
                    let (a, b) = (self.a_ring.tile(stage_index), self.b_ring.tile(stage_index));
                    // A select on the walk, not a duplicated MMA chain: an
                    // `OperandWalk` carries its own transpose bit, so both
                    // layouts issue through one loop.
                    let (a_walk, b_walk) = if self.transposed {
                        (
                            MnAStage::from_raw(a.base()).mn_walk(),
                            MnBStage::from_raw(b.base()).mn_walk(),
                        )
                    } else {
                        (a.k_walk(), b.k_walk())
                    };
                    mma_walk_cg2::<Bf16, CHUNKS>(
                        target.raw(),
                        a_walk,
                        b_walk,
                        MmaShape::M256_N128,
                        k > 0,
                    );
                    // `free` is the one release that stays cluster-wide. A
                    // pair leader multicasts `A` into the *other* pair's ring
                    // at the same stage index, so a slot may not be refilled
                    // until **both** MMAs have read it — which is why this
                    // commit reaches all four CTAs and why `free` is armed for
                    // [`PAIRS`] arrivals rather than one. Getting this wrong is
                    // not a hang: the accounting still balances and one pair
                    // simply overwrites an operand the other is still reading,
                    // which measured as 0.77% of `C` wrong.
                    commit_multicast_cg2(self.free.sem(stage_index), CLUSTER_MASK);
                    k += 1;
                    stage_index += 1;
                }
                commit_multicast_cg2(self.full.sem(sequence), self.pair_mask);
                sequence += 1;
            }
        }
    }

    /// Every epilogue this cluster will run: wait the item's MMA out, lift the
    /// band, hand the accumulator's columns to the *next* item through `empty`,
    /// and store behind the release.
    ///
    /// The pre-arrival is the first item's: nothing has written the accumulator
    /// yet, so its columns are free before the loop starts. It is a plain
    /// arrival rather than a [`Release`] because there is no `tcgen05.ld` in
    /// front of it to fence.
    ///
    /// # Safety
    /// Every lane of every band warp of both ranks, once per launch.
    #[inline(always)]
    unsafe fn epilogue(&self, mut walk: Walk) {
        unsafe {
            if lane() == 0 {
                let mut slot = 0u32;
                while slot < SLOTS {
                    self.empty.sem(slot).at_rank(self.leader).arrive();
                    slot += 1;
                }
            }
            let mut sequence = 0u32;
            while let Some(item) = walk.next() {
                self.full.wait(sequence);
                let (row, column) = self.origin(item);
                let release = Release {
                    sem: self.empty.sem(sequence + SLOTS).at_rank(self.leader),
                };
                self.out
                    .drain(self.slot(sequence), self.stage, row, column, release);
                sequence += 1;
            }
        }
    }

    /// Arm every barrier for the whole launch, publish the write to the async
    /// proxy, and rendezvous — the only cluster-wide synchronization the item
    /// stream has, and it happens once.
    ///
    /// # Safety
    /// Every thread of both ranks, before any role starts.
    #[inline(always)]
    unsafe fn arm(&self) {
        unsafe {
            if thread::threadIdx_x() == 0 {
                self.load.init_all(1);
                // One arrival per pair: a stage is recyclable when both MMAs
                // have consumed it, since both read `A` this producer wrote.
                self.free.init_all(PAIRS);
                self.full.init_all(1);
                // One arrival per band warp per rank *of the pair*: the MMA
                // writes both of its ranks' tensor memory, and the other pair's
                // drains are nothing to it.
                self.empty.init_all(DRAIN_WARPS as u32 * PAIR_RANKS);
                publish_to_async_proxy();
            }
            cluster::cluster_sync();
        }
    }

    /// # Safety
    /// Every thread of both ranks, with every role's loop finished.
    #[inline(always)]
    unsafe fn retire(&self) {
        unsafe {
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();
            if thread::threadIdx_x() == 0 {
                self.load.inval_all();
                self.free.inval_all();
                self.full.inval_all();
                self.empty.inval_all();
            }
        }
    }

    /// The three roles, each walking the same static schedule for itself.
    ///
    /// # Safety
    /// Every thread of both ranks, between [`Tile::arm`] and [`Tile::retire`].
    #[inline(always)]
    unsafe fn run(&self, items: u32) {
        unsafe {
            let walk = Walk::open(items);
            let warp = warp_id();
            if warp == PRODUCER {
                if lane() == 0 {
                    self.produce(walk);
                }
            } else if warp == ISSUER {
                if self.rank == self.leader && lane() == 0 {
                    self.multiply(walk);
                }
            } else {
                self.epilogue(walk);
            }
        }
    }

    /// The `[2·BLOCK_M, BLOCK_N]` tile this CTA's pair covers for `item`.
    ///
    /// An item is a **region** of [`PAIRS`] adjacent tiles of one tile-row, and
    /// the two pairs take one each — which is the whole reason they share `A`.
    /// The walk is therefore over regions, and only the last step differs
    /// between the pairs, so nothing about the schedule is dynamic and no role
    /// has to be told which tile it is on.
    ///
    /// Wave quantization is the narrow tile's, not the region's: a cluster is
    /// [`CLUSTER_RANKS`] CTAs where it used to be [`PAIR_RANKS`], so there are
    /// half as many clusters and each does one region per item — the same
    /// tiles-per-wave the `[256, 128]` tile gave on twice the clusters.
    #[inline(always)]
    fn locate(&self, item: u32) -> (u32, u32) {
        let (tile_m, region) =
            pipeline::grouped(item, self.tiles_m, self.tiles_n / PAIRS, GROUP);
        (tile_m, PAIRS * region + self.pair)
    }

    /// The accumulator segment `sequence` owns: consecutive items alternate, so
    /// an item's MMA runs into one [`BLOCK_N`]-column slot while the previous
    /// item's drain still reads the other.
    #[inline(always)]
    fn slot(&self, sequence: u32) -> Accumulator {
        self.accumulator
            .columns_right(BLOCK_N as u32 * (sequence % SLOTS))
    }

    /// This warp's origin in `C` for `item`: the tile, this rank's half of its
    /// rows, and the 32 rows the warp owns.
    #[inline(always)]
    fn origin(&self, item: u32) -> (u32, u32) {
        let (tile_m, tile_n) = self.locate(item);
        (
            2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank_in_pair + 32 * warp_id(),
            BLOCK_N as u32 * tile_n,
        )
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// # Safety
    /// Both maps must describe live buffers covering the walk the item loop
    /// takes, and the launch must declare [`SHARED_BYTES`]. `alloc_cluster` is
    /// a whole-cluster collective with a `cluster_sync` in it, so this must not
    /// be reached from inside the item loop.
    #[inline(always)]
    unsafe fn attach<D: Drain>(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        tiles_m: u32,
        tiles_n: u32,
        k_blocks: u32,
        transposed: bool,
        out: D,
    ) -> Tile<D> {
        unsafe {
            let shared = shared(SharedPlan::attach());
            let (run, _) = staged(shared.plan);
            let rank = cluster::block_rank();
            Tile {
                a_ring: shared.a_ring,
                b_ring: shared.b_ring,
                load: shared.load,
                free: shared.free,
                a_map,
                b_map,
                accumulator: Accumulator::from_raw(alloc_cluster(shared.tmem_slot, ACCUM_COLS)),
                stage: run.tile(warp_id() % DRAIN_WARPS as u32),
                out,
                tiles_m,
                tiles_n,
                k_blocks,
                transposed,
                rank,
                pair: pair_of(rank),
                rank_in_pair: rank_in_pair(rank),
                leader: leader_of(pair_of(rank)),
                pair_mask: pair_mask(pair_of(rank)),
                a_mask: a_mask(pair_of(rank)),
                full: shared.full,
                empty: shared.empty,
            }
        }
    }

    /// Run every item this cluster owns, from the one barrier arming to the
    /// one teardown — the whole schedule, in the three lines it now is.
    ///
    /// # Safety
    /// Every thread of both ranks, once, with `tile` attached.
    #[inline(always)]
    unsafe fn sweep<D: Drain>(tile: &Tile<D>, items: u32) {
        unsafe {
            tile.arm();
            tile.run(items);
            // `retire`'s `cluster_sync` is also what covers the cluster that got
            // no items at all, which a capped grid can leave having allocated
            // and never looped, and what keeps the accumulator's columns alive
            // until the last drain's reads retire.
            tile.retire();
            dealloc_cluster(tile.accumulator.raw(), ACCUM_COLS);
        }
    }

    /// Packed-bf16 `C`, one `[2·BLOCK_M, BLOCK_N]` output tile per work item.
    ///
    /// `mode` picks the fold — even overwrites, odd accumulates — and
    /// `transposed` picks the operand layout. `0` is the default `C = A·Bᵀ`
    /// over K-major `[M, K]` and `[N, K]` operands; `1` sets the instruction
    /// descriptor's transpose bits so both operands are read MN-major, `A` as
    /// `[K, M]` and `B` as `[K, N]` — the *native* row-major activation and
    /// output-gradient panels of a weight gradient `dW += Aᵀ·B`, with nothing
    /// transposed in global memory.
    ///
    /// `tiles_m` and `tiles_n` count `[2·BLOCK_M, BLOCK_N]` tiles and their
    /// product is the item count. A grouped item map has to know how many tile
    /// *rows* there are, so `tiles_m` is a parameter rather than a quotient.
    ///
    /// # Safety
    /// [`attach`]'s, plus: `c` must hold `n` columns for every row the walk
    /// reaches, and the grid must be a whole number of clusters — see
    /// [`super::host::tcgen05_launch_config`].
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_tcgen05_bf16_optimized(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut c: DisjointSlice<u32>,
        n: i32,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
        mode: u32,
        transposed: u32,
    ) {
        unsafe {
            // Packed pairs: the slice's storage word is two elements wide, so
            // the cursor comes from the address rather than from the slice.
            let out = Packed {
                c: GlobalRows::<Bf16>::from_raw(c.as_mut_ptr() as *mut u8, n as usize),
                fold: mode % 2 == 1,
            };
            let tile = attach(
                a_map,
                b_map,
                tiles_m,
                tiles_n,
                k as u32 / BLOCK_K as u32,
                transposed != 0,
                out,
            );
            sweep(&tile, tiles_m * tiles_n / PAIRS);
        }
    }

    /// [`gemm_tcgen05_bf16_optimized`] with an fp32 `C`, **overwrite only**.
    /// `c_offset` selects one matrix in a stacked allocation without host-side
    /// pointer marshalling. The accumulating fp32 form is
    /// [`gemm_tcgen05_f32_accumulate`], which folds at the copy engine instead
    /// of here.
    ///
    /// # Safety
    /// As [`gemm_tcgen05_bf16_optimized`], with `c_offset..c_offset + m * n`
    /// inside `c`.
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_tcgen05_f32_optimized(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut c: DisjointSlice<f32>,
        c_offset: usize,
        n: i32,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
        transposed: u32,
    ) {
        unsafe {
            let out = Wide {
                c: GlobalRows::<F32>::from_raw(c.as_mut_ptr().add(c_offset) as *mut u8, n as usize),
            };
            let tile = attach(
                a_map,
                b_map,
                tiles_m,
                tiles_n,
                k as u32 / BLOCK_K as u32,
                transposed != 0,
                out,
            );
            sweep(&tile, tiles_m * tiles_n / PAIRS);
        }
    }

    /// `C += A·Bᵀ` (or `dW += Aᵀ·B` under `transposed`) into an fp32 `C`
    /// described by a rank-2 tensor map, with the fold done by the copy engine:
    /// the epilogue *reduce-stores* its tile (`cp.reduce...add`, ferro #42) and
    /// never reads `C` — oxide-train#80's remedy 1 for the +25–29% the
    /// register fold cost over the plain store.
    ///
    /// `C`'s geometry — base, `n`, the row stride of a region inside a stacked
    /// gradient allocation — all live in `c_map`, so the kernel takes no `C`
    /// slice at all; the box is `[16, 32]` fp32 under SWIZZLE_128B, the shape
    /// [`Reduce`]'s staging tile stores through.
    ///
    /// # Safety
    /// As [`gemm_tcgen05_bf16_optimized`] for the operand maps; `c_map` must
    /// describe a live fp32 matrix holding initialized values (a reduction
    /// reads what a store would ignore) covering every tile the item walk
    /// reaches, and nothing else may write it during the launch.
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub unsafe fn gemm_tcgen05_f32_accumulate(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        c_map: *const TmaDescriptor,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
        transposed: u32,
    ) {
        unsafe {
            let tile = attach(
                a_map,
                b_map,
                tiles_m,
                tiles_n,
                k as u32 / BLOCK_K as u32,
                transposed != 0,
                Reduce { c_map },
            );
            sweep(&tile, tiles_m * tiles_n / PAIRS);
        }
    }
}
