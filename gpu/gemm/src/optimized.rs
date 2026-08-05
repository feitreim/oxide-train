//! # The tcgen05 GEMMs — `C = A·Bᵀ` and `dW += Aᵀ·B` on the `cta_group::2` path
//!
//! bf16 operands, an fp32 TMEM accumulator, and a `C` that is either packed
//! bf16 or fp32, either overwritten or accumulated into. The training variants
//! share one compute pipeline: the operand layout is a runtime flag on the
//! *walk* and the epilogue is a type parameter, so neither is a second MMA
//! chain.
//!
//! A cluster of two CTAs shares one `M256_N256` UMMA. Each rank stages its own
//! [`BLOCK_M`] rows of `A` and its own [`HALF_N`] columns of `B` at the same
//! shared offsets, the instruction reads both ranks' shared memory, and each
//! rank drains its own `[BLOCK_M, BLOCK_N]` band. Rank [`LEADER`] owns the MMA,
//! the accumulator and the stage barriers, so the item boundary is
//! `barrier.cluster` rather than `bar.sync`.
//!
//! The grid is **persistent** — [`MAX_CLUSTERS`] clusters, two CTAs an SM — and
//! one work item is one `[2·BLOCK_M, BLOCK_N]` output tile.
//! [`pipeline::run`] hands out item indices and [`pipeline::grouped`] maps them
//! to tiles in blocks of [`GROUP`] tile-rows. K is [`STAGES`] deep over a pair
//! of tile rings; the TMA fills `load` and the MMA's own commit releases `free`,
//! so what proves an operand has been read is the accumulator instruction and
//! not a thread.
//!
//! ## What replaced what
//!
//! This is a rewrite of the 2026-07-26 extraction-point kernel, not a port of
//! it. That one launched **one cluster per output tile** with a hand-unrolled
//! four-stage pipeline over eight `SharedArray` statics, six warps (a TMA warp,
//! an MMA warp and four epilogue warps), and an inert two-slot TMEM ping-pong
//! left over from a CLC schedule it had removed. All of that is gone: the ring
//! is a [`SharedTileRing`] over a [`SharedPlan`], the schedule is
//! [`pipeline::run`], and the epilogue warps are the CTA's four band warps —
//! joined, since oxide-train#80 remedy 2, by one scheduler warp that produces
//! and multiplies so the band warps never block the schedule.
//!
//! The tile is `[256, 256]` at `STAGES = 3` because ferro-kittens' own sweep
//! (#87) put it **+11.6% / +21.6%** at 8192³ / 16384³ over `[256, 128]` at the
//! same depth — with `[256, 128] @ STAGES = 2`, which *raises* residency to 4
//! CTAs/SM, the worst rung in that table at −35.3%. Two CTAs an SM is forced
//! here anyway: at `BLOCK_N = 256` the tensor-memory term binds first
//! (`512 / 256 = 2`), before shared memory is consulted.
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
//! epilogue is deferred one item, each drain releases the accumulator's
//! columns through `Release` the moment its last `tcgen05.ld` retires, and
//! the store tail plus the whole next TMA fill run beside the next MMA.
//!
//! Within a pass the drain keeps the shape #116/#117 measured: TMEM → registers
//! (`tcgen05.ld.16x256b.x8`) → `stmatrix.m8n8.x4` into a per-warp
//! `[32, STAGE_N]` shared tile → 16-byte stores, four passes a band. Not a TMA
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
use cuda_device::{DisjointSlice, cluster, cluster_launch, kernel, warp};
use cuda_host::cuda_module;

use kittens::epilogue::{StoreRing, Warp};
use kittens::global::{GlobalRows, accumulate_shared_rows, store_rows, store_shared_rows};
use kittens::ldst::{scatter_tile, store_tile_x4};
use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline::{self, Job};
use kittens::plan::SharedPlan;
use kittens::shared::{Bf16, F32, SharedTile, SharedTileRing, Swizzle128B};
use kittens::sync::{ClusterSemaphore, Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};
use kittens::{BaseLdtm, RegTile, lane, warp_id};

use super::phase_probe;

/// Rows of the pair tile one rank stages and drains.
pub const BLOCK_M: usize = 128;
/// Columns of the pair tile — the widest `cta_group::2` MMA shape there is.
pub const BLOCK_N: usize = 256;
/// Columns of `B` one rank stages: the pair's tile, split down the middle.
const HALF_N: usize = BLOCK_N / 2;
/// One 128-byte swizzle atom of bf16, and the only width
/// [`SharedTile::k_walk`] accepts — so `BLOCK_K` and [`STAGES`] are a
/// factorization of the shared budget rather than two free axes.
pub const BLOCK_K: usize = 64;
/// K=16 chunks one stage chains, both layouts alike.
const CHUNKS: usize = BLOCK_K / 16;
/// Ring depth. 3 → 2 is −11.8% / −7.3% at unchanged residency (ferro #87).
const STAGES: usize = 3;
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
/// CTAs of the cluster that share one accumulator and one barrier set.
///
/// Named for the pair rather than `RANKS` so `Job::RANKS` can be written
/// beside it without either shadowing the other.
const CLUSTER_RANKS: u32 = 2;
const PAIR: u16 = ((1u32 << CLUSTER_RANKS) - 1) as u16;
const LEADER: u32 = 0;

/// A K-major `[BLOCK_M, BLOCK_K]` operand stage. `A`'s and `B`'s are the same
/// type because a rank stages [`HALF_N`] columns of `B`, and `HALF_N` is
/// [`BLOCK_M`].
type Stage = SharedTile<Bf16, BLOCK_M, BLOCK_K, Swizzle128B>;
/// The same bytes read MN-major: `[BLOCK_K, BLOCK_M]` as two stacked 64-wide
/// subtiles, K along the rows. A 128-byte swizzle caps a TMA box at 128 bytes,
/// so a rank's 128 MN values arrive as two boxes rather than one.
type MnStage = SharedTile<Bf16, BLOCK_K, BLOCK_M, Swizzle128B>;
type Ring = SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type StageTile = SharedTile<Bf16, 32, STAGE_N, Swizzle128B>;
type StageRun = SharedTileRing<Bf16, 32, STAGE_N, Swizzle128B, DRAIN_WARPS>;
type Band = RegTile<32, STAGE_N, BaseLdtm>;
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;

const _: () = assert!(HALF_N == BLOCK_M, "one stage type serves both operands");
const _: () = assert!(
    Stage::BYTES == MnStage::BYTES,
    "one ring serves both operand walks"
);

/// SMs on a B200, and the CTAs of one a `BLOCK_N = 256` accumulator admits:
/// `512 / BLOCK_N` tensor-memory columns, which binds before shared memory
/// does.
const SMS: u32 = 148;
const CTAS_PER_SM: u32 = (512 / BLOCK_N) as u32;
pub const MAX_CLUSTERS: u32 = SMS * CTAS_PER_SM / CLUSTER_RANKS;

/// [`pipeline::grouped`]'s width, swept at this tile shape by ferro-kittens
/// #89 — a tile change is a reason to re-run that sweep.
pub const GROUP: u32 = 8;

struct Shared {
    a_ring: Ring,
    b_ring: Ring,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    done: Semaphore,
    acc_free: Semaphore,
    tmem_slot: *mut u32,
    plan: SharedPlan,
}

#[inline(always)]
const fn shared(at: SharedPlan) -> Shared {
    let (a_ring, at) = at.tile_ring::<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>();
    let (b_ring, at) = at.tile_ring::<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>();
    let (load, at) = at.semaphores::<STAGES>();
    let (free, at) = at.semaphores::<STAGES>();
    let (done, at) = at.semaphore();
    let (acc_free, at) = at.semaphore();
    let (tmem_slot, at) = at.tmem_slot();
    Shared {
        a_ring,
        b_ring,
        load,
        free,
        done,
        acc_free,
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
    assert!(THREADS == 192 && MAX_CLUSTERS == 148);
    // `acc_free` costs the plan nothing: the eight bytes land in the
    // 128-byte alignment padding in front of the staging tiles.
    assert!(SHARED_BYTES == 114_816 && SHARED_BYTES <= 116_736);
    assert!(
        BLOCK_N == 4 * STAGE_N,
        "every drain spells its four passes out to hoist the loads ahead of the stores"
    );
};

/// The moment a drain has read everything it will read from tensor memory.
///
/// The deferred epilogue's whole point (oxide-train#80 remedy 2): the next
/// item's MMA needs the accumulator's *columns*, not its values in `C`, so a
/// drain that has retired its last `tcgen05.ld` hands the columns back through
/// the leader's `acc_free` barrier and issues the stores it still owes behind
/// the release. Between the fence and the arrival every lane of the warp
/// synchronizes, so one lane's arrival covers thirty-two lanes' loads.
#[derive(Clone, Copy)]
struct Release {
    sem: ClusterSemaphore,
    /// The teardown drain after the item loop runs on invalidated barriers,
    /// so it fences and arrives nowhere.
    live: bool,
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
            if self.live && lane() == 0 {
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
/// Every impl owes `release` the same discipline: its loads are hoisted one
/// pass ahead of its stores — two bands live instead of one, which is what the
/// register budgets in `main.rs` price — and [`Release::now`] is called the
/// instant the last load has its registers, so the tail of the store work runs
/// beside the next item's MMA rather than in front of it.
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
            let first: Band = accumulator.tile_x8(band_row, 0);
            let second: Band = accumulator.tile_x8(band_row, n);
            self.pass(stage, first, row, column);
            let third: Band = accumulator.tile_x8(band_row, 2 * n);
            self.pass(stage, second, row, column + n);
            let fourth: Band = accumulator.tile_x8(band_row, 3 * n);
            release.now();
            self.pass(stage, third, row, column + 2 * n);
            self.pass(stage, fourth, row, column + 3 * n);
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
#[derive(Clone, Copy)]
struct Wide {
    c: GlobalRows<F32>,
}

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
            // One band live, not two: this drain already holds the fattest
            // registers in the file, and the two-band hoist measured 181 —
            // past the 170 the register file grants 12 warps an SM, which is
            // the 2 → 1 CTA cliff by another name. Releasing after the fourth
            // load still overlaps the last store pass and every store's
            // completion with the next item's MMA.
            let (lane, band_row) = (lane(), 32 * warp_id());
            let n = STAGE_N as u32;
            let first: Band = accumulator.tile_x8(band_row, 0);
            store_rows(self.c, row, column, lane, first);
            let second: Band = accumulator.tile_x8(band_row, n);
            store_rows(self.c, row, column + n, lane, second);
            let third: Band = accumulator.tile_x8(band_row, 2 * n);
            store_rows(self.c, row, column + 2 * n, lane, third);
            let fourth: Band = accumulator.tile_x8(band_row, 3 * n);
            release.now();
            store_rows(self.c, row, column + 3 * n, lane, fourth);
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
    unsafe fn emit(
        self,
        ring: &mut ReduceRing,
        lane: u32,
        band: RegTile<16, STAGE_N, BaseLdtm>,
        row: u32,
        column: u32,
    ) {
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
            // Eight passes spelled out rather than looped — the same argument
            // `kittens::tmem`'s batching section makes: a loop-carried band
            // wants a runtime index and lands in local memory. The loads run
            // one pass ahead of the engine so the eighth is waited out while
            // the seventh is still being scattered, and the release sits
            // between them: only the last pass's scatter and the engine's
            // final read run after the accumulator is handed back.
            let n = STAGE_N as u32;
            let (top, bottom) = (band_row, band_row + 16);
            let b0: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(top, 0);
            let b1: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(top, n);
            self.emit(&mut ring, lane, b0, row, column);
            let b2: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(top, 2 * n);
            self.emit(&mut ring, lane, b1, row, column + n);
            let b3: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(top, 3 * n);
            self.emit(&mut ring, lane, b2, row, column + 2 * n);
            let b4: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(bottom, 0);
            self.emit(&mut ring, lane, b3, row, column + 3 * n);
            let b5: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(bottom, n);
            self.emit(&mut ring, lane, b4, row + 16, column);
            let b6: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(bottom, 2 * n);
            self.emit(&mut ring, lane, b5, row + 16, column + n);
            let b7: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(bottom, 3 * n);
            self.emit(&mut ring, lane, b6, row + 16, column + 2 * n);
            release.now();
            self.emit(&mut ring, lane, b7, row + 16, column + 3 * n);
            ring.drain();
        }
    }
}

#[derive(Clone, Copy)]
struct Tile<D: Drain> {
    a_ring: Ring,
    b_ring: Ring,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    done: Semaphore,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    accumulator: Accumulator,
    stage: StageTile,
    out: D,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    transposed: bool,
    rank: u32,
    acc_free: Semaphore,
    /// The item whose accumulator is still in tensor memory, plus one — the
    /// deferred epilogue's cursor, `0` before the first item. Register state,
    /// advanced by every thread in lockstep.
    pending: u32,
}

impl<D: Drain> Tile<D> {
    /// # Safety
    /// One thread of the CTA, once per item.
    #[inline(always)]
    unsafe fn produce(&self, tile_m: u32, tile_n: u32) {
        unsafe {
            // All four of the pair's tiles complete on the leader's copy of the
            // stage barrier, and only the leader charges it: `expect_tx` is
            // `.shared::cta`, so a peer could not charge that barrier even
            // holding its address. Both ranks derive the same half-stage charge
            // from the loads they just issued; the leader scales its own by
            // `RANKS` to cover the peer's, and the peer drops it.
            let a_line = (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank) as i32;
            let b_line = (BLOCK_N as u32 * tile_n + HALF_N as u32 * self.rank) as i32;
            let mut k = 0u32;
            while k < self.k_blocks {
                self.free.wait_recycled(k);
                let stage = self.load.sem(k).at_rank(LEADER);
                let depth = (BLOCK_K as u32 * k) as i32;
                let (a, b) = (self.a_ring.tile(k), self.b_ring.tile(k));
                // The map's fast axis dictates the coordinate order: a K-major
                // operand is one box at `(k, mn)`, an MN-major one a box per
                // 64-wide subtile at `(mn, k)`. Same transaction bytes either
                // way, so the charge does not depend on the branch.
                let bytes = if self.transposed {
                    MnStage::from_raw(a.base())
                        .tma_load_2d_arriving_at(self.a_map, a_line, depth, stage)
                        + MnStage::from_raw(b.base())
                            .tma_load_2d_arriving_at(self.b_map, b_line, depth, stage)
                } else {
                    a.tma_load_2d_arriving_at(self.a_map, depth, a_line, stage)
                        + b.tma_load_2d_arriving_at(self.b_map, depth, b_line, stage)
                };
                if self.rank == LEADER {
                    self.load
                        .sem(k)
                        .expect_tx(bytes.across_ranks(CLUSTER_RANKS));
                }
                k += 1;
            }
        }
    }

    /// # Safety
    /// One thread of the leader rank, with the accumulator's previous contents
    /// already read: only the first stage of an item starts it fresh.
    #[inline(always)]
    unsafe fn multiply(&self) {
        unsafe {
            let mut k = 0u32;
            while k < self.k_blocks {
                self.load.wait(k);
                let (a, b) = (self.a_ring.tile(k), self.b_ring.tile(k));
                // A select on the walk, not a duplicated MMA chain: an
                // `OperandWalk` carries its own transpose bit, so both layouts
                // issue through one loop.
                let (a_walk, b_walk) = if self.transposed {
                    (
                        MnStage::from_raw(a.base()).mn_walk(),
                        MnStage::from_raw(b.base()).mn_walk(),
                    )
                } else {
                    (a.k_walk(), b.k_walk())
                };
                mma_walk_cg2::<Bf16, CHUNKS>(
                    self.accumulator.raw(),
                    a_walk,
                    b_walk,
                    MmaShape::M256_N256,
                    k > 0,
                );
                commit_multicast_cg2(self.free.sem(k), PAIR);
                k += 1;
            }
            commit_multicast_cg2(self.done, PAIR);
        }
    }

    /// This warp's origin in `C` for `item`: the pair tile, this rank's half of
    /// it, and the 32 rows the warp owns.
    #[inline(always)]
    fn origin(&self, item: u32) -> (u32, u32) {
        let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, GROUP);
        (
            2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank + 32 * warp_id(),
            BLOCK_N as u32 * tile_n,
        )
    }
}

impl<D: Drain> Job for Tile<D> {
    const RANKS: u32 = CLUSTER_RANKS;

    /// # Safety
    /// As [`Semaphore::init`].
    #[inline(always)]
    unsafe fn init(&self, _item: u32) {
        unsafe {
            self.load.init_all(1);
            self.free.init_all(1);
            self.done.init(1);
            // One arrival per band warp per rank: the MMA writes both ranks'
            // tensor memory, so it waits for both ranks' drains.
            self.acc_free.init(DRAIN_WARPS as u32 * CLUSTER_RANKS);
        }
    }

    /// # Safety
    /// As [`Semaphore::inval`].
    #[inline(always)]
    unsafe fn inval(&self) {
        unsafe {
            self.load.inval_all();
            self.free.inval_all();
            self.done.inval();
            self.acc_free.inval();
        }
    }

    /// # Safety
    /// Every thread of both CTAs of the cluster must enter with the same
    /// `item`, and the maps must cover the tile it names — and the one before
    /// it, whose drain this item runs.
    ///
    /// The epilogue is **deferred one item** (the load-compute-store-finish
    /// form `pipeline::Job`'s doc sanctions): the band warps open item `i` by
    /// draining item `i - 1`'s accumulator, releasing its columns to this
    /// item's MMA the moment their loads retire, and close it waiting out this
    /// item's `done` — so the barrier the boundary re-arms has completed
    /// inside the item that armed it, and what runs beside the MMA is the
    /// previous drain's store tail plus this item's whole TMA fill.
    #[inline(always)]
    unsafe fn work(&mut self, item: u32) {
        unsafe {
            let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, GROUP);
            let (warp, lane) = (warp_id(), lane());

            if warp == PRODUCER {
                if lane == 0 {
                    self.produce(tile_m, tile_n);
                }
            } else if warp == ISSUER {
                if self.rank == LEADER && lane == 0 {
                    self.acc_free.wait(0);
                    self.multiply();
                }
            } else {
                let release = Release {
                    sem: self.acc_free.at_rank(LEADER),
                    live: true,
                };
                if self.pending == 0 {
                    // Nothing to read: the first item's accumulator is free
                    // the moment the barrier says so.
                    release.now();
                } else {
                    let (row, column) = self.origin(self.pending - 1);
                    self.out
                        .drain(self.accumulator, self.stage, row, column, release);
                }
                self.done.wait(0);
            }
            self.pending = item + 1;
        }
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
            Tile {
                a_ring: shared.a_ring,
                b_ring: shared.b_ring,
                load: shared.load,
                free: shared.free,
                done: shared.done,
                a_map,
                b_map,
                accumulator: Accumulator::from_raw(alloc_cluster(shared.tmem_slot, BLOCK_N as u32)),
                stage: run.tile(warp_id() % DRAIN_WARPS as u32),
                out,
                tiles_m,
                tiles_n,
                k_blocks,
                transposed,
                rank: cluster::block_rank(),
                acc_free: shared.acc_free,
                pending: 0,
            }
        }
    }

    /// Drain the accumulator the deferred epilogue still holds after the item
    /// loop — the last item's, which `work` waited complete but did not read.
    /// `pipeline::run` has already invalidated the barriers, so this drain
    /// releases nothing and nobody waits on it: [`release`]'s own
    /// `cluster_sync` is what keeps the columns alive until the reads retire.
    ///
    /// # Safety
    /// After [`pipeline::run`] returns and before [`release`], every thread.
    #[inline(always)]
    unsafe fn drain_last<D: Drain>(tile: &Tile<D>) {
        unsafe {
            if tile.pending != 0 && warp_id() < DRAIN_WARPS as u32 {
                let (row, column) = tile.origin(tile.pending - 1);
                let release = Release {
                    sem: tile.acc_free.at_rank(LEADER),
                    live: false,
                };
                tile.out
                    .drain(tile.accumulator, tile.stage, row, column, release);
            }
        }
    }

    /// # Safety
    /// Every thread of every rank must arrive, with the accumulator's last
    /// reader retired.
    #[inline(always)]
    unsafe fn release<D: Drain>(tile: &Tile<D>) {
        unsafe {
            // The item boundary already retired the pair's reads; this
            // `cluster_sync` is for the cluster that got no items at all, which
            // a capped grid can leave having allocated and never looped.
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();
            dealloc_cluster(tile.accumulator.raw(), BLOCK_N as u32);
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
    #[cluster_launch(2, 1, 1)]
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
            let mut tile = attach(
                a_map,
                b_map,
                tiles_m,
                tiles_n,
                k as u32 / BLOCK_K as u32,
                transposed != 0,
                out,
            );
            pipeline::run(&mut tile, tiles_m * tiles_n);
            drain_last(&tile);
            release(&tile);
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
    #[cluster_launch(2, 1, 1)]
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
            let mut tile = attach(
                a_map,
                b_map,
                tiles_m,
                tiles_n,
                k as u32 / BLOCK_K as u32,
                transposed != 0,
                out,
            );
            pipeline::run(&mut tile, tiles_m * tiles_n);
            drain_last(&tile);
            release(&tile);
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
    #[cluster_launch(2, 1, 1)]
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
            let mut tile = attach(
                a_map,
                b_map,
                tiles_m,
                tiles_n,
                k as u32 / BLOCK_K as u32,
                transposed != 0,
                Reduce { c_map },
            );
            pipeline::run(&mut tile, tiles_m * tiles_n);
            drain_last(&tile);
            release(&tile);
        }
    }

    /// [`gemm_tcgen05_f32_optimized`] with a stopwatch on each phase of an item
    /// — oxide-train#80's forensics, whose body is [`phase_probe`]
    /// and whose only caller is `bin/budget.rs`. It computes the same `C`.
    ///
    /// The entry point lives here rather than in a module of its own because a
    /// binary gets one device artifact: a second `#[cuda_module]` beside this
    /// one loads, and then every symbol of the other is missing.
    ///
    /// # Safety
    /// As [`gemm_tcgen05_f32_optimized`], plus `clocks` holding
    /// [`phase_probe::COUNTERS`] zeroed `u64` per cluster.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_probe_f32_store(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut c: DisjointSlice<f32>,
        mut clocks: DisjointSlice<u64>,
        n: i32,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
        transposed: u32,
    ) {
        unsafe {
            phase_probe::probe::<true>(
                a_map,
                b_map,
                &mut c,
                &mut clocks,
                n,
                k,
                tiles_m,
                tiles_n,
                transposed,
            )
        }
    }

    /// [`gemm_probe_f32_store`] with the epilogue deleted — ferro #114's
    /// `no drain` floor, which writes no `C` and is a timing arm only.
    ///
    /// # Safety
    /// As [`gemm_probe_f32_store`]; `c` is untouched.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_probe_f32_nodrain(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        mut c: DisjointSlice<f32>,
        mut clocks: DisjointSlice<u64>,
        n: i32,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
        transposed: u32,
    ) {
        unsafe {
            phase_probe::probe::<false>(
                a_map,
                b_map,
                &mut c,
                &mut clocks,
                n,
                k,
                tiles_m,
                tiles_n,
                transposed,
            )
        }
    }
}
