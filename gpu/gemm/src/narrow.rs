//! # The `[256, 128]` pair-tile variant of [`super::optimized`]
//!
//! The same kernel — the persistent grid, the deferred drain, the three
//! epilogues — at half the pair tile's columns, compiled as a second set of
//! entry points and selected per shape by [`super::host::tcgen05_launch_config`]
//! (oxide-train#80 remedy 3). Nothing here replaces the `[256, 256]` kernel:
//! ferro-kittens #87 measured the wide tile +11.6% / +21.6% at 8192³ / 16384³,
//! and the model's deep-K rows sit at 0.94–0.97 of cuBLASLt on it. What the
//! wide tile cannot fix is what this file exists for:
//!
//! - **Wave quantization.** One work item is a `[256, 256]` output tile, and
//!   four of the training step's shapes leave the 148-cluster grid a ragged
//!   last wave (down dW's 192 tiles are 1.3 waves — 64.9% efficient). Halving
//!   the tile doubles the item count and halves the quantum the last wave
//!   rounds up by.
//! - **The drain-release cap.** A 256-column accumulator cannot be
//!   double-buffered — two slots at two CTAs an SM would need 1024 of the SM's
//!   512 tensor-memory columns — so the wide kernel's drain must release
//!   *early*, and the loads it cannot hoist stay serial. At 128 columns two
//!   slots fit exactly (`2 × 128 × 2 = 512`), and this kernel takes them:
//!   the MMA of one item and the drain of the previous one run on disjoint
//!   column halves of a 256-column allocation, so **the whole drain — loads
//!   included — runs beside the MMA**, and the wide kernel's release barrier
//!   disappears outright. The rendezvous that keeps the slots exclusive is
//!   `pipeline::run`'s own item boundary: the drain of item `i-2` retired
//!   inside item `i-1` (its warps' last `tcgen05.ld` is fenced by the
//!   boundary's `tcgen05_fence_before_thread_sync`), so when item `i` opens,
//!   its slot is free by construction and the MMA issuer waits on nothing but
//!   its operands.
//!
//! The tile is `[256, 128]` at `STAGES = 4`: the narrower B ring frees exactly
//! one more K stage inside the same 114 816-byte plan, so the shared envelope,
//! the two-CTAs-per-SM residency and [`MAX_CLUSTERS`] all match the wide
//! kernel's — the A/B between the two variants moves the tile and nothing
//! else. (ferro #87 priced this rung's occupancy against the wide tile's at
//! −7.7% / +2.1% on big squares; the dispatch keeps squares on the wide
//! kernel.)
//!
//! Halving `BLOCK_N` halves per-tile B reuse (`M·N/(M+N)`: 128 → 85.3) and
//! doubles the per-output epilogue count; the double-buffered drain is what
//! pays those costs back, which is why this kernel ships in the composed form
//! rather than as a wave-math-only variant (that increment was measured
//! separately on the way here — oxide-train#80).

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
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};
use kittens::{BaseLdtm, RegTile, lane, warp_id};

/// Rows of the pair tile one rank stages and drains — the same 128 as the wide
/// kernel: `M` is fixed at 256 by the widest `cta_group::2` `MmaShape` there
/// is, so a smaller pair tile can only be narrower, not shorter.
pub const BLOCK_M: usize = 128;
/// Columns of the pair tile: half the wide kernel's, the `M256_N128` shape.
pub const BLOCK_N: usize = 128;
/// Columns of `B` one rank stages. At 64 it no longer equals [`BLOCK_M`], so
/// `B`'s stage is its own type here where the wide kernel reuses `A`'s.
const HALF_N: usize = BLOCK_N / 2;
/// One 128-byte swizzle atom of bf16 — see the wide kernel.
pub const BLOCK_K: usize = 64;
/// K=16 chunks one stage chains, both layouts alike.
const CHUNKS: usize = BLOCK_K / 16;
/// Ring depth. One deeper than the wide kernel's three: the B ring shrank by
/// exactly one stage-pair's worth of bytes, so the fourth stage lands the plan
/// on the same 114 816 B and costs residency nothing.
const STAGES: usize = 4;
/// Four band warps, the producer and the MMA issuer — the wide kernel's
/// role split, unchanged.
pub const THREADS: u32 = (BLOCK_M / 32) as u32 * 32 + 64;
/// Warps that own a 32-row band of the accumulator.
const DRAIN_WARPS: usize = BLOCK_M / 32;
/// The warp whose lane 0 issues the item's TMA loads.
const PRODUCER: u32 = DRAIN_WARPS as u32;
/// The warp whose lane 0 (leader rank only) waits the accumulator free and
/// issues the MMA chain — its own warp, not a diverged lane of the producer's,
/// for the reason the wide kernel records.
const ISSUER: u32 = PRODUCER + 1;
/// The staged drain's band width. A `[32, BLOCK_N]` band is two passes of it
/// where the wide kernel's is four.
const STAGE_N: usize = 64;
/// CTAs of the cluster that share one accumulator and one barrier set.
const CLUSTER_RANKS: u32 = 2;
const PAIR: u16 = ((1u32 << CLUSTER_RANKS) - 1) as u16;
const LEADER: u32 = 0;

/// A K-major `[BLOCK_M, BLOCK_K]` `A` stage.
type AStage = SharedTile<Bf16, BLOCK_M, BLOCK_K, Swizzle128B>;
/// `A` read MN-major: `[BLOCK_K, BLOCK_M]` as two stacked 64-wide subtiles.
type AMnStage = SharedTile<Bf16, BLOCK_K, BLOCK_M, Swizzle128B>;
/// This rank's `[HALF_N, BLOCK_K]` of `B` — and, read MN-major, the same
/// `[64, 64]` shape with the axes swapped, so one type serves both walks and
/// the TMA box is `[64, 64]` either way (the *narrow* box of the operand's
/// tensor-map pair; the wide kernel's K-major `[64, 128]` box does not fit a
/// 64-row stage).
type BStage = SharedTile<Bf16, HALF_N, BLOCK_K, Swizzle128B>;
type ARing = SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type BRing = SharedTileRing<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>;
type StageTile = SharedTile<Bf16, 32, STAGE_N, Swizzle128B>;
type StageRun = SharedTileRing<Bf16, 32, STAGE_N, Swizzle128B, DRAIN_WARPS>;
type Band = RegTile<32, STAGE_N, BaseLdtm>;
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;

const _: () = assert!(HALF_N == BLOCK_K, "one [64, 64] stage serves both B walks");
const _: () = assert!(
    AStage::BYTES == AMnStage::BYTES,
    "one ring serves both A walks"
);

/// SMs on a B200. Tensor memory alone would admit `512 / 128 = 4` CTAs of an
/// SM here, but the shared plan holds two — and the double-buffered form of
/// this kernel allocates 256 columns anyway, so two is the number this tile is
/// designed around, not a compromise.
const SMS: u32 = 148;
const CTAS_PER_SM: u32 = 2;
pub const MAX_CLUSTERS: u32 = SMS * CTAS_PER_SM / CLUSTER_RANKS;

/// [`pipeline::grouped`]'s width. Kept at the wide kernel's 8: ferro #89 swept
/// it at a `[256, 128]` tile originally, so this tile is the one it was
/// measured at.
pub const GROUP: u32 = 8;

/// Tensor-memory columns one CTA allocates: two [`BLOCK_N`]-column slots,
/// which at [`CTAS_PER_SM`] = 2 is the SM's whole 512 — exactly.
const TMEM_COLUMNS: u32 = 2 * BLOCK_N as u32;

struct Shared {
    a_ring: ARing,
    b_ring: BRing,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    done: Semaphore,
    tmem_slot: *mut u32,
    plan: SharedPlan,
}

#[inline(always)]
const fn shared(at: SharedPlan) -> Shared {
    let (a_ring, at) = at.tile_ring::<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>();
    let (b_ring, at) = at.tile_ring::<Bf16, HALF_N, BLOCK_K, Swizzle128B, STAGES>();
    let (load, at) = at.semaphores::<STAGES>();
    let (free, at) = at.semaphores::<STAGES>();
    let (done, at) = at.semaphore();
    let (tmem_slot, at) = at.tmem_slot();
    Shared {
        a_ring,
        b_ring,
        load,
        free,
        done,
        tmem_slot,
        plan: at,
    }
}

/// One `[32, STAGE_N]` staging tile per warp, past the operand plan.
#[inline(always)]
const fn staged(at: SharedPlan) -> (StageRun, SharedPlan) {
    at.tile_ring::<Bf16, 32, STAGE_N, Swizzle128B, DRAIN_WARPS>()
}

/// Dynamic shared memory both entry points declare — the wide kernel's exact
/// 114 816 B, by construction rather than coincidence: the fourth K stage's
/// 24 576 B are the two stages of B-ring the narrower tile gave back.
pub const SHARED_BYTES: usize = staged(shared(SharedPlan::sizing()).plan).1.bytes();

const _: () = {
    assert!(THREADS == 192 && MAX_CLUSTERS == 148);
    assert!(SHARED_BYTES == 114_816 && SHARED_BYTES <= 116_736);
    assert!(
        BLOCK_N == 2 * STAGE_N,
        "every drain spells its two passes out to hoist the loads ahead of the stores"
    );
};

/// Where a band of the accumulator goes, and how — the wide kernel's trait at
/// two passes a band instead of four, and with no release parameter: the
/// double-buffered accumulator frees a drain from ever having to hand its
/// columns to anyone mid-flight. Loads still run a pass ahead of stores for
/// their latency, not for a barrier.
trait Drain: Copy {
    /// Push this warp's whole `[32, BLOCK_N]` band out to `C` at
    /// `(row, column)`.
    ///
    /// # Safety
    ///
    /// - Every lane of the warp calls this together, with the accumulator
    ///   slot's MMA complete and fenced, and nothing writing that slot until
    ///   the item boundary retires these reads.
    /// - The band's rectangle lies inside `C`.
    unsafe fn drain(self, accumulator: Accumulator, stage: StageTile, row: u32, column: u32);
}

/// Packed-bf16 `C`, through the staging tile — see the wide kernel's `Packed`.
#[derive(Clone, Copy)]
struct Packed {
    c: GlobalRows<Bf16>,
    fold: bool,
}

impl Packed {
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
    unsafe fn drain(self, accumulator: Accumulator, stage: StageTile, row: u32, column: u32) {
        unsafe {
            let band_row = 32 * warp_id();
            let n = STAGE_N as u32;
            let first: Band = accumulator.tile_x8(band_row, 0);
            let second: Band = accumulator.tile_x8(band_row, n);
            self.pass(stage, first, row, column);
            self.pass(stage, second, row, column + n);
        }
    }
}

/// Overwriting fp32 `C`, straight out of the registers — the wide kernel's
/// `Wide` at one band held, the shape GAPS.md §2.6 explains.
#[derive(Clone, Copy)]
struct WideOut {
    c: GlobalRows<F32>,
}

impl Drain for WideOut {
    #[inline(always)]
    unsafe fn drain(self, accumulator: Accumulator, _stage: StageTile, row: u32, column: u32) {
        unsafe {
            let (lane, band_row) = (lane(), 32 * warp_id());
            let n = STAGE_N as u32;
            let first: Band = accumulator.tile_x8(band_row, 0);
            store_rows(self.c, row, column, lane, first);
            let second: Band = accumulator.tile_x8(band_row, n);
            store_rows(self.c, row, column + n, lane, second);
        }
    }
}

/// The accumulating drain's staging ring — byte-for-byte the wide kernel's
/// reinterpretation of one bf16 staging tile.
type ReduceRing = StoreRing<F32, 16, STAGE_N, Swizzle128B, 0, Warp>;

const _: () = assert!(
    ReduceRing::BYTES == StageTile::BYTES,
    "the reduce ring reinterprets one bf16 staging tile exactly"
);

/// Accumulating fp32 `C` through the copy engine's reduction store — the wide
/// kernel's `Reduce` at four `[16, STAGE_N]` passes instead of eight.
#[derive(Clone, Copy)]
struct Reduce {
    c_map: *const TmaDescriptor,
}

impl Reduce {
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
    unsafe fn drain(self, accumulator: Accumulator, stage: StageTile, row: u32, column: u32) {
        unsafe {
            let (lane, band_row) = (lane(), 32 * warp_id());
            let mut ring = ReduceRing::attach(stage.base());
            // Four passes spelled out, loads one pass ahead of the engine —
            // the wide kernel's schedule at half the passes.
            let n = STAGE_N as u32;
            let (top, bottom) = (band_row, band_row + 16);
            let b0: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(top, 0);
            let b1: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(top, n);
            self.emit(&mut ring, lane, b0, row, column);
            let b2: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(bottom, 0);
            self.emit(&mut ring, lane, b1, row, column + n);
            let b3: RegTile<16, STAGE_N, BaseLdtm> = accumulator.tile_x8(bottom, n);
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
    /// The item whose accumulator is still in tensor memory, plus one — the
    /// deferred epilogue's cursor, `0` before the first item.
    pending: u32,
    /// Items this *cluster* has started — the slot-parity counter. Not
    /// derivable from the item index: the static schedule strides items by the
    /// cluster count, which is even, so a global index's parity would park
    /// every cluster on one slot forever.
    cycle: u32,
}

impl<D: Drain> Tile<D> {
    /// # Safety
    /// One thread of the CTA, once per item.
    #[inline(always)]
    unsafe fn produce(&self, tile_m: u32, tile_n: u32) {
        unsafe {
            // The pair's four tiles complete on the leader's copy of the stage
            // barrier, leader-charged — the wide kernel's protocol. What the
            // narrower tile changes is only `B`'s geometry: a rank stages
            // [`HALF_N`] = 64 columns, one `[64, 64]` box in either layout.
            let a_line = (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank) as i32;
            let b_line = (BLOCK_N as u32 * tile_n + HALF_N as u32 * self.rank) as i32;
            let mut k = 0u32;
            while k < self.k_blocks {
                self.free.wait_recycled(k);
                let stage = self.load.sem(k).at_rank(LEADER);
                let depth = (BLOCK_K as u32 * k) as i32;
                let (a, b) = (self.a_ring.tile(k), self.b_ring.tile(k));
                let bytes = if self.transposed {
                    AMnStage::from_raw(a.base())
                        .tma_load_2d_arriving_at(self.a_map, a_line, depth, stage)
                        + b.tma_load_2d_arriving_at(self.b_map, b_line, depth, stage)
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
    /// One thread of the leader rank, with `accumulator` — this item's slot —
    /// free: its last reader retired behind an item boundary.
    #[inline(always)]
    unsafe fn multiply(&self, accumulator: Accumulator) {
        unsafe {
            let mut k = 0u32;
            while k < self.k_blocks {
                self.load.wait(k);
                let (a, b): (AStage, BStage) = (self.a_ring.tile(k), self.b_ring.tile(k));
                // `B`'s `[64, 64]` stage is the same tile either way; the walk
                // carries the transpose. `A` keeps the wide kernel's pair.
                let (a_walk, b_walk) = if self.transposed {
                    (AMnStage::from_raw(a.base()).mn_walk(), b.mn_walk())
                } else {
                    (a.k_walk(), b.k_walk())
                };
                mma_walk_cg2::<Bf16, CHUNKS>(
                    accumulator.raw(),
                    a_walk,
                    b_walk,
                    MmaShape::M256_N128,
                    k > 0,
                );
                commit_multicast_cg2(self.free.sem(k), PAIR);
                k += 1;
            }
            commit_multicast_cg2(self.done, PAIR);
        }
    }

    /// One of the allocation's two [`BLOCK_N`]-column slots, by parity.
    #[inline(always)]
    fn slot(&self, parity: u32) -> Accumulator {
        self.accumulator
            .columns_right(BLOCK_N as u32 * (parity & 1))
    }

    /// This warp's origin in `C` for `item`.
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
        }
    }

    /// # Safety
    /// Every thread of both CTAs of the cluster must enter with the same
    /// `item`, and the maps must cover the tile it names — and the one before
    /// it, whose drain this item runs.
    ///
    /// The epilogue is deferred one item, as in the wide kernel — but the two
    /// accumulator slots make the MMA wait for **nothing**: item `i` multiplies
    /// into slot `cycle & 1` while the band warps drain item `i-1` out of the
    /// other slot, loads and all. The slot being free needs no barrier — its
    /// last reader was item `i-2`'s drain, which ran inside item `i-1` and was
    /// retired by the item boundary's `tcgen05` fence and cluster sync.
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
                    self.multiply(self.slot(self.cycle));
                }
            } else {
                if self.pending != 0 {
                    let (row, column) = self.origin(self.pending - 1);
                    self.out
                        .drain(self.slot(self.cycle + 1), self.stage, row, column);
                }
                self.done.wait(0);
            }
            self.pending = item + 1;
            self.cycle += 1;
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// # Safety
    /// As the wide kernel's `attach`.
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
                accumulator: Accumulator::from_raw(alloc_cluster(shared.tmem_slot, TMEM_COLUMNS)),
                stage: run.tile(warp_id() % DRAIN_WARPS as u32),
                out,
                tiles_m,
                tiles_n,
                k_blocks,
                transposed,
                rank: cluster::block_rank(),
                pending: 0,
                cycle: 0,
            }
        }
    }

    /// Drain the accumulator slot the deferred epilogue still holds after the
    /// item loop — the last item's, at the parity its `work` multiplied into.
    ///
    /// # Safety
    /// After [`pipeline::run`] returns and before [`release`], every thread.
    #[inline(always)]
    unsafe fn drain_last<D: Drain>(tile: &Tile<D>) {
        unsafe {
            if tile.pending != 0 && warp_id() < DRAIN_WARPS as u32 {
                let (row, column) = tile.origin(tile.pending - 1);
                tile.out
                    .drain(tile.slot(tile.cycle + 1), tile.stage, row, column);
            }
        }
    }

    /// # Safety
    /// Every thread of every rank must arrive, with the accumulator's last
    /// reader retired.
    #[inline(always)]
    unsafe fn release<D: Drain>(tile: &Tile<D>) {
        unsafe {
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();
            dealloc_cluster(tile.accumulator.raw(), TMEM_COLUMNS);
        }
    }

    /// Packed-bf16 `C`, one `[2·BLOCK_M, BLOCK_N]` = `[256, 128]` output tile
    /// per work item — [`super::super::optimized::kernels::gemm_tcgen05_bf16_optimized`]
    /// at the narrow tile; the contract is that kernel's with `tiles_n`
    /// counting 128-column tiles and `b_map`'s box `[64, 64]` in the K-major
    /// layout.
    ///
    /// # Safety
    /// As the wide entry point, at this file's tile and [`SHARED_BYTES`].
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_tcgen05_bf16_narrow(
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

    /// [`gemm_tcgen05_bf16_narrow`] with an fp32 `C`, overwrite only.
    ///
    /// # Safety
    /// As [`gemm_tcgen05_bf16_narrow`], with `c_offset..c_offset + m * n`
    /// inside `c`.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_tcgen05_f32_narrow(
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
            let out = WideOut {
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
    /// through the copy engine's reduction store — the wide accumulate kernel
    /// at the narrow tile. `c_map`'s box stays `[16, 32]`.
    ///
    /// # Safety
    /// As the wide accumulate entry point, at this file's tile.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn gemm_tcgen05_f32_accumulate_narrow(
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
}
