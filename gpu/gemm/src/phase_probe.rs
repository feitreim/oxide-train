//! The shipped tcgen05 GEMM with a stopwatch on every phase of an item
//! (oxide-train#80 forensics).
//!
//! `optimized.rs`'s fp32 store path, copied and instrumented rather than
//! parameterized: the shipped kernel keeps no counters, no `clock64` and no
//! extra argument. This module compiles **only** into `src/bin/budget.rs` —
//! `src/transpose_probe.rs` is the same arrangement — so nothing here reaches
//! `gemm.ptx` or the model.
//!
//! ## What it answers
//!
//! `model_shapes` puts the shallow-K store rows 10–17 points under cuBLASLt
//! while their deep-K siblings reach 0.95–0.97, and the store-vs-fold
//! decomposition never isolated what shallow K pays extra. Per item the loop is
//!
//! ```text
//!   inval + init (leader thread) → cluster_sync → work → fence → cluster_sync
//! ```
//!
//! and `work` is: the band warps drain item `i-1` and release the accumulator,
//! the issuer waits that release and walks `k_blocks` MMAs each gated on its
//! stage's TMA, the producer issues those TMAs. Every one of those waits is a
//! candidate for the fixed per-item term, and they are separated here:
//!
//! | counter | phase |
//! |---|---|
//! | [`PRE`] | the leader's `inval`+`init` and the opening `cluster_sync` |
//! | [`POST`] | the closing tcgen05 fence and `cluster_sync` |
//! | [`ACC`] | the issuer waiting the previous drain's release |
//! | [`FILL`] | the issuer waiting stage 0 — the **pipeline fill**, per item |
//! | [`FEED`] | the issuer waiting stages 1.. — the **steady-state feed stall** |
//! | [`MMA`] | the issuer's whole multiply, fill and feed included |
//! | [`DRAIN`] | a band warp's epilogue |
//! | [`DONE`] | a band warp waiting the item's MMA chain out |
//! | [`FREE`] | the producer waiting a stage back from the MMA (back-pressure) |
//!
//! Ticks are `clock64` read on rank 0's SM, so every counter shares one clock.
//!
//! `DRAIN_ON = false` is the epilogue-free floor (ferro #114's `no drain`),
//! which prices what the drain still costs the item stream after #83 deferred
//! it. That arm writes no `C` and is a timing arm only; the `true` arm keeps
//! the trailing drain and so computes the same `C` the shipped kernel does.

use cuda_device::debug::clock64;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster, cluster_launch, kernel, thread, warp};
use cuda_host::cuda_module;

use kittens::global::{GlobalRows, store_rows};
use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline;
use kittens::plan::SharedPlan;
use kittens::shared::{Bf16, F32, SharedTile, SharedTileRing, Swizzle128B, publish_to_async_proxy};
use kittens::sync::{ClusterSemaphore, Semaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};
use kittens::{BaseLdtm, RegTile, lane, warp_id};

pub const BLOCK_M: usize = 128;
pub const BLOCK_N: usize = 256;
const HALF_N: usize = BLOCK_N / 2;
pub const BLOCK_K: usize = 64;
const CHUNKS: usize = BLOCK_K / 16;
const STAGES: usize = 3;
pub const THREADS: u32 = (BLOCK_M / 32) as u32 * 32 + 64;
const DRAIN_WARPS: usize = BLOCK_M / 32;
const PRODUCER: u32 = DRAIN_WARPS as u32;
const ISSUER: u32 = PRODUCER + 1;
const STAGE_N: usize = 64;
const CLUSTER_RANKS: u32 = 2;
const PAIR: u16 = ((1u32 << CLUSTER_RANKS) - 1) as u16;
const LEADER: u32 = 0;

type Stage = SharedTile<Bf16, BLOCK_M, BLOCK_K, Swizzle128B>;
type MnStage = SharedTile<Bf16, BLOCK_K, BLOCK_M, Swizzle128B>;
type Ring = SharedTileRing<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>;
type StageRun = SharedTileRing<Bf16, 32, STAGE_N, Swizzle128B, DRAIN_WARPS>;
type Band = RegTile<32, STAGE_N, BaseLdtm>;
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;

const SMS: u32 = 148;
const CTAS_PER_SM: u32 = (512 / BLOCK_N) as u32;
pub const MAX_CLUSTERS: u32 = SMS * CTAS_PER_SM / CLUSTER_RANKS;
pub const GROUP: u32 = 8;

/// Counters a cluster keeps, and the stride of the host's readback.
pub const COUNTERS: usize = 16;
/// Items this cluster ran — every other counter divides by it.
pub const ITEMS: usize = 0;
pub const PRE: usize = 1;
pub const POST: usize = 2;
pub const WORK: usize = 3;
pub const ACC: usize = 4;
pub const FILL: usize = 5;
pub const FEED: usize = 6;
pub const MMA: usize = 7;
pub const PROD: usize = 8;
pub const FREE: usize = 9;
pub const DRAIN: usize = 10;
pub const DONE: usize = 11;
/// First `clock64` to last, over the whole item loop.
pub const SPAN: usize = 12;

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

#[inline(always)]
const fn staged(at: SharedPlan) -> (StageRun, SharedPlan) {
    at.tile_ring::<Bf16, 32, STAGE_N, Swizzle128B, DRAIN_WARPS>()
}

/// The shipped kernel's plan, byte for byte — the fp32 drain never touches the
/// staging run and declares it anyway, so the launch geometry is identical.
pub const SHARED_BYTES: usize = staged(shared(SharedPlan::sizing()).plan).1.bytes();

const _: () = {
    assert!(THREADS == 192 && MAX_CLUSTERS == 148);
    assert!(SHARED_BYTES == 114_816);
    assert!(Stage::BYTES == MnStage::BYTES);
    assert!(HALF_N == BLOCK_M);
};

/// This cluster's counter block.
#[derive(Clone, Copy)]
struct Clocks {
    at: *mut u64,
}

impl Clocks {
    /// # Safety
    /// `at` addresses [`COUNTERS`] live `u64`, and one lane of one warp of rank
    /// [`LEADER`] writes any given slot.
    #[inline(always)]
    unsafe fn put(self, slot: usize, ticks: u64) {
        unsafe { *self.at.add(slot) = ticks }
    }
}

#[derive(Clone, Copy)]
struct Release {
    sem: ClusterSemaphore,
    live: bool,
}

impl Release {
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

/// `optimized::Wide`, verbatim: one band live, released after the fourth load.
#[derive(Clone, Copy)]
struct Wide {
    c: GlobalRows<F32>,
}

impl Wide {
    #[inline(always)]
    unsafe fn drain<const DRAIN_ON: bool>(
        self,
        accumulator: Accumulator,
        row: u32,
        column: u32,
        release: Release,
    ) {
        unsafe {
            if !DRAIN_ON {
                release.now();
                return;
            }
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

#[derive(Clone, Copy)]
struct Tile {
    a_ring: Ring,
    b_ring: Ring,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    done: Semaphore,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    accumulator: Accumulator,
    out: Wide,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    transposed: bool,
    rank: u32,
    acc_free: Semaphore,
}

/// One role's tick totals, kept in registers across the item loop.
#[derive(Clone, Copy, Default)]
struct Sums {
    span: u32,
    first: u32,
    rest: u32,
}

impl Tile {
    /// The producer, with `free.wait_recycled` timed: `sums.span` is the whole
    /// span and `sums.rest` the back-pressure inside it.
    #[inline(always)]
    unsafe fn produce(&self, tile_m: u32, tile_n: u32, sums: &mut Sums) {
        unsafe {
            let opened = clock64();
            let a_line = (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank) as i32;
            let b_line = (BLOCK_N as u32 * tile_n + HALF_N as u32 * self.rank) as i32;
            let mut k = 0u32;
            while k < self.k_blocks {
                let waited = clock64();
                self.free.wait_recycled(k);
                sums.rest += clock64().wrapping_sub(waited) as u32;
                let stage = self.load.sem(k).at_rank(LEADER);
                let depth = (BLOCK_K as u32 * k) as i32;
                let (a, b) = (self.a_ring.tile(k), self.b_ring.tile(k));
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
            sums.span += clock64().wrapping_sub(opened) as u32;
        }
    }

    /// The MMA chain, with stage 0's wait (`sums.first` — the pipeline fill)
    /// kept apart from stages 1.. (`sums.rest` — the steady-state feed stall).
    /// `sums.span` is the whole multiply.
    #[inline(always)]
    unsafe fn multiply(&self, sums: &mut Sums) {
        unsafe {
            let opened = clock64();
            let mut k = 0u32;
            while k < self.k_blocks {
                let waited = clock64();
                self.load.wait(k);
                let stalled = clock64().wrapping_sub(waited) as u32;
                if k == 0 {
                    sums.first += stalled;
                } else {
                    sums.rest += stalled;
                }
                let (a, b) = (self.a_ring.tile(k), self.b_ring.tile(k));
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
            sums.span += clock64().wrapping_sub(opened) as u32;
        }
    }

    #[inline(always)]
    fn origin(&self, item: u32) -> (u32, u32) {
        let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, GROUP);
        (
            2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank + 32 * warp_id(),
            BLOCK_N as u32 * tile_n,
        )
    }

    #[inline(always)]
    unsafe fn arm(&self) {
        unsafe {
            self.load.init_all(1);
            self.free.init_all(1);
            self.done.init(1);
            self.acc_free.init(DRAIN_WARPS as u32 * CLUSTER_RANKS);
        }
    }

    #[inline(always)]
    unsafe fn disarm(&self) {
        unsafe {
            self.load.inval_all();
            self.free.inval_all();
            self.done.inval();
            self.acc_free.inval();
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `pipeline::run`'s loop, opened up so the item boundary can be timed
    /// apart from the work it separates. Structurally identical: leader-only
    /// inval-then-init, a proxy publish, the boundary, the item's roles, the
    /// tcgen05 fence, the boundary again. Returns the last item plus one, which
    /// is the deferred epilogue's cursor.
    ///
    /// # Safety
    /// As `pipeline::run`, plus `clocks` addressing this cluster's block.
    #[inline(always)]
    unsafe fn run<const DRAIN_ON: bool>(tile: &Tile, items: u32, clocks: Clocks) -> u32 {
        unsafe {
            let leader_thread = thread::threadIdx_x() == 0;
            let (warp, is_lane_0) = (warp_id(), lane() == 0);
            let mut initialized = false;
            let mut item = cluster::cluster_idx();
            let mut pending = 0u32;
            let mut ran = 0u64;

            let (mut pre, mut post, mut work) = (0u64, 0u64, 0u64);
            let mut acc = 0u64;
            let mut issue = Sums::default();
            let mut feed = Sums::default();
            let mut band = Sums::default();
            let opened = clock64();

            while item < items {
                let entered = clock64();
                if leader_thread {
                    if initialized {
                        tile.disarm();
                    }
                    tile.arm();
                    publish_to_async_proxy();
                }
                initialized = true;
                cluster::cluster_sync();
                let working = clock64();
                pre += working.wrapping_sub(entered);

                let (tile_m, tile_n) = pipeline::grouped(item, tile.tiles_m, tile.tiles_n, GROUP);
                if warp == PRODUCER {
                    if is_lane_0 {
                        tile.produce(tile_m, tile_n, &mut feed);
                    }
                } else if warp == ISSUER {
                    if tile.rank == LEADER && is_lane_0 {
                        let waited = clock64();
                        tile.acc_free.wait(0);
                        acc += clock64().wrapping_sub(waited);
                        tile.multiply(&mut issue);
                    }
                } else {
                    let release = Release {
                        sem: tile.acc_free.at_rank(LEADER),
                        live: true,
                    };
                    let drained = clock64();
                    if pending == 0 {
                        release.now();
                    } else {
                        let (row, column) = tile.origin(pending - 1);
                        tile.out
                            .drain::<DRAIN_ON>(tile.accumulator, row, column, release);
                    }
                    let stored = clock64();
                    band.span += stored.wrapping_sub(drained) as u32;
                    tile.done.wait(0);
                    band.rest += clock64().wrapping_sub(stored) as u32;
                }
                pending = item + 1;
                ran += 1;

                let worked = clock64();
                work += worked.wrapping_sub(working);
                tcgen05_fence_before_thread_sync();
                cluster::cluster_sync();
                post += clock64().wrapping_sub(worked);
                item += cluster::num_clusters();
            }
            if leader_thread && initialized {
                tile.disarm();
            }

            // One SM's clock backs every counter, so only rank `LEADER` writes,
            // and one lane of each role's warp owns its own slots.
            if tile.rank == LEADER && is_lane_0 {
                if warp == ISSUER {
                    clocks.put(ITEMS, ran);
                    clocks.put(PRE, pre);
                    clocks.put(POST, post);
                    clocks.put(WORK, work);
                    clocks.put(ACC, acc);
                    clocks.put(FILL, issue.first as u64);
                    clocks.put(FEED, issue.rest as u64);
                    clocks.put(MMA, issue.span as u64);
                    clocks.put(SPAN, clock64().wrapping_sub(opened));
                } else if warp == PRODUCER {
                    clocks.put(PROD, feed.span as u64);
                    clocks.put(FREE, feed.rest as u64);
                } else if warp == 0 {
                    clocks.put(DRAIN, band.span as u64);
                    clocks.put(DONE, band.rest as u64);
                }
            }
            pending
        }
    }

    /// # Safety
    /// As `gemm_tcgen05_f32_optimized`, plus `clocks` holding
    /// [`COUNTERS`]` * MAX_CLUSTERS` zeroed `u64`.
    #[inline(always)]
    unsafe fn probe<const DRAIN_ON: bool>(
        a_map: *const TmaDescriptor,
        b_map: *const TmaDescriptor,
        c: &mut DisjointSlice<f32>,
        clocks: &mut DisjointSlice<u64>,
        n: i32,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
        transposed: u32,
    ) {
        unsafe {
            let shared = shared(SharedPlan::attach());
            let tile = Tile {
                a_ring: shared.a_ring,
                b_ring: shared.b_ring,
                load: shared.load,
                free: shared.free,
                done: shared.done,
                a_map,
                b_map,
                accumulator: Accumulator::from_raw(alloc_cluster(shared.tmem_slot, BLOCK_N as u32)),
                out: Wide {
                    c: GlobalRows::<F32>::from_raw(c.as_mut_ptr() as *mut u8, n as usize),
                },
                tiles_m,
                tiles_n,
                k_blocks: k as u32 / BLOCK_K as u32,
                transposed: transposed != 0,
                rank: cluster::block_rank(),
                acc_free: shared.acc_free,
            };
            let at = clocks
                .as_mut_ptr()
                .add(COUNTERS * cluster::cluster_idx() as usize);
            let pending = run::<DRAIN_ON>(&tile, tiles_m * tiles_n, Clocks { at });
            if DRAIN_ON && pending != 0 && warp_id() < DRAIN_WARPS as u32 {
                let (row, column) = tile.origin(pending - 1);
                let release = Release {
                    sem: tile.acc_free.at_rank(LEADER),
                    live: false,
                };
                tile.out
                    .drain::<DRAIN_ON>(tile.accumulator, row, column, release);
            }
            tcgen05_fence_before_thread_sync();
            cluster::cluster_sync();
            dealloc_cluster(tile.accumulator.raw(), BLOCK_N as u32);
        }
    }

    /// The shipped fp32 store kernel with the stopwatch, computing the same
    /// `C`, plus a `clocks` block of [`COUNTERS`] `u64` per cluster.
    ///
    /// # Safety
    /// As [`probe`].
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
            probe::<true>(
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
    /// `no drain` floor, which writes no `C`.
    ///
    /// # Safety
    /// As [`probe`]; `c` is untouched.
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
            probe::<false>(
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
