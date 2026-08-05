//! The shipped tcgen05 GEMM with a stopwatch on every phase of an item
//! (oxide-train#80 forensics).
//!
//! `optimized.rs`'s fp32 store path, copied and instrumented rather than
//! parameterized: the shipped kernels keep no counters, no `clock64` and no
//! extra argument, and this module's [`probe`] reaches the device only through
//! the two entry points `optimized::kernels` declares for it — which nothing
//! but `src/bin/budget.rs` loads.
//!
//! ## What it answers
//!
//! `model_shapes` puts the shallow-K store rows 10–17 points under cuBLASLt
//! while their deep-K siblings reach 0.95–0.97, and the store-vs-fold
//! decomposition never isolated what shallow K pays extra. Per item the loop is
//!
//! and each role runs one uninterrupted loop: the band warps wait an item's
//! MMA out, drain it and release the accumulator to the next item; the issuer
//! waits that release and walks `k_blocks` MMAs each gated on its stage's TMA;
//! the producer issues those TMAs across item boundaries. Every one of those
//! waits is a candidate for the residual per-item term, and they are separated
//! here:
//!
//! | counter | phase |
//! |---|---|
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
use cuda_device::{DisjointSlice, cluster, thread, warp};

use kittens::global::{GlobalRows, store_rows};
use kittens::mma::{MmaShape, commit_multicast_cg2, mma_walk_cg2};
use kittens::pipeline;
use kittens::plan::SharedPlan;
use kittens::shared::{Bf16, F32, SharedTile, SharedTileRing, Swizzle128B, publish_to_async_proxy};
use kittens::sync::{ClusterSemaphore, SemaphoreRing};
use kittens::tmem::{TmemTile, alloc_cluster, dealloc_cluster};
use kittens::{BaseLdtm, RegTile, lane, warp_id};

pub const BLOCK_M: usize = 128;
pub const BLOCK_N: usize = 256;
const HALF_N: usize = BLOCK_N / 2;
/// Accumulator segments one allocation carries. The ping-pong is a *runtime*
/// choice here — `PING` selects one or two — so the before and the after are
/// the same binary at the same residency, ring depth and plan.
const SLOTS: u32 = 2;
/// The SM's whole tensor memory, at both arms: what pins residency at one CTA
/// an SM must not move between them, or the arms are not comparable.
const ACCUM_COLS: u32 = SLOTS * BLOCK_N as u32;
pub const BLOCK_K: usize = 64;
const CHUNKS: usize = BLOCK_K / 16;
const STAGES: usize = 6;
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
type Band = RegTile<16, BLOCK_N, BaseLdtm>;
const ISSUES: usize = BLOCK_N / 64;
type Accumulator = TmemTile<BLOCK_M, BLOCK_N>;

const SMS: u32 = 148;
const CTAS_PER_SM: u32 = 512 / ACCUM_COLS;
pub const MAX_CLUSTERS: u32 = SMS * CTAS_PER_SM / CLUSTER_RANKS;
pub const GROUP: u32 = 8;

/// Counters a cluster keeps, and the stride of the host's readback.
pub const COUNTERS: usize = 16;
/// Items this cluster ran — every other counter divides by it.
pub const ITEMS: usize = 0;
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

const ITEMS_DEEP: usize = 4;

struct Shared {
    a_ring: Ring,
    b_ring: Ring,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    full: SemaphoreRing<ITEMS_DEEP>,
    empty: SemaphoreRing<ITEMS_DEEP>,
    tmem_slot: *mut u32,
    plan: SharedPlan,
}

#[inline(always)]
const fn shared(at: SharedPlan) -> Shared {
    let (a_ring, at) = at.tile_ring::<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>();
    let (b_ring, at) = at.tile_ring::<Bf16, BLOCK_M, BLOCK_K, Swizzle128B, STAGES>();
    let (load, at) = at.semaphores::<STAGES>();
    let (free, at) = at.semaphores::<STAGES>();
    let (full, at) = at.semaphores::<ITEMS_DEEP>();
    let (empty, at) = at.semaphores::<ITEMS_DEEP>();
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

#[inline(always)]
const fn staged(at: SharedPlan) -> (StageRun, SharedPlan) {
    at.tile_ring::<Bf16, 32, STAGE_N, Swizzle128B, DRAIN_WARPS>()
}

/// The shipped kernel's plan, byte for byte — the fp32 drain never touches the
/// staging run and declares it anyway, so the launch geometry is identical.
pub const SHARED_BYTES: usize = staged(shared(SharedPlan::sizing()).plan).1.bytes();

const _: () = {
    assert!(THREADS == 192 && MAX_CLUSTERS == 74);
    assert!(SHARED_BYTES <= 232_448);
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
}

impl Release {
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
            let top: Band = accumulator.tile_x8_batched::<16, BLOCK_N, ISSUES>(band_row, 0);
            store_rows(self.c, row, column, lane, top);
            let bottom: Band = accumulator.tile_x8_batched::<16, BLOCK_N, ISSUES>(band_row + 16, 0);
            release.now();
            store_rows(self.c, row + 16, column, lane, bottom);
        }
    }
}

#[derive(Clone, Copy)]
struct Tile {
    a_ring: Ring,
    b_ring: Ring,
    load: SemaphoreRing<STAGES>,
    free: SemaphoreRing<STAGES>,
    a_map: *const TmaDescriptor,
    b_map: *const TmaDescriptor,
    accumulator: Accumulator,
    out: Wide,
    tiles_m: u32,
    tiles_n: u32,
    k_blocks: u32,
    transposed: bool,
    rank: u32,
    full: SemaphoreRing<ITEMS_DEEP>,
    empty: SemaphoreRing<ITEMS_DEEP>,
}

/// One role's tick totals, kept in registers across its loop.
#[derive(Clone, Copy, Default)]
struct Sums {
    span: u32,
    first: u32,
    rest: u32,
}

/// `optimized::Walk`, verbatim.
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

impl Tile {
    /// The producer, with `free.wait_recycled` timed: `sums.span` is the whole
    /// span and `sums.rest` the back-pressure inside it.
    #[inline(always)]
    unsafe fn produce(&self, mut walk: Walk, sums: &mut Sums) {
        unsafe {
            let opened = clock64();
            let mut stage_index = 0u32;
            while let Some(item) = walk.next() {
                let (tile_m, tile_n) = pipeline::grouped(item, self.tiles_m, self.tiles_n, GROUP);
                let a_line = (2 * BLOCK_M as u32 * tile_m + BLOCK_M as u32 * self.rank) as i32;
                let b_line = (BLOCK_N as u32 * tile_n + HALF_N as u32 * self.rank) as i32;
                let mut k = 0u32;
                while k < self.k_blocks {
                    let waited = clock64();
                    self.free.wait_recycled(stage_index);
                    sums.rest += clock64().wrapping_sub(waited) as u32;
                    let stage = self.load.sem(stage_index).at_rank(LEADER);
                    let depth = (BLOCK_K as u32 * k) as i32;
                    let (a, b) = (self.a_ring.tile(stage_index), self.b_ring.tile(stage_index));
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
                            .sem(stage_index)
                            .expect_tx(bytes.across_ranks(CLUSTER_RANKS));
                    }
                    k += 1;
                    stage_index += 1;
                }
            }
            sums.span += clock64().wrapping_sub(opened) as u32;
        }
    }

    /// The MMA chains, with each item's `empty` wait in `acc`, its stage-0 wait
    /// (the pipeline fill) in `sums.first`, its stages 1.. in `sums.rest`, and
    /// the multiply itself in `sums.span`.
    #[inline(always)]
    unsafe fn multiply(&self, mut walk: Walk, slots: u32, acc: &mut u64, sums: &mut Sums) {
        unsafe {
            let mut sequence = 0u32;
            let mut stage_index = 0u32;
            while walk.next().is_some() {
                let waited = clock64();
                self.empty.wait(sequence);
                let target = self.slot(sequence, slots);
                let opened = clock64();
                *acc += opened.wrapping_sub(waited);
                let mut k = 0u32;
                while k < self.k_blocks {
                    let waited = clock64();
                    self.load.wait(stage_index);
                    let stalled = clock64().wrapping_sub(waited) as u32;
                    if k == 0 {
                        sums.first += stalled;
                    } else {
                        sums.rest += stalled;
                    }
                    let (a, b) = (self.a_ring.tile(stage_index), self.b_ring.tile(stage_index));
                    let (a_walk, b_walk) = if self.transposed {
                        (
                            MnStage::from_raw(a.base()).mn_walk(),
                            MnStage::from_raw(b.base()).mn_walk(),
                        )
                    } else {
                        (a.k_walk(), b.k_walk())
                    };
                    mma_walk_cg2::<Bf16, CHUNKS>(
                        target.raw(),
                        a_walk,
                        b_walk,
                        MmaShape::M256_N256,
                        k > 0,
                    );
                    commit_multicast_cg2(self.free.sem(stage_index), PAIR);
                    k += 1;
                    stage_index += 1;
                }
                commit_multicast_cg2(self.full.sem(sequence), PAIR);
                sequence += 1;
                sums.span += clock64().wrapping_sub(opened) as u32;
            }
        }
    }

    /// The epilogue, with the wait for the item's MMA in `sums.rest` and the
    /// drain itself in `sums.span`.
    #[inline(always)]
    unsafe fn epilogue<const DRAIN_ON: bool>(&self, mut walk: Walk, slots: u32, sums: &mut Sums) {
        unsafe {
            if lane() == 0 {
                let mut slot = 0u32;
                while slot < slots {
                    self.empty.sem(slot).at_rank(LEADER).arrive();
                    slot += 1;
                }
            }
            let mut sequence = 0u32;
            while let Some(item) = walk.next() {
                let waited = clock64();
                self.full.wait(sequence);
                let opened = clock64();
                sums.rest += opened.wrapping_sub(waited) as u32;
                let (row, column) = self.origin(item);
                let release = Release {
                    sem: self.empty.sem(sequence + slots).at_rank(LEADER),
                };
                self.out
                    .drain::<DRAIN_ON>(self.slot(sequence, slots), row, column, release);
                sums.span += clock64().wrapping_sub(opened) as u32;
                sequence += 1;
            }
        }
    }

    /// The accumulator segment `sequence` owns. At `slots == 1` every item
    /// takes the same one and the kernel is the deep ring without a ping-pong;
    /// at `SLOTS` consecutive items alternate.
    #[inline(always)]
    fn slot(&self, sequence: u32, slots: u32) -> Accumulator {
        self.accumulator
            .columns_right(BLOCK_N as u32 * (sequence % slots))
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
            if thread::threadIdx_x() == 0 {
                self.load.init_all(1);
                self.free.init_all(1);
                self.full.init_all(1);
                self.empty.init_all(DRAIN_WARPS as u32 * CLUSTER_RANKS);
                publish_to_async_proxy();
            }
            cluster::cluster_sync();
        }
    }

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
}

/// The shipped schedule with a stopwatch on each role.
///
/// # Safety
/// As `gemm_tcgen05_f32_optimized`, plus `clocks` holding [`COUNTERS`] zeroed
/// `u64` per cluster.
#[inline(always)]
pub unsafe fn probe<const DRAIN_ON: bool, const PING: bool>(
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
            a_map,
            b_map,
            accumulator: Accumulator::from_raw(alloc_cluster(shared.tmem_slot, ACCUM_COLS)),
            out: Wide {
                c: GlobalRows::<F32>::from_raw(c.as_mut_ptr() as *mut u8, n as usize),
            },
            tiles_m,
            tiles_n,
            k_blocks: k as u32 / BLOCK_K as u32,
            transposed: transposed != 0,
            rank: cluster::block_rank(),
            full: shared.full,
            empty: shared.empty,
        };
        tile.arm();

        let items = tiles_m * tiles_n;
        let walk = Walk::open(items);
        let ran = {
            let mut counted = Walk::open(items);
            let mut n = 0u64;
            while counted.next().is_some() {
                n += 1;
            }
            n
        };
        let slots = if PING { SLOTS } else { 1 };
        let (warp, is_lane_0) = (warp_id(), lane() == 0);
        let mut acc = 0u64;
        let mut issue = Sums::default();
        let mut feed = Sums::default();
        let mut band = Sums::default();
        let opened = clock64();
        if warp == PRODUCER {
            if is_lane_0 {
                tile.produce(walk, &mut feed);
            }
        } else if warp == ISSUER {
            if tile.rank == LEADER && is_lane_0 {
                tile.multiply(walk, slots, &mut acc, &mut issue);
            }
        } else {
            tile.epilogue::<DRAIN_ON>(walk, slots, &mut band);
        }
        let span = clock64().wrapping_sub(opened);

        // One SM's clock backs every counter, so only rank `LEADER` writes, and
        // one lane of each role's warp owns its own slots.
        if tile.rank == LEADER && is_lane_0 && ran != 0 {
            let at = Clocks {
                at: clocks
                    .as_mut_ptr()
                    .add(COUNTERS * cluster::cluster_idx() as usize),
            };
            if warp == ISSUER {
                at.put(ITEMS, ran);
                at.put(ACC, acc / ran);
                at.put(FILL, issue.first as u64 / ran);
                at.put(FEED, issue.rest as u64 / ran);
                at.put(MMA, issue.span as u64 / ran);
                at.put(SPAN, span / ran);
            } else if warp == PRODUCER {
                at.put(PROD, feed.span as u64 / ran);
                at.put(FREE, feed.rest as u64 / ran);
            } else if warp == 0 {
                at.put(DRAIN, band.span as u64 / ran);
                at.put(DONE, band.rest as u64 / ran);
            }
        }

        tile.retire();
        dealloc_cluster(tile.accumulator.raw(), ACCUM_COLS);
    }
}
