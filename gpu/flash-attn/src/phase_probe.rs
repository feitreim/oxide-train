//! The two shipped flash backward kernels with a stopwatch on every phase of a
//! key-tile visit (oxide-train's flash-backward forensics front).
//!
//! `tcgen05.rs`'s `flash_backward_q` / `flash_backward_kv`, copied and
//! instrumented rather than parameterized: the shipped kernels keep no
//! counters, no `clock64` and no extra argument, and this module reaches the
//! device only through the two entry points `tcgen05::kernels` declares for it
//! — which nothing but `src/bin/probe.rs` loads. This is a child module of
//! `tcgen05` so that it sees the plans, the tile types and the tile contract
//! without any of them becoming crate surface.
//!
//! ## What it answers
//!
//! The bench puts the pair at 2.66 ms where the forward is 0.832, and the
//! *issued* tensor-core work is 3 and 4 `[128, 64, 128]`-equivalent MMAs per
//! key-tile visit against the forward's 2. Divide the measured milliseconds by
//! the issued flops and all three kernels land near 300 TFLOP/s — an eighth of
//! the B200's dense bf16 peak, and a fifth of what this repo's own GEMMs
//! reach. Neither the MMA nor HBM explains that (the same arithmetic puts the
//! pair at ~25% of 8 TB/s), so the time is in between and nothing so far has
//! said *where*.
//!
//! Both kernels run one uninterrupted loop over key tiles (kernel A) or query
//! tiles (kernel B), with two roles inside one 128-thread warpgroup: the leader
//! issues TMA and MMA ahead of the pass, and all four warps run the register
//! pass and then meet at a block barrier. Every wait in that loop is a
//! candidate, and they are separated here:
//!
//! | counter | phase |
//! |---|---|
//! | [`FEED`] | the leader waiting the streamed operand stage before the score MMAs |
//! | [`RECYCLE`] | the leader waiting the gradient MMA that frees a stage, before refilling it |
//! | [`ISSUE`] | the leader's own MMA issue, feed and recycle excluded |
//! | [`SCORED`] | the warpgroup waiting the score MMAs of *this* tile — exposed tensor-core latency |
//! | [`GRAD`] | the warpgroup waiting the gradient MMA that frees the gradient operand slot |
//! | [`STAT`] | the per-tile statistic loads (`load_row_vec` per item, `load_col_vec` per chunk) |
//! | [`PASS`] | the register pass — mask, `exp2`, the `dP − D` product, the packed store |
//! | [`SYNC`] | the proxy fence and the one block barrier per tile |
//! | [`EPI`] | the trailing accumulator wait and the `store_rows` drain, per item |
//! | [`SPAN`] | first `clock64` to last, over the whole persistent loop |
//!
//! Every counter but [`SPAN`], [`EPI`] and [`ITEMS`] is divided by the CTA's
//! **key-tile visits**, so the columns are directly comparable between the two
//! kernels and across shapes; [`EPI`] is divided by items, since it runs once
//! per item. `FEED + RECYCLE + ISSUE` is the leader's loop and
//! `SCORED + GRAD + STAT + PASS + SYNC` is the warpgroup's, and the two roles
//! are the same threads at different times — the leader's terms are what it
//! does *before* joining the pass, so the visit's cost is the warpgroup column
//! plus whatever of the leader's column the other three warps wait out at
//! [`SYNC`].
//!
//! Ticks are `clock64` on the CTA's own SM, so every counter of a CTA shares
//! one clock. Reading it perturbs: the probe's own wall time is reported beside
//! the shipped kernel's so the inflation is on the record rather than assumed
//! away.

use cuda_device::DisjointSlice;
use cuda_device::barrier::fence_proxy_async_shared_cta;
use cuda_device::debug::clock64;
use cuda_device::tcgen05::{tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{cluster, thread, warp};

use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor,
};
use kittens::global::{GlobalRows, load_col_vec, load_row_vec, store_rows};
use kittens::ldst::store_tile;
use kittens::mma::{self, mma_ab, mma_abt};
use kittens::pipeline;
use kittens::plan::SharedPlan;
use kittens::reg::Exp2Hw;
use kittens::shared::{F32, MmaElement};
use kittens::tmem::{alloc_block, dealloc_block, warp_lanes};

use super::*;

// `tcgen05::kernels`' own scale constants, which live inside that module and so
// cannot be reached from here. Same literals, and the probe would be measuring
// different arithmetic if they ever stopped being.
const LN2: f32 = 0.693_147_18;
const SCALE: f32 = 0.088_388_35;
const LOG2E: f32 = 1.442_695_04;

/// `u64` counters one CTA owns.
pub const COUNTERS: usize = 24;

/// Key-tile (kernel A) or query-tile (kernel B) visits this CTA made.
pub const VISITS: usize = 0;
/// Work items this CTA ran.
pub const ITEMS: usize = 1;
/// The leader waiting the streamed operand stage its next score MMA reads.
pub const FEED: usize = 2;
/// The leader waiting the gradient MMA that frees the stage it is about to
/// refill.
pub const RECYCLE: usize = 3;
/// The leader's MMA issue itself.
pub const ISSUE: usize = 4;
/// The warpgroup waiting this tile's score MMAs.
pub const SCORED: usize = 5;
/// The warpgroup waiting the gradient MMA that frees the gradient operand slot
/// this tile's pass writes.
pub const GRAD: usize = 6;
/// The saved-statistic loads.
pub const STAT: usize = 7;
/// The register pass.
pub const PASS: usize = 8;
/// The **pass warps** at the proxy fence and the block barrier — how long
/// warps 0-3 wait for the issue warp, and so how much of the tile the issue
/// warp is the critical path for.
pub const SYNC: usize = 9;
/// The **issue warp** at the same barrier — the mirror of [`SYNC`]. One of the
/// two is near zero and the other is the slack; which one says whether the
/// tile is issue-bound or pass-bound.
pub const IDLE: usize = 12;
/// The trailing drain, per item.
pub const EPI: usize = 10;
/// First `clock64` to last.
pub const SPAN: usize = 11;

// --- The issue warp's [`ISSUE`], split by what it is issuing (kernel A). ---
// `TMA + SMMA + GMMA == ISSUE`, so the split is a partition and not a sample.
/// The issue warp's TMA issue — `load_kv`'s two `tma_load`s and the
/// `expect_tx` that charges them, plus the item head's four `tma_load_at`.
pub const TMA: usize = 13;
/// The issue warp's score-MMA issue: the two `mma_abt` walks (eight chained
/// `tcgen05.mma` each, `M128_N64`) and their shared commit.
pub const SMMA: usize = 14;
/// The issue warp's gradient-MMA issue: `dQ += dS·K`, one `mma_ab` walk of two
/// `N=64` bands × four K chunks, and its commit.
pub const GMMA: usize = 15;

// --- The pass warps' [`PASS`], split the same way (kernel A). ---
// `TREAD + ARITH + SSTORE == PASS`, with `ARITH` by subtraction.
/// The pass warps inside `TmemTile::tile` — the `tcgen05.ld` of a
/// `SCORE_CHUNK` of the score band and of the gradient band, each of which is
/// four `.16x256b` issues with a `tcgen05.wait::ld` after every one of them.
pub const TREAD: usize = 16;
/// The pass warps inside `store_tile` — the bf16 convert and the swizzled
/// shared-memory store of the finished dS chunk.
pub const SSTORE: usize = 17;
/// The pass warps **between** items: everything `pipeline::run` does around
/// `work` — the two block boundaries, the leader's barrier inval/init and the
/// proxy fence. The `residual` the report derives, measured rather than
/// subtracted.
pub const GAP: usize = 18;
/// The pass warps' *first* `scored.wait` of an item — the pipeline fill, which
/// the per-visit [`SCORED`] mean otherwise smears across every visit.
pub const FILL: usize = 19;
/// The half of [`EPI`] that is the wait for the item's last gradient MMA. It
/// has to happen before the item boundary, because `pipeline::run` re-arms the
/// barrier set there; the rest of [`EPI`] does not, which is the whole of the
/// deferred-drain question.
pub const DRAINWAIT: usize = 20;

/// Per-CTA tick sums, all in one struct so the instrumented loops stay legible.
#[derive(Clone, Copy, Default)]
struct Sums {
    feed: u64,
    recycle: u64,
    issue: u64,
    scored: u64,
    grad: u64,
    stat: u64,
    pass: u64,
    sync: u64,
    idle: u64,
    epi: u64,
    visits: u64,
    items: u64,
    tma: u64,
    smma: u64,
    gmma: u64,
    tread: u64,
    sstore: u64,
    gap: u64,
    fill: u64,
    drainwait: u64,
    /// The pass warps' last `clock64` of the previous item, so [`GAP`] can be
    /// measured across `pipeline::run`'s own per-item work instead of being
    /// derived by subtracting the columns from the span.
    ended: u64,
}

/// This CTA's counter block.
#[derive(Clone, Copy)]
struct Clocks {
    at: *mut u64,
}

impl Clocks {
    /// # Safety
    /// `at` addresses [`COUNTERS`] live `u64`, written by one lane of one warp
    /// of this CTA.
    #[inline(always)]
    unsafe fn put(self, slot: usize, ticks: u64) {
        unsafe { *self.at.add(slot) = ticks }
    }

    /// The pass warps' column, from warp 0. Owns [`VISITS`], [`ITEMS`] and
    /// [`SPAN`] because it is the role that runs the item to its end.
    ///
    /// # Safety
    /// As [`Self::put`]; `sums.visits` and `sums.items` are non-zero.
    #[inline(always)]
    unsafe fn write_pass(self, sums: &Sums, span: u64) {
        unsafe {
            let (visits, items) = (sums.visits.max(1), sums.items.max(1));
            self.put(VISITS, sums.visits);
            self.put(ITEMS, sums.items);
            self.put(SCORED, sums.scored / visits);
            self.put(GRAD, sums.grad / visits);
            self.put(STAT, sums.stat / visits);
            self.put(PASS, sums.pass / visits);
            self.put(SYNC, sums.sync / visits);
            self.put(EPI, sums.epi / items);
            self.put(SPAN, span / items);
            self.put(TREAD, sums.tread / visits);
            self.put(SSTORE, sums.sstore / visits);
            self.put(GAP, sums.gap / items);
            self.put(FILL, sums.fill / items);
            self.put(DRAINWAIT, sums.drainwait / items);
        }
    }

    /// The issue warp's column. Disjoint slots from [`Self::write_pass`], so
    /// the two roles publish into one CTA block without a race.
    ///
    /// # Safety
    /// As [`Self::write_pass`].
    #[inline(always)]
    unsafe fn write_issue(self, sums: &Sums) {
        unsafe {
            let visits = sums.visits.max(1);
            self.put(FEED, sums.feed / visits);
            self.put(RECYCLE, sums.recycle / visits);
            self.put(ISSUE, sums.issue / visits);
            self.put(IDLE, sums.idle / visits);
            self.put(TMA, sums.tma / visits);
            self.put(SMMA, sums.smma / visits);
            self.put(GMMA, sums.gmma / visits);
        }
    }
}

/// `BackwardQStream` with the stopwatch. Every field and every line of `work`
/// is the shipped kernel's; the `clock64` pairs and `sums` are the diff.
struct BackwardQProbe {
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
    scores: STmem,
    gradients: STmem,
    accumulator: AccTmem,
    lse: GlobalRows<F32>,
    dot: GlobalRows<F32>,
    dq: GlobalRows<F32>,
    sums: Sums,
}

impl BackwardQProbe {
    #[inline(always)]
    fn buffer(&self, segment: STmem, key_tile: u32) -> STmem {
        if key_tile & 1 == 0 {
            segment
        } else {
            segment.columns_right(TILE as u32)
        }
    }

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

    #[inline(always)]
    unsafe fn score_mmas(&self, key_tile: u32) {
        unsafe {
            mma_abt(
                self.buffer(self.scores, key_tile).raw(),
                self.shared.q,
                self.shared.k.tile(key_tile),
                SCORE_SHAPE,
                false,
            );
            mma_abt(
                self.buffer(self.gradients, key_tile).raw(),
                self.shared.dy,
                self.shared.v.tile(key_tile),
                SCORE_SHAPE,
                false,
            );
            mma::commit(self.shared.scored.sem(key_tile));
        }
    }
}

impl pipeline::Job for BackwardQProbe {
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
            let query_block = self.tiles - 1 - item / self.planes;
            let plane = item % self.planes;
            let head = plane % self.h;
            let batch = plane / self.h;
            let query_base = query_block * QUERIES as u32;
            let key_tiles = 2 * query_block + 2;
            let first_masked = 2 * query_block;

            let (leader, warp_id, lane) = (self.leader, self.warp_id, self.lane);
            let band = warp_lanes();
            let group = warp_id / DRAIN_WARPS;
            let issuing = warp_id == ISSUE_WARP;

            if leader {
                tcgen05_fence_after_thread_sync();
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
                // The item's pipeline fill, charged to FEED like every other
                // wait on a streamed stage.
                let waited = clock64();
                full.wait(0);
                self.shared.kv_loaded.wait(0);
                let opened = clock64();
                self.sums.feed += opened.wrapping_sub(waited);
                self.score_mmas(0);
                let issued = clock64().wrapping_sub(opened);
                self.sums.issue += issued;
                self.sums.smma += issued;
            }

            if issuing {
                let mut key_tile = 0u32;
                while key_tile < key_tiles {
                    if lane == 0 {
                        let refill = key_tile + BACKWARD_STAGES as u32 - 1;
                        if key_tile > 0 && refill < key_tiles {
                            let waited = clock64();
                            self.shared.accumulated.wait(key_tile - 1);
                            let opened = clock64();
                            self.sums.recycle += opened.wrapping_sub(waited);
                            self.load_kv(refill, plane as i32);
                            let issued = clock64().wrapping_sub(opened);
                            self.sums.issue += issued;
                            self.sums.tma += issued;
                        }
                        if key_tile + 1 < key_tiles {
                            let waited = clock64();
                            self.shared.kv_loaded.wait(key_tile + 1);
                            let opened = clock64();
                            self.sums.feed += opened.wrapping_sub(waited);
                            self.score_mmas(key_tile + 1);
                            let issued = clock64().wrapping_sub(opened);
                            self.sums.issue += issued;
                            self.sums.smma += issued;
                        }
                    }
                    let at = clock64();
                    tcgen05_fence_before_thread_sync();
                    thread::sync_threads();
                    let synced = clock64();
                    self.sums.idle += synced.wrapping_sub(at);
                    if lane == 0 {
                        tcgen05_fence_after_thread_sync();
                        mma_ab(
                            self.accumulator.raw(),
                            self.shared.ds.tile(key_tile),
                            self.shared.k.tile(key_tile),
                            OUTPUT_SHAPE,
                            key_tile != 0,
                        );
                        mma::commit(self.shared.accumulated.sem(key_tile));
                        let issued = clock64().wrapping_sub(synced);
                        self.sums.issue += issued;
                        self.sums.gmma += issued;
                    }
                    self.sums.visits += 1;
                    key_tile += 1;
                }
            } else {
                let waited = clock64();
                if self.sums.items > 0 {
                    self.sums.gap += waited.wrapping_sub(self.sums.ended);
                }
                let row = batch * self.t + query_base + band;
                let mut lse2: Rows = load_row_vec(self.lse, row, head, lane);
                lse2.scale_assign(LOG2E);
                let dots: Rows = load_row_vec(self.dot, row, head, lane);
                self.sums.stat += clock64().wrapping_sub(waited);

                let mut key_tile = 0u32;
                while key_tile < key_tiles {
                    let waited = clock64();
                    self.shared.scored.wait(key_tile);
                    let scored_at = clock64();
                    self.sums.scored += scored_at.wrapping_sub(waited);
                    if key_tile == 0 {
                        self.sums.fill += scored_at.wrapping_sub(waited);
                    }
                    if key_tile >= GRADIENT_STAGES as u32 {
                        self.shared
                            .accumulated
                            .wait(key_tile - GRADIENT_STAGES as u32);
                    }
                    let opened = clock64();
                    self.sums.grad += opened.wrapping_sub(scored_at);

                    let scores = self.buffer(self.scores, key_tile);
                    let gradients = self.buffer(self.gradients, key_tile);
                    let key_base = key_tile * TILE as u32;
                    let masked = key_tile >= first_masked;
                    let ds = self.shared.ds.tile(key_tile).chunk_writer();

                    let mut column = group * PASS_COLUMNS;
                    let last = column + PASS_COLUMNS;
                    while column < last {
                        // Five stops rather than one, because the question this
                        // loop is asked is which of its three kinds of work the
                        // pass floor is: the `tcgen05.ld` pair, the register
                        // arithmetic between them, or the swizzled store.
                        let read_from = clock64();
                        let mut dscore: ScoreChunk = scores.tile(band, column);
                        let scored_read = clock64();
                        if masked {
                            dscore.make_causal_at(
                                lane,
                                query_base + band,
                                key_base + column,
                                MASKED_SCORE,
                            );
                        }
                        dscore.sub_row_assign(lse2);
                        dscore.unary_map_assign::<Exp2Hw>();
                        let grad_from = clock64();
                        let mut dp: ScoreChunk = gradients.tile(band, column);
                        let grad_read = clock64();
                        dp.sub_row_assign(dots);
                        dscore.mul_assign(dp);
                        dscore.scale_assign(SCALE);
                        let store_from = clock64();
                        store_tile(ds, band, column, lane, dscore);
                        let stored = clock64();
                        self.sums.tread +=
                            scored_read.wrapping_sub(read_from) + grad_read.wrapping_sub(grad_from);
                        self.sums.sstore += stored.wrapping_sub(store_from);
                        column += SCORE_CHUNK as u32;
                    }
                    let passed = clock64();
                    self.sums.pass += passed.wrapping_sub(opened);

                    fence_proxy_async_shared_cta();
                    tcgen05_fence_before_thread_sync();
                    thread::sync_threads();
                    self.sums.sync += clock64().wrapping_sub(passed);
                    self.sums.visits += 1;
                    key_tile += 1;
                }

                let waited = clock64();
                self.shared.accumulated.wait(key_tiles - 1);
                let drained = clock64();
                let columns = group * DRAIN_COLUMNS as u32;
                let dq: OutHalf = self.accumulator.tile_x8(band, columns);
                store_rows(self.dq, row, head * HD as u32 + columns, lane, dq);
                let ended = clock64();
                self.sums.epi += ended.wrapping_sub(waited);
                self.sums.drainwait += drained.wrapping_sub(waited);
                self.sums.ended = ended;
            }
            self.sums.items += 1;
        }
    }
}

/// `flash_backward_q` with the stopwatch.
///
/// # Safety
/// As `tcgen05::kernels::flash_backward_q`, plus `clocks` holding [`COUNTERS`]
/// zeroed `u64` per CTA of the grid.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn backward_q(
    q_tma: *const TmaDescriptor,
    k_tma: *const TmaDescriptor,
    v_tma: *const TmaDescriptor,
    dy_tma: *const TmaDescriptor,
    logsumexp: &[f32],
    dot: &[f32],
    sequence_length: u32,
    heads: u32,
    batches: u32,
    dq: &mut DisjointSlice<f32>,
    clocks: &mut DisjointSlice<u64>,
) {
    unsafe {
        if thread::blockDim_x() as usize != FLASH_BACKWARD_BLOCK {
            return;
        }
        let shared = backward_q_plan(SharedPlan::attach());
        let tmem = alloc_block(shared.tmem_slot, BACKWARD_TMEM_COLUMNS);
        let scores = STmem::from_raw(tmem);
        let gradients = STmem::from_raw(tmem + 2 * TILE as u32);
        let accumulator = AccTmem::from_raw(tmem + 4 * TILE as u32);

        let tiles = sequence_length / QUERIES as u32;
        let planes = heads * batches;
        let stats = heads as usize;
        let mut job = BackwardQProbe {
            q_tma,
            k_tma,
            v_tma,
            dy_tma,
            t: sequence_length,
            h: heads,
            tiles,
            planes,
            // The issue warp's lane 0, as the shipped kernels have it.
            // #96 moved `leader` there and left the probe's at thread 0, so
            // until now the instrument had a pass warp doing the item head's
            // TMA loads and first score MMA. Per item rather than per visit,
            // which is why it stayed inside the probe's 1% overhead band —
            // and still not what ships.
            leader: thread::threadIdx_x() == ISSUE_WARP * 32,
            warp_id: warp::warp_id(),
            lane: warp::lane_id(),
            shared,
            scores,
            gradients,
            accumulator,
            lse: GlobalRows::<F32>::from_raw(logsumexp.as_ptr().cast_mut().cast(), stats),
            dot: GlobalRows::<F32>::from_raw(dot.as_ptr().cast_mut().cast(), stats),
            dq: GlobalRows::<F32>::from_slice(dq, heads as usize * HD),
            sums: Sums::default(),
        };
        let opened = clock64();
        pipeline::run(&mut job, tiles * planes);
        let span = clock64().wrapping_sub(opened);
        report(clocks, &job.sums, span, job.warp_id, job.lane);
        dealloc_block(tmem, BACKWARD_TMEM_COLUMNS);
    }
}

/// `BackwardKvStream` with the stopwatch.
struct BackwardKvProbe {
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
    scores: STmem,
    gradients: STmem,
    dv_acc: AccTmem,
    dk_acc: AccTmem,
    lse: GlobalRows<F32>,
    dot: GlobalRows<F32>,
    dk: GlobalRows<F32>,
    dv: GlobalRows<F32>,
    sums: Sums,
}

impl BackwardKvProbe {
    #[inline(always)]
    fn buffer(&self, segment: STmem, step: u32) -> STmem {
        if step & 1 == 0 {
            segment
        } else {
            segment.columns_right(TILE as u32)
        }
    }

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

    #[inline(always)]
    unsafe fn score_mmas(&self, step: u32) {
        unsafe {
            mma_abt(
                self.buffer(self.scores, step).raw(),
                self.shared.k,
                self.shared.q.tile(step),
                SCORE_SHAPE,
                false,
            );
            mma_abt(
                self.buffer(self.gradients, step).raw(),
                self.shared.v,
                self.shared.dy.tile(step),
                SCORE_SHAPE,
                false,
            );
            mma::commit(self.shared.scored.sem(step));
        }
    }
}

impl pipeline::Job for BackwardKvProbe {
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
            let key_block = item / self.planes;
            let plane = item % self.planes;
            let head = plane % self.h;
            let batch = plane / self.h;
            let key_base = key_block * QUERIES as u32;
            let steps = 2 * (self.tiles - key_block);

            let (leader, warp_id, lane) = (self.leader, self.warp_id, self.lane);
            let band = warp_lanes();
            let group = warp_id / DRAIN_WARPS;
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
                let waited = clock64();
                full.wait(0);
                self.shared.qdy_loaded.wait(0);
                let opened = clock64();
                self.sums.feed += opened.wrapping_sub(waited);
                self.score_mmas(0);
                self.sums.issue += clock64().wrapping_sub(opened);
            }

            if issuing {
                let mut step = 0u32;
                while step < steps {
                    if lane == 0 {
                        let refill = step + BACKWARD_STAGES as u32 - 1;
                        if step > 0 && refill < steps {
                            let waited = clock64();
                            self.shared.accumulated.wait(step - 1);
                            let opened = clock64();
                            self.sums.recycle += opened.wrapping_sub(waited);
                            self.load_qdy(refill, key_base + refill * TILE as u32, plane as i32);
                            self.sums.issue += clock64().wrapping_sub(opened);
                        }
                        if step + 1 < steps {
                            let waited = clock64();
                            self.shared.qdy_loaded.wait(step + 1);
                            let opened = clock64();
                            self.sums.feed += opened.wrapping_sub(waited);
                            self.score_mmas(step + 1);
                            self.sums.issue += clock64().wrapping_sub(opened);
                        }
                    }
                    let at = clock64();
                    tcgen05_fence_before_thread_sync();
                    thread::sync_threads();
                    let synced = clock64();
                    self.sums.idle += synced.wrapping_sub(at);
                    if lane == 0 {
                        tcgen05_fence_after_thread_sync();
                        let accumulate = step != 0;
                        mma_ab(
                            self.dv_acc.raw(),
                            self.shared.p.tile(step),
                            self.shared.dy.tile(step),
                            OUTPUT_SHAPE,
                            accumulate,
                        );
                        mma_ab(
                            self.dk_acc.raw(),
                            self.shared.ds.tile(step),
                            self.shared.q.tile(step),
                            OUTPUT_SHAPE,
                            accumulate,
                        );
                        mma::commit(self.shared.accumulated.sem(step));
                        self.sums.issue += clock64().wrapping_sub(synced);
                    }
                    self.sums.visits += 1;
                    step += 1;
                }
            } else {
                let row = batch * self.t + key_base + band;
                let mut step = 0u32;
                while step < steps {
                    let query_base = key_base + step * TILE as u32;
                    let waited = clock64();
                    self.shared.scored.wait(step);
                    let scored_at = clock64();
                    self.sums.scored += scored_at.wrapping_sub(waited);
                    if step >= GRADIENT_STAGES as u32 {
                        self.shared.accumulated.wait(step - GRADIENT_STAGES as u32);
                    }
                    let opened = clock64();
                    self.sums.grad += opened.wrapping_sub(scored_at);

                    let scores = self.buffer(self.scores, step);
                    let gradients = self.buffer(self.gradients, step);
                    let masked = step < 2;
                    let p = self.shared.p.tile(step).chunk_writer();
                    let ds = self.shared.ds.tile(step).chunk_writer();

                    // The statistic is per streamed query, so it is read a
                    // chunk at a time inside the pass rather than once at the
                    // item head — which is the one structural difference from
                    // kernel A's loop, and the reason STAT is separated from
                    // PASS on both sides.
                    let mut stat = 0u64;
                    let mut column = group * PASS_COLUMNS;
                    let last = column + PASS_COLUMNS;
                    while column < last {
                        let query = batch * self.t + query_base + column;
                        let at = clock64();
                        let mut lse2: Cols = load_col_vec(self.lse, query, head, lane);
                        lse2.scale_assign(LOG2E);
                        let dots: Cols = load_col_vec(self.dot, query, head, lane);
                        stat += clock64().wrapping_sub(at);

                        let mut probability: ScoreChunk = scores.tile(band, column);
                        if masked {
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
                        probability.scale_assign(LN2);
                        store_tile(ds, band, column, lane, probability);
                        column += SCORE_CHUNK as u32;
                    }
                    let passed = clock64();
                    self.sums.stat += stat;
                    self.sums.pass += passed.wrapping_sub(opened) - stat;

                    fence_proxy_async_shared_cta();
                    tcgen05_fence_before_thread_sync();
                    thread::sync_threads();
                    self.sums.sync += clock64().wrapping_sub(passed);
                    self.sums.visits += 1;
                    step += 1;
                }

                let waited = clock64();
                self.shared.accumulated.wait(steps - 1);
                let columns = group * DRAIN_COLUMNS as u32;
                let dv: OutHalf = self.dv_acc.tile_x8(band, columns);
                store_rows(self.dv, row, head * HD as u32 + columns, lane, dv);
                let dk: OutHalf = self.dk_acc.tile_x8(band, columns);
                store_rows(self.dk, row, head * HD as u32 + columns, lane, dk);
                self.sums.epi += clock64().wrapping_sub(waited);
            }
            self.sums.items += 1;
        }
    }
}

/// `flash_backward_kv` with the stopwatch.
///
/// # Safety
/// As `tcgen05::kernels::flash_backward_kv`, plus `clocks` holding
/// [`COUNTERS`] zeroed `u64` per CTA of the grid.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn backward_kv(
    q_tma: *const TmaDescriptor,
    k_tma: *const TmaDescriptor,
    v_tma: *const TmaDescriptor,
    dy_tma: *const TmaDescriptor,
    logsumexp: &[f32],
    dot: &[f32],
    sequence_length: u32,
    heads: u32,
    batches: u32,
    dk: &mut DisjointSlice<f32>,
    dv: &mut DisjointSlice<f32>,
    clocks: &mut DisjointSlice<u64>,
) {
    unsafe {
        if thread::blockDim_x() as usize != FLASH_BACKWARD_BLOCK {
            return;
        }
        let shared = backward_kv_plan(SharedPlan::attach());
        let tmem = alloc_block(shared.tmem_slot, BACKWARD_TMEM_COLUMNS);
        let scores = STmem::from_raw(tmem);
        let gradients = STmem::from_raw(tmem + 2 * TILE as u32);
        let dv_acc = AccTmem::from_raw(tmem + 4 * TILE as u32);
        let dk_acc = AccTmem::from_raw(tmem + 4 * TILE as u32 + HD as u32);

        let tiles = sequence_length / QUERIES as u32;
        let planes = heads * batches;
        let stats = heads as usize;
        let mut job = BackwardKvProbe {
            q_tma,
            k_tma,
            v_tma,
            dy_tma,
            t: sequence_length,
            h: heads,
            tiles,
            planes,
            // The issue warp's lane 0, as the shipped kernels have it.
            // #96 moved `leader` there and left the probe's at thread 0, so
            // until now the instrument had a pass warp doing the item head's
            // TMA loads and first score MMA. Per item rather than per visit,
            // which is why it stayed inside the probe's 1% overhead band —
            // and still not what ships.
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
            dk: GlobalRows::<F32>::from_slice(dk, heads as usize * HD),
            dv: GlobalRows::<F32>::from_slice(dv, heads as usize * HD),
            sums: Sums::default(),
        };
        let opened = clock64();
        pipeline::run(&mut job, tiles * planes);
        let span = clock64().wrapping_sub(opened);
        report(clocks, &job.sums, span, job.warp_id, job.lane);
        dealloc_block(tmem, BACKWARD_TMEM_COLUMNS);
    }
}

/// Publish this CTA's counters — the pass warps' column from warp 0, the issue
/// warp's from warp [`ISSUE_WARP`], into disjoint slots of one block.
///
/// The two roles are now two warps (issue #94), so no single thread holds both
/// halves and neither can report the other's. What each *can* report is its own
/// wait at the shared barrier: the pass warps' [`SYNC`] against the issue
/// warp's [`IDLE`], which is the tile's critical path stated twice from
/// opposite sides.
///
/// # Safety
/// `clocks` holds [`COUNTERS`] `u64` per CTA of the grid, and each slot is
/// written by one thread alone.
#[inline(always)]
unsafe fn report(clocks: &mut DisjointSlice<u64>, sums: &Sums, span: u64, warp_id: u32, lane: u32) {
    unsafe {
        if lane != 0 || sums.items == 0 {
            return;
        }
        let at = Clocks {
            at: clocks
                .as_mut_ptr()
                .add(COUNTERS * cluster::cluster_idx() as usize),
        };
        if warp_id == 0 {
            at.write_pass(sums, span);
        } else if warp_id == ISSUE_WARP {
            at.write_issue(sums);
        }
    }
}

// ---------------------------------------------------------------------------
// The MMA cadence probe: what one `tcgen05.mma` costs the issuing warp, and
// how that cost moves with `N`.
// ---------------------------------------------------------------------------
//
// The phase budget says kernel A's issue warp spends ~1 700 ticks a visit
// pushing 24 `M128_N64` instructions — ~70 ticks each against a ~26-tick
// full-rate figure. Two readings fit that number and they want opposite
// remedies:
//
// - **Back-pressure.** The tensor core is busy and the issuing thread stalls
//   on a full MMA queue. Then the cost is per *unit of work*, a wider `N`
//   halves it for the same instruction count, and a second issue warp buys
//   nothing.
// - **Issue overhead.** Each instruction costs the front end a fixed toll
//   whatever its shape. Then the cost is per *instruction*, a wider `N` is
//   free work, and a second issuer would help too.
//
// This kernel separates them by holding everything but `N` fixed: the same
// eight chained K=16 MMAs over the same `[128, 128]` A operand, with a B
// operand of 64, 128 and 256 rows. Under back-pressure the three cost the same
// per unit of work; under a fixed toll they cost the same per instruction. It
// also runs kernel A's gradient MMA both ways — `mma_ab`'s two `N=64` bands
// against one `N=128` band off the same tile's `mn_walk` — which is the same
// question asked at the shape a remedy would actually take.

/// Variants the cadence probe times.
pub const CADENCE_VARIANTS: usize = 5;
/// `u64` per variant: sustained issue ticks, sustained ticks to retirement,
/// one drained chain's round trip, and the instructions the variant issued.
pub const CADENCE_SLOTS: usize = 4;
/// Counters the cadence probe writes.
pub const CADENCE_COUNTERS: usize = CADENCE_VARIANTS * CADENCE_SLOTS;

/// Bytes the cadence probe's shared plan takes: the two operand panels, plus
/// room for the completion ring and the TMEM staging word. Rounded up rather
/// than carved exactly — one CTA of one warp is not competing for an SM.
pub const CADENCE_SMEM: usize = QUERIES * HD * 2 + 256 * HD * 2 + 1024;

/// `dQ += dS·K` the way `kittens::mma::mma_ab` walked it before
/// ferro-kittens#194: one `N = 64` instruction per stacked `B` subtile, into
/// `tmem + 64 * subtile`.
///
/// Kept here, and only here, because the cadence table's point is the
/// *comparison* — the shipped walk is the wide one now, and a table with one
/// row is not a measurement of anything.
///
/// # Safety
///
/// As `kittens::mma::mma_ab`, plus: `shape` is `M128_N64` and `tmem` owns
/// `64 * B::SUBTILES` fp32 columns.
#[inline(always)]
unsafe fn gradient_mma_banded<const AR: usize, const K: usize, const N: usize>(
    tmem: u32,
    a: SharedTile<Bf16, AR, K, Swizzle128B>,
    b: SharedTile<Bf16, K, N, Swizzle128B>,
    accumulate: bool,
) {
    unsafe {
        let instruction = Tcgen05InstructionDescriptor::builder()
            .shape(MmaShape::M128_N64)
            .element_type(Tcgen05ElementType::BF16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .transpose_a(false)
            .transpose_b(true)
            .build()
            .raw();
        let mut band = 0usize;
        while band < SharedTile::<Bf16, K, N, Swizzle128B>::SUBTILES {
            let base = band * SharedTile::<Bf16, K, N, Swizzle128B>::SUBTILE_BYTES;
            let mut chunk = 0usize;
            while chunk < K / 16 {
                // A is K-major and K spans one swizzle atom at this shape, so
                // its chunks step 32 bytes along the row; B steps 16 rows of
                // the 128-byte atom.
                Bf16::mma(
                    tmem + (band as u32) * 64,
                    a.operand_descriptor(chunk * 32),
                    b.operand_descriptor(base + chunk * 16 * 128),
                    instruction,
                    accumulate || chunk > 0,
                );
                chunk += 1;
            }
            band += 1;
        }
    }
}

/// One warp's timing of `rounds` chained walks, written into `clocks` at
/// `variant`.
///
/// A macro and not a function: the walk is a different call at every variant
/// and the device backend gets no closures out of this file.
///
/// The pipe is drained before the sustained window opens, so the window is the
/// steady state and the trailing wait is what the last chain still owed.
macro_rules! time_walk {
    ($clocks:expr, $variant:expr, $instructions:expr, $rounds:expr, $sem:expr, $at:expr, $walk:expr) => {{
        // A drained pipe first: one chain, committed and waited, so the
        // sustained window below opens with nothing outstanding and this
        // number is the shape's own latency.
        let opened = clock64();
        $walk;
        mma::commit($sem.sem($at));
        $sem.wait($at);
        let latency = clock64().wrapping_sub(opened);

        let started = clock64();
        let mut round = 0u32;
        while round < $rounds {
            $walk;
            round += 1;
        }
        let issued = clock64();
        mma::commit($sem.sem($at + 1));
        $sem.wait($at + 1);
        let retired = clock64();

        let out = $clocks.as_mut_ptr().add($variant * CADENCE_SLOTS);
        *out = issued.wrapping_sub(started);
        *out.add(1) = retired.wrapping_sub(started);
        *out.add(2) = latency;
        *out.add(3) = ($instructions as u64) * $rounds as u64;
    }};
}

/// Time one `tcgen05.mma` chain per variant, from a single warp of a single
/// CTA — the body of `tcgen05::kernels::mma_cadence`.
///
/// # Safety
///
/// One CTA of 32 threads, `CADENCE_SMEM` dynamic shared bytes, and `clocks`
/// holding [`CADENCE_COUNTERS`] zeroed `u64`.
#[inline(always)]
pub unsafe fn mma_cadence(rounds: u32, clocks: &mut DisjointSlice<u64>) {
    unsafe {
        let at = SharedPlan::attach();
        let (a, at) = at.tile::<Bf16, QUERIES, HD, Swizzle128B>();
        let (b, at) = at.tile::<Bf16, 256, HD, Swizzle128B>();
        let (sem, at) = at.semaphores::<16>();
        let (tmem_slot, _) = at.tmem_slot();

        // Operand *values* are irrelevant to a cadence, but a shared tile that
        // was never written can hold anything, and "anything" includes bit
        // patterns whose timing is not the timing of a number. Zero both.
        let words = a.base().cast::<u32>();
        let total = (QUERIES * HD + 256 * HD) / 2;
        let mut i = thread::threadIdx_x() as usize;
        while i < total {
            *words.add(i) = 0;
            i += 32;
        }

        let tmem = alloc_block(tmem_slot, BACKWARD_TMEM_COLUMNS);
        if thread::threadIdx_x() == 0 {
            sem.init_all(1);
        }
        fence_proxy_async_shared_cta();
        thread::sync_threads();

        if thread::threadIdx_x() == 0 {
            // The score MMA's shape at three widths. Same A, same eight K=16
            // chunks, same instruction count — only the B operand's row count
            // and the accumulator band move.
            let b64 = SharedTile::<Bf16, TILE, HD, Swizzle128B>::from_raw(b.base());
            let b128 = SharedTile::<Bf16, QUERIES, HD, Swizzle128B>::from_raw(b.base());
            time_walk!(
                clocks,
                0,
                8,
                rounds,
                sem,
                0,
                mma_abt(tmem, a, b64, MmaShape::M128_N64, true)
            );
            time_walk!(
                clocks,
                1,
                8,
                rounds,
                sem,
                2,
                mma_abt(tmem, a, b128, MmaShape::M128_N128, true)
            );
            time_walk!(
                clocks,
                2,
                8,
                rounds,
                sem,
                4,
                mma_abt(tmem, a, b, MmaShape::M128_N256, true)
            );

            // The gradient MMA's shape both ways — `[QUERIES, TILE]` dS
            // against a `[TILE, HD]` K panel — as two `N = 64` bands and as
            // the one `N = HD` band the shipped walk is since
            // ferro-kittens#194.
            let ds = SharedTile::<Bf16, QUERIES, TILE, Swizzle128B>::from_raw(a.base());
            let k = SharedTile::<Bf16, TILE, HD, Swizzle128B>::from_raw(b.base());
            time_walk!(
                clocks,
                3,
                8,
                rounds,
                sem,
                6,
                gradient_mma_banded(tmem, ds, k, true)
            );
            time_walk!(
                clocks,
                4,
                4,
                rounds,
                sem,
                8,
                mma_ab(tmem, ds, k, MmaShape::M128_N128, true)
            );
        }

        tcgen05_fence_before_thread_sync();
        thread::sync_threads();
        dealloc_block(tmem, BACKWARD_TMEM_COLUMNS);
        if thread::threadIdx_x() == 0 {
            sem.inval_all();
        }
    }
}
