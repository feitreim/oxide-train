//! Where a replicating TMA multicast's transaction bytes are counted
//! (oxide-train#80, ferro-kittens `shared.rs`'s open question).
//!
//! `SharedTile::tma_load_2d_multicast_cg2` takes an arbitrary `cta_mask` and
//! its documentation says the one thing nobody has established:
//!
//! > The charge handed back is **one tile** — what a single destination
//! > receives. A caller replicating into several CTAs owns the question of
//! > whether the one barrier it named sees that once or once per destination;
//! > it has not been answered against hardware.
//!
//! Every multicast in this repo and in cuda-oxide's own `gemm_sol_final` uses a
//! **one-bit** mask — the own-bit form, which replicates nothing and exists
//! only to reach a `.shared::cluster` barrier operand. So the replicating form
//! is unexercised, and a 4-CTA cluster that multicasts `A` between its two
//! `cta_group::2` pairs cannot be written until this is known.
//!
//! ## The question, and why the answer changes the kernel
//!
//! Two readings of the `[mbar]` operand are consistent with what is written
//! down:
//!
//! - **per destination** — the operand is an *offset*, and the copy that lands
//!   in CTA `d` completes on the barrier at that offset in `d`. This is how
//!   `tcgen05.commit.multicast` behaves (`commit_multicast_cg2` takes a plain
//!   local [`kittens::sync::Semaphore`], and every rank of the pair waits its
//!   own copy), and it is what CUTLASS's cluster GEMMs assume.
//! - **absolute** — the operand is a `.shared::cluster` address naming exactly
//!   one CTA's barrier, so a replicating copy completes there and nowhere else.
//!   This is the reading `tma_load_2d_arriving_at` is built on: it hands over a
//!   `mapa`'d *leader* address with the caller's own bit as the mask, and the
//!   leader really does get charged.
//!
//! Under the first, a 4-CTA cluster works: every CTA charges its own stage
//! barrier for exactly the bytes it receives and waits it, and the two ranks
//! whose `A` arrives without a leader behind it tell their leader through
//! [`kittens::sync::ClusterSemaphore::arrive`]. Under the second, a
//! replicating multicast can signal only one CTA, and every other destination
//! receives bytes that nothing accounts for — a silent hang, which is exactly
//! the failure mode oxide-train#80's split-chain probe spent 75 minutes in.
//!
//! ## The instrument
//!
//! Three 4-CTA clusters, one variant each, and **no variant can hang**: every
//! wait is against a `globaltimer` deadline, so a barrier that never flips
//! writes `completed = 0` and lives to report it. Each CTA also writes back the
//! first and last bf16 element of its staged tile, which separates *delivery*
//! from *accounting* — a rank holding the right bytes with `completed = 0`
//! says the mask worked and only the charge went elsewhere.
//!
//! | variant | who issues | mask | `[mbar]` | what a completion means |
//! |---|---|---|---|---|
//! | 0 | rank 0 | `{0, 2}` | rank 0's own | the offset is applied per destination |
//! | 1 | rank 1 | `{1, 3}` | rank 0's, `mapa`'d | an absolute address takes a replicating copy |
//! | 2 | rank 1 | `{1, 3}` | rank 0's, `mapa`'d | ...and is charged *twice*, once per destination |
//!
//! Variant 0 is the one the kernel needs. Variants 1 and 2 are the same launch
//! for free and pin down the other reading rather than leaving it inferred:
//! 1 charges one tile at rank 0, 2 charges two, and only one of them can
//! complete.
//!
//! Deliberately built out of the **raw** `cuda_device` intrinsics rather than
//! `kittens`' wrappers, for ferro's own reason: what is under test is the
//! instruction and its lowering, so putting a library wrapper in the path would
//! make a failure ambiguous between the two.

use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_inval,
    mbarrier_try_wait_parity,
};
use cuda_device::shared::SharedArray;
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s_multicast_cg2};
use cuda_device::{DisjointSlice, cluster, cluster_launch, debug, kernel, thread};
use cuda_host::cuda_module;

use kittens::sync::Semaphore;

/// Rows of the staged tile — one `A` half of the 4-CTA GEMM's pair tile.
pub const TILE_ROWS: usize = 128;
/// Columns, in bf16: one 128-byte `SWIZZLE_128B` atom, so the tile is a single
/// TMA box and the probe needs no subtile loop.
pub const TILE_COLS: usize = 64;
pub const TILE_BYTES: usize = TILE_ROWS * TILE_COLS * 2;
/// CTAs of the cluster under test.
pub const RANKS: u32 = 4;
/// Variants, one cluster each — see the module header.
pub const VARIANTS: u32 = 5;
/// `u64`s of the report one CTA writes:
/// `[entered, completed, first, last, charged, mask]`.
pub const FIELDS: usize = 6;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// One row of the report, per CTA. `entered` separates "waited and timed
    /// out" from "never ran", which a buffer of zeros cannot do on its own.
    #[inline(always)]
    unsafe fn record(out: &mut DisjointSlice<u64>, field: usize, value: u64) {
        unsafe {
            let base = FIELDS * thread::blockIdx_x() as usize;
            *out.get_unchecked_mut(base + field) = value;
        }
    }

    /// Ask the hardware where a replicating multicast's bytes are counted, and
    /// write down what came back.
    ///
    /// Launch with grid `(VARIANTS * RANKS, 1, 1)`, 32 threads, no dynamic
    /// shared memory; `out` holds `FIELDS * VARIANTS * RANKS` `u64`s. `map`
    /// describes a bf16 matrix of at least `RANKS * TILE_ROWS` rows and
    /// `TILE_COLS` columns under a `[TILE_ROWS, TILE_COLS]` `SWIZZLE_128B` box.
    ///
    /// # Safety
    /// `map` must describe a live matrix covering row `TILE_ROWS * 2`, and the
    /// launch must be shaped as above.
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub unsafe fn multicast_accounting_probe(
        map: *const TmaDescriptor,
        deadline_ns: u64,
        mut out: DisjointSlice<u64>,
    ) {
        unsafe {
            static mut TILE: SharedArray<u8, TILE_BYTES, 128> = SharedArray::UNINIT;
            static mut BAR: Barrier = Barrier::UNINIT;

            let rank = cluster::block_rank();
            let variant = cluster::cluster_idx();
            let leader = thread::threadIdx_x() == 0;
            let tile = &raw mut TILE as *mut u8;

            if leader {
                // Before anything can block: a CTA that timed out and a CTA
                // that never ran write the same zeros otherwise.
                record(&mut out, 0, 1);
                mbarrier_init(&raw mut BAR, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            // Every rank's barrier has to exist before any rank asks the
            // hardware to complete transactions on it.
            cluster::cluster_sync();

            // What this rank claims it will receive. A rank that charges
            // nothing still reports its tile, which is what separates delivery
            // from accounting.
            let tiles = match (variant, rank) {
                (0, 0) | (0, 2) => 1,
                (1, 0) => 1,
                (2, 0) => 2,
                (3, 0) | (3, 2) => 1,
                (4, 0) | (4, 2) => 2,
                _ => 0,
            };
            let charged = tiles * TILE_BYTES as u32;
            if leader && charged > 0 {
                mbarrier_arrive_expect_tx(&raw const BAR, 1, charged);
            }

            // Which ranks issue, and the row each fetches: an `A` half is
            // identified by its rows, so a destination holding rows
            // `TILE_ROWS..` proves the mask rather than the coordinate.
            // Variant 4 is the kernel's own arrangement — both halves at once,
            // both aimed at the even rank of the pair.
            let issues = match variant {
                0 => rank == 0,
                4 => rank < 2,
                _ => rank == 1,
            };
            if leader && issues {
                let mask = ((1u32 << rank) | (1u32 << (rank + 2))) as u16;
                let bar = if variant == 0 {
                    &raw mut BAR
                } else {
                    Semaphore::attach(&raw mut BAR).at_rank(0).raw()
                };
                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                    tile,
                    map,
                    0,
                    (TILE_ROWS as u32 * rank) as i32,
                    bar,
                    mask,
                );
                record(&mut out, 5, mask as u64);
            }

            // A deadline rather than the barrier, so a wrong answer is a row of
            // a table instead of a launch that never returns.
            let deadline = debug::globaltimer() + deadline_ns;
            let mut completed = 0u64;
            if charged > 0 {
                while debug::globaltimer() < deadline {
                    if mbarrier_try_wait_parity(&raw const BAR, 0) {
                        completed = 1;
                        break;
                    }
                }
            }
            thread::sync_threads();

            if leader {
                record(&mut out, 1, completed);
                record(&mut out, 2, *(tile as *const u16) as u64);
                record(
                    &mut out,
                    3,
                    *(tile.add(TILE_BYTES - 2) as *const u16) as u64,
                );
                record(&mut out, 4, charged as u64);
            }
            // The cluster may not retire while another rank's copy is still in
            // flight into this one's shared memory.
            cluster::cluster_sync();
            if leader {
                mbarrier_inval(&raw mut BAR);
            }
        }
    }
}
