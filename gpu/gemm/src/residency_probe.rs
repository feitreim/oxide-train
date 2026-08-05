//! How many CTAs of a persistent cluster launch are *actually* co-resident.
//!
//! `MAX_CLUSTERS` is derived from per-CTA resources — `512 / ACCUM_COLS` tensor
//! memory columns, then shared memory, then registers — and the whole
//! grouped-walk and wave arithmetic in [`super::optimized`] assumes all 148
//! clusters (296 CTAs, two an SM) run at once. `cuOccupancyMaxActiveClusters`
//! disagrees: it reports **74** for that shape, which would make the persistent
//! wave two sequential waves and reinterpret every wave-quantization number
//! oxide-train#80 has recorded.
//!
//! The driver's answer is a model. This is the measurement. Every CTA reads
//! `%smid`, holds for a fixed **wall-clock** interval — `%globaltimer`, the one
//! clock every SM shares, since `clock64` is per-SM and cannot order two CTAs
//! against each other — and reports its entry and exit. Three things then fall
//! out of one launch:
//!
//! - **Wall time.** A hold of `h` over a grid the device runs in one wave takes
//!   `h`; over a grid it runs in two takes `2h`. That is the whole question,
//!   and it needs no timestamp arithmetic at all.
//! - **The packing.** Distinct `%smid`s, and how many CTAs each SM hosted.
//! - **The overlap.** The largest set of CTAs whose hold intervals intersect,
//!   swept over the timeline — the direct count of what was resident at once.
//!
//! The probe carries the shipped kernel's *resources* rather than its work: the
//! launch declares [`super::SHARED_BYTES`] (the driver reserves a launch's
//! dynamic shared memory whether or not a thread reads it), the block is
//! [`super::optimized::THREADS`] wide, the cluster is the shipped
//! `cluster_launch(2, 1, 1)`, and each CTA takes `columns` of its own SM's 512
//! tensor-memory columns exactly as the accumulator does. What it does *not*
//! carry is the shipped kernel's 168 registers, so it can only ever report
//! residency at or above the real kernel's — which is why `bin/residency.rs`
//! also times the **shipped** kernel across grid widths, where the register
//! count is the real one.

use cuda_device::debug::globaltimer;
use cuda_device::{DisjointSlice, cluster, thread};

use kittens::plan::SharedPlan;
use kittens::tmem::{alloc_block, alloc_cluster, dealloc_block, dealloc_cluster};

/// `u64`s a CTA reports: its SM, its cluster, and the wall-clock interval it
/// held that SM for.
pub const SLOTS: usize = 4;

/// Which allocator a probe drives for its tensor-memory columns.
///
/// The shipped kernel's is `cta_group::2`, whose two ranks are charged
/// `columns` against *their own* SM's 512 — so the per-SM charge is the same
/// either way, and the wider cluster takes the block-scoped form because a
/// 4-CTA cluster is not a pair.
#[derive(Clone, Copy, PartialEq)]
pub enum Columns {
    Pair(u32),
    Block(u32),
}

/// # Safety
///
/// - Every thread of every CTA of the cluster calls this together, once.
/// - `out` holds [`SLOTS`] `u64` per CTA of the grid.
/// - The launch declares enough dynamic shared memory for a
///   [`SharedPlan::tmem_slot`], and `columns` is within one SM's 512 at the
///   residency being probed.
#[inline(always)]
pub unsafe fn hold(out: &mut DisjointSlice<u64>, hold_ns: u64, columns: Columns) {
    unsafe {
        let (slot, _) = SharedPlan::attach().tmem_slot();
        let address = match columns {
            Columns::Pair(columns) => alloc_cluster(slot, columns),
            Columns::Block(columns) => alloc_block(slot, columns),
        };

        // Entry is read after the allocation, so a CTA that had to *wait* for
        // tensor memory reports the interval it actually owned the SM's
        // resources — which is the interval residency is about.
        let entry = globaltimer();
        while globaltimer().wrapping_sub(entry) < hold_ns {}
        let exit = globaltimer();

        if thread::threadIdx_x() == 0 {
            let at = SLOTS * thread::blockIdx_x() as usize;
            let report = out.as_mut_ptr().add(at);
            report.write(thread::smid() as u64);
            report.add(1).write(cluster::cluster_idx() as u64);
            report.add(2).write(entry);
            report.add(3).write(exit);
        }

        // No tcgen05 read was issued, so the fence the drain needs has nothing
        // to retire; the rendezvous is still owed, because a peer must not free
        // the pair's columns while the other rank is still inside the hold.
        thread::sync_threads();
        match columns {
            Columns::Pair(columns) => {
                cluster::cluster_sync();
                dealloc_cluster(address, columns)
            }
            Columns::Block(columns) => dealloc_block(address, columns),
        }
    }
}
