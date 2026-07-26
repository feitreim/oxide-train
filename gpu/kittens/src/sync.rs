//! Semaphores over mbarrier intrinsics — the phase-parity idiom, first-class.
//!
//! cuda-oxide exposes no named barriers, so every cross-warp handoff in a
//! warp-specialized kernel is an mbarrier phase-parity spin (FA4 does the
//! same). The soundness rule these types encode: parity arithmetic works
//! because every barrier's completions lead its waiter by at most one phase —
//! each producer's next completion transitively requires the previous
//! consumer wait.
//!
//! A [`Semaphore`] is a stateless handle over one mbarrier word; a
//! [`SemaphoreRing`] owns the `index → (stage, parity)` arithmetic for the
//! `N`-deep producer/consumer rings that today thread parity bits by hand.

use cuda_device::barrier::{
    Barrier, mbarrier_arrive, mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_inval,
    mbarrier_try_wait_parity,
};

/// One mbarrier, addressed as the 64-bit state word it is. Handles are Copy
/// and carry no phase: parity comes from the caller's tile index (see
/// [`SemaphoreRing`]) so the same storage can back producer and consumer
/// handles at different pipeline positions.
#[derive(Clone, Copy)]
pub struct Semaphore {
    bar: *mut Barrier,
}

impl Semaphore {
    /// Wrap an mbarrier word (typically one slot of a `SharedArray<u64, N, 8>`
    /// static cast to `*mut Barrier`).
    ///
    /// # Safety
    ///
    /// `bar` must point to shared memory that lives as long as every use of
    /// the returned handle.
    #[inline(always)]
    pub const unsafe fn attach(bar: *mut Barrier) -> Self {
        Self { bar }
    }

    /// Initialize with `arriving` expected arrivals per phase. One thread
    /// per barrier, before any use, behind a block sync.
    ///
    /// # Safety
    ///
    /// Must not race any other access to the barrier.
    #[inline(always)]
    pub unsafe fn init(self, arriving: u32) {
        unsafe { mbarrier_init(self.bar, arriving) }
    }

    /// Invalidate before the block exits (or between persistent work items),
    /// wiping whatever unbalanced arrivals the phase left behind.
    ///
    /// # Safety
    ///
    /// No thread may still be arriving at or waiting on the barrier.
    #[inline(always)]
    pub unsafe fn inval(self) {
        unsafe { mbarrier_inval(self.bar) }
    }

    /// Count one arrival.
    ///
    /// # Safety
    ///
    /// The barrier must be initialized and this arrival balanced by the
    /// phase's expected count.
    #[inline(always)]
    pub unsafe fn arrive(self) {
        unsafe {
            mbarrier_arrive(self.bar);
        }
    }

    /// Count one arrival and register `bytes` of expected TMA transactions —
    /// the producer side of a `load → wait` handoff. The issuing thread
    /// charges every byte its `cp.async.bulk.tensor` calls will complete
    /// against this barrier.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::arrive`]; `bytes` must equal the bytes
    /// actually in flight or the phase never completes.
    #[inline(always)]
    pub unsafe fn expect_tx(self, bytes: u32) {
        unsafe {
            mbarrier_arrive_expect_tx(self.bar, 1, bytes);
        }
    }

    /// Spin until the barrier completes the phase with the given parity.
    ///
    /// # Safety
    ///
    /// The barrier must be initialized, and `parity` must follow the
    /// one-phase-lead rule from the module docs.
    #[inline(always)]
    pub unsafe fn wait(self, parity: u32) {
        unsafe { while !mbarrier_try_wait_parity(self.bar, parity) {} }
    }

    /// The raw barrier word, for intrinsics that consume one directly
    /// (`tcgen05_commit_shared_cluster`, TMA loads).
    #[inline(always)]
    pub const fn raw(self) -> *mut Barrier {
        self.bar
    }

    /// The barrier's cluster-multicast alias: a multicast TMA load completes
    /// on *each* receiving CTA's copy of the barrier, addressed by masking
    /// the issuing CTA's rank bit (and sub-word bits) out of the local
    /// address — the validated gemm cta_group::2 idiom. Pass the result to
    /// [`crate::shared::SharedTile::tma_load_2d_multicast_cg2`].
    #[inline(always)]
    pub fn multicast_alias(self) -> Semaphore {
        Semaphore {
            bar: ((self.bar as u32) & 0xFEFF_FFF8) as *mut Barrier,
        }
    }
}

/// `N` semaphores backing an `N`-stage pipeline ring, with the
/// `index → (stage, parity)` arithmetic in one place: tile `i` uses stage
/// `i % N`, whose barrier completes once per `N` tiles, so tile `i`'s
/// completion carries parity `(i / N) & 1`.
#[derive(Clone, Copy)]
pub struct SemaphoreRing<const N: usize> {
    base: *mut Barrier,
}

impl<const N: usize> SemaphoreRing<N> {
    /// Wrap `N` consecutive mbarrier words.
    ///
    /// # Safety
    ///
    /// `base` must point to `N` barrier words living as long as every use.
    #[inline(always)]
    pub const unsafe fn attach(base: *mut Barrier) -> Self {
        Self { base }
    }

    /// The semaphore of tile `index`'s stage.
    #[inline(always)]
    pub fn sem(self, index: u32) -> Semaphore {
        unsafe { Semaphore::attach(self.base.add(index as usize % N)) }
    }

    /// Consumer wait: spin until tile `index`'s stage completes its
    /// `(i / N) & 1` phase.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::wait`].
    #[inline(always)]
    pub unsafe fn wait(self, index: u32) {
        unsafe { self.sem(index).wait((index / N as u32) & 1) }
    }

    /// Producer wait for a recycled slot: a producer running the full `N`
    /// stages ahead of its consumer fills tile `index`'s stage only after the
    /// consumer's release from `N` tiles ago — the previous ring cycle, hence
    /// parity `(i / N - 1) & 1`. The first `N` tiles fill virgin slots and
    /// skip the wait.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::wait`]; the consumer must release this
    /// ring exactly once per tile consumed.
    #[inline(always)]
    pub unsafe fn wait_recycled(self, index: u32) {
        unsafe {
            if index as usize >= N {
                self.sem(index).wait((index / N as u32).wrapping_sub(1) & 1);
            }
        }
    }

    /// Initialize all `N` barriers with the same expected arrival count.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::init`].
    #[inline(always)]
    pub unsafe fn init_all(self, arriving: u32) {
        unsafe {
            let mut stage = 0u32;
            while (stage as usize) < N {
                self.sem(stage).init(arriving);
                stage += 1;
            }
        }
    }

    /// Invalidate all `N` barriers.
    ///
    /// # Safety
    ///
    /// Same contract as [`Semaphore::inval`].
    #[inline(always)]
    pub unsafe fn inval_all(self) {
        unsafe {
            let mut stage = 0u32;
            while (stage as usize) < N {
                self.sem(stage).inval();
                stage += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_stage_addressing_wraps() {
        // Pointer math only — no barrier is ever touched.
        let base = 0x100usize as *mut Barrier;
        let ring = unsafe { SemaphoreRing::<3>::attach(base) };
        assert_eq!(ring.sem(0).raw(), base);
        assert_eq!(ring.sem(4).raw(), unsafe { base.add(1) });
        assert_eq!(ring.sem(6).raw(), base);
    }
}
