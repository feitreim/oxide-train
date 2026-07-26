//! Persistent-grid harness (issue #61 phase 4) — the scaffold of
//! `flash_forward_persistent`, extracted.
//!
//! CTAs run a static strided work-item loop (`blockIdx.x`, stepping by
//! `gridDim.x`). Every mbarrier is re-initialized per item behind a block
//! sync, so each item's phase arithmetic starts from zero and unbalanced
//! arrivals (a consumer that never arrives for work the item didn't have)
//! are wiped, not threaded through parity math. ThunderKittens calls this
//! shape `prototype::lcf`; here the scaffold is [`run`] and the kernel
//! supplies a [`Job`].

use cuda_device::barrier::fence_proxy_async_shared_cta;
use cuda_device::tcgen05::tcgen05_fence_before_thread_sync;
use cuda_device::thread;

/// One persistent kernel's work, split at the points the scaffold owns.
/// Implementations are plain structs of tile/semaphore handles built once
/// before [`run`]; every method must be `#[inline(always)]` so the job
/// scalarizes into the kernel like the hand-written loop it replaces.
pub trait Job {
    /// (Re)initialize every mbarrier `item` will use — arrival counts may
    /// depend on the item's shape. Called on thread 0 only; [`run`] fences
    /// and syncs before any other thread touches the barriers.
    ///
    /// # Safety
    ///
    /// Same contract as [`crate::sync::Semaphore::init`].
    unsafe fn init(&self, item: u32);

    /// Invalidate the item's barrier set, wiping whatever unbalanced
    /// arrivals the finished item left. Thread 0 only.
    ///
    /// # Safety
    ///
    /// Same contract as [`crate::sync::Semaphore::inval`].
    unsafe fn inval(&self);

    /// One work item, entered by every thread of the block. Role dispatch
    /// (which warps produce, consume, issue MMAs) lives here.
    ///
    /// # Safety
    ///
    /// Kernel-specific: whatever the item's barrier protocol requires.
    unsafe fn work(&mut self, item: u32);
}

/// Run `job` over `items` work items on the static strided schedule. Owns
/// the barrier lifecycle (leader-only inval-then-init, published by a
/// proxy fence and a block sync) and the per-item tcgen05 fence pairing:
/// `work` returns with its MMAs committed, the harness fences them before
/// the item-boundary sync. TMEM allocation stays with the caller — it
/// spans items, not one.
///
/// # Safety
///
/// Every thread of the block must call this together, with `job`'s barrier
/// storage unused by anything else for the duration.
#[inline(always)]
pub unsafe fn run<J: Job>(job: &mut J, items: u32) {
    unsafe {
        let leader = thread::threadIdx_x() == 0;
        let mut initialized = false;
        let mut item = thread::blockIdx_x();
        while item < items {
            if leader {
                if initialized {
                    job.inval();
                }
                job.init(item);
                fence_proxy_async_shared_cta();
            }
            initialized = true;
            thread::sync_threads();
            job.work(item);
            tcgen05_fence_before_thread_sync();
            thread::sync_threads();
            item += thread::gridDim_x();
        }
        if leader && initialized {
            job.inval();
        }
    }
}
