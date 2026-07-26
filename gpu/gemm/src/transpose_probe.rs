//! tcgen05 descriptor-transpose probe (issue #53): one CTA, one MMA tile, a
//! host-chosen operand geometry.
//!
//! The weight-gradient GEMMs want `dW += dYᵀ·X` with both operands consumed in
//! their native row-major `[T, feature]` layout, i.e. both MN-major. The
//! instruction descriptor's `transpose_a`/`transpose_b` bits do exactly that,
//! but only `transpose_b` has ever run in this repo (flash-attn's `O = P·V`
//! and the gradient MMAs), and nothing anywhere sets `transpose_a`. This
//! module isolates the operand path from every kernel that would consume it:
//! TMA one A tile and one B tile, chain four `K = 16` MMAs, drain the fp32
//! TMEM accumulator straight to global memory, and let the host compare
//! against a CPU oracle.
//!
//! Everything about the operand walk is a launch parameter — the smem
//! descriptor's LBO/SBO/swizzle bits (pre-packed by the host as `a_layout` /
//! `b_layout`, address field zero), how many TMA loads stack into the tile,
//! and the byte step between `K = 16` chunks. One Modal run therefore sweeps a
//! table of candidate geometries instead of costing one run per guess, which
//! is what deferred this work in #43/#50.
//!
//! This compiles ONLY into `src/bin/transpose_probe.rs`, whose device artifact
//! ships as `transpose_probe.ptx` on the pure-PTX path. Alongside the tcgen05
//! geometry sweep it deliberately carries libdevice-backed math and a device
//! atomic regression probe. At cuda-oxide b099f64 those lowerings remain legal
//! in the same pure-PTX artifact as tcgen05; losing that property must fail this
//! harness before the model's single-artifact build can regress silently.
//!
//! Shape contract: `C[128, 64] = A[128, 64] · Bᵀ[64, 64]` over one `K = 64`
//! stage (four chained `K = 16` bf16 MMAs), `M128_N64` or `M64_N64`. `N = 64`
//! is deliberate: 64 bf16 elements are exactly one 128-byte `SWIZZLE_128B`
//! row, so a B operand fits one subtile in either layout.

use cuda_device::DisjointSlice;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive_expect_tx, mbarrier_init,
    mbarrier_inval, mbarrier_try_wait_parity,
};
use cuda_device::shared::SharedArray;
use cuda_device::tcgen05::{
    tcgen05_alloc, tcgen05_commit_shared_cluster, tcgen05_dealloc, tcgen05_ld_16x256b_pure,
    tcgen05_load_wait, tcgen05_mma_f16,
};
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

/// Accumulator rows the probe drains (the `M128` shapes fill all of them; the
/// `M64` cases fill rows 0..64 and the host ignores the rest).
pub const PROBE_M: usize = 128;
/// Accumulator columns — one 128-byte `SWIZZLE_128B` row of bf16.
pub const PROBE_N: usize = 64;
/// Reduction depth of the single stage.
pub const PROBE_K: usize = 64;
/// bf16 tcgen05 MMAs are `K = 16`, so one stage is four chained instructions.
const CHUNKS: u32 = (PROBE_K / 16) as u32;
/// MN elements in one 128-byte-row subtile; the second TMA load of a stacked
/// operand starts here along the tile's fast axis.
const SUBTILE_ELEMENTS: i32 = 64;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Libdevice regression gate for the unified tcgen05 artifact.
    #[kernel]
    pub unsafe fn libdevice_math_probe(input: &[f32], mut output: DisjointSlice<f32>) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if i < input.len() {
            let value = input[i];
            unsafe {
                *output.as_mut_ptr().add(i) =
                    value.sqrt() + value.exp() + value.ln() + value.max(0.5);
            }
        }
    }

    /// Exact contended sum: threads add the integers 1..=64 to one fp32 slot.
    #[kernel]
    pub unsafe fn device_atomic_probe(mut output: DisjointSlice<f32>) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if i < 64 {
            let slot = unsafe { DeviceAtomicF32::from_ptr(output.as_mut_ptr()) };
            slot.fetch_add((i + 1) as f32, AtomicOrdering::Relaxed);
        }
    }

    /// Fold a shared address into a host-packed layout word (LBO/SBO/swizzle
    /// bits, address field zero). Same encoding as `gemm::kernels`'.
    #[inline(always)]
    fn smem_descriptor(layout: u64, address: u64) -> u64 {
        layout | ((address >> 4) & 0x3fff)
    }

    /// Stack `tiles` TMA loads into one operand. A stacked operand advances
    /// `tile_bytes` in shared memory and one subtile along the map's fast
    /// axis, exactly like flash-attn's two-subtile `load_panel`.
    #[inline(always)]
    unsafe fn load_operand(
        destination: *mut u8,
        tma: *const TmaDescriptor,
        tiles: u32,
        tile_bytes: u32,
        barrier: *mut Barrier,
    ) {
        unsafe {
            let mut tile = 0u32;
            while tile < tiles {
                cp_async_bulk_tensor_2d_g2s(
                    destination.add((tile * tile_bytes) as usize),
                    tma,
                    tile as i32 * SUBTILE_ELEMENTS,
                    0,
                    barrier,
                );
                tile += 1;
            }
        }
    }

    /// One `C = A·Bᵀ` tile under a host-chosen operand geometry. Launch with
    /// grid `(1,1,1)`, 128 threads, no dynamic shared memory; `output` holds
    /// `PROBE_M * PROBE_N` fp32 elements.
    ///
    /// The shared scratch is far larger than the tiles (64 KiB for A, 32 KiB
    /// for B): a candidate LBO/SBO the hardware interprets differently than
    /// intended still lands inside the CTA's own shared window, so a wrong
    /// guess reads garbage and fails the oracle instead of faulting and
    /// killing the whole sweep.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub unsafe fn descriptor_transpose_probe(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_layout: u64,
        b_layout: u64,
        a_tiles: u32,
        a_tile_bytes: u32,
        a_chunk_step: u32,
        b_tiles: u32,
        b_tile_bytes: u32,
        b_chunk_step: u32,
        instruction: u32,
        mut output: DisjointSlice<f32>,
    ) {
        unsafe {
            static mut SMEM_A: SharedArray<u8, 65536, 128> = SharedArray::UNINIT;
            static mut SMEM_B: SharedArray<u8, 32768, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDRESS: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut MMA_BARRIER: Barrier = Barrier::UNINIT;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane = warp::lane_id();
            let is_leader = tid == 0;

            if is_leader {
                mbarrier_init(&raw mut TMA_BARRIER, 1);
                mbarrier_init(&raw mut MMA_BARRIER, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDRESS as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem = *(&raw const TMEM_ADDRESS as *const u32);

            let a_smem = &raw mut SMEM_A as *mut u8;
            let b_smem = &raw mut SMEM_B as *mut u8;
            if is_leader {
                load_operand(a_smem, a_tma, a_tiles, a_tile_bytes, &raw mut TMA_BARRIER);
                load_operand(b_smem, b_tma, b_tiles, b_tile_bytes, &raw mut TMA_BARRIER);
                mbarrier_arrive_expect_tx(
                    &raw const TMA_BARRIER,
                    1,
                    a_tiles * a_tile_bytes + b_tiles * b_tile_bytes,
                );
            }
            while !mbarrier_try_wait_parity(&raw const TMA_BARRIER, 0) {}
            thread::sync_threads();

            if is_leader {
                let mut chunk = 0u32;
                while chunk < CHUNKS {
                    tcgen05_mma_f16(
                        tmem,
                        smem_descriptor(a_layout, a_smem as u64 + (chunk * a_chunk_step) as u64),
                        smem_descriptor(b_layout, b_smem as u64 + (chunk * b_chunk_step) as u64),
                        instruction,
                        chunk > 0,
                    );
                    chunk += 1;
                }
                tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
            }
            while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, 0) {}
            thread::sync_threads();

            // Base-LDTM 16x256b fragment map, same as flash-attn's probes: for
            // each 16-row block this thread owns rows `lane/4` and `+8`, and
            // columns `2*(lane%4)` and `+1` of each 8-column half. All 128
            // lanes are drained; the host reads only the rows the shape fills.
            let quad = (lane % 4) as usize;
            let row_in_16 = (lane / 4) as usize;
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 4 {
                    let column = (column_block * 16) as usize;
                    let low = tcgen05_ld_16x256b_pure(tmem + (tmem_row << 16) + column as u32);
                    tcgen05_load_wait();
                    let high = tcgen05_ld_16x256b_pure(tmem + (tmem_row << 16) + column as u32 + 8);
                    tcgen05_load_wait();
                    let row_a = tmem_row as usize + row_in_16;
                    let row_b = row_a + 8;
                    let col = column + 2 * quad;
                    *output.get_unchecked_mut(row_a * PROBE_N + col) = low[0];
                    *output.get_unchecked_mut(row_a * PROBE_N + col + 1) = low[1];
                    *output.get_unchecked_mut(row_b * PROBE_N + col) = low[2];
                    *output.get_unchecked_mut(row_b * PROBE_N + col + 1) = low[3];
                    *output.get_unchecked_mut(row_a * PROBE_N + col + 8) = high[0];
                    *output.get_unchecked_mut(row_a * PROBE_N + col + 9) = high[1];
                    *output.get_unchecked_mut(row_b * PROBE_N + col + 8) = high[2];
                    *output.get_unchecked_mut(row_b * PROBE_N + col + 9) = high[3];
                    column_block += 1;
                }
                row_block += 1;
            }

            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(tmem, 512);
            }
            if is_leader {
                mbarrier_inval(&raw mut TMA_BARRIER);
                mbarrier_inval(&raw mut MMA_BARRIER);
            }
        }
    }
}
