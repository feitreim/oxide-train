/*
 * The tcgen05 path is adapted from cuda-oxide's Apache-2.0-licensed
 * `examples/gemm_sol` at the repository's pinned v0.2.1 tag.
 */

//! GEMM performance ladder for the B200 training target.
//!
//! The portable first rung is an fp32 shared-memory/register-tiled kernel.
//! The Blackwell rung consumes row-major bf16 `A[M,K]` and transposed,
//! row-major bf16 `B[N,K]`. Its B200 path uses cta_group::2 CTA-pair UMMA,
//! four TMA stages, and a fp32 TMEM accumulator to produce 256x256 output tiles,
//! one cluster per tile (exact-cover, no CLC work-stealing). Packed-bf16/fp32
//! overwrite and accumulation modes share that compute pipeline, so backward
//! parameter gradients need no separate add.
//!
//! bf16 is represented as raw `u16`/packed `u32` until milestone 7d adds the
//! shared `Element` plumbing.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use cuda_device::barrier::{Barrier, fence_proxy_async_shared_cta, mbarrier_arrive_cluster};
use cuda_device::cluster;
use cuda_device::shared::SharedArray;
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
    cvt_f32x2_bf16x2, tcgen05_alloc, tcgen05_alloc_cg2, tcgen05_dealloc, tcgen05_dealloc_cg2,
    tcgen05_relinquish_alloc_permit_cg2,
};
use cuda_device::tma::TmaDescriptor;
use cuda_device::{DisjointSlice, cluster_launch, kernel, thread, warp};
use cuda_host::cuda_module;
use kittens::ldst::stmatrix_m8n8_x2;
use kittens::mma;
use kittens::shared::{Bf16, SharedTile, Swizzle128B};
use kittens::sync::{Semaphore, SemaphoreRing};
use kittens::tmem::TmemTile;

pub mod fp32;
pub use fp32::{BK, BM, BN, TM, TN, launch_config as fp32_launch_config};
pub mod host;
pub use host::{
    Bf16PairsTmaMap, Bf16TmaMap, TC_BK, TC_K_PIPELINE, TC_M_TILE, TC_N_TILE, TC_TILE, Tcgen05Gemm,
    TmaLayout, create_bf16_pairs_tma_map, create_bf16_pairs_tma_map_prefix, create_bf16_tma_map,
    tcgen05_launch_config,
};

/// A `[TC_TILE, TC_BK]` K-major bf16 operand stage: one swizzle subtile,
/// four chained K=16 MMA chunks.
type KStage = SharedTile<Bf16, TC_TILE, TC_BK, Swizzle128B>;

/// The same stage bytes read MN-major (`transpose_a`/`transpose_b` set):
/// `[TC_BK, TC_TILE]` as two stacked 64-wide subtiles, K along the rows.
type MnStage = SharedTile<Bf16, TC_BK, TC_TILE, Swizzle128B>;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[inline(always)]
    pub(super) fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    #[inline(always)]
    fn f32_to_bf16_rne(value: f32) -> u16 {
        let bits = value.to_bits();
        let round = 0x7fffu32 + ((bits >> 16) & 1);
        (bits.wrapping_add(round) >> 16) as u16
    }

    #[inline(always)]
    pub(super) fn accumulate_bf16_pair(current: u32, update: u32) -> u32 {
        let lo = bf16_to_f32(current as u16) + bf16_to_f32(update as u16);
        let hi = bf16_to_f32((current >> 16) as u16) + bf16_to_f32((update >> 16) as u16);
        f32_to_bf16_rne(lo) as u32 | ((f32_to_bf16_rne(hi) as u32) << 16)
    }

    /// Blackwell TMA + tcgen05 GEMM.
    ///
    /// One CTA computes a 128x128 tile. TMA loads swizzled 128x64 bf16 tiles;
    /// four `tcgen05` MMAs consume each K tile and accumulate in fp32 TMEM.
    #[inline(always)]
    unsafe fn gemm_tcgen05_bf16_impl<const ACCUMULATE: bool>(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut output: DisjointSlice<u32>,
        n: u32,
        k: u32,
    ) {
        unsafe {
            static mut SMEM_A: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDRESS: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut MMA_BARRIER: Barrier = Barrier::UNINIT;

            const TILE_BYTES: u32 = (TC_TILE * TC_BK * 2) as u32;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;
            let is_leader = tid == 0;
            let tile_m = thread::blockIdx_x();
            let tile_n = thread::blockIdx_y();

            let tma_sem = Semaphore::attach(&raw mut TMA_BARRIER);
            let mma_sem = Semaphore::attach(&raw mut MMA_BARRIER);
            let a_tile = KStage::from_raw(&raw mut SMEM_A as *mut u8);
            let b_tile = KStage::from_raw(&raw mut SMEM_B as *mut u8);

            if is_leader {
                tma_sem.init(1);
                mma_sem.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDRESS as *mut u32, 512);
            }
            thread::sync_threads();
            let accumulator =
                TmemTile::<TC_TILE, TC_TILE>::from_raw(*(&raw const TMEM_ADDRESS as *const u32));

            let instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let mut k_tile = 0u32;
            while k_tile < k / TC_BK as u32 {
                let phase = k_tile & 1;
                if is_leader {
                    let k_offset = (k_tile * TC_BK as u32) as i32;
                    a_tile.tma_load_2d(a_tma, k_offset, (tile_m * TC_TILE as u32) as i32, tma_sem);
                    b_tile.tma_load_2d(b_tma, k_offset, (tile_n * TC_TILE as u32) as i32, tma_sem);
                    tma_sem.expect_tx(TILE_BYTES * 2);
                }

                tma_sem.wait(phase);
                thread::sync_threads();

                if is_leader {
                    // PTX names this the 16-bit floating-point MMA family;
                    // the instruction descriptor selects bf16 inputs.
                    mma::mma_abt(accumulator.raw(), a_tile, b_tile, instruction, k_tile > 0);
                    mma::commit(mma_sem);
                }

                mma_sem.wait(phase);
                thread::sync_threads();
                k_tile += 1;
            }

            // TMEM fp32 -> registers -> packed bf16 shared-memory tile.
            let warp_row_base = (warp_id * 32) as usize;
            let row_in_matrix = (lane_id % 8) as usize;
            let matrix_column_offset = if (8..16).contains(&lane_id) { 16 } else { 0 };
            let row_stride_bytes = TC_TILE * 2;
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 8 {
                    let column = (column_block * 16) as usize;
                    let (low, high) = accumulator.fragment(tmem_row, column as u32);

                    let output_row = warp_row_base + row_block as usize * 16 + row_in_matrix;
                    let output_address = (&raw mut SMEM_OUT as *mut u8)
                        .add(output_row * row_stride_bytes + column * 2 + matrix_column_offset);
                    stmatrix_m8n8_x2(
                        output_address,
                        cvt_f32x2_bf16x2(low[0], low[1]),
                        cvt_f32x2_bf16x2(high[0], high[1]),
                    );

                    let output_address = (&raw mut SMEM_OUT as *mut u8).add(
                        (output_row + 8) * row_stride_bytes + column * 2 + matrix_column_offset,
                    );
                    stmatrix_m8n8_x2(
                        output_address,
                        cvt_f32x2_bf16x2(low[2], low[3]),
                        cvt_f32x2_bf16x2(high[2], high[3]),
                    );
                    column_block += 1;
                }
                row_block += 1;
            }
            thread::sync_threads();

            let packed_n = n as usize / 2;
            let tile_row_base = tile_m as usize * TC_TILE;
            let tile_column_base = tile_n as usize * (TC_TILE / 2);
            let mut local = tid as usize;
            while local < TC_TILE * TC_TILE / 2 {
                let local_row = local / (TC_TILE / 2);
                let local_column = local % (TC_TILE / 2);
                let global =
                    (tile_row_base + local_row) * packed_n + tile_column_base + local_column;
                let update = SMEM_OUT[local];
                if ACCUMULATE {
                    let slot = output.get_unchecked_mut(global);
                    *slot = accumulate_bf16_pair(*slot, update);
                } else {
                    *output.get_unchecked_mut(global) = update;
                }
                local += TC_TILE;
            }

            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(accumulator.raw(), 512);
            }
            if is_leader {
                tma_sem.inval();
                mma_sem.inval();
            }
        }
    }

    /// Blackwell TMA + tcgen05 GEMM with an fp32 global-memory epilogue.
    ///
    /// This intentionally remains a concrete variant rather than an epilogue
    /// framework: block-linears keep fp32 activations and gradients, while the
    /// tensor-core operands are bf16. One CTA computes a 128x128 tile.
    #[inline(always)]
    unsafe fn gemm_tcgen05_bf16_f32_impl<const ACCUMULATE: bool>(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut output: DisjointSlice<f32>,
        n: u32,
        k: u32,
    ) {
        unsafe {
            static mut SMEM_A: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDRESS: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TMA_BARRIER: Barrier = Barrier::UNINIT;
            static mut MMA_BARRIER: Barrier = Barrier::UNINIT;

            const TILE_BYTES: u32 = (TC_TILE * TC_BK * 2) as u32;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;
            let is_leader = tid == 0;
            let tile_m = thread::blockIdx_x();
            let tile_n = thread::blockIdx_y();

            let tma_sem = Semaphore::attach(&raw mut TMA_BARRIER);
            let mma_sem = Semaphore::attach(&raw mut MMA_BARRIER);
            let a_tile = KStage::from_raw(&raw mut SMEM_A as *mut u8);
            let b_tile = KStage::from_raw(&raw mut SMEM_B as *mut u8);

            if is_leader {
                tma_sem.init(1);
                mma_sem.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDRESS as *mut u32, 512);
            }
            thread::sync_threads();
            let accumulator =
                TmemTile::<TC_TILE, TC_TILE>::from_raw(*(&raw const TMEM_ADDRESS as *const u32));

            let instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M128_N128)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let mut k_tile = 0u32;
            while k_tile < k / TC_BK as u32 {
                let phase = k_tile & 1;
                if is_leader {
                    let k_offset = (k_tile * TC_BK as u32) as i32;
                    a_tile.tma_load_2d(a_tma, k_offset, (tile_m * TC_TILE as u32) as i32, tma_sem);
                    b_tile.tma_load_2d(b_tma, k_offset, (tile_n * TC_TILE as u32) as i32, tma_sem);
                    tma_sem.expect_tx(TILE_BYTES * 2);
                }

                tma_sem.wait(phase);
                thread::sync_threads();

                if is_leader {
                    mma::mma_abt(accumulator.raw(), a_tile, b_tile, instruction, k_tile > 0);
                    mma::commit(mma_sem);
                }

                mma_sem.wait(phase);
                thread::sync_threads();
                k_tile += 1;
            }

            // Reuse the proven TMEM-to-shared mapping from the packed-bf16
            // epilogue. The global drain widens each pair back to fp32, avoiding
            // a separate dequantization launch while preserving bf16 outputs.
            let warp_row_base = (warp_id * 32) as usize;
            let row_in_matrix = (lane_id % 8) as usize;
            let matrix_column_offset = if (8..16).contains(&lane_id) { 16 } else { 0 };
            let row_stride_bytes = TC_TILE * 2;
            let mut row_block = 0u32;
            while row_block < 2 {
                let tmem_row = warp_id * 32 + row_block * 16;
                let mut column_block = 0u32;
                while column_block < 8 {
                    let column = (column_block * 16) as usize;
                    let (low, high) = accumulator.fragment(tmem_row, column as u32);

                    let output_row = warp_row_base + row_block as usize * 16 + row_in_matrix;
                    let output_address = (&raw mut SMEM_OUT as *mut u8)
                        .add(output_row * row_stride_bytes + column * 2 + matrix_column_offset);
                    stmatrix_m8n8_x2(
                        output_address,
                        cvt_f32x2_bf16x2(low[0], low[1]),
                        cvt_f32x2_bf16x2(high[0], high[1]),
                    );

                    let output_address = (&raw mut SMEM_OUT as *mut u8).add(
                        (output_row + 8) * row_stride_bytes + column * 2 + matrix_column_offset,
                    );
                    stmatrix_m8n8_x2(
                        output_address,
                        cvt_f32x2_bf16x2(low[2], low[3]),
                        cvt_f32x2_bf16x2(high[2], high[3]),
                    );
                    column_block += 1;
                }
                row_block += 1;
            }
            thread::sync_threads();

            let packed_n = n as usize / 2;
            let tile_row_base = tile_m as usize * TC_TILE;
            let tile_column_base = tile_n as usize * (TC_TILE / 2);
            let mut local = tid as usize;
            while local < TC_TILE * TC_TILE / 2 {
                let local_row = local / (TC_TILE / 2);
                let local_column = local % (TC_TILE / 2);
                let global_pair =
                    (tile_row_base + local_row) * packed_n + tile_column_base + local_column;
                let update = SMEM_OUT[local];
                let lo = bf16_to_f32(update as u16);
                let hi = bf16_to_f32((update >> 16) as u16);
                let global = global_pair * 2;
                if ACCUMULATE {
                    *output.get_unchecked_mut(global) += lo;
                    *output.get_unchecked_mut(global + 1) += hi;
                } else {
                    *output.get_unchecked_mut(global) = lo;
                    *output.get_unchecked_mut(global + 1) = hi;
                }
                local += TC_TILE;
            }

            thread::sync_threads();
            if warp_id == 0 {
                tcgen05_dealloc(accumulator.raw(), 512);
            }
            if is_leader {
                tma_sem.inval();
                mma_sem.inval();
            }
        }
    }

    /// Blackwell bf16 `C = A B^T`, with fp32 tensor-core accumulation.
    #[kernel]
    pub unsafe fn gemm_tcgen05_bf16_store(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: DisjointSlice<u32>,
        n: u32,
        k: u32,
    ) {
        unsafe { gemm_tcgen05_bf16_impl::<false>(a_tma, b_tma, output, n, k) }
    }

    /// Blackwell bf16 `C += A B^T`, with fp32 tensor-core accumulation.
    #[kernel]
    pub unsafe fn gemm_tcgen05_bf16_accumulate(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: DisjointSlice<u32>,
        n: u32,
        k: u32,
    ) {
        unsafe { gemm_tcgen05_bf16_impl::<true>(a_tma, b_tma, output, n, k) }
    }

    /// Blackwell bf16 `C = A B^T`, widened to row-major fp32 output.
    #[kernel]
    pub unsafe fn gemm_tcgen05_bf16_f32_store(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: DisjointSlice<f32>,
        n: u32,
        k: u32,
    ) {
        unsafe { gemm_tcgen05_bf16_f32_impl::<false>(a_tma, b_tma, output, n, k) }
    }

    /// Blackwell bf16 `C += A B^T`, accumulating into row-major fp32 output.
    #[kernel]
    pub unsafe fn gemm_tcgen05_bf16_f32_accumulate(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: DisjointSlice<f32>,
        n: u32,
        k: u32,
    ) {
        unsafe { gemm_tcgen05_bf16_f32_impl::<true>(a_tma, b_tma, output, n, k) }
    }
}

include!("optimized.rs");
