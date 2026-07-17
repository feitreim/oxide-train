/*
 * The tcgen05 path is adapted from cuda-oxide's Apache-2.0-licensed
 * `examples/gemm_sol` at the repository's pinned v0.2.1 tag.
 */

//! GEMM performance ladder for the B200 training target.
//!
//! The portable first rung is an fp32 shared-memory/register-tiled kernel.
//! The Blackwell rung consumes row-major bf16 `A[M,K]` and transposed,
//! row-major bf16 `B[N,K]`, accumulates in fp32 TMEM with `tcgen05`, and emits
//! row-major packed-bf16 `C[M,N]`. Both rungs have overwrite and `C += A B`
//! variants so backward parameter gradients do not need a separate add kernel.
//!
//! bf16 is represented as raw `u16`/packed `u32` until milestone 7d adds the
//! shared `Element` plumbing.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::{marker::PhantomData, mem::MaybeUninit};

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive_expect_tx, mbarrier_init,
    mbarrier_inval, mbarrier_try_wait_parity,
};
use cuda_device::shared::SharedArray;
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
    cvt_f32x2_bf16x2, stmatrix_m8n8_x2, tcgen05_alloc, tcgen05_commit_shared_cluster,
    tcgen05_dealloc, tcgen05_ld_16x256b_pure, tcgen05_load_wait, tcgen05_mma_f16,
};
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

pub mod fp32;
pub use fp32::{BK, BM, BN, TM, TN, launch_config as fp32_launch_config};

/// tcgen05 CTA output tile edge: `M` and `N` must be multiples of this.
pub const TC_TILE: usize = 128;
/// tcgen05 reduction tile: `K` must be a multiple of this.
pub const TC_BK: usize = 64;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[inline(always)]
    fn smem_descriptor(
        smem_address: u64,
        leading_bytes: u32,
        stride_bytes: u32,
        swizzle: u8,
    ) -> u64 {
        let address = (smem_address >> 4) & 0x3fff;
        let leading = ((leading_bytes >> 4) & 0x3fff) as u64;
        let stride = ((stride_bytes >> 4) & 0x3fff) as u64;
        address | (leading << 16) | (stride << 32) | (1u64 << 46) | ((swizzle as u64) << 61)
    }

    #[inline(always)]
    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    #[inline(always)]
    fn f32_to_bf16_rne(value: f32) -> u16 {
        let bits = value.to_bits();
        let round = 0x7fffu32 + ((bits >> 16) & 1);
        (bits.wrapping_add(round) >> 16) as u16
    }

    #[inline(always)]
    fn accumulate_bf16_pair(current: u32, update: u32) -> u32 {
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
            const LEADING_BYTES: u32 = 16;
            const STRIDE_BYTES: u32 = 1024;
            const SWIZZLE_128B: u8 = 2;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;
            let is_leader = tid == 0;
            let tile_m = thread::blockIdx_x();
            let tile_n = thread::blockIdx_y();

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
            let tmem_address = *(&raw const TMEM_ADDRESS as *const u32);

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
                    cp_async_bulk_tensor_2d_g2s(
                        &raw mut SMEM_A as *mut u8,
                        a_tma,
                        k_offset,
                        (tile_m * TC_TILE as u32) as i32,
                        &raw mut TMA_BARRIER,
                    );
                    cp_async_bulk_tensor_2d_g2s(
                        &raw mut SMEM_B as *mut u8,
                        b_tma,
                        k_offset,
                        (tile_n * TC_TILE as u32) as i32,
                        &raw mut TMA_BARRIER,
                    );
                    mbarrier_arrive_expect_tx(&raw const TMA_BARRIER, 1, TILE_BYTES * 2);
                }

                while !mbarrier_try_wait_parity(&raw const TMA_BARRIER, phase) {}
                thread::sync_threads();

                if is_leader {
                    let a_base = &raw const SMEM_A as u64;
                    let b_base = &raw const SMEM_B as u64;
                    let mut mma = 0u32;
                    while mma < 4 {
                        let byte_offset = (mma * 32) as u64;
                        let a_descriptor = smem_descriptor(
                            a_base + byte_offset,
                            LEADING_BYTES,
                            STRIDE_BYTES,
                            SWIZZLE_128B,
                        );
                        let b_descriptor = smem_descriptor(
                            b_base + byte_offset,
                            LEADING_BYTES,
                            STRIDE_BYTES,
                            SWIZZLE_128B,
                        );
                        // PTX names this the 16-bit floating-point MMA family;
                        // the instruction descriptor selects bf16 inputs.
                        tcgen05_mma_f16(
                            tmem_address,
                            a_descriptor,
                            b_descriptor,
                            instruction,
                            k_tile > 0 || mma > 0,
                        );
                        mma += 1;
                    }
                    tcgen05_commit_shared_cluster(&raw mut MMA_BARRIER as *mut u64);
                }

                while !mbarrier_try_wait_parity(&raw const MMA_BARRIER, phase) {}
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
                    let low =
                        tcgen05_ld_16x256b_pure(tmem_address + (tmem_row << 16) + column as u32);
                    tcgen05_load_wait();
                    let high = tcgen05_ld_16x256b_pure(
                        tmem_address + (tmem_row << 16) + column as u32 + 8,
                    );
                    tcgen05_load_wait();

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
                tcgen05_dealloc(tmem_address, 512);
            }
            if is_leader {
                mbarrier_inval(&raw mut TMA_BARRIER);
                mbarrier_inval(&raw mut MMA_BARRIER);
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
}

/// Launch for the fixed Blackwell 128x128 tcgen05 output tile.
pub fn tcgen05_launch_config(m: usize, n: usize, k: usize) -> LaunchConfig {
    assert!(m.is_multiple_of(TC_TILE));
    assert!(n.is_multiple_of(TC_TILE) && n.is_multiple_of(2));
    assert!(k.is_multiple_of(TC_BK));
    assert!(m <= u32::MAX as usize && n <= u32::MAX as usize && k <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: ((m / TC_TILE) as u32, (n / TC_TILE) as u32, 1),
        block_dim: (TC_TILE as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Device-resident CUDA tensor map for a row-major bf16 matrix.
///
/// The map owns only the descriptor. The mapped matrix buffer must outlive all
/// launches that use this value.
pub struct Bf16TmaMap<'matrix> {
    descriptor: DeviceBuffer<u64>,
    _matrix: PhantomData<&'matrix DeviceBuffer<u16>>,
}

impl Bf16TmaMap<'_> {
    pub fn as_ptr(&self) -> *const TmaDescriptor {
        self.descriptor.cu_deviceptr() as *const TmaDescriptor
    }
}

/// Encode a `SWIZZLE_128B` tensor map loading 128x64 bf16 tiles from a
/// row-major `[height, width]` bf16 matrix at `base` (a device pointer).
fn encode_bf16_tma_map(
    stream: &CudaStream,
    base: u64,
    width: usize,
    height: usize,
) -> Result<DeviceBuffer<u64>, Box<dyn std::error::Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    assert!(width.is_multiple_of(TC_BK));
    assert!(height.is_multiple_of(TC_TILE));
    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dimensions = [width as u64, height as u64];
    let global_strides = [(width * 2) as u64];
    let box_dimensions = [TC_BK as u32, TC_TILE as u32];
    let element_strides = [1u32, 1u32];
    let status = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            base as *mut std::ffi::c_void,
            global_dimensions.as_ptr(),
            global_strides.as_ptr(),
            box_dimensions.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if status != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled(bf16) failed: {status:?}").into());
    }
    let tensor_map = unsafe { tensor_map.assume_init() };
    Ok(DeviceBuffer::from_host(stream, &tensor_map.opaque)?)
}

/// Build a `SWIZZLE_128B` tensor map loading a 128x64 bf16 tile.
pub fn create_bf16_tma_map<'matrix>(
    stream: &CudaStream,
    matrix: &'matrix DeviceBuffer<u16>,
    width: usize,
    height: usize,
) -> Result<Bf16TmaMap<'matrix>, Box<dyn std::error::Error>> {
    assert_eq!(matrix.len(), width * height);
    Ok(Bf16TmaMap {
        descriptor: encode_bf16_tma_map(stream, matrix.cu_deviceptr(), width, height)?,
        _matrix: PhantomData,
    })
}

/// Tensor map over packed-pair bf16 storage (`u32` = two adjacent row
/// elements), for owners that hold the mapped buffer alongside the map.
///
/// Unlike [`Bf16TmaMap`] this does not borrow the matrix: the constructor is
/// `unsafe` and the caller promises the mapped allocation outlives every
/// launch that consumes the map.
pub struct Bf16PairsTmaMap {
    descriptor: DeviceBuffer<u64>,
}

impl Bf16PairsTmaMap {
    pub fn as_ptr(&self) -> *const TmaDescriptor {
        self.descriptor.cu_deviceptr() as *const TmaDescriptor
    }
}

/// Build a `SWIZZLE_128B` tensor map over a row-major `[height, width]` bf16
/// matrix stored as packed pairs.
///
/// # Safety
///
/// `matrix` must stay allocated at the same device address for every kernel
/// launch that consumes the returned map.
pub unsafe fn create_bf16_pairs_tma_map(
    stream: &CudaStream,
    matrix: &DeviceBuffer<u32>,
    width: usize,
    height: usize,
) -> Result<Bf16PairsTmaMap, Box<dyn std::error::Error>> {
    assert!(width.is_multiple_of(2));
    assert_eq!(matrix.len() * 2, width * height);
    Ok(Bf16PairsTmaMap {
        descriptor: encode_bf16_tma_map(stream, matrix.cu_deviceptr(), width, height)?,
    })
}
