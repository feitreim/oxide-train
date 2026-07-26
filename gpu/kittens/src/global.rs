//! Global-memory layouts and their TMA tensor maps.
//!
//! The device side of a global operand is just a `*const TmaDescriptor`
//! kernel parameter; what the type system can hold is the *host* side — the
//! `cuTensorMapEncodeTiled` call whose box shape must agree with the
//! [`crate::shared::SharedTile`] the kernel loads into. The builder here is
//! generic over that tile, so the agreement is by construction: one box per
//! subtile, `SUBTILE_COLS` wide, `R` rows tall.
//!
//! Host-only (`feature = "host"`): the device crates never see cuda-core.

#[cfg(feature = "host")]
pub use host::{PanelMap, encode_bf16_panels};

#[cfg(feature = "host")]
mod host {
    use std::error::Error;
    use std::mem::MaybeUninit;

    use cuda_core::{CudaStream, DeviceBuffer};
    use cuda_device::tma::TmaDescriptor;

    use crate::shared::{Bf16, SharedTile, Swizzle128B};

    /// A device-resident TMA tensor map over a packed `[planes, rows, C]`
    /// bf16 buffer, boxed as swizzled `[R, 64]` subtiles.
    ///
    /// Does not borrow the mapped buffer: the constructor is `unsafe` and the
    /// caller promises the allocation outlives every launch consuming the map.
    pub struct PanelMap {
        descriptor: DeviceBuffer<u64>,
    }

    impl PanelMap {
        /// The pointer kernels take as their TMA parameter.
        pub fn as_ptr(&self) -> *const TmaDescriptor {
            self.descriptor.cu_deviceptr() as *const TmaDescriptor
        }
    }

    /// Encode a SWIZZLE_128B tensor map loading `[R, 64]` bf16 subtiles of a
    /// `SharedTile<Bf16, R, C, Swizzle128B>` from one `[rows, C]` panel of a
    /// packed `[planes, rows, C]` staging buffer. The kernel selects the
    /// panel via the third coordinate, the row range via the second, and the
    /// stacked subtile (columns `64*i..64*(i+1)`) via the first — which
    /// [`SharedTile::tma_load`] walks automatically.
    ///
    /// # Safety
    ///
    /// `base` must be the device address of a live buffer of at least
    /// `planes * rows * C` bf16 elements, staying allocated at that address
    /// for every launch that consumes the returned map.
    pub unsafe fn encode_bf16_panels<const R: usize, const C: usize>(
        stream: &CudaStream,
        base: u64,
        rows: usize,
        planes: usize,
    ) -> Result<PanelMap, Box<dyn Error>> {
        use cuda_core::sys::{
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
            cudaError_enum_CUDA_SUCCESS,
        };

        type Tile<const R: usize, const C: usize> = SharedTile<Bf16, R, C, Swizzle128B>;

        assert!(rows.is_multiple_of(R));
        let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
        let global_dimensions = [C as u64, rows as u64, planes as u64];
        let global_strides = [(C * 2) as u64, (rows * C * 2) as u64];
        let box_dimensions = [Tile::<R, C>::SUBTILE_COLS as u32, R as u32, 1u32];
        let element_strides = [1u32, 1u32, 1u32];
        let status = unsafe {
            cuTensorMapEncodeTiled(
                tensor_map.as_mut_ptr(),
                CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                3,
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
            return Err(format!("cuTensorMapEncodeTiled(bf16 panels) failed: {status:?}").into());
        }
        let tensor_map = unsafe { tensor_map.assume_init() };
        Ok(PanelMap {
            descriptor: DeviceBuffer::from_host(stream, &tensor_map.opaque)?,
        })
    }
}
