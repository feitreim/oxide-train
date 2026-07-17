//! Host-side tcgen05 support: tile contracts, TMA tensor maps, and raw
//! launchers for the bf16 GEMM kernels.
//!
//! This file deliberately contains no `#[cuda_module]`. Binaries whose own
//! device artifact must stay free of tcgen05 lowerings (gpu/llama-model: its
//! libdevice math forces the artifact through libNVVM, which rejects tcgen05
//! constructs) include only this module and load the kernels from a
//! `gemm.ptx` built separately by this crate, which takes the pure-PTX path.

use std::error::Error;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cuda_core::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig,
};
use cuda_device::tma::TmaDescriptor;

/// tcgen05 CTA output tile edge: `M` and `N` must be multiples of this.
pub const TC_TILE: usize = 128;
/// tcgen05 reduction tile: `K` must be a multiple of this.
pub const TC_BK: usize = 64;

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

/// Encode a `SWIZZLE_128B` tensor map loading 128x64 bf16 tiles from a
/// row-major `[height, width]` bf16 matrix at `base` (a device pointer).
fn encode_bf16_tma_map(
    stream: &CudaStream,
    base: u64,
    width: usize,
    height: usize,
) -> Result<DeviceBuffer<u64>, Box<dyn Error>> {
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

/// Build a `SWIZZLE_128B` tensor map loading a 128x64 bf16 tile.
pub fn create_bf16_tma_map<'matrix>(
    stream: &CudaStream,
    matrix: &'matrix DeviceBuffer<u16>,
    width: usize,
    height: usize,
) -> Result<Bf16TmaMap<'matrix>, Box<dyn Error>> {
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
) -> Result<Bf16PairsTmaMap, Box<dyn Error>> {
    assert!(width.is_multiple_of(2));
    assert_eq!(matrix.len() * 2, width * height);
    Ok(Bf16PairsTmaMap {
        descriptor: encode_bf16_tma_map(stream, matrix.cu_deviceptr(), width, height)?,
    })
}

/// The two tcgen05 bf16 GEMM kernels, loaded from a `gemm.ptx` built by this
/// crate rather than from the calling binary's embedded artifact.
///
/// The launchers mirror the `#[cuda_module]`-generated marshalling exactly:
/// TMA descriptor pointers and dimensions as scalars, the packed output as a
/// `(pointer, length)` device-slice pair.
pub struct Tcgen05Gemm {
    store: CudaFunction,
    accumulate: CudaFunction,
    _module: Arc<CudaModule>,
}

impl Tcgen05Gemm {
    pub fn load_from_ptx(ctx: &Arc<CudaContext>, path: &str) -> Result<Self, Box<dyn Error>> {
        let module = ctx.load_module_from_file(path).map_err(|error| {
            format!(
                "loading {path} failed ({error:?}); build gpu/gemm first so its \
                 pure-PTX artifact exists (run.sh does this for llama-model)"
            )
        })?;
        Ok(Self {
            store: module.load_function("gemm_tcgen05_bf16_store")?,
            accumulate: module.load_function("gemm_tcgen05_bf16_accumulate")?,
            _module: module,
        })
    }

    /// Blackwell bf16 `C = A B^T`; see the kernel for the full contract.
    ///
    /// # Safety
    ///
    /// Same contract as the generated launcher: the TMA maps must describe
    /// live matrices matching the launch dimensions, and `output` must hold
    /// exactly `m * n / 2` packed pairs.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn store(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<u32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe { launch_tcgen05(&self.store, stream, config, a_tma, b_tma, output, n, k) }
    }

    /// Blackwell bf16 `C += A B^T`; see the kernel for the full contract.
    ///
    /// # Safety
    ///
    /// Same contract as [`Tcgen05Gemm::store`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn accumulate(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<u32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe { launch_tcgen05(&self.accumulate, stream, config, a_tma, b_tma, output, n, k) }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_tcgen05(
    function: &CudaFunction,
    stream: &CudaStream,
    config: LaunchConfig,
    mut a_tma: *const TmaDescriptor,
    mut b_tma: *const TmaDescriptor,
    output: &mut DeviceBuffer<u32>,
    mut n: u32,
    mut k: u32,
) -> Result<(), DriverError> {
    let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
    cuda_host::push_kernel_scalar(&mut args, &mut a_tma);
    cuda_host::push_kernel_scalar(&mut args, &mut b_tma);
    let (mut output_ptr, mut output_len) = cuda_host::writable_device_buffer_arg(output);
    cuda_host::push_kernel_device_slice(&mut args, &mut output_ptr, &mut output_len);
    cuda_host::push_kernel_scalar(&mut args, &mut n);
    cuda_host::push_kernel_scalar(&mut args, &mut k);
    unsafe {
        cuda_core::launch_kernel_on_stream(
            function,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            stream,
            &mut args,
        )
    }
}
