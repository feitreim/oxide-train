//! Host-side support for the tcgen05 attention kernels: staging-buffer TMA
//! maps, launch configs, and raw launchers for the kernels in `flash.ptx`.
//!
//! This file deliberately contains no `#[cuda_module]` (the gemm `host.rs`
//! pattern). Binaries whose own device artifact must stay free of tcgen05
//! lowerings — `main.rs` here (libdevice oracle kernels) and later
//! `gpu/model` — include only this module and load the kernels from a
//! `flash.ptx` built separately by `src/bin/flash.rs`, which takes the
//! pure-PTX path.

use std::error::Error;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cuda_core::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig,
};
use cuda_device::tma::TmaDescriptor;

/// Query/key tile edge: `T` must be a multiple of this to use the tcgen05
/// forward; other shapes stay on the fp32 tiled kernels.
pub const FLASH_TILE: usize = 128;
/// The only head width the tcgen05 forward supports.
pub const FLASH_HD: usize = 64;
/// Dynamic shared bytes of the synchronous forward kernel: Q, K, V tiles
/// plus the two stacked P subtiles. Mirrors `FLASH_DYNAMIC_SMEM` in
/// `tcgen05.rs`.
pub const FLASH_DYNAMIC_SMEM_BYTES: u32 = (5 * FLASH_TILE * FLASH_HD * 2) as u32;
/// Dynamic shared bytes of the transpose_b probe: A (two subtiles) plus B.
pub const PROBE_DYNAMIC_SMEM_BYTES: u32 = (3 * FLASH_TILE * FLASH_HD * 2) as u32;
/// Dynamic shared allocation for the pipelined forward: Q + K/V rings sized
/// for the deepest supported `PIPELINE_STAGES` (4) + the two P subtiles.
/// The kernel's actual plan (`FLASH_PIPELINE_SMEM`, a function of the swept
/// `PIPELINE_STAGES` in `tcgen05.rs`) must stay at or under this; the flash
/// bin asserts it. Allocating the ceiling keeps stage sweeps a one-const
/// edit, and costs nothing: TMEM (512 columns per CTA against a 512-column
/// SM budget) already pins occupancy to one CTA per SM.
pub const FLASH_PIPELINE_SMEM_BYTES: u32 = ((3 + 2 * 4) * FLASH_TILE * FLASH_HD * 2) as u32;
/// Threads of the pipelined forward: the 128-thread softmax warpgroup plus
/// the TMA-load warp and the MMA-issue warp. Mirrors `FLASH_PIPELINE_BLOCK`.
pub const FLASH_PIPELINE_BLOCK_THREADS: u32 = (FLASH_TILE + 64) as u32;

/// Launch for the synchronous tcgen05 forward over `batches` packed
/// sequences.
pub fn flash_forward_config(batches: usize, sequence_length: usize, heads: usize) -> LaunchConfig {
    assert!(sequence_length.is_multiple_of(FLASH_TILE));
    assert!(batches <= u16::MAX as usize && heads <= u16::MAX as usize);
    assert!(sequence_length / FLASH_TILE <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            (sequence_length / FLASH_TILE) as u32,
            heads as u32,
            batches as u32,
        ),
        block_dim: (FLASH_TILE as u32, 1, 1),
        shared_mem_bytes: FLASH_DYNAMIC_SMEM_BYTES,
    }
}

/// Launch for the warp-specialized pipelined forward: same grid, the wider
/// block, the ring-sized dynamic shared allocation.
pub fn flash_pipelined_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
) -> LaunchConfig {
    let base = flash_forward_config(batches, sequence_length, heads);
    LaunchConfig {
        grid_dim: base.grid_dim,
        block_dim: (FLASH_PIPELINE_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: FLASH_PIPELINE_SMEM_BYTES,
    }
}

/// Encode a `SWIZZLE_128B` tensor map loading swizzled `[128, 64]` bf16 tiles
/// from one `[T, 64]` head panel of a packed `[planes, T, 64]` staging buffer
/// (`planes = B*H`); the kernel selects the panel via the third coordinate.
fn encode_bf16_head_tma_map(
    stream: &CudaStream,
    base: u64,
    sequence_length: usize,
    planes: usize,
) -> Result<DeviceBuffer<u64>, Box<dyn Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    assert!(sequence_length.is_multiple_of(FLASH_TILE));
    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dimensions = [FLASH_HD as u64, sequence_length as u64, planes as u64];
    let global_strides = [
        (FLASH_HD * 2) as u64,
        (sequence_length * FLASH_HD * 2) as u64,
    ];
    let box_dimensions = [FLASH_HD as u32, FLASH_TILE as u32, 1u32];
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
        return Err(format!("cuTensorMapEncodeTiled(bf16 head panel) failed: {status:?}").into());
    }
    let tensor_map = unsafe { tensor_map.assume_init() };
    Ok(DeviceBuffer::from_host(stream, &tensor_map.opaque)?)
}

/// Tensor map over one packed-bf16 `[planes, T, 64]` staging buffer.
///
/// Does not borrow the buffer: the constructor is `unsafe` and the caller
/// promises the mapped allocation outlives every launch consuming the map.
pub struct FlashHeadTmaMap {
    descriptor: DeviceBuffer<u64>,
}

impl FlashHeadTmaMap {
    pub fn as_ptr(&self) -> *const TmaDescriptor {
        self.descriptor.cu_deviceptr() as *const TmaDescriptor
    }
}

/// Build a head-panel tensor map over a packed-pair staging buffer holding
/// `planes` panels of `[sequence_length, 64]` bf16 values.
///
/// # Safety
///
/// `buffer` must stay allocated at the same device address for every kernel
/// launch that consumes the returned map.
pub unsafe fn create_flash_head_tma_map(
    stream: &CudaStream,
    buffer: &DeviceBuffer<u32>,
    sequence_length: usize,
    planes: usize,
) -> Result<FlashHeadTmaMap, Box<dyn Error>> {
    assert_eq!(buffer.len() * 2, planes * sequence_length * FLASH_HD);
    Ok(FlashHeadTmaMap {
        descriptor: encode_bf16_head_tma_map(
            stream,
            buffer.cu_deviceptr(),
            sequence_length,
            planes,
        )?,
    })
}

/// Raise a kernel's dynamic-shared-memory ceiling above the 48 KiB default.
fn opt_in_dynamic_smem(function: &CudaFunction, bytes: u32) -> Result<(), Box<dyn Error>> {
    use cuda_core::sys::{
        CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        cuFuncSetAttribute, cudaError_enum_CUDA_SUCCESS,
    };
    let status = unsafe {
        cuFuncSetAttribute(
            function.cu_function(),
            CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )
    };
    if status != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuFuncSetAttribute(dynamic smem {bytes}) failed: {status:?}").into());
    }
    Ok(())
}

/// The tcgen05 attention kernels, loaded from a `flash.ptx` built by
/// `src/bin/flash.rs` rather than from the calling binary's embedded
/// artifact. The launchers mirror the `#[cuda_module]`-generated
/// marshalling exactly.
pub struct Tcgen05Flash {
    forward: CudaFunction,
    forward_pipelined: CudaFunction,
    transpose_probe: CudaFunction,
    swizzle_probe: CudaFunction,
    exp2: CudaFunction,
    log2: CudaFunction,
    _module: Arc<CudaModule>,
}

impl Tcgen05Flash {
    pub fn load_from_ptx(ctx: &Arc<CudaContext>, path: &str) -> Result<Self, Box<dyn Error>> {
        let module = ctx.load_module_from_file(path).map_err(|error| {
            format!(
                "loading {path} failed ({error:?}); build gpu/flash-attn's `flash` \
                 binary first so its pure-PTX artifact exists"
            )
        })?;
        let forward = module.load_function("flash_forward_tcgen05")?;
        let forward_pipelined = module.load_function("flash_forward_pipelined")?;
        let transpose_probe = module.load_function("transpose_b_probe")?;
        opt_in_dynamic_smem(&forward, FLASH_DYNAMIC_SMEM_BYTES)?;
        opt_in_dynamic_smem(&forward_pipelined, FLASH_PIPELINE_SMEM_BYTES)?;
        opt_in_dynamic_smem(&transpose_probe, PROBE_DYNAMIC_SMEM_BYTES)?;
        Ok(Self {
            forward,
            forward_pipelined,
            transpose_probe,
            swizzle_probe: module.load_function("swizzle_probe")?,
            exp2: module.load_function("software_exp2")?,
            log2: module.load_function("software_log2")?,
            _module: module,
        })
    }

    /// Synchronous tcgen05 causal attention forward over bf16 head-panel
    /// staging buffers. Launch with `flash_forward_config`.
    ///
    /// # Safety
    ///
    /// The maps must describe live `[B*H, T, 64]` staging buffers matching
    /// the launch config, `output` must hold `B*T*H*64` elements and
    /// `logsumexp` `B*T*H` elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        sequence_length: u32,
        heads: u32,
        output: &mut DeviceBuffer<f32>,
        logsumexp: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.launch_forward(
                &self.forward,
                stream,
                config,
                q_tma,
                k_tma,
                v_tma,
                sequence_length,
                heads,
                output,
                logsumexp,
            )
        }
    }

    /// Warp-specialized pipelined forward (issue #35, phase 2): identical
    /// contract to `forward`, launched with `flash_pipelined_config`.
    ///
    /// # Safety
    ///
    /// Same contract as `forward`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_pipelined(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        sequence_length: u32,
        heads: u32,
        output: &mut DeviceBuffer<f32>,
        logsumexp: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.launch_forward(
                &self.forward_pipelined,
                stream,
                config,
                q_tma,
                k_tma,
                v_tma,
                sequence_length,
                heads,
                output,
                logsumexp,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_forward(
        &self,
        function: &CudaFunction,
        stream: &CudaStream,
        config: LaunchConfig,
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        sequence_length: u32,
        heads: u32,
        output: &mut DeviceBuffer<f32>,
        logsumexp: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        let mut q_tma = q_tma;
        let mut k_tma = k_tma;
        let mut v_tma = v_tma;
        let mut sequence_length = sequence_length;
        let mut heads = heads;
        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        cuda_host::push_kernel_scalar(&mut args, &mut q_tma);
        cuda_host::push_kernel_scalar(&mut args, &mut k_tma);
        cuda_host::push_kernel_scalar(&mut args, &mut v_tma);
        cuda_host::push_kernel_scalar(&mut args, &mut sequence_length);
        cuda_host::push_kernel_scalar(&mut args, &mut heads);
        let (mut output_ptr, mut output_len) = cuda_host::writable_device_buffer_arg(output);
        cuda_host::push_kernel_device_slice(&mut args, &mut output_ptr, &mut output_len);
        let (mut lse_ptr, mut lse_len) = cuda_host::writable_device_buffer_arg(logsumexp);
        cuda_host::push_kernel_device_slice(&mut args, &mut lse_ptr, &mut lse_len);
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

    /// One-CTA `C[128,64] = A[128,128]·B[128,64]` with B consumed through the
    /// transposed-descriptor path. `A` is staged as two `[128, 64]` head
    /// panels (planes 0/1 hold columns 0..64 / 64..128), `B` as one panel.
    ///
    /// # Safety
    ///
    /// The maps must describe live staging buffers of those shapes and
    /// `output` must hold `128 * 64` elements.
    pub unsafe fn transpose_probe(
        &self,
        stream: &CudaStream,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        let mut a_tma = a_tma;
        let mut b_tma = b_tma;
        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        cuda_host::push_kernel_scalar(&mut args, &mut a_tma);
        cuda_host::push_kernel_scalar(&mut args, &mut b_tma);
        let (mut output_ptr, mut output_len) = cuda_host::writable_device_buffer_arg(output);
        cuda_host::push_kernel_device_slice(&mut args, &mut output_ptr, &mut output_len);
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.transpose_probe,
                (1, 1, 1),
                (FLASH_TILE as u32, 1, 1),
                PROBE_DYNAMIC_SMEM_BYTES,
                stream,
                &mut args,
            )
        }
    }

    /// Dump one TMA-loaded `[128, 64]` bf16 tile's raw shared-memory words.
    ///
    /// # Safety
    ///
    /// The map must describe a live staging buffer with at least one
    /// `[128, 64]` panel; `output` must hold `128 * 32` words.
    pub unsafe fn swizzle_probe(
        &self,
        stream: &CudaStream,
        src_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<u32>,
    ) -> Result<(), DriverError> {
        let mut src_tma = src_tma;
        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        cuda_host::push_kernel_scalar(&mut args, &mut src_tma);
        let (mut output_ptr, mut output_len) = cuda_host::writable_device_buffer_arg(output);
        cuda_host::push_kernel_device_slice(&mut args, &mut output_ptr, &mut output_len);
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.swizzle_probe,
                (1, 1, 1),
                (FLASH_TILE as u32, 1, 1),
                (FLASH_TILE * FLASH_HD * 2) as u32,
                stream,
                &mut args,
            )
        }
    }

    /// Elementwise software-`exp2` accuracy oracle.
    pub fn software_exp2(
        &self,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        self.launch_elementwise(&self.exp2, stream, input, output)
    }

    /// Elementwise software-`log2` accuracy oracle.
    pub fn software_log2(
        &self,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        self.launch_elementwise(&self.log2, stream, input, output)
    }

    fn launch_elementwise(
        &self,
        function: &CudaFunction,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        assert_eq!(input.len(), output.len());
        let config = LaunchConfig::for_num_elems(output.len() as u32);
        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        let (mut input_ptr, mut input_len) = cuda_host::read_only_device_buffer_arg(input);
        cuda_host::push_kernel_device_slice(&mut args, &mut input_ptr, &mut input_len);
        let (mut output_ptr, mut output_len) = cuda_host::writable_device_buffer_arg(output);
        cuda_host::push_kernel_device_slice(&mut args, &mut output_ptr, &mut output_len);
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
}
