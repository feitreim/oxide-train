//! Host-side tcgen05 support: tile contracts, TMA tensor maps, and ergonomic
//! adapters over cuda-oxide's generated typed launchers.

use std::collections::HashMap;
use std::error::Error;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Mutex;

use cuda_core::{CudaContext, CudaFunction, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::tma::TmaDescriptor;

/// TMA panel edge and packed-storage alignment.
pub const TC_TILE: usize = 128;
/// tcgen05 reduction tile: `K` must be a multiple of this.
pub const TC_BK: usize = 64;
/// Optimized CTA-pair output rows.
pub const TC_M_TILE: usize = 256;
/// Output columns callers test a shape's eligibility against.
///
/// Two work items rather than one since the tile halved
/// ([`gemm::BLOCK_N`](super::optimized::BLOCK_N)): kept at 256 because it is
/// what `gpu/model` gates on and a conservative gate costs nothing, while the
/// kernel's own unit is [`TC_N_ITEM`].
pub const TC_N_TILE: usize = 256;
/// Output columns of one work item — the kernel's tile, and the divisor its
/// `tiles_n` argument is counted in.
pub const TC_N_ITEM: usize = super::optimized::BLOCK_N;
/// Rows of `B` one rank stages: a `cta_group::2` `M256_N128` reads `TC_TILE`
/// rows of `A` and this many columns of `B` from each rank, so a bf16 matrix
/// needs a second TMA box to be usable as the `B` operand.
pub const TC_B_TILE: usize = TC_N_ITEM / 2;
/// Bytes of one CUDA tensor map, and the stride between the two boxes every
/// bf16 map encodes.
const TC_MAP_BYTES: usize = 128;
/// The K unit callers still test shapes for eligibility against.
///
/// It was a hard requirement when the pipeline was four hand-unrolled stages
/// and `k` had to fill a whole cycle; the three-deep ring does not care, and
/// [`tcgen05_launch_config`] asks only for a whole number of [`TC_BK`]. Kept as
/// a caller-facing constant because `gpu/model` gates on it, where it is now a
/// conservative test rather than a necessary one.
pub const TC_K_PIPELINE: usize = 256;

/// Persistent launch grid: [`gemm::MAX_CLUSTERS`](super::MAX_CLUSTERS) CTA
/// pairs, or fewer where the problem has fewer tiles.
///
/// The kernel is a work-item loop, not an exact cover, so the grid is a
/// property of the *device* and only capped by the problem: past
/// `MAX_CLUSTERS`, extra tiles arrive as extra items on clusters that already
/// exist. That is what stops a 16384³ launch paying 37 waves of cluster
/// start-up and lets one operand panel stay resident across the columns that
/// re-read it (`pipeline::grouped`, `GROUP = 8`).
///
/// `cluster_launch(2, 1, 1)` requires a whole number of clusters, which
/// `TC_RANKS *` guarantees.
///
/// `k` needs only to be a whole number of `TC_BK` stages — a three-deep ring
/// does not care how the block count divides. `TC_K_PIPELINE` survives as the
/// *caller-facing* eligibility unit some model shapes are still tested against;
/// nothing here requires it.
pub fn tcgen05_launch_config(m: usize, n: usize, k: usize) -> LaunchConfig {
    assert!(m.is_multiple_of(TC_M_TILE));
    assert!(n.is_multiple_of(TC_N_TILE));
    assert!(k.is_multiple_of(TC_BK));
    assert!(m <= u32::MAX as usize && n <= u32::MAX as usize && k <= u32::MAX as usize);
    let tiles = (m / TC_M_TILE)
        .checked_mul(n / TC_N_ITEM)
        .expect("tcgen05 work grid overflow");
    LaunchConfig {
        grid_dim: (
            TC_RANKS * tiles.min(super::MAX_CLUSTERS as usize) as u32,
            1,
            1,
        ),
        block_dim: (TC_THREADS, 1, 1),
        shared_mem_bytes: super::SHARED_BYTES as u32,
    }
}

/// CTAs of a cluster — the `cluster_launch` dimension, said once.
const TC_RANKS: u32 = 2;
/// Threads a CTA launches with: one warp per 32 accumulator rows, and every one
/// of them drains.
const TC_THREADS: u32 = super::optimized::THREADS;

/// Operand orientation, which fixes the TMA box.
///
/// A K-major operand is stored `[MN, K]` and streams `TC_TILE x TC_BK` tiles —
/// K contiguous, one 128-byte swizzle row per MN index. An MN-major operand is
/// stored `[K, MN]` and streams `TC_BK x TC_BK` subtiles instead: 128B swizzle
/// caps a TMA box at 128 bytes, so a CTA's 128 MN values arrive as two stacked
/// subtiles and the MMA's smem descriptor jumps between them via its LBO. See
/// `src/bin/transpose_probe.rs` for the geometry validation.
#[derive(Clone, Copy, PartialEq)]
pub enum TmaLayout {
    KMajor,
    MnMajor,
}

/// Which side of the MMA a map is being encoded for.
///
/// The `cta_group::2` `M256_N128` reads [`TC_TILE`] rows of `A` and
/// [`TC_B_TILE`] columns of `B` from each rank, so the two operands want
/// different box heights over the same matrix — and a matrix is routinely both
/// (a weight panel is `B` in the forward and `A` in the backward). Rather than
/// make callers say which role a map is for, every bf16 map encodes **both**
/// boxes, adjacent in one buffer, and the launcher picks with [`b_operand`].
#[derive(Clone, Copy)]
enum Operand {
    A,
    B,
}

impl TmaLayout {
    fn box_dimensions(self, operand: Operand) -> [u32; 2] {
        let rows = match (self, operand) {
            // MN-major boxes are `TC_BK` square either way: the swizzle caps a
            // box at 128 bytes, so `A`'s 128 MN values arrive as two of them
            // and `B`'s 64 as one.
            (TmaLayout::MnMajor, _) => TC_BK,
            (TmaLayout::KMajor, Operand::A) => TC_TILE,
            (TmaLayout::KMajor, Operand::B) => TC_B_TILE,
        };
        [TC_BK as u32, rows as u32]
    }
}

/// The `B`-operand box of a map pair — see [`Operand`].
fn b_operand(map: *const TmaDescriptor) -> *const TmaDescriptor {
    // SAFETY: every map this crate hands out is two adjacent descriptors, and
    // the second is in bounds of the same allocation.
    unsafe { (map as *const u8).add(TC_MAP_BYTES) as *const TmaDescriptor }
}

/// Encode a `SWIZZLE_128B` tensor map over a row-major `[height, width]` bf16
/// matrix at `base` (a device pointer).
fn encode_bf16_tma_map(
    stream: &CudaStream,
    base: u64,
    width: usize,
    height: usize,
    layout: TmaLayout,
) -> Result<DeviceBuffer<u64>, Box<dyn Error>> {
    encode_bf16_tma_map_strided(stream, base, width, height, width, layout)
}

/// Encode a bf16 tensor map whose logical rows are prefixes of wider physical
/// rows. This addresses one expert inside a globally transposed stacked
/// weight or activation buffer.
fn encode_bf16_tma_map_strided(
    stream: &CudaStream,
    base: u64,
    width: usize,
    height: usize,
    row_stride: usize,
    layout: TmaLayout,
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
    assert!(height.is_multiple_of(match layout {
        TmaLayout::KMajor => TC_TILE,
        TmaLayout::MnMajor => TC_BK,
    }));
    assert!(row_stride >= width);
    let global_dimensions = [width as u64, height as u64];
    let global_strides = [(row_stride * 2) as u64];
    let element_strides = [1u32, 1u32];
    let encode = |operand| -> Result<[u64; 16], Box<dyn Error>> {
        let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
        let box_dimensions = layout.box_dimensions(operand);
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
        Ok(unsafe { tensor_map.assume_init() }.opaque)
    };
    // The `A` box first, so a bare map pointer keeps meaning what it did.
    let mut pair = [0u64; 32];
    pair[..16].copy_from_slice(&encode(Operand::A)?);
    pair[16..].copy_from_slice(&encode(Operand::B)?);
    Ok(DeviceBuffer::from_host(stream, &pair)?)
}

/// Encode a `SWIZZLE_128B` fp32 tensor map delivering `[16, 32]` boxes of a
/// row-major `[height, width]` fp32 matrix whose rows are `row_stride`
/// elements apart — the reduction-store epilogue's `C`
/// (`gemm_tcgen05_f32_accumulate`, ferro-kittens #42). The box is the
/// `[16, STAGE_N]` fp32 staging tile's subtile: 32 fp32 columns is one
/// 128-byte swizzle atom.
///
/// Panics if the driver rejects the descriptor: every input is derived from
/// shapes the launch entry points already assert, so a rejection is a
/// programming error, not a runtime condition.
fn encode_f32_tma_map(
    stream: &CudaStream,
    base: u64,
    width: usize,
    height: usize,
    row_stride: usize,
) -> Result<DeviceBuffer<u64>, DriverError> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    assert!(width.is_multiple_of(TC_N_TILE));
    assert!(height.is_multiple_of(TC_M_TILE));
    assert!(row_stride >= width);
    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dimensions = [width as u64, height as u64];
    let global_strides = [(row_stride * 4) as u64];
    let box_dimensions = [32u32, 16u32];
    let element_strides = [1u32, 1u32];
    let status = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
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
    assert!(
        status == cudaError_enum_CUDA_SUCCESS,
        "cuTensorMapEncodeTiled(f32 [{height}, {width}] stride {row_stride}) failed: {status:?}"
    );
    let tensor_map = unsafe { tensor_map.assume_init() };
    DeviceBuffer::from_host(stream, &tensor_map.opaque)
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

/// Build a `SWIZZLE_128B` map pair over a bf16 matrix: the `A`-operand box
/// (`TC_TILE` rows) and the `B`-operand box (`TC_B_TILE` rows), adjacent in one
/// buffer, so one map serves a matrix that is `A` in one GEMM and `B` in
/// another.
pub fn create_bf16_tma_map<'matrix>(
    stream: &CudaStream,
    matrix: &'matrix DeviceBuffer<u16>,
    width: usize,
    height: usize,
    layout: TmaLayout,
) -> Result<Bf16TmaMap<'matrix>, Box<dyn Error>> {
    assert_eq!(matrix.len(), width * height);
    Ok(Bf16TmaMap {
        descriptor: encode_bf16_tma_map(stream, matrix.cu_deviceptr(), width, height, layout)?,
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
    layout: TmaLayout,
) -> Result<Bf16PairsTmaMap, Box<dyn Error>> {
    assert!(width.is_multiple_of(2));
    assert_eq!(matrix.len() * 2, width * height);
    Ok(Bf16PairsTmaMap {
        descriptor: encode_bf16_tma_map(stream, matrix.cu_deviceptr(), width, height, layout)?,
    })
}

/// Build a tensor map over a prefix of a larger packed-pair scratch buffer.
///
/// # Safety
///
/// `matrix` must remain at the same address and contain at least
/// `width * height / 2` words for every launch using the returned map.
pub unsafe fn create_bf16_pairs_tma_map_prefix(
    stream: &CudaStream,
    matrix: &DeviceBuffer<u32>,
    width: usize,
    height: usize,
    layout: TmaLayout,
) -> Result<Bf16PairsTmaMap, Box<dyn Error>> {
    assert!(width.is_multiple_of(2));
    assert!(matrix.len() * 2 >= width * height);
    Ok(Bf16PairsTmaMap {
        descriptor: encode_bf16_tma_map(stream, matrix.cu_deviceptr(), width, height, layout)?,
    })
}

/// Build a tensor map over a rectangular region of packed-pair bf16 storage.
///
/// `word_offset` locates the first logical element pair and `row_stride` is
/// measured in bf16 elements. A stride larger than `width` permits one expert
/// to be addressed inside a globally transposed `[height, E * width]` buffer.
///
/// # Safety
///
/// `matrix` must remain at the same address and the described strided region
/// must stay within the allocation for every launch using the returned map.
pub unsafe fn create_bf16_pairs_tma_map_region(
    stream: &CudaStream,
    matrix: &DeviceBuffer<u32>,
    word_offset: usize,
    width: usize,
    height: usize,
    row_stride: usize,
    layout: TmaLayout,
) -> Result<Bf16PairsTmaMap, Box<dyn Error>> {
    assert!(width.is_multiple_of(2));
    assert!(row_stride.is_multiple_of(2));
    assert!(row_stride >= width);
    let required_bf16 = if height == 0 {
        0
    } else {
        (height - 1)
            .checked_mul(row_stride)
            .and_then(|prefix| prefix.checked_add(width))
            .expect("bf16 TMA region size overflow")
    };
    assert!(
        word_offset
            .checked_add(required_bf16 / 2)
            .is_some_and(|end| end <= matrix.len()),
        "bf16 TMA region exceeds its packed allocation"
    );
    let byte_offset = word_offset
        .checked_mul(std::mem::size_of::<u32>())
        .expect("bf16 TMA region byte offset overflow");
    let base = matrix
        .cu_deviceptr()
        .checked_add(byte_offset as u64)
        .expect("bf16 TMA region device pointer overflow");
    Ok(Bf16PairsTmaMap {
        descriptor: encode_bf16_tma_map_strided(stream, base, width, height, row_stride, layout)?,
    })
}

/// The optimized tcgen05 bf16 GEMM loaded from the calling binary's single
/// embedded device artifact.
pub struct Tcgen05Gemm {
    generated: super::optimized::kernels::LoadedModule,
    optimized: CudaFunction,
    optimized_f32: CudaFunction,
    optimized_f32_accumulate: CudaFunction,
    /// Reduction-store `C` maps, keyed by `(device address, n, m)` and kept
    /// for the module's lifetime. The cache is the lifetime guarantee the
    /// async launch needs — a descriptor built per call would be freed while
    /// the kernel still reads it — and gradient buffers are stable across
    /// steps, so after the first step every accumulate launch is a lookup. A
    /// key's address being reused by a later allocation is harmless: the map's
    /// contents are a pure function of the key.
    c_maps: Mutex<HashMap<(u64, usize, usize), DeviceBuffer<u64>>>,
}

/// Raise a kernel's dynamic-shared-memory ceiling above the 48 KiB default.
///
/// Both entry points plan 114 816 B, so neither can be launched without this.
fn opt_in_dynamic_smem(function: &CudaFunction, bytes: u32) -> Result<(), Box<dyn Error>> {
    use cuda_core::sys::{
        CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        cuFuncSetAttribute, cudaError_enum_CUDA_SUCCESS,
    };
    // SAFETY: `function` is a live entry point of a loaded module, and the
    // attribute takes an `int`.
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

impl Tcgen05Gemm {
    pub fn load(ctx: &std::sync::Arc<CudaContext>) -> Result<Self, Box<dyn Error>> {
        let generated = super::optimized::kernels::load(ctx)?;
        let optimized = generated
            .as_cuda_module()
            .load_function("gemm_tcgen05_bf16_optimized")?;
        let optimized_f32 = generated
            .as_cuda_module()
            .load_function("gemm_tcgen05_f32_optimized")?;
        let optimized_f32_accumulate = generated
            .as_cuda_module()
            .load_function("gemm_tcgen05_f32_accumulate")?;
        opt_in_dynamic_smem(&optimized, super::SHARED_BYTES as u32)?;
        opt_in_dynamic_smem(&optimized_f32, super::SHARED_BYTES as u32)?;
        opt_in_dynamic_smem(&optimized_f32_accumulate, super::SHARED_BYTES as u32)?;
        Ok(Self {
            generated,
            optimized,
            optimized_f32,
            optimized_f32_accumulate,
            c_maps: Mutex::new(HashMap::new()),
        })
    }

    /// The loaded kernels, named for `bench_util::enforce_kernel_budgets`.
    pub fn kernels(&self) -> [(&'static str, &CudaFunction); 3] {
        [
            ("gemm_tcgen05_bf16_optimized", &self.optimized),
            ("gemm_tcgen05_f32_optimized", &self.optimized_f32),
            (
                "gemm_tcgen05_f32_accumulate",
                &self.optimized_f32_accumulate,
            ),
        ]
    }

    /// The cached reduction-store map for `output[offset..offset + elements]`
    /// read as an `[m, n]` fp32 matrix — see [`Tcgen05Gemm::c_maps`] for why
    /// the descriptor must outlive the call that built it.
    fn reduce_c_map(
        &self,
        stream: &CudaStream,
        output: &DeviceBuffer<f32>,
        output_offset: usize,
        output_elements: usize,
        n: u32,
    ) -> Result<*const TmaDescriptor, DriverError> {
        let base = output.cu_deviceptr() + (output_offset * std::mem::size_of::<f32>()) as u64;
        let m = output_elements / n as usize;
        let key = (base, n as usize, m);
        let mut maps = self.c_maps.lock().expect("reduction-map cache poisoned");
        if !maps.contains_key(&key) {
            let map = encode_f32_tma_map(stream, base, n as usize, m, n as usize)?;
            maps.insert(key, map);
        }
        Ok(maps[&key].cu_deviceptr() as *const TmaDescriptor)
    }

    /// Launch the reduction-store accumulate kernel: every `f32_accumulate*`
    /// adapter lands here, so the fold's read of `C` is gone from all of them
    /// at once (#80 remedy 1).
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_f32_accumulate(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<f32>,
        output_offset: usize,
        output_elements: usize,
        n: u32,
        k: u32,
        layout: TmaLayout,
    ) -> Result<(), DriverError> {
        let output_end = output_offset
            .checked_add(output_elements)
            .expect("tcgen05 fp32 output region overflow");
        assert!(output_end <= output.len());
        let c_map = self.reduce_c_map(stream, output, output_offset, output_elements, n)?;
        let m = output_elements / n as usize;
        unsafe {
            self.generated.gemm_tcgen05_f32_accumulate(
                stream,
                config,
                a_tma,
                b_operand(b_tma),
                c_map,
                k as i32,
                (m / TC_M_TILE) as u32,
                (n as usize / TC_N_ITEM) as u32,
                u32::from(layout == TmaLayout::MnMajor),
            )
        }
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
        unsafe {
            launch_tcgen05(
                &self.generated,
                stream,
                config,
                a_tma,
                b_tma,
                output,
                n,
                k,
                0,
                TmaLayout::KMajor,
            )
        }
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
        unsafe {
            launch_tcgen05(
                &self.generated,
                stream,
                config,
                a_tma,
                b_tma,
                output,
                n,
                k,
                1,
                TmaLayout::KMajor,
            )
        }
    }

    /// Blackwell bf16 `C = A B^T` with row-major fp32 output.
    ///
    /// # Safety
    ///
    /// The maps must describe live matrices matching the launch dimensions,
    /// and `output` must contain exactly `m * n` elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f32_store(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        let output_elements = output.len();
        unsafe {
            launch_tcgen05_f32(
                &self.generated,
                stream,
                config,
                a_tma,
                b_tma,
                output,
                0,
                output_elements,
                n,
                k,
                TmaLayout::KMajor,
            )
        }
    }

    /// Offset form of [`Tcgen05Gemm::f32_store`] for one matrix inside a
    /// stacked fp32 output allocation.
    ///
    /// # Safety
    ///
    /// The TMA maps must match the launch dimensions and the selected output
    /// region must contain exactly one `m * n` matrix.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f32_store_at(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<f32>,
        output_offset: usize,
        output_elements: usize,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            launch_tcgen05_f32(
                &self.generated,
                stream,
                config,
                a_tma,
                b_tma,
                output,
                output_offset,
                output_elements,
                n,
                k,
                TmaLayout::KMajor,
            )
        }
    }

    /// Blackwell bf16 `C += A B^T` with row-major fp32 output.
    ///
    /// # Safety
    ///
    /// Same contract as [`Tcgen05Gemm::f32_store`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f32_accumulate(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        let output_elements = output.len();
        unsafe {
            self.launch_f32_accumulate(
                stream,
                config,
                a_tma,
                b_tma,
                output,
                0,
                output_elements,
                n,
                k,
                TmaLayout::KMajor,
            )
        }
    }

    /// Offset form of [`Tcgen05Gemm::f32_accumulate`] for one matrix inside a
    /// stacked fp32 gradient allocation.
    ///
    /// # Safety
    ///
    /// Same contract as [`Tcgen05Gemm::f32_store_at`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f32_accumulate_at(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<f32>,
        output_offset: usize,
        output_elements: usize,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            self.launch_f32_accumulate(
                stream,
                config,
                a_tma,
                b_tma,
                output,
                output_offset,
                output_elements,
                n,
                k,
                TmaLayout::KMajor,
            )
        }
    }

    /// Packed-bf16 weight-gradient form: `C += Aᵀ·B` with both operands read
    /// MN-major from their native `[K, M]` / `[K, N]` panels (#53). This is the
    /// lm-head's gradient, whose operands are already bf16.
    ///
    /// # Safety
    ///
    /// Same contract as [`Tcgen05Gemm::f32_accumulate_transposed_at`], with a
    /// packed-pair output of `m * n / 2` words.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn accumulate_transposed(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<u32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            launch_tcgen05(
                &self.generated,
                stream,
                config,
                a_tma,
                b_tma,
                output,
                n,
                k,
                1,
                TmaLayout::MnMajor,
            )
        }
    }

    /// Weight-gradient form: `C += Aᵀ·B` with both operands read MN-major
    /// straight out of their native row-major `[K, M]` and `[K, N]` panels,
    /// via the descriptor's `transpose_a`/`transpose_b` bits (#53). Nothing is
    /// transposed in global memory, so the caller stages plain quantized
    /// activations and output gradients.
    ///
    /// # Safety
    ///
    /// The maps must be [`TmaLayout::MnMajor`] maps over live `[k, m]` and
    /// `[k, n]` matrices, and the selected output region must hold exactly one
    /// `m * n` matrix.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f32_accumulate_transposed_at(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<f32>,
        output_offset: usize,
        output_elements: usize,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            self.launch_f32_accumulate(
                stream,
                config,
                a_tma,
                b_tma,
                output,
                output_offset,
                output_elements,
                n,
                k,
                TmaLayout::MnMajor,
            )
        }
    }

    /// Overwrite form of [`Tcgen05Gemm::f32_accumulate_transposed`]: the same
    /// MN-major walk with the fold turned off, `C = Aᵀ·B`. The mode grid had
    /// this hole — the kernel takes `mode` and `transposed` independently and
    /// no adapter said so. It is what a caller writing the first micro-batch
    /// of a zeroed gradient wants, and what lets `bin/model_shapes.rs` price
    /// the fold's extra read of `C` apart from the MN-major operand walk.
    ///
    /// # Safety
    ///
    /// Same contract as [`Tcgen05Gemm::f32_accumulate_transposed`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f32_store_transposed(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        let output_elements = output.len();
        unsafe {
            launch_tcgen05_f32(
                &self.generated,
                stream,
                config,
                a_tma,
                b_tma,
                output,
                0,
                output_elements,
                n,
                k,
                TmaLayout::MnMajor,
            )
        }
    }

    /// Whole-buffer form of [`Tcgen05Gemm::f32_accumulate_transposed_at`].
    ///
    /// # Safety
    ///
    /// Same contract as [`Tcgen05Gemm::f32_accumulate_transposed_at`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f32_accumulate_transposed(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: &mut DeviceBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        let output_elements = output.len();
        unsafe {
            self.f32_accumulate_transposed_at(
                stream,
                config,
                a_tma,
                b_tma,
                output,
                0,
                output_elements,
                n,
                k,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_tcgen05(
    module: &super::optimized::kernels::LoadedModule,
    stream: &CudaStream,
    config: LaunchConfig,
    a_tma: *const TmaDescriptor,
    b_tma: *const TmaDescriptor,
    output: &mut DeviceBuffer<u32>,
    n: u32,
    k: u32,
    mode: u32,
    layout: TmaLayout,
) -> Result<(), DriverError> {
    let m = output
        .len()
        .checked_mul(2)
        .expect("tcgen05 packed output size overflow")
        / n as usize;
    unsafe {
        module.gemm_tcgen05_bf16_optimized(
            stream,
            config,
            a_tma,
            b_operand(b_tma),
            output,
            n as i32,
            k as i32,
            (m / TC_M_TILE) as u32,
            (n as usize / TC_N_ITEM) as u32,
            mode,
            u32::from(layout == TmaLayout::MnMajor),
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_tcgen05_f32(
    module: &super::optimized::kernels::LoadedModule,
    stream: &CudaStream,
    config: LaunchConfig,
    a_tma: *const TmaDescriptor,
    b_tma: *const TmaDescriptor,
    output: &mut DeviceBuffer<f32>,
    output_offset: usize,
    output_elements: usize,
    n: u32,
    k: u32,
    layout: TmaLayout,
) -> Result<(), DriverError> {
    let output_end = output_offset
        .checked_add(output_elements)
        .expect("tcgen05 fp32 output region overflow");
    assert!(output_end <= output.len());
    let m = output_elements / n as usize;
    unsafe {
        module.gemm_tcgen05_f32_optimized(
            stream,
            config,
            a_tma,
            b_operand(b_tma),
            output,
            output_offset,
            n as i32,
            k as i32,
            (m / TC_M_TILE) as u32,
            (n as usize / TC_N_ITEM) as u32,
            u32::from(layout == TmaLayout::MnMajor),
        )
    }
}
