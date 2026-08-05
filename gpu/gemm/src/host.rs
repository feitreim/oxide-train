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
/// Optimized CTA-pair output columns — the wide tile's.
pub const TC_N_TILE: usize = 256;
/// The narrow variant's CTA-pair output columns (oxide-train#80 remedy 3).
pub const TC_N_NARROW: usize = 128;

/// Which compiled pair tile a launch runs on. Same pipeline, same epilogues,
/// same 114 816 B shared plan and 148-cluster grid; what moves is the output
/// tile a work item covers, and with it the wave arithmetic and the TMEM
/// columns an accumulator pins.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TcTile {
    /// `[256, 256]` — [`super::optimized`], the shipped kernel. Deep-K rows
    /// and every square-bench shape stay here.
    Wide,
    /// `[256, 128]` — [`super::narrow`], selected where the halved tile
    /// quantum improves the last wave.
    Narrow,
}

impl TcTile {
    /// Output columns of one work item's pair tile.
    pub const fn n_tile(self) -> usize {
        match self {
            TcTile::Wide => TC_N_TILE,
            TcTile::Narrow => TC_N_NARROW,
        }
    }

    const fn max_clusters(self) -> u32 {
        match self {
            TcTile::Wide => super::optimized::MAX_CLUSTERS,
            TcTile::Narrow => super::narrow::MAX_CLUSTERS,
        }
    }

    const fn shared_bytes(self) -> usize {
        match self {
            TcTile::Wide => super::optimized::SHARED_BYTES,
            TcTile::Narrow => super::narrow::SHARED_BYTES,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TcTile::Wide => "[256,256]",
            TcTile::Narrow => "[256,128]",
        }
    }
}

/// A launch geometry plus the tile it was computed for. The launch adapters
/// take this rather than a bare `LaunchConfig` because the grid, the shared
/// envelope, the entry point and `B`'s tensor-map box all follow the tile —
/// a config applied to the other variant's kernel would be silently wrong in
/// all four.
#[derive(Clone, Copy)]
pub struct Tcgen05Launch {
    pub tile: TcTile,
    pub config: LaunchConfig,
}

/// The pair tile a shape dispatches to.
///
/// The rule: prefer the wide tile — it carries 1.5× the arithmetic intensity
/// (`M·N/(M+N)`: 128 vs 85.3) and won ferro #87's sweep at every square at or
/// above 2048³ — and take the narrow tile only where it strictly improves the
/// persistent grid's last-wave efficiency, `tiles / (⌈tiles/148⌉ · 148)`.
/// Halving the tile doubles the item count, so the comparison reduces to
/// `2·⌈t/148⌉ > ⌈2t/148⌉`. Equal efficiency keeps the wide tile, which is what
/// structurally protects the deep-K model rows (0.94–0.97 of cuBLASLt) and
/// every square-bench shape: their wave math is tile-indifferent, so they
/// never leave the wide kernel.
///
/// Of the training step's fourteen shapes this routes exactly three narrow:
/// qkv fwd (0.973 → 0.994), gate_up fwd (0.865 → 0.943) and down dW
/// (0.649 → 0.865). `k` is accepted so a measured K-depth guard has somewhere
/// to live if the numbers ask for one.
pub fn tcgen05_tile(m: usize, n: usize, _k: usize) -> TcTile {
    let clusters = TcTile::Wide.max_clusters() as usize;
    let tiles = (m / TC_M_TILE) * (n / TC_N_TILE);
    let waves = |tiles: usize| tiles.div_ceil(clusters);
    if 2 * waves(tiles) > waves(2 * tiles) {
        TcTile::Narrow
    } else {
        TcTile::Wide
    }
}
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
pub fn tcgen05_launch_config(m: usize, n: usize, k: usize) -> Tcgen05Launch {
    tcgen05_launch_config_tiled(m, n, k, tcgen05_tile(m, n, k))
}

/// [`tcgen05_launch_config`] at a caller-chosen tile — the benches' instrument
/// for pricing the two variants against each other on one shape.
pub fn tcgen05_launch_config_tiled(m: usize, n: usize, k: usize, tile: TcTile) -> Tcgen05Launch {
    assert!(m.is_multiple_of(TC_M_TILE));
    assert!(n.is_multiple_of(TC_N_TILE));
    assert!(k.is_multiple_of(TC_BK));
    assert!(m <= u32::MAX as usize && n <= u32::MAX as usize && k <= u32::MAX as usize);
    let tiles = (m / TC_M_TILE)
        .checked_mul(n / tile.n_tile())
        .expect("tcgen05 work grid overflow");
    Tcgen05Launch {
        tile,
        config: LaunchConfig {
            grid_dim: (
                TC_RANKS * tiles.min(tile.max_clusters() as usize) as u32,
                1,
                1,
            ),
            block_dim: (TC_THREADS, 1, 1),
            shared_mem_bytes: tile.shared_bytes() as u32,
        },
    }
}

/// CTAs of a cluster — the `cluster_launch` dimension, said once.
const TC_RANKS: u32 = 2;
/// Threads a CTA launches with: one warp per 32 accumulator rows plus the two
/// role warps — the same 192 in both variants, which the assert holds still.
const TC_THREADS: u32 = super::optimized::THREADS;
const _: () = assert!(TC_THREADS == super::narrow::THREADS);

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

impl TmaLayout {
    /// The box the *wide* kernel's `B` stage (and both kernels' `A` stage)
    /// loads through.
    fn box_dimensions(self) -> [u32; 2] {
        match self {
            TmaLayout::KMajor => [TC_BK as u32, TC_TILE as u32],
            TmaLayout::MnMajor => [TC_BK as u32, TC_BK as u32],
        }
    }

    /// The box the *narrow* kernel's 64-row `B` stage needs, where it differs.
    ///
    /// Only the K-major layout does: MN-major boxes are already `[64, 64]`,
    /// and `A`'s stage is `[128, TC_BK]` under both tiles. `None` means the
    /// wide descriptor serves both variants.
    fn narrow_box(self) -> Option<[u32; 2]> {
        match self {
            TmaLayout::KMajor => Some([TC_BK as u32, TC_BK as u32]),
            TmaLayout::MnMajor => None,
        }
    }
}

/// One operand's tensor maps, as the launch adapters consume them: the wide
/// descriptor every `A` walk and the wide kernel's `B` walk load through, and
/// the `[64, 64]`-box twin the narrow kernel's `B` stage needs in the K-major
/// layout. Built by the map owners' `operand()`; `Copy` so closures can hold
/// it across timed launches.
#[derive(Clone, Copy)]
pub struct TmaOperand {
    wide: *const TmaDescriptor,
    narrow: *const TmaDescriptor,
}

impl TmaOperand {
    /// The descriptor an `A` operand loads through — box `[TC_BK, 128]`
    /// K-major or `[64, 64]` MN-major, identical under both tiles.
    fn a(self) -> *const TmaDescriptor {
        self.wide
    }

    /// The descriptor a `B` operand loads through at `tile`.
    fn b(self, tile: TcTile) -> *const TmaDescriptor {
        match tile {
            TcTile::Wide => self.wide,
            TcTile::Narrow => self.narrow,
        }
    }
}

/// Both variants' descriptors over a row-major `[height, width]` bf16 matrix:
/// the wide box always, the narrow `[64, 64]` K-major twin where the layout
/// makes them differ. One extra 128-byte descriptor per K-major operand is the
/// whole cost of keeping the launch free to dispatch either kernel.
struct OperandDescriptors {
    wide: DeviceBuffer<u64>,
    narrow: Option<DeviceBuffer<u64>>,
}

impl OperandDescriptors {
    fn encode(
        stream: &CudaStream,
        base: u64,
        width: usize,
        height: usize,
        row_stride: usize,
        layout: TmaLayout,
    ) -> Result<Self, Box<dyn Error>> {
        let wide = encode_bf16_tma_map_strided(stream, base, width, height, row_stride, layout)?;
        let narrow = match layout.narrow_box() {
            Some(narrow_box) => Some(encode_bf16_tma_map_boxed(
                stream, base, width, height, row_stride, narrow_box,
            )?),
            None => None,
        };
        Ok(Self { wide, narrow })
    }

    fn operand(&self) -> TmaOperand {
        let wide = self.wide.cu_deviceptr() as *const TmaDescriptor;
        TmaOperand {
            wide,
            narrow: self
                .narrow
                .as_ref()
                .map_or(wide, |map| map.cu_deviceptr() as *const TmaDescriptor),
        }
    }

    fn as_ptr(&self) -> *const TmaDescriptor {
        self.wide.cu_deviceptr() as *const TmaDescriptor
    }
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
    encode_bf16_tma_map_boxed(
        stream,
        base,
        width,
        height,
        row_stride,
        layout.box_dimensions(),
    )
}

/// The encoder itself, at an explicit box — [`TmaLayout::box_dimensions`] for
/// the wide kernel's stages, [`TmaLayout::narrow_box`] for the narrow
/// kernel's 64-row `B` stage.
fn encode_bf16_tma_map_boxed(
    stream: &CudaStream,
    base: u64,
    width: usize,
    height: usize,
    row_stride: usize,
    box_dimensions: [u32; 2],
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
    assert!(height.is_multiple_of(box_dimensions[1] as usize));
    assert!(row_stride >= width);
    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dimensions = [width as u64, height as u64];
    let global_strides = [(row_stride * 2) as u64];
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

    // The narrower tile constraint: both variants' `C` widths satisfy it, and
    // the box below is `[16, 32]` under either.
    assert!(width.is_multiple_of(TC_N_NARROW));
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
    descriptors: OperandDescriptors,
    _matrix: PhantomData<&'matrix DeviceBuffer<u16>>,
}

impl Bf16TmaMap<'_> {
    /// The wide descriptor alone — for consumers outside the tcgen05 launch
    /// adapters, which take [`Bf16TmaMap::operand`] and pick per tile.
    pub fn as_ptr(&self) -> *const TmaDescriptor {
        self.descriptors.as_ptr()
    }

    pub fn operand(&self) -> TmaOperand {
        self.descriptors.operand()
    }
}

/// Build a `SWIZZLE_128B` tensor map loading a 128x64 bf16 tile.
pub fn create_bf16_tma_map<'matrix>(
    stream: &CudaStream,
    matrix: &'matrix DeviceBuffer<u16>,
    width: usize,
    height: usize,
    layout: TmaLayout,
) -> Result<Bf16TmaMap<'matrix>, Box<dyn Error>> {
    assert_eq!(matrix.len(), width * height);
    Ok(Bf16TmaMap {
        descriptors: OperandDescriptors::encode(
            stream,
            matrix.cu_deviceptr(),
            width,
            height,
            width,
            layout,
        )?,
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
    descriptors: OperandDescriptors,
}

impl Bf16PairsTmaMap {
    /// The wide descriptor alone — see [`Bf16TmaMap::as_ptr`].
    pub fn as_ptr(&self) -> *const TmaDescriptor {
        self.descriptors.as_ptr()
    }

    pub fn operand(&self) -> TmaOperand {
        self.descriptors.operand()
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
        descriptors: OperandDescriptors::encode(
            stream,
            matrix.cu_deviceptr(),
            width,
            height,
            width,
            layout,
        )?,
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
        descriptors: OperandDescriptors::encode(
            stream,
            matrix.cu_deviceptr(),
            width,
            height,
            width,
            layout,
        )?,
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
        descriptors: OperandDescriptors::encode(stream, base, width, height, row_stride, layout)?,
    })
}

/// The optimized tcgen05 bf16 GEMM loaded from the calling binary's single
/// embedded device artifact.
pub struct Tcgen05Gemm {
    generated: super::optimized::kernels::LoadedModule,
    narrow: super::narrow::kernels::LoadedModule,
    optimized: CudaFunction,
    optimized_f32: CudaFunction,
    optimized_f32_accumulate: CudaFunction,
    narrow_bf16: CudaFunction,
    narrow_f32: CudaFunction,
    narrow_f32_accumulate: CudaFunction,
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
        let narrow = super::narrow::kernels::load(ctx)?;
        let optimized = generated
            .as_cuda_module()
            .load_function("gemm_tcgen05_bf16_optimized")?;
        let optimized_f32 = generated
            .as_cuda_module()
            .load_function("gemm_tcgen05_f32_optimized")?;
        let optimized_f32_accumulate = generated
            .as_cuda_module()
            .load_function("gemm_tcgen05_f32_accumulate")?;
        let narrow_bf16 = narrow
            .as_cuda_module()
            .load_function("gemm_tcgen05_bf16_narrow")?;
        let narrow_f32 = narrow
            .as_cuda_module()
            .load_function("gemm_tcgen05_f32_narrow")?;
        let narrow_f32_accumulate = narrow
            .as_cuda_module()
            .load_function("gemm_tcgen05_f32_accumulate_narrow")?;
        opt_in_dynamic_smem(&optimized, super::optimized::SHARED_BYTES as u32)?;
        opt_in_dynamic_smem(&optimized_f32, super::optimized::SHARED_BYTES as u32)?;
        opt_in_dynamic_smem(
            &optimized_f32_accumulate,
            super::optimized::SHARED_BYTES as u32,
        )?;
        opt_in_dynamic_smem(&narrow_bf16, super::narrow::SHARED_BYTES as u32)?;
        opt_in_dynamic_smem(&narrow_f32, super::narrow::SHARED_BYTES as u32)?;
        opt_in_dynamic_smem(&narrow_f32_accumulate, super::narrow::SHARED_BYTES as u32)?;
        Ok(Self {
            generated,
            narrow,
            optimized,
            optimized_f32,
            optimized_f32_accumulate,
            narrow_bf16,
            narrow_f32,
            narrow_f32_accumulate,
            c_maps: Mutex::new(HashMap::new()),
        })
    }

    /// The loaded kernels, named for `bench_util::enforce_kernel_budgets`.
    pub fn kernels(&self) -> [(&'static str, &CudaFunction); 6] {
        [
            ("gemm_tcgen05_bf16_optimized", &self.optimized),
            ("gemm_tcgen05_f32_optimized", &self.optimized_f32),
            (
                "gemm_tcgen05_f32_accumulate",
                &self.optimized_f32_accumulate,
            ),
            ("gemm_tcgen05_bf16_narrow", &self.narrow_bf16),
            ("gemm_tcgen05_f32_narrow", &self.narrow_f32),
            (
                "gemm_tcgen05_f32_accumulate_narrow",
                &self.narrow_f32_accumulate,
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
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
        let tiles_m = (m / TC_M_TILE) as u32;
        let tiles_n = (n as usize / launch.tile.n_tile()) as u32;
        let transposed = u32::from(layout == TmaLayout::MnMajor);
        unsafe {
            match launch.tile {
                TcTile::Wide => self.generated.gemm_tcgen05_f32_accumulate(
                    stream,
                    launch.config,
                    a.a(),
                    b.b(TcTile::Wide),
                    c_map,
                    k as i32,
                    tiles_m,
                    tiles_n,
                    transposed,
                ),
                TcTile::Narrow => self.narrow.gemm_tcgen05_f32_accumulate_narrow(
                    stream,
                    launch.config,
                    a.a(),
                    b.b(TcTile::Narrow),
                    c_map,
                    k as i32,
                    tiles_m,
                    tiles_n,
                    transposed,
                ),
            }
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
        output: &mut DeviceBuffer<u32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            launch_tcgen05(
                self,
                stream,
                launch,
                a,
                b,
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
        output: &mut DeviceBuffer<u32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            launch_tcgen05(
                self,
                stream,
                launch,
                a,
                b,
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
        output: &mut DeviceBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        let output_elements = output.len();
        unsafe {
            launch_tcgen05_f32(
                self,
                stream,
                launch,
                a,
                b,
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
        output: &mut DeviceBuffer<f32>,
        output_offset: usize,
        output_elements: usize,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            launch_tcgen05_f32(
                self,
                stream,
                launch,
                a,
                b,
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
        output: &mut DeviceBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        let output_elements = output.len();
        unsafe {
            self.launch_f32_accumulate(
                stream,
                launch,
                a,
                b,
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
        output: &mut DeviceBuffer<f32>,
        output_offset: usize,
        output_elements: usize,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            self.launch_f32_accumulate(
                stream,
                launch,
                a,
                b,
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
        output: &mut DeviceBuffer<u32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            launch_tcgen05(
                self,
                stream,
                launch,
                a,
                b,
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
        output: &mut DeviceBuffer<f32>,
        output_offset: usize,
        output_elements: usize,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            self.launch_f32_accumulate(
                stream,
                launch,
                a,
                b,
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
        output: &mut DeviceBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        let output_elements = output.len();
        unsafe {
            launch_tcgen05_f32(
                self,
                stream,
                launch,
                a,
                b,
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
        launch: Tcgen05Launch,
        a: TmaOperand,
        b: TmaOperand,
        output: &mut DeviceBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        let output_elements = output.len();
        unsafe {
            self.f32_accumulate_transposed_at(
                stream,
                launch,
                a,
                b,
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
    gemm: &Tcgen05Gemm,
    stream: &CudaStream,
    launch: Tcgen05Launch,
    a: TmaOperand,
    b: TmaOperand,
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
    let tiles_m = (m / TC_M_TILE) as u32;
    let tiles_n = (n as usize / launch.tile.n_tile()) as u32;
    let transposed = u32::from(layout == TmaLayout::MnMajor);
    unsafe {
        match launch.tile {
            TcTile::Wide => gemm.generated.gemm_tcgen05_bf16_optimized(
                stream,
                launch.config,
                a.a(),
                b.b(TcTile::Wide),
                output,
                n as i32,
                k as i32,
                tiles_m,
                tiles_n,
                mode,
                transposed,
            ),
            TcTile::Narrow => gemm.narrow.gemm_tcgen05_bf16_narrow(
                stream,
                launch.config,
                a.a(),
                b.b(TcTile::Narrow),
                output,
                n as i32,
                k as i32,
                tiles_m,
                tiles_n,
                mode,
                transposed,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_tcgen05_f32(
    gemm: &Tcgen05Gemm,
    stream: &CudaStream,
    launch: Tcgen05Launch,
    a: TmaOperand,
    b: TmaOperand,
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
    let tiles_m = (m / TC_M_TILE) as u32;
    let tiles_n = (n as usize / launch.tile.n_tile()) as u32;
    let transposed = u32::from(layout == TmaLayout::MnMajor);
    unsafe {
        match launch.tile {
            TcTile::Wide => gemm.generated.gemm_tcgen05_f32_optimized(
                stream,
                launch.config,
                a.a(),
                b.b(TcTile::Wide),
                output,
                output_offset,
                n as i32,
                k as i32,
                tiles_m,
                tiles_n,
                transposed,
            ),
            TcTile::Narrow => gemm.narrow.gemm_tcgen05_f32_narrow(
                stream,
                launch.config,
                a.a(),
                b.b(TcTile::Narrow),
                output,
                output_offset,
                n as i32,
                k as i32,
                tiles_m,
                tiles_n,
                transposed,
            ),
        }
    }
}
