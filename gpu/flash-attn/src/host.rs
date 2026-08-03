//! Host-side support for the tcgen05 attention kernels: staging-buffer TMA
//! maps, launch configs, and ergonomic adapters over generated typed launchers.

use std::error::Error;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaFunction, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::tma::TmaDescriptor;

/// Key tile edge — and half a query block, since the forward's `M128` MMAs
/// take two stacked tiles of queries.
pub const FLASH_TILE: usize = 64;
/// Query rows one forward CTA owns. `T` must be a multiple of this to use the
/// tcgen05 forward; other shapes stay on the fp32 tiled kernels.
pub const FLASH_QUERIES: usize = 2 * FLASH_TILE;
/// The only head width the tcgen05 forward supports.
pub const FLASH_HD: usize = 128;
/// SWIZZLE_128B subtile width: a 128-wide operand is two stacked `[TILE, 64]`
/// subtiles, which is also the TMA descriptor's box column count.
pub const FLASH_SUBTILE_HD: usize = 64;
/// Bytes of one full-width `[TILE, HD]` bf16 panel (two stacked subtiles).
const TILE_BYTES: usize = FLASH_TILE * FLASH_HD * 2;
/// Dynamic shared bytes of the score_mma probe: A panel plus B panel.
pub const PROBE_DYNAMIC_SMEM_BYTES: u32 = (2 * TILE_BYTES) as u32;
/// Dynamic shared bytes of the PAIRED query-parallel backward (kernel A,
/// Design B): resident stacked `[Q_A;Q_B]`/`[dY_A;dY_B]` (`2 * TILE_BYTES`
/// each), streamed K/V panels, and the stacked `[128, 64]` dS tile. Mirrors
/// `FLASH_BACKWARD_Q_SMEM` in `tcgen05.rs`.
pub const FLASH_BACKWARD_Q_SMEM_BYTES: u32 = (7 * TILE_BYTES) as u32;
/// Dynamic shared bytes of the PAIRED key-parallel backward (kernel B, Design
/// B): resident stacked `[K_A;K_B]`/`[V_A;V_B]`, streamed Q/dY panels, and the
/// stacked Pᵀ and dSᵀ tiles. Mirrors `FLASH_BACKWARD_KV_SMEM`.
pub const FLASH_BACKWARD_KV_SMEM_BYTES: u32 = (8 * TILE_BYTES) as u32;
/// Dynamic shared allocation for the PIPELINED query-parallel backward: the
/// resident Q/dY pairs, K/V rings sized for the deepest supported
/// `BACKWARD_STAGES` (4), and the single stacked dS tile. As with the forward,
/// the ceiling is allocated so stage sweeps are a one-const edit and the flash
/// bin asserts the kernel's actual plan fits. Mirrors
/// `FLASH_BACKWARD_Q_PIPELINED_SMEM` in `tcgen05.rs`.
pub const FLASH_BACKWARD_Q_PIPELINED_SMEM_BYTES: u32 = ((5 + 2 * 4) * TILE_BYTES) as u32;
/// Threads of the pipelined backward: the 128-thread gradient warpgroup plus
/// the TMA-load and MMA-issue warps. Mirrors
/// `FLASH_BACKWARD_Q_PIPELINED_BLOCK`.
pub const FLASH_BACKWARD_Q_PIPELINED_BLOCK_THREADS: u32 = (FLASH_QUERIES + 64) as u32;
/// Dynamic shared allocation for the PIPELINED key-parallel backward: the
/// resident K/V pairs, Q/dY rings sized for the deepest supported
/// `BACKWARD_STAGES` (4), and the stacked Pᵀ and dSᵀ tiles. Mirrors
/// `FLASH_BACKWARD_KV_PIPELINED_SMEM` in `tcgen05.rs`.
pub const FLASH_BACKWARD_KV_PIPELINED_SMEM_BYTES: u32 = ((6 + 2 * 4) * TILE_BYTES) as u32;
/// Threads of the pipelined key-parallel backward. Mirrors
/// `FLASH_BACKWARD_KV_PIPELINED_BLOCK`.
pub const FLASH_BACKWARD_KV_PIPELINED_BLOCK_THREADS: u32 = (FLASH_QUERIES + 64) as u32;
/// Dynamic shared allocation for the forward: the resident `[QUERIES, HD]`
/// query block (`2 * TILE_BYTES`), K/V rings sized for the deepest supported
/// `FORWARD_STAGES` (4), the two-deep `[QUERIES, TILE]` probability ring
/// (`2 * TILE_BYTES`), and a page for the barriers and the vote scratch the
/// plan carves after them. The kernel's actual plan (`FLASH_FORWARD_SMEM`, a
/// function of the swept `FORWARD_STAGES`) must stay at or under this; the
/// flash bin asserts it. Allocating the ceiling keeps stage sweeps a one-const
/// edit, and costs nothing at 1 CTA/SM.
pub const FLASH_FORWARD_SMEM_BYTES: u32 = ((4 + 2 * 4) * TILE_BYTES + 1024) as u32;
/// Threads of the forward: the four warps an `M128` accumulator's 128 TMEM
/// lanes are drained by, and no others. Mirrors `FLASH_FORWARD_BLOCK`.
pub const FLASH_FORWARD_BLOCK_THREADS: u32 = FLASH_QUERIES as u32;

/// Launch for both PAIRED tcgen05 backward kernels (Design B): each CTA owns a
/// tile PAIR, so the grid is `(T/128, H, B)` and the block is 128 threads (the
/// 4-warp paired warpgroup draining 128 rows). `T` must be a multiple of 128
/// (two 64-row tiles); non-pairable shapes stay on the fp32 tiled backward.
/// `dynamic_smem` is the caller's kernel-A or kernel-B shared-memory plan.
fn flash_backward_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
    dynamic_smem: u32,
) -> LaunchConfig {
    assert!(sequence_length.is_multiple_of(2 * FLASH_TILE));
    assert!(batches <= u16::MAX as usize && heads <= u16::MAX as usize);
    assert!(sequence_length / (2 * FLASH_TILE) <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            (sequence_length / (2 * FLASH_TILE)) as u32,
            heads as u32,
            batches as u32,
        ),
        block_dim: ((2 * FLASH_TILE) as u32, 1, 1),
        shared_mem_bytes: dynamic_smem,
    }
}

/// Launch for the query-parallel backward (kernel A).
pub fn flash_backward_q_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
) -> LaunchConfig {
    flash_backward_config(batches, sequence_length, heads, FLASH_BACKWARD_Q_SMEM_BYTES)
}

/// Launch for the key-parallel backward (kernel B).
pub fn flash_backward_kv_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
) -> LaunchConfig {
    flash_backward_config(
        batches,
        sequence_length,
        heads,
        FLASH_BACKWARD_KV_SMEM_BYTES,
    )
}

/// Launch for the warp-specialized pipelined query-parallel backward: same
/// grid as kernel A, the wider block, the ring-sized shared allocation.
pub fn flash_backward_q_pipelined_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
) -> LaunchConfig {
    let base = flash_backward_q_config(batches, sequence_length, heads);
    LaunchConfig {
        grid_dim: base.grid_dim,
        block_dim: (FLASH_BACKWARD_Q_PIPELINED_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: FLASH_BACKWARD_Q_PIPELINED_SMEM_BYTES,
    }
}

/// Launch for the warp-specialized pipelined key-parallel backward.
pub fn flash_backward_kv_pipelined_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
) -> LaunchConfig {
    let base = flash_backward_kv_config(batches, sequence_length, heads);
    LaunchConfig {
        grid_dim: base.grid_dim,
        block_dim: (FLASH_BACKWARD_KV_PIPELINED_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: FLASH_BACKWARD_KV_PIPELINED_SMEM_BYTES,
    }
}

/// Work items of the forward — one per (query block, head, batch), which is
/// also the length of the correction-count buffer.
pub fn flash_work_items(batches: usize, sequence_length: usize, heads: usize) -> usize {
    assert!(sequence_length.is_multiple_of(FLASH_QUERIES));
    (sequence_length / FLASH_QUERIES) * heads * batches
}

/// Elements of the correction-count output: one word per work item, indexed
/// `plane * tiles + query_tile`.
pub fn correction_count_len(batches: usize, sequence_length: usize, heads: usize) -> usize {
    flash_work_items(batches, sequence_length, heads)
}

/// SM count of the device backing `ctx`, for sizing the persistent grid.
pub fn device_sm_count(ctx: &CudaContext) -> Result<usize, Box<dyn Error>> {
    use cuda_core::sys::{
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDeviceGetAttribute,
        cudaError_enum_CUDA_SUCCESS,
    };
    let mut count = 0i32;
    let status = unsafe {
        cuDeviceGetAttribute(
            &mut count,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            ctx.cu_device(),
        )
    };
    if status != cudaError_enum_CUDA_SUCCESS {
        return Err(
            format!("cuDeviceGetAttribute(multiprocessor count) failed: {status:?}").into(),
        );
    }
    Ok(count as usize)
}

/// Launch for the forward: a 1-D persistent grid of `cta_count` CTAs (normally
/// the SM count; clamped to the work-item count, so passing the item count
/// degenerates to one item per CTA for hang debugging).
pub fn flash_forward_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
    cta_count: usize,
) -> LaunchConfig {
    assert!(batches <= u16::MAX as usize && heads <= u16::MAX as usize);
    let items = flash_work_items(batches, sequence_length, heads);
    assert!(items > 0 && items <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (items.min(cta_count.max(1)) as u32, 1, 1),
        block_dim: (FLASH_FORWARD_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: FLASH_FORWARD_SMEM_BYTES,
    }
}

/// Tensor map over one packed-bf16 `[planes, T, 128]` staging buffer,
/// encoded by kittens' layout-generic builder (`[FLASH_TILE, 64]` swizzled
/// boxes, one per stacked HD subtile).
pub type FlashHeadTmaMap = kittens::global::PanelMap;

/// Build a head-panel tensor map over a packed-pair staging buffer holding
/// `planes` panels of `[sequence_length, 128]` bf16 values.
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
    unsafe {
        kittens::global::encode_bf16_panels::<FLASH_TILE, FLASH_HD>(
            stream,
            buffer.cu_deviceptr(),
            sequence_length,
            planes,
        )
    }
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

/// The tcgen05 attention kernels loaded from the calling binary's single
/// embedded device artifact.
pub struct Tcgen05Flash {
    generated: super::tcgen05::kernels::LoadedModule,
    forward: CudaFunction,
    backward_q: CudaFunction,
    backward_q_pipelined: CudaFunction,
    backward_kv: CudaFunction,
    backward_kv_pipelined: CudaFunction,
    sm_count: usize,
}

impl Tcgen05Flash {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn Error>> {
        let generated = super::tcgen05::kernels::load(ctx)?;
        let module = generated.as_cuda_module().clone();
        let forward = module.load_function("flash_forward")?;
        let backward_q = module.load_function("flash_backward_q_tcgen05")?;
        let backward_q_pipelined = module.load_function("flash_backward_q_pipelined")?;
        let backward_kv = module.load_function("flash_backward_kv_tcgen05")?;
        let backward_kv_pipelined = module.load_function("flash_backward_kv_pipelined")?;
        let transpose_probe = module.load_function("transpose_b_probe")?;
        opt_in_dynamic_smem(&forward, FLASH_FORWARD_SMEM_BYTES)?;
        opt_in_dynamic_smem(&backward_q, FLASH_BACKWARD_Q_SMEM_BYTES)?;
        opt_in_dynamic_smem(&backward_q_pipelined, FLASH_BACKWARD_Q_PIPELINED_SMEM_BYTES)?;
        opt_in_dynamic_smem(&backward_kv, FLASH_BACKWARD_KV_SMEM_BYTES)?;
        opt_in_dynamic_smem(
            &backward_kv_pipelined,
            FLASH_BACKWARD_KV_PIPELINED_SMEM_BYTES,
        )?;
        opt_in_dynamic_smem(&transpose_probe, PROBE_DYNAMIC_SMEM_BYTES)?;
        Ok(Self {
            generated,
            forward,
            backward_q,
            backward_q_pipelined,
            backward_kv,
            backward_kv_pipelined,
            sm_count: device_sm_count(ctx)?,
        })
    }

    /// SM count captured at load time — the natural `cta_count` for
    /// `flash_forward_config`.
    pub fn sm_count(&self) -> usize {
        self.sm_count
    }

    /// The launched kernels paired with their names, for reporting what ptxas
    /// gave each one.
    pub fn kernels(&self) -> [(&'static str, &CudaFunction); 5] {
        [
            ("forward", &self.forward),
            ("backward q", &self.backward_q),
            ("backward q pipelined", &self.backward_q_pipelined),
            ("backward kv", &self.backward_kv),
            ("backward kv pipelined", &self.backward_kv_pipelined),
        ]
    }

    /// tcgen05 causal attention forward over bf16 head-panel staging buffers.
    /// Launch with `flash_forward_config`.
    ///
    /// # Safety
    ///
    /// The maps must describe live `[B*H, T, HD]` staging buffers matching the
    /// launch config, `output` must hold `B*T*H*HD` elements, `logsumexp`
    /// `B*T*H` elements, and `correction_counts` `correction_count_len`
    /// elements.
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
        batches: u32,
        output: &mut DeviceBuffer<f32>,
        logsumexp: &mut DeviceBuffer<f32>,
        correction_counts: &mut DeviceBuffer<u32>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.generated.flash_forward(
                stream,
                config,
                q_tma,
                k_tma,
                v_tma,
                sequence_length,
                heads,
                batches,
                output,
                logsumexp,
                correction_counts,
            )
        }
    }

    /// Synchronous tcgen05 query-parallel backward (kernel A): writes fp32
    /// `dq[B*T, H*64]` from the bf16 head-panel staging buffers plus the saved
    /// `logsumexp[B*T, H]` (natural log) and `dot[B*T, H]`. Launch with
    /// `flash_backward_q_config`.
    ///
    /// # Safety
    ///
    /// The maps must describe live `[B*H, T, 64]` staging buffers matching the
    /// launch config (`dy` staged unscaled like K/V), `logsumexp`/`dot` must
    /// hold `B*T*H` elements, and `dq` must hold `B*T*H*64` elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn backward_q(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        logsumexp: &DeviceBuffer<f32>,
        dot: &DeviceBuffer<f32>,
        sequence_length: u32,
        heads: u32,
        dq: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.generated.flash_backward_q_tcgen05(
                stream,
                config,
                q_tma,
                k_tma,
                v_tma,
                dy_tma,
                logsumexp,
                dot,
                sequence_length,
                heads,
                dq,
            )
        }
    }

    /// The warp-specialized pipelined query-parallel backward: identical
    /// contract to [`Self::backward_q`], launched with
    /// `flash_backward_q_pipelined_config`.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::backward_q`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn backward_q_pipelined(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        logsumexp: &DeviceBuffer<f32>,
        dot: &DeviceBuffer<f32>,
        sequence_length: u32,
        heads: u32,
        dq: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.generated.flash_backward_q_pipelined(
                stream,
                config,
                q_tma,
                k_tma,
                v_tma,
                dy_tma,
                logsumexp,
                dot,
                sequence_length,
                heads,
                dq,
            )
        }
    }

    /// Synchronous tcgen05 key-parallel backward (kernel B): writes fp32
    /// `dk`/`dv` `[B*T, H*64]` from the same staged operands and statistics.
    /// Launch with `flash_backward_kv_config`.
    ///
    /// # Safety
    ///
    /// Same operand/statistic contract as `backward_q`; `dk` and `dv` must
    /// each hold `B*T*H*64` elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn backward_kv(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        logsumexp: &DeviceBuffer<f32>,
        dot: &DeviceBuffer<f32>,
        sequence_length: u32,
        heads: u32,
        dk: &mut DeviceBuffer<f32>,
        dv: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.generated.flash_backward_kv_tcgen05(
                stream,
                config,
                q_tma,
                k_tma,
                v_tma,
                dy_tma,
                logsumexp,
                dot,
                sequence_length,
                heads,
                dk,
                dv,
            )
        }
    }

    /// The warp-specialized pipelined key-parallel backward: identical
    /// contract to [`Self::backward_kv`], launched with
    /// `flash_backward_kv_pipelined_config`.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::backward_kv`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn backward_kv_pipelined(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q_tma: *const TmaDescriptor,
        k_tma: *const TmaDescriptor,
        v_tma: *const TmaDescriptor,
        dy_tma: *const TmaDescriptor,
        logsumexp: &DeviceBuffer<f32>,
        dot: &DeviceBuffer<f32>,
        sequence_length: u32,
        heads: u32,
        dk: &mut DeviceBuffer<f32>,
        dv: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.generated.flash_backward_kv_pipelined(
                stream,
                config,
                q_tma,
                k_tma,
                v_tma,
                dy_tma,
                logsumexp,
                dot,
                sequence_length,
                heads,
                dk,
                dv,
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
        unsafe {
            self.generated.transpose_b_probe(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (FLASH_TILE as u32, 1, 1),
                    shared_mem_bytes: PROBE_DYNAMIC_SMEM_BYTES,
                },
                a_tma,
                b_tma,
                output,
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
        unsafe {
            self.generated.swizzle_probe(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (FLASH_TILE as u32, 1, 1),
                    shared_mem_bytes: (FLASH_TILE * FLASH_HD * 2) as u32,
                },
                src_tma,
                output,
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
        assert_eq!(input.len(), output.len());
        unsafe {
            self.generated.software_exp2(
                stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                input,
                output,
            )
        }
    }

    /// Elementwise software-`log2` accuracy oracle.
    pub fn software_log2(
        &self,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        assert_eq!(input.len(), output.len());
        unsafe {
            self.generated.software_log2(
                stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                input,
                output,
            )
        }
    }
}
