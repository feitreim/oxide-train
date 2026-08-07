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
/// Dynamic shared bytes of the score_mma probe: the paired `[QUERIES, HD]` A
/// operand the `M128` accumulator names, plus the B panel.
pub const PROBE_DYNAMIC_SMEM_BYTES: u32 = (3 * TILE_BYTES) as u32;
/// Dynamic shared bytes of the MMA cadence probe: its `[128, HD]` A panel and
/// the `[256, HD]` B panel the three widths are read out of.
pub const CADENCE_SMEM_BYTES: u32 = super::tcgen05::phase_probe::CADENCE_SMEM as u32;
/// Dynamic shared bytes of the query-parallel backward — **exactly** the
/// kernel's own plan, for the reason the forward's is exact. Neither backward
/// kernel can reach two CTAs an SM (both take all 512 tensor-memory columns),
/// so nothing here is bought by trimming it; an exact request is simply the
/// only one that stays true when `BACKWARD_STAGES` moves. Mirrors
/// `FLASH_BACKWARD_Q_SMEM` in `tcgen05.rs`.
pub const FLASH_BACKWARD_Q_SMEM_BYTES: u32 = super::tcgen05::FLASH_BACKWARD_Q_SMEM as u32;
/// Dynamic shared bytes of the key-parallel backward. The largest plan in the
/// repo: it carries a second gradient operand ring, since its MMA pair reads
/// `Pᵀ` and `dSᵀ` where kernel A's reads only `dS`.
pub const FLASH_BACKWARD_KV_SMEM_BYTES: u32 = super::tcgen05::FLASH_BACKWARD_KV_SMEM as u32;
/// Threads of either backward kernel: the `PASS_GROUPS` warpgroups that share
/// an `M128` accumulator's 128 TMEM lanes and split its columns between them,
/// plus the warp that issues every TMA and MMA. Mirrors
/// `FLASH_BACKWARD_BLOCK`, and `flash.rs`'s `main` asserts the two agree.
pub const FLASH_BACKWARD_BLOCK_THREADS: u32 = super::tcgen05::FLASH_BACKWARD_BLOCK as u32;
/// Dynamic shared allocation for the forward: **exactly** the kernel's own
/// plan, not a ceiling sized for the deepest supported `FORWARD_STAGES`.
///
/// The launch's request is what the driver charges an SM, so a ceiling is not
/// free the moment the plan is small enough to admit a second CTA — it pins
/// residency at the ceiling's. That was invisible while the plan was 164 KiB
/// and is the whole point at 112.
pub const FLASH_FORWARD_SMEM_BYTES: u32 = super::tcgen05::FLASH_FORWARD_SMEM as u32;
/// Shared memory an SM gives its CTAs, and tensor-memory columns it has.
/// B200 numbers, which is the only target (decision #14); ferro-kittens #84
/// measured residency as exactly the `min` of what the two admit, at every
/// rung it tried.
const SM_SHARED_BYTES: usize = 233_472;
const SM_TMEM_COLUMNS: usize = 512;
/// CTAs of the forward an SM admits, and so the multiplier the persistent grid
/// takes over the SM count. Two at `FORWARD_STAGES = 2`, one above it.
pub const FLASH_FORWARD_CTAS_PER_SM: usize = {
    let by_shared = SM_SHARED_BYTES / FLASH_FORWARD_SMEM_BYTES as usize;
    let by_tmem = SM_TMEM_COLUMNS / super::tcgen05::FORWARD_TMEM_COLUMNS as usize;
    if by_shared < by_tmem {
        by_shared
    } else {
        by_tmem
    }
};
const _: () = assert!(
    FLASH_FORWARD_CTAS_PER_SM >= 1,
    "the forward's plan does not fit one CTA on an SM"
);
/// Threads of the forward: the four warps an `M128` accumulator's 128 TMEM
/// lanes are drained by, and no others. Mirrors `FLASH_FORWARD_BLOCK`.
pub const FLASH_FORWARD_BLOCK_THREADS: u32 = FLASH_QUERIES as u32;

/// Launch for either backward kernel: a 1-D persistent grid over the same
/// work-item space the forward's takes, one item per (block, head, batch).
///
/// No residency multiplier, unlike `flash_forward_config`: both kernels
/// allocate all 512 tensor-memory columns, so an SM holds one whatever their
/// shared plan costs. `T` must be a multiple of `FLASH_QUERIES`; other shapes
/// stay on the fp32 tiled backward.
fn flash_backward_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
    cta_count: usize,
    dynamic_smem: u32,
) -> LaunchConfig {
    assert!(batches <= u16::MAX as usize && heads <= u16::MAX as usize);
    let items = flash_work_items(batches, sequence_length, heads);
    assert!(items > 0 && items <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (items.min(cta_count.max(1)) as u32, 1, 1),
        block_dim: (FLASH_BACKWARD_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: dynamic_smem,
    }
}

/// Launch for the query-parallel backward (kernel A).
pub fn flash_backward_q_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
    cta_count: usize,
) -> LaunchConfig {
    flash_backward_config(
        batches,
        sequence_length,
        heads,
        cta_count,
        FLASH_BACKWARD_Q_SMEM_BYTES,
    )
}

/// Launch for the key-parallel backward (kernel B).
pub fn flash_backward_kv_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
    cta_count: usize,
) -> LaunchConfig {
    flash_backward_config(
        batches,
        sequence_length,
        heads,
        cta_count,
        FLASH_BACKWARD_KV_SMEM_BYTES,
    )
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

/// Launch for the forward: a 1-D persistent grid of
/// `cta_count * FLASH_FORWARD_CTAS_PER_SM` CTAs (`cta_count` is normally the SM
/// count; clamped to the work-item count, so passing the item count degenerates
/// to one item per CTA for hang debugging).
///
/// The residency multiplier is the caller's business only in that it must not
/// have to know it: a grid of one CTA per SM would leave half the SM idle at a
/// plan that admits two.
pub fn flash_forward_config(
    batches: usize,
    sequence_length: usize,
    heads: usize,
    cta_count: usize,
) -> LaunchConfig {
    assert!(batches <= u16::MAX as usize && heads <= u16::MAX as usize);
    let items = flash_work_items(batches, sequence_length, heads);
    assert!(items > 0 && items <= u32::MAX as usize);
    let ctas = cta_count.max(1) * FLASH_FORWARD_CTAS_PER_SM;
    LaunchConfig {
        grid_dim: (items.min(ctas) as u32, 1, 1),
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

/// Build a head-panel tensor map over a **row-major** packed-pair
/// `[batches * sequence_length, heads * 128]` panel — the layout a projection
/// GEMM writes, addressed as if it were the `[batches * heads, T, 128]`
/// staging buffer [`create_flash_head_tma_map`] describes.
///
/// The relayout is the descriptor's, not a kernel's: a head is a column band
/// `FLASH_HD` wide, so it is dimension 2 at stride `FLASH_HD`, while a token
/// is dimension 1 at the panel's full row stride. `(batch, token)` collapses
/// into that one row coordinate because `batch * T + token` is exactly the
/// panel's row index, which is why an operand nobody relaid out still arrives
/// as `[TILE, HD]` head tiles. Consumers pass `row = batch * T + token` and
/// `plane = head` where a staged panel takes `row = token` and
/// `plane = batch * heads + head`.
///
/// # Safety
///
/// `buffer` must stay allocated at the same device address for every kernel
/// launch that consumes the returned map.
pub unsafe fn create_flash_row_major_tma_map(
    stream: &CudaStream,
    buffer: &DeviceBuffer<u32>,
    rows: usize,
    heads: usize,
) -> Result<FlashHeadTmaMap, Box<dyn Error>> {
    assert_eq!(buffer.len() * 2, rows * heads * FLASH_HD);
    assert!(rows.is_multiple_of(FLASH_TILE));
    let width = heads * FLASH_HD;
    let layout = unsafe {
        kittens::global::GlobalLayout::<kittens::shared::Bf16, 3>::strided(
            buffer.cu_deviceptr(),
            [FLASH_HD, rows, heads],
            [1, width, FLASH_HD],
        )
    };
    layout.tensor_map::<kittens::shared::SharedTile<
        kittens::shared::Bf16,
        FLASH_TILE,
        FLASH_HD,
        kittens::shared::Swizzle128B,
    >>(stream)
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
    backward_kv: CudaFunction,
    sm_count: usize,
}

impl Tcgen05Flash {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn Error>> {
        let generated = super::tcgen05::kernels::load(ctx)?;
        let module = generated.as_cuda_module().clone();
        let forward = module.load_function("flash_forward")?;
        let backward_q = module.load_function("flash_backward_q")?;
        let backward_kv = module.load_function("flash_backward_kv")?;
        let transpose_probe = module.load_function("transpose_b_probe")?;
        let backward_q_probe = module.load_function("flash_backward_q_probe")?;
        let backward_kv_probe = module.load_function("flash_backward_kv_probe")?;
        let cadence = module.load_function("mma_cadence")?;
        opt_in_dynamic_smem(&cadence, CADENCE_SMEM_BYTES)?;
        opt_in_dynamic_smem(&forward, FLASH_FORWARD_SMEM_BYTES)?;
        opt_in_dynamic_smem(&backward_q, FLASH_BACKWARD_Q_SMEM_BYTES)?;
        opt_in_dynamic_smem(&backward_kv, FLASH_BACKWARD_KV_SMEM_BYTES)?;
        opt_in_dynamic_smem(&transpose_probe, PROBE_DYNAMIC_SMEM_BYTES)?;
        opt_in_dynamic_smem(&backward_q_probe, FLASH_BACKWARD_Q_SMEM_BYTES)?;
        opt_in_dynamic_smem(&backward_kv_probe, FLASH_BACKWARD_KV_SMEM_BYTES)?;
        Ok(Self {
            generated,
            forward,
            backward_q,
            backward_kv,
            sm_count: device_sm_count(ctx)?,
        })
    }

    /// Time one chained `tcgen05.mma` walk per shape variant, from a single
    /// warp of a single CTA.
    ///
    /// # Safety
    ///
    /// `clocks` holds `tcgen05::phase_probe::CADENCE_COUNTERS` zeroed `u64`.
    pub unsafe fn mma_cadence(
        &self,
        stream: &CudaStream,
        rounds: u32,
        clocks: &mut DeviceBuffer<u64>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.generated.mma_cadence(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: CADENCE_SMEM_BYTES,
                },
                rounds,
                clocks,
            )
        }
    }

    /// [`Self::backward_q`] with the phase stopwatch, writing
    /// `tcgen05::phase_probe::COUNTERS` ticks per CTA into `clocks`.
    ///
    /// # Safety
    ///
    /// As [`Self::backward_q`]; `clocks` holds `COUNTERS * config.grid_dim.0`
    /// zeroed elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn backward_q_probe(
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
        batches: u32,
        dq: &mut DeviceBuffer<f32>,
        clocks: &mut DeviceBuffer<u64>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.generated.flash_backward_q_probe(
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
                batches,
                dq,
                clocks,
            )
        }
    }

    /// [`Self::backward_kv`] with the phase stopwatch.
    ///
    /// # Safety
    ///
    /// As [`Self::backward_kv`] and [`Self::backward_q_probe`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn backward_kv_probe(
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
        batches: u32,
        dk: &mut DeviceBuffer<f32>,
        dv: &mut DeviceBuffer<f32>,
        clocks: &mut DeviceBuffer<u64>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.generated.flash_backward_kv_probe(
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
                batches,
                dk,
                dv,
                clocks,
            )
        }
    }

    /// SM count captured at load time — the natural `cta_count` for
    /// `flash_forward_config`.
    pub fn sm_count(&self) -> usize {
        self.sm_count
    }

    /// The launched kernels paired with their names, for reporting what ptxas
    /// gave each one.
    pub fn kernels(&self) -> [(&'static str, &CudaFunction); 3] {
        [
            ("forward", &self.forward),
            ("backward q", &self.backward_q),
            ("backward kv", &self.backward_kv),
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

    /// tcgen05 causal attention backward, query-parallel half: writes fp32
    /// `dq[B*T, H*HD]` from the bf16 head-panel staging buffers plus the saved
    /// `logsumexp[B*T, H]` (natural log) and `dot[B*T, H]`. Launch with
    /// `flash_backward_q_config`.
    ///
    /// # Safety
    ///
    /// The maps must describe live `[B*H, T, HD]` staging buffers matching the
    /// launch config (`dy` staged unscaled like K/V), `logsumexp`/`dot` must
    /// hold `B*T*H` elements, and `dq` must hold `B*T*H*HD` elements.
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
        batches: u32,
        dq: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.generated.flash_backward_q(
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
                batches,
                dq,
            )
        }
    }

    /// tcgen05 causal attention backward, key-parallel half: writes fp32
    /// `dk`/`dv` `[B*T, H*HD]` from the same staged operands and statistics.
    /// Launch with `flash_backward_kv_config`.
    ///
    /// # Safety
    ///
    /// Same operand/statistic contract as [`Self::backward_q`]; `dk` and `dv`
    /// must each hold `B*T*H*HD` elements.
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
        batches: u32,
        dk: &mut DeviceBuffer<f32>,
        dv: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        unsafe {
            self.generated.flash_backward_kv(
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
                batches,
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
