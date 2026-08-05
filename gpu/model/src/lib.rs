//! GPU forward and backward for the single-block reference Dense. Aligned
//! training shapes run the lm-head and block linears through bf16 tcgen05
//! against bf16 master weights (#57) and fp32 gradients; small parity shapes
//! retain the fp32 register-tiled block-linear oracle, which widens the master
//! on the way in.
//!
//! Parameters, gradients, and saved activations remain GPU-resident. The
//! implementation mirrors `nn::Dense` explicitly so residual splits and the
//! aliasing story stay visible. Since 7e2, activations and scratch live in a
//! persistent `GpuDenseWorkspace` reused across steps; safety comes from
//! disjoint workspace fields (each saved activation has a dedicated buffer),
//! not from the CPU reference's by-value Ctx ownership.
//!
//! Two padded dimensions keep every head GEMM inside the tcgen05 tile
//! contract without touching the tuned kernel:
//! - `VP` pads the vocabulary (50,257 -> 50,432). The padded weight columns
//!   are zero at initialization and stay zero: the classifier backward writes
//!   exact zeros there, so their gradients, moments, and decayed masters never
//!   move. Checkpoints store the unpadded columns only.
//! - `NP` pads the token rows to the 128-row tile. Padded rows of the head
//!   input are zeroed once and never written, so they contribute exactly
//!   nothing to any product (including the `K = NP` weight-gradient GEMM).

use std::error::Error;

use bench_util::{KernelProfiler, NoopProfiler};
use cuda_core::{
    CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig, PinnedHostBuffer,
};
use nn::{Dense, MoeBlock, MoeDense};
use optim::{
    AdamWConfig, AuxLossSchedule, MasterRounding, MuonConfig, NEWTON_SCHULZ_A, NEWTON_SCHULZ_B,
    NEWTON_SCHULZ_C, NEWTON_SCHULZ_EPSILON,
};
use tensor_core::{Rank1, Rank2, Rank3, Rank4, Shape, rng::stream_seed};

// cuda-oxide collects kernels from the selected binary target. The binary
// includes this file as a module, which in turn includes each canonical kernel
// source here instead of copying definitions or relying on dependency PTX.
//
// At cuda-oxide b099f64, libdevice calls and tcgen05 lowerings coexist on the
// pure-PTX path. The canonical GEMM and flash modules are therefore included
// in this binary and loaded from the same embedded artifact as every other
// model kernel.
#[path = "../../ops/src/lib.rs"]
mod dense_device;
#[path = "../../flash-attn/src/lib.rs"]
mod flash_device;
#[path = "../../gemm/src/lib.rs"]
#[allow(unused_imports)]
mod gemm_device;
#[path = "../../tensor-gpu/src/lib.rs"]
#[allow(dead_code)]
pub mod tensor_device;

pub use dense_device::kernels as dense_kernels;
use dense_device::{
    NORM_TILE_BLOCK_ROWS, NORM_TILE_CHUNK, NORM_TILE_THREADS, QUAD_LANES, SWIGLU_TILE_BLOCK_ROWS,
    SWIGLU_TILE_CHUNK, SWIGLU_TILE_THREADS,
};

/// The tile RMSNorm forward's launch, or `None` at a shape it cannot cover.
///
/// The kernel bounds-checks nothing, so this is the single place the
/// divisibility is decided and `rms_norm_forward_fast` is the arm every other
/// shape takes.
fn norm_tiles(rows: usize, columns: usize) -> Option<LaunchConfig> {
    (rows.is_multiple_of(NORM_TILE_BLOCK_ROWS) && columns.is_multiple_of(NORM_TILE_CHUNK)).then(
        || LaunchConfig {
            grid_dim: ((rows / NORM_TILE_BLOCK_ROWS) as u32, 1, 1),
            block_dim: (NORM_TILE_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        },
    )
}

/// The tile-SwiGLU launch for a `rows x columns` rectangle, or `None` when the
/// shape does not divide the tile the way the kernels require.
///
/// The tile kernels bounds-check nothing, so this is where the divisibility is
/// decided; every training shape in `bin/train.rs` divides, and the flat
/// kernels stay as the arm anything else takes. The bf16 forward has no tile
/// arm on purpose — it measured 0.70x of the flat one, which already stores a
/// packed pair per thread (#70). Only the dense (non-MoE) FFN takes this
/// launch now: the expert path reads its gate/up activation interleaved and
/// its fused kernels size their own flat launches.
fn swiglu_tiles(rows: usize, columns: usize) -> Option<LaunchConfig> {
    (rows.is_multiple_of(SWIGLU_TILE_BLOCK_ROWS) && columns.is_multiple_of(SWIGLU_TILE_CHUNK)).then(
        || LaunchConfig {
            grid_dim: ((rows / SWIGLU_TILE_BLOCK_ROWS) as u32, 1, 1),
            block_dim: (SWIGLU_TILE_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        },
    )
}
pub use flash_device::host::Tcgen05Flash;
pub use flash_device::kernels as flash_kernels;
pub use gemm_device::fp32::kernels as gemm_kernels;
pub use gemm_device::host::Tcgen05Gemm;
pub use tensor_device::kernels as tensor_kernels;

use flash_device::host as flash_host;
// `flash_forward_config` is deliberately NOT imported: this module has its own
// generic one for the fp32 tiled path (below), and #74 renamed the host
// function into that collision. The tcgen05 call site qualifies it, which is
// what the backward configs beside it already did.
use flash_device::host::{
    FLASH_HD, FLASH_QUERIES, FlashHeadTmaMap, correction_count_len, create_flash_head_tma_map,
};
use gemm_device::fp32_launch_config;
use gemm_device::host::{
    Bf16PairsTmaMap, TC_K_PIPELINE, TC_M_TILE, TC_N_TILE, TC_TILE, TmaLayout,
    create_bf16_pairs_tma_map, create_bf16_pairs_tma_map_prefix, create_bf16_pairs_tma_map_region,
    tcgen05_launch_config,
};
use tensor_device::{
    GpuAdamWMoments, GpuBf16Tensor, GpuMuonMomentum, GpuTensor, MASTER_ROUNDING_NEAREST,
    MASTER_ROUNDING_STOCHASTIC, MasterAdamW, master_transpose_config, pack_bf16_pairs,
    transpose_pairs_config,
};

pub mod checkpoint;

/// How every bf16 master commits its fp32 update (#57).
///
/// One constant switches the whole model, kernels included, because both modes
/// live in the same fused kernel. Nearest is the shipped default: every overfit
/// gate converges on it with margin, so nothing yet demands the alternative,
/// and it costs no arithmetic. Stochastic rounding is there for when they stop
/// — its draws come from a splitmix64 stream keyed on `(step, parameter id,
/// element index)` and never from runtime entropy, so a rerun and a checkpoint
/// resume reproduce the same weights either way.
///
/// Flipping this also breaks the GPU/CPU master parity gates, and not because
/// either side is wrong: the two draw independent streams (the grouped `qkv`
/// and `gate_up` masters are one interleaved parameter on the GPU and three or
/// two separate ones on the CPU, so no keying can align them), a single step
/// stays within the gates' one-ulp budget, and then a near-cancelling update
/// turns that ulp into many at the much smaller result. Adopting stochastic
/// rounding means re-deciding what those gates compare, not just widening them.
pub const MASTER_ROUNDING: MasterRounding = MasterRounding::Nearest;

const fn master_rounding_selector() -> u32 {
    match MASTER_ROUNDING {
        MasterRounding::Nearest => MASTER_ROUNDING_NEAREST,
        MasterRounding::Stochastic => MASTER_ROUNDING_STOCHASTIC,
    }
}

/// Bundle the AdamW hyperparameters, this step's bias corrections, and the
/// write-back's deterministic noise seed for one parameter.
fn master_adamw(
    config: AdamWConfig,
    weight_decay: f32,
    corrections: (f32, f32),
    step: u64,
    parameter_id: u64,
) -> MasterAdamW {
    MasterAdamW {
        learning_rate: config.learning_rate,
        beta1: config.beta1,
        beta2: config.beta2,
        epsilon: config.epsilon,
        weight_decay,
        first_correction: corrections.0,
        second_correction: corrections.1,
        rounding: master_rounding_selector(),
        seed: stream_seed(step, parameter_id),
    }
}

/// Stable ids for the deterministic write-back noise stream.
///
/// The values are structural — a parameter's slot in the model, not its
/// allocation order — so they survive a checkpoint resume unchanged.
mod parameter_id {
    pub const EMBEDDING: u64 = 0;
    pub const LM_HEAD: u64 = 1;
    pub const QKV_PROJ: u64 = 2;
    pub const O_PROJ: u64 = 3;
    pub const GATE_UP_PROJ: u64 = 4;
    pub const DOWN_PROJ: u64 = 5;
    pub const EXPERT_GATE_UP: u64 = 6;
    pub const EXPERT_DOWN: u64 = 7;
    /// Ids of block-local parameters are offset by the block index times this,
    /// which is comfortably larger than the per-block id space above.
    pub const PER_BLOCK: u64 = 16;

    pub const fn in_block(block: usize, slot: u64) -> u64 {
        PER_BLOCK * (block as u64 + 1) + slot
    }
}

fn elementwise_config<S: Shape>() -> LaunchConfig {
    assert!(S::NUM_ELEMENTS <= u32::MAX as usize);
    LaunchConfig::for_num_elems(S::NUM_ELEMENTS as u32)
}

fn reduction_config() -> LaunchConfig {
    assert!(tensor_device::REDUCE_THREADS.is_power_of_two());
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (tensor_device::REDUCE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn classifier_config<const N: usize>() -> LaunchConfig {
    assert!(dense_device::CLASSIFIER_THREADS.is_power_of_two());
    assert!(N <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (dense_device::CLASSIFIER_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn norm_config<const N: usize>() -> LaunchConfig {
    assert!(dense_device::NORM_THREADS.is_power_of_two());
    assert!(N <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (dense_device::NORM_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn norm_weight_config<const N: usize, const D: usize>() -> LaunchConfig {
    let threads = dense_device::NORM_THREADS;
    let rows_per_block = dense_device::NORM_WEIGHT_ROWS_PER_BLOCK;
    assert!(threads.is_power_of_two());
    assert!(N <= u32::MAX as usize && D <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            D.div_ceil(threads) as u32,
            N.div_ceil(rows_per_block) as u32,
            1,
        ),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn moe_scatter_dy_config(pairs: usize) -> LaunchConfig {
    assert!(dense_device::MOE_SCATTER_DY_THREADS.is_power_of_two());
    assert!(pairs <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (pairs as u32, 1, 1),
        block_dim: (dense_device::MOE_SCATTER_DY_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch for the MoE dead-slot zeroing pass: one block per `(expert, slot)`.
fn moe_zero_bins_config(bins: usize) -> LaunchConfig {
    assert!(bins <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (bins as u32, 1, 1),
        block_dim: (dense_device::MOE_ZERO_BINS_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn moe_assign_config<const E: usize>() -> LaunchConfig {
    assert!(dense_device::MOE_ASSIGN_THREADS.is_power_of_two());
    assert!(E <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (E as u32, 1, 1),
        block_dim: (dense_device::MOE_ASSIGN_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch for a router GEMM producing an `[m, n]` output: one lane per element
/// of a `[ROUTER_GEMM_BM, ROUTER_GEMM_BN]` output tile.
fn router_gemm_config(m: usize, n: usize) -> LaunchConfig {
    assert!(m <= u32::MAX as usize && n <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            (n as u32).div_ceil(dense_device::ROUTER_GEMM_BN as u32),
            (m as u32).div_ceil(dense_device::ROUTER_GEMM_BM as u32),
            1,
        ),
        block_dim: (dense_device::ROUTER_GEMM_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch for the router input-backward kernel: one block per
/// `[ROUTER_INPUT_TOKENS, ROUTER_INPUT_BN]` tile of `dx`.
fn router_input_config<const N: usize, const D: usize>() -> LaunchConfig {
    assert!(N <= u32::MAX as usize && D <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            D.div_ceil(dense_device::ROUTER_INPUT_BN) as u32,
            N.div_ceil(dense_device::ROUTER_INPUT_TOKENS) as u32,
            1,
        ),
        block_dim: (dense_device::ROUTER_INPUT_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch for one router weight-gradient token partition: one block per
/// `ROUTER_WGRAD_BM` model rows of each of the `ROUTER_WGRAD_SPLITS`
/// partitions.
fn router_wgrad_split_config<const D: usize>() -> LaunchConfig {
    assert!(D <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            D.div_ceil(dense_device::ROUTER_WGRAD_BM) as u32,
            dense_device::ROUTER_WGRAD_SPLITS as u32,
            1,
        ),
        block_dim: (dense_device::ROUTER_WGRAD_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn pairs_config(words: usize) -> LaunchConfig {
    assert!(words <= u32::MAX as usize);
    LaunchConfig::for_num_elems(words as u32)
}

fn flash_forward_config<const N: usize, const T: usize, const H: usize, const HD: usize>()
-> LaunchConfig {
    assert_eq!(N % T, 0);
    flash_device::tiled_forward_config(N / T, T, H, HD)
}

fn flash_dot_config<const N: usize, const H: usize, const HD: usize>() -> LaunchConfig {
    flash_device::dot_config(N, H, HD)
}

/// Launch contract for the per-row oracle attention kernels: one block of
/// `HD` lanes per `(row, head)`.
fn per_row_flash_config<const N: usize, const H: usize, const HD: usize>() -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((N * H) as u32, 1, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn flash_backward_q_config<const N: usize, const T: usize, const H: usize, const HD: usize>()
-> LaunchConfig {
    assert_eq!(N % T, 0);
    flash_device::tiled_backward_q_config(N / T, T, H, HD)
}

fn flash_backward_kv_config<const N: usize, const T: usize, const H: usize, const HD: usize>()
-> LaunchConfig {
    assert_eq!(N % T, 0);
    flash_device::tiled_backward_kv_config(N / T, T, H, HD)
}

fn add_into<S: Shape, P: KernelProfiler>(
    lhs: &GpuTensor<f32, S>,
    rhs: &GpuTensor<f32, S>,
    output: &mut GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    // SAFETY: launch dimensions come from S and all buffers have shape S.
    profiler.measure(stream, name, || unsafe {
        kernels.add(
            stream,
            elementwise_config::<S>(),
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })
}

fn fill_zero<S: Shape, P: KernelProfiler>(
    output: &mut GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    // SAFETY: launch dimensions come from S and output has shape S.
    profiler.measure(stream, name, || unsafe {
        kernels.fill(
            stream,
            elementwise_config::<S>(),
            0.0,
            output.as_device_buffer_mut(),
        )
    })
}

fn sum_into<S: Shape, P: KernelProfiler>(
    input: &GpuTensor<f32, S>,
    output: &mut GpuTensor<f32, Rank1<1>>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    // SAFETY: the reduction config and scalar output satisfy the kernel contract.
    profiler.measure(stream, name, || unsafe {
        kernels.sum(
            stream,
            reduction_config(),
            input.as_device_buffer(),
            S::NUM_ELEMENTS as u32,
            output.as_device_buffer_mut(),
        )
    })
}

fn scale_into<S: Shape, P: KernelProfiler>(
    input: &GpuTensor<f32, S>,
    factor: f32,
    output: &mut GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    // SAFETY: launch dimensions come from S and both buffers have shape S.
    profiler.measure(stream, name, || unsafe {
        kernels.scale(
            stream,
            elementwise_config::<S>(),
            input.as_device_buffer(),
            factor,
            output.as_device_buffer_mut(),
        )
    })
}

/// `output = lhs · rhs` where `rhs` is a `[K, N]` weight shadow.
///
/// The weight operand is an unshaped buffer because it is the widened view of
/// a bf16 master, not a tensor in its own right; its extent is checked by the
/// caller's const generics.
fn gemm_into<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &DeviceBuffer<f32>,
    output: &mut GpuTensor<f32, Rank2<M, N>>,
    stream: &CudaStream,
    kernels: &gemm_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || unsafe {
        kernels.register_gemm_store(
            stream,
            fp32_launch_config(M, N),
            M,
            N,
            K,
            lhs.as_device_buffer(),
            rhs,
            output.as_device_buffer_mut(),
        )
    })
}

fn gemm_tn_accumulate_into<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &GpuTensor<f32, Rank2<M, N>>,
    output: &mut GpuTensor<f32, Rank2<K, N>>,
    stream: &CudaStream,
    kernels: &gemm_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || unsafe {
        kernels.register_gemm_tn_accumulate(
            stream,
            fp32_launch_config(K, N),
            K,
            N,
            M,
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })
}

/// `output = lhs · rhsᵀ`; see [`gemm_into`] for why `rhs` is unshaped.
fn gemm_nt_into<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &DeviceBuffer<f32>,
    output: &mut GpuTensor<f32, Rank2<M, N>>,
    stream: &CudaStream,
    kernels: &gemm_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || unsafe {
        kernels.register_gemm_nt_store(
            stream,
            fp32_launch_config(M, N),
            M,
            N,
            K,
            lhs.as_device_buffer(),
            rhs,
            output.as_device_buffer_mut(),
        )
    })
}

fn copy_device_region<E: DeviceCopy>(
    destination: &mut DeviceBuffer<E>,
    destination_offset: usize,
    source: &DeviceBuffer<E>,
    source_offset: usize,
    elements: usize,
    stream: &CudaStream,
) -> Result<(), DriverError> {
    let destination_end = destination_offset
        .checked_add(elements)
        .expect("device copy destination region overflow");
    let source_end = source_offset
        .checked_add(elements)
        .expect("device copy source region overflow");
    assert!(destination_end <= destination.len());
    assert!(source_end <= source.len());
    let bytes = elements
        .checked_mul(std::mem::size_of::<E>())
        .expect("device copy byte count overflow");
    let destination_bytes = destination_offset
        .checked_mul(std::mem::size_of::<E>())
        .expect("device copy destination byte offset overflow");
    let source_bytes = source_offset
        .checked_mul(std::mem::size_of::<E>())
        .expect("device copy source byte offset overflow");
    let destination = destination
        .cu_deviceptr()
        .checked_add(destination_bytes as u64)
        .expect("device copy destination pointer overflow");
    let source = source
        .cu_deviceptr()
        .checked_add(source_bytes as u64)
        .expect("device copy source pointer overflow");
    // SAFETY: the checked element ranges above are within their allocations,
    // and expert oracle staging — the only remaining caller — always copies
    // between distinct allocations.
    unsafe { cuda_core::memory::memcpy_dtod_async(destination, source, bytes, stream.cu_stream()) }
}

/// The two tcgen05 `B` operands of one linear, in the layouts the `C = A B^T`
/// form needs as a K-contiguous operand: `[rows, columns]` for the input
/// gradient `dx = dy·Wᵀ` and `[columns, rows]` for the forward `y = x·W`.
///
/// The `[rows, columns]` operand *is* the bf16 master. #57 made master and
/// compute copy the same dtype and layout, so #58 encoded `normal_tma` against
/// the master's own words instead of a byte-identical duplicate: only
/// `transposed` is still stored, and only it is refreshed after a step.
struct Bf16LinearWeights {
    transposed: DeviceBuffer<u32>,
    normal_tma: Bf16PairsTmaMap,
    transposed_tma: Bf16PairsTmaMap,
}

impl Bf16LinearWeights {
    /// `master` is the linear's packed `[rows, columns]` master, mapped in
    /// place; `values` are the same weights on the host, for the transpose.
    fn new(
        stream: &CudaStream,
        master: &DeviceBuffer<u32>,
        values: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Self, Box<dyn Error>> {
        assert_eq!(values.len(), rows * columns);
        let mut transposed_values = vec![0.0f32; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                transposed_values[column * rows + row] = values[row * columns + column];
            }
        }
        let transposed = DeviceBuffer::from_host(stream, &pack_bf16_pairs(&transposed_values))?;
        // SAFETY: `transposed` lives beside its map here and the master beside
        // both in the owning linear; neither is ever reallocated, including on
        // checkpoint resume, which refills masters in place. Optimizer
        // write-back and the transpose refresh mutate contents only.
        let normal_tma =
            unsafe { create_bf16_pairs_tma_map(stream, master, columns, rows, TmaLayout::KMajor)? };
        let transposed_tma = unsafe {
            create_bf16_pairs_tma_map(stream, &transposed, rows, columns, TmaLayout::KMajor)?
        };
        Ok(Self {
            transposed,
            normal_tma,
            transposed_tma,
        })
    }

    /// Re-transpose the master the optimizer just wrote. The `[rows, columns]`
    /// operand needs no refresh at all: the fused write-back already stored the
    /// bytes its descriptor reads.
    fn sync_from_master(
        &mut self,
        master: &DeviceBuffer<u32>,
        rows: usize,
        columns: usize,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        // SAFETY: the master holds rows * columns packed pairs and
        // `transposed` is the same allocation size.
        unsafe {
            kernels.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(rows, columns),
                master,
                rows as u32,
                columns as u32,
                &mut self.transposed,
            )
        }
    }
}

struct Bf16LinearMaps {
    d: Bf16PairsTmaMap,
    ff: Bf16PairsTmaMap,
    qkv: Bf16PairsTmaMap,
    gate_up: Bf16PairsTmaMap,
}

impl Bf16LinearMaps {
    fn get<const D: usize, const FF: usize>(&self, width: usize) -> &Bf16PairsTmaMap {
        if width == D {
            &self.d
        } else if width == FF {
            &self.ff
        } else if width == 3 * D {
            &self.qkv
        } else if width == 2 * FF {
            &self.gate_up
        } else {
            panic!("unsupported tcgen05 linear width {width}")
        }
    }
}

/// Reusable packed-bf16 operand storage for all block-linear GEMMs.
///
/// Both operands are staged in their natural `[N, width]` row-major layout:
/// `rows` takes the output gradient (also the input GEMM's row operand) and
/// `lhs` the activation. The weight gradient reads them MN-major through the
/// tcgen05 descriptor transpose (#53), so each buffer carries a K-major map
/// set and an MN-major one and nothing is ever transposed in global memory.
struct Bf16LinearScratch<const N: usize, const D: usize, const FF: usize> {
    rows: DeviceBuffer<u32>,
    lhs: DeviceBuffer<u32>,
    row_maps: Bf16LinearMaps,
    row_mn_maps: Bf16LinearMaps,
    lhs_mn_maps: Bf16LinearMaps,
}

impl<const N: usize, const D: usize, const FF: usize> Bf16LinearScratch<N, D, FF> {
    fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        let max_width = D.max(FF).max(3 * D).max(2 * FF);
        let rows = DeviceBuffer::zeroed(stream, N * max_width / 2)?;
        let lhs = DeviceBuffer::zeroed(stream, N * max_width / 2)?;

        let row_maps = Self::maps(stream, &rows, TmaLayout::KMajor)?;
        let row_mn_maps = Self::maps(stream, &rows, TmaLayout::MnMajor)?;
        let lhs_mn_maps = Self::maps(stream, &lhs, TmaLayout::MnMajor)?;
        Ok(Self {
            rows,
            lhs,
            row_maps,
            row_mn_maps,
            lhs_mn_maps,
        })
    }

    fn maps(
        stream: &CudaStream,
        buffer: &DeviceBuffer<u32>,
        layout: TmaLayout,
    ) -> Result<Bf16LinearMaps, Box<dyn Error>> {
        let make =
            |width| unsafe { create_bf16_pairs_tma_map_prefix(stream, buffer, width, N, layout) };
        Ok(Bf16LinearMaps {
            d: make(D)?,
            ff: make(FF)?,
            qkv: make(3 * D)?,
            gate_up: make(2 * FF)?,
        })
    }
}

/// Operand staging shared by every block linear.
///
/// `bf16` holds the packed tcgen05 panels; `oracle_weights` is the fp32 buffer
/// the register-tiled fallback needs, because the masters are bf16 (#57) and
/// that GEMM family reads fp32. Which one a given call uses depends on the
/// token count as well as the weight shape, so both can be live in one model —
/// and `oracle_weights` is skipped entirely when no linear can reach the
/// fallback, which is the case for every real training shape.
struct LinearScratch<const N: usize, const D: usize, const FF: usize> {
    bf16: Option<Bf16LinearScratch<N, D, FF>>,
    oracle_weights: Option<DeviceBuffer<f32>>,
}

impl<const N: usize, const D: usize, const FF: usize> LinearScratch<N, D, FF> {
    /// `widths` are the `(input, output)` shapes of the linears this scratch
    /// serves; `oracle_weights` is sized for the largest and allocated only if
    /// one of them can miss the tcgen05 contract.
    fn new(stream: &CudaStream, widths: &[(usize, usize)]) -> Result<Self, Box<dyn Error>> {
        let bf16 =
            if N.is_multiple_of(TC_TILE) && D.is_multiple_of(TC_TILE) && FF.is_multiple_of(TC_TILE)
            {
                Some(Bf16LinearScratch::new(stream)?)
            } else {
                None
            };
        let all_tcgen05 = bf16.is_some()
            && widths
                .iter()
                .all(|&(input, output)| tcgen05_linear_eligible(N, input, output));
        let oracle_weights = if all_tcgen05 {
            None
        } else {
            let elements = widths
                .iter()
                .map(|&(input, output)| input * output)
                .max()
                .expect("a block has at least one linear");
            Some(DeviceBuffer::zeroed(stream, elements)?)
        };
        Ok(Self {
            bf16,
            oracle_weights,
        })
    }

    fn oracle_weights(&mut self) -> &mut DeviceBuffer<f32> {
        self.oracle_weights
            .as_mut()
            .expect("the fp32 oracle path needs its widened weight staging")
    }
}

fn tcgen05_linear_eligible(m: usize, k: usize, n: usize) -> bool {
    m.is_multiple_of(TC_M_TILE) && k.is_multiple_of(TC_K_PIPELINE) && n.is_multiple_of(TC_N_TILE)
}

fn tcgen05_attention_eligible(t: usize, head_dim: usize) -> bool {
    // Every MMA in the tcgen05 attention kernels fills 128 real rows — the
    // forward's query block (#68) and the backward's Design-B tile pair (#47
    // item 2) — so `T` must be a multiple of 128; other shapes fall to the
    // per-row oracle. The canonical T=2048 satisfies it.
    t.is_multiple_of(FLASH_QUERIES) && head_dim == FLASH_HD
}

/// One operand's packed-bf16 `[B*H, T, HD]` head panel and the TMA map that
/// streams it.
struct StagedHeads {
    words: DeviceBuffer<u32>,
    tma: FlashHeadTmaMap,
}

impl StagedHeads {
    fn new(
        stream: &CudaStream,
        words: usize,
        sequence_length: usize,
        planes: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let words = DeviceBuffer::zeroed(stream, words)?;
        // SAFETY: the mapped buffer lives beside its map and is never
        // reallocated.
        let tma = unsafe { create_flash_head_tma_map(stream, &words, sequence_length, planes)? };
        Ok(Self { words, tma })
    }
}

/// Where a block keeps Q, K and V between the projection and attention.
///
/// The tcgen05 path never wants them fp32. Its only consumers are the two
/// flash passes, which stream packed-bf16 head panels, so the projection's
/// fp32 panel is rotated and quantized straight into those panels once and
/// read twice — the fp32 triple, the split that filled it, the two rotation
/// passes and the backward's re-staging all go away with it (SPEC §7.1).
/// Shapes the tcgen05 kernels do not cover keep the fp32 triple its oracle
/// kernels read.
enum AttentionOperands<const N: usize, const D: usize> {
    Staged {
        q: StagedHeads,
        k: StagedHeads,
        v: StagedHeads,
    },
    Wide {
        q: GpuTensor<f32, Rank2<N, D>>,
        k: GpuTensor<f32, Rank2<N, D>>,
        v: GpuTensor<f32, Rank2<N, D>>,
    },
}

impl<const N: usize, const D: usize> AttentionOperands<N, D> {
    fn new(
        stream: &CudaStream,
        sequence_length: usize,
        heads: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if !tcgen05_attention_eligible(sequence_length, D / heads) {
            return Ok(Self::Wide {
                q: GpuTensor::zeros(stream)?,
                k: GpuTensor::zeros(stream)?,
                v: GpuTensor::zeros(stream)?,
            });
        }
        let planes = N / sequence_length * heads;
        let panel = || StagedHeads::new(stream, N * D / 2, sequence_length, planes);
        Ok(Self::Staged {
            q: panel()?,
            k: panel()?,
            v: panel()?,
        })
    }
}

/// The backward's staged dY panel and the per-workstream correction-count
/// output, shared across blocks.
///
/// Q/K/V are not here any more: they are per-block saved activations that the
/// forward already staged (see [`AttentionOperands`]). dY is not — it is a
/// backward temporary, so one buffer serves every block.
struct FlashAttentionScratch<const N: usize, const T: usize, const D: usize, const H: usize> {
    dy: StagedHeads,
    correction_counts: DeviceBuffer<u32>,
}

impl<const N: usize, const T: usize, const D: usize, const H: usize>
    FlashAttentionScratch<N, T, D, H>
{
    fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            dy: StagedHeads::new(stream, N * D / 2, T, N / T * H)?,
            correction_counts: DeviceBuffer::zeroed(stream, correction_count_len(N / T, T, H))?,
        })
    }
}

pub struct GpuLinear<const IN: usize, const OUT: usize> {
    pub w: GpuBf16Tensor<Rank2<IN, OUT>>,
    pub dw: GpuTensor<f32, Rank2<IN, OUT>>,
    compute: Option<Bf16LinearWeights>,
}

impl<const IN: usize, const OUT: usize> GpuLinear<IN, OUT> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layer: &nn::Linear<N, IN, OUT>,
    ) -> Result<Self, Box<dyn Error>> {
        let w = GpuBf16Tensor::from_f32_host(stream, layer.w.as_slice())?;
        let compute = if IN.is_multiple_of(TC_TILE) && OUT.is_multiple_of(TC_TILE) {
            let values = layer.w.as_slice();
            Some(Bf16LinearWeights::new(
                stream,
                w.as_words(),
                values,
                IN,
                OUT,
            )?)
        } else {
            None
        };
        Ok(Self {
            w,
            dw: GpuTensor::zeros(stream)?,
            compute,
        })
    }

    #[allow(clippy::too_many_arguments, unused_unsafe)]
    fn forward_into<const N: usize, const D: usize, const FF: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        output: &mut GpuTensor<f32, Rank2<N, OUT>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        scratch: &mut LinearScratch<N, D, FF>,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        if let (Some(compute), Some(staging)) = (&self.compute, scratch.bf16.as_mut())
            && tcgen05_linear_eligible(N, IN, OUT)
        {
            // SAFETY: buffers and launch tiles are validated by the eligibility check.
            profiler.measure(stream, name, || unsafe {
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * IN / 2),
                    x.as_device_buffer(),
                    &mut staging.rows,
                )?;
                unsafe {
                    tcgen05.f32_store(
                        stream,
                        tcgen05_launch_config(N, OUT, IN),
                        staging.row_maps.get::<D, FF>(IN).as_ptr(),
                        compute.transposed_tma.as_ptr(),
                        output.as_device_buffer_mut(),
                        OUT as u32,
                        IN as u32,
                    )
                }
            })
        } else {
            let weights = scratch.oracle_weights();
            self.w.widen_into(weights, stream, tensor)?;
            gemm_into(x, weights, output, stream, fp32, profiler, name)
        }
    }

    #[allow(clippy::too_many_arguments, unused_unsafe)]
    fn backward_into<const N: usize, const D: usize, const FF: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        dy: &GpuTensor<f32, Rank2<N, OUT>>,
        dx: &mut GpuTensor<f32, Rank2<N, IN>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        scratch: &mut LinearScratch<N, D, FF>,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<(), DriverError> {
        if let (Some(compute), Some(staging)) = (&self.compute, scratch.bf16.as_mut())
            && tcgen05_linear_eligible(N, IN, OUT)
        {
            // SAFETY: both quantize launches cover exactly the panels they
            // write, and the accumulate's tiles come from the eligibility check.
            profiler.measure(stream, names[0], || unsafe {
                // `dW += xᵀ·dy` reads both operands MN-major straight out of
                // their native `[N, width]` panels, so staging is a plain
                // quantize each and no transpose runs at all. `rows` doubles as
                // the input GEMM's row operand below.
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * IN / 2),
                    x.as_device_buffer(),
                    &mut staging.lhs,
                )?;
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * OUT / 2),
                    dy.as_device_buffer(),
                    &mut staging.rows,
                )?;
                unsafe {
                    tcgen05.f32_accumulate_transposed(
                        stream,
                        tcgen05_launch_config(IN, OUT, N),
                        staging.lhs_mn_maps.get::<D, FF>(IN).as_ptr(),
                        staging.row_mn_maps.get::<D, FF>(OUT).as_ptr(),
                        self.dw.as_device_buffer_mut(),
                        OUT as u32,
                        N as u32,
                    )
                }
            })?;
            // `staging.rows` still holds the quantized `dy` written by the
            // weight-gradient pass above; this launch consumes it as its row
            // operand, so nothing may overwrite `rows` between the two.
            profiler.measure(stream, names[1], || unsafe {
                tcgen05.f32_store(
                    stream,
                    tcgen05_launch_config(N, IN, OUT),
                    staging.row_maps.get::<D, FF>(OUT).as_ptr(),
                    compute.normal_tma.as_ptr(),
                    dx.as_device_buffer_mut(),
                    IN as u32,
                    OUT as u32,
                )
            })
        } else {
            gemm_tn_accumulate_into(x, dy, &mut self.dw, stream, fp32, profiler, names[0])?;
            let weights = scratch.oracle_weights();
            self.w.widen_into(weights, stream, tensor)?;
            gemm_nt_into(dy, weights, dx, stream, fp32, profiler, names[1])
        }
    }

    /// Rebuild the transposed compute operand from the master. The optimizer
    /// no longer needs this — its write-back emits both layouts — but a
    /// checkpoint resume refills masters in place and must catch the
    /// transpose up.
    fn sync_compute(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        if let Some(compute) = &mut self.compute {
            compute.sync_from_master(self.w.as_words(), IN, OUT, stream, kernels)?;
        }
        Ok(())
    }

    /// One AdamW step over the master, which also refreshes the transposed
    /// compute operand and clears the gradient. A linear without a compute
    /// copy needs no transpose, so it takes the flat write-back.
    fn adamw_step(
        &mut self,
        moments: &mut GpuAdamWMoments<Rank2<IN, OUT>>,
        config: MasterAdamW,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        match &mut self.compute {
            Some(compute) => self.w.adamw_step_transposed(
                &mut self.dw,
                moments,
                config,
                IN,
                OUT,
                &mut compute.transposed,
                stream,
                kernels,
            ),
            None => self
                .w
                .adamw_step(&mut self.dw, moments, config, stream, kernels),
        }
    }
}

pub struct GpuGroupedLinear<const IN: usize, const GROUPS: usize, const OUT: usize> {
    pub w: GpuBf16Tensor<Rank3<IN, GROUPS, OUT>>,
    pub dw: GpuTensor<f32, Rank3<IN, GROUPS, OUT>>,
    compute: Option<Bf16LinearWeights>,
}

/// How a grouped linear's backward gets its upstream gradient.
///
/// `Staged` means the kernel that produced it wrote the packed-bf16 words
/// straight into the scratch's row operand, so the backward's own quantize
/// does not run. The weight product reads that buffer MN-major and the input
/// product K-major, both through descriptors over the same bytes, so one
/// packed layout serves both and the panel is rounded once either way (SPEC
/// §7.1). Ask [`GpuGroupedLinear::packed_row_gradient`] for the buffer; a
/// `None` there means the fp32 fallback runs and only `Wide` is valid.
#[derive(Clone, Copy)]
enum RowGradient<'a, const N: usize, const GROUPS: usize, const OUT: usize> {
    Staged,
    Wide(&'a GpuTensor<f32, Rank3<N, GROUPS, OUT>>),
}

impl<const IN: usize, const GROUPS: usize, const OUT: usize> GpuGroupedLinear<IN, GROUPS, OUT> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layers: [&nn::Linear<N, IN, OUT>; GROUPS],
    ) -> Result<Self, Box<dyn Error>> {
        let mut weights = vec![0.0; IN * GROUPS * OUT];
        for input in 0..IN {
            for (group, layer) in layers.iter().enumerate() {
                let source = &layer.w.as_slice()[input * OUT..(input + 1) * OUT];
                let destination = (input * GROUPS + group) * OUT;
                weights[destination..destination + OUT].copy_from_slice(source);
            }
        }
        let w = GpuBf16Tensor::from_f32_host(stream, &weights)?;
        let compute = if IN.is_multiple_of(TC_TILE) && (GROUPS * OUT).is_multiple_of(TC_TILE) {
            let master = w.as_words();
            Some(Bf16LinearWeights::new(
                stream,
                master,
                &weights,
                IN,
                GROUPS * OUT,
            )?)
        } else {
            None
        };
        Ok(Self {
            w,
            dw: GpuTensor::zeros(stream)?,
            compute,
        })
    }

    /// The packed row operand [`Self::backward_into`] will read when this
    /// linear takes the tcgen05 path, for a producing kernel to write itself
    /// in place of an fp32 panel and the quantize over it.
    ///
    /// `None` means the fp32 fallback runs and the producer owes a wide
    /// `[N, GROUPS * OUT]` panel instead; the condition is `backward_into`'s
    /// own, so the two never disagree.
    fn packed_row_gradient<'a, const N: usize, const D: usize, const FF: usize>(
        &self,
        scratch: &'a mut LinearScratch<N, D, FF>,
    ) -> Option<&'a mut DeviceBuffer<u32>> {
        let staging = scratch.bf16.as_mut()?;
        (self.compute.is_some() && tcgen05_linear_eligible(N, IN, GROUPS * OUT))
            .then_some(&mut staging.rows)
    }

    #[allow(clippy::too_many_arguments, unused_unsafe)]
    fn forward_into<const N: usize, const D: usize, const FF: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        output: &mut GpuTensor<f32, Rank3<N, GROUPS, OUT>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        scratch: &mut LinearScratch<N, D, FF>,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        let width = GROUPS * OUT;
        if let (Some(compute), Some(staging)) = (&self.compute, scratch.bf16.as_mut())
            && tcgen05_linear_eligible(N, IN, width)
        {
            // SAFETY: buffers and launch tiles are validated by the eligibility check.
            profiler.measure(stream, name, || unsafe {
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * IN / 2),
                    x.as_device_buffer(),
                    &mut staging.rows,
                )?;
                unsafe {
                    tcgen05.f32_store(
                        stream,
                        tcgen05_launch_config(N, width, IN),
                        staging.row_maps.get::<D, FF>(IN).as_ptr(),
                        compute.transposed_tma.as_ptr(),
                        output.as_device_buffer_mut(),
                        width as u32,
                        IN as u32,
                    )
                }
            })
        } else {
            let weights = scratch.oracle_weights();
            self.w.widen_into(weights, stream, tensor)?;
            profiler.measure(stream, name, || unsafe {
                fp32.register_gemm_store(
                    stream,
                    fp32_launch_config(N, width),
                    N,
                    width,
                    IN,
                    x.as_device_buffer(),
                    weights,
                    output.as_device_buffer_mut(),
                )
            })
        }
    }

    #[allow(clippy::too_many_arguments, unused_unsafe)]
    fn backward_into<const N: usize, const D: usize, const FF: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        dy: RowGradient<'_, N, GROUPS, OUT>,
        dx: &mut GpuTensor<f32, Rank2<N, IN>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        scratch: &mut LinearScratch<N, D, FF>,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<(), DriverError> {
        let width = GROUPS * OUT;
        if let (Some(compute), Some(staging)) = (&self.compute, scratch.bf16.as_mut())
            && tcgen05_linear_eligible(N, IN, width)
        {
            // SAFETY: both quantize launches cover exactly the panels they
            // write, and the accumulate's tiles come from the eligibility check.
            profiler.measure(stream, names[0], || unsafe {
                // See `GpuLinear::backward_into`: both weight-gradient operands
                // are consumed MN-major from their native panels, so each is a
                // plain quantize and nothing is transposed.
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * IN / 2),
                    x.as_device_buffer(),
                    &mut staging.lhs,
                )?;
                if let RowGradient::Wide(dy) = dy {
                    tensor.convert_f32_to_bf16_pairs(
                        stream,
                        pairs_config(N * width / 2),
                        dy.as_device_buffer(),
                        &mut staging.rows,
                    )?;
                }
                unsafe {
                    tcgen05.f32_accumulate_transposed(
                        stream,
                        tcgen05_launch_config(IN, width, N),
                        staging.lhs_mn_maps.get::<D, FF>(IN).as_ptr(),
                        staging.row_mn_maps.get::<D, FF>(width).as_ptr(),
                        self.dw.as_device_buffer_mut(),
                        width as u32,
                        N as u32,
                    )
                }
            })?;
            // `staging.rows` holds the quantized `dy` — written by the pass
            // above, or by the caller's own kernel — and this launch consumes
            // it as its row operand, so nothing may overwrite `rows` between
            // the two.
            profiler.measure(stream, names[1], || unsafe {
                tcgen05.f32_store(
                    stream,
                    tcgen05_launch_config(N, IN, width),
                    staging.row_maps.get::<D, FF>(width).as_ptr(),
                    compute.normal_tma.as_ptr(),
                    dx.as_device_buffer_mut(),
                    IN as u32,
                    width as u32,
                )
            })
        } else {
            let RowGradient::Wide(dy) = dy else {
                panic!("the fp32 fallback reads a wide row gradient, as `packed_row_gradient` says")
            };
            profiler.measure(stream, names[0], || unsafe {
                fp32.register_gemm_tn_accumulate(
                    stream,
                    fp32_launch_config(IN, width),
                    IN,
                    width,
                    N,
                    x.as_device_buffer(),
                    dy.as_device_buffer(),
                    self.dw.as_device_buffer_mut(),
                )
            })?;
            let weights = scratch.oracle_weights();
            self.w.widen_into(weights, stream, tensor)?;
            profiler.measure(stream, names[1], || unsafe {
                fp32.register_gemm_nt_store(
                    stream,
                    fp32_launch_config(N, IN),
                    N,
                    IN,
                    width,
                    dy.as_device_buffer(),
                    weights,
                    dx.as_device_buffer_mut(),
                )
            })
        }
    }

    /// [`GpuLinear::sync_compute`] over the interleaved master.
    fn sync_compute(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        if let Some(compute) = &mut self.compute {
            compute.sync_from_master(self.w.as_words(), IN, GROUPS * OUT, stream, kernels)?;
        }
        Ok(())
    }

    /// [`GpuLinear::adamw_step`] over the interleaved `[IN, GROUPS * OUT]`
    /// master.
    fn adamw_step(
        &mut self,
        moments: &mut GpuAdamWMoments<Rank3<IN, GROUPS, OUT>>,
        config: MasterAdamW,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        match &mut self.compute {
            Some(compute) => self.w.adamw_step_transposed(
                &mut self.dw,
                moments,
                config,
                IN,
                GROUPS * OUT,
                &mut compute.transposed,
                stream,
                kernels,
            ),
            None => self
                .w
                .adamw_step(&mut self.dw, moments, config, stream, kernels),
        }
    }
}

/// The tcgen05 operands for `experts` row-major `[rows, columns]` matrices held
/// in one stacked allocation.
///
/// `normal_maps` address the stacked bf16 master itself, one strided descriptor
/// per expert (#58). `transposed` is the transpose of the global
/// `[experts * rows, columns]` matrix, likewise one strided descriptor per
/// expert, which avoids an allocation or a transpose launch each.
struct StackedBf16Weights {
    transposed: DeviceBuffer<u32>,
    normal_maps: Vec<Bf16PairsTmaMap>,
    transposed_maps: Vec<Bf16PairsTmaMap>,
    experts: usize,
    rows: usize,
    columns: usize,
}

impl StackedBf16Weights {
    /// `master` is the stacked packed master `normal_maps` address in place;
    /// `values` are the same weights on the host, for the transpose.
    fn new(
        stream: &CudaStream,
        master: &DeviceBuffer<u32>,
        values: &[f32],
        experts: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Self, Box<dyn Error>> {
        assert_eq!(values.len(), experts * rows * columns);
        let total_rows = experts * rows;
        let mut transposed_values = vec![0.0f32; values.len()];
        for row in 0..total_rows {
            for column in 0..columns {
                transposed_values[column * total_rows + row] = values[row * columns + column];
            }
        }

        let transposed = DeviceBuffer::from_host(stream, &pack_bf16_pairs(&transposed_values))?;
        // SAFETY: `transposed` lives beside its maps here and the master beside
        // both in the owning expert FFN; neither is ever reallocated, including
        // on checkpoint resume, which refills masters in place.
        let normal_maps = (0..experts)
            .map(|expert| unsafe {
                create_bf16_pairs_tma_map_region(
                    stream,
                    master,
                    expert * rows * columns / 2,
                    columns,
                    rows,
                    columns,
                    TmaLayout::KMajor,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let transposed_maps = (0..experts)
            .map(|expert| unsafe {
                create_bf16_pairs_tma_map_region(
                    stream,
                    &transposed,
                    expert * rows / 2,
                    rows,
                    columns,
                    total_rows,
                    TmaLayout::KMajor,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            transposed,
            normal_maps,
            transposed_maps,
            experts,
            rows,
            columns,
        })
    }

    /// Re-transpose the master; see [`Bf16LinearWeights::sync_from_master`] for
    /// why the untransposed operand needs nothing.
    fn sync_from_master(
        &mut self,
        master: &DeviceBuffer<u32>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        // SAFETY: `transposed` is allocated to the master's packed-pair length.
        unsafe {
            kernels.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(self.experts * self.rows, self.columns),
                master,
                (self.experts * self.rows) as u32,
                self.columns as u32,
                &mut self.transposed,
            )
        }
    }
}

/// Whether these expert shapes route every expert GEMM through tcgen05, which
/// is also what decides whether a panel can be stored packed.
///
/// This has to be *exactly* the predicate the GEMMs test, not a weaker
/// alignment check: a packed panel has no fp32 copy, so if a shape could store
/// panels packed and still fall back to the register-tiled oracle, that oracle
/// would ask a packed panel for `wide()` and panic. Testing both expert GEMMs
/// (`D → 2·FF` and `FF → D`) makes packed storage and the tcgen05 path the same
/// condition, which is what lets `wide()` and the `fp32_staging` expect be
/// invariants rather than hazards.
fn expert_tcgen05_aligned<const C: usize, const D: usize, const FF: usize>() -> bool {
    tcgen05_linear_eligible(C, D, 2 * FF) && tcgen05_linear_eligible(C, FF, D)
}

/// One expert panel `[E, C, width]` stored in the dtype its consumers read.
///
/// A panel whose only readers are bf16 tcgen05 operands is stored `Packed`:
/// the producing kernel rounds once and every GEMM addresses the panel in
/// place through its own descriptors, so no `convert_f32_to_bf16_pairs` runs
/// and the wide copy never exists (SPEC §7.1, #59). The fp32 oracle path and
/// panels a pointwise kernel still reads keep the `Wide` copy.
pub enum ExpertPanel {
    Packed(PackedExpertPanel),
    Wide(DeviceBuffer<f32>),
}

/// Packed-bf16 panel storage plus the per-expert descriptors addressing it as
/// a GEMM row operand (`k_maps`) and as a weight-gradient operand (`mn_maps`).
pub struct PackedExpertPanel {
    words: DeviceBuffer<u32>,
    k_maps: Vec<Bf16PairsTmaMap>,
    mn_maps: Vec<Bf16PairsTmaMap>,
}

impl ExpertPanel {
    fn new(
        stream: &CudaStream,
        experts: usize,
        capacity: usize,
        width: usize,
        packed: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let elements = experts * capacity * width;
        if !packed {
            return Ok(Self::Wide(DeviceBuffer::zeroed(stream, elements)?));
        }
        assert!(
            width.is_multiple_of(2),
            "packed expert panel needs an even width"
        );
        let words = DeviceBuffer::zeroed(stream, elements / 2)?;
        let k_maps =
            expert_panel_maps(stream, &words, experts, capacity, width, TmaLayout::KMajor)?;
        let mn_maps =
            expert_panel_maps(stream, &words, experts, capacity, width, TmaLayout::MnMajor)?;
        Ok(Self::Packed(PackedExpertPanel {
            words,
            k_maps,
            mn_maps,
        }))
    }

    fn wide(&self) -> &DeviceBuffer<f32> {
        match self {
            Self::Wide(values) => values,
            Self::Packed(_) => panic!("packed expert panel has no fp32 copy"),
        }
    }

    /// Host upload in element order, rounding to bf16 for packed storage.
    /// Parity-gate entry point; it synchronizes the stream.
    fn upload(&mut self, values: &[f32], stream: &CudaStream) -> Result<(), DriverError> {
        match self {
            Self::Wide(wide) => {
                assert_eq!(values.len(), wide.len());
                upload_device(wide.cu_deviceptr(), values, stream)
            }
            Self::Packed(panel) => {
                assert_eq!(values.len(), panel.words.len() * 2);
                upload_device(panel.words.cu_deviceptr(), &pack_bf16_pairs(values), stream)
            }
        }
    }
}

/// Blocking host-to-device copy of `values` to the device address `destination`.
fn upload_device<T>(
    destination: u64,
    values: &[T],
    stream: &CudaStream,
) -> Result<(), DriverError> {
    // SAFETY: callers size `values` against the destination allocation, and the
    // synchronization keeps the host slice borrowed until the copy completes.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            destination,
            values.as_ptr(),
            std::mem::size_of_val(values),
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()
}

/// Per-expert descriptors over the `[E, C, width]` packed panel at `words`.
///
/// SAFETY: every caller keeps the maps beside the buffer they describe inside
/// the same never-reallocated activation or scratch struct.
fn expert_panel_maps(
    stream: &CudaStream,
    words: &DeviceBuffer<u32>,
    experts: usize,
    capacity: usize,
    width: usize,
    layout: TmaLayout,
) -> Result<Vec<Bf16PairsTmaMap>, Box<dyn Error>> {
    (0..experts)
        .map(|expert| unsafe {
            create_bf16_pairs_tma_map_region(
                stream,
                words,
                expert * capacity * width / 2,
                width,
                capacity,
                width,
                layout,
            )
        })
        .collect()
}

/// Staging used only by the non-aligned fp32 oracle. One expert is copied into
/// these buffers and passed to the existing register-tiled GEMM launchers.
///
/// `b` receives the weights widened from the bf16 master (#57) rather than a
/// copy: #59 removed the operand staging the aligned path used to need, but the
/// oracle still reads fp32, so this is now the only place a master is widened.
struct ExpertFp32Scratch {
    a: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    c: DeviceBuffer<f32>,
}

impl ExpertFp32Scratch {
    fn new<const C: usize, const D: usize, const FF: usize>(
        stream: &CudaStream,
    ) -> Result<Self, DriverError> {
        let max_width = D.max(FF).max(2 * FF);
        let max_elements = (C * max_width).max(D * 2 * FF).max(FF * D);
        Ok(Self {
            a: DeviceBuffer::zeroed(stream, max_elements)?,
            b: DeviceBuffer::zeroed(stream, max_elements)?,
            c: DeviceBuffer::zeroed(stream, max_elements)?,
        })
    }
}

#[allow(clippy::too_many_arguments, unused_unsafe)]
fn expert_linear_forward<const E: usize, const C: usize, P: KernelProfiler>(
    input: &ExpertPanel,
    weights: &DeviceBuffer<u32>,
    compute: Option<&StackedBf16Weights>,
    output: &mut DeviceBuffer<f32>,
    input_width: usize,
    output_width: usize,
    staging: &mut Option<ExpertFp32Scratch>,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    fp32: &gemm_kernels::LoadedModule,
    tcgen05: &Tcgen05Gemm,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    if let (Some(compute), ExpertPanel::Packed(input)) = (compute, input)
        && tcgen05_linear_eligible(C, input_width, output_width)
    {
        profiler.measure(stream, name, || {
            for expert in 0..E {
                unsafe {
                    tcgen05.f32_store_at(
                        stream,
                        tcgen05_launch_config(C, output_width, input_width),
                        input.k_maps[expert].as_ptr(),
                        compute.transposed_maps[expert].as_ptr(),
                        output,
                        expert * C * output_width,
                        C * output_width,
                        output_width as u32,
                        input_width as u32,
                    )?;
                }
            }
            Ok(())
        })
    } else {
        let fp32_scratch = staging
            .as_mut()
            .expect("non-aligned experts must own fp32 staging");
        // SAFETY: every expert's slice fits the oracle scratch, which is sized
        // for the widest operand this call can see.
        profiler.measure(stream, name, || unsafe {
            for expert in 0..E {
                copy_device_region(
                    &mut fp32_scratch.a,
                    0,
                    input.wide(),
                    expert * C * input_width,
                    C * input_width,
                    stream,
                )?;
                tensor.widen_bf16_region(
                    stream,
                    pairs_config(input_width * output_width),
                    weights,
                    (expert * input_width * output_width) as u32,
                    (input_width * output_width) as u32,
                    &mut fp32_scratch.b,
                )?;
                unsafe {
                    fp32.register_gemm_store(
                        stream,
                        fp32_launch_config(C, output_width),
                        C,
                        output_width,
                        input_width,
                        &fp32_scratch.a,
                        &fp32_scratch.b,
                        &mut fp32_scratch.c,
                    )?;
                }
                copy_device_region(
                    output,
                    expert * C * output_width,
                    &fp32_scratch.c,
                    0,
                    C * output_width,
                    stream,
                )?;
            }
            Ok(())
        })
    }
}

#[allow(clippy::too_many_arguments, unused_unsafe)]
fn expert_linear_backward<const E: usize, const C: usize, P: KernelProfiler>(
    input: &ExpertPanel,
    output_gradient: &ExpertPanel,
    weights: &DeviceBuffer<u32>,
    weight_gradient: &mut DeviceBuffer<f32>,
    compute: Option<&StackedBf16Weights>,
    input_gradient: &mut DeviceBuffer<f32>,
    input_width: usize,
    output_width: usize,
    staging: &mut Option<ExpertFp32Scratch>,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    fp32: &gemm_kernels::LoadedModule,
    tcgen05: &Tcgen05Gemm,
    profiler: &mut P,
    names: [&'static str; 2],
) -> Result<(), DriverError> {
    if let (Some(compute), ExpertPanel::Packed(input), ExpertPanel::Packed(dy)) =
        (compute, input, output_gradient)
        && tcgen05_linear_eligible(C, input_width, output_width)
    {
        // Both weight-gradient operands are read MN-major out of their native
        // `[E*C, width]` panels (#53), and since #59 both panels are stored
        // packed, so no operand is staged or transposed at all.
        profiler.measure(stream, names[0], || {
            for expert in 0..E {
                unsafe {
                    tcgen05.f32_accumulate_transposed_at(
                        stream,
                        tcgen05_launch_config(input_width, output_width, C),
                        input.mn_maps[expert].as_ptr(),
                        dy.mn_maps[expert].as_ptr(),
                        weight_gradient,
                        expert * input_width * output_width,
                        input_width * output_width,
                        output_width as u32,
                        C as u32,
                    )?;
                }
            }
            Ok(())
        })?;
        profiler.measure(stream, names[1], || {
            for expert in 0..E {
                unsafe {
                    tcgen05.f32_store_at(
                        stream,
                        tcgen05_launch_config(C, input_width, output_width),
                        dy.k_maps[expert].as_ptr(),
                        compute.normal_maps[expert].as_ptr(),
                        input_gradient,
                        expert * C * input_width,
                        C * input_width,
                        input_width as u32,
                        output_width as u32,
                    )?;
                }
            }
            Ok(())
        })
    } else {
        let fp32_scratch = staging
            .as_mut()
            .expect("non-aligned experts must own fp32 staging");
        profiler.measure(stream, names[0], || {
            for expert in 0..E {
                copy_device_region(
                    &mut fp32_scratch.a,
                    0,
                    input.wide(),
                    expert * C * input_width,
                    C * input_width,
                    stream,
                )?;
                copy_device_region(
                    &mut fp32_scratch.b,
                    0,
                    output_gradient.wide(),
                    expert * C * output_width,
                    C * output_width,
                    stream,
                )?;
                copy_device_region(
                    &mut fp32_scratch.c,
                    0,
                    weight_gradient,
                    expert * input_width * output_width,
                    input_width * output_width,
                    stream,
                )?;
                unsafe {
                    fp32.register_gemm_tn_accumulate(
                        stream,
                        fp32_launch_config(input_width, output_width),
                        input_width,
                        output_width,
                        C,
                        &fp32_scratch.a,
                        &fp32_scratch.b,
                        &mut fp32_scratch.c,
                    )?;
                }
                copy_device_region(
                    weight_gradient,
                    expert * input_width * output_width,
                    &fp32_scratch.c,
                    0,
                    input_width * output_width,
                    stream,
                )?;
            }
            Ok(())
        })?;
        // SAFETY: every expert's slice fits the oracle scratch, which is sized
        // for the widest operand this call can see.
        profiler.measure(stream, names[1], || unsafe {
            for expert in 0..E {
                copy_device_region(
                    &mut fp32_scratch.a,
                    0,
                    output_gradient.wide(),
                    expert * C * output_width,
                    C * output_width,
                    stream,
                )?;
                tensor.widen_bf16_region(
                    stream,
                    pairs_config(input_width * output_width),
                    weights,
                    (expert * input_width * output_width) as u32,
                    (input_width * output_width) as u32,
                    &mut fp32_scratch.b,
                )?;
                unsafe {
                    fp32.register_gemm_nt_store(
                        stream,
                        fp32_launch_config(C, input_width),
                        C,
                        input_width,
                        output_width,
                        &fp32_scratch.a,
                        &fp32_scratch.b,
                        &mut fp32_scratch.c,
                    )?;
                }
                copy_device_region(
                    input_gradient,
                    expert * C * input_width,
                    &fp32_scratch.c,
                    0,
                    C * input_width,
                    stream,
                )?;
            }
            Ok(())
        })
    }
}

/// Stacked GPU weights for `E` capacity-binned SwiGLU experts.
///
/// Gate and up projections share one `[E, D, 2, FF]` master/gradient entry;
/// down projections share one `[E, FF, D]` entry. Aligned shapes also own one
/// persistent packed-bf16 compute allocation per entry.
pub struct GpuExpertFfn<const E: usize, const D: usize, const FF: usize> {
    pub gate_up: GpuBf16Tensor<Rank4<E, D, 2, FF>>,
    pub d_gate_up: GpuTensor<f32, Rank4<E, D, 2, FF>>,
    pub down: GpuBf16Tensor<Rank3<E, FF, D>>,
    pub d_down: GpuTensor<f32, Rank3<E, FF, D>>,
    gate_up_compute: Option<StackedBf16Weights>,
    down_compute: Option<StackedBf16Weights>,
}

impl<const E: usize, const D: usize, const FF: usize> GpuExpertFfn<E, D, FF> {
    pub fn from_cpu<const C: usize>(
        stream: &CudaStream,
        experts: &[nn::ExpertFfn<C, D, FF>; E],
    ) -> Result<Self, Box<dyn Error>> {
        assert!(E > 0, "GPU expert count must be non-zero");
        let mut gate_up = vec![0.0; E * D * 2 * FF];
        let mut down = vec![0.0; E * FF * D];
        for (expert, source) in experts.iter().enumerate() {
            for input in 0..D {
                let destination = (expert * D + input) * 2 * FF;
                gate_up[destination..destination + FF]
                    .copy_from_slice(&source.gate_proj.w.as_slice()[input * FF..(input + 1) * FF]);
                gate_up[destination + FF..destination + 2 * FF]
                    .copy_from_slice(&source.up_proj.w.as_slice()[input * FF..(input + 1) * FF]);
            }
            down[expert * FF * D..(expert + 1) * FF * D]
                .copy_from_slice(source.down_proj.w.as_slice());
        }
        let aligned = D.is_multiple_of(TC_TILE) && FF.is_multiple_of(TC_TILE);
        let gate_up_master = GpuBf16Tensor::from_f32_host(stream, &gate_up)?;
        let down_master = GpuBf16Tensor::from_f32_host(stream, &down)?;
        let gate_up_compute = aligned
            .then(|| {
                StackedBf16Weights::new(stream, gate_up_master.as_words(), &gate_up, E, D, 2 * FF)
            })
            .transpose()?;
        let down_compute = aligned
            .then(|| StackedBf16Weights::new(stream, down_master.as_words(), &down, E, FF, D))
            .transpose()?;
        Ok(Self {
            gate_up: gate_up_master,
            d_gate_up: GpuTensor::zeros(stream)?,
            down: down_master,
            d_down: GpuTensor::zeros(stream)?,
            gate_up_compute,
            down_compute,
        })
    }

    pub fn forward<const C: usize>(
        &self,
        workspace: &mut GpuExpertWorkspace<E, C, D, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        dense: &dense_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.forward_profiled(
            &mut workspace.acts,
            &mut workspace.scratch,
            stream,
            tensor,
            fp32,
            tcgen05,
            dense,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_profiled<const C: usize, P: KernelProfiler>(
        &self,
        acts: &mut GpuExpertActs<E, C, D, FF>,
        scratch: &mut GpuExpertScratch<E, C, D, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        dense: &dense_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        let GpuExpertActs {
            bin_input,
            gate_up,
            activated,
            ..
        } = &mut *acts;
        // The GEMM writes the interleaved `[E, C, 2, FF]` activation directly
        // and SwiGLU reads it in place, so the split copy that used to build
        // separate gate/up buffers never runs.
        expert_linear_forward::<E, C, P>(
            bin_input,
            self.gate_up.as_words(),
            self.gate_up_compute.as_ref(),
            gate_up.as_device_buffer_mut(),
            D,
            2 * FF,
            &mut scratch.fp32_staging,
            stream,
            tensor,
            fp32,
            tcgen05,
            profiler,
            "forward.experts.gate_up_gemm",
        )?;
        // SAFETY: `gate_up` holds E * C interleaved rows of 2 * FF elements,
        // and each arm launches over the element count its own output dtype
        // packs them into; the packed arm's FF is tcgen05-aligned, hence a
        // multiple of QUAD_LANES.
        profiler.measure(stream, "forward.experts.swiglu", || unsafe {
            match activated {
                ExpertPanel::Packed(panel) => dense.swiglu_forward_interleaved_bf16(
                    stream,
                    LaunchConfig::for_num_elems((E * C * FF / QUAD_LANES) as u32),
                    gate_up.as_device_buffer(),
                    FF as u32,
                    &mut panel.words,
                ),
                ExpertPanel::Wide(values) => dense.swiglu_forward_interleaved(
                    stream,
                    LaunchConfig::for_num_elems((E * C * FF) as u32),
                    gate_up.as_device_buffer(),
                    FF as u32,
                    values,
                ),
            }
        })?;
        expert_linear_forward::<E, C, P>(
            &acts.activated,
            self.down.as_words(),
            self.down_compute.as_ref(),
            acts.bin_output.as_device_buffer_mut(),
            FF,
            D,
            &mut scratch.fp32_staging,
            stream,
            tensor,
            fp32,
            tcgen05,
            profiler,
            "forward.experts.down_gemm",
        )?;
        Ok(())
    }

    pub fn backward<const C: usize>(
        &mut self,
        workspace: &mut GpuExpertWorkspace<E, C, D, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        dense: &dense_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.backward_profiled(
            &workspace.acts,
            &mut workspace.scratch,
            stream,
            tensor,
            fp32,
            tcgen05,
            dense,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_profiled<const C: usize, P: KernelProfiler>(
        &mut self,
        acts: &GpuExpertActs<E, C, D, FF>,
        scratch: &mut GpuExpertScratch<E, C, D, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        dense: &dense_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        expert_linear_backward::<E, C, P>(
            &acts.activated,
            &scratch.d_bin_output,
            self.down.as_words(),
            self.d_down.as_device_buffer_mut(),
            self.down_compute.as_ref(),
            scratch.d_activated.as_device_buffer_mut(),
            FF,
            D,
            &mut scratch.fp32_staging,
            stream,
            tensor,
            fp32,
            tcgen05,
            profiler,
            [
                "backward.experts.down_weight_gemm",
                "backward.experts.down_input_gemm",
            ],
        )?;
        let GpuExpertScratch {
            d_activated,
            d_gate_up,
            ..
        } = &mut *scratch;
        // One fused pass reads the interleaved gate/up activation and the
        // downstream gradient once and writes both interleaved gradient
        // halves in place, so the separate gate/up gradient buffers and the
        // join pass that merged them never exist.
        // SAFETY: `gate_up` holds E * C interleaved rows of 2 * FF elements,
        // `d_activated` E * C * FF, and each arm launches over the element
        // count its own output dtype packs them into; the packed arm's FF is
        // tcgen05-aligned, hence a multiple of QUAD_LANES.
        profiler.measure(stream, "backward.experts.swiglu", || unsafe {
            match d_gate_up {
                ExpertPanel::Packed(panel) => dense.swiglu_backward_interleaved_bf16(
                    stream,
                    LaunchConfig::for_num_elems((E * C * FF / QUAD_LANES) as u32),
                    acts.gate_up.as_device_buffer(),
                    d_activated.as_device_buffer(),
                    FF as u32,
                    &mut panel.words,
                ),
                ExpertPanel::Wide(values) => dense.swiglu_backward_interleaved(
                    stream,
                    LaunchConfig::for_num_elems((E * C * FF) as u32),
                    acts.gate_up.as_device_buffer(),
                    d_activated.as_device_buffer(),
                    FF as u32,
                    values,
                ),
            }
        })?;
        expert_linear_backward::<E, C, P>(
            &acts.bin_input,
            &scratch.d_gate_up,
            self.gate_up.as_words(),
            self.d_gate_up.as_device_buffer_mut(),
            self.gate_up_compute.as_ref(),
            scratch.d_bin_input.as_device_buffer_mut(),
            D,
            2 * FF,
            &mut scratch.fp32_staging,
            stream,
            tensor,
            fp32,
            tcgen05,
            profiler,
            [
                "backward.experts.gate_up_weight_gemm",
                "backward.experts.gate_up_input_gemm",
            ],
        )?;
        Ok(())
    }

    /// Clear both stacked gradients. An AdamW step clears the gradient it
    /// consumes, so a training loop never calls this; a backward checked on
    /// its own still needs a way back to a known-zero start.
    pub fn zero_grad(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        fill_zero(
            &mut self.d_gate_up,
            stream,
            tensor,
            &mut profiler,
            "zero_grad.experts.gate_up",
        )?;
        fill_zero(
            &mut self.d_down,
            stream,
            tensor,
            &mut profiler,
            "zero_grad.experts.down",
        )
    }

    /// [`GpuLinear::sync_compute`] over both stacked expert masters.
    pub fn sync_compute(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        if let Some(compute) = &mut self.gate_up_compute {
            compute.sync_from_master(self.gate_up.as_words(), stream, tensor)?;
        }
        if let Some(compute) = &mut self.down_compute {
            compute.sync_from_master(self.down.as_words(), stream, tensor)?;
        }
        Ok(())
    }

    /// [`GpuLinear::adamw_step`] over the stacked `[E * D, 2 * FF]` gate/up
    /// master.
    fn adamw_step_gate_up(
        &mut self,
        moments: &mut GpuAdamWMoments<Rank4<E, D, 2, FF>>,
        config: MasterAdamW,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        match &mut self.gate_up_compute {
            Some(compute) => self.gate_up.adamw_step_transposed(
                &mut self.d_gate_up,
                moments,
                config,
                E * D,
                2 * FF,
                &mut compute.transposed,
                stream,
                tensor,
            ),
            None => self
                .gate_up
                .adamw_step(&mut self.d_gate_up, moments, config, stream, tensor),
        }
    }

    /// [`Self::adamw_step_gate_up`] for the stacked `[E * FF, D]` down master.
    fn adamw_step_down(
        &mut self,
        moments: &mut GpuAdamWMoments<Rank3<E, FF, D>>,
        config: MasterAdamW,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        match &mut self.down_compute {
            Some(compute) => self.down.adamw_step_transposed(
                &mut self.d_down,
                moments,
                config,
                E * FF,
                D,
                &mut compute.transposed,
                stream,
                tensor,
            ),
            None => self
                .down
                .adamw_step(&mut self.d_down, moments, config, stream, tensor),
        }
    }

    /// The stacked global transpose of the gate/up master. Parity-test
    /// accessor: binaries other than the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn gate_up_transposed_words(&self) -> Option<&DeviceBuffer<u32>> {
        self.gate_up_compute
            .as_ref()
            .map(|weights| &weights.transposed)
    }

    /// The stacked global transpose of the down master; see
    /// [`Self::gate_up_transposed_words`].
    #[allow(dead_code)]
    pub fn down_transposed_words(&self) -> Option<&DeviceBuffer<u32>> {
        self.down_compute
            .as_ref()
            .map(|weights| &weights.transposed)
    }
}

/// Capacity-bin activations one backward pass will read again: forward writes
/// them and every block in a deep model owns its own copy.
///
/// `gate_up` is the interleaved `[E, C, 2, FF]` gate/up activation the fused
/// GEMM writes; it stays fp32 for the SwiGLU pointwise math, as does
/// `bin_output` for the gate-gradient dot product. `bin_input` and
/// `activated` are packed bf16 because every reader of those two is a tcgen05
/// operand (SPEC §7.1). The fp32 buffers occupy `4 * E * C * (D + 2 * FF)`
/// bytes plus `2 * E * C * (D + FF)` packed.
pub struct GpuExpertActs<const E: usize, const C: usize, const D: usize, const FF: usize> {
    pub bin_input: ExpertPanel,
    gate_up: GpuTensor<f32, Rank4<E, C, 2, FF>>,
    activated: ExpertPanel,
    pub bin_output: GpuTensor<f32, Rank3<E, C, D>>,
}

impl<const E: usize, const C: usize, const D: usize, const FF: usize> GpuExpertActs<E, C, D, FF> {
    pub fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        assert!(E > 0 && C > 0 && D > 0 && FF > 0);
        assert!(E * C * D <= u32::MAX as usize);
        assert!(E * C * FF <= u32::MAX as usize);
        let aligned = expert_tcgen05_aligned::<C, D, FF>();
        Ok(Self {
            bin_input: ExpertPanel::new(stream, E, C, D, aligned)?,
            gate_up: GpuTensor::zeros(stream)?,
            activated: ExpertPanel::new(stream, E, C, FF, aligned)?,
            bin_output: GpuTensor::zeros(stream)?,
        })
    }
}

/// Expert buffers no backward pass reads after the launch that consumed them,
/// so one instance serves every block of a deep model.
///
/// `d_bin_output` and `d_gate_up` are packed bf16 (every reader is a tcgen05
/// operand, SPEC §7.1); the rest stay fp32 as epilogue targets or pointwise
/// operands, occupying `4 * E * C * (D + FF)` bytes plus
/// `2 * E * C * (D + 2 * FF)` packed. Aligned tcgen05 shapes now stage nothing
/// at all — every operand is addressed in place — so only the non-aligned
/// oracle allocates the three one-expert fp32 staging buffers.
pub struct GpuExpertScratch<const E: usize, const C: usize, const D: usize, const FF: usize> {
    pub d_bin_output: ExpertPanel,
    d_activated: GpuTensor<f32, Rank3<E, C, FF>>,
    d_gate_up: ExpertPanel,
    pub d_bin_input: GpuTensor<f32, Rank3<E, C, D>>,
    fp32_staging: Option<ExpertFp32Scratch>,
}

impl<const E: usize, const C: usize, const D: usize, const FF: usize>
    GpuExpertScratch<E, C, D, FF>
{
    pub fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        assert!(E > 0 && C > 0 && D > 0 && FF > 0);
        let aligned = expert_tcgen05_aligned::<C, D, FF>();
        Ok(Self {
            d_bin_output: ExpertPanel::new(stream, E, C, D, aligned)?,
            d_activated: GpuTensor::zeros(stream)?,
            d_gate_up: ExpertPanel::new(stream, E, C, 2 * FF, aligned)?,
            d_bin_input: GpuTensor::zeros(stream)?,
            fp32_staging: (!aligned)
                .then(|| ExpertFp32Scratch::new::<C, D, FF>(stream))
                .transpose()?,
        })
    }

    pub fn tcgen05_active(&self) -> bool {
        self.fp32_staging.is_none()
    }
}

/// One block's activations plus the shared scratch, bundled for the expert
/// parity gates and other single-block callers.
pub struct GpuExpertWorkspace<const E: usize, const C: usize, const D: usize, const FF: usize> {
    pub acts: GpuExpertActs<E, C, D, FF>,
    pub scratch: GpuExpertScratch<E, C, D, FF>,
}

impl<const E: usize, const C: usize, const D: usize, const FF: usize>
    GpuExpertWorkspace<E, C, D, FF>
{
    pub fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            acts: GpuExpertActs::new(stream)?,
            scratch: GpuExpertScratch::new(stream)?,
        })
    }

    /// Uploads in place: the panels' TMA descriptors pin their allocations.
    pub fn upload_bins(&mut self, values: &[f32], stream: &CudaStream) -> Result<(), DriverError> {
        assert_eq!(values.len(), E * C * D);
        self.acts.bin_input.upload(values, stream)
    }

    pub fn upload_output_gradient(
        &mut self,
        values: &[f32],
        stream: &CudaStream,
    ) -> Result<(), DriverError> {
        assert_eq!(values.len(), E * C * D);
        self.scratch.d_bin_output.upload(values, stream)
    }

    pub fn tcgen05_active(&self) -> bool {
        self.scratch.tcgen05_active()
    }
}

/// GPU AdamW state for the two stacked expert parameter entries.
pub struct GpuExpertAdamW<const E: usize, const D: usize, const FF: usize> {
    config: AdamWConfig,
    step: u64,
    pub gate_up: GpuAdamWMoments<Rank4<E, D, 2, FF>>,
    pub down: GpuAdamWMoments<Rank3<E, FF, D>>,
}

impl<const E: usize, const D: usize, const FF: usize> GpuExpertAdamW<E, D, FF> {
    pub fn new(stream: &CudaStream, config: AdamWConfig) -> Result<Self, DriverError> {
        config.validate();
        Ok(Self {
            config,
            step: 0,
            gate_up: GpuAdamWMoments::zeros(stream)?,
            down: GpuAdamWMoments::zeros(stream)?,
        })
    }

    pub fn update(
        &mut self,
        experts: &mut GpuExpertFfn<E, D, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        self.step = self
            .step
            .checked_add(1)
            .expect("expert AdamW step overflow");
        let corrections = self.config.bias_correction(self.step);
        let decay = self.config.weight_decay;
        experts.adamw_step_gate_up(
            &mut self.gate_up,
            master_adamw(
                self.config,
                decay,
                corrections,
                self.step,
                parameter_id::EXPERT_GATE_UP,
            ),
            stream,
            tensor,
        )?;
        experts.adamw_step_down(
            &mut self.down,
            master_adamw(
                self.config,
                decay,
                corrections,
                self.step,
                parameter_id::EXPERT_DOWN,
            ),
            stream,
            tensor,
        )
    }
}

pub struct GpuRmsNorm<const D: usize> {
    pub w: GpuTensor<f32, Rank1<D>>,
    pub dw: GpuTensor<f32, Rank1<D>>,
    eps: f32,
}

impl<const D: usize> GpuRmsNorm<D> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layer: &nn::RmsNorm<N, D>,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            w: GpuTensor::from_cpu(stream, &layer.w)?,
            dw: GpuTensor::zeros(stream)?,
            eps: layer.eps,
        })
    }

    fn forward_into<const N: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        y: &mut GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &dense_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        // SAFETY: norm launch dimensions and Rank2 buffers agree on N * D, and
        // the tile arm is only taken at a shape `norm_tiles` accepted.
        profiler.measure(stream, name, || unsafe {
            match norm_tiles(N, D) {
                Some(tiles) => kernels.rms_norm_forward_tile(
                    stream,
                    tiles,
                    x.as_device_buffer(),
                    self.w.as_device_buffer(),
                    self.eps,
                    D as u32,
                    y.as_device_buffer_mut(),
                ),
                None => kernels.rms_norm_forward_fast(
                    stream,
                    norm_config::<N>(),
                    x.as_device_buffer(),
                    self.w.as_device_buffer(),
                    self.eps,
                    D as u32,
                    y.as_device_buffer_mut(),
                ),
            }
        })
    }

    fn backward_into<const N: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        dy: &GpuTensor<f32, Rank2<N, D>>,
        dx: &mut GpuTensor<f32, Rank2<N, D>>,
        inv: &mut GpuTensor<f32, Rank1<N>>,
        stream: &CudaStream,
        kernels: &dense_kernels::LoadedModule,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<(), DriverError> {
        // SAFETY: norm launch dimensions and saved/output buffers agree on N * D.
        profiler.measure(stream, names[0], || unsafe {
            kernels.rms_norm_backward_x_fast(
                stream,
                norm_config::<N>(),
                x.as_device_buffer(),
                self.w.as_device_buffer(),
                dy.as_device_buffer(),
                self.eps,
                D as u32,
                dx.as_device_buffer_mut(),
                inv.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, names[1], || unsafe {
            kernels.rms_norm_backward_weight_fast(
                stream,
                norm_weight_config::<N, D>(),
                x.as_device_buffer(),
                dy.as_device_buffer(),
                inv.as_device_buffer(),
                N as u32,
                D as u32,
                self.dw.as_device_buffer_mut(),
            )
        })
    }
}

/// Token embedding with a bf16 master and an fp32, atomically accumulated
/// gradient. It owns no compute copy: the lookup kernel reads the packed
/// master directly.
pub struct GpuEmbedding<const VOCAB: usize, const D: usize> {
    pub w: GpuBf16Tensor<Rank2<VOCAB, D>>,
    pub dw: GpuTensor<f32, Rank2<VOCAB, D>>,
}

impl<const VOCAB: usize, const D: usize> GpuEmbedding<VOCAB, D> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layer: &nn::Embedding<N, VOCAB, D>,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            w: GpuBf16Tensor::from_f32_host(stream, layer.w.as_slice())?,
            dw: GpuTensor::zeros(stream)?,
        })
    }

    fn forward_into<const N: usize, P: KernelProfiler>(
        &self,
        tokens: &GpuTensor<u32, Rank1<N>>,
        y: &mut GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &dense_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        // SAFETY: token and output shapes satisfy the embedding launch contract.
        profiler.measure(stream, name, || unsafe {
            kernels.embedding_forward(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                self.w.as_words(),
                tokens.as_device_buffer(),
                D as u32,
                y.as_device_buffer_mut(),
            )
        })
    }

    fn backward<const N: usize, P: KernelProfiler>(
        &mut self,
        tokens: &GpuTensor<u32, Rank1<N>>,
        dy: &GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &dense_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.embedding_backward_scatter(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                tokens.as_device_buffer(),
                dy.as_device_buffer(),
                D as u32,
                self.dw.as_device_buffer_mut(),
            )
        })
    }
}

/// bf16 lm-head over a bf16 master (§7; masters converted in #57).
///
/// The weight is needed in both layouts the tcgen05 `C = A B^T` form takes as
/// a K-contiguous `B` operand: `[D, VP]` for the input-gradient GEMM and
/// `[VP, D]` for the forward. The first *is* the master — `w_tma` maps its
/// words in place (#58) — so only `w_t` is stored beside it, and only `w_t` is
/// refreshed after a step. `dw` accumulates in packed bf16, produced directly
/// by the tcgen05 accumulate epilogue.
pub struct GpuBf16Head<const D: usize, const VP: usize> {
    pub master: GpuBf16Tensor<Rank2<D, VP>>,
    w_t: DeviceBuffer<u32>,
    dw: DeviceBuffer<u32>,
    w_tma: Bf16PairsTmaMap,
    w_t_tma: Bf16PairsTmaMap,
}

impl<const D: usize, const VP: usize> GpuBf16Head<D, VP> {
    fn from_cpu<const N: usize, const VOCAB: usize>(
        stream: &CudaStream,
        layer: &nn::Linear<N, D, VOCAB>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(VP >= VOCAB);
        let mut padded = vec![0.0f32; D * VP];
        for row in 0..D {
            padded[row * VP..row * VP + VOCAB]
                .copy_from_slice(&layer.w.as_slice()[row * VOCAB..(row + 1) * VOCAB]);
        }
        Self::from_master_values(stream, &padded)
    }

    /// Rebuild the head from padded `[D, VP]` fp32 values, rounding the master
    /// and its transpose on the host.
    pub(crate) fn from_master_values(
        stream: &CudaStream,
        values: &[f32],
    ) -> Result<Self, Box<dyn Error>> {
        assert_eq!(values.len(), D * VP);
        let mut transposed_values = vec![0.0f32; VP * D];
        for row in 0..D {
            for column in 0..VP {
                transposed_values[column * D + row] = values[row * VP + column];
            }
        }

        let master = GpuBf16Tensor::from_f32_host(stream, values)?;
        let w_t = DeviceBuffer::from_host(stream, &pack_bf16_pairs(&transposed_values))?;
        let dw = DeviceBuffer::zeroed(stream, D * VP / 2)?;
        // SAFETY: the master and `w_t` live in this struct beside their maps
        // and are never reallocated — checkpoint resume rebuilds the whole
        // head through this constructor, maps included.
        let w_tma = unsafe {
            create_bf16_pairs_tma_map(stream, master.as_words(), VP, D, TmaLayout::KMajor)?
        };
        let w_t_tma = unsafe { create_bf16_pairs_tma_map(stream, &w_t, D, VP, TmaLayout::KMajor)? };
        Ok(Self {
            master,
            w_t,
            dw,
            w_tma,
            w_t_tma,
        })
    }

    /// Packed-bf16 weight gradient. Parity-test accessor: binaries other than
    /// the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn dw_words(&self) -> &DeviceBuffer<u32> {
        &self.dw
    }

    /// Packed-bf16 `[VP, D]` transposed compute weights. Parity-test accessor:
    /// binaries other than the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn w_t_words(&self) -> &DeviceBuffer<u32> {
        &self.w_t
    }

    fn zero_grad<P: KernelProfiler>(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        // SAFETY: dw contains exactly D * VP / 2 packed words.
        profiler.measure(stream, name, || unsafe {
            kernels.fill_u32(stream, pairs_config(D * VP / 2), 0, &mut self.dw)
        })
    }

    fn forward_into<const NP: usize, P: KernelProfiler>(
        &self,
        x_tma: &Bf16PairsTmaMap,
        logits: &mut DeviceBuffer<u32>,
        stream: &CudaStream,
        kernels: &Tcgen05Gemm,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.store(
                stream,
                tcgen05_launch_config(NP, VP, D),
                x_tma.as_ptr(),
                self.w_t_tma.as_ptr(),
                logits,
                VP as u32,
                D as u32,
            )
        })
    }

    /// `dw += x^T dlogits` from the transposed operands staged in the
    /// workspace; padded token rows and vocabulary columns contribute zeros.
    /// `dW += head_inputᵀ·dlogits`, reading both operands MN-major straight
    /// out of the forward's own `[NP, D]` and `[NP, VP]` bf16 panels (#53) —
    /// the two whole-panel transposes this used to need are gone.
    fn backward_weight<const NP: usize, P: KernelProfiler>(
        &mut self,
        x_mn_tma: &Bf16PairsTmaMap,
        dlogits_mn_tma: &Bf16PairsTmaMap,
        stream: &CudaStream,
        kernels: &Tcgen05Gemm,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.accumulate_transposed(
                stream,
                tcgen05_launch_config(D, VP, NP),
                x_mn_tma.as_ptr(),
                dlogits_mn_tma.as_ptr(),
                &mut self.dw,
                VP as u32,
                NP as u32,
            )
        })
    }

    fn backward_input<const NP: usize, P: KernelProfiler>(
        &self,
        dlogits_tma: &Bf16PairsTmaMap,
        dx: &mut DeviceBuffer<u32>,
        stream: &CudaStream,
        kernels: &Tcgen05Gemm,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.store(
                stream,
                tcgen05_launch_config(NP, D, VP),
                dlogits_tma.as_ptr(),
                self.w_tma.as_ptr(),
                dx,
                D as u32,
                VP as u32,
            )
        })
    }

    fn adamw_step(
        &mut self,
        moments: &mut GpuAdamWMoments<Rank2<D, VP>>,
        config: MasterAdamW,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        // SAFETY: packed gradients, packed masters, both fp32 moments, and the
        // `[VP, D]` transpose all describe the same D * VP parameter matrix,
        // and the launch tiles it exactly.
        unsafe {
            kernels.adamw_bf16_master_packed_grad_transposed(
                stream,
                master_transpose_config(D, VP),
                &mut self.dw,
                D as u32,
                VP as u32,
                config.learning_rate,
                config.beta1,
                config.beta2,
                config.epsilon,
                config.weight_decay,
                config.first_correction,
                config.second_correction,
                config.rounding,
                config.seed,
                self.master.as_words_mut(),
                moments.first.as_device_buffer_mut(),
                moments.second.as_device_buffer_mut(),
                &mut self.w_t,
            )
        }
    }
}

pub struct GpuDenseDense<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> {
    pub embedding: GpuEmbedding<VOCAB, D>,
    pub attention_norm: GpuRmsNorm<D>,
    pub qkv_proj: GpuGroupedLinear<D, 3, D>,
    pub o_proj: GpuLinear<D, D>,
    pub ffn_norm: GpuRmsNorm<D>,
    pub gate_up_proj: GpuGroupedLinear<D, 2, FF>,
    pub down_proj: GpuLinear<FF, D>,
    pub final_norm: GpuRmsNorm<D>,
    pub lm_head: GpuBf16Head<D, VP>,
}

/// GPU-resident AdamW state mirroring every model parameter.
///
/// The lm-head moments span the padded `[D, VP]` master; padded columns hold
/// zeros forever because their gradients are exactly zero.
pub struct GpuDenseDenseAdamW<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize>
{
    config: AdamWConfig,
    step: u64,
    pub embedding: GpuAdamWMoments<Rank2<VOCAB, D>>,
    pub attention_norm: GpuAdamWMoments<Rank1<D>>,
    pub qkv_proj: GpuAdamWMoments<Rank3<D, 3, D>>,
    pub o_proj: GpuAdamWMoments<Rank2<D, D>>,
    pub ffn_norm: GpuAdamWMoments<Rank1<D>>,
    pub gate_up_proj: GpuAdamWMoments<Rank3<D, 2, FF>>,
    pub down_proj: GpuAdamWMoments<Rank2<FF, D>>,
    pub final_norm: GpuAdamWMoments<Rank1<D>>,
    pub lm_head: GpuAdamWMoments<Rank2<D, VP>>,
}

impl<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize>
    GpuDenseDenseAdamW<VOCAB, VP, D, FF>
{
    pub fn new(stream: &CudaStream, config: AdamWConfig) -> Result<Self, DriverError> {
        config.validate();
        Ok(Self {
            config,
            step: 0,
            embedding: GpuAdamWMoments::zeros(stream)?,
            attention_norm: GpuAdamWMoments::zeros(stream)?,
            qkv_proj: GpuAdamWMoments::zeros(stream)?,
            o_proj: GpuAdamWMoments::zeros(stream)?,
            ffn_norm: GpuAdamWMoments::zeros(stream)?,
            gate_up_proj: GpuAdamWMoments::zeros(stream)?,
            down_proj: GpuAdamWMoments::zeros(stream)?,
            final_norm: GpuAdamWMoments::zeros(stream)?,
            lm_head: GpuAdamWMoments::zeros(stream)?,
        })
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    pub(crate) fn restore_step(&mut self, step: u64) {
        self.step = step;
    }

    pub fn update<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
    >(
        &mut self,
        model: &mut GpuDenseDense<N, NP, T, VOCAB, VP, D, H, HD, FF>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.update_profiled(model, stream, kernels, &mut profiler)
    }

    pub fn update_profiled<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        P: KernelProfiler,
    >(
        &mut self,
        model: &mut GpuDenseDense<N, NP, T, VOCAB, VP, D, H, HD, FF>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        self.step = self.step.checked_add(1).expect("AdamW step overflow");
        let (first_correction, second_correction) = self.config.bias_correction(self.step);

        let corrections = (first_correction, second_correction);
        let step = self.step;

        // bf16 masters: one fused kernel does the fp32 update and the rounded
        // write-back. Norms keep fp32 storage and the plain `adamw` kernel.
        macro_rules! embedding {
            ($field:ident, $id:expr) => {
                profiler.measure(
                    stream,
                    concat!("optimizer.", stringify!($field), ".adamw"),
                    || {
                        model.$field.w.adamw_step(
                            &mut model.$field.dw,
                            &mut self.$field,
                            master_adamw(
                                self.config,
                                self.config.weight_decay,
                                corrections,
                                step,
                                $id,
                            ),
                            stream,
                            kernels,
                        )
                    },
                )?;
            };
        }
        macro_rules! master {
            ($field:ident, $id:expr) => {
                profiler.measure(
                    stream,
                    concat!("optimizer.", stringify!($field), ".adamw"),
                    || {
                        model.$field.adamw_step(
                            &mut self.$field,
                            master_adamw(
                                self.config,
                                self.config.weight_decay,
                                corrections,
                                step,
                                $id,
                            ),
                            stream,
                            kernels,
                        )
                    },
                )?;
            };
        }
        macro_rules! norm {
            ($field:ident) => {
                profiler.measure(
                    stream,
                    concat!("optimizer.", stringify!($field), ".adamw"),
                    || {
                        model.$field.w.adamw_step(
                            &mut model.$field.dw,
                            &mut self.$field,
                            self.config.learning_rate,
                            self.config.beta1,
                            self.config.beta2,
                            self.config.epsilon,
                            0.0,
                            first_correction,
                            second_correction,
                            stream,
                            kernels,
                        )
                    },
                )?;
            };
        }

        embedding!(embedding, parameter_id::EMBEDDING);
        norm!(attention_norm);
        master!(qkv_proj, parameter_id::QKV_PROJ);
        master!(o_proj, parameter_id::O_PROJ);
        norm!(ffn_norm);
        master!(gate_up_proj, parameter_id::GATE_UP_PROJ);
        master!(down_proj, parameter_id::DOWN_PROJ);
        norm!(final_norm);
        profiler.measure(stream, "optimizer.lm_head.adamw", || {
            model.lm_head.adamw_step(
                &mut self.lm_head,
                master_adamw(
                    self.config,
                    self.config.weight_decay,
                    corrections,
                    step,
                    parameter_id::LM_HEAD,
                ),
                stream,
                kernels,
            )
        })?;
        Ok(())
    }
}

/// Single-block Dense with the dense SwiGLU branch substituted by a statically
/// shaped mixture of experts. Routing remains runtime data.
/// Weights for one MoE decoder block: pre-norm attention followed by a
/// top-k routed expert FFN.
pub struct GpuBlock<const D: usize, const FF: usize, const E: usize> {
    pub attention_norm: GpuRmsNorm<D>,
    pub qkv_proj: GpuGroupedLinear<D, 3, D>,
    pub o_proj: GpuLinear<D, D>,
    pub ffn_norm: GpuRmsNorm<D>,
    pub router: GpuTensor<f32, Rank2<D, E>>,
    pub d_router: GpuTensor<f32, Rank2<D, E>>,
    pub experts: GpuExpertFfn<E, D, FF>,
}

/// An `L`-deep stack of [`GpuBlock`]s between a token embedding and a bf16
/// lm-head.
pub struct GpuDense<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
    const L: usize = 1,
> {
    pub embedding: GpuEmbedding<VOCAB, D>,
    pub blocks: Vec<GpuBlock<D, FF, E>>,
    pub final_norm: GpuRmsNorm<D>,
    pub lm_head: GpuBf16Head<D, VP>,
}

/// AdamW moments for one [`GpuBlock`]'s parameters.
pub struct GpuBlockAdamW<const D: usize, const FF: usize, const E: usize> {
    pub attention_norm: GpuAdamWMoments<Rank1<D>>,
    pub qkv_proj: GpuAdamWMoments<Rank3<D, 3, D>>,
    pub o_proj: GpuAdamWMoments<Rank2<D, D>>,
    pub ffn_norm: GpuAdamWMoments<Rank1<D>>,
    pub router: GpuAdamWMoments<Rank2<D, E>>,
    pub expert_gate_up: GpuAdamWMoments<Rank4<E, D, 2, FF>>,
    pub expert_down: GpuAdamWMoments<Rank3<E, FF, D>>,
}

impl<const D: usize, const FF: usize, const E: usize> GpuBlockAdamW<D, FF, E> {
    fn zeros(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            attention_norm: GpuAdamWMoments::zeros(stream)?,
            qkv_proj: GpuAdamWMoments::zeros(stream)?,
            o_proj: GpuAdamWMoments::zeros(stream)?,
            ffn_norm: GpuAdamWMoments::zeros(stream)?,
            router: GpuAdamWMoments::zeros(stream)?,
            expert_gate_up: GpuAdamWMoments::zeros(stream)?,
            expert_down: GpuAdamWMoments::zeros(stream)?,
        })
    }
}

/// AdamW state for every MoE model parameter. The router remains on AdamW
/// regardless of future hidden-matrix Muon routing.
pub struct GpuDenseAdamW<
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const FF: usize,
    const E: usize,
> {
    config: AdamWConfig,
    aux_schedule: AuxLossSchedule,
    step: u64,
    pub embedding: GpuAdamWMoments<Rank2<VOCAB, D>>,
    pub blocks: Vec<GpuBlockAdamW<D, FF, E>>,
    pub final_norm: GpuAdamWMoments<Rank1<D>>,
    pub lm_head: GpuAdamWMoments<Rank2<D, VP>>,
}

impl<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize, const E: usize>
    GpuDenseAdamW<VOCAB, VP, D, FF, E>
{
    pub fn new(
        stream: &CudaStream,
        config: AdamWConfig,
        aux_schedule: AuxLossSchedule,
        layers: usize,
    ) -> Result<Self, DriverError> {
        config.validate();
        aux_schedule.validate();
        assert!(layers > 0, "optimizer needs at least one block");
        Ok(Self {
            config,
            aux_schedule,
            step: 0,
            embedding: GpuAdamWMoments::zeros(stream)?,
            blocks: (0..layers)
                .map(|_| GpuBlockAdamW::zeros(stream))
                .collect::<Result<_, _>>()?,
            final_norm: GpuAdamWMoments::zeros(stream)?,
            lm_head: GpuAdamWMoments::zeros(stream)?,
        })
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    pub fn aux_schedule(&self) -> AuxLossSchedule {
        self.aux_schedule
    }

    pub fn aux_coefficient(&self) -> f32 {
        self.aux_schedule.coefficient(self.step)
    }

    pub(crate) fn restore_step(&mut self, step: u64) {
        self.step = step;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        const K: usize,
        const C: usize,
        const L: usize,
    >(
        &mut self,
        model: &mut GpuDense<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.update_profiled(model, stream, kernels, &mut profiler)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_profiled<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        const K: usize,
        const C: usize,
        const L: usize,
        P: KernelProfiler,
    >(
        &mut self,
        model: &mut GpuDense<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        assert_eq!(self.blocks.len(), model.blocks.len());
        self.step = self.step.checked_add(1).expect("AdamW step overflow");
        let (first_correction, second_correction) = self.config.bias_correction(self.step);

        let corrections = (first_correction, second_correction);
        let step = self.step;
        let decay = self.config.weight_decay;
        let config = self.config;

        // bf16 masters take the fused update-and-round kernel, which also
        // refreshes any transposed compute operand and clears the gradient it
        // consumed; the fp32 parameters (norms, and the router by decision
        // #22) keep the plain fp32 `adamw`, which likewise clears its own.
        macro_rules! master {
            ($name:literal, $parameter:expr, $moments:expr, $id:expr) => {
                profiler.measure(stream, $name, || {
                    $parameter.adamw_step(
                        $moments,
                        master_adamw(config, decay, corrections, step, $id),
                        stream,
                        kernels,
                    )
                })?;
            };
        }
        macro_rules! fp32 {
            ($name:literal, $parameter:expr, $gradient:expr, $moments:expr, $decay:expr) => {
                profiler.measure(stream, $name, || {
                    $parameter.adamw_step(
                        $gradient,
                        $moments,
                        config.learning_rate,
                        config.beta1,
                        config.beta2,
                        config.epsilon,
                        $decay,
                        first_correction,
                        second_correction,
                        stream,
                        kernels,
                    )
                })?;
            };
        }

        profiler.measure(stream, "optimizer.embedding.adamw", || {
            model.embedding.w.adamw_step(
                &mut model.embedding.dw,
                &mut self.embedding,
                master_adamw(config, decay, corrections, step, parameter_id::EMBEDDING),
                stream,
                kernels,
            )
        })?;
        for (index, (block, moments)) in model
            .blocks
            .iter_mut()
            .zip(self.blocks.iter_mut())
            .enumerate()
        {
            fp32!(
                "optimizer.attention_norm.adamw",
                block.attention_norm.w,
                &mut block.attention_norm.dw,
                &mut moments.attention_norm,
                0.0
            );
            master!(
                "optimizer.qkv_proj.adamw",
                block.qkv_proj,
                &mut moments.qkv_proj,
                parameter_id::in_block(index, parameter_id::QKV_PROJ)
            );
            master!(
                "optimizer.o_proj.adamw",
                block.o_proj,
                &mut moments.o_proj,
                parameter_id::in_block(index, parameter_id::O_PROJ)
            );
            fp32!(
                "optimizer.ffn_norm.adamw",
                block.ffn_norm.w,
                &mut block.ffn_norm.dw,
                &mut moments.ffn_norm,
                0.0
            );
            fp32!(
                "optimizer.router.adamw",
                block.router,
                &mut block.d_router,
                &mut moments.router,
                decay
            );
            profiler.measure(stream, "optimizer.experts.gate_up.adamw", || {
                block.experts.adamw_step_gate_up(
                    &mut moments.expert_gate_up,
                    master_adamw(
                        config,
                        decay,
                        corrections,
                        step,
                        parameter_id::in_block(index, parameter_id::EXPERT_GATE_UP),
                    ),
                    stream,
                    kernels,
                )
            })?;
            profiler.measure(stream, "optimizer.experts.down.adamw", || {
                block.experts.adamw_step_down(
                    &mut moments.expert_down,
                    master_adamw(
                        config,
                        decay,
                        corrections,
                        step,
                        parameter_id::in_block(index, parameter_id::EXPERT_DOWN),
                    ),
                    stream,
                    kernels,
                )
            })?;
        }
        fp32!(
            "optimizer.final_norm.adamw",
            model.final_norm.w,
            &mut model.final_norm.dw,
            &mut self.final_norm,
            0.0
        );
        profiler.measure(stream, "optimizer.lm_head.adamw", || {
            model.lm_head.adamw_step(
                &mut self.lm_head,
                master_adamw(config, decay, corrections, step, parameter_id::LM_HEAD),
                stream,
                kernels,
            )
        })
    }
}

/// Device scratch for Muon's Newton–Schulz orthogonalization.
///
/// Every buffer is sized once for the largest hidden matrix and reused for
/// all of them via prefixes, so a steady-state Muon step performs no device
/// allocation. Gram-side buffers hold `min(rows, cols)^2` elements, which the
/// model bounds by `D^2`.
pub struct GpuMuonScratch {
    update: DeviceBuffer<f32>,
    x: DeviceBuffer<f32>,
    x_next: DeviceBuffer<f32>,
    product: DeviceBuffer<f32>,
    gram: DeviceBuffer<f32>,
    gram_squared: DeviceBuffer<f32>,
    polynomial: DeviceBuffer<f32>,
    sum_squares: DeviceBuffer<f32>,
}

impl GpuMuonScratch {
    pub fn new(
        stream: &CudaStream,
        max_update_elements: usize,
        max_matrix_elements: usize,
        max_gram_side: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            update: DeviceBuffer::zeroed(stream, max_update_elements)?,
            x: DeviceBuffer::zeroed(stream, max_matrix_elements)?,
            x_next: DeviceBuffer::zeroed(stream, max_matrix_elements)?,
            product: DeviceBuffer::zeroed(stream, max_matrix_elements)?,
            gram: DeviceBuffer::zeroed(stream, max_gram_side * max_gram_side)?,
            gram_squared: DeviceBuffer::zeroed(stream, max_gram_side * max_gram_side)?,
            polynomial: DeviceBuffer::zeroed(stream, max_gram_side * max_gram_side)?,
            sum_squares: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    /// Test hook mirroring `optim::zeroth_power_via_newton_schulz`: copy
    /// `input` (a dense `[rows, cols]` matrix) through the iteration and read
    /// the result back. Parity-binary only; other binaries see dead code.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn zeroth_power(
        &mut self,
        input: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        steps: usize,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
    ) -> Result<Vec<f32>, DriverError> {
        // SAFETY: the copied prefix and launch dimensions are bounded by the
        // scratch allocations validated by the constructor.
        unsafe {
            let elements = rows * cols;
            tensor.gather_group(
                stream,
                pairs_config(elements),
                input,
                1,
                0,
                cols as u32,
                elements as u32,
                &mut self.x,
            )?;
            newton_schulz_orthogonalize(self, rows, cols, steps, stream, tensor, gemm)?;
            let mut values = self.x.to_host_vec(stream)?;
            values.truncate(elements);
            Ok(values)
        }
    }
}

/// Orthogonalize the `[rows, cols]` prefix of `scratch.x` in place with the
/// quintic Newton–Schulz iteration, matching the CPU reference's math.
///
/// The Gram matrix always lives on the smaller axis. For wide matrices the
/// iteration is the reference's `X = aX + (bA + cA^2) X` with `A = X X^T`.
/// For tall matrices the reference transposes, iterates, and transposes back;
/// since `A = X^T X` and its polynomial `B` are symmetric, that whole
/// round-trip collapses to `X = aX + X B`, so no f32 transpose kernel exists.
#[allow(unused_unsafe)]
fn newton_schulz_orthogonalize(
    scratch: &mut GpuMuonScratch,
    rows: usize,
    cols: usize,
    steps: usize,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    gemm: &gemm_kernels::LoadedModule,
) -> Result<(), DriverError> {
    // SAFETY: every launch is bounded by elements or gram_elements, both of
    // which fit the preallocated Muon scratch buffers.
    unsafe {
        assert!(steps < 100, "Newton-Schulz steps must be less than 100");
        assert!(rows > 0 && cols > 0, "Muon matrices must be non-empty");
        let elements = rows * cols;
        let gram_side = rows.min(cols);
        let gram_elements = gram_side * gram_side;

        tensor.sum_squares(
            stream,
            reduction_config(),
            &scratch.x,
            elements as u32,
            &mut scratch.sum_squares,
        )?;
        tensor.scale_by_inv_norm(
            stream,
            pairs_config(elements),
            &scratch.x,
            &scratch.sum_squares,
            NEWTON_SCHULZ_EPSILON,
            elements as u32,
            &mut scratch.x_next,
        )?;
        std::mem::swap(&mut scratch.x, &mut scratch.x_next);

        for _ in 0..steps {
            if rows <= cols {
                // A = X X^T
                unsafe {
                    gemm.register_gemm_nt_store(
                        stream,
                        fp32_launch_config(rows, rows),
                        rows,
                        rows,
                        cols,
                        &scratch.x,
                        &scratch.x,
                        &mut scratch.gram,
                    )?;
                }
            } else {
                // A = X^T X; the fp32 family has no TN store, so zero + accumulate.
                tensor.fill(stream, pairs_config(gram_elements), 0.0, &mut scratch.gram)?;
                unsafe {
                    gemm.register_gemm_tn_accumulate(
                        stream,
                        fp32_launch_config(gram_side, gram_side),
                        gram_side,
                        gram_side,
                        rows,
                        &scratch.x,
                        &scratch.x,
                        &mut scratch.gram,
                    )?;
                }
            }
            unsafe {
                gemm.register_gemm_store(
                    stream,
                    fp32_launch_config(gram_side, gram_side),
                    gram_side,
                    gram_side,
                    gram_side,
                    &scratch.gram,
                    &scratch.gram,
                    &mut scratch.gram_squared,
                )?;
            }
            // B = b A + c A^2
            tensor.scaled_sum(
                stream,
                pairs_config(gram_elements),
                NEWTON_SCHULZ_B,
                &scratch.gram,
                NEWTON_SCHULZ_C,
                &scratch.gram_squared,
                gram_elements as u32,
                &mut scratch.polynomial,
            )?;
            if rows <= cols {
                // X = a X + B X
                unsafe {
                    gemm.register_gemm_store(
                        stream,
                        fp32_launch_config(rows, cols),
                        rows,
                        cols,
                        rows,
                        &scratch.polynomial,
                        &scratch.x,
                        &mut scratch.product,
                    )?;
                }
            } else {
                // X = a X + X B
                unsafe {
                    gemm.register_gemm_store(
                        stream,
                        fp32_launch_config(rows, cols),
                        rows,
                        cols,
                        cols,
                        &scratch.x,
                        &scratch.polynomial,
                        &mut scratch.product,
                    )?;
                }
            }
            tensor.scaled_sum(
                stream,
                pairs_config(elements),
                NEWTON_SCHULZ_A,
                &scratch.x,
                1.0,
                &scratch.product,
                elements as u32,
                &mut scratch.x_next,
            )?;
            std::mem::swap(&mut scratch.x, &mut scratch.x_next);
        }
        Ok(())
    }
}

/// One Muon update over a `[rows, groups, cols]` parameter whose groups are
/// independent `[rows, cols]` matrices (`groups = 1` for plain linears).
///
/// Momentum and the Nesterov interpolation are elementwise and run over the
/// whole interleaved buffer; orthogonalization and the fused decay/apply then
/// run per group so each projection is orthogonalized on its own, matching
/// the CPU reference's separate `q/k/v` and `gate/up` matrices.
///
/// Momentum, the Newton--Schulz iteration, and its f64 norm accumulation are
/// deliberately fp32 (SPEC §7); the bf16 master is widened on read inside the
/// apply kernel and rounded once on write-back.
#[allow(clippy::too_many_arguments)]
#[allow(unused_unsafe)]
fn muon_step_raw(
    parameter: &mut DeviceBuffer<u32>,
    gradient: &DeviceBuffer<f32>,
    momentum: &mut DeviceBuffer<f32>,
    rows: usize,
    groups: usize,
    cols: usize,
    config: MuonConfig,
    rounding: u32,
    seed: u64,
    scratch: &mut GpuMuonScratch,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    gemm: &gemm_kernels::LoadedModule,
) -> Result<(), DriverError> {
    // SAFETY: group offsets partition the parameter matrix and all scratch
    // launches are bounded by total or per_group.
    unsafe {
        let total = rows * groups * cols;
        let per_group = rows * cols;
        tensor.ema_momentum(
            stream,
            pairs_config(total),
            gradient,
            config.momentum,
            momentum,
        )?;
        let (gradient_weight, momentum_weight) = if config.nesterov {
            (1.0 - config.momentum, config.momentum)
        } else {
            (0.0, 1.0)
        };
        tensor.scaled_sum(
            stream,
            pairs_config(total),
            gradient_weight,
            gradient,
            momentum_weight,
            momentum,
            total as u32,
            &mut scratch.update,
        )?;

        let aspect_ratio_scale = ((rows as f32 / cols as f32).max(1.0)).sqrt();
        let decay = 1.0 - config.learning_rate * config.weight_decay;
        let update_scale = config.learning_rate * aspect_ratio_scale;
        for group in 0..groups {
            tensor.gather_group(
                stream,
                pairs_config(per_group),
                &scratch.update,
                groups as u32,
                group as u32,
                cols as u32,
                per_group as u32,
                &mut scratch.x,
            )?;
            newton_schulz_orthogonalize(
                scratch,
                rows,
                cols,
                config.newton_schulz_steps,
                stream,
                tensor,
                gemm,
            )?;
            // Group `g`'s slots of `update` are read only by group `g`'s gather
            // above, so writing the orthogonalized result back over them leaves
            // every later group's input intact.
            unsafe {
                tensor.scatter_group(
                    stream,
                    pairs_config(per_group),
                    &scratch.x,
                    groups as u32,
                    group as u32,
                    cols as u32,
                    per_group as u32,
                    &mut scratch.update,
                )?;
            }
        }
        tensor.muon_apply_bf16(
            stream,
            pairs_config(total / 2),
            &scratch.update,
            decay,
            update_scale,
            rounding,
            seed,
            parameter,
        )
    }
}

/// GPU-resident mixed Muon/AdamW state mirroring `optim::DenseMuon`'s
/// routing: hidden projection matrices take Muon, while the embedding,
/// norms, and lm-head keep AdamW (the lm-head over its padded `[D, VP]`
/// master, exactly as in [`GpuDenseDenseAdamW`]).
pub struct GpuDenseMuon<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize> {
    muon_config: MuonConfig,
    adamw_config: AdamWConfig,
    step: u64,
    scratch: GpuMuonScratch,
    pub embedding: GpuAdamWMoments<Rank2<VOCAB, D>>,
    pub attention_norm: GpuAdamWMoments<Rank1<D>>,
    pub qkv_proj: GpuMuonMomentum<Rank3<D, 3, D>>,
    pub o_proj: GpuMuonMomentum<Rank2<D, D>>,
    pub ffn_norm: GpuAdamWMoments<Rank1<D>>,
    pub gate_up_proj: GpuMuonMomentum<Rank3<D, 2, FF>>,
    pub down_proj: GpuMuonMomentum<Rank2<FF, D>>,
    pub final_norm: GpuAdamWMoments<Rank1<D>>,
    pub lm_head: GpuAdamWMoments<Rank2<D, VP>>,
}

impl<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize>
    GpuDenseMuon<VOCAB, VP, D, FF>
{
    pub fn new(
        stream: &CudaStream,
        muon_config: MuonConfig,
        adamw_config: AdamWConfig,
    ) -> Result<Self, DriverError> {
        muon_config.validate();
        adamw_config.validate();
        // Largest interleaved hidden parameter, largest single matrix, and
        // the Gram side (min dimension), which every hidden matrix bounds by D.
        let max_update_elements = (3 * D * D).max(2 * D * FF);
        let max_matrix_elements = (D * D).max(D * FF);
        Ok(Self {
            muon_config,
            adamw_config,
            step: 0,
            scratch: GpuMuonScratch::new(stream, max_update_elements, max_matrix_elements, D)?,
            embedding: GpuAdamWMoments::zeros(stream)?,
            attention_norm: GpuAdamWMoments::zeros(stream)?,
            qkv_proj: GpuMuonMomentum::zeros(stream)?,
            o_proj: GpuMuonMomentum::zeros(stream)?,
            ffn_norm: GpuAdamWMoments::zeros(stream)?,
            gate_up_proj: GpuMuonMomentum::zeros(stream)?,
            down_proj: GpuMuonMomentum::zeros(stream)?,
            final_norm: GpuAdamWMoments::zeros(stream)?,
            lm_head: GpuAdamWMoments::zeros(stream)?,
        })
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn muon_config(&self) -> MuonConfig {
        self.muon_config
    }

    pub fn adamw_config(&self) -> AdamWConfig {
        self.adamw_config
    }

    pub fn update<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
    >(
        &mut self,
        model: &mut GpuDenseDense<N, NP, T, VOCAB, VP, D, H, HD, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        self.step = self.step.checked_add(1).expect("Muon step overflow");
        let (first_correction, second_correction) = self.adamw_config.bias_correction(self.step);

        let corrections = (first_correction, second_correction);
        let step = self.step;
        let config = self.adamw_config;

        macro_rules! master_adamw_step {
            ($field:ident, $id:expr) => {
                model.$field.w.adamw_step(
                    &mut model.$field.dw,
                    &mut self.$field,
                    master_adamw(config, config.weight_decay, corrections, step, $id),
                    stream,
                    tensor,
                )?;
            };
        }
        macro_rules! norm {
            ($field:ident) => {
                model.$field.w.adamw_step(
                    &mut model.$field.dw,
                    &mut self.$field,
                    config.learning_rate,
                    config.beta1,
                    config.beta2,
                    config.epsilon,
                    0.0,
                    first_correction,
                    second_correction,
                    stream,
                    tensor,
                )?;
            };
        }
        macro_rules! muon {
            ($field:ident, $rows:expr, $groups:expr, $cols:expr, $id:expr) => {
                muon_step_raw(
                    model.$field.w.as_words_mut(),
                    model.$field.dw.as_device_buffer(),
                    self.$field.momentum.as_device_buffer_mut(),
                    $rows,
                    $groups,
                    $cols,
                    self.muon_config,
                    master_rounding_selector(),
                    stream_seed(step, $id),
                    &mut self.scratch,
                    stream,
                    tensor,
                    gemm,
                )?;
            };
        }

        master_adamw_step!(embedding, parameter_id::EMBEDDING);
        norm!(attention_norm);
        muon!(qkv_proj, D, 3, D, parameter_id::QKV_PROJ);
        muon!(o_proj, D, 1, D, parameter_id::O_PROJ);
        norm!(ffn_norm);
        muon!(gate_up_proj, D, 2, FF, parameter_id::GATE_UP_PROJ);
        muon!(down_proj, FF, 1, D, parameter_id::DOWN_PROJ);
        norm!(final_norm);
        // Muon writes its masters through `muon_apply_bf16`, which has no
        // transpose to emit, so those four still need the standalone pass.
        model.sync_linear_compute(stream, tensor)?;
        model.lm_head.adamw_step(
            &mut self.lm_head,
            master_adamw(
                config,
                config.weight_decay,
                corrections,
                step,
                parameter_id::LM_HEAD,
            ),
            stream,
            tensor,
        )
    }
}

struct InputStaging<const N: usize> {
    tokens: PinnedHostBuffer<u32>,
    targets: PinnedHostBuffer<u32>,
    copied: CudaEvent,
    pending: bool,
}

impl<const N: usize> InputStaging<N> {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            tokens: PinnedHostBuffer::zeroed(stream.context(), N)?,
            targets: PinnedHostBuffer::zeroed(stream.context(), N)?,
            copied: stream.context().new_event(None)?,
            pending: false,
        })
    }
}

/// Uploads one `(tokens, targets)` batch through double-buffered pinned
/// staging, validating every id against the vocabulary.
fn upload_token_inputs<const N: usize, const VOCAB: usize>(
    device_tokens: &mut GpuTensor<u32, Rank1<N>>,
    device_targets: &mut GpuTensor<u32, Rank1<N>>,
    staging: &mut [InputStaging<N>; 2],
    next_staging: &mut usize,
    tokens: &[usize; N],
    targets: &[usize; N],
    stream: &CudaStream,
) -> Result<(), DriverError> {
    let slot = &mut staging[*next_staging];
    if slot.pending {
        slot.copied.synchronize()?;
    }
    for i in 0..N {
        assert!(tokens[i] < VOCAB);
        assert!(targets[i] < VOCAB);
        slot.tokens[i] = tokens[i] as u32;
        slot.targets[i] = targets[i] as u32;
    }

    // SAFETY: the staging slot remains owned by the caller's workspace and is
    // not read, mutated, or dropped until `copied` has synchronized before its
    // next reuse. The event is recorded after both copies on this stream.
    unsafe {
        device_tokens
            .as_device_buffer_mut()
            .copy_from_pinned_host_async(stream, &slot.tokens)?;
        device_targets
            .as_device_buffer_mut()
            .copy_from_pinned_host_async(stream, &slot.targets)?;
    }
    slot.copied.record(stream)?;
    slot.pending = true;
    *next_staging ^= 1;
    Ok(())
}

/// Routing decisions one backward pass will read again; each block of a deep
/// model owns its own copy.
struct GpuRoutingActs<const N: usize, const E: usize, const K: usize> {
    probabilities: GpuTensor<f32, Rank2<N, E>>,
    selected_experts: GpuTensor<u32, Rank2<N, K>>,
    gate_weights: GpuTensor<f32, Rank2<N, K>>,
    slots: GpuTensor<u32, Rank2<N, K>>,
    assignment_counts: GpuTensor<u32, Rank1<E>>,
}

impl<const N: usize, const E: usize, const K: usize> GpuRoutingActs<N, E, K> {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        assert!(E > 0, "MoE must have at least one expert");
        assert!(K > 0 && K <= E, "MoE top-k must be in 1..=E");
        Ok(Self {
            probabilities: GpuTensor::zeros(stream)?,
            selected_experts: GpuTensor::zeros(stream)?,
            gate_weights: GpuTensor::zeros(stream)?,
            slots: GpuTensor::zeros(stream)?,
            assignment_counts: GpuTensor::zeros(stream)?,
        })
    }
}

/// One [`GpuBlock`]'s saved forward activations: everything its backward pass
/// reads again. A deep model allocates `L` of these.
struct GpuBlockActs<
    const N: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> {
    input: GpuTensor<f32, Rank2<N, D>>,
    attention_normalized: GpuTensor<f32, Rank2<N, D>>,
    attention: AttentionOperands<N, D>,
    attended: GpuTensor<f32, Rank2<N, D>>,
    attention_logsumexp: GpuTensor<f32, Rank2<N, H>>,
    ffn_input: GpuTensor<f32, Rank2<N, D>>,
    ffn_normalized: GpuTensor<f32, Rank2<N, D>>,
    routing: GpuRoutingActs<N, E, K>,
    experts: GpuExpertActs<E, C, D, FF>,
}

impl<
    const N: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> GpuBlockActs<N, D, H, FF, E, K, C>
{
    fn new(stream: &CudaStream, sequence_length: usize) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            input: GpuTensor::zeros(stream)?,
            attention_normalized: GpuTensor::zeros(stream)?,
            attention: AttentionOperands::new(stream, sequence_length, H)?,
            attended: GpuTensor::zeros(stream)?,
            attention_logsumexp: GpuTensor::zeros(stream)?,
            ffn_input: GpuTensor::zeros(stream)?,
            ffn_normalized: GpuTensor::zeros(stream)?,
            routing: GpuRoutingActs::new(stream)?,
            experts: GpuExpertActs::new(stream)?,
        })
    }
}

/// Temporaries no launch reads after the launch that consumed them; one
/// instance serves every block of a deep model.
struct GpuBlockScratch<
    const N: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> {
    qkv: GpuTensor<f32, Rank3<N, 3, D>>,
    projection_output: GpuTensor<f32, Rank2<N, D>>,
    attention_dot: GpuTensor<f32, Rank2<N, H>>,
    router_logits: GpuTensor<f32, Rank2<N, E>>,
    probability_sums: GpuTensor<f32, Rank1<E>>,
    gate_gradients: GpuTensor<f32, Rank2<N, K>>,
    dlogits: GpuTensor<f32, Rank2<N, E>>,
    router_dx: GpuTensor<f32, Rank2<N, D>>,
    /// One `[E,D]` router weight gradient per token partition, merged in
    /// ascending partition order by `router_backward_weight_merge`.
    router_dweight_partials: GpuTensor<f32, Rank3<{ dense_device::ROUTER_WGRAD_SPLITS }, E, D>>,
    experts: GpuExpertScratch<E, C, D, FF>,
    /// `[T, HD/2]` `(cos, sin)` couples, uploaded once. Shared by every block
    /// and every step: RoPE's angles depend only on `(position, pair)`.
    rope_table: DeviceBuffer<f32>,
    d_model_0: GpuTensor<f32, Rank2<N, D>>,
    d_model_1: GpuTensor<f32, Rank2<N, D>>,
    d_model_2: GpuTensor<f32, Rank2<N, D>>,
    d_model_3: GpuTensor<f32, Rank2<N, D>>,
    d_model_4: GpuTensor<f32, Rank2<N, D>>,
    norm_backward_inv: GpuTensor<f32, Rank1<N>>,
}

impl<
    const N: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> GpuBlockScratch<N, D, H, FF, E, K, C>
{
    fn new(stream: &CudaStream, sequence_length: usize) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            qkv: GpuTensor::zeros(stream)?,
            projection_output: GpuTensor::zeros(stream)?,
            attention_dot: GpuTensor::zeros(stream)?,
            router_logits: GpuTensor::zeros(stream)?,
            probability_sums: GpuTensor::zeros(stream)?,
            gate_gradients: GpuTensor::zeros(stream)?,
            dlogits: GpuTensor::zeros(stream)?,
            router_dx: GpuTensor::zeros(stream)?,
            router_dweight_partials: GpuTensor::zeros(stream)?,
            experts: GpuExpertScratch::new(stream)?,
            rope_table: DeviceBuffer::from_host(
                stream,
                &dense_device::rope_table(sequence_length, D / H),
            )?,
            d_model_0: GpuTensor::zeros(stream)?,
            d_model_1: GpuTensor::zeros(stream)?,
            d_model_2: GpuTensor::zeros(stream)?,
            d_model_3: GpuTensor::zeros(stream)?,
            d_model_4: GpuTensor::zeros(stream)?,
            norm_backward_inv: GpuTensor::zeros(stream)?,
        })
    }
}

/// Persistent device and pinned-host storage for one deep MoE model's
/// training steps.
///
/// Create this once and pass it to every forward/backward call. All operator
/// outputs are written into these allocations, so a steady-state step performs
/// no device allocation or synchronous device free. Per-block saved
/// activations dominate the footprint; every backward-only temporary is
/// shared across blocks.
///
/// The packed lm-head buffers are `NP` rows tall. Rows `N..NP` of
/// `head_input` are zeroed at allocation and never written afterwards
/// (`convert_f32_to_bf16_pairs` stops at the input length), and the same rows
/// of `logits` always hold zeros: the forward GEMM computes them from the
/// zero input rows and the classifier backward never touches them.
pub struct GpuMoeWorkspace<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
    const L: usize = 1,
> {
    tokens: GpuTensor<u32, Rank1<N>>,
    targets: GpuTensor<u32, Rank1<N>>,
    staging: [InputStaging<N>; 2],
    next_staging: usize,
    block_acts: Vec<GpuBlockActs<N, D, H, FF, E, K, C>>,
    block_scratch: GpuBlockScratch<N, D, H, FF, E, K, C>,
    final_input: GpuTensor<f32, Rank2<N, D>>,
    final_normalized: GpuTensor<f32, Rank2<N, D>>,
    head_input: DeviceBuffer<u32>,
    logits: DeviceBuffer<u32>,
    d_head_input: DeviceBuffer<u32>,
    head_input_tma: Bf16PairsTmaMap,
    head_input_mn_tma: Bf16PairsTmaMap,
    logits_tma: Bf16PairsTmaMap,
    logits_mn_tma: Bf16PairsTmaMap,
    linear_scratch: LinearScratch<N, D, FF>,
    flash_scratch: Option<FlashAttentionScratch<N, T, D, H>>,
    losses: GpuTensor<f32, Rank1<N>>,
    loss_sum: GpuTensor<f32, Rank1<1>>,
    loss: GpuTensor<f32, Rank1<1>>,
}

impl<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
    const L: usize,
> GpuMoeWorkspace<N, NP, T, VOCAB, VP, D, H, FF, E, K, C, L>
{
    pub fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        assert!(L > 0, "workspace needs at least one block");
        // MoE blocks keep only the attention projections as plain linears; the
        // expert FFN owns its own staging.
        let linear_scratch = LinearScratch::new(stream, &[(D, 3 * D), (D, D)])?;
        let head_input = DeviceBuffer::zeroed(stream, NP * D / 2)?;
        let logits = DeviceBuffer::zeroed(stream, NP * VP / 2)?;
        // SAFETY: the mapped buffers live in this workspace beside their maps
        // and are never reallocated.
        let head_input_tma =
            unsafe { create_bf16_pairs_tma_map(stream, &head_input, D, NP, TmaLayout::KMajor)? };
        let head_input_mn_tma =
            unsafe { create_bf16_pairs_tma_map(stream, &head_input, D, NP, TmaLayout::MnMajor)? };
        let logits_tma =
            unsafe { create_bf16_pairs_tma_map(stream, &logits, VP, NP, TmaLayout::KMajor)? };
        let logits_mn_tma =
            unsafe { create_bf16_pairs_tma_map(stream, &logits, VP, NP, TmaLayout::MnMajor)? };
        Ok(Self {
            tokens: GpuTensor::zeros(stream)?,
            targets: GpuTensor::zeros(stream)?,
            staging: [InputStaging::new(stream)?, InputStaging::new(stream)?],
            next_staging: 0,
            block_acts: (0..L)
                .map(|_| GpuBlockActs::new(stream, T))
                .collect::<Result<_, _>>()?,
            block_scratch: GpuBlockScratch::new(stream, T)?,
            final_input: GpuTensor::zeros(stream)?,
            final_normalized: GpuTensor::zeros(stream)?,
            head_input,
            logits,
            d_head_input: DeviceBuffer::zeroed(stream, NP * D / 2)?,
            head_input_tma,
            head_input_mn_tma,
            logits_tma,
            logits_mn_tma,
            linear_scratch,
            flash_scratch: if tcgen05_attention_eligible(T, D / H) {
                Some(FlashAttentionScratch::new(stream)?)
            } else {
                None
            },
            losses: GpuTensor::zeros(stream)?,
            loss_sum: GpuTensor::zeros(stream)?,
            loss: GpuTensor::zeros(stream)?,
        })
    }

    /// Packed-bf16 logits (dlogits after a backward pass). Parity-test
    /// accessor: binaries other than the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn logits_words(&self) -> &DeviceBuffer<u32> {
        &self.logits
    }

    pub fn loss(&self) -> &GpuTensor<f32, Rank1<1>> {
        &self.loss
    }

    /// Whether this workspace's shapes route the block linears through the
    /// bf16 tcgen05 path. Lets the aligned parity gate assert it is actually
    /// exercising that path rather than silently falling back to fp32.
    pub fn tcgen05_linears_active(&self) -> bool {
        self.linear_scratch.bf16.is_some()
    }

    /// Whether the expert GEMMs run on the bf16 tcgen05 path. Parity-gate
    /// accessor.
    #[allow(dead_code)]
    pub fn tcgen05_experts_active(&self) -> bool {
        self.block_scratch.experts.tcgen05_active()
    }

    /// Host readback of one packed-bf16 logits row, widened to f32.
    ///
    /// Sampling and debugging only: this synchronizes the stream after copying
    /// only the requested row.
    #[allow(dead_code)]
    pub fn logits_row(&self, row: usize, stream: &CudaStream) -> Result<Vec<f32>, DriverError> {
        assert!(row < NP);
        let stride = VP / 2;
        let byte_offset = row
            .checked_mul(stride)
            .and_then(|offset| offset.checked_mul(std::mem::size_of::<u32>()))
            .expect("logits row byte offset overflow");
        let source = self
            .logits
            .cu_deviceptr()
            .checked_add(byte_offset as u64)
            .expect("logits row device pointer overflow");
        let mut words = vec![0u32; stride];
        // SAFETY: `row < NP` and the workspace allocation of `NP * VP / 2`
        // words guarantee that `source` has `words.len()` readable elements.
        // The initialized host vector remains live until stream synchronization.
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                words.as_mut_ptr(),
                source,
                std::mem::size_of_val(words.as_slice()),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        Ok(words
            .iter()
            .flat_map(|&word| {
                [
                    f32::from_bits((word & 0xFFFF) << 16),
                    f32::from_bits((word >> 16) << 16),
                ]
            })
            .collect())
    }

    fn upload_inputs(
        &mut self,
        tokens: &[usize; N],
        targets: &[usize; N],
        stream: &CudaStream,
    ) -> Result<(), DriverError> {
        upload_token_inputs::<N, VOCAB>(
            &mut self.tokens,
            &mut self.targets,
            &mut self.staging,
            &mut self.next_staging,
            tokens,
            targets,
            stream,
        )
    }
}

/// Workspace for the single-block dense (non-MoE) reference model.
pub struct GpuDenseWorkspace<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
> {
    tokens: GpuTensor<u32, Rank1<N>>,
    targets: GpuTensor<u32, Rank1<N>>,
    staging: [InputStaging<N>; 2],
    next_staging: usize,
    attention_input: GpuTensor<f32, Rank2<N, D>>,
    attention_normalized: GpuTensor<f32, Rank2<N, D>>,
    qkv: GpuTensor<f32, Rank3<N, 3, D>>,
    attention: AttentionOperands<N, D>,
    attended: GpuTensor<f32, Rank2<N, D>>,
    attention_logsumexp: GpuTensor<f32, Rank2<N, H>>,
    attention_dot: GpuTensor<f32, Rank2<N, H>>,
    ffn_input: GpuTensor<f32, Rank2<N, D>>,
    ffn_normalized: GpuTensor<f32, Rank2<N, D>>,
    gate_up: GpuTensor<f32, Rank3<N, 2, FF>>,
    gate: GpuTensor<f32, Rank2<N, FF>>,
    up: GpuTensor<f32, Rank2<N, FF>>,
    activated: GpuTensor<f32, Rank2<N, FF>>,
    final_input: GpuTensor<f32, Rank2<N, D>>,
    final_normalized: GpuTensor<f32, Rank2<N, D>>,
    projection_output: GpuTensor<f32, Rank2<N, D>>,
    head_input: DeviceBuffer<u32>,
    logits: DeviceBuffer<u32>,
    d_head_input: DeviceBuffer<u32>,
    head_input_tma: Bf16PairsTmaMap,
    head_input_mn_tma: Bf16PairsTmaMap,
    logits_tma: Bf16PairsTmaMap,
    logits_mn_tma: Bf16PairsTmaMap,
    linear_scratch: LinearScratch<N, D, FF>,
    flash_scratch: Option<FlashAttentionScratch<N, T, D, H>>,
    /// See `GpuBlockScratch::rope_table`.
    rope_table: DeviceBuffer<f32>,
    norm_backward_inv: GpuTensor<f32, Rank1<N>>,
    losses: GpuTensor<f32, Rank1<N>>,
    loss_sum: GpuTensor<f32, Rank1<1>>,
    loss: GpuTensor<f32, Rank1<1>>,
    d_model_0: GpuTensor<f32, Rank2<N, D>>,
    d_model_1: GpuTensor<f32, Rank2<N, D>>,
    d_model_2: GpuTensor<f32, Rank2<N, D>>,
    d_model_3: GpuTensor<f32, Rank2<N, D>>,
    d_model_4: GpuTensor<f32, Rank2<N, D>>,
    d_ff_0: GpuTensor<f32, Rank2<N, FF>>,
    d_ff_1: GpuTensor<f32, Rank2<N, FF>>,
    d_ff_2: GpuTensor<f32, Rank2<N, FF>>,
}

impl<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
> GpuDenseWorkspace<N, NP, T, VOCAB, VP, D, H, FF>
{
    pub fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        let linear_scratch =
            LinearScratch::new(stream, &[(D, 3 * D), (D, D), (D, 2 * FF), (FF, D)])?;
        let head_input = DeviceBuffer::zeroed(stream, NP * D / 2)?;
        let logits = DeviceBuffer::zeroed(stream, NP * VP / 2)?;
        // SAFETY: the mapped buffers live in this workspace beside their maps
        // and are never reallocated.
        let head_input_tma =
            unsafe { create_bf16_pairs_tma_map(stream, &head_input, D, NP, TmaLayout::KMajor)? };
        let head_input_mn_tma =
            unsafe { create_bf16_pairs_tma_map(stream, &head_input, D, NP, TmaLayout::MnMajor)? };
        let logits_tma =
            unsafe { create_bf16_pairs_tma_map(stream, &logits, VP, NP, TmaLayout::KMajor)? };
        let logits_mn_tma =
            unsafe { create_bf16_pairs_tma_map(stream, &logits, VP, NP, TmaLayout::MnMajor)? };
        Ok(Self {
            tokens: GpuTensor::zeros(stream)?,
            targets: GpuTensor::zeros(stream)?,
            staging: [InputStaging::new(stream)?, InputStaging::new(stream)?],
            next_staging: 0,
            attention_input: GpuTensor::zeros(stream)?,
            attention_normalized: GpuTensor::zeros(stream)?,
            qkv: GpuTensor::zeros(stream)?,
            attention: AttentionOperands::new(stream, T, H)?,
            attended: GpuTensor::zeros(stream)?,
            attention_logsumexp: GpuTensor::zeros(stream)?,
            attention_dot: GpuTensor::zeros(stream)?,
            ffn_input: GpuTensor::zeros(stream)?,
            ffn_normalized: GpuTensor::zeros(stream)?,
            gate_up: GpuTensor::zeros(stream)?,
            gate: GpuTensor::zeros(stream)?,
            up: GpuTensor::zeros(stream)?,
            activated: GpuTensor::zeros(stream)?,
            final_input: GpuTensor::zeros(stream)?,
            final_normalized: GpuTensor::zeros(stream)?,
            projection_output: GpuTensor::zeros(stream)?,
            head_input,
            logits,
            d_head_input: DeviceBuffer::zeroed(stream, NP * D / 2)?,
            head_input_tma,
            head_input_mn_tma,
            logits_tma,
            logits_mn_tma,
            linear_scratch,
            flash_scratch: if tcgen05_attention_eligible(T, D / H) {
                Some(FlashAttentionScratch::new(stream)?)
            } else {
                None
            },
            rope_table: DeviceBuffer::from_host(stream, &dense_device::rope_table(T, D / H))?,
            norm_backward_inv: GpuTensor::zeros(stream)?,
            losses: GpuTensor::zeros(stream)?,
            loss_sum: GpuTensor::zeros(stream)?,
            loss: GpuTensor::zeros(stream)?,
            d_model_0: GpuTensor::zeros(stream)?,
            d_model_1: GpuTensor::zeros(stream)?,
            d_model_2: GpuTensor::zeros(stream)?,
            d_model_3: GpuTensor::zeros(stream)?,
            d_model_4: GpuTensor::zeros(stream)?,
            d_ff_0: GpuTensor::zeros(stream)?,
            d_ff_1: GpuTensor::zeros(stream)?,
            d_ff_2: GpuTensor::zeros(stream)?,
        })
    }

    /// Packed-bf16 logits (dlogits after a backward pass). Parity-test
    /// accessor: binaries other than the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn logits_words(&self) -> &DeviceBuffer<u32> {
        &self.logits
    }

    pub fn loss(&self) -> &GpuTensor<f32, Rank1<1>> {
        &self.loss
    }

    /// Whether this workspace's shapes route the block linears through the
    /// bf16 tcgen05 path. Lets the aligned parity gate assert it is actually
    /// exercising that path rather than silently falling back to fp32.
    pub fn tcgen05_linears_active(&self) -> bool {
        self.linear_scratch.bf16.is_some()
    }

    /// Host readback of one packed-bf16 logits row, widened to f32.
    ///
    /// Sampling and debugging only: this synchronizes the stream after copying
    /// only the requested row.
    pub fn logits_row(&self, row: usize, stream: &CudaStream) -> Result<Vec<f32>, DriverError> {
        assert!(row < NP);
        let stride = VP / 2;
        let byte_offset = row
            .checked_mul(stride)
            .and_then(|offset| offset.checked_mul(std::mem::size_of::<u32>()))
            .expect("logits row byte offset overflow");
        let source = self
            .logits
            .cu_deviceptr()
            .checked_add(byte_offset as u64)
            .expect("logits row device pointer overflow");
        let mut words = vec![0u32; stride];
        // SAFETY: `row < NP` and the workspace allocation of `NP * VP / 2`
        // words guarantee that `source` has `words.len()` readable elements.
        // The initialized host vector remains live until stream synchronization.
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                words.as_mut_ptr(),
                source,
                std::mem::size_of_val(words.as_slice()),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        Ok(words
            .iter()
            .flat_map(|&word| {
                [
                    f32::from_bits((word & 0xFFFF) << 16),
                    f32::from_bits((word >> 16) << 16),
                ]
            })
            .collect())
    }

    fn upload_inputs(
        &mut self,
        tokens: &[usize; N],
        targets: &[usize; N],
        stream: &CudaStream,
    ) -> Result<(), DriverError> {
        upload_token_inputs::<N, VOCAB>(
            &mut self.tokens,
            &mut self.targets,
            &mut self.staging,
            &mut self.next_staging,
            tokens,
            targets,
            stream,
        )
    }
}

impl<const D: usize, const FF: usize, const E: usize> GpuBlock<D, FF, E> {
    pub fn from_cpu<
        const N: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        const K: usize,
        const C: usize,
    >(
        stream: &CudaStream,
        block: &MoeBlock<N, T, D, H, HD, FF, E, K, C>,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            attention_norm: GpuRmsNorm::from_cpu(stream, &block.attention_norm)?,
            qkv_proj: GpuGroupedLinear::from_cpu(
                stream,
                [&block.q_proj, &block.k_proj, &block.v_proj],
            )?,
            o_proj: GpuLinear::from_cpu(stream, &block.o_proj)?,
            ffn_norm: GpuRmsNorm::from_cpu(stream, &block.ffn_norm)?,
            router: GpuTensor::from_host(stream, block.ffn.router.w.as_slice())?,
            d_router: GpuTensor::zeros(stream)?,
            experts: GpuExpertFfn::from_cpu(stream, &block.ffn.experts)?,
        })
    }

    fn sync_compute(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        self.qkv_proj.sync_compute(stream, kernels)?;
        self.o_proj.sync_compute(stream, kernels)?;
        self.experts.sync_compute(stream, kernels)
    }

    /// Runs one block forward from `acts.input`, saving backward state into
    /// `acts` and writing the block output into `output` (the next block's
    /// input buffer, or the workspace's final-norm input for the last block).
    #[allow(clippy::too_many_arguments)]
    fn forward_profiled<
        const N: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        const K: usize,
        const C: usize,
        P: KernelProfiler,
    >(
        &self,
        acts: &mut GpuBlockActs<N, D, H, FF, E, K, C>,
        output: &mut GpuTensor<f32, Rank2<N, D>>,
        scratch: &mut GpuBlockScratch<N, D, H, FF, E, K, C>,
        linear_scratch: &mut LinearScratch<N, D, FF>,
        mut flash_scratch: Option<&mut FlashAttentionScratch<N, T, D, H>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        flash_bf16: &Tcgen05Flash,
        dense: &dense_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        self.attention_norm.forward_into(
            &acts.input,
            &mut acts.attention_normalized,
            stream,
            dense,
            profiler,
            "forward.attention_norm",
        )?;
        self.qkv_proj.forward_into(
            &acts.attention_normalized,
            &mut scratch.qkv,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            linear_scratch,
            profiler,
            "forward.qkv_proj.gemm",
        )?;
        stage_attention_operands::<N, T, D, H, HD, P>(
            &scratch.qkv,
            &mut acts.attention,
            &mut scratch.d_model_0,
            &scratch.rope_table,
            stream,
            dense,
            flash,
            profiler,
        )?;
        flash_attention_forward_into::<N, T, D, H, HD, P>(
            &acts.attention,
            &mut acts.attended,
            &mut acts.attention_logsumexp,
            flash_scratch.as_deref_mut(),
            stream,
            flash,
            flash_bf16,
            profiler,
        )?;
        self.o_proj.forward_into(
            &acts.attended,
            &mut scratch.projection_output,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            linear_scratch,
            profiler,
            "forward.o_proj.gemm",
        )?;
        add_into(
            &acts.input,
            &scratch.projection_output,
            &mut acts.ffn_input,
            stream,
            tensor,
            profiler,
            "forward.attention_residual",
        )?;
        self.ffn_norm.forward_into(
            &acts.ffn_input,
            &mut acts.ffn_normalized,
            stream,
            dense,
            profiler,
            "forward.ffn_norm",
        )?;

        profiler.measure(stream, "forward.router.logits", || unsafe {
            dense.router_logits(
                stream,
                router_gemm_config(N, E),
                acts.ffn_normalized.as_device_buffer(),
                self.router.as_device_buffer(),
                D as u32,
                E as u32,
                scratch.router_logits.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "forward.router.topk", || unsafe {
            dense.router_softmax_topk(
                stream,
                LaunchConfig::for_num_elems(N as u32),
                scratch.router_logits.as_device_buffer(),
                E as u32,
                K as u32,
                acts.routing.probabilities.as_device_buffer_mut(),
                acts.routing.selected_experts.as_device_buffer_mut(),
                acts.routing.gate_weights.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "forward.router.assign", || unsafe {
            dense.moe_bin_assign_parallel(
                stream,
                moe_assign_config::<E>(),
                acts.routing.selected_experts.as_device_buffer(),
                N as u32,
                E as u32,
                K as u32,
                C as u32,
                acts.routing.slots.as_device_buffer_mut(),
                acts.routing.assignment_counts.as_device_buffer_mut(),
            )
        })?;
        let GpuBlockActs {
            ffn_normalized,
            routing,
            experts,
            ..
        } = &mut *acts;
        let bin_input = &mut experts.bin_input;
        // The scatter below overwrites every assigned bin row, so only the
        // unassigned capacity tail needs clearing — mirroring the backward's
        // dead-bin pass rather than pre-filling the whole `E·C·D` panel.
        profiler.measure(stream, "forward.router.zero_bins", || match bin_input {
            ExpertPanel::Packed(panel) => unsafe {
                dense.moe_zero_dead_bins_bf16(
                    stream,
                    moe_zero_bins_config(E * C),
                    routing.assignment_counts.as_device_buffer(),
                    D as u32,
                    C as u32,
                    &mut panel.words,
                )
            },
            ExpertPanel::Wide(values) => unsafe {
                dense.moe_zero_dead_bins(
                    stream,
                    moe_zero_bins_config(E * C),
                    routing.assignment_counts.as_device_buffer(),
                    D as u32,
                    C as u32,
                    values,
                )
            },
        })?;
        profiler.measure(stream, "forward.router.scatter", || match bin_input {
            // The packed panel's tcgen05 alignment makes D a multiple of
            // QUAD_LANES, so the quad walk always applies.
            ExpertPanel::Packed(panel) => unsafe {
                dense.moe_scatter_bf16_quad(
                    stream,
                    LaunchConfig::for_num_elems((N * K * D / QUAD_LANES) as u32),
                    ffn_normalized.as_device_buffer(),
                    routing.selected_experts.as_device_buffer(),
                    routing.slots.as_device_buffer(),
                    D as u32,
                    K as u32,
                    C as u32,
                    &mut panel.words,
                )
            },
            ExpertPanel::Wide(values) => unsafe {
                dense.moe_scatter(
                    stream,
                    LaunchConfig::for_num_elems((N * K * D) as u32),
                    ffn_normalized.as_device_buffer(),
                    routing.selected_experts.as_device_buffer(),
                    routing.slots.as_device_buffer(),
                    D as u32,
                    K as u32,
                    C as u32,
                    values,
                )
            },
        })?;
        self.experts.forward_profiled(
            &mut acts.experts,
            &mut scratch.experts,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            dense,
            profiler,
        )?;
        // The gather folds the ffn residual add in, writing the block output
        // directly instead of an intermediate a separate add pass re-reads.
        // SAFETY: routing indices and slots were produced for the same N/K/C shape.
        profiler.measure(stream, "forward.router.gather", || unsafe {
            if D.is_multiple_of(QUAD_LANES) {
                dense.moe_gather_combine_add_quad(
                    stream,
                    LaunchConfig::for_num_elems((N * D / QUAD_LANES) as u32),
                    acts.experts.bin_output.as_device_buffer(),
                    acts.routing.selected_experts.as_device_buffer(),
                    acts.routing.gate_weights.as_device_buffer(),
                    acts.routing.slots.as_device_buffer(),
                    acts.ffn_input.as_device_buffer(),
                    D as u32,
                    K as u32,
                    C as u32,
                    output.as_device_buffer_mut(),
                )
            } else {
                dense.moe_gather_combine_add(
                    stream,
                    LaunchConfig::for_num_elems((N * D) as u32),
                    acts.experts.bin_output.as_device_buffer(),
                    acts.routing.selected_experts.as_device_buffer(),
                    acts.routing.gate_weights.as_device_buffer(),
                    acts.routing.slots.as_device_buffer(),
                    acts.ffn_input.as_device_buffer(),
                    D as u32,
                    K as u32,
                    C as u32,
                    output.as_device_buffer_mut(),
                )
            }
        })
    }

    /// Runs one block backward.
    ///
    /// Contract: `scratch.d_model_1` holds the loss gradient with respect to
    /// this block's output on entry and holds the gradient with respect to
    /// this block's input on exit, so a reverse loop over blocks needs no
    /// copies between them.
    #[allow(clippy::too_many_arguments)]
    fn backward_profiled<
        const N: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        const K: usize,
        const C: usize,
        P: KernelProfiler,
    >(
        &mut self,
        aux_coefficient: f32,
        acts: &GpuBlockActs<N, D, H, FF, E, K, C>,
        scratch: &mut GpuBlockScratch<N, D, H, FF, E, K, C>,
        linear_scratch: &mut LinearScratch<N, D, FF>,
        mut flash_scratch: Option<&mut FlashAttentionScratch<N, T, D, H>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        flash_bf16: &Tcgen05Flash,
        dense: &dense_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        // `scatter_dy` below overwrites every assigned bin row, so only the
        // unassigned capacity tail needs clearing -- a full `E·C·D` pre-fill
        // would rewrite the whole buffer to no effect.
        let GpuBlockScratch {
            experts: expert_scratch,
            d_model_1,
            gate_gradients,
            ..
        } = &mut *scratch;
        let counts = acts.routing.assignment_counts.as_device_buffer();
        profiler.measure(
            stream,
            "backward.router.zero_dead_bins",
            || match &mut expert_scratch.d_bin_output {
                ExpertPanel::Packed(panel) => unsafe {
                    dense.moe_zero_dead_bins_bf16(
                        stream,
                        moe_zero_bins_config(E * C),
                        counts,
                        D as u32,
                        C as u32,
                        &mut panel.words,
                    )
                },
                ExpertPanel::Wide(values) => unsafe {
                    dense.moe_zero_dead_bins(
                        stream,
                        moe_zero_bins_config(E * C),
                        counts,
                        D as u32,
                        C as u32,
                        values,
                    )
                },
            },
        )?;
        let bin_output = acts.experts.bin_output.as_device_buffer();
        let selected = acts.routing.selected_experts.as_device_buffer();
        let gates = acts.routing.gate_weights.as_device_buffer();
        let slots = acts.routing.slots.as_device_buffer();
        profiler.measure(
            stream,
            "backward.router.scatter_dy",
            || match &mut expert_scratch.d_bin_output {
                ExpertPanel::Packed(panel) => unsafe {
                    dense.moe_scatter_dy_bf16(
                        stream,
                        moe_scatter_dy_config(N * K),
                        bin_output,
                        d_model_1.as_device_buffer(),
                        selected,
                        gates,
                        slots,
                        D as u32,
                        K as u32,
                        C as u32,
                        &mut panel.words,
                        gate_gradients.as_device_buffer_mut(),
                    )
                },
                ExpertPanel::Wide(values) => unsafe {
                    dense.moe_scatter_dy(
                        stream,
                        moe_scatter_dy_config(N * K),
                        bin_output,
                        d_model_1.as_device_buffer(),
                        selected,
                        gates,
                        slots,
                        D as u32,
                        K as u32,
                        C as u32,
                        values,
                        gate_gradients.as_device_buffer_mut(),
                    )
                },
            },
        )?;
        self.experts.backward_profiled(
            &acts.experts,
            &mut scratch.experts,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            dense,
            profiler,
        )?;
        profiler.measure(stream, "backward.router.softmax", || unsafe {
            dense.router_backward(
                stream,
                LaunchConfig::for_num_elems(N as u32),
                acts.routing.probabilities.as_device_buffer(),
                acts.routing.selected_experts.as_device_buffer(),
                acts.routing.gate_weights.as_device_buffer(),
                scratch.gate_gradients.as_device_buffer(),
                acts.routing.assignment_counts.as_device_buffer(),
                N as u32,
                E as u32,
                K as u32,
                aux_coefficient,
                scratch.dlogits.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "backward.router.input", || unsafe {
            dense.router_backward_input(
                stream,
                router_input_config::<N, D>(),
                scratch.dlogits.as_device_buffer(),
                self.router.as_device_buffer(),
                E as u32,
                scratch.router_dx.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "backward.router.weight", || unsafe {
            dense.router_backward_weight_split(
                stream,
                router_wgrad_split_config::<D>(),
                acts.ffn_normalized.as_device_buffer(),
                scratch.dlogits.as_device_buffer(),
                N as u32,
                E as u32,
                D as u32,
                scratch.router_dweight_partials.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "backward.router.weight_merge", || unsafe {
            dense.router_backward_weight_merge(
                stream,
                LaunchConfig::for_num_elems((D * E) as u32),
                scratch.router_dweight_partials.as_device_buffer(),
                E as u32,
                self.d_router.as_device_buffer_mut(),
            )
        })?;
        // The gather folds the router input-gradient add in, writing the
        // combined ffn input gradient directly instead of an intermediate a
        // separate add pass re-reads; it runs after `router_backward_input`
        // so `router_dx` is ready.
        // SAFETY: routing indices and slots were produced for the same N/K/C shape.
        profiler.measure(stream, "backward.router.gather_dx", || unsafe {
            if D.is_multiple_of(QUAD_LANES) {
                dense.moe_gather_dx_add_quad(
                    stream,
                    LaunchConfig::for_num_elems((N * D / QUAD_LANES) as u32),
                    scratch.experts.d_bin_input.as_device_buffer(),
                    acts.routing.selected_experts.as_device_buffer(),
                    acts.routing.slots.as_device_buffer(),
                    scratch.router_dx.as_device_buffer(),
                    D as u32,
                    K as u32,
                    C as u32,
                    scratch.d_model_4.as_device_buffer_mut(),
                )
            } else {
                dense.moe_gather_dx_add(
                    stream,
                    LaunchConfig::for_num_elems((N * D) as u32),
                    scratch.experts.d_bin_input.as_device_buffer(),
                    acts.routing.selected_experts.as_device_buffer(),
                    acts.routing.slots.as_device_buffer(),
                    scratch.router_dx.as_device_buffer(),
                    D as u32,
                    K as u32,
                    C as u32,
                    scratch.d_model_4.as_device_buffer_mut(),
                )
            }
        })?;
        self.ffn_norm.backward_into(
            &acts.ffn_input,
            &scratch.d_model_4,
            &mut scratch.d_model_0,
            &mut scratch.norm_backward_inv,
            stream,
            dense,
            profiler,
            ["backward.ffn_norm.input", "backward.ffn_norm.weight"],
        )?;
        add_into(
            &scratch.d_model_1,
            &scratch.d_model_0,
            &mut scratch.d_model_2,
            stream,
            tensor,
            profiler,
            "backward.ffn_residual",
        )?;
        self.o_proj.backward_into(
            &acts.attended,
            &scratch.d_model_2,
            &mut scratch.d_model_0,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            linear_scratch,
            profiler,
            ["backward.o_proj.weight_gemm", "backward.o_proj.input_gemm"],
        )?;
        flash_attention_backward_into::<N, T, D, H, HD, P>(
            &acts.attention,
            &acts.attended,
            &acts.attention_logsumexp,
            &mut scratch.attention_dot,
            &scratch.d_model_0,
            &mut scratch.d_model_1,
            &mut scratch.d_model_3,
            &mut scratch.d_model_4,
            flash_scratch.as_deref_mut(),
            stream,
            flash,
            flash_bf16,
            profiler,
        )?;
        // dQ, dK and dV go straight into the projection's row gradient,
        // un-rotated on the way: the two `backward.*_rope` passes this used to
        // need are the join's own arithmetic now.
        let dqkv = join_qkv_gradient::<N, T, D, H, HD, FF, P>(
            &self.qkv_proj,
            &scratch.d_model_1,
            &scratch.d_model_3,
            &scratch.d_model_4,
            &scratch.rope_table,
            &mut scratch.qkv,
            linear_scratch,
            stream,
            dense,
            profiler,
        )?;
        self.qkv_proj.backward_into(
            &acts.attention_normalized,
            dqkv,
            &mut scratch.d_model_3,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            linear_scratch,
            profiler,
            [
                "backward.qkv_proj.weight_gemm",
                "backward.qkv_proj.input_gemm",
            ],
        )?;
        self.attention_norm.backward_into(
            &acts.input,
            &scratch.d_model_3,
            &mut scratch.d_model_0,
            &mut scratch.norm_backward_inv,
            stream,
            dense,
            profiler,
            [
                "backward.attention_norm.input",
                "backward.attention_norm.weight",
            ],
        )?;
        add_into(
            &scratch.d_model_2,
            &scratch.d_model_0,
            &mut scratch.d_model_1,
            stream,
            tensor,
            profiler,
            "backward.attention_residual",
        )
    }

    fn zero_grad_profiled<P: KernelProfiler>(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        macro_rules! zero {
            ($name:literal, $gradient:expr) => {
                fill_zero($gradient, stream, tensor, profiler, $name)?;
            };
        }
        zero!("zero_grad.attention_norm", &mut self.attention_norm.dw);
        zero!("zero_grad.qkv_proj", &mut self.qkv_proj.dw);
        zero!("zero_grad.o_proj", &mut self.o_proj.dw);
        zero!("zero_grad.ffn_norm", &mut self.ffn_norm.dw);
        zero!("zero_grad.router", &mut self.d_router);
        zero!("zero_grad.experts.gate_up", &mut self.experts.d_gate_up);
        zero!("zero_grad.experts.down", &mut self.experts.d_down);
        Ok(())
    }
}

impl<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
    const L: usize,
> GpuDense<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>
{
    fn assert_shape() {
        assert!(N <= u32::MAX as usize);
        assert_eq!(N % T, 0);
        assert_eq!(D, H * HD);
        assert_eq!(NP, N.next_multiple_of(TC_M_TILE));
        assert!(VP >= VOCAB);
        assert_eq!(VP % TC_N_TILE, 0);
        assert_eq!(D % TC_K_PIPELINE, 0);
        assert!(E > 0 && K > 0 && K <= E && C > 0);
        assert!(L > 0, "model needs at least one block");
    }

    pub fn from_cpu(
        stream: &CudaStream,
        model: &MoeDense<N, T, VOCAB, D, H, HD, FF, E, K, C, L>,
    ) -> Result<Self, Box<dyn Error>> {
        Self::assert_shape();
        assert_eq!(model.blocks.len(), L);
        Ok(Self {
            embedding: GpuEmbedding::from_cpu(stream, &model.embedding)?,
            blocks: model
                .blocks
                .iter()
                .map(|block| GpuBlock::from_cpu(stream, block))
                .collect::<Result<_, _>>()?,
            final_norm: GpuRmsNorm::from_cpu(stream, &model.final_norm)?,
            lm_head: GpuBf16Head::from_cpu(stream, &model.lm_head)?,
        })
    }

    /// Deterministic scaled initialization equal to
    /// `Self::from_cpu(&MoeDense::new(seed, aux_coefficient))`, but holding at
    /// most one CPU block in host memory at a time so large configurations do
    /// not need the whole fp32 model host-side.
    pub fn initialized(
        stream: &CudaStream,
        seed: u64,
        aux_coefficient: f32,
    ) -> Result<Self, Box<dyn Error>> {
        Self::assert_shape();
        let hidden_scale = (D as f32).sqrt().recip();
        let mut next_seed = seed;
        let mut take_seed = move || {
            let current = next_seed;
            next_seed += 1;
            current
        };

        let embedding = {
            let cpu = nn::Embedding::<N, VOCAB, D>::new(
                tensor_cpu::CpuTensor::uniform(take_seed()).scale(hidden_scale),
            );
            GpuEmbedding::from_cpu(stream, &cpu)?
        };
        let blocks = (0..L)
            .map(|_| {
                let cpu =
                    MoeBlock::<N, T, D, H, HD, FF, E, K, C>::new(&mut take_seed, aux_coefficient);
                GpuBlock::from_cpu(stream, &cpu)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let final_norm = GpuRmsNorm::from_cpu(stream, &nn::RmsNorm::<N, D>::ones(1e-5))?;
        let lm_head = {
            let cpu = nn::Linear::<N, D, VOCAB>::new(
                tensor_cpu::CpuTensor::uniform(take_seed()).scale(hidden_scale),
            );
            GpuBf16Head::from_cpu(stream, &cpu)?
        };
        Ok(Self {
            embedding,
            blocks,
            final_norm,
            lm_head,
        })
    }

    pub(crate) fn sync_compute(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        for block in &mut self.blocks {
            block.sync_compute(stream, kernels)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: &[usize; N],
        targets: &[usize; N],
        aux_coefficient: f32,
        workspace: &mut GpuMoeWorkspace<N, NP, T, VOCAB, VP, D, H, FF, E, K, C, L>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        flash_bf16: &Tcgen05Flash,
        dense: &dense_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.forward_profiled(
            tokens,
            targets,
            aux_coefficient,
            workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments, unused_unsafe)]
    pub fn forward_profiled<P: KernelProfiler>(
        &self,
        tokens: &[usize; N],
        targets: &[usize; N],
        aux_coefficient: f32,
        workspace: &mut GpuMoeWorkspace<N, NP, T, VOCAB, VP, D, H, FF, E, K, C, L>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        flash_bf16: &Tcgen05Flash,
        dense: &dense_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        assert!(aux_coefficient.is_finite() && aux_coefficient >= 0.0);
        assert_eq!(self.blocks.len(), L);
        // SAFETY: workspace construction fixes every buffer to this model's
        // const-generic shape and each helper validates its launch dimensions.
        unsafe {
            workspace.upload_inputs(tokens, targets, stream)?;
            self.embedding.forward_into(
                &workspace.tokens,
                &mut workspace.block_acts[0].input,
                stream,
                dense,
                profiler,
                "forward.embedding",
            )?;
            for (index, block) in self.blocks.iter().enumerate() {
                let (current, rest) = workspace.block_acts[index..]
                    .split_first_mut()
                    .expect("one activation set per block");
                let output = match rest.first_mut() {
                    Some(next) => &mut next.input,
                    None => &mut workspace.final_input,
                };
                block.forward_profiled::<N, T, H, HD, K, C, P>(
                    current,
                    output,
                    &mut workspace.block_scratch,
                    &mut workspace.linear_scratch,
                    workspace.flash_scratch.as_mut(),
                    stream,
                    tensor,
                    gemm,
                    gemm_bf16,
                    flash,
                    flash_bf16,
                    dense,
                    profiler,
                )?;
            }
            self.final_norm.forward_into(
                &workspace.final_input,
                &mut workspace.final_normalized,
                stream,
                dense,
                profiler,
                "forward.final_norm",
            )?;
            // Rows N..NP of head_input were zeroed at allocation and the convert
            // stops at the fp32 input's length, so they stay zero.
            profiler.measure(stream, "forward.lm_head.quantize", || {
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * D / 2),
                    workspace.final_normalized.as_device_buffer(),
                    &mut workspace.head_input,
                )
            })?;
            self.lm_head.forward_into::<NP, P>(
                &workspace.head_input_tma,
                &mut workspace.logits,
                stream,
                gemm_bf16,
                profiler,
                "forward.lm_head.gemm",
            )?;
            cross_entropy_into::<N, VOCAB, VP, P>(
                &workspace.logits,
                &workspace.targets,
                &mut workspace.losses,
                &mut workspace.loss_sum,
                &mut workspace.loss,
                stream,
                tensor,
                dense,
                profiler,
            )?;
            for acts in &workspace.block_acts {
                fill_zero(
                    &mut workspace.block_scratch.probability_sums,
                    stream,
                    tensor,
                    profiler,
                    "forward.router.zero_probability_sums",
                )?;
                profiler.measure(stream, "forward.router.aux_probability_sums", || unsafe {
                    dense.moe_probability_sums(
                        stream,
                        LaunchConfig {
                            grid_dim: (E as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        acts.routing.probabilities.as_device_buffer(),
                        N as u32,
                        E as u32,
                        workspace
                            .block_scratch
                            .probability_sums
                            .as_device_buffer_mut(),
                    )
                })?;
                profiler.measure(stream, "forward.router.aux_loss", || unsafe {
                    dense.moe_aux_loss(
                        stream,
                        LaunchConfig::for_num_elems(1),
                        workspace.block_scratch.probability_sums.as_device_buffer(),
                        acts.routing.assignment_counts.as_device_buffer(),
                        N as u32,
                        E as u32,
                        K as u32,
                        aux_coefficient,
                        workspace.loss.as_device_buffer_mut(),
                    )
                })?;
            }
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &mut self,
        aux_coefficient: f32,
        workspace: &mut GpuMoeWorkspace<N, NP, T, VOCAB, VP, D, H, FF, E, K, C, L>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        flash_bf16: &Tcgen05Flash,
        dense: &dense_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.backward_profiled(
            aux_coefficient,
            workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments, unused_unsafe)]
    pub fn backward_profiled<P: KernelProfiler>(
        &mut self,
        aux_coefficient: f32,
        workspace: &mut GpuMoeWorkspace<N, NP, T, VOCAB, VP, D, H, FF, E, K, C, L>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        flash_bf16: &Tcgen05Flash,
        dense: &dense_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        assert!(aux_coefficient.is_finite() && aux_coefficient >= 0.0);
        assert_eq!(self.blocks.len(), L);
        // SAFETY: workspace construction fixes every buffer to this model's
        // const-generic shape and each helper validates its launch dimensions.
        unsafe {
            cross_entropy_backward_into::<N, VOCAB, VP, P>(
                &workspace.targets,
                &mut workspace.logits,
                stream,
                dense,
                profiler,
            )?;
            // Rows N..NP of head_input and logits hold zeros (forward computed
            // them from the zero-padded head input and the classifier backward
            // skips them), so the MN-major operands feed exact zeros into the
            // weight GEMM's padded reduction slice.
            self.lm_head.backward_weight::<NP, P>(
                &workspace.head_input_mn_tma,
                &workspace.logits_mn_tma,
                stream,
                gemm_bf16,
                profiler,
                "backward.lm_head.weight_gemm",
            )?;
            self.lm_head.backward_input::<NP, P>(
                &workspace.logits_tma,
                &mut workspace.d_head_input,
                stream,
                gemm_bf16,
                profiler,
                "backward.lm_head.input_gemm",
            )?;
            profiler.measure(stream, "backward.lm_head.dequantize", || {
                tensor.convert_bf16_pairs_to_f32(
                    stream,
                    elementwise_config::<Rank2<N, D>>(),
                    &workspace.d_head_input,
                    workspace.block_scratch.d_model_0.as_device_buffer_mut(),
                )
            })?;
            self.final_norm.backward_into(
                &workspace.final_input,
                &workspace.block_scratch.d_model_0,
                &mut workspace.block_scratch.d_model_1,
                &mut workspace.block_scratch.norm_backward_inv,
                stream,
                dense,
                profiler,
                ["backward.final_norm.input", "backward.final_norm.weight"],
            )?;
            for (block, acts) in self
                .blocks
                .iter_mut()
                .zip(workspace.block_acts.iter())
                .rev()
            {
                block.backward_profiled::<N, T, H, HD, K, C, P>(
                    aux_coefficient,
                    acts,
                    &mut workspace.block_scratch,
                    &mut workspace.linear_scratch,
                    workspace.flash_scratch.as_mut(),
                    stream,
                    tensor,
                    gemm,
                    gemm_bf16,
                    flash,
                    flash_bf16,
                    dense,
                    profiler,
                )?;
            }
            self.embedding.backward(
                &workspace.tokens,
                &workspace.block_scratch.d_model_1,
                stream,
                dense,
                profiler,
                "backward.embedding",
            )
        }
    }

    pub fn zero_grad(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.zero_grad_profiled(stream, tensor, &mut profiler)
    }

    pub fn zero_grad_profiled<P: KernelProfiler>(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        fill_zero(
            &mut self.embedding.dw,
            stream,
            tensor,
            profiler,
            "zero_grad.embedding",
        )?;
        for block in &mut self.blocks {
            block.zero_grad_profiled(stream, tensor, profiler)?;
        }
        fill_zero(
            &mut self.final_norm.dw,
            stream,
            tensor,
            profiler,
            "zero_grad.final_norm",
        )?;
        self.lm_head
            .zero_grad(stream, tensor, profiler, "zero_grad.lm_head")
    }
}

impl<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> GpuDenseDense<N, NP, T, VOCAB, VP, D, H, HD, FF>
{
    pub fn from_cpu(
        stream: &CudaStream,
        model: &Dense<N, T, VOCAB, D, H, HD, FF>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(N <= u32::MAX as usize);
        assert_eq!(N % T, 0);
        assert_eq!(D, H * HD);
        // tcgen05 head contract: padded tokens and vocabulary are tile
        // multiples, and D serves as both an output dimension (input-gradient
        // N, weight-gradient M) and a reduction width.
        assert_eq!(NP, N.next_multiple_of(TC_M_TILE));
        assert!(VP >= VOCAB);
        assert_eq!(VP % TC_N_TILE, 0);
        assert_eq!(D % TC_K_PIPELINE, 0);
        Ok(Self {
            embedding: GpuEmbedding::from_cpu(stream, &model.embedding)?,
            attention_norm: GpuRmsNorm::from_cpu(stream, &model.attention_norm)?,
            qkv_proj: GpuGroupedLinear::from_cpu(
                stream,
                [&model.q_proj, &model.k_proj, &model.v_proj],
            )?,
            o_proj: GpuLinear::from_cpu(stream, &model.o_proj)?,
            ffn_norm: GpuRmsNorm::from_cpu(stream, &model.ffn_norm)?,
            gate_up_proj: GpuGroupedLinear::from_cpu(stream, [&model.gate_proj, &model.up_proj])?,
            down_proj: GpuLinear::from_cpu(stream, &model.down_proj)?,
            final_norm: GpuRmsNorm::from_cpu(stream, &model.final_norm)?,
            lm_head: GpuBf16Head::from_cpu(stream, &model.lm_head)?,
        })
    }

    pub(crate) fn sync_linear_compute(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        self.qkv_proj.sync_compute(stream, kernels)?;
        self.o_proj.sync_compute(stream, kernels)?;
        self.gate_up_proj.sync_compute(stream, kernels)?;
        self.down_proj.sync_compute(stream, kernels)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: &[usize; N],
        targets: &[usize; N],
        workspace: &mut GpuDenseWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        flash_bf16: &Tcgen05Flash,
        dense: &dense_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.forward_profiled(
            tokens,
            targets,
            workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments, unused_unsafe)]
    pub fn forward_profiled<P: KernelProfiler>(
        &self,
        tokens: &[usize; N],
        targets: &[usize; N],
        workspace: &mut GpuDenseWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        flash_bf16: &Tcgen05Flash,
        dense: &dense_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        // SAFETY: workspace construction fixes every buffer to this model's
        // const-generic shape and each helper validates its launch dimensions.
        unsafe {
            workspace.upload_inputs(tokens, targets, stream)?;
            self.embedding.forward_into(
                &workspace.tokens,
                &mut workspace.attention_input,
                stream,
                dense,
                profiler,
                "forward.embedding",
            )?;
            self.attention_norm.forward_into(
                &workspace.attention_input,
                &mut workspace.attention_normalized,
                stream,
                dense,
                profiler,
                "forward.attention_norm",
            )?;
            self.qkv_proj.forward_into(
                &workspace.attention_normalized,
                &mut workspace.qkv,
                stream,
                tensor,
                gemm,
                gemm_bf16,
                &mut workspace.linear_scratch,
                profiler,
                "forward.qkv_proj.gemm",
            )?;
            stage_attention_operands::<N, T, D, H, HD, P>(
                &workspace.qkv,
                &mut workspace.attention,
                &mut workspace.d_model_0,
                &workspace.rope_table,
                stream,
                dense,
                flash,
                profiler,
            )?;
            flash_attention_forward_into::<N, T, D, H, HD, P>(
                &workspace.attention,
                &mut workspace.attended,
                &mut workspace.attention_logsumexp,
                workspace.flash_scratch.as_mut(),
                stream,
                flash,
                flash_bf16,
                profiler,
            )?;
            self.o_proj.forward_into(
                &workspace.attended,
                &mut workspace.projection_output,
                stream,
                tensor,
                gemm,
                gemm_bf16,
                &mut workspace.linear_scratch,
                profiler,
                "forward.o_proj.gemm",
            )?;
            add_into(
                &workspace.attention_input,
                &workspace.projection_output,
                &mut workspace.ffn_input,
                stream,
                tensor,
                profiler,
                "forward.attention_residual",
            )?;

            self.ffn_norm.forward_into(
                &workspace.ffn_input,
                &mut workspace.ffn_normalized,
                stream,
                dense,
                profiler,
                "forward.ffn_norm",
            )?;
            self.gate_up_proj.forward_into(
                &workspace.ffn_normalized,
                &mut workspace.gate_up,
                stream,
                tensor,
                gemm,
                gemm_bf16,
                &mut workspace.linear_scratch,
                profiler,
                "forward.gate_up_proj.gemm",
            )?;
            profiler.measure(stream, "forward.gate_up_proj.split", || {
                dense.split_group2(
                    stream,
                    LaunchConfig::for_num_elems((N * FF) as u32),
                    workspace.gate_up.as_device_buffer(),
                    FF as u32,
                    workspace.gate.as_device_buffer_mut(),
                    workspace.up.as_device_buffer_mut(),
                )
            })?;
            swiglu_into(
                &workspace.gate,
                &workspace.up,
                &mut workspace.activated,
                stream,
                dense,
                profiler,
                "forward.swiglu",
            )?;
            self.down_proj.forward_into(
                &workspace.activated,
                &mut workspace.projection_output,
                stream,
                tensor,
                gemm,
                gemm_bf16,
                &mut workspace.linear_scratch,
                profiler,
                "forward.down_proj.gemm",
            )?;
            add_into(
                &workspace.ffn_input,
                &workspace.projection_output,
                &mut workspace.final_input,
                stream,
                tensor,
                profiler,
                "forward.ffn_residual",
            )?;

            self.final_norm.forward_into(
                &workspace.final_input,
                &mut workspace.final_normalized,
                stream,
                dense,
                profiler,
                "forward.final_norm",
            )?;
            // Rows N..NP of head_input were zeroed at allocation and the convert
            // stops at the fp32 input's length, so they stay zero.
            profiler.measure(stream, "forward.lm_head.quantize", || {
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * D / 2),
                    workspace.final_normalized.as_device_buffer(),
                    &mut workspace.head_input,
                )
            })?;
            self.lm_head.forward_into::<NP, P>(
                &workspace.head_input_tma,
                &mut workspace.logits,
                stream,
                gemm_bf16,
                profiler,
                "forward.lm_head.gemm",
            )?;
            cross_entropy_into::<N, VOCAB, VP, P>(
                &workspace.logits,
                &workspace.targets,
                &mut workspace.losses,
                &mut workspace.loss_sum,
                &mut workspace.loss,
                stream,
                tensor,
                dense,
                profiler,
            )
        }
    }

    pub fn backward(
        &mut self,
        workspace: &mut GpuDenseWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        flash_bf16: &Tcgen05Flash,
        dense: &dense_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.backward_profiled(
            workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            flash_bf16,
            dense,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments, unused_unsafe)]
    pub fn backward_profiled<P: KernelProfiler>(
        &mut self,
        workspace: &mut GpuDenseWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        flash_bf16: &Tcgen05Flash,
        dense: &dense_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        // SAFETY: workspace construction fixes every buffer to this model's
        // const-generic shape and each helper validates its launch dimensions.
        unsafe {
            cross_entropy_backward_into::<N, VOCAB, VP, P>(
                &workspace.targets,
                &mut workspace.logits,
                stream,
                dense,
                profiler,
            )?;
            // Rows N..NP of head_input and logits hold zeros (forward computed
            // them from the zero-padded head input and the classifier backward
            // skips them), so the MN-major operands feed exact zeros into the
            // weight GEMM's padded reduction slice.
            self.lm_head.backward_weight::<NP, P>(
                &workspace.head_input_mn_tma,
                &workspace.logits_mn_tma,
                stream,
                gemm_bf16,
                profiler,
                "backward.lm_head.weight_gemm",
            )?;
            self.lm_head.backward_input::<NP, P>(
                &workspace.logits_tma,
                &mut workspace.d_head_input,
                stream,
                gemm_bf16,
                profiler,
                "backward.lm_head.input_gemm",
            )?;
            profiler.measure(stream, "backward.lm_head.dequantize", || {
                tensor.convert_bf16_pairs_to_f32(
                    stream,
                    elementwise_config::<Rank2<N, D>>(),
                    &workspace.d_head_input,
                    workspace.d_model_0.as_device_buffer_mut(),
                )
            })?;
            self.final_norm.backward_into(
                &workspace.final_input,
                &workspace.d_model_0,
                &mut workspace.d_model_1,
                &mut workspace.norm_backward_inv,
                stream,
                dense,
                profiler,
                ["backward.final_norm.input", "backward.final_norm.weight"],
            )?;

            self.down_proj.backward_into(
                &workspace.activated,
                &workspace.d_model_1,
                &mut workspace.d_ff_0,
                stream,
                tensor,
                gemm,
                gemm_bf16,
                &mut workspace.linear_scratch,
                profiler,
                [
                    "backward.down_proj.weight_gemm",
                    "backward.down_proj.input_gemm",
                ],
            )?;
            swiglu_backward_into(
                &workspace.gate,
                &workspace.up,
                &workspace.d_ff_0,
                &mut workspace.d_ff_1,
                &mut workspace.d_ff_2,
                stream,
                dense,
                profiler,
            )?;
            profiler.measure(stream, "backward.gate_up_proj.join", || unsafe {
                dense.join_group2(
                    stream,
                    LaunchConfig::for_num_elems((N * FF) as u32),
                    workspace.d_ff_1.as_device_buffer(),
                    workspace.d_ff_2.as_device_buffer(),
                    FF as u32,
                    workspace.gate_up.as_device_buffer_mut(),
                )
            })?;
            self.gate_up_proj.backward_into(
                &workspace.ffn_normalized,
                RowGradient::Wide(&workspace.gate_up),
                &mut workspace.d_model_3,
                stream,
                tensor,
                gemm,
                gemm_bf16,
                &mut workspace.linear_scratch,
                profiler,
                [
                    "backward.gate_up_proj.weight_gemm",
                    "backward.gate_up_proj.input_gemm",
                ],
            )?;
            self.ffn_norm.backward_into(
                &workspace.ffn_input,
                &workspace.d_model_3,
                &mut workspace.d_model_0,
                &mut workspace.norm_backward_inv,
                stream,
                dense,
                profiler,
                ["backward.ffn_norm.input", "backward.ffn_norm.weight"],
            )?;
            add_into(
                &workspace.d_model_1,
                &workspace.d_model_0,
                &mut workspace.d_model_2,
                stream,
                tensor,
                profiler,
                "backward.ffn_residual",
            )?;

            self.o_proj.backward_into(
                &workspace.attended,
                &workspace.d_model_2,
                &mut workspace.d_model_0,
                stream,
                tensor,
                gemm,
                gemm_bf16,
                &mut workspace.linear_scratch,
                profiler,
                ["backward.o_proj.weight_gemm", "backward.o_proj.input_gemm"],
            )?;
            flash_attention_backward_into::<N, T, D, H, HD, P>(
                &workspace.attention,
                &workspace.attended,
                &workspace.attention_logsumexp,
                &mut workspace.attention_dot,
                &workspace.d_model_0,
                &mut workspace.d_model_1,
                &mut workspace.d_model_3,
                &mut workspace.d_model_4,
                workspace.flash_scratch.as_mut(),
                stream,
                flash,
                flash_bf16,
                profiler,
            )?;
            // See the MoE block backward: the inverse rotation is the join's.
            let dqkv = join_qkv_gradient::<N, T, D, H, HD, FF, P>(
                &self.qkv_proj,
                &workspace.d_model_1,
                &workspace.d_model_3,
                &workspace.d_model_4,
                &workspace.rope_table,
                &mut workspace.qkv,
                &mut workspace.linear_scratch,
                stream,
                dense,
                profiler,
            )?;
            self.qkv_proj.backward_into(
                &workspace.attention_normalized,
                dqkv,
                &mut workspace.d_model_3,
                stream,
                tensor,
                gemm,
                gemm_bf16,
                &mut workspace.linear_scratch,
                profiler,
                [
                    "backward.qkv_proj.weight_gemm",
                    "backward.qkv_proj.input_gemm",
                ],
            )?;
            self.attention_norm.backward_into(
                &workspace.attention_input,
                &workspace.d_model_3,
                &mut workspace.d_model_0,
                &mut workspace.norm_backward_inv,
                stream,
                dense,
                profiler,
                [
                    "backward.attention_norm.input",
                    "backward.attention_norm.weight",
                ],
            )?;
            add_into(
                &workspace.d_model_2,
                &workspace.d_model_0,
                &mut workspace.d_model_1,
                stream,
                tensor,
                profiler,
                "backward.attention_residual",
            )?;
            self.embedding.backward(
                &workspace.tokens,
                &workspace.d_model_1,
                stream,
                dense,
                profiler,
                "backward.embedding",
            )
        }
    }

    pub fn zero_grad(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.zero_grad_profiled(stream, tensor, &mut profiler)
    }

    pub fn zero_grad_profiled<P: KernelProfiler>(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        macro_rules! zero {
            ($field:ident) => {
                fill_zero(
                    &mut self.$field.dw,
                    stream,
                    tensor,
                    profiler,
                    concat!("zero_grad.", stringify!($field)),
                )?;
            };
        }
        zero!(embedding);
        zero!(attention_norm);
        zero!(qkv_proj);
        zero!(o_proj);
        zero!(ffn_norm);
        zero!(gate_up_proj);
        zero!(down_proj);
        zero!(final_norm);
        self.lm_head
            .zero_grad(stream, tensor, profiler, "zero_grad.lm_head")?;
        Ok(())
    }
}

fn rope_into<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    x: &GpuTensor<f32, Rank2<N, D>>,
    y: &mut GpuTensor<f32, Rank2<N, D>>,
    table: &DeviceBuffer<f32>,
    stream: &CudaStream,
    kernels: &dense_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    // SAFETY: all buffers contain N * D elements, and the launch is one thread
    // per rotated pair.
    profiler.measure(stream, name, || unsafe {
        kernels.rope_forward(
            stream,
            LaunchConfig::for_num_elems((N * D / 2) as u32),
            x.as_device_buffer(),
            table,
            T as u32,
            H as u32,
            HD as u32,
            y.as_device_buffer_mut(),
        )
    })
}

/// Q, K and V out of the projection's fp32 `[N, 3D]` panel and into whatever
/// this shape's attention reads.
///
/// The staged path does it in one pass — split, rotate, scale, quantize and
/// relayout head-major — where the fp32 path still needs the split and two
/// rotation passes it always did. `rotated` is a scratch `[N, D]` the fp32
/// path rotates through and the staged path never touches.
#[allow(clippy::too_many_arguments)]
fn stage_attention_operands<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    qkv: &GpuTensor<f32, Rank3<N, 3, D>>,
    operands: &mut AttentionOperands<N, D>,
    rotated: &mut GpuTensor<f32, Rank2<N, D>>,
    table: &DeviceBuffer<f32>,
    stream: &CudaStream,
    dense: &dense_kernels::LoadedModule,
    flash: &flash_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    match operands {
        AttentionOperands::Staged { q, k, v } => {
            // Fold softmax_scale * log2(e) into Q so the kernel's softmax is
            // base-2 native; K/V quantize unscaled.
            let q_scale = std::f32::consts::LOG2_E / (HD as f32).sqrt();
            // SAFETY: the panel is [N, 3D] and the three staged buffers match
            // the N/T/H/HD attention layout.
            profiler.measure(stream, "forward.attention.stage_qkv", || unsafe {
                flash.stage_qkv_heads_bf16(
                    stream,
                    flash_device::stage_heads_config(N, H, HD),
                    qkv.as_device_buffer(),
                    table,
                    T as u32,
                    H as u32,
                    q_scale,
                    &mut q.words,
                    &mut k.words,
                    &mut v.words,
                )
            })
        }
        AttentionOperands::Wide { q, k, v } => {
            // SAFETY: qkv contains three contiguous N * D groups.
            profiler.measure(stream, "forward.qkv_proj.split", || unsafe {
                dense.split_group3(
                    stream,
                    LaunchConfig::for_num_elems((N * D) as u32),
                    qkv.as_device_buffer(),
                    D as u32,
                    q.as_device_buffer_mut(),
                    k.as_device_buffer_mut(),
                    v.as_device_buffer_mut(),
                )
            })?;
            rope_into::<N, T, D, H, HD, P>(
                q,
                rotated,
                table,
                stream,
                dense,
                profiler,
                "forward.q_rope",
            )?;
            std::mem::swap(q, rotated);
            rope_into::<N, T, D, H, HD, P>(
                k,
                rotated,
                table,
                stream,
                dense,
                profiler,
                "forward.k_rope",
            )?;
            std::mem::swap(k, rotated);
            Ok(())
        }
    }
}

/// Un-rotate dQ and dK and join them with dV into the qkv projection's row
/// gradient.
///
/// The two backward GEMMs read that panel as bf16, so on the tcgen05 path the
/// join writes their operand buffer directly and the quantize `backward_into`
/// used to run over `[N, 3D]` disappears — the join's own writes halve with
/// it. Shapes on the fp32 fallback still get the wide panel its GEMMs read.
#[allow(clippy::too_many_arguments)]
fn join_qkv_gradient<
    'a,
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    P: KernelProfiler,
>(
    qkv_proj: &GpuGroupedLinear<D, 3, D>,
    dq: &GpuTensor<f32, Rank2<N, D>>,
    dk: &GpuTensor<f32, Rank2<N, D>>,
    dv: &GpuTensor<f32, Rank2<N, D>>,
    table: &DeviceBuffer<f32>,
    wide: &'a mut GpuTensor<f32, Rank3<N, 3, D>>,
    linear_scratch: &mut LinearScratch<N, D, FF>,
    stream: &CudaStream,
    dense: &dense_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<RowGradient<'a, N, 3, D>, DriverError> {
    let pairs = LaunchConfig::for_num_elems((N * D / 2) as u32);
    // SAFETY: the three gradients are [N, D] and either destination holds
    // N * 3 * D elements of its own width.
    match qkv_proj.packed_row_gradient(linear_scratch) {
        Some(rows) => {
            profiler.measure(stream, "backward.qkv_proj.join", || unsafe {
                dense.join_group3_rope_bf16(
                    stream,
                    pairs,
                    dq.as_device_buffer(),
                    dk.as_device_buffer(),
                    dv.as_device_buffer(),
                    table,
                    T as u32,
                    H as u32,
                    HD as u32,
                    rows,
                )
            })?;
            Ok(RowGradient::Staged)
        }
        None => {
            profiler.measure(stream, "backward.qkv_proj.join", || unsafe {
                dense.join_group3_rope(
                    stream,
                    pairs,
                    dq.as_device_buffer(),
                    dk.as_device_buffer(),
                    dv.as_device_buffer(),
                    table,
                    T as u32,
                    H as u32,
                    HD as u32,
                    wide.as_device_buffer_mut(),
                )
            })?;
            Ok(RowGradient::Wide(wide))
        }
    }
}

/// Attention forward dispatch: tile-aligned shapes read the packed-bf16 head
/// panels the qkv staging pass wrote and run the persistent tcgen05 forward
/// (issue #35 phase 3); other shapes stay on the fp32 tiled kernel. Both paths
/// write the same fp32 `y`/natural-log LSE contract, so the (still fp32)
/// backward is oblivious.
#[allow(clippy::too_many_arguments)]
fn flash_attention_forward_into<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    operands: &AttentionOperands<N, D>,
    output: &mut GpuTensor<f32, Rank2<N, D>>,
    logsumexp: &mut GpuTensor<f32, Rank2<N, H>>,
    scratch: Option<&mut FlashAttentionScratch<N, T, D, H>>,
    stream: &CudaStream,
    kernels: &flash_kernels::LoadedModule,
    flash_bf16: &Tcgen05Flash,
    profiler: &mut P,
) -> Result<(), DriverError> {
    match operands {
        AttentionOperands::Staged { q, k, v } => {
            let scratch = scratch.expect("staged operands allocate the flash scratch beside them");
            // SAFETY: the panels and their maps match the N/T/H/HD layout.
            profiler.measure(stream, "forward.attention.flash", || unsafe {
                flash_bf16.forward(
                    stream,
                    flash_host::flash_forward_config(N / T, T, H, flash_bf16.sm_count()),
                    q.tma.as_ptr(),
                    k.tma.as_ptr(),
                    v.tma.as_ptr(),
                    T as u32,
                    H as u32,
                    (N / T) as u32,
                    output.as_device_buffer_mut(),
                    logsumexp.as_device_buffer_mut(),
                    &mut scratch.correction_counts,
                )
            })
        }
        AttentionOperands::Wide { q, k, v } if HD == flash_device::TILE_HD => {
            // SAFETY: tiled config is selected only for its specialized head width.
            profiler.measure(stream, "forward.attention.flash", || unsafe {
                kernels.flash_attention_forward_tiled(
                    stream,
                    flash_forward_config::<N, T, H, HD>(),
                    q.as_device_buffer(),
                    k.as_device_buffer(),
                    v.as_device_buffer(),
                    T as u32,
                    H as u32,
                    output.as_device_buffer_mut(),
                    logsumexp.as_device_buffer_mut(),
                )
            })
        }
        // Head widths neither flash generation specializes on fall back to
        // the per-row oracle kernels: correct for any power-of-two `HD` up to
        // `MAX_HEAD_DIM`, but serial over keys — a stopgap until the tcgen05
        // flash learns this head width.
        AttentionOperands::Wide { q, k, v } => {
            // SAFETY: per-row config covers N * H rows with HD lanes.
            profiler.measure(stream, "forward.attention.flash", || unsafe {
                kernels.flash_attention_forward(
                    stream,
                    per_row_flash_config::<N, H, HD>(),
                    q.as_device_buffer(),
                    k.as_device_buffer(),
                    v.as_device_buffer(),
                    T as u32,
                    H as u32,
                    HD as u32,
                    output.as_device_buffer_mut(),
                    logsumexp.as_device_buffer_mut(),
                )
            })
        }
    }
}

/// Attention backward dispatch (issue #35 phase 4): tile-aligned shapes stage
/// dY into the shared scratch, read the forward's per-block Q/K/V panels, and
/// run the tcgen05 query-parallel dQ and key-parallel dK/dV kernels; other
/// shapes stay on the fp32 tiled kernels. Both paths first run the fp32
/// `backward_dot` over the forward `y` — the tcgen05 kernels consume the same
/// `Σ dy·y` and the saved natural-log LSE as read-only device slices.
#[allow(clippy::too_many_arguments)]
fn flash_attention_backward_into<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    operands: &AttentionOperands<N, D>,
    output: &GpuTensor<f32, Rank2<N, D>>,
    logsumexp: &GpuTensor<f32, Rank2<N, H>>,
    softmax_dot: &mut GpuTensor<f32, Rank2<N, H>>,
    dy: &GpuTensor<f32, Rank2<N, D>>,
    dq: &mut GpuTensor<f32, Rank2<N, D>>,
    dk: &mut GpuTensor<f32, Rank2<N, D>>,
    dv: &mut GpuTensor<f32, Rank2<N, D>>,
    scratch: Option<&mut FlashAttentionScratch<N, T, D, H>>,
    stream: &CudaStream,
    kernels: &flash_kernels::LoadedModule,
    flash_bf16: &Tcgen05Flash,
    profiler: &mut P,
) -> Result<(), DriverError> {
    // SAFETY: dot config and buffers agree on N, H, and HD.
    profiler.measure(stream, "backward.attention.flash_dot", || unsafe {
        kernels.flash_attention_backward_dot(
            stream,
            flash_dot_config::<N, H, HD>(),
            dy.as_device_buffer(),
            output.as_device_buffer(),
            HD as u32,
            softmax_dot.as_device_buffer_mut(),
        )
    })?;
    match operands {
        AttentionOperands::Staged { q, k, v } => {
            let scratch = scratch.expect("staged operands allocate the flash scratch beside them");
            // Q, K and V were staged once by the forward and are this block's
            // saved activations; only dY, a backward temporary sharing one
            // buffer across blocks, is staged here.
            // SAFETY: staged buffers match the N/T/H/HD attention layout.
            profiler.measure(stream, "backward.attention.stage_bf16", || unsafe {
                kernels.stage_attention_heads_bf16(
                    stream,
                    flash_device::stage_heads_config(N, H, HD),
                    dy.as_device_buffer(),
                    T as u32,
                    H as u32,
                    1.0,
                    &mut scratch.dy.words,
                )
            })?;
            profiler.measure(stream, "backward.attention.flash_q", || unsafe {
                flash_bf16.backward_q(
                    stream,
                    flash_host::flash_backward_q_config(N / T, T, H, flash_bf16.sm_count()),
                    q.tma.as_ptr(),
                    k.tma.as_ptr(),
                    v.tma.as_ptr(),
                    scratch.dy.tma.as_ptr(),
                    logsumexp.as_device_buffer(),
                    softmax_dot.as_device_buffer(),
                    T as u32,
                    H as u32,
                    (N / T) as u32,
                    dq.as_device_buffer_mut(),
                )
            })?;
            profiler.measure(stream, "backward.attention.flash_kv", || unsafe {
                flash_bf16.backward_kv(
                    stream,
                    flash_host::flash_backward_kv_config(N / T, T, H, flash_bf16.sm_count()),
                    q.tma.as_ptr(),
                    k.tma.as_ptr(),
                    v.tma.as_ptr(),
                    scratch.dy.tma.as_ptr(),
                    logsumexp.as_device_buffer(),
                    softmax_dot.as_device_buffer(),
                    T as u32,
                    H as u32,
                    (N / T) as u32,
                    dk.as_device_buffer_mut(),
                    dv.as_device_buffer_mut(),
                )
            })
        }
        AttentionOperands::Wide { q, k, v } if HD == flash_device::TILE_HD => {
            // SAFETY: tiled config is selected only for its specialized head width.
            profiler.measure(stream, "backward.attention.flash_q", || unsafe {
                kernels.flash_attention_backward_q_tiled(
                    stream,
                    flash_backward_q_config::<N, T, H, HD>(),
                    q.as_device_buffer(),
                    k.as_device_buffer(),
                    v.as_device_buffer(),
                    dy.as_device_buffer(),
                    logsumexp.as_device_buffer(),
                    softmax_dot.as_device_buffer(),
                    T as u32,
                    H as u32,
                    dq.as_device_buffer_mut(),
                )
            })?;
            // SAFETY: tiled config is selected only for its specialized head width.
            profiler.measure(stream, "backward.attention.flash_kv", || unsafe {
                kernels.flash_attention_backward_kv_tiled(
                    stream,
                    flash_backward_kv_config::<N, T, H, HD>(),
                    q.as_device_buffer(),
                    k.as_device_buffer(),
                    v.as_device_buffer(),
                    dy.as_device_buffer(),
                    logsumexp.as_device_buffer(),
                    softmax_dot.as_device_buffer(),
                    T as u32,
                    H as u32,
                    dk.as_device_buffer_mut(),
                    dv.as_device_buffer_mut(),
                )
            })
        }
        // Per-row oracle fallback; see the forward dispatch for the contract.
        AttentionOperands::Wide { q, k, v } => {
            // SAFETY: per-row config covers N * H rows with HD lanes.
            profiler.measure(stream, "backward.attention.flash_q", || unsafe {
                kernels.flash_attention_backward_q(
                    stream,
                    per_row_flash_config::<N, H, HD>(),
                    q.as_device_buffer(),
                    k.as_device_buffer(),
                    v.as_device_buffer(),
                    output.as_device_buffer(),
                    dy.as_device_buffer(),
                    logsumexp.as_device_buffer(),
                    T as u32,
                    H as u32,
                    HD as u32,
                    dq.as_device_buffer_mut(),
                )
            })?;
            // SAFETY: per-row config covers N * H rows with HD lanes.
            profiler.measure(stream, "backward.attention.flash_kv", || unsafe {
                kernels.flash_attention_backward_kv(
                    stream,
                    per_row_flash_config::<N, H, HD>(),
                    q.as_device_buffer(),
                    k.as_device_buffer(),
                    v.as_device_buffer(),
                    output.as_device_buffer(),
                    dy.as_device_buffer(),
                    logsumexp.as_device_buffer(),
                    T as u32,
                    H as u32,
                    HD as u32,
                    dk.as_device_buffer_mut(),
                    dv.as_device_buffer_mut(),
                )
            })
        }
    }
}

fn swiglu_into<const N: usize, const FF: usize, P: KernelProfiler>(
    gate: &GpuTensor<f32, Rank2<N, FF>>,
    up: &GpuTensor<f32, Rank2<N, FF>>,
    output: &mut GpuTensor<f32, Rank2<N, FF>>,
    stream: &CudaStream,
    kernels: &dense_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    // SAFETY: all elementwise buffers contain N * FF elements, and the tile
    // arm is only taken at a shape `swiglu_tiles` accepted.
    profiler.measure(stream, name, || unsafe {
        match swiglu_tiles(N, FF) {
            Some(tiles) => kernels.swiglu_forward_tile(
                stream,
                tiles,
                gate.as_device_buffer(),
                up.as_device_buffer(),
                FF as u32,
                output.as_device_buffer_mut(),
            ),
            None => kernels.swiglu_forward(
                stream,
                LaunchConfig::for_num_elems((N * FF) as u32),
                gate.as_device_buffer(),
                up.as_device_buffer(),
                output.as_device_buffer_mut(),
            ),
        }
    })?;
    Ok(())
}

fn swiglu_backward_into<const N: usize, const FF: usize, P: KernelProfiler>(
    gate: &GpuTensor<f32, Rank2<N, FF>>,
    up: &GpuTensor<f32, Rank2<N, FF>>,
    dy: &GpuTensor<f32, Rank2<N, FF>>,
    dgate: &mut GpuTensor<f32, Rank2<N, FF>>,
    dup: &mut GpuTensor<f32, Rank2<N, FF>>,
    stream: &CudaStream,
    kernels: &dense_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    let config = LaunchConfig::for_num_elems((N * FF) as u32);
    let tiles = swiglu_tiles(N, FF);
    // SAFETY: all elementwise buffers contain N * FF elements, and the tile
    // arms are only taken at a shape `swiglu_tiles` accepted.
    profiler.measure(stream, "backward.swiglu.gate", || unsafe {
        match tiles {
            Some(tiles) => kernels.swiglu_backward_gate_tile(
                stream,
                tiles,
                gate.as_device_buffer(),
                up.as_device_buffer(),
                dy.as_device_buffer(),
                FF as u32,
                dgate.as_device_buffer_mut(),
            ),
            None => kernels.swiglu_backward_gate(
                stream,
                config,
                gate.as_device_buffer(),
                up.as_device_buffer(),
                dy.as_device_buffer(),
                dgate.as_device_buffer_mut(),
            ),
        }
    })?;
    // SAFETY: as above.
    profiler.measure(stream, "backward.swiglu.up", || unsafe {
        match tiles {
            Some(tiles) => kernels.swiglu_backward_up_tile(
                stream,
                tiles,
                gate.as_device_buffer(),
                dy.as_device_buffer(),
                FF as u32,
                dup.as_device_buffer_mut(),
            ),
            None => kernels.swiglu_backward_up(
                stream,
                config,
                gate.as_device_buffer(),
                dy.as_device_buffer(),
                dup.as_device_buffer_mut(),
            ),
        }
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cross_entropy_into<const N: usize, const VOCAB: usize, const VP: usize, P: KernelProfiler>(
    logits: &DeviceBuffer<u32>,
    targets: &GpuTensor<u32, Rank1<N>>,
    losses: &mut GpuTensor<f32, Rank1<N>>,
    loss_sum: &mut GpuTensor<f32, Rank1<1>>,
    loss: &mut GpuTensor<f32, Rank1<1>>,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    dense: &dense_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    // SAFETY: logits use N padded-VP rows and targets/losses use N entries.
    profiler.measure(stream, "forward.loss.fused_classifier", || unsafe {
        dense.fused_classifier_forward_bf16(
            stream,
            classifier_config::<N>(),
            logits,
            targets.as_device_buffer(),
            N as u32,
            VOCAB as u32,
            VP as u32,
            losses.as_device_buffer_mut(),
        )
    })?;
    sum_into(
        losses,
        loss_sum,
        stream,
        tensor,
        profiler,
        "forward.loss.reduction",
    )?;
    scale_into(
        loss_sum,
        1.0 / N as f32,
        loss,
        stream,
        tensor,
        profiler,
        "forward.loss.mean",
    )
}

fn cross_entropy_backward_into<
    const N: usize,
    const VOCAB: usize,
    const VP: usize,
    P: KernelProfiler,
>(
    targets: &GpuTensor<u32, Rank1<N>>,
    dlogits: &mut DeviceBuffer<u32>,
    stream: &CudaStream,
    kernels: &dense_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    // SAFETY: dlogits uses N padded-VP rows and targets contain N entries.
    profiler.measure(stream, "backward.loss.fused_classifier", || unsafe {
        kernels.fused_classifier_backward_in_place_bf16(
            stream,
            classifier_config::<N>(),
            targets.as_device_buffer(),
            1.0,
            N as u32,
            VOCAB as u32,
            VP as u32,
            dlogits,
        )
    })
}
