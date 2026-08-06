//! Reference CUDA kernels for the first Dense modules.
//!
//! These favor direct, auditable implementations over performance. They are
//! the GPU correctness baseline that later optimized kernels must match.
//!
//! With the pinned stock cuda-oxide backend, kernels are collected from the
//! selected binary target rather than a separately compiled library target.
//! Host binaries should include this file as a module (see `main.rs`) so this
//! remains the single source of kernel definitions while the selected target
//! receives an embedded CUDA artifact.

use cuda_device::{
    DisjointSlice, SharedArray,
    atomic::{AtomicOrdering, DeviceAtomicF32},
    cuda_module, kernel, thread,
};

use kittens::global::{GlobalRows, load_cols, load_rows, store_rows};
use kittens::reg::{BaseLdtm, ColVec, RegTile, RegVec};
use kittens::shared::F32;
use kittens::{lane, warp_id};

/// Threads in the row-parallel fused classifier kernels.
///
/// Each block owns one row and lanes stride over the vocabulary. Keeping this
/// fixed and power-of-two makes the online `(max, sum_exp)` reduction valid for
/// arbitrary vocabulary sizes.
pub const CLASSIFIER_THREADS: usize = 256;

/// Threads in the block-per-row RMSNorm factor reduction. Must remain a power
/// of two.
pub const NORM_THREADS: usize = 256;

/// Rows accumulated by one RMSNorm weight-gradient block.
///
/// Splitting a large batch across the grid's Y dimension exposes enough
/// parallelism to saturate the GPU. Each block performs one atomic add per
/// owned column, rather than one atomic per input element.
pub const NORM_WEIGHT_ROWS_PER_BLOCK: usize = 256;

/// Threads in one expert's deterministic MoE capacity-assignment block.
///
/// Each lane owns one contiguous range of token/rank pairs. A block-wide
/// prefix sum then gives every range its exact token-order starting slot.
pub const MOE_ASSIGN_THREADS: usize = 256;

/// Upper bound on routed experts, sizing the shared broadcast tiles.
pub const ROUTER_MAX_EXPERTS: usize = 8;

/// Router forward logits GEMM CTA output rows (token tile). Tiled so the router
/// weight is loaded once per token tile instead of re-read per token.
pub const ROUTER_GEMM_BM: usize = 32;
/// Router logits GEMM CTA output columns. `8` matches the routed expert width,
/// so the skinny expert dimension carries no tile padding.
pub const ROUTER_GEMM_BN: usize = 8;
/// Router logits GEMM reduction tile.
pub const ROUTER_GEMM_BK: usize = 16;
/// Threads in a router logits GEMM block: one lane per output element.
pub const ROUTER_GEMM_THREADS: usize = ROUTER_GEMM_BM * ROUTER_GEMM_BN;

/// Threads in one router input-backward block.
pub const ROUTER_INPUT_THREADS: usize = 256;
/// Token rows one router input-backward block sweeps. The block's slice of the
/// `[D,E]` weight is read once and reused across all of them, so weight traffic
/// falls by this factor.
pub const ROUTER_INPUT_TOKENS: usize = 64;
/// Model columns each router input-backward lane owns. The lane keeps their
/// `[E]` weight rows in registers for the whole token sweep.
pub const ROUTER_INPUT_COLUMNS: usize = 4;
/// Model columns one router input-backward block owns. Lane-major so every
/// `dx` store is a full coalesced sector.
pub const ROUTER_INPUT_BN: usize = ROUTER_INPUT_THREADS * ROUTER_INPUT_COLUMNS;

/// Threads in one router weight-gradient partition block.
pub const ROUTER_WGRAD_THREADS: usize = 256;
/// Model rows each router weight-gradient lane owns. The lane holds their
/// `[E]` accumulators in registers, so one coalesced `x` load feeds `E` FMAs.
pub const ROUTER_WGRAD_ROWS: usize = 4;
/// Model rows one router weight-gradient block owns.
pub const ROUTER_WGRAD_BM: usize = ROUTER_WGRAD_THREADS * ROUTER_WGRAD_ROWS;
/// Contiguous token partitions the router weight gradient is split into. Each
/// partition is summed by one block and the partitions are merged in ascending
/// order, which fixes the reduction order independently of block scheduling.
pub const ROUTER_WGRAD_SPLITS: usize = 256;

/// Sentinel written by deterministic MoE binning for a capacity-dropped pair.
pub const MOE_DROPPED_SLOT: u32 = u32::MAX;

/// Threads in one block-per-pair MoE backward scatter. Lanes stride the `D`
/// gradient row for a coalesced copy and a tree-reduced gate dot. Must remain a
/// power of two.
pub const MOE_SCATTER_DY_THREADS: usize = 256;

/// Threads in one block-per-expert routing-probability reduction. Lanes stride
/// the tokens before a tree reduction, so this must remain a power of two.
///
/// One block per expert is only `E` blocks, so the token loop's depth is all
/// the parallelism there is: `N` runs to 24576 and the trip count is a runtime
/// value NVVM will not unroll, which leaves each lane one load in flight and
/// the launch `N / threads` load latencies deep. A full block is 24 of them
/// rather than 96 (#99).
pub const MOE_PROBABILITY_SUMS_THREADS: usize = 1024;

/// Threads in one MoE dead-slot zeroing block.
pub const MOE_ZERO_BINS_THREADS: usize = 256;

/// Blocks per expert in the MoE dead-slot zeroing pass.
///
/// Each strides its expert's dead tail, so this bounds the dispatch rather than
/// describing the work: enough blocks to fill the device when the routing drops
/// a lot of tokens, and when it drops none they each load one count and exit.
pub const MOE_ZERO_BINS_BLOCKS: usize = 256;

/// Rows one warp owns in the tile RMSNorm forward.
///
/// `BaseLdtm` hands a warp its rows in 16-row blocks, so this is the tile's
/// `M` and the unit the grid is cut in. 16 and not 32: what this kernel wants
/// is columns per lane, and spending the register budget on rows instead
/// measures 1.026x against 16's 1.246x.
pub const NORM_TILE_ROWS: usize = 16;

/// Columns a tile-RMSNorm warp holds in registers at once.
///
/// The row statistic is carried across chunks, so this trades registers for
/// the number of times the row is walked, and it is the only knob that buys a
/// lane more loads in flight — the one thing this kernel was short of. `dim`
/// must be a multiple of it.
///
/// 64 is the peak and the edge: 96 falls to 0.387x, which is the `[16, 96]`
/// band no longer fitting the register file rather than the rolled-walk depot
/// that used to collapse everything past 32 (ferro-kittens#180 fixed that, and
/// fixing it is what moved this shape from 0.280x to 1.246x).
pub const NORM_TILE_CHUNK: usize = 64;

/// Warps in one tile-RMSNorm block. Each owns its own [`NORM_TILE_ROWS`] rows
/// and never talks to the others, so this only sets how many bands a CTA
/// carries; 2 and 4 measure the same and 8 gives up the win.
pub const NORM_TILE_WARPS: usize = 4;

/// Threads in one tile-RMSNorm block.
pub const NORM_TILE_THREADS: usize = 32 * NORM_TILE_WARPS;

/// Rows one tile-RMSNorm block owns — the grid's row quantum.
pub const NORM_TILE_BLOCK_ROWS: usize = NORM_TILE_ROWS * NORM_TILE_WARPS;

/// Rows one warp owns in the tile SwiGLU kernels.
///
/// SwiGLU has no reduction in it, so this is only how a warp's elements are
/// spread over rows: the whole shape question here is how many accesses a lane
/// has in flight, which is [`SWIGLU_TILE_CHUNK`]'s.
pub const SWIGLU_TILE_ROWS: usize = 16;

/// Columns a tile-SwiGLU warp holds in registers at once.
pub const SWIGLU_TILE_CHUNK: usize = 32;

/// Warps in one tile-SwiGLU block.
pub const SWIGLU_TILE_WARPS: usize = 4;

/// Threads in one tile-SwiGLU block.
pub const SWIGLU_TILE_THREADS: usize = 32 * SWIGLU_TILE_WARPS;

/// Rows one tile-SwiGLU block owns — the grid's row quantum.
pub const SWIGLU_TILE_BLOCK_ROWS: usize = SWIGLU_TILE_ROWS * SWIGLU_TILE_WARPS;

/// `f32` lanes in one 16-byte vector memory access.
///
/// Rows are walked through `*const u128` / `*mut u128` rather than an
/// `align(16)` `[f32; 4]` newtype: the codegen scalarizes aggregate loads back
/// into four `ld.global.b32`, while a `u128` stays one `ld/st.global.v2.b64`.
/// A row base `row * dim` is 16-byte aligned whenever `dim % QUAD_LANES == 0`,
/// which is the guard every vectorized row walk below checks.
pub const QUAD_LANES: usize = 4;

/// The RoPE cos/sin table for one sequence: `[sequence_length, head_dim / 2]`
/// entries of `(cos, sin)`, laid out so a rotated pair reads one adjacent
/// `f32` couple.
///
/// The rotation kernels used to recompute `powf`, `sin` and `cos` per element,
/// which is what they were waiting on rather than memory (#70). The angles are
/// a function of `(position, pair)` only — `sequence_length * head_dim` floats,
/// 1 MiB at the training shape — so they are built once and read from L2.
///
/// Built on the *host*, with the same expression `nn::Rope` uses, so one set of
/// angles serves the CPU reference and every device consumer instead of
/// comparing libdevice's `sinf` against the host's.
pub fn rope_table(sequence_length: usize, head_dim: usize) -> Vec<f32> {
    assert_eq!(head_dim % 2, 0, "RoPE head dimension must be even");
    let mut table = Vec::with_capacity(sequence_length * head_dim);
    for position in 0..sequence_length {
        for pair in 0..head_dim / 2 {
            let frequency = 10_000.0f32.powf(-((2 * pair) as f32) / head_dim as f32);
            let (sin, cos) = (position as f32 * frequency).sin_cos();
            table.push(cos);
            table.push(sin);
        }
    }
    table
}

/// One warp's slice of a row band, [`NORM_TILE_CHUNK`] columns wide.
type NormChunk = RegTile<NORM_TILE_ROWS, NORM_TILE_CHUNK, BaseLdtm>;

/// One `f32` per row of a [`NormChunk`], replicated across each quad — where a
/// row statistic lives between the passes that produce and consume it.
type NormRows = RegVec<NORM_TILE_ROWS, BaseLdtm>;

/// The chunk's slice of the `[dim]` weight, one `f32` per column this thread
/// owns rather than one per (row, column) pair.
type NormColumns = ColVec<NORM_TILE_CHUNK, BaseLdtm>;

/// One warp's slice of a SwiGLU row band.
type SwigluChunk = RegTile<SWIGLU_TILE_ROWS, SWIGLU_TILE_CHUNK, BaseLdtm>;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Unpacks one 16-byte vector load into its `f32` lanes. The shifts are
    /// register moves after codegen, not shift instructions.
    #[inline(always)]
    fn quad_lanes(bits: u128) -> [f32; QUAD_LANES] {
        [
            f32::from_bits(bits as u32),
            f32::from_bits((bits >> 32) as u32),
            f32::from_bits((bits >> 64) as u32),
            f32::from_bits((bits >> 96) as u32),
        ]
    }

    /// Packs `f32` lanes back into one 16-byte vector store.
    #[inline(always)]
    fn quad_bits(lanes: [f32; QUAD_LANES]) -> u128 {
        (lanes[0].to_bits() as u128)
            | ((lanes[1].to_bits() as u128) << 32)
            | ((lanes[2].to_bits() as u128) << 64)
            | ((lanes[3].to_bits() as u128) << 96)
    }

    #[inline(always)]
    fn bf16_bits_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    #[inline(always)]
    fn f32_to_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let round = 0x7fffu32 + ((bits >> 16) & 1);
        (bits.wrapping_add(round) >> 16) as u16
    }

    /// Two adjacent f32s as the one packed word `convert_f32_to_bf16_pairs`
    /// would have written for them: low half first, both round to nearest even.
    #[inline(always)]
    fn bf16_pair(low: f32, high: f32) -> u32 {
        f32_to_bf16_bits(low) as u32 | ((f32_to_bf16_bits(high) as u32) << 16)
    }

    /// The logical element `index` of a packed-bf16 buffer: low half is even.
    #[inline(always)]
    fn bf16_at(words: &[u32], index: usize) -> f32 {
        let word = words[index / 2];
        bf16_bits_to_f32((if index % 2 == 0 { word } else { word >> 16 }) as u16)
    }

    /// Both halves of one packed word, low first.
    #[inline(always)]
    fn bf16_halves(word: u32) -> [f32; 2] {
        [
            bf16_bits_to_f32(word as u16),
            bf16_bits_to_f32((word >> 16) as u16),
        ]
    }

    #[kernel]
    pub fn rms_norm_forward(
        x: &[f32],
        weight: &[f32],
        eps: f32,
        dim: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let d = dim as usize;
        if d == 0 || i >= x.len() || i >= y.len() || d > weight.len() {
            return;
        }
        let row = i / d;
        let base = row * d;
        if base + d > x.len() {
            return;
        }
        let mut sum_sq = 0.0f32;
        for col in 0..d {
            let value = x[base + col];
            sum_sq += value * value;
        }
        let inv = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        if let Some(slot) = y.get_mut(index) {
            let col = i % d;
            *slot = x[i] * inv * weight[col];
        }
    }

    #[kernel]
    pub fn rms_norm_backward_x(
        x: &[f32],
        weight: &[f32],
        dy: &[f32],
        eps: f32,
        dim: u32,
        mut dx: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let d = dim as usize;
        if d == 0 || i >= x.len() || i >= dy.len() || i >= dx.len() || d > weight.len() {
            return;
        }
        let row = i / d;
        let base = row * d;
        if base + d > x.len() || base + d > dy.len() {
            return;
        }
        let mut sum_sq = 0.0f32;
        let mut dot = 0.0f32;
        for col in 0..d {
            let value = x[base + col];
            sum_sq += value * value;
            dot += dy[base + col] * weight[col] * x[base + col];
        }
        let inv = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        let correction = inv * inv * inv * dot / dim as f32;
        if let Some(slot) = dx.get_mut(index) {
            let col = i % d;
            *slot = dy[i] * weight[col] * inv - x[i] * correction;
        }
    }

    /// Block-per-row RMSNorm forward.
    ///
    /// Unlike [`rms_norm_forward`], which is retained as the direct oracle,
    /// this computes the row reduction once and has lanes write a strided
    /// slice of the output.
    #[kernel]
    pub fn rms_norm_forward_fast(
        x: &[f32],
        weight: &[f32],
        eps: f32,
        dim: u32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut PARTIALS: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != NORM_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        let d = dim as usize;
        let base = row * d;
        if d == 0 || base + d > x.len() || base + d > y.len() || d > weight.len() {
            return;
        }

        let mut sum_sq = 0.0f32;
        let mut col = tid;
        while col < d {
            let value = x[base + col];
            sum_sq += value * value;
            col += NORM_THREADS;
        }
        unsafe {
            PARTIALS[tid] = sum_sq;
        }
        thread::sync_threads();

        let mut stride = NORM_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIALS[tid] += PARTIALS[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                PARTIALS[0] = 1.0 / (PARTIALS[0] / dim as f32 + eps).sqrt();
            }
        }
        thread::sync_threads();

        let inv = unsafe { PARTIALS[0] };
        col = tid;
        while col < d {
            // SAFETY: each lane owns distinct columns of this block's row.
            unsafe {
                *y.get_unchecked_mut(base + col) = x[base + col] * inv * weight[col];
            }
            col += NORM_THREADS;
        }
    }

    /// [`rms_norm_forward_fast`] over a packed-bf16 activation stream.
    ///
    /// A lane owns whole words, so it carries two values per step; the sum of
    /// squares, the scale and the weight product all stay in fp32 registers
    /// and only the store rounds. There is no tile arm for this dtype: the
    /// tile kernel's advantage over the block-per-row one was loads in flight
    /// (see [`rms_norm_forward_tile`]), and packing the stream doubles the
    /// elements each of these lanes already carries.
    #[kernel]
    pub fn rms_norm_forward_fast_bf16(
        x: &[u32],
        weight: &[f32],
        eps: f32,
        dim: u32,
        mut y: DisjointSlice<u32>,
    ) {
        static mut PARTIALS: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != NORM_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        let d = dim as usize;
        if d == 0 || !d.is_multiple_of(2) {
            return;
        }
        let words = d / 2;
        let base = row * words;
        if base + words > x.len() || base + words > y.len() || d > weight.len() {
            return;
        }

        let mut sum_sq = 0.0f32;
        let mut word = tid;
        while word < words {
            let [low, high] = bf16_halves(x[base + word]);
            sum_sq += low * low + high * high;
            word += NORM_THREADS;
        }
        unsafe {
            PARTIALS[tid] = sum_sq;
        }
        thread::sync_threads();

        let mut stride = NORM_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIALS[tid] += PARTIALS[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                PARTIALS[0] = 1.0 / (PARTIALS[0] / dim as f32 + eps).sqrt();
            }
        }
        thread::sync_threads();

        let inv = unsafe { PARTIALS[0] };
        word = tid;
        while word < words {
            let [low, high] = bf16_halves(x[base + word]);
            // SAFETY: each lane owns distinct words of this block's row.
            unsafe {
                *y.get_unchecked_mut(base + word) = bf16_pair(
                    low * inv * weight[2 * word],
                    high * inv * weight[2 * word + 1],
                );
            }
            word += NORM_THREADS;
        }
    }

    /// Block-per-row RMSNorm input backward, also producing the row inverse
    /// factors consumed by the weight-gradient kernel.
    ///
    /// [`rms_norm_backward_x`] recomputes both reductions once per output
    /// element. This variant computes them once per row and fuses the otherwise
    /// separate inverse-factor pass.
    #[kernel]
    pub fn rms_norm_backward_x_fast(
        x: &[f32],
        weight: &[f32],
        dy: &[f32],
        eps: f32,
        dim: u32,
        mut dx: DisjointSlice<f32>,
        mut inv: DisjointSlice<f32>,
    ) {
        static mut SUM_SQ: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;
        static mut DOT: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != NORM_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        let d = dim as usize;
        let base = row * d;
        if d == 0
            || base + d > x.len()
            || base + d > dy.len()
            || base + d > dx.len()
            || d > weight.len()
            || row >= inv.len()
        {
            return;
        }

        let mut sum_sq = 0.0f32;
        let mut dot = 0.0f32;
        let mut col = tid;
        while col < d {
            let value = x[base + col];
            sum_sq += value * value;
            dot += dy[base + col] * weight[col] * value;
            col += NORM_THREADS;
        }
        unsafe {
            SUM_SQ[tid] = sum_sq;
            DOT[tid] = dot;
        }
        thread::sync_threads();

        let mut stride = NORM_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    SUM_SQ[tid] += SUM_SQ[tid + stride];
                    DOT[tid] += DOT[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                let row_inv = 1.0 / (SUM_SQ[0] / dim as f32 + eps).sqrt();
                SUM_SQ[0] = row_inv;
                DOT[0] = row_inv * row_inv * row_inv * DOT[0] / dim as f32;
                *inv.get_unchecked_mut(row) = row_inv;
            }
        }
        thread::sync_threads();

        let row_inv = unsafe { SUM_SQ[0] };
        let correction = unsafe { DOT[0] };
        col = tid;
        while col < d {
            // SAFETY: each lane owns distinct columns of this block's row.
            unsafe {
                *dx.get_unchecked_mut(base + col) =
                    dy[base + col] * weight[col] * row_inv - x[base + col] * correction;
            }
            col += NORM_THREADS;
        }
    }

    /// [`rms_norm_backward_x_fast`] reading its saved input packed.
    ///
    /// Only `x` changes dtype: the incoming and outgoing gradients are
    /// backward temporaries and stay fp32, and both reductions still
    /// accumulate in fp32 registers.
    #[kernel]
    pub fn rms_norm_backward_x_fast_bf16(
        x: &[u32],
        weight: &[f32],
        dy: &[f32],
        eps: f32,
        dim: u32,
        mut dx: DisjointSlice<f32>,
        mut inv: DisjointSlice<f32>,
    ) {
        static mut SUM_SQ: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;
        static mut DOT: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != NORM_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        let d = dim as usize;
        let base = row * d;
        if d == 0
            || !d.is_multiple_of(2)
            || base + d > x.len() * 2
            || base + d > dy.len()
            || base + d > dx.len()
            || d > weight.len()
            || row >= inv.len()
        {
            return;
        }

        let mut sum_sq = 0.0f32;
        let mut dot = 0.0f32;
        let mut col = tid;
        while col < d {
            let value = bf16_at(x, base + col);
            sum_sq += value * value;
            dot += dy[base + col] * weight[col] * value;
            col += NORM_THREADS;
        }
        unsafe {
            SUM_SQ[tid] = sum_sq;
            DOT[tid] = dot;
        }
        thread::sync_threads();

        let mut stride = NORM_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    SUM_SQ[tid] += SUM_SQ[tid + stride];
                    DOT[tid] += DOT[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                let row_inv = 1.0 / (SUM_SQ[0] / dim as f32 + eps).sqrt();
                SUM_SQ[0] = row_inv;
                DOT[0] = row_inv * row_inv * row_inv * DOT[0] / dim as f32;
                *inv.get_unchecked_mut(row) = row_inv;
            }
        }
        thread::sync_threads();

        let row_inv = unsafe { SUM_SQ[0] };
        let correction = unsafe { DOT[0] };
        col = tid;
        while col < d {
            // SAFETY: each lane owns distinct columns of this block's row.
            unsafe {
                *dx.get_unchecked_mut(base + col) =
                    dy[base + col] * weight[col] * row_inv - bf16_at(x, base + col) * correction;
            }
            col += NORM_THREADS;
        }
    }

    #[kernel]
    pub fn rms_norm_backward_weight(
        x: &[f32],
        dy: &[f32],
        eps: f32,
        rows: u32,
        dim: u32,
        mut dweight: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let col = index.get();
        if col >= dim as usize {
            return;
        }
        let d = dim as usize;
        let mut grad = 0.0f32;
        for row in 0..rows as usize {
            let base = row * d;
            let mut sum_sq = 0.0f32;
            for feature in 0..d {
                let value = x[base + feature];
                sum_sq += value * value;
            }
            let inv = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
            grad += dy[base + col] * x[base + col] * inv;
        }
        if let Some(slot) = dweight.get_mut(index) {
            *slot += grad;
        }
    }

    /// Per-row `1 / sqrt(mean(x^2) + eps)` factors, one block per row.
    ///
    /// Feeds [`rms_norm_backward_weight_fast`], which would otherwise
    /// recompute every row's norm once per column.
    #[kernel]
    pub fn rms_norm_row_inv(x: &[f32], eps: f32, dim: u32, mut inv: DisjointSlice<f32>) {
        static mut PARTIALS: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != NORM_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        let d = dim as usize;
        let base = row * d;
        if base + d > x.len() {
            return;
        }

        let mut partial = 0.0f32;
        let mut col = tid;
        while col < d {
            let value = x[base + col];
            partial += value * value;
            col += NORM_THREADS;
        }
        unsafe {
            PARTIALS[tid] = partial;
        }
        thread::sync_threads();

        let mut stride = NORM_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIALS[tid] += PARTIALS[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            // SAFETY: this block exclusively owns `row`.
            unsafe {
                *inv.get_unchecked_mut(row) = 1.0 / (PARTIALS[0] / dim as f32 + eps).sqrt();
            }
        }
    }

    /// Tiled RMSNorm weight gradient from precomputed row factors.
    ///
    /// A block owns a column tile and a bounded row chunk. The Y grid exposes
    /// parallelism across large batches, and each thread atomically contributes
    /// one chunk sum to its column. [`rms_norm_backward_weight`] stays as the
    /// naive parity oracle.
    #[kernel]
    pub unsafe fn rms_norm_backward_weight_fast(
        x: &[f32],
        dy: &[f32],
        inv: &[f32],
        rows: u32,
        dim: u32,
        mut dweight: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != NORM_THREADS {
            return;
        }
        let d = dim as usize;
        let col = thread::blockIdx_x() as usize * NORM_THREADS + tid;
        if col >= d || col >= dweight.len() {
            return;
        }
        let row_start = thread::blockIdx_y() as usize * NORM_WEIGHT_ROWS_PER_BLOCK;
        let row_end = (row_start + NORM_WEIGHT_ROWS_PER_BLOCK).min(rows as usize);
        let mut grad = 0.0f32;
        for row in row_start..row_end {
            grad += dy[row * d + col] * x[row * d + col] * inv[row];
        }

        // SAFETY: `col` was bounds-checked and every access to this location
        // in this kernel is atomic. Stream ordering covers the preceding
        // zero/accumulation state and subsequent optimizer read.
        let slot = unsafe { DeviceAtomicF32::from_ptr(dweight.as_mut_ptr().add(col)) };
        slot.fetch_add(grad, AtomicOrdering::Relaxed);
    }

    /// [`rms_norm_backward_weight_fast`] reading its saved input packed.
    ///
    /// # Safety
    ///
    /// Same contract as [`rms_norm_backward_weight_fast`], and `dim` must be
    /// even so a row starts on a word boundary.
    #[kernel]
    pub unsafe fn rms_norm_backward_weight_fast_bf16(
        x: &[u32],
        dy: &[f32],
        inv: &[f32],
        rows: u32,
        dim: u32,
        mut dweight: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != NORM_THREADS {
            return;
        }
        let d = dim as usize;
        let col = thread::blockIdx_x() as usize * NORM_THREADS + tid;
        if col >= d || col >= dweight.len() {
            return;
        }
        let row_start = thread::blockIdx_y() as usize * NORM_WEIGHT_ROWS_PER_BLOCK;
        let row_end = (row_start + NORM_WEIGHT_ROWS_PER_BLOCK).min(rows as usize);
        let mut grad = 0.0f32;
        for row in row_start..row_end {
            grad += dy[row * d + col] * bf16_at(x, row * d + col) * inv[row];
        }

        // SAFETY: `col` was bounds-checked and every access to this location
        // in this kernel is atomic. Stream ordering covers the preceding
        // zero/accumulation state and subsequent optimizer read.
        let slot = unsafe { DeviceAtomicF32::from_ptr(dweight.as_mut_ptr().add(col)) };
        slot.fetch_add(grad, AtomicOrdering::Relaxed);
    }

    /// Warp-per-band RMSNorm forward on kittens register tiles.
    ///
    /// [`rms_norm_forward_fast`] gives a row to a 256-thread block and pays a
    /// shared-memory tree and eight barriers for its one statistic. Here a warp
    /// owns [`NORM_TILE_ROWS`] rows at once, so the same statistic is
    /// `RegTile::row_sum` — two shuffles, no shared memory and no barrier —
    /// and the row is walked [`NORM_TILE_CHUNK`] columns at a time through
    /// `load_rows`, which pairs adjacent columns into one access.
    ///
    /// The reduction is not what buys the 1.246x, and #70 was right that it
    /// never was: this kernel trades 6.3M threads for 98k, and it wins anyway
    /// because 64 columns per lane put more loads in flight than the 12
    /// elements per thread the block-per-row kernel carries. At 4167 GB/s
    /// against the 5641 the *backward* reaches over the same buffers on the
    /// same card, work per thread was the whole of what the forward was short
    /// of; collecting it lands at 5184 GB/s, and the backward is where that
    /// road ends, which is why it has no tile version.
    ///
    /// # Safety
    ///
    /// Launch with [`NORM_TILE_THREADS`] threads and a grid covering exactly
    /// `rows / NORM_TILE_BLOCK_ROWS` blocks: `rows` must be a multiple of
    /// [`NORM_TILE_BLOCK_ROWS`] and `dim` of [`NORM_TILE_CHUNK`], both checked
    /// by the launcher rather than the kernel, which never bounds-checks.
    #[kernel]
    pub unsafe fn rms_norm_forward_tile(
        x: &[f32],
        weight: &[f32],
        eps: f32,
        dim: u32,
        mut y: DisjointSlice<f32>,
    ) {
        unsafe {
            let lane = lane();
            let row = NORM_TILE_BLOCK_ROWS as u32 * thread::blockIdx_x()
                + NORM_TILE_ROWS as u32 * warp_id();
            let d = dim as usize;
            let source = GlobalRows::<F32>::from_raw(x.as_ptr() as *mut u8, d);
            let destination = GlobalRows::<F32>::from_slice(&mut y, d);
            // A one-row cursor: the weight is a per-column operand, and
            // `load_cols` reads it as one. Reading it through `load_rows`
            // instead would hold every value once per row the thread owns and
            // put the chunk width on a spill cliff (ferro-kittens#172).
            let parameters = GlobalRows::<F32>::from_raw(weight.as_ptr() as *mut u8, 0);

            let mut total = NormRows::splat(0.0);
            let mut column = 0u32;
            while column < dim {
                let v: NormChunk = load_rows(source, row, column, lane);
                total.add_assign(v.mul(v).row_sum());
                column += NORM_TILE_CHUNK as u32;
            }
            let inv = total.scale(1.0 / dim as f32).shift(eps).rsqrt();

            column = 0;
            while column < dim {
                let v: NormChunk = load_rows(source, row, column, lane);
                let w: NormColumns = load_cols(parameters, 0, column, lane);
                store_rows(destination, row, column, lane, v.mul_row(inv).mul_col(w));
                column += NORM_TILE_CHUNK as u32;
            }
        }
    }

    #[kernel]
    pub fn swiglu_forward(gate: &[f32], up: &[f32], mut y: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = y.get_mut(index) {
            let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
            *slot = gate[i] * sigmoid * up[i];
        }
    }

    /// [`swiglu_forward`] storing packed-bf16 pairs, one word per thread.
    ///
    /// The expert `activated` panel is only ever read as a bf16 tcgen05
    /// operand, so it is rounded once here instead of stored wide and
    /// quantized again by each of the two GEMMs that read it (#59).
    #[kernel]
    pub fn swiglu_forward_bf16(gate: &[f32], up: &[f32], mut y: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let pair = index.get();
        if 2 * pair + 1 >= gate.len() || 2 * pair + 1 >= up.len() {
            return;
        }
        if let Some(slot) = y.get_mut(index) {
            let mut packed = 0u32;
            for half in 0..2 {
                let i = 2 * pair + half;
                let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
                packed |= (f32_to_bf16_bits(gate[i] * sigmoid * up[i]) as u32) << (16 * half);
            }
            *slot = packed;
        }
    }

    #[kernel]
    pub fn swiglu_backward_gate(
        gate: &[f32],
        up: &[f32],
        dy: &[f32],
        mut dgate: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(gate_slot) = dgate.get_mut(index) {
            let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
            let dsilu = sigmoid * (1.0 + gate[i] * (1.0 - sigmoid));
            *gate_slot = dy[i] * up[i] * dsilu;
        }
    }

    #[kernel]
    pub fn swiglu_backward_up(gate: &[f32], dy: &[f32], mut dup: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(up_slot) = dup.get_mut(index) {
            let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
            *up_slot = dy[i] * gate[i] * sigmoid;
        }
    }

    /// [`swiglu_forward`] reading gate and up out of one interleaved
    /// `[rows, 2, ff]` panel — the layout the fused gate/up GEMM writes — so
    /// no split pass ever copies the panel into separate gate/up buffers.
    #[kernel]
    pub fn swiglu_forward_interleaved(gate_up: &[f32], ff: u32, mut y: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        let ff = ff as usize;
        if ff == 0 {
            return;
        }
        let row = i / ff;
        let column = i % ff;
        let base = row * 2 * ff + column;
        if base + ff >= gate_up.len() {
            return;
        }
        if let Some(slot) = y.get_mut(index) {
            let gate = gate_up[base];
            let sigmoid = 1.0 / (1.0 + (-gate).exp());
            *slot = gate * sigmoid * gate_up[base + ff];
        }
    }

    /// [`swiglu_forward_interleaved`] storing packed-bf16 pairs: one 16-byte
    /// gate load, one 16-byte up load, and one 8-byte packed store per
    /// thread. `ff` must be a multiple of [`QUAD_LANES`], which the tcgen05
    /// alignment gate on packed panels already guarantees.
    #[kernel]
    pub unsafe fn swiglu_forward_interleaved_bf16(
        gate_up: &[f32],
        ff: u32,
        mut y: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        let ff = ff as usize;
        if ff == 0 || !ff.is_multiple_of(QUAD_LANES) {
            return;
        }
        let row_quads = ff / QUAD_LANES;
        let row = i / row_quads;
        let quad = i % row_quads;
        let base = row * 2 * ff + QUAD_LANES * quad;
        if base + ff + QUAD_LANES > gate_up.len() || 2 * i + 1 >= y.len() {
            return;
        }
        // SAFETY: `base` and `base + ff` are multiples of `QUAD_LANES`, so
        // both 16-byte loads are aligned; bounds were checked above, and this
        // thread exclusively owns output words `2i` and `2i + 1`.
        let gates = quad_lanes(unsafe { *(gate_up.as_ptr().add(base) as *const u128) });
        let ups = quad_lanes(unsafe { *(gate_up.as_ptr().add(base + ff) as *const u128) });
        let mut packed = 0u64;
        for lane in 0..QUAD_LANES {
            let gate = gates[lane];
            let sigmoid = 1.0 / (1.0 + (-gate).exp());
            packed |= (f32_to_bf16_bits(gate * sigmoid * ups[lane]) as u64) << (16 * lane);
        }
        unsafe {
            *(y.as_mut_ptr() as *mut u64).add(i) = packed;
        }
    }

    /// Fused [`swiglu_backward_gate`] + [`swiglu_backward_up`]: reads the
    /// interleaved `[rows, 2, ff]` gate/up panel once and writes both halves
    /// of the interleaved `[rows, 2, ff]` gradient, so the two separate
    /// gradient buffers and the join pass that merged them never exist.
    #[kernel]
    pub unsafe fn swiglu_backward_interleaved(
        gate_up: &[f32],
        dy: &[f32],
        ff: u32,
        mut d_gate_up: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let ff = ff as usize;
        if ff == 0 {
            return;
        }
        let row = i / ff;
        let column = i % ff;
        let base = row * 2 * ff + column;
        if i >= dy.len() || base + ff >= gate_up.len() || base + ff >= d_gate_up.len() {
            return;
        }
        let gate = gate_up[base];
        let up = gate_up[base + ff];
        let grad = dy[i];
        let sigmoid = 1.0 / (1.0 + (-gate).exp());
        let dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid));
        // SAFETY: this thread exclusively owns both halves of column `i`.
        unsafe {
            *d_gate_up.get_unchecked_mut(base) = grad * up * dsilu;
            *d_gate_up.get_unchecked_mut(base + ff) = grad * gate * sigmoid;
        }
    }

    /// [`swiglu_backward_interleaved`] storing packed-bf16 pairs (every
    /// reader of the gate/up gradient panel is a tcgen05 operand, #59).
    /// Three 16-byte loads and two 8-byte packed stores per thread; `ff`
    /// must be a multiple of [`QUAD_LANES`].
    #[kernel]
    pub unsafe fn swiglu_backward_interleaved_bf16(
        gate_up: &[f32],
        dy: &[f32],
        ff: u32,
        mut d_gate_up: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        let ff = ff as usize;
        if ff == 0 || !ff.is_multiple_of(QUAD_LANES) {
            return;
        }
        let row_quads = ff / QUAD_LANES;
        let row = i / row_quads;
        let quad = i % row_quads;
        let base = row * 2 * ff + QUAD_LANES * quad;
        let dy_base = row * ff + QUAD_LANES * quad;
        // The packed row holds `ff` words: this thread's two gate words start
        // at `gate_word` and its two up words `ff / 2` words later.
        let gate_word = row * ff + 2 * quad;
        if base + ff + QUAD_LANES > gate_up.len()
            || dy_base + QUAD_LANES > dy.len()
            || gate_word + ff / 2 + 1 >= d_gate_up.len()
        {
            return;
        }
        // SAFETY: `base`, `base + ff`, and `dy_base` are multiples of
        // `QUAD_LANES` so the 16-byte loads are aligned; `gate_word` and
        // `gate_word + ff / 2` are even so the 8-byte stores are aligned;
        // bounds were checked above and this thread owns both word pairs.
        let gates = quad_lanes(unsafe { *(gate_up.as_ptr().add(base) as *const u128) });
        let ups = quad_lanes(unsafe { *(gate_up.as_ptr().add(base + ff) as *const u128) });
        let grads = quad_lanes(unsafe { *(dy.as_ptr().add(dy_base) as *const u128) });
        let mut dgate = 0u64;
        let mut dup = 0u64;
        for lane in 0..QUAD_LANES {
            let gate = gates[lane];
            let grad = grads[lane];
            let sigmoid = 1.0 / (1.0 + (-gate).exp());
            let dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid));
            dgate |= (f32_to_bf16_bits(grad * ups[lane] * dsilu) as u64) << (16 * lane);
            dup |= (f32_to_bf16_bits(grad * gate * sigmoid) as u64) << (16 * lane);
        }
        unsafe {
            *(d_gate_up.as_mut_ptr().add(gate_word) as *mut u64) = dgate;
            *(d_gate_up.as_mut_ptr().add(gate_word + ff / 2) as *mut u64) = dup;
        }
    }

    #[kernel]
    pub fn split_group2(
        input: &[f32],
        width: u32,
        mut first: DisjointSlice<f32>,
        mut second: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let width = width as usize;
        let row = i / width;
        let column = i % width;
        let base = row * 2 * width + column;
        if let Some(slot) = first.get_mut(thread::index_1d()) {
            *slot = input[base];
        }
        if let Some(slot) = second.get_mut(thread::index_1d()) {
            *slot = input[base + width];
        }
    }

    #[kernel]
    pub unsafe fn join_group2(
        first: &[f32],
        second: &[f32],
        width: u32,
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        // Launches round up to whole blocks; excess threads must not write.
        if 2 * i >= output.len() {
            return;
        }
        let width = width as usize;
        let row = i / width;
        let column = i % width;
        let base = row * 2 * width + column;
        unsafe {
            *output.get_unchecked_mut(base) = first[i];
            *output.get_unchecked_mut(base + width) = second[i];
        }
    }

    /// [`join_group2`] storing packed-bf16 pairs, one word per thread per
    /// group. `width` counts f32 columns per group and must be even; the two
    /// groups land `width / 2` words apart inside each interleaved row (#59).
    #[kernel]
    pub unsafe fn join_group2_bf16(
        first: &[f32],
        second: &[f32],
        width: u32,
        mut output: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        let width = width as usize;
        if width == 0 || !width.is_multiple_of(2) {
            return;
        }
        let half = width / 2;
        let row = i / half;
        let column = i % half;
        let source = row * width + 2 * column;
        if source + 1 >= first.len() || source + 1 >= second.len() {
            return;
        }
        let base = row * width + column;
        if base + half >= output.len() {
            return;
        }
        let low = f32_to_bf16_bits(first[source]) as u32
            | ((f32_to_bf16_bits(first[source + 1]) as u32) << 16);
        let high = f32_to_bf16_bits(second[source]) as u32
            | ((f32_to_bf16_bits(second[source + 1]) as u32) << 16);
        // SAFETY: both indices were bounds-checked and one thread owns each.
        unsafe {
            *output.get_unchecked_mut(base) = low;
            *output.get_unchecked_mut(base + half) = high;
        }
    }

    #[kernel]
    pub fn split_group3(
        input: &[f32],
        width: u32,
        mut first: DisjointSlice<f32>,
        mut second: DisjointSlice<f32>,
        mut third: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let width = width as usize;
        let row = i / width;
        let column = i % width;
        let base = row * 3 * width + column;
        if let Some(slot) = first.get_mut(thread::index_1d()) {
            *slot = input[base];
        }
        if let Some(slot) = second.get_mut(thread::index_1d()) {
            *slot = input[base + width];
        }
        if let Some(slot) = third.get_mut(thread::index_1d()) {
            *slot = input[base + 2 * width];
        }
    }

    #[kernel]
    pub unsafe fn join_group3(
        first: &[f32],
        second: &[f32],
        third: &[f32],
        width: u32,
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        // Launches round up to whole blocks; excess threads must not write.
        if 3 * i >= output.len() {
            return;
        }
        let width = width as usize;
        let row = i / width;
        let column = i % width;
        let base = row * 3 * width + column;
        unsafe {
            *output.get_unchecked_mut(base) = first[i];
            *output.get_unchecked_mut(base + width) = second[i];
            *output.get_unchecked_mut(base + 2 * width) = third[i];
        }
    }

    /// [`join_group3`] with RoPE's transposed rotation folded into the first
    /// two groups.
    ///
    /// Attention backward hands out gradients for the *rotated* Q and K, which
    /// the qkv projection wants un-rotated. Doing that on the way into the
    /// joined panel replaces three passes over `[N, D]` triples with one: the
    /// arithmetic is [`rope_backward`]'s, elementwise and in fp32, so the
    /// result is the composition's bit for bit.
    ///
    /// One thread per rotated pair; `width` is `heads * head_dim`.
    ///
    /// # Safety
    ///
    /// `dq`, `dk` and `dv` are `[N, heads * head_dim]` and `output` is
    /// `[N, 3 * heads * head_dim]`.
    #[kernel]
    pub unsafe fn join_group3_rope(
        dq: &[f32],
        dk: &[f32],
        dv: &[f32],
        table: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut output: DisjointSlice<f32>,
    ) {
        let pair = thread::index_1d().get();
        // Launches round up to whole blocks; excess threads must not write.
        if 6 * pair >= output.len() {
            return;
        }
        let width = heads as usize * head_dim as usize;
        let base = (pair / (width / 2)) * 3 * width + 2 * (pair % (width / 2));
        let angle = rope_angle(pair, sequence_length, heads, head_dim);
        let (cos, sin) = (table[angle], table[angle + 1]);
        let (q0, q1) = (dq[2 * pair], dq[2 * pair + 1]);
        let (k0, k1) = (dk[2 * pair], dk[2 * pair + 1]);
        unsafe {
            *output.get_unchecked_mut(base) = q0 * cos + q1 * sin;
            *output.get_unchecked_mut(base + 1) = -q0 * sin + q1 * cos;
            *output.get_unchecked_mut(base + width) = k0 * cos + k1 * sin;
            *output.get_unchecked_mut(base + width + 1) = -k0 * sin + k1 * cos;
            *output.get_unchecked_mut(base + 2 * width) = dv[2 * pair];
            *output.get_unchecked_mut(base + 2 * width + 1) = dv[2 * pair + 1];
        }
    }

    /// [`join_group3_rope`] writing the packed-bf16 panel the qkv projection's
    /// backward GEMMs read, rather than an fp32 one a quantize pass turns into
    /// it.
    ///
    /// Both GEMMs consume that panel out of the same buffer through their own
    /// descriptors — K-major for the input product, MN-major for the weight
    /// product — so one packed layout serves both and this kernel is the whole
    /// of what the quantize did. The arithmetic is [`join_group3_rope`]'s, and
    /// an fp32 expression stored and reloaded before rounding rounds the same
    /// as one rounded in registers, so the words are that composition's bit
    /// for bit.
    ///
    /// Each rotated pair is one packed word per group: a group's offset into
    /// the row is a multiple of `width` and a pair starts at an even column,
    /// so the couple a thread owns is exactly the couple
    /// `convert_f32_to_bf16_pairs` would have packed into one word.
    ///
    /// # Safety
    ///
    /// `dq`, `dk` and `dv` are `[N, heads * head_dim]` and `output` holds at
    /// least `N * 3 * heads * head_dim / 2` words.
    #[kernel]
    pub unsafe fn join_group3_rope_bf16(
        dq: &[f32],
        dk: &[f32],
        dv: &[f32],
        table: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut output: DisjointSlice<u32>,
    ) {
        let pair = thread::index_1d().get();
        // Launches round up to whole blocks; excess threads must not write.
        // `output` is a shared operand buffer sized for the widest linear, so
        // the input length is what bounds this and not the output's.
        if 2 * pair >= dq.len() {
            return;
        }
        let half = heads as usize * head_dim as usize / 2;
        let word = (pair / half) * 3 * half + pair % half;
        let angle = rope_angle(pair, sequence_length, heads, head_dim);
        let (cos, sin) = (table[angle], table[angle + 1]);
        let (q0, q1) = (dq[2 * pair], dq[2 * pair + 1]);
        let (k0, k1) = (dk[2 * pair], dk[2 * pair + 1]);
        unsafe {
            *output.get_unchecked_mut(word) = bf16_pair(q0 * cos + q1 * sin, -q0 * sin + q1 * cos);
            *output.get_unchecked_mut(word + half) =
                bf16_pair(k0 * cos + k1 * sin, -k0 * sin + k1 * cos);
            *output.get_unchecked_mut(word + 2 * half) = bf16_pair(dv[2 * pair], dv[2 * pair + 1]);
        }
    }

    #[kernel]
    pub fn embedding_forward(weight: &[u32], tokens: &[u32], dim: u32, mut y: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = y.get_mut(index) {
            let d = dim as usize;
            let row = i / d;
            let col = i % d;
            // The embedding master is bf16 (#57), stored as packed pairs.
            let element = tokens[row] as usize * d + col;
            let word = weight[element / 2];
            let bits = (if element % 2 == 0 { word } else { word >> 16 }) as u16;
            *slot = f32::from_bits((bits as u32) << 16);
        }
    }

    /// [`embedding_forward`] writing the packed activation stream.
    ///
    /// The master is already bf16 (#57) and so is the stream, so a row of the
    /// table reaches the first block's input as whole words: this lookup
    /// neither widens nor rounds, and moves half the bytes of the fp32 one.
    #[kernel]
    pub fn embedding_forward_bf16(
        weight: &[u32],
        tokens: &[u32],
        dim: u32,
        mut y: DisjointSlice<u32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let d = dim as usize;
        if d == 0 || !d.is_multiple_of(2) {
            return;
        }
        let words = d / 2;
        if let Some(slot) = y.get_mut(index) {
            *slot = weight[tokens[i / words] as usize * words + i % words];
        }
    }

    /// Reference embedding backward without atomics: one thread owns each
    /// vocabulary/feature slot and scans token positions.
    #[kernel]
    pub fn embedding_backward(
        tokens: &[u32],
        dy: &[f32],
        token_count: u32,
        dim: u32,
        mut dweight: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = dweight.get_mut(index) {
            let d = dim as usize;
            let token = i / d;
            let col = i % d;
            let mut grad = 0.0f32;
            for row in 0..token_count as usize {
                if tokens[row] as usize == token {
                    grad += dy[row * d + col];
                }
            }
            *slot += grad;
        }
    }

    /// Embedding backward scatter: one thread owns each upstream-gradient
    /// element and atomically accumulates it into the selected vocabulary row.
    ///
    /// Unlike [`embedding_backward`], this does O(token_count * dim) work
    /// rather than making every vocabulary/feature slot scan all token
    /// positions. Device-scope relaxed atomics are sufficient: the stream
    /// orders this kernel with the gradient fill before it and optimizer use
    /// after it, while the atomic only needs to serialize colliding tokens
    /// within this launch.
    #[kernel]
    pub unsafe fn embedding_backward_scatter(
        tokens: &[u32],
        dy: &[f32],
        dim: u32,
        mut dweight: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= dy.len() {
            return;
        }
        let d = dim as usize;
        let row = i / d;
        let col = i % d;
        let output = tokens[row] as usize * d + col;
        if output >= dweight.len() {
            return;
        }

        // SAFETY: `output` was bounds-checked above and the pointer remains
        // valid for the kernel launch. Multiple token positions may select the
        // same output, so every access to that location in this kernel is an
        // atomic fetch-add.
        let slot = unsafe { DeviceAtomicF32::from_ptr(dweight.as_mut_ptr().add(output)) };
        slot.fetch_add(dy[i], AtomicOrdering::Relaxed);
    }

    #[kernel]
    pub fn softmax_forward(logits: &[f32], classes: u32, mut probabilities: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        let c = classes as usize;
        if c == 0 || i >= logits.len() || i >= probabilities.len() {
            return;
        }
        let row = i / c;
        let base = row * c;
        if base + c > logits.len() {
            return;
        }
        let mut max = f32::NEG_INFINITY;
        for col in 0..c {
            max = max.max(logits[base + col]);
        }
        let mut sum_exp = 0.0f32;
        for col in 0..c {
            let value = (logits[base + col] - max).exp();
            sum_exp += value;
        }
        if let Some(slot) = probabilities.get_mut(index) {
            *slot = (logits[i] - max).exp() / sum_exp;
        }
    }

    #[kernel]
    pub fn cross_entropy_loss(
        logits: &[f32],
        targets: &[u32],
        rows: u32,
        classes: u32,
        mut losses: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if row >= rows as usize {
            return;
        }
        let c = classes as usize;
        let base = row * c;
        let mut max = f32::NEG_INFINITY;
        for col in 0..c {
            max = max.max(logits[base + col]);
        }
        let mut sum_exp = 0.0f32;
        for col in 0..c {
            sum_exp += (logits[base + col] - max).exp();
        }
        if let Some(slot) = losses.get_mut(index) {
            *slot = max + sum_exp.ln() - logits[base + targets[row] as usize];
        }
    }

    #[kernel]
    pub fn softmax_cross_entropy_backward(
        probabilities: &[f32],
        targets: &[u32],
        upstream: f32,
        rows: u32,
        classes: u32,
        mut dlogits: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = dlogits.get_mut(index) {
            let c = classes as usize;
            let row = i / c;
            let col = i % c;
            let target = targets[row] as usize;
            let indicator = if col == target { 1.0 } else { 0.0 };
            *slot = upstream * (probabilities[i] - indicator) / rows as f32;
        }
    }

    /// Fused row-parallel softmax and cross-entropy forward.
    ///
    /// One block owns one logits row. Every lane computes an online softmax
    /// summary over its strided vocabulary slice, then the block combines those
    /// summaries without materializing probabilities.
    #[kernel]
    pub fn fused_classifier_forward(
        logits: &[f32],
        targets: &[u32],
        rows: u32,
        classes: u32,
        mut losses: DisjointSlice<f32>,
    ) {
        static mut MAXIMA: SharedArray<f32, CLASSIFIER_THREADS> = SharedArray::UNINIT;
        static mut SUMS: SharedArray<f32, CLASSIFIER_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != CLASSIFIER_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        if row >= rows as usize {
            return;
        }

        let c = classes as usize;
        let base = row * c;
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut col = tid;
        while col < c {
            let value = logits[base + col];
            let next_max = running_max.max(value);
            running_sum = running_sum * (running_max - next_max).exp() + (value - next_max).exp();
            running_max = next_max;
            col += CLASSIFIER_THREADS;
        }
        unsafe {
            MAXIMA[tid] = running_max;
            SUMS[tid] = running_sum;
        }
        thread::sync_threads();

        let mut stride = CLASSIFIER_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let right_sum = SUMS[tid + stride];
                    if right_sum > 0.0 {
                        let left_sum = SUMS[tid];
                        if left_sum > 0.0 {
                            let left_max = MAXIMA[tid];
                            let right_max = MAXIMA[tid + stride];
                            let next_max = left_max.max(right_max);
                            SUMS[tid] = left_sum * (left_max - next_max).exp()
                                + right_sum * (right_max - next_max).exp();
                            MAXIMA[tid] = next_max;
                        } else {
                            SUMS[tid] = right_sum;
                            MAXIMA[tid] = MAXIMA[tid + stride];
                        }
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            let target = targets[row] as usize;
            unsafe {
                *losses.get_unchecked_mut(row) = MAXIMA[0] + SUMS[0].ln() - logits[base + target];
            }
        }
    }

    /// Recompute softmax and overwrite logits with cross-entropy gradients.
    ///
    /// The block reduction matches `fused_classifier_forward`; after all lanes
    /// have consumed the logits, each lane rewrites its disjoint strided slice.
    #[kernel]
    pub fn fused_classifier_backward_in_place(
        targets: &[u32],
        upstream: f32,
        rows: u32,
        classes: u32,
        mut logits: DisjointSlice<f32>,
    ) {
        static mut MAXIMA: SharedArray<f32, CLASSIFIER_THREADS> = SharedArray::UNINIT;
        static mut SUMS: SharedArray<f32, CLASSIFIER_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != CLASSIFIER_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        if row >= rows as usize {
            return;
        }

        let c = classes as usize;
        let base = row * c;
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut col = tid;
        while col < c {
            // SAFETY: the row belongs to this block and striding by the block
            // width gives each lane exclusive ownership of this element.
            let value = unsafe { *logits.get_unchecked_mut(base + col) };
            let next_max = running_max.max(value);
            running_sum = running_sum * (running_max - next_max).exp() + (value - next_max).exp();
            running_max = next_max;
            col += CLASSIFIER_THREADS;
        }
        unsafe {
            MAXIMA[tid] = running_max;
            SUMS[tid] = running_sum;
        }
        thread::sync_threads();

        let mut stride = CLASSIFIER_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let right_sum = SUMS[tid + stride];
                    if right_sum > 0.0 {
                        let left_sum = SUMS[tid];
                        if left_sum > 0.0 {
                            let left_max = MAXIMA[tid];
                            let right_max = MAXIMA[tid + stride];
                            let next_max = left_max.max(right_max);
                            SUMS[tid] = left_sum * (left_max - next_max).exp()
                                + right_sum * (right_max - next_max).exp();
                            MAXIMA[tid] = next_max;
                        } else {
                            SUMS[tid] = right_sum;
                            MAXIMA[tid] = MAXIMA[tid + stride];
                        }
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        let row_max = unsafe { MAXIMA[0] };
        let inverse_sum = 1.0 / unsafe { SUMS[0] };
        let target = targets[row] as usize;
        let scale = upstream / rows as f32;
        let mut col = tid;
        while col < c {
            let index = base + col;
            // SAFETY: this lane exclusively owns `index` for both the read and
            // the subsequent in-place gradient write.
            let value = unsafe { *logits.get_unchecked_mut(index) };
            let probability = (value - row_max).exp() * inverse_sum;
            let indicator = if col == target { 1.0 } else { 0.0 };
            unsafe {
                *logits.get_unchecked_mut(index) = scale * (probability - indicator);
            }
            col += CLASSIFIER_THREADS;
        }
    }

    /// [`fused_classifier_forward`] over packed-bf16 logits rows.
    ///
    /// Rows are `padded_classes` elements wide (packed two per word) but the
    /// softmax and loss only see the first `classes` columns; the padded tail
    /// holds the lm-head's zero-weight vocabulary columns.
    #[kernel]
    pub fn fused_classifier_forward_bf16(
        logits: &[u32],
        targets: &[u32],
        rows: u32,
        classes: u32,
        padded_classes: u32,
        mut losses: DisjointSlice<f32>,
    ) {
        static mut MAXIMA: SharedArray<f32, CLASSIFIER_THREADS> = SharedArray::UNINIT;
        static mut SUMS: SharedArray<f32, CLASSIFIER_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != CLASSIFIER_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        if row >= rows as usize {
            return;
        }

        let c = classes as usize;
        let base = row * padded_classes as usize / 2;
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut pair = tid;
        while 2 * pair < c {
            let word = logits[base + pair];
            let mut half = 0;
            while half < 2 {
                let col = 2 * pair + half;
                if col < c {
                    let value = bf16_bits_to_f32((word >> (16 * half)) as u16);
                    let next_max = running_max.max(value);
                    running_sum =
                        running_sum * (running_max - next_max).exp() + (value - next_max).exp();
                    running_max = next_max;
                }
                half += 1;
            }
            pair += CLASSIFIER_THREADS;
        }
        unsafe {
            MAXIMA[tid] = running_max;
            SUMS[tid] = running_sum;
        }
        thread::sync_threads();

        let mut stride = CLASSIFIER_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let right_sum = SUMS[tid + stride];
                    if right_sum > 0.0 {
                        let left_sum = SUMS[tid];
                        if left_sum > 0.0 {
                            let left_max = MAXIMA[tid];
                            let right_max = MAXIMA[tid + stride];
                            let next_max = left_max.max(right_max);
                            SUMS[tid] = left_sum * (left_max - next_max).exp()
                                + right_sum * (right_max - next_max).exp();
                            MAXIMA[tid] = next_max;
                        } else {
                            SUMS[tid] = right_sum;
                            MAXIMA[tid] = MAXIMA[tid + stride];
                        }
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            let target = targets[row] as usize;
            let word = logits[base + target / 2];
            let bits = (if target % 2 == 0 { word } else { word >> 16 }) as u16;
            unsafe {
                *losses.get_unchecked_mut(row) = MAXIMA[0] + SUMS[0].ln() - bf16_bits_to_f32(bits);
            }
        }
    }

    /// [`fused_classifier_backward_in_place`] over packed-bf16 logits rows.
    ///
    /// The recomputed softmax sees the first `classes` columns; the write-back
    /// covers the full `padded_classes` stride so padded vocabulary columns
    /// carry exactly-zero gradients into the weight GEMM.
    #[kernel]
    pub fn fused_classifier_backward_in_place_bf16(
        targets: &[u32],
        upstream: f32,
        rows: u32,
        classes: u32,
        padded_classes: u32,
        mut logits: DisjointSlice<u32>,
    ) {
        static mut MAXIMA: SharedArray<f32, CLASSIFIER_THREADS> = SharedArray::UNINIT;
        static mut SUMS: SharedArray<f32, CLASSIFIER_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != CLASSIFIER_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        if row >= rows as usize {
            return;
        }

        let c = classes as usize;
        let stride_words = padded_classes as usize / 2;
        let base = row * stride_words;
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut pair = tid;
        while 2 * pair < c {
            // SAFETY: the row belongs to this block and striding by the block
            // width gives each lane exclusive ownership of this word.
            let word = unsafe { *logits.get_unchecked_mut(base + pair) };
            let mut half = 0;
            while half < 2 {
                let col = 2 * pair + half;
                if col < c {
                    let value = bf16_bits_to_f32((word >> (16 * half)) as u16);
                    let next_max = running_max.max(value);
                    running_sum =
                        running_sum * (running_max - next_max).exp() + (value - next_max).exp();
                    running_max = next_max;
                }
                half += 1;
            }
            pair += CLASSIFIER_THREADS;
        }
        unsafe {
            MAXIMA[tid] = running_max;
            SUMS[tid] = running_sum;
        }
        thread::sync_threads();

        let mut stride = CLASSIFIER_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let right_sum = SUMS[tid + stride];
                    if right_sum > 0.0 {
                        let left_sum = SUMS[tid];
                        if left_sum > 0.0 {
                            let left_max = MAXIMA[tid];
                            let right_max = MAXIMA[tid + stride];
                            let next_max = left_max.max(right_max);
                            SUMS[tid] = left_sum * (left_max - next_max).exp()
                                + right_sum * (right_max - next_max).exp();
                            MAXIMA[tid] = next_max;
                        } else {
                            SUMS[tid] = right_sum;
                            MAXIMA[tid] = MAXIMA[tid + stride];
                        }
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        let row_max = unsafe { MAXIMA[0] };
        let inverse_sum = 1.0 / unsafe { SUMS[0] };
        let target = targets[row] as usize;
        let scale = upstream / rows as f32;
        let mut pair = tid;
        while pair < stride_words {
            // SAFETY: this lane exclusively owns the word for both the read
            // and the in-place gradient write.
            let word = unsafe { *logits.get_unchecked_mut(base + pair) };
            let mut packed = 0u32;
            let mut half = 0;
            while half < 2 {
                let col = 2 * pair + half;
                if col < c {
                    let value = bf16_bits_to_f32((word >> (16 * half)) as u16);
                    let probability = (value - row_max).exp() * inverse_sum;
                    let indicator = if col == target { 1.0 } else { 0.0 };
                    let gradient = scale * (probability - indicator);
                    packed |= (f32_to_bf16_bits(gradient) as u32) << (16 * half);
                }
                half += 1;
            }
            unsafe {
                *logits.get_unchecked_mut(base + pair) = packed;
            }
            pair += CLASSIFIER_THREADS;
        }
    }

    /// The SwiGLU row band a tile kernel reads: `silu(gate) * up`.
    ///
    /// Split out because all three tile kernels of this family walk the same
    /// rectangle and differ only in what they compute on it.
    #[inline(always)]
    fn swiglu_row(row: u32) -> u32 {
        SWIGLU_TILE_BLOCK_ROWS as u32 * thread::blockIdx_x() + SWIGLU_TILE_ROWS as u32 * row
    }

    /// `sigmoid(x)`, as a tile map.
    #[inline(always)]
    fn tile_sigmoid(x: SwigluChunk) -> SwigluChunk {
        x.neg().exp().shift(1.0).recip()
    }

    /// [`swiglu_forward`] over register tiles.
    ///
    /// The flat kernel gives one output element to one thread, which is one
    /// 4-byte load from each input and one 4-byte store per thread and a grid
    /// of `rows * columns` of them. Here a warp owns a
    /// `SWIGLU_TILE_ROWS x SWIGLU_TILE_CHUNK` rectangle, so adjacent columns
    /// pair into one 8-byte access and a lane carries a chunk's worth of them
    /// in flight at once. There is no reduction in this kernel and no statistic
    /// to share — the tile buys memory-level parallelism and nothing else,
    /// which on this family is the whole of what was missing.
    ///
    /// # Safety
    ///
    /// Launch with [`SWIGLU_TILE_THREADS`] threads and exactly
    /// `rows / SWIGLU_TILE_BLOCK_ROWS` blocks over `rows x columns` buffers;
    /// `rows` must be a multiple of [`SWIGLU_TILE_BLOCK_ROWS`] and `columns` of
    /// [`SWIGLU_TILE_CHUNK`], both the launcher's to check.
    #[kernel]
    pub unsafe fn swiglu_forward_tile(
        gate: &[f32],
        up: &[f32],
        columns: u32,
        mut y: DisjointSlice<f32>,
    ) {
        unsafe {
            let lane = lane();
            let row = swiglu_row(warp_id());
            let width = columns as usize;
            let gates = GlobalRows::<F32>::from_raw(gate.as_ptr() as *mut u8, width);
            let ups = GlobalRows::<F32>::from_raw(up.as_ptr() as *mut u8, width);
            let destination = GlobalRows::<F32>::from_slice(&mut y, width);

            let mut column = 0u32;
            while column < columns {
                let g: SwigluChunk = load_rows(gates, row, column, lane);
                let u: SwigluChunk = load_rows(ups, row, column, lane);
                store_rows(
                    destination,
                    row,
                    column,
                    lane,
                    g.mul(tile_sigmoid(g)).mul(u),
                );
                column += SWIGLU_TILE_CHUNK as u32;
            }
        }
    }

    /// [`swiglu_backward_gate`] over register tiles.
    ///
    /// # Safety
    ///
    /// As [`swiglu_forward_tile`].
    #[kernel]
    pub unsafe fn swiglu_backward_gate_tile(
        gate: &[f32],
        up: &[f32],
        dy: &[f32],
        columns: u32,
        mut dgate: DisjointSlice<f32>,
    ) {
        unsafe {
            let lane = lane();
            let row = swiglu_row(warp_id());
            let width = columns as usize;
            let gates = GlobalRows::<F32>::from_raw(gate.as_ptr() as *mut u8, width);
            let ups = GlobalRows::<F32>::from_raw(up.as_ptr() as *mut u8, width);
            let upstream = GlobalRows::<F32>::from_raw(dy.as_ptr() as *mut u8, width);
            let destination = GlobalRows::<F32>::from_slice(&mut dgate, width);

            let mut column = 0u32;
            while column < columns {
                let g: SwigluChunk = load_rows(gates, row, column, lane);
                let u: SwigluChunk = load_rows(ups, row, column, lane);
                let d: SwigluChunk = load_rows(upstream, row, column, lane);
                let s = tile_sigmoid(g);
                // silu'(g) = s * (1 + g * (1 - s)), the flat kernel's `dsilu`.
                let dsilu = s.mul(g.mul(s.neg().shift(1.0)).shift(1.0));
                store_rows(destination, row, column, lane, d.mul(u).mul(dsilu));
                column += SWIGLU_TILE_CHUNK as u32;
            }
        }
    }

    /// [`swiglu_backward_up`] over register tiles.
    ///
    /// # Safety
    ///
    /// As [`swiglu_forward_tile`].
    #[kernel]
    pub unsafe fn swiglu_backward_up_tile(
        gate: &[f32],
        dy: &[f32],
        columns: u32,
        mut dup: DisjointSlice<f32>,
    ) {
        unsafe {
            let lane = lane();
            let row = swiglu_row(warp_id());
            let width = columns as usize;
            let gates = GlobalRows::<F32>::from_raw(gate.as_ptr() as *mut u8, width);
            let upstream = GlobalRows::<F32>::from_raw(dy.as_ptr() as *mut u8, width);
            let destination = GlobalRows::<F32>::from_slice(&mut dup, width);

            let mut column = 0u32;
            while column < columns {
                let g: SwigluChunk = load_rows(gates, row, column, lane);
                let d: SwigluChunk = load_rows(upstream, row, column, lane);
                store_rows(
                    destination,
                    row,
                    column,
                    lane,
                    d.mul(g).mul(tile_sigmoid(g)),
                );
                column += SWIGLU_TILE_CHUNK as u32;
            }
        }
    }

    /// Index of a rotated pair's `(cos, sin)` couple in a [`rope_table`].
    ///
    /// `pair` counts rotated pairs across the whole `[N, heads * head_dim]`
    /// row, and pairs never straddle a head, so the head falls out of the
    /// modulus without ever being named.
    #[inline(always)]
    fn rope_angle(pair: usize, sequence_length: u32, heads: u32, head_dim: u32) -> usize {
        let half = head_dim as usize / 2;
        let position = (pair / (heads as usize * half)) % sequence_length as usize;
        2 * (position * half + pair % half)
    }

    /// RoPE forward over `[N, heads * head_dim]`, one thread per rotated pair.
    ///
    /// # Safety
    ///
    /// `y` and `x` hold the same element count and `table` is a
    /// [`rope_table`] for this `sequence_length` and `head_dim`.
    #[kernel]
    pub unsafe fn rope_forward(
        x: &[f32],
        table: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let pair = thread::index_1d().get();
        // Launches round up to whole blocks; excess threads must not write.
        if 2 * pair + 1 >= y.len() {
            return;
        }
        let angle = rope_angle(pair, sequence_length, heads, head_dim);
        let (cos, sin) = (table[angle], table[angle + 1]);
        let (x0, x1) = (x[2 * pair], x[2 * pair + 1]);
        unsafe {
            *y.get_unchecked_mut(2 * pair) = x0 * cos - x1 * sin;
            *y.get_unchecked_mut(2 * pair + 1) = x0 * sin + x1 * cos;
        }
    }

    /// RoPE backward: the transposed rotation, same pair per thread.
    ///
    /// # Safety
    ///
    /// As [`rope_forward`].
    #[kernel]
    pub unsafe fn rope_backward(
        dy: &[f32],
        table: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut dx: DisjointSlice<f32>,
    ) {
        let pair = thread::index_1d().get();
        if 2 * pair + 1 >= dx.len() {
            return;
        }
        let angle = rope_angle(pair, sequence_length, heads, head_dim);
        let (cos, sin) = (table[angle], table[angle + 1]);
        let (d0, d1) = (dy[2 * pair], dy[2 * pair + 1]);
        unsafe {
            *dx.get_unchecked_mut(2 * pair) = d0 * cos + d1 * sin;
            *dx.get_unchecked_mut(2 * pair + 1) = -d0 * sin + d1 * cos;
        }
    }

    /// Materialize causal softmax probabilities as `[N,H,T]`.
    #[kernel]
    pub fn attention_probabilities(
        q: &[f32],
        k: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut probabilities: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = probabilities.get_mut(index) {
            let t = sequence_length as usize;
            let h = heads as usize;
            let hd = head_dim as usize;
            let key_position = i % t;
            let head = (i / t) % h;
            let query_row = i / (t * h);
            let query_position = query_row % t;
            if key_position > query_position {
                *slot = 0.0;
                return;
            }
            let sequence_start = query_row - query_position;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut max_score = f32::NEG_INFINITY;
            for candidate in 0..=query_position {
                let key_row = sequence_start + candidate;
                let mut dot = 0.0f32;
                for dim in 0..hd {
                    dot += q[query_row * h * hd + head * hd + dim]
                        * k[key_row * h * hd + head * hd + dim];
                }
                max_score = max_score.max(dot * scale);
            }
            let mut denominator = 0.0f32;
            let mut selected = 0.0f32;
            for candidate in 0..=query_position {
                let key_row = sequence_start + candidate;
                let mut dot = 0.0f32;
                for dim in 0..hd {
                    dot += q[query_row * h * hd + head * hd + dim]
                        * k[key_row * h * hd + head * hd + dim];
                }
                let exponential = (dot * scale - max_score).exp();
                denominator += exponential;
                if candidate == key_position {
                    selected = exponential;
                }
            }
            *slot = selected / denominator;
        }
    }

    #[kernel]
    pub fn attention_output(
        probabilities: &[f32],
        v: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = output.get_mut(index) {
            let t = sequence_length as usize;
            let h = heads as usize;
            let hd = head_dim as usize;
            let dim = i % hd;
            let head = (i / hd) % h;
            let query_row = i / (h * hd);
            let query_position = query_row % t;
            let sequence_start = query_row - query_position;
            let mut value = 0.0f32;
            for key_position in 0..=query_position {
                let key_row = sequence_start + key_position;
                let p = probabilities[(query_row * h + head) * t + key_position];
                value += p * v[key_row * h * hd + head * hd + dim];
            }
            *slot = value;
        }
    }

    #[kernel]
    pub fn attention_backward_q(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        probabilities: &[f32],
        dy: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut dq: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = dq.get_mut(index) {
            let t = sequence_length as usize;
            let h = heads as usize;
            let hd = head_dim as usize;
            let dim = i % hd;
            let head = (i / hd) % h;
            let query_row = i / (h * hd);
            let query_position = query_row % t;
            let sequence_start = query_row - query_position;
            let mut softmax_dot = 0.0f32;
            for key_position in 0..=query_position {
                let key_row = sequence_start + key_position;
                let mut dp = 0.0f32;
                for d in 0..hd {
                    dp += dy[query_row * h * hd + head * hd + d]
                        * v[key_row * h * hd + head * hd + d];
                }
                softmax_dot += probabilities[(query_row * h + head) * t + key_position] * dp;
            }
            let mut value = 0.0f32;
            let scale = 1.0 / (head_dim as f32).sqrt();
            for key_position in 0..=query_position {
                let key_row = sequence_start + key_position;
                let mut dp = 0.0f32;
                for d in 0..hd {
                    dp += dy[query_row * h * hd + head * hd + d]
                        * v[key_row * h * hd + head * hd + d];
                }
                let p = probabilities[(query_row * h + head) * t + key_position];
                value += p * (dp - softmax_dot) * scale * k[key_row * h * hd + head * hd + dim];
            }
            *slot = value;
            let _ = q;
        }
    }

    #[kernel]
    pub fn attention_backward_k(
        q: &[f32],
        v: &[f32],
        probabilities: &[f32],
        dy: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut dk: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = dk.get_mut(index) {
            let t = sequence_length as usize;
            let h = heads as usize;
            let hd = head_dim as usize;
            let dim = i % hd;
            let head = (i / hd) % h;
            let key_row = i / (h * hd);
            let key_position = key_row % t;
            let sequence_start = key_row - key_position;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut value = 0.0f32;
            for query_position in key_position..t {
                let query_row = sequence_start + query_position;
                let mut softmax_dot = 0.0f32;
                let mut selected_dp = 0.0f32;
                for candidate in 0..=query_position {
                    let candidate_row = sequence_start + candidate;
                    let mut dp = 0.0f32;
                    for d in 0..hd {
                        dp += dy[query_row * h * hd + head * hd + d]
                            * v[candidate_row * h * hd + head * hd + d];
                    }
                    softmax_dot += probabilities[(query_row * h + head) * t + candidate] * dp;
                    if candidate == key_position {
                        selected_dp = dp;
                    }
                }
                let p = probabilities[(query_row * h + head) * t + key_position];
                value += p
                    * (selected_dp - softmax_dot)
                    * scale
                    * q[query_row * h * hd + head * hd + dim];
            }
            *slot = value;
        }
    }

    #[kernel]
    pub fn attention_backward_v(
        probabilities: &[f32],
        dy: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut dv: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = dv.get_mut(index) {
            let t = sequence_length as usize;
            let h = heads as usize;
            let hd = head_dim as usize;
            let dim = i % hd;
            let head = (i / hd) % h;
            let key_row = i / (h * hd);
            let key_position = key_row % t;
            let sequence_start = key_row - key_position;
            let mut value = 0.0f32;
            for query_position in key_position..t {
                let query_row = sequence_start + query_position;
                value += probabilities[(query_row * h + head) * t + key_position]
                    * dy[query_row * h * hd + head * hd + dim];
            }
            *slot = value;
        }
    }

    /// Register-tiled fp32 `C[m,n] = A[m,k] B[k,n]` for the router logits. The
    /// token tile `BM` is loaded once per `BK` step and reused across the
    /// experts, so the router weight is streamed from L2 rather than re-read
    /// per token. One lane owns one output; the skinny expert width fits `BN`.
    #[inline(always)]
    unsafe fn router_gemm_impl(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
    ) {
        static mut TILE_A: SharedArray<f32, { ROUTER_GEMM_BM * ROUTER_GEMM_BK }> =
            SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, { ROUTER_GEMM_BK * ROUTER_GEMM_BN }> =
            SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != ROUTER_GEMM_THREADS {
            return;
        }
        let thread_row = tid / ROUTER_GEMM_BN;
        let thread_col = tid % ROUTER_GEMM_BN;
        let block_row = thread::blockIdx_y() as usize * ROUTER_GEMM_BM;
        let block_col = thread::blockIdx_x() as usize * ROUTER_GEMM_BN;
        let mut accumulator = 0.0f32;

        let mut k_base = 0usize;
        while k_base < k {
            let mut local = tid;
            while local < ROUTER_GEMM_BM * ROUTER_GEMM_BK {
                let tile_row = local / ROUTER_GEMM_BK;
                let tile_col = local % ROUTER_GEMM_BK;
                let global_row = block_row + tile_row;
                let global_col = k_base + tile_col;
                unsafe {
                    TILE_A[tile_row * ROUTER_GEMM_BK + tile_col] =
                        if global_row < m && global_col < k {
                            a[global_row * k + global_col]
                        } else {
                            0.0
                        };
                }
                local += ROUTER_GEMM_THREADS;
            }

            local = tid;
            while local < ROUTER_GEMM_BK * ROUTER_GEMM_BN {
                let tile_row = local / ROUTER_GEMM_BN;
                let tile_col = local % ROUTER_GEMM_BN;
                let global_row = k_base + tile_row;
                let global_col = block_col + tile_col;
                unsafe {
                    TILE_B[tile_row * ROUTER_GEMM_BN + tile_col] =
                        if global_row < k && global_col < n {
                            b[global_row * n + global_col]
                        } else {
                            0.0
                        };
                }
                local += ROUTER_GEMM_THREADS;
            }
            thread::sync_threads();

            let mut inner = 0usize;
            while inner < ROUTER_GEMM_BK {
                unsafe {
                    accumulator += TILE_A[thread_row * ROUTER_GEMM_BK + inner]
                        * TILE_B[inner * ROUTER_GEMM_BN + thread_col];
                }
                inner += 1;
            }
            thread::sync_threads();
            k_base += ROUTER_GEMM_BK;
        }

        let global_row = block_row + thread_row;
        let global_col = block_col + thread_col;
        if global_row < m && global_col < n {
            unsafe {
                *c.get_unchecked_mut(global_row * n + global_col) = accumulator;
            }
        }
    }

    /// Router logits for a skinny `[N,D] x [D,E]` fp32 matrix multiply.
    #[kernel]
    pub unsafe fn router_logits(
        x: &[f32],
        weight: &[f32],
        dim: u32,
        experts: u32,
        logits: DisjointSlice<f32>,
    ) {
        let d = dim as usize;
        let e = experts as usize;
        if d == 0 || e == 0 || !x.len().is_multiple_of(d) {
            return;
        }
        let n = x.len() / d;
        unsafe { router_gemm_impl(n, e, d, x, weight, logits) }
    }

    /// [`router_gemm_impl`] with a packed-bf16 `A`.
    ///
    /// Only the token tile's staging load differs: it widens on the way into
    /// shared memory, so the inner product and the accumulator are the fp32
    /// ones the twin uses and the router's own arithmetic is unchanged.
    #[inline(always)]
    unsafe fn router_gemm_bf16_impl(
        m: usize,
        n: usize,
        k: usize,
        a: &[u32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
    ) {
        static mut TILE_A: SharedArray<f32, { ROUTER_GEMM_BM * ROUTER_GEMM_BK }> =
            SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, { ROUTER_GEMM_BK * ROUTER_GEMM_BN }> =
            SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != ROUTER_GEMM_THREADS {
            return;
        }
        let thread_row = tid / ROUTER_GEMM_BN;
        let thread_col = tid % ROUTER_GEMM_BN;
        let block_row = thread::blockIdx_y() as usize * ROUTER_GEMM_BM;
        let block_col = thread::blockIdx_x() as usize * ROUTER_GEMM_BN;
        let mut accumulator = 0.0f32;

        let mut k_base = 0usize;
        while k_base < k {
            let mut local = tid;
            while local < ROUTER_GEMM_BM * ROUTER_GEMM_BK {
                let tile_row = local / ROUTER_GEMM_BK;
                let tile_col = local % ROUTER_GEMM_BK;
                let global_row = block_row + tile_row;
                let global_col = k_base + tile_col;
                unsafe {
                    TILE_A[tile_row * ROUTER_GEMM_BK + tile_col] =
                        if global_row < m && global_col < k {
                            bf16_at(a, global_row * k + global_col)
                        } else {
                            0.0
                        };
                }
                local += ROUTER_GEMM_THREADS;
            }

            local = tid;
            while local < ROUTER_GEMM_BK * ROUTER_GEMM_BN {
                let tile_row = local / ROUTER_GEMM_BN;
                let tile_col = local % ROUTER_GEMM_BN;
                let global_row = k_base + tile_row;
                let global_col = block_col + tile_col;
                unsafe {
                    TILE_B[tile_row * ROUTER_GEMM_BN + tile_col] =
                        if global_row < k && global_col < n {
                            b[global_row * n + global_col]
                        } else {
                            0.0
                        };
                }
                local += ROUTER_GEMM_THREADS;
            }
            thread::sync_threads();

            let mut inner = 0usize;
            while inner < ROUTER_GEMM_BK {
                unsafe {
                    accumulator += TILE_A[thread_row * ROUTER_GEMM_BK + inner]
                        * TILE_B[inner * ROUTER_GEMM_BN + thread_col];
                }
                inner += 1;
            }
            thread::sync_threads();
            k_base += ROUTER_GEMM_BK;
        }

        let global_row = block_row + thread_row;
        let global_col = block_col + thread_col;
        if global_row < m && global_col < n {
            unsafe {
                *c.get_unchecked_mut(global_row * n + global_col) = accumulator;
            }
        }
    }

    /// [`router_logits`] over a packed-bf16 token stream. The router weight
    /// and the logits stay fp32: the router is an fp32 parameter by design.
    #[kernel]
    pub unsafe fn router_logits_bf16(
        x: &[u32],
        weight: &[f32],
        dim: u32,
        experts: u32,
        logits: DisjointSlice<f32>,
    ) {
        let d = dim as usize;
        let e = experts as usize;
        if d == 0 || e == 0 || !(x.len() * 2).is_multiple_of(d) {
            return;
        }
        let n = x.len() * 2 / d;
        unsafe { router_gemm_bf16_impl(n, e, d, x, weight, logits) }
    }

    /// Per-token softmax, deterministic top-k, and selected-probability
    /// renormalization. Ties select the lower expert index.
    #[kernel]
    pub unsafe fn router_softmax_topk(
        logits: &[f32],
        experts: u32,
        top_k: u32,
        mut probabilities: DisjointSlice<f32>,
        mut selected_experts: DisjointSlice<u32>,
        mut gate_weights: DisjointSlice<f32>,
    ) {
        let token = thread::index_1d().get();
        let e = experts as usize;
        let k = top_k as usize;
        if e == 0
            || k == 0
            || k > e
            || token * e + e > logits.len()
            || token * e + e > probabilities.len()
            || token * k + k > selected_experts.len()
            || token * k + k > gate_weights.len()
        {
            return;
        }

        let mut maximum = f32::NEG_INFINITY;
        for expert in 0..e {
            maximum = maximum.max(logits[token * e + expert]);
        }
        let mut denominator = 0.0f32;
        for expert in 0..e {
            denominator += (logits[token * e + expert] - maximum).exp();
        }
        for expert in 0..e {
            unsafe {
                *probabilities.get_unchecked_mut(token * e + expert) =
                    (logits[token * e + expert] - maximum).exp() / denominator;
            }
        }

        for rank in 0..k {
            let mut best_expert = 0usize;
            let mut best_probability = f32::NEG_INFINITY;
            for expert in 0..e {
                let mut already_selected = false;
                for previous_rank in 0..rank {
                    if unsafe { *selected_experts.as_mut_ptr().add(token * k + previous_rank) }
                        as usize
                        == expert
                    {
                        already_selected = true;
                    }
                }
                let probability = unsafe { *probabilities.as_mut_ptr().add(token * e + expert) };
                if !already_selected
                    && (probability > best_probability
                        || (probability == best_probability && expert < best_expert))
                {
                    best_probability = probability;
                    best_expert = expert;
                }
            }
            unsafe {
                *selected_experts.get_unchecked_mut(token * k + rank) = best_expert as u32;
            }
        }

        let mut selected_sum = 0.0f32;
        for rank in 0..k {
            let expert = unsafe { *selected_experts.as_mut_ptr().add(token * k + rank) } as usize;
            selected_sum += unsafe { *probabilities.as_mut_ptr().add(token * e + expert) };
        }
        for rank in 0..k {
            let expert = unsafe { *selected_experts.as_mut_ptr().add(token * k + rank) } as usize;
            unsafe {
                *gate_weights.get_unchecked_mut(token * k + rank) =
                    *probabilities.as_mut_ptr().add(token * e + expert) / selected_sum;
            }
        }
    }

    /// Deterministic capacity assignment: one block serially scans token order
    /// for one expert, avoiding nondeterministic atomic slot claims.
    #[kernel]
    pub unsafe fn moe_bin_assign(
        selected_experts: &[u32],
        tokens: u32,
        experts: u32,
        top_k: u32,
        capacity: u32,
        mut slots: DisjointSlice<u32>,
        mut assignment_counts: DisjointSlice<u32>,
    ) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        let expert = thread::blockIdx_x() as usize;
        let n = tokens as usize;
        let e = experts as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if expert >= e
            || n * k > selected_experts.len()
            || n * k > slots.len()
            || expert >= assignment_counts.len()
        {
            return;
        }

        let mut count = 0usize;
        for token in 0..n {
            for rank in 0..k {
                let pair = token * k + rank;
                if selected_experts[pair] as usize == expert {
                    unsafe {
                        *slots.get_unchecked_mut(pair) = if count < c {
                            count as u32
                        } else {
                            MOE_DROPPED_SLOT
                        };
                    }
                    count += 1;
                }
            }
        }
        unsafe {
            *assignment_counts.get_unchecked_mut(expert) = count as u32;
        }
    }

    /// Block-parallel deterministic capacity assignment.
    ///
    /// One block owns an expert and partitions the flattened token/rank order
    /// into contiguous lane ranges. The exclusive prefix of each range's match
    /// count is its first slot, preserving the serial kernel's exact ordering,
    /// capacity-drop behavior, and assignment counts without atomic claims.
    #[kernel]
    pub unsafe fn moe_bin_assign_parallel(
        selected_experts: &[u32],
        tokens: u32,
        experts: u32,
        top_k: u32,
        capacity: u32,
        mut slots: DisjointSlice<u32>,
        mut assignment_counts: DisjointSlice<u32>,
    ) {
        static mut RANGE_COUNTS: SharedArray<u32, MOE_ASSIGN_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != MOE_ASSIGN_THREADS {
            return;
        }
        let expert = thread::blockIdx_x() as usize;
        let n = tokens as usize;
        let e = experts as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        let pairs = n * k;
        if expert >= e
            || pairs > selected_experts.len()
            || pairs > slots.len()
            || expert >= assignment_counts.len()
        {
            return;
        }

        let pairs_per_lane = pairs.div_ceil(MOE_ASSIGN_THREADS);
        let start = (tid * pairs_per_lane).min(pairs);
        let end = (start + pairs_per_lane).min(pairs);
        let mut range_count = 0u32;
        for pair in start..end {
            if selected_experts[pair] as usize == expert {
                range_count += 1;
            }
        }
        unsafe {
            RANGE_COUNTS[tid] = range_count;
        }
        thread::sync_threads();

        // Inclusive Hillis-Steele scan. The barrier before each write keeps
        // every lane's read on the previous iteration's shared-memory state.
        let mut offset = 1usize;
        while offset < MOE_ASSIGN_THREADS {
            let preceding = if tid >= offset {
                unsafe { RANGE_COUNTS[tid - offset] }
            } else {
                0
            };
            thread::sync_threads();
            if tid >= offset {
                unsafe {
                    RANGE_COUNTS[tid] += preceding;
                }
            }
            thread::sync_threads();
            offset *= 2;
        }

        let range_start = if tid == 0 {
            0
        } else {
            unsafe { RANGE_COUNTS[tid - 1] }
        };
        let mut local_count = 0u32;
        for pair in start..end {
            if selected_experts[pair] as usize == expert {
                let slot = range_start + local_count;
                unsafe {
                    *slots.get_unchecked_mut(pair) = if slot < c as u32 {
                        slot
                    } else {
                        MOE_DROPPED_SLOT
                    };
                }
                local_count += 1;
            }
        }
        if tid == 0 {
            unsafe {
                *assignment_counts.get_unchecked_mut(expert) = RANGE_COUNTS[MOE_ASSIGN_THREADS - 1];
            }
        }
    }

    /// Parallel token reduction for each expert's mean routing probability.
    ///
    /// One block owns an expert, and so owns that expert's slot outright: lanes
    /// accumulate strided token slices, a tree reduction combines them, and
    /// lane 0 stores. Accumulating the lanes with same-address atomics instead
    /// needed the buffer pre-zeroed by a `fill` launch per layer, serialized
    /// [`MOE_PROBABILITY_SUMS_THREADS`] adds on one word, and left the summation
    /// order — and so the reported auxiliary loss — dependent on their arrival
    /// order (#99). The auxiliary *gradient* never reads this: `router_backward`
    /// derives it from the assignment counts.
    #[kernel]
    pub unsafe fn moe_probability_sums(
        probabilities: &[f32],
        tokens: u32,
        experts: u32,
        mut probability_sums: DisjointSlice<f32>,
    ) {
        static mut SUMS: SharedArray<f32, MOE_PROBABILITY_SUMS_THREADS> = SharedArray::UNINIT;

        let expert = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let n = tokens as usize;
        let e = experts as usize;
        // Uniform over the block, so no lane reaches a barrier the rest skip.
        if expert >= e || expert >= probability_sums.len() || probabilities.len() < n * e {
            return;
        }
        let mut sum = 0.0f32;
        let mut token = lane;
        while token < n {
            sum += probabilities[token * e + expert];
            token += MOE_PROBABILITY_SUMS_THREADS;
        }

        // SAFETY: each lane owns its own slot of the block's scratch.
        unsafe {
            SUMS[lane] = sum;
        }
        thread::sync_threads();
        let mut stride = MOE_PROBABILITY_SUMS_THREADS / 2;
        while stride > 0 {
            if lane < stride {
                // SAFETY: the surviving half of the lanes own disjoint slots.
                unsafe {
                    SUMS[lane] += SUMS[lane + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            // SAFETY: this block exclusively owns `expert`.
            unsafe {
                *probability_sums.get_unchecked_mut(expert) = SUMS[0];
            }
        }
    }

    /// Add the weighted Switch-style load-balancing loss to the scalar training
    /// loss from the already-reduced expert probability sums.
    #[kernel]
    pub unsafe fn moe_aux_loss(
        probability_sums: &[f32],
        assignment_counts: &[u32],
        tokens: u32,
        experts: u32,
        top_k: u32,
        coefficient: f32,
        mut loss: DisjointSlice<f32>,
    ) {
        if thread::index_1d().get() != 0 || loss.is_empty() {
            return;
        }
        let n = tokens as usize;
        let e = experts as usize;
        let k = top_k as usize;
        if n == 0 || e == 0 || k == 0 || probability_sums.len() < e || assignment_counts.len() < e {
            return;
        }
        let mut auxiliary = 0.0f32;
        for expert in 0..e {
            let assignment_fraction = assignment_counts[expert] as f32 / (n * k) as f32;
            auxiliary += assignment_fraction * probability_sums[expert] / n as f32;
        }
        unsafe {
            *loss.get_unchecked_mut(0) += coefficient * e as f32 * auxiliary;
        }
    }

    /// Copy surviving token rows into `[E,C,D]` capacity-padded expert bins.
    /// The destination must be zeroed before launch so unused slots stay inert.
    #[kernel]
    pub unsafe fn moe_scatter(
        x: &[f32],
        selected_experts: &[u32],
        slots: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut expert_input: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        let pair = i / d;
        let column = i % d;
        if pair >= selected_experts.len() || pair >= slots.len() || d == 0 || k == 0 {
            return;
        }
        let slot = slots[pair];
        if slot == MOE_DROPPED_SLOT {
            return;
        }
        let expert = selected_experts[pair] as usize;
        let token = pair / k;
        let output = (expert * c + slot as usize) * d + column;
        if token * d + column >= x.len() || output >= expert_input.len() {
            return;
        }
        // Deterministic bin assignment guarantees one writer per accepted slot.
        unsafe {
            *expert_input.get_unchecked_mut(output) = x[token * d + column];
        }
    }

    /// [`moe_scatter`] storing packed-bf16 pairs, one word per thread.
    ///
    /// Both readers of the expert input panel are bf16 tcgen05 operands, so
    /// the routing copy rounds once here rather than writing a wide panel that
    /// two quantize launches then re-read (#59). `dim` must be even.
    #[kernel]
    pub unsafe fn moe_scatter_bf16(
        x: &[f32],
        selected_experts: &[u32],
        slots: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut expert_input: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if d == 0 || k == 0 || !d.is_multiple_of(2) {
            return;
        }
        let half = d / 2;
        let pair = i / half;
        let column = i % half;
        if pair >= selected_experts.len() || pair >= slots.len() {
            return;
        }
        let slot = slots[pair];
        if slot == MOE_DROPPED_SLOT {
            return;
        }
        let expert = selected_experts[pair] as usize;
        let token = pair / k;
        let output = (expert * c + slot as usize) * half + column;
        let source = token * d + 2 * column;
        if source + 1 >= x.len() || output >= expert_input.len() {
            return;
        }
        // Deterministic bin assignment guarantees one writer per accepted slot.
        unsafe {
            *expert_input.get_unchecked_mut(output) = f32_to_bf16_bits(x[source]) as u32
                | ((f32_to_bf16_bits(x[source + 1]) as u32) << 16);
        }
    }

    /// [`moe_scatter_bf16`] moving one 16-byte source quad and one 8-byte
    /// packed store per thread. `dim` must be a multiple of [`QUAD_LANES`].
    #[kernel]
    pub unsafe fn moe_scatter_bf16_quad(
        x: &[f32],
        selected_experts: &[u32],
        slots: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut expert_input: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if d == 0 || k == 0 || !d.is_multiple_of(QUAD_LANES) {
            return;
        }
        let row_quads = d / QUAD_LANES;
        let pair = i / row_quads;
        let quad = i % row_quads;
        if pair >= selected_experts.len() || pair >= slots.len() {
            return;
        }
        let slot = slots[pair];
        if slot == MOE_DROPPED_SLOT {
            return;
        }
        let expert = selected_experts[pair] as usize;
        let token = pair / k;
        let source = token * d + QUAD_LANES * quad;
        let word = (expert * c + slot as usize) * (d / 2) + 2 * quad;
        if source + QUAD_LANES > x.len() || word + 1 >= expert_input.len() {
            return;
        }
        // SAFETY: `source` is a multiple of `QUAD_LANES` so the 16-byte load
        // is aligned; `word` is even so the 8-byte store is aligned; bounds
        // were checked above and deterministic bin assignment guarantees one
        // writer per accepted slot.
        let values = quad_lanes(unsafe { *(x.as_ptr().add(source) as *const u128) });
        let mut packed = 0u64;
        for lane in 0..QUAD_LANES {
            packed |= (f32_to_bf16_bits(values[lane]) as u64) << (16 * lane);
        }
        unsafe {
            *(expert_input.as_mut_ptr().add(word) as *mut u64) = packed;
        }
    }

    /// [`moe_scatter_bf16_quad`] with a packed source.
    ///
    /// The token stream is already the dtype the bins hold, so the scatter is
    /// one 8-byte copy per thread: it reads half the bytes of the fp32 twin
    /// and rounds nothing at all. `dim` must be a multiple of [`QUAD_LANES`].
    #[kernel]
    pub unsafe fn moe_scatter_packed_quad(
        x: &[u32],
        selected_experts: &[u32],
        slots: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut expert_input: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if d == 0 || k == 0 || !d.is_multiple_of(QUAD_LANES) {
            return;
        }
        let row_quads = d / QUAD_LANES;
        let pair = i / row_quads;
        let quad = i % row_quads;
        if pair >= selected_experts.len() || pair >= slots.len() {
            return;
        }
        let slot = slots[pair];
        if slot == MOE_DROPPED_SLOT {
            return;
        }
        let expert = selected_experts[pair] as usize;
        let token = pair / k;
        let source = token * (d / 2) + 2 * quad;
        let word = (expert * c + slot as usize) * (d / 2) + 2 * quad;
        if source + 1 >= x.len() || word + 1 >= expert_input.len() {
            return;
        }
        // SAFETY: `source` and `word` are both even, so the 8-byte load and
        // store are aligned; bounds were checked above and deterministic bin
        // assignment guarantees one writer per accepted slot.
        let packed = unsafe { *(x.as_ptr().add(source) as *const u64) };
        unsafe {
            *(expert_input.as_mut_ptr().add(word) as *mut u64) = packed;
        }
    }

    /// [`moe_scatter`] with a packed source and an fp32 destination, for the
    /// shapes whose expert panels miss the tcgen05 contract and stay wide.
    /// The destination must be pre-zeroed.
    #[kernel]
    pub unsafe fn moe_scatter_packed_wide(
        x: &[u32],
        selected_experts: &[u32],
        slots: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut expert_input: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if d == 0 || k == 0 {
            return;
        }
        let pair = i / d;
        let column = i % d;
        if pair >= selected_experts.len() || pair >= slots.len() {
            return;
        }
        let slot = slots[pair];
        if slot == MOE_DROPPED_SLOT {
            return;
        }
        let expert = selected_experts[pair] as usize;
        let token = pair / k;
        let output = (expert * c + slot as usize) * d + column;
        let source = token * d + column;
        if source >= x.len() * 2 || output >= expert_input.len() {
            return;
        }
        // Deterministic bin assignment guarantees one writer per accepted slot.
        unsafe {
            *expert_input.get_unchecked_mut(output) = bf16_at(x, source);
        }
    }

    /// Gather expert outputs to token order using the renormalized gate weights.
    #[kernel]
    pub fn moe_gather_combine(
        expert_output: &[f32],
        selected_experts: &[u32],
        gate_weights: &[f32],
        slots: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if i >= output.len() || d == 0 || k == 0 {
            return;
        }
        let token = i / d;
        let column = i % d;
        let mut value = 0.0f32;
        for rank in 0..k {
            let pair = token * k + rank;
            if pair >= slots.len() || pair >= selected_experts.len() || pair >= gate_weights.len() {
                return;
            }
            let slot = slots[pair];
            if slot != MOE_DROPPED_SLOT {
                let expert = selected_experts[pair] as usize;
                let input = (expert * c + slot as usize) * d + column;
                if input >= expert_output.len() {
                    return;
                }
                value += gate_weights[pair] * expert_output[input];
            }
        }
        if let Some(slot) = output.get_mut(index) {
            *slot = value;
        }
    }

    /// [`moe_gather_combine`] with the residual add folded in: each output
    /// element is `residual + Σ_k gate · expert_output`, so the separate
    /// `[N, D]` residual-add pass and the intermediate it read never run.
    #[kernel]
    pub fn moe_gather_combine_add(
        expert_output: &[f32],
        selected_experts: &[u32],
        gate_weights: &[f32],
        slots: &[u32],
        residual: &[f32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if i >= output.len() || i >= residual.len() || d == 0 || k == 0 {
            return;
        }
        let token = i / d;
        let column = i % d;
        let mut value = residual[i];
        for rank in 0..k {
            let pair = token * k + rank;
            if pair >= slots.len() || pair >= selected_experts.len() || pair >= gate_weights.len() {
                return;
            }
            let slot = slots[pair];
            if slot != MOE_DROPPED_SLOT {
                let expert = selected_experts[pair] as usize;
                let input = (expert * c + slot as usize) * d + column;
                if input >= expert_output.len() {
                    return;
                }
                value += gate_weights[pair] * expert_output[input];
            }
        }
        if let Some(slot) = output.get_mut(index) {
            *slot = value;
        }
    }

    /// [`moe_gather_combine_add`] moving one 16-byte quad per access. `dim`
    /// must be a multiple of [`QUAD_LANES`].
    #[kernel]
    pub unsafe fn moe_gather_combine_add_quad(
        expert_output: &[f32],
        selected_experts: &[u32],
        gate_weights: &[f32],
        slots: &[u32],
        residual: &[f32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut output: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if d == 0 || k == 0 || !d.is_multiple_of(QUAD_LANES) {
            return;
        }
        let row_quads = d / QUAD_LANES;
        let token = i / row_quads;
        let quad = i % row_quads;
        let base = token * d + QUAD_LANES * quad;
        if base + QUAD_LANES > residual.len()
            || base + QUAD_LANES > output.len()
            || (token + 1) * k > slots.len()
            || (token + 1) * k > selected_experts.len()
            || (token + 1) * k > gate_weights.len()
        {
            return;
        }
        // SAFETY: `base` and every bin row offset are multiples of
        // `QUAD_LANES`, so all 16-byte accesses are aligned; bounds were
        // checked above and this thread exclusively owns its output quad.
        let mut value = quad_lanes(unsafe { *(residual.as_ptr().add(base) as *const u128) });
        for rank in 0..k {
            let pair = token * k + rank;
            let slot = slots[pair];
            if slot != MOE_DROPPED_SLOT {
                let expert = selected_experts[pair] as usize;
                let input = (expert * c + slot as usize) * d + QUAD_LANES * quad;
                if input + QUAD_LANES > expert_output.len() {
                    return;
                }
                let gate = gate_weights[pair];
                let outputs =
                    quad_lanes(unsafe { *(expert_output.as_ptr().add(input) as *const u128) });
                for lane in 0..QUAD_LANES {
                    value[lane] += gate * outputs[lane];
                }
            }
        }
        unsafe {
            *(output.as_mut_ptr().add(base) as *mut u128) = quad_bits(value);
        }
    }

    /// [`moe_gather_combine_add`] on a packed residual stream, one word per
    /// thread. The arm for a `dim` the quad twin's 16-byte accesses cannot
    /// cover; `dim` must still be even.
    #[kernel]
    pub fn moe_gather_combine_add_bf16(
        expert_output: &[f32],
        selected_experts: &[u32],
        gate_weights: &[f32],
        slots: &[u32],
        residual: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut output: DisjointSlice<u32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if d == 0 || k == 0 || !d.is_multiple_of(2) {
            return;
        }
        let words = d / 2;
        let token = i / words;
        let column = 2 * (i % words);
        if (token + 1) * k > slots.len()
            || (token + 1) * k > selected_experts.len()
            || (token + 1) * k > gate_weights.len()
            || i >= residual.len()
        {
            return;
        }
        let mut value = bf16_halves(residual[i]);
        for rank in 0..k {
            let pair = token * k + rank;
            let slot = slots[pair];
            if slot != MOE_DROPPED_SLOT {
                let expert = selected_experts[pair] as usize;
                let input = (expert * c + slot as usize) * d + column;
                if input + 1 >= expert_output.len() {
                    return;
                }
                let gate = gate_weights[pair];
                value[0] += gate * expert_output[input];
                value[1] += gate * expert_output[input + 1];
            }
        }
        if let Some(slot) = output.get_mut(index) {
            *slot = bf16_pair(value[0], value[1]);
        }
    }

    /// [`moe_gather_combine_add_quad`] on a packed residual stream.
    ///
    /// The expert outputs stay fp32 — they are the down projection's epilogue
    /// target — and the combine still sums in fp32 registers. Only the
    /// residual read and the block-output store are packed, so the block
    /// boundary rounds exactly once.
    #[kernel]
    pub unsafe fn moe_gather_combine_add_quad_bf16(
        expert_output: &[f32],
        selected_experts: &[u32],
        gate_weights: &[f32],
        slots: &[u32],
        residual: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut output: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if d == 0 || k == 0 || !d.is_multiple_of(QUAD_LANES) {
            return;
        }
        let row_quads = d / QUAD_LANES;
        let token = i / row_quads;
        let quad = i % row_quads;
        let base = token * (d / 2) + 2 * quad;
        if base + 1 >= residual.len()
            || base + 1 >= output.len()
            || (token + 1) * k > slots.len()
            || (token + 1) * k > selected_experts.len()
            || (token + 1) * k > gate_weights.len()
        {
            return;
        }
        // SAFETY: `base` is even so the 8-byte residual load and output store
        // are aligned, and every bin row offset is a multiple of `QUAD_LANES`
        // so the 16-byte expert reads are; bounds were checked above and this
        // thread exclusively owns its output quad.
        let packed = unsafe { *(residual.as_ptr().add(base) as *const u64) };
        let mut value = [
            bf16_bits_to_f32(packed as u16),
            bf16_bits_to_f32((packed >> 16) as u16),
            bf16_bits_to_f32((packed >> 32) as u16),
            bf16_bits_to_f32((packed >> 48) as u16),
        ];
        for rank in 0..k {
            let pair = token * k + rank;
            let slot = slots[pair];
            if slot != MOE_DROPPED_SLOT {
                let expert = selected_experts[pair] as usize;
                let input = (expert * c + slot as usize) * d + QUAD_LANES * quad;
                if input + QUAD_LANES > expert_output.len() {
                    return;
                }
                let gate = gate_weights[pair];
                let outputs =
                    quad_lanes(unsafe { *(expert_output.as_ptr().add(input) as *const u128) });
                for lane in 0..QUAD_LANES {
                    value[lane] += gate * outputs[lane];
                }
            }
        }
        let mut store = 0u64;
        for lane in 0..QUAD_LANES {
            store |= (f32_to_bf16_bits(value[lane]) as u64) << (16 * lane);
        }
        unsafe {
            *(output.as_mut_ptr().add(base) as *mut u64) = store;
        }
    }

    /// Scatter `gate * dy` to expert-output order and compute one gate gradient
    /// dot product per accepted pair, one block per `(token, slot)`.
    ///
    /// Lanes stride the `D` row so the bin copy is fully coalesced, and the
    /// gate dot `Σ_d expert_output·dy` reduces in shared memory. The prior
    /// thread-per-pair variant walked the whole row serially on a single lane.
    ///
    /// A `dim` divisible by [`QUAD_LANES`] makes every row base 16-byte
    /// aligned, so each lane moves a whole quad per step instead of one `f32`;
    /// other `dim` take the scalar walk.
    ///
    /// Rows the routing left unassigned are cleared by [`moe_zero_dead_bins`],
    /// not by this kernel.
    #[kernel]
    pub unsafe fn moe_scatter_dy(
        expert_output: &[f32],
        dy: &[f32],
        selected_experts: &[u32],
        gate_weights: &[f32],
        slots: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut expert_output_gradient: DisjointSlice<f32>,
        mut gate_gradients: DisjointSlice<f32>,
    ) {
        static mut DOT: SharedArray<f32, MOE_SCATTER_DY_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != MOE_SCATTER_DY_THREADS {
            return;
        }
        let pair = thread::blockIdx_x() as usize;
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if pair >= selected_experts.len()
            || pair >= gate_weights.len()
            || pair >= slots.len()
            || pair >= gate_gradients.len()
            || d == 0
            || k == 0
        {
            return;
        }

        let slot = slots[pair];
        if slot == MOE_DROPPED_SLOT {
            if tid == 0 {
                // SAFETY: this block exclusively owns `pair`.
                unsafe {
                    *gate_gradients.get_unchecked_mut(pair) = 0.0;
                }
            }
            return;
        }

        let token = pair / k;
        let expert = selected_experts[pair] as usize;
        let bin_base = (expert * c + slot as usize) * d;
        let token_base = token * d;
        if bin_base + d > expert_output.len()
            || bin_base + d > expert_output_gradient.len()
            || token_base + d > dy.len()
        {
            return;
        }

        let gate = gate_weights[pair];
        let mut dot = 0.0f32;
        if d.is_multiple_of(QUAD_LANES) {
            // SAFETY: `bin_base` and `token_base` are multiples of `dim`, hence
            // of `QUAD_LANES`, so both rows are 16-byte aligned inside device
            // allocations; the row bounds were checked above. Each lane owns
            // distinct quads of this block's bin row.
            let dy_row = unsafe { dy.as_ptr().add(token_base) as *const u128 };
            let output_row = unsafe { expert_output.as_ptr().add(bin_base) as *const u128 };
            let gradient_row =
                unsafe { expert_output_gradient.as_mut_ptr().add(bin_base) as *mut u128 };
            let mut quad = tid;
            while quad < d / QUAD_LANES {
                let grad = quad_lanes(unsafe { *dy_row.add(quad) });
                let output = quad_lanes(unsafe { *output_row.add(quad) });
                let mut scaled = [0.0f32; QUAD_LANES];
                for lane in 0..QUAD_LANES {
                    dot += output[lane] * grad[lane];
                    scaled[lane] = gate * grad[lane];
                }
                unsafe {
                    *gradient_row.add(quad) = quad_bits(scaled);
                }
                quad += MOE_SCATTER_DY_THREADS;
            }
        } else {
            let mut column = tid;
            while column < d {
                let grad = dy[token_base + column];
                dot += expert_output[bin_base + column] * grad;
                // SAFETY: each lane owns distinct columns of this block's bin row.
                unsafe {
                    *expert_output_gradient.get_unchecked_mut(bin_base + column) = gate * grad;
                }
                column += MOE_SCATTER_DY_THREADS;
            }
        }
        unsafe {
            DOT[tid] = dot;
        }
        thread::sync_threads();

        let mut stride = MOE_SCATTER_DY_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    DOT[tid] += DOT[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            // SAFETY: this block exclusively owns `pair`.
            unsafe {
                *gate_gradients.get_unchecked_mut(pair) = DOT[0];
            }
        }
    }

    /// [`moe_scatter_dy`] storing the bin gradient as packed-bf16 pairs.
    ///
    /// Both readers of that panel are bf16 tcgen05 operands (#59). The gate dot
    /// product still accumulates in fp32 over the same quads in the same lane
    /// order, so it is bit-identical to the wide kernel wherever `dim` is a
    /// multiple of [`QUAD_LANES`]; only the store narrows.
    #[kernel]
    pub unsafe fn moe_scatter_dy_bf16(
        expert_output: &[f32],
        dy: &[f32],
        selected_experts: &[u32],
        gate_weights: &[f32],
        slots: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut expert_output_gradient: DisjointSlice<u32>,
        mut gate_gradients: DisjointSlice<f32>,
    ) {
        static mut DOT: SharedArray<f32, MOE_SCATTER_DY_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != MOE_SCATTER_DY_THREADS {
            return;
        }
        let pair = thread::blockIdx_x() as usize;
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if pair >= selected_experts.len()
            || pair >= gate_weights.len()
            || pair >= slots.len()
            || pair >= gate_gradients.len()
            || d == 0
            || k == 0
            || !d.is_multiple_of(2)
        {
            return;
        }

        let slot = slots[pair];
        if slot == MOE_DROPPED_SLOT {
            if tid == 0 {
                // SAFETY: this block exclusively owns `pair`.
                unsafe {
                    *gate_gradients.get_unchecked_mut(pair) = 0.0;
                }
            }
            return;
        }

        let token = pair / k;
        let expert = selected_experts[pair] as usize;
        let bin_base = (expert * c + slot as usize) * d;
        let token_base = token * d;
        if bin_base + d > expert_output.len()
            || (bin_base + d) / 2 > expert_output_gradient.len()
            || token_base + d > dy.len()
        {
            return;
        }

        let gate = gate_weights[pair];
        let mut dot = 0.0f32;
        if d.is_multiple_of(QUAD_LANES) {
            // SAFETY: `bin_base` and `token_base` are multiples of `dim`, hence
            // of `QUAD_LANES`, so both fp32 rows are 16-byte aligned and the
            // packed row's `bin_base / 2` base is 8-byte aligned; the row
            // bounds were checked above. Each lane owns distinct quads.
            let dy_row = unsafe { dy.as_ptr().add(token_base) as *const u128 };
            let output_row = unsafe { expert_output.as_ptr().add(bin_base) as *const u128 };
            let gradient_row =
                unsafe { expert_output_gradient.as_mut_ptr().add(bin_base / 2) as *mut u64 };
            let mut quad = tid;
            while quad < d / QUAD_LANES {
                let grad = quad_lanes(unsafe { *dy_row.add(quad) });
                let output = quad_lanes(unsafe { *output_row.add(quad) });
                let mut packed = 0u64;
                for lane in 0..QUAD_LANES {
                    dot += output[lane] * grad[lane];
                    packed |= (f32_to_bf16_bits(gate * grad[lane]) as u64) << (16 * lane);
                }
                unsafe {
                    *gradient_row.add(quad) = packed;
                }
                quad += MOE_SCATTER_DY_THREADS;
            }
        } else {
            let mut word = tid;
            while word < d / 2 {
                let column = 2 * word;
                let low = dy[token_base + column];
                let high = dy[token_base + column + 1];
                dot += expert_output[bin_base + column] * low
                    + expert_output[bin_base + column + 1] * high;
                // SAFETY: each lane owns distinct words of this block's bin row.
                unsafe {
                    *expert_output_gradient.get_unchecked_mut(bin_base / 2 + word) =
                        f32_to_bf16_bits(gate * low) as u32
                            | ((f32_to_bf16_bits(gate * high) as u32) << 16);
                }
                word += MOE_SCATTER_DY_THREADS;
            }
        }
        unsafe {
            DOT[tid] = dot;
        }
        thread::sync_threads();

        let mut stride = MOE_SCATTER_DY_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    DOT[tid] += DOT[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            // SAFETY: this block exclusively owns `pair`.
            unsafe {
                *gate_gradients.get_unchecked_mut(pair) = DOT[0];
            }
        }
    }

    /// Zero the expert-bin gradient rows the routing left unassigned.
    ///
    /// Capacity assignment hands expert `e` the slots `0..min(count[e], C)` and
    /// [`moe_scatter_dy`] overwrites exactly those, so the dead tail
    /// `min(count[e], C)..C` is all that still needs clearing. Together the two
    /// passes cover the whole `E·C·D` buffer, replacing a full pre-fill.
    ///
    /// Blocks walk that tail and nothing else: one block per `(expert, slot)`
    /// dispatched `E · C` of them — 590K per step across the layers — to
    /// discover that a balanced router leaves almost every slot live, and cost
    /// 0.45 ms doing it (#99). The expert is the grid's second dimension, so a
    /// block reads its count once and then strides only dead slots; lanes
    /// stride the row.
    #[kernel]
    pub unsafe fn moe_zero_dead_bins(
        assignment_counts: &[u32],
        dim: u32,
        capacity: u32,
        mut expert_output_gradient: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let threads = thread::blockDim_x() as usize;
        let expert = thread::blockIdx_y() as usize;
        let d = dim as usize;
        let c = capacity as usize;
        if d == 0 || c == 0 || expert >= assignment_counts.len() {
            return;
        }
        let live = (assignment_counts[expert] as usize).min(c);
        let mut slot = live + thread::blockIdx_x() as usize;
        while slot < c {
            let base = (expert * c + slot) * d;
            if base + d > expert_output_gradient.len() {
                return;
            }
            if d.is_multiple_of(QUAD_LANES) {
                // SAFETY: `base` is a multiple of `dim`, hence of `QUAD_LANES`,
                // so the row is 16-byte aligned; bounds were checked above.
                // Each lane owns distinct quads of this block's dead bin row.
                let row = unsafe { expert_output_gradient.as_mut_ptr().add(base) as *mut u128 };
                let mut quad = tid;
                while quad < d / QUAD_LANES {
                    unsafe {
                        *row.add(quad) = 0;
                    }
                    quad += threads;
                }
            } else {
                let mut column = tid;
                while column < d {
                    // SAFETY: each lane owns distinct columns of this dead bin
                    // row.
                    unsafe {
                        *expert_output_gradient.get_unchecked_mut(base + column) = 0.0;
                    }
                    column += threads;
                }
            }
            slot += thread::gridDim_x() as usize;
        }
    }

    /// [`moe_zero_dead_bins`] over a packed-bf16 gradient panel (#59).
    #[kernel]
    pub unsafe fn moe_zero_dead_bins_bf16(
        assignment_counts: &[u32],
        dim: u32,
        capacity: u32,
        mut expert_output_gradient: DisjointSlice<u32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let threads = thread::blockDim_x() as usize;
        let expert = thread::blockIdx_y() as usize;
        let d = dim as usize;
        let c = capacity as usize;
        if d == 0 || c == 0 || !d.is_multiple_of(2) || expert >= assignment_counts.len() {
            return;
        }
        let words = d / 2;
        let live = (assignment_counts[expert] as usize).min(c);
        let mut slot = live + thread::blockIdx_x() as usize;
        while slot < c {
            let base = (expert * c + slot) * words;
            if base + words > expert_output_gradient.len() {
                return;
            }
            if words.is_multiple_of(QUAD_LANES) {
                // SAFETY: `base` is a multiple of `words`, hence of
                // `QUAD_LANES`, so the row is 16-byte aligned; bounds were
                // checked above. Each lane owns distinct quads of this block's
                // dead bin row.
                let row = unsafe { expert_output_gradient.as_mut_ptr().add(base) as *mut u128 };
                let mut quad = tid;
                while quad < words / QUAD_LANES {
                    unsafe {
                        *row.add(quad) = 0;
                    }
                    quad += threads;
                }
            } else {
                let mut column = tid;
                while column < words {
                    // SAFETY: each lane owns distinct words of this dead bin
                    // row.
                    unsafe {
                        *expert_output_gradient.get_unchecked_mut(base + column) = 0;
                    }
                    column += threads;
                }
            }
            slot += thread::gridDim_x() as usize;
        }
    }

    /// Gather expert-input gradients back to token order, summing top-k paths.
    #[kernel]
    pub fn moe_gather_dx(
        expert_input_gradient: &[f32],
        selected_experts: &[u32],
        slots: &[u32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut dx: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if i >= dx.len() || d == 0 || k == 0 {
            return;
        }
        let token = i / d;
        let column = i % d;
        let mut value = 0.0f32;
        for rank in 0..k {
            let pair = token * k + rank;
            if pair >= selected_experts.len() || pair >= slots.len() {
                return;
            }
            let slot = slots[pair];
            if slot != MOE_DROPPED_SLOT {
                let expert = selected_experts[pair] as usize;
                let input = (expert * c + slot as usize) * d + column;
                if input >= expert_input_gradient.len() {
                    return;
                }
                value += expert_input_gradient[input];
            }
        }
        if let Some(slot) = dx.get_mut(index) {
            *slot = value;
        }
    }

    /// [`moe_gather_dx`] with the router input-gradient add folded in: each
    /// output element is `router_dx + Σ_k expert_input_gradient`, so the
    /// separate `[N, D]` combine pass and the intermediate it read never run.
    #[kernel]
    pub fn moe_gather_dx_add(
        expert_input_gradient: &[f32],
        selected_experts: &[u32],
        slots: &[u32],
        router_dx: &[f32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut dx: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if i >= dx.len() || i >= router_dx.len() || d == 0 || k == 0 {
            return;
        }
        let token = i / d;
        let column = i % d;
        let mut value = router_dx[i];
        for rank in 0..k {
            let pair = token * k + rank;
            if pair >= selected_experts.len() || pair >= slots.len() {
                return;
            }
            let slot = slots[pair];
            if slot != MOE_DROPPED_SLOT {
                let expert = selected_experts[pair] as usize;
                let input = (expert * c + slot as usize) * d + column;
                if input >= expert_input_gradient.len() {
                    return;
                }
                value += expert_input_gradient[input];
            }
        }
        if let Some(slot) = dx.get_mut(index) {
            *slot = value;
        }
    }

    /// [`moe_gather_dx_add`] moving one 16-byte quad per access. `dim` must
    /// be a multiple of [`QUAD_LANES`].
    #[kernel]
    pub unsafe fn moe_gather_dx_add_quad(
        expert_input_gradient: &[f32],
        selected_experts: &[u32],
        slots: &[u32],
        router_dx: &[f32],
        dim: u32,
        top_k: u32,
        capacity: u32,
        mut dx: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let d = dim as usize;
        let k = top_k as usize;
        let c = capacity as usize;
        if d == 0 || k == 0 || !d.is_multiple_of(QUAD_LANES) {
            return;
        }
        let row_quads = d / QUAD_LANES;
        let token = i / row_quads;
        let quad = i % row_quads;
        let base = token * d + QUAD_LANES * quad;
        if base + QUAD_LANES > router_dx.len()
            || base + QUAD_LANES > dx.len()
            || (token + 1) * k > slots.len()
            || (token + 1) * k > selected_experts.len()
        {
            return;
        }
        // SAFETY: `base` and every bin row offset are multiples of
        // `QUAD_LANES`, so all 16-byte accesses are aligned; bounds were
        // checked above and this thread exclusively owns its output quad.
        let mut value = quad_lanes(unsafe { *(router_dx.as_ptr().add(base) as *const u128) });
        for rank in 0..k {
            let pair = token * k + rank;
            let slot = slots[pair];
            if slot != MOE_DROPPED_SLOT {
                let expert = selected_experts[pair] as usize;
                let input = (expert * c + slot as usize) * d + QUAD_LANES * quad;
                if input + QUAD_LANES > expert_input_gradient.len() {
                    return;
                }
                let gradients = quad_lanes(unsafe {
                    *(expert_input_gradient.as_ptr().add(input) as *const u128)
                });
                for lane in 0..QUAD_LANES {
                    value[lane] += gradients[lane];
                }
            }
        }
        unsafe {
            *(dx.as_mut_ptr().add(base) as *mut u128) = quad_bits(value);
        }
    }

    /// Backward through selected-probability renormalization and router
    /// softmax, including the Switch-style auxiliary loss gradient.
    #[kernel]
    pub unsafe fn router_backward(
        probabilities: &[f32],
        selected_experts: &[u32],
        gate_weights: &[f32],
        gate_gradients: &[f32],
        assignment_counts: &[u32],
        tokens: u32,
        experts: u32,
        top_k: u32,
        aux_coefficient: f32,
        mut dlogits: DisjointSlice<f32>,
    ) {
        let token = thread::index_1d().get();
        let n = tokens as usize;
        let e = experts as usize;
        let k = top_k as usize;
        if token >= n
            || e == 0
            || k == 0
            || token * e + e > probabilities.len()
            || token * e + e > dlogits.len()
            || token * k + k > selected_experts.len()
            || token * k + k > gate_weights.len()
            || token * k + k > gate_gradients.len()
            || e > assignment_counts.len()
        {
            return;
        }

        let mut weighted_gate_gradient = 0.0f32;
        let mut selected_probability_sum = 0.0f32;
        for rank in 0..k {
            let pair = token * k + rank;
            let expert = selected_experts[pair] as usize;
            weighted_gate_gradient += gate_gradients[pair] * gate_weights[pair];
            selected_probability_sum += probabilities[token * e + expert];
        }

        let mut softmax_dot = 0.0f32;
        for expert in 0..e {
            let mut probability_gradient = 0.0f32;
            for rank in 0..k {
                let pair = token * k + rank;
                if selected_experts[pair] as usize == expert {
                    probability_gradient +=
                        (gate_gradients[pair] - weighted_gate_gradient) / selected_probability_sum;
                }
            }
            let assignment_fraction = assignment_counts[expert] as f32 / (n * k) as f32;
            probability_gradient += aux_coefficient * e as f32 * assignment_fraction / n as f32;
            softmax_dot += probabilities[token * e + expert] * probability_gradient;
        }

        for expert in 0..e {
            let mut probability_gradient = 0.0f32;
            for rank in 0..k {
                let pair = token * k + rank;
                if selected_experts[pair] as usize == expert {
                    probability_gradient +=
                        (gate_gradients[pair] - weighted_gate_gradient) / selected_probability_sum;
                }
            }
            let assignment_fraction = assignment_counts[expert] as f32 / (n * k) as f32;
            probability_gradient += aux_coefficient * e as f32 * assignment_fraction / n as f32;
            unsafe {
                *dlogits.get_unchecked_mut(token * e + expert) =
                    probabilities[token * e + expert] * (probability_gradient - softmax_dot);
            }
        }
    }

    /// Router linear backward with respect to its input:
    /// `dx[N,D] = dlogits[N,E] x weight[D,E]^T`.
    ///
    /// A block owns a `[ROUTER_INPUT_TOKENS, ROUTER_INPUT_BN]` output tile. Its
    /// slice of the `[D,E]` weight is read once into registers — each lane keeps
    /// the `[E]` weight rows of its `ROUTER_INPUT_COLUMNS` columns — and reused
    /// across the whole token sweep, so the weight is read `N /
    /// ROUTER_INPUT_TOKENS` times instead of once per token. Registers rather
    /// than shared memory hold the staged weight because a lane's columns are
    /// private to it: the shared tile would just be the same values with an
    /// `LDS` in the inner loop. Lane-major columns keep every `dx` store a full
    /// coalesced sector, which is what the write-bound tile is paced by.
    #[kernel]
    pub unsafe fn router_backward_input(
        dlogits: &[f32],
        weight: &[f32],
        experts: u32,
        mut dx: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != ROUTER_INPUT_THREADS {
            return;
        }
        let e = experts as usize;
        if e == 0 || e > ROUTER_MAX_EXPERTS || !weight.len().is_multiple_of(e) {
            return;
        }
        let d = weight.len() / e;
        if d == 0 || !dx.len().is_multiple_of(d) {
            return;
        }
        let n = dx.len() / d;
        let token_base = thread::blockIdx_y() as usize * ROUTER_INPUT_TOKENS;
        let column_base = thread::blockIdx_x() as usize * ROUTER_INPUT_BN + tid;
        if token_base >= n || n * e > dlogits.len() {
            return;
        }

        let mut staged = [[0.0f32; ROUTER_MAX_EXPERTS]; ROUTER_INPUT_COLUMNS];
        let mut slot = 0usize;
        while slot < ROUTER_INPUT_COLUMNS {
            let column = column_base + slot * ROUTER_INPUT_THREADS;
            let mut expert = 0usize;
            while expert < ROUTER_MAX_EXPERTS {
                staged[slot][expert] = if column < d && expert < e {
                    weight[column * e + expert]
                } else {
                    0.0
                };
                expert += 1;
            }
            slot += 1;
        }

        let token_end = (token_base + ROUTER_INPUT_TOKENS).min(n);
        let mut token = token_base;
        while token < token_end {
            // The gate row is warp-uniform, so this is one broadcast load per
            // expert for the whole token, amortized over the register tile.
            let mut gates = [0.0f32; ROUTER_MAX_EXPERTS];
            let mut expert = 0usize;
            while expert < ROUTER_MAX_EXPERTS {
                gates[expert] = if expert < e {
                    dlogits[token * e + expert]
                } else {
                    0.0
                };
                expert += 1;
            }

            let row_base = token * d;
            let mut slot = 0usize;
            while slot < ROUTER_INPUT_COLUMNS {
                let column = column_base + slot * ROUTER_INPUT_THREADS;
                if column < d {
                    let mut value = 0.0f32;
                    let mut expert = 0usize;
                    while expert < ROUTER_MAX_EXPERTS {
                        value += gates[expert] * staged[slot][expert];
                        expert += 1;
                    }
                    unsafe {
                        *dx.get_unchecked_mut(row_base + column) = value;
                    }
                }
                slot += 1;
            }
            token += 1;
        }
    }

    /// Router linear backward with respect to its weight.
    #[kernel]
    pub fn router_backward_weight(
        x: &[f32],
        dlogits: &[f32],
        tokens: u32,
        experts: u32,
        mut dweight: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let n = tokens as usize;
        let e = experts as usize;
        if i >= dweight.len() || n == 0 || e == 0 {
            return;
        }
        let d = dweight.len() / e;
        let column = i / e;
        let expert = i % e;
        let mut value = 0.0f32;
        for token in 0..n {
            value += x[token * d + column] * dlogits[token * e + expert];
        }
        if let Some(slot) = dweight.get_mut(index) {
            *slot += value;
        }
    }

    /// One contiguous token partition of the router weight gradient
    /// `dweight[D,E] = x[N,D]^T dlogits[N,E]`, written to
    /// `partials[SPLITS, E, D]` for `router_backward_weight_merge` to sum.
    ///
    /// `D*E` is far too few outputs to fill the machine on its own, so the
    /// token dimension is split across blocks. A block owns
    /// `ROUTER_WGRAD_BM` model rows of one partition, lane-major so each `x`
    /// read is a full coalesced sector, and each lane keeps `[E]` accumulators
    /// per owned row in registers: one `x` load feeds `E` FMAs and the gate row
    /// is a warp-uniform broadcast shared by the whole register tile.
    ///
    /// The reduction order is fixed: a lane owns its outputs alone and sums its
    /// partition in ascending token order, and the merge sums partitions in
    /// ascending order. No lane, block, or launch ordering can perturb it.
    #[kernel]
    pub unsafe fn router_backward_weight_split(
        x: &[f32],
        dlogits: &[f32],
        tokens: u32,
        experts: u32,
        dim: u32,
        mut partials: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != ROUTER_WGRAD_THREADS {
            return;
        }
        let n = tokens as usize;
        let e = experts as usize;
        let d = dim as usize;
        if e == 0 || e > ROUTER_MAX_EXPERTS || d == 0 || n * d > x.len() || n * e > dlogits.len() {
            return;
        }
        let split = thread::blockIdx_y() as usize;
        let row_base = thread::blockIdx_x() as usize * ROUTER_WGRAD_BM + tid;
        if (split + 1) * e * d > partials.len() {
            return;
        }

        let partition = n.div_ceil(ROUTER_WGRAD_SPLITS);
        let token_end = ((split + 1) * partition).min(n);
        let mut token = (split * partition).min(n);

        let mut accumulators = [[0.0f32; ROUTER_MAX_EXPERTS]; ROUTER_WGRAD_ROWS];
        while token < token_end {
            let mut gates = [0.0f32; ROUTER_MAX_EXPERTS];
            let mut expert = 0usize;
            while expert < ROUTER_MAX_EXPERTS {
                gates[expert] = if expert < e {
                    dlogits[token * e + expert]
                } else {
                    0.0
                };
                expert += 1;
            }

            let row_offset = token * d;
            let mut slot = 0usize;
            while slot < ROUTER_WGRAD_ROWS {
                let row = row_base + slot * ROUTER_WGRAD_THREADS;
                let value = if row < d { x[row_offset + row] } else { 0.0 };
                let mut expert = 0usize;
                while expert < ROUTER_MAX_EXPERTS {
                    accumulators[slot][expert] += value * gates[expert];
                    expert += 1;
                }
                slot += 1;
            }
            token += 1;
        }

        let mut slot = 0usize;
        while slot < ROUTER_WGRAD_ROWS {
            let row = row_base + slot * ROUTER_WGRAD_THREADS;
            if row < d {
                let mut expert = 0usize;
                while expert < e {
                    unsafe {
                        *partials.get_unchecked_mut((split * e + expert) * d + row) =
                            accumulators[slot][expert];
                    }
                    expert += 1;
                }
            }
            slot += 1;
        }
    }

    /// [`router_backward_weight_split`] reading its saved input packed.
    ///
    /// The partition boundaries, the per-lane register tile and the ascending
    /// reduction order are the twin's, so this stays as deterministic as it
    /// was; only the `x` load widens.
    #[kernel]
    pub unsafe fn router_backward_weight_split_bf16(
        x: &[u32],
        dlogits: &[f32],
        tokens: u32,
        experts: u32,
        dim: u32,
        mut partials: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != ROUTER_WGRAD_THREADS {
            return;
        }
        let n = tokens as usize;
        let e = experts as usize;
        let d = dim as usize;
        if e == 0
            || e > ROUTER_MAX_EXPERTS
            || d == 0
            || n * d > x.len() * 2
            || n * e > dlogits.len()
        {
            return;
        }
        let split = thread::blockIdx_y() as usize;
        let row_base = thread::blockIdx_x() as usize * ROUTER_WGRAD_BM + tid;
        if (split + 1) * e * d > partials.len() {
            return;
        }

        let partition = n.div_ceil(ROUTER_WGRAD_SPLITS);
        let token_end = ((split + 1) * partition).min(n);
        let mut token = (split * partition).min(n);

        let mut accumulators = [[0.0f32; ROUTER_MAX_EXPERTS]; ROUTER_WGRAD_ROWS];
        while token < token_end {
            let mut gates = [0.0f32; ROUTER_MAX_EXPERTS];
            let mut expert = 0usize;
            while expert < ROUTER_MAX_EXPERTS {
                gates[expert] = if expert < e {
                    dlogits[token * e + expert]
                } else {
                    0.0
                };
                expert += 1;
            }

            let row_offset = token * d;
            let mut slot = 0usize;
            while slot < ROUTER_WGRAD_ROWS {
                let row = row_base + slot * ROUTER_WGRAD_THREADS;
                let value = if row < d {
                    bf16_at(x, row_offset + row)
                } else {
                    0.0
                };
                let mut expert = 0usize;
                while expert < ROUTER_MAX_EXPERTS {
                    accumulators[slot][expert] += value * gates[expert];
                    expert += 1;
                }
                slot += 1;
            }
            token += 1;
        }

        let mut slot = 0usize;
        while slot < ROUTER_WGRAD_ROWS {
            let row = row_base + slot * ROUTER_WGRAD_THREADS;
            if row < d {
                let mut expert = 0usize;
                while expert < e {
                    unsafe {
                        *partials.get_unchecked_mut((split * e + expert) * d + row) =
                            accumulators[slot][expert];
                    }
                    expert += 1;
                }
            }
            slot += 1;
        }
    }

    /// Sum the router weight-gradient token partitions in ascending order and
    /// accumulate into `dweight[D,E]`. Threads walk `partials` expert-major so
    /// the wide read is coalesced; the narrow `dweight` update is not, and does
    /// not matter at `D*E` elements.
    #[kernel]
    pub unsafe fn router_backward_weight_merge(
        partials: &[f32],
        experts: u32,
        mut dweight: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let e = experts as usize;
        if e == 0 || !dweight.len().is_multiple_of(e) || i >= dweight.len() {
            return;
        }
        let d = dweight.len() / e;
        let expert = i / d;
        let row = i % d;
        if ROUTER_WGRAD_SPLITS * e * d > partials.len() {
            return;
        }

        let mut value = 0.0f32;
        let mut split = 0usize;
        while split < ROUTER_WGRAD_SPLITS {
            value += partials[(split * e + expert) * d + row];
            split += 1;
        }
        unsafe {
            *dweight.get_unchecked_mut(row * e + expert) += value;
        }
    }
}
