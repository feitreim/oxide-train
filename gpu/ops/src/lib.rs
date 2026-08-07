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
    cuda_module, kernel, launch_bounds, thread,
};

use kittens::global::{GlobalRows, load_cols, load_rows, store_rows};
use kittens::reg::{BaseLdtm, ColVec, RegTile, RegVec};
use kittens::shared::{Bf16, F32};
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

/// Rows one fused RMSNorm-backward block owns.
///
/// The block carries a shared column-partial accumulator for the weight
/// gradient across all of them and pays one atomic per column at the end, so
/// this is the whole trade: fewer rows is more blocks to fill the device and
/// proportionally more atomics on the same `dim` addresses. Occupancy is
/// register-bound at 4 blocks/SM either way, so this only buys waves against
/// atomics — and it is not monotone. Paired trainer runs at B=16 measure
/// +2.76% at 8, **+3.59% at 16** and +2.24% at 32, on baselines 0.27% apart.
/// 16 leaves 2048 blocks at the training shape, where the split kernel it
/// replaces launched 32768 one-row blocks.
pub const NORM_BACKWARD_ROWS_PER_BLOCK: usize = 16;

/// Widest row the fused RMSNorm kernels carry in shared memory.
///
/// The backward's column partials and the forward pair's staged row are both
/// sized by it, and it is the one shape condition on the fused arms — the
/// split kernels stay as what anything wider takes.
pub const NORM_MAX_COLUMNS: usize = 4096;

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
/// Router logits GEMM reduction tile, in the [`reference`] kernel.
pub const ROUTER_GEMM_BK: usize = 16;
/// Threads in a router logits GEMM block: one lane per output element.
pub const ROUTER_GEMM_THREADS: usize = ROUTER_GEMM_BM * ROUTER_GEMM_BN;

/// Reduction tile of the *shipped* router logits GEMM.
///
/// This is the only thing that puts more than one `x` load in flight per lane.
/// A block stages `ROUTER_GEMM_BM * ROUTER_LOGITS_BK` elements between two
/// barriers, so a lane carries `BM * BK / THREADS` of them at once and the
/// whole reduction costs `2 * dim / BK` barriers. At the reference's 16 that is
/// **eight bytes a lane and 384 barriers a block** at the training shape, which
/// is the whole of why this kernel measured 453 GB/s — 6% of HBM — while
/// reading nothing but a `[N, D]` stream. 128 is 64 bytes a lane and 48
/// barriers, and it costs 20.6 KiB of shared memory, which is still eight
/// blocks an SM.
pub const ROUTER_LOGITS_BK: usize = 128;

/// Row stride of both router logits GEMM tiles, in `f32`.
///
/// The pad is what makes the tile reads conflict-free. A warp spans four token
/// rows of the `x` tile and all eight expert columns of the weight tile, and
/// both strides would otherwise be a multiple of the 32 banks — every one of
/// those rows landing on the same four banks at a different address. One
/// [`QUAD_LANES`] vector of pad rotates each successive row by four banks, so
/// the four rows a warp reads cover sixteen distinct banks and the eight
/// expert columns cover all thirty-two.
pub const ROUTER_LOGITS_STRIDE: usize = ROUTER_LOGITS_BK + QUAD_LANES;

/// Token rows one staging pass of the router logits GEMM covers.
///
/// The staging map is the transpose of the accumulating one: a warp covers one
/// tile row, so its 32 vector loads are one contiguous 256-byte run of `x`,
/// where the inner product wants a lane per expert.
pub const ROUTER_LOGITS_STAGE_ROWS: usize = ROUTER_GEMM_THREADS * QUAD_LANES / ROUTER_LOGITS_BK;

/// Staging passes one block makes over its `x` tile per reduction step.
pub const ROUTER_LOGITS_STAGES: usize = ROUTER_GEMM_BM / ROUTER_LOGITS_STAGE_ROWS;

/// Staging passes one block makes over its weight tile per reduction step.
pub const ROUTER_LOGITS_WEIGHT_STAGES: usize =
    ROUTER_LOGITS_BK * ROUTER_MAX_EXPERTS / ROUTER_GEMM_THREADS;

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
/// Model rows each router weight-gradient lane owns: both halves of one packed
/// `x` word, so a warp's load is a full 128-byte sector and the lane holds
/// `ROWS * E` accumulators in registers.
pub const ROUTER_WGRAD_ROWS: usize = 2;
/// Model rows one router weight-gradient block owns.
pub const ROUTER_WGRAD_BM: usize = ROUTER_WGRAD_THREADS * ROUTER_WGRAD_ROWS;
/// Tokens a lane loads before it multiplies any of them.
///
/// The token loop's trip count is a runtime value NVVM will not unroll, so this
/// is the only thing putting more than one `x` load in flight per lane — the
/// same fact `MOE_PROBABILITY_SUMS_THREADS` records, and the reason this kernel
/// used to run at a tenth of memory rate.
///
/// The staged values land in a `.local` depot because the multiply loop that
/// reads them stays rolled, and that is the faster arrangement: `#[unroll]`ing
/// it folds the depot away into 128 straight-line FMAs and measured **4.3205 ms
/// against 3.1397** over the step's twelve launches. The depot is L1-resident
/// and the registers it saves are what keep the warps that hide the loads.
pub const ROUTER_WGRAD_TOKENS: usize = 8;
/// Contiguous token partitions the router weight gradient is split into. Each
/// partition is summed by one block and the partitions are merged in ascending
/// order, which fixes the reduction order independently of block scheduling.
///
/// It also sizes the `[SPLITS, E, D]` partial buffer the merge reads back, so
/// it is worth no more than it buys: `D * SPLITS * TOKENS` loads in flight is
/// already past memory rate at 64, where 256 cost four times the partials.
pub const ROUTER_WGRAD_SPLITS: usize = 64;

/// Sentinel written by deterministic MoE binning for a capacity-dropped pair.
pub const MOE_DROPPED_SLOT: u32 = u32::MAX;

/// Threads in one block-per-pair MoE backward scatter. Lanes stride the `D`
/// gradient row for a coalesced copy and a tree-reduced gate dot. Must remain a
/// power of two.
pub const MOE_SCATTER_DY_THREADS: usize = 256;

/// Threads in one block-per-expert auxiliary-loss term reduction. Lanes stride
/// the tokens before a tree reduction, so this must remain a power of two.
///
/// One block per expert is only `E` blocks, so the token loop's depth is all
/// the parallelism there is: `N` runs to 24576 and the trip count is a runtime
/// value NVVM will not unroll, which leaves each lane one load in flight and
/// the launch `N / threads` load latencies deep. A full block is 24 of them
/// rather than 96 (#99).
pub const MOE_AUX_TERMS_THREADS: usize = 1024;

/// Threads in the single-block loss tail. Lanes stride the per-token losses
/// before a tree reduction, so this must remain a power of two.
pub const LOSS_TAIL_THREADS: usize = 1024;

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

/// Columns a tile-RMSNorm *backward* warp holds in registers at once.
///
/// Half [`NORM_TILE_CHUNK`], and for the reason that constant's own note gives
/// for stopping at 64: the forward carries one band, and this kernel carries
/// three at once — the saved input, the upstream gradient and the residual it
/// adds — so the same register file buys a third of the columns. 64 here is
/// the `[16, 96]` cliff again, one band earlier.
pub const NORM_BACKWARD_TILE_CHUNK: usize = 32;

/// One warp's slice of a backward row band, [`NORM_BACKWARD_TILE_CHUNK`] wide.
type NormBackChunk = RegTile<NORM_TILE_ROWS, NORM_BACKWARD_TILE_CHUNK, BaseLdtm>;

/// The backward chunk's slice of the `[dim]` weight — and, on the way out, of
/// its gradient: the one statistic in this kernel that runs down a column
/// rather than across a row.
type NormBackColumns = ColVec<NORM_BACKWARD_TILE_CHUNK, BaseLdtm>;

/// One warp's slice of a SwiGLU row band.
type SwigluChunk = RegTile<SWIGLU_TILE_ROWS, SWIGLU_TILE_CHUNK, BaseLdtm>;

/// The kernels no training step launches: the correctness baseline the gates
/// compare the ones below against.
pub mod reference;

/// Resident blocks an SM every entry point below declares, beside its block
/// width, in `#[launch_bounds]`.
///
/// The block width is #122's fix: an entry point that declares no `.maxntid`
/// lets the driver's JIT *derive* the launchable block from whatever
/// allocation it chose, and a derived block narrower than the launch is a
/// `701` rather than a slow kernel. Every kernel here derived **1024**, which
/// is safe today only by accident.
///
/// The second number is not optional, and that is what pass two measured. A
/// declared `.maxntid` is an input to ptxas' heuristics and not only the
/// register budget's divisor: at a bare `#[launch_bounds(256)]` the budget goes
/// from the derived 1024's 64 registers to 255, and the allocator spends it —
/// `router_backward_weight_split_bf16` went 32 registers and 8 resident blocks
/// to **93 and 2**, and its span with it. So the count that ships is the one
/// that hands ptxas back the budget the derived `.maxntid` gave it,
/// `65536 / (threads * 64)`: 4 for a 256-thread block, 1 for 1024, 6 for the
/// 128-thread tile norm. Every kernel keeps its unpinned allocation under it,
/// with one exception: `router_backward_weight_split_bf16` loses at every
/// target a 256-thread block has, and its own note carries the four
/// measurements that say so.
///
/// Tighter is not better either: `(256, 8)` caps the budget at the 32 registers
/// three of these kernels already used, and all three still acquired a local
/// frame — the harder unrolling a declared 256 buys has to go somewhere.
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

    /// One 8-byte packed load as [`QUAD_LANES`] `f32` — the packed twin of
    /// [`quad_lanes`], and what a bf16 panel costs half the bytes to fill.
    #[inline(always)]
    fn bf16_quad(packed: u64) -> [f32; QUAD_LANES] {
        [
            bf16_bits_to_f32(packed as u16),
            bf16_bits_to_f32((packed >> 16) as u16),
            bf16_bits_to_f32((packed >> 32) as u16),
            bf16_bits_to_f32((packed >> 48) as u16),
        ]
    }

    /// One 16-byte vector load as the [`QUAD_LANES`] packed words it holds —
    /// [`quad_lanes`] for a stream that stays packed rather than widening.
    #[inline(always)]
    fn quad_words(bits: u128) -> [u32; QUAD_LANES] {
        [
            bits as u32,
            (bits >> 32) as u32,
            (bits >> 64) as u32,
            (bits >> 96) as u32,
        ]
    }

    /// The inverse of [`quad_words`]: [`QUAD_LANES`] packed words as one
    /// 16-byte vector store.
    #[inline(always)]
    fn quad_word_bits(words: [u32; QUAD_LANES]) -> u128 {
        let mut bits = 0u128;
        let mut lane = 0usize;
        while lane < QUAD_LANES {
            bits |= (words[lane] as u128) << (32 * lane);
            lane += 1;
        }
        bits
    }

    /// The inverse of [`bf16_quad`]: [`QUAD_LANES`] `f32` as one 8-byte store.
    #[inline(always)]
    fn bf16_quad_bits(lanes: [f32; QUAD_LANES]) -> u64 {
        let mut packed = 0u64;
        for lane in 0..QUAD_LANES {
            packed |= (f32_to_bf16_bits(lanes[lane]) as u64) << (16 * lane);
        }
        packed
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
    #[launch_bounds(256, 4)]
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

    /// [`rms_norm_backward_x_fast`] reading its saved input packed.
    ///
    /// Only `x` changes dtype: the incoming and outgoing gradients are
    /// backward temporaries and stay fp32, and both reductions still
    /// accumulate in fp32 registers.
    #[kernel]
    #[launch_bounds(256, 4)]
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

    /// [`rms_norm_backward_weight_fast`] reading its saved input packed.
    ///
    /// # Safety
    ///
    /// Same contract as [`rms_norm_backward_weight_fast`], and `dim` must be
    /// even so a row starts on a word boundary.
    #[kernel]
    #[launch_bounds(256, 4)]
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

    /// The whole RMSNorm backward in one launch: the input gradient, the
    /// residual branch's add, the weight gradient, and — when the caller
    /// supplies a destination — the forward's normalized output recomputed
    /// for a consumer that would otherwise have kept it saved all step.
    ///
    /// The split form reads `x` and `dy` three times (twice for the input
    /// gradient, once more for the weight gradient), writes a row-factor
    /// buffer only to read it back, and then a separate `add` reads and
    /// rewrites the whole gradient. Here a block owns
    /// [`NORM_BACKWARD_ROWS_PER_BLOCK`] rows and takes each of them apart in
    /// one go: reduce the row, then immediately spend the same values on all
    /// four outputs while they are still in cache.
    ///
    /// The weight gradient is what forces the shape. A column's partial sum
    /// spans rows, so it cannot live in the register of a lane that owns one
    /// row's column — it lives in [`WEIGHT_PARTIALS`], one shared slot per
    /// column, disjointly owned by the lane that strides onto it. Rows per
    /// block is then purely how many atomics the pass costs: one per column
    /// per block, paid once at the end.
    ///
    /// # Safety
    ///
    /// `dim` must be even so a row starts on a word boundary, and at most
    /// [`NORM_MAX_COLUMNS`] so the column partials fit; `dweight` must hold
    /// `dim` accumulators this launch may atomically update.
    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn rms_norm_backward_fused_bf16(
        x: &[u32],
        weight: &[f32],
        dy: &[f32],
        residual: &[f32],
        eps: f32,
        rows: u32,
        dim: u32,
        mut dx: DisjointSlice<f32>,
        mut dweight: DisjointSlice<f32>,
        mut normalized: DisjointSlice<u32>,
    ) {
        static mut SUM_SQ: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;
        static mut DOT: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;
        static mut WEIGHT_PARTIALS: SharedArray<f32, NORM_MAX_COLUMNS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != NORM_THREADS {
            return;
        }
        let d = dim as usize;
        if d == 0 || !d.is_multiple_of(2) || d > NORM_MAX_COLUMNS {
            return;
        }
        let words = d / 2;
        let row_start = thread::blockIdx_x() as usize * NORM_BACKWARD_ROWS_PER_BLOCK;
        let row_end = (row_start + NORM_BACKWARD_ROWS_PER_BLOCK).min(rows as usize);
        if row_start >= row_end {
            return;
        }
        let elements = row_end * d;
        if elements > 2 * x.len()
            || elements > dy.len()
            || elements > residual.len()
            || elements > dx.len()
            || d > weight.len()
            || d > dweight.len()
        {
            return;
        }
        // An absent destination is how the caller says it kept the forward's
        // output instead; every site that recomputes hands over a whole one.
        let recompute = 2 * normalized.len() >= elements;

        let mut column = tid;
        while column < d {
            unsafe {
                WEIGHT_PARTIALS[column] = 0.0;
            }
            column += NORM_THREADS;
        }
        thread::sync_threads();

        for row in row_start..row_end {
            let base = row * d;
            let word_base = row * words;

            let mut sum_sq = 0.0f32;
            let mut dot = 0.0f32;
            let mut word = tid;
            while word < words {
                let [low, high] = bf16_halves(x[word_base + word]);
                sum_sq += low * low + high * high;
                dot += dy[base + 2 * word] * weight[2 * word] * low
                    + dy[base + 2 * word + 1] * weight[2 * word + 1] * high;
                word += NORM_THREADS;
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
                }
            }
            thread::sync_threads();

            let row_inv = unsafe { SUM_SQ[0] };
            let correction = unsafe { DOT[0] };
            word = tid;
            while word < words {
                let low_column = 2 * word;
                let high_column = low_column + 1;
                let low_weight = weight[low_column];
                let high_weight = weight[high_column];
                let [low, high] = bf16_halves(x[word_base + word]);
                let low_dy = dy[base + low_column];
                let high_dy = dy[base + high_column];
                // SAFETY: each lane owns distinct words of this row and the
                // column partials the same striding gave it.
                unsafe {
                    *dx.get_unchecked_mut(base + low_column) = residual[base + low_column]
                        + low_dy * low_weight * row_inv
                        - low * correction;
                    *dx.get_unchecked_mut(base + high_column) = residual[base + high_column]
                        + high_dy * high_weight * row_inv
                        - high * correction;
                    if recompute {
                        *normalized.get_unchecked_mut(word_base + word) =
                            bf16_pair(low * row_inv * low_weight, high * row_inv * high_weight);
                    }
                    WEIGHT_PARTIALS[low_column] += low_dy * low * row_inv;
                    WEIGHT_PARTIALS[high_column] += high_dy * high * row_inv;
                }
                word += NORM_THREADS;
            }
            // The next row's reduction overwrites this one's partials.
            thread::sync_threads();
        }

        column = tid;
        while column < d {
            // SAFETY: `column` was bounds-checked and every access to this
            // location in this kernel is atomic. Stream ordering covers the
            // preceding zero/accumulation state and subsequent optimizer read.
            let slot = unsafe { DeviceAtomicF32::from_ptr(dweight.as_mut_ptr().add(column)) };
            slot.fetch_add(unsafe { WEIGHT_PARTIALS[column] }, AtomicOrdering::Relaxed);
            column += NORM_THREADS;
        }
    }

    /// [`rms_norm_backward_fused_bf16`] on kittens register tiles.
    ///
    /// The block-per-row kernel pays for its two row statistics in shared
    /// memory: a 256-lane tree is eight barriers, and it runs that tree once
    /// per row for all [`NORM_BACKWARD_ROWS_PER_BLOCK`] of them — 160 barriers
    /// a block, to produce two numbers a row. Here a warp owns
    /// [`NORM_TILE_ROWS`] rows at once and both statistics are
    /// `RegTile::row_sum`: two shuffles, no shared memory, no barrier anywhere
    /// in the kernel.
    ///
    /// The weight gradient is the axis that does not fit that picture. It is a
    /// sum down a column, so it cannot live in a row statistic — but it also
    /// does not have to leave the warp: `col_sum` gives the chunk's 32 columns
    /// summed over the warp's 16 rows — folding across the column group itself,
    /// so the eight lanes that hold a column already agree on the whole of it —
    /// and the four lanes holding distinct columns pay one atomic each. That is
    /// `dim` atomics per 16 rows, which is
    /// exactly what the shared-memory kernel already paid — the shared
    /// accumulator was never buying fewer of them, only spreading them over a
    /// block that owned the same 16 rows.
    ///
    /// No `normalized` output: every training call site hands this kernel an
    /// empty one (`no_normalized`), so the arm that recomputed the forward's
    /// result is dead weight here and the shape stays three bands.
    ///
    /// # Safety
    ///
    /// Launch with [`NORM_TILE_THREADS`] threads over exactly
    /// `rows / NORM_TILE_BLOCK_ROWS` blocks: `rows` must be a multiple of
    /// [`NORM_TILE_BLOCK_ROWS`] and `dim` of [`NORM_BACKWARD_TILE_CHUNK`], both
    /// the launcher's to check, since this kernel never bounds-checks. As
    /// [`rms_norm_backward_fused_bf16`], `dweight` is accumulated into and the
    /// caller owes it a zeroed buffer.
    #[kernel]
    #[launch_bounds(128, 6)]
    pub unsafe fn rms_norm_backward_fused_tile_bf16(
        x: &[u32],
        weight: &[f32],
        dy: &[f32],
        residual: &[f32],
        eps: f32,
        dim: u32,
        mut dx: DisjointSlice<f32>,
        mut dweight: DisjointSlice<f32>,
    ) {
        unsafe {
            let lane = lane();
            let row = NORM_TILE_BLOCK_ROWS as u32 * thread::blockIdx_x()
                + NORM_TILE_ROWS as u32 * warp_id();
            let d = dim as usize;
            // The saved input is packed two bf16 to a `u32`; the cursor names
            // the element, so the stride is the row's `dim` values and the
            // widening happens in the load instruction.
            let source = GlobalRows::<Bf16>::from_raw(x.as_ptr() as *mut u8, d);
            let upstream = GlobalRows::<F32>::from_raw(dy.as_ptr() as *mut u8, d);
            let carried = GlobalRows::<F32>::from_raw(residual.as_ptr() as *mut u8, d);
            let destination = GlobalRows::<F32>::from_slice(&mut dx, d);
            // A one-row cursor: the weight is a per-column operand, read once
            // per chunk by `load_cols` rather than once per row (#172).
            let parameters = GlobalRows::<F32>::from_raw(weight.as_ptr() as *mut u8, 0);

            let mut squares = NormRows::splat(0.0);
            let mut dots = NormRows::splat(0.0);
            let mut column = 0u32;
            while column < dim {
                let v: NormBackChunk = load_rows(source, row, column, lane);
                let g: NormBackChunk = load_rows(upstream, row, column, lane);
                let w: NormBackColumns = load_cols(parameters, 0, column, lane);
                squares.add_assign(v.mul(v).row_sum());
                dots.add_assign(g.mul_col(w).mul(v).row_sum());
                column += NORM_BACKWARD_TILE_CHUNK as u32;
            }
            let inv = squares.scale(1.0 / dim as f32).shift(eps).rsqrt();
            // `inv³ · dot / dim`, the block-per-row kernel's `correction`.
            let correction = dots.mul(inv).mul(inv).mul(inv).scale(1.0 / dim as f32);

            column = 0;
            while column < dim {
                let v: NormBackChunk = load_rows(source, row, column, lane);
                let g: NormBackChunk = load_rows(upstream, row, column, lane);
                let r: NormBackChunk = load_rows(carried, row, column, lane);
                let w: NormBackColumns = load_cols(parameters, 0, column, lane);
                store_rows(
                    destination,
                    row,
                    column,
                    lane,
                    r.add(g.mul_col(w).mul_row(inv)).sub(v.mul_row(correction)),
                );

                // The column statistic, paid for by the four lanes that hold
                // distinct columns. `col_sum` has already folded across the
                // column group, so the eight lanes holding a column agree on
                // the whole of it; folding again here would sum those eight
                // agreeing copies and multiply `dweight` by 8.
                let partials = g.mul(v).mul_row(inv).col_sum();
                if lane < 4 {
                    let mut value = 0usize;
                    while value < NormBackColumns::VALUES {
                        let at = column as usize + NormBackColumns::column(lane, value) as usize;
                        // SAFETY: `at` is inside the `dim` accumulators the
                        // launcher promised, and every access to that location
                        // in this kernel is atomic.
                        let slot = DeviceAtomicF32::from_ptr(dweight.as_mut_ptr().add(at));
                        slot.fetch_add(partials.get(value), AtomicOrdering::Relaxed);
                        value += 1;
                    }
                }
                column += NORM_BACKWARD_TILE_CHUNK as u32;
            }
        }
    }

    /// The residual add and the RMSNorm that reads its result, in one launch.
    ///
    /// `stream_input` and `projection` are the two branches meeting at a layer
    /// boundary; `sum` is the rounded residual stream the backward will read
    /// and `y` its normalization. The row is staged in [`ROW`] on the way
    /// through, so the second walk costs no traffic at all and the statistic
    /// is taken over exactly the rounded values `sum` holds — the same
    /// quantity the separate norm would have read back.
    ///
    /// # Safety
    ///
    /// `dim` must be even and at most [`NORM_MAX_COLUMNS`].
    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn rms_norm_forward_residual_bf16(
        stream_input: &[u32],
        projection: &[f32],
        weight: &[f32],
        eps: f32,
        dim: u32,
        mut sum: DisjointSlice<u32>,
        mut y: DisjointSlice<u32>,
    ) {
        static mut PARTIALS: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;
        static mut ROW: SharedArray<u32, { NORM_MAX_COLUMNS / 2 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != NORM_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        let d = dim as usize;
        if d == 0 || !d.is_multiple_of(2) || d > NORM_MAX_COLUMNS {
            return;
        }
        let words = d / 2;
        let base = row * words;
        if base + words > stream_input.len()
            || base + words > sum.len()
            || base + words > y.len()
            || 2 * (base + words) > projection.len()
            || d > weight.len()
        {
            return;
        }

        let mut sum_sq = 0.0f32;
        let mut word = tid;
        while word < words {
            let [low, high] = bf16_halves(stream_input[base + word]);
            let packed = bf16_pair(
                low + projection[2 * (base + word)],
                high + projection[2 * (base + word) + 1],
            );
            // SAFETY: each lane owns distinct words of this block's row.
            unsafe {
                *sum.get_unchecked_mut(base + word) = packed;
                ROW[word] = packed;
            }
            let [low, high] = bf16_halves(packed);
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
            let [low, high] = bf16_halves(unsafe { ROW[word] });
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

    /// The token lookup and the RMSNorm that reads it, in one launch.
    ///
    /// Same shape as [`rms_norm_forward_residual_bf16`] with the gather in
    /// place of the add: `y` is the embedded row the backward scatters into
    /// and `normalized` the first block's attention input.
    ///
    /// # Safety
    ///
    /// `dim` must be even and at most [`NORM_MAX_COLUMNS`].
    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn embedding_forward_norm_bf16(
        table: &[u32],
        tokens: &[u32],
        weight: &[f32],
        eps: f32,
        dim: u32,
        mut y: DisjointSlice<u32>,
        mut normalized: DisjointSlice<u32>,
    ) {
        static mut PARTIALS: SharedArray<f32, NORM_THREADS> = SharedArray::UNINIT;
        static mut ROW: SharedArray<u32, { NORM_MAX_COLUMNS / 2 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != NORM_THREADS {
            return;
        }
        let row = thread::blockIdx_x() as usize;
        let d = dim as usize;
        if d == 0 || !d.is_multiple_of(2) || d > NORM_MAX_COLUMNS {
            return;
        }
        let words = d / 2;
        let base = row * words;
        if row >= tokens.len()
            || base + words > y.len()
            || base + words > normalized.len()
            || d > weight.len()
        {
            return;
        }
        let token_base = tokens[row] as usize * words;
        if token_base + words > table.len() {
            return;
        }

        let mut sum_sq = 0.0f32;
        let mut word = tid;
        while word < words {
            let packed = table[token_base + word];
            // SAFETY: each lane owns distinct words of this block's row.
            unsafe {
                *y.get_unchecked_mut(base + word) = packed;
                ROW[word] = packed;
            }
            let [low, high] = bf16_halves(packed);
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
            let [low, high] = bf16_halves(unsafe { ROW[word] });
            // SAFETY: each lane owns distinct words of this block's row.
            unsafe {
                *normalized.get_unchecked_mut(base + word) = bf16_pair(
                    low * inv * weight[2 * word],
                    high * inv * weight[2 * word + 1],
                );
            }
            word += NORM_THREADS;
        }
    }

    /// [`swiglu_forward`] reading gate and up out of one interleaved
    /// `[rows, 2, ff]` panel — the layout the fused gate/up GEMM writes — so
    /// no split pass ever copies the panel into separate gate/up buffers.
    #[kernel]
    #[launch_bounds(256, 4)]
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

    /// [`swiglu_forward_interleaved`] on a packed gate/up panel, storing packed
    /// pairs: two 8-byte loads and one 8-byte store per thread.
    ///
    /// The panel's producer is the fused gate/up GEMM's own epilogue
    /// (`Tcgen05Gemm::store_at`), so the activation is never fp32 in memory at
    /// all and this pass reads half the bytes its fp32 twin did. The SwiGLU
    /// itself is unchanged: gate, sigmoid and product are fp32 registers, and
    /// only the store rounds.
    ///
    /// `ff` must be a multiple of [`QUAD_LANES`], which the tcgen05 alignment
    /// gate on packed panels already guarantees.
    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn swiglu_forward_interleaved_packed(
        gate_up: &[u32],
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
        // The packed row holds `ff` words: this thread's two gate words start
        // at `gate_word` and its two up words `ff / 2` words later.
        let gate_word = row * ff + 2 * quad;
        if gate_word + ff / 2 + 1 >= gate_up.len() || 2 * i + 1 >= y.len() {
            return;
        }
        // SAFETY: `ff` is a multiple of `QUAD_LANES`, so `gate_word` and
        // `gate_word + ff / 2` are even and both 8-byte loads are aligned;
        // bounds were checked above, and this thread exclusively owns output
        // words `2i` and `2i + 1`.
        let gates = bf16_quad(unsafe { *(gate_up.as_ptr().add(gate_word) as *const u64) });
        let ups = bf16_quad(unsafe { *(gate_up.as_ptr().add(gate_word + ff / 2) as *const u64) });
        let mut activated = [0.0f32; QUAD_LANES];
        for lane in 0..QUAD_LANES {
            let gate = gates[lane];
            let sigmoid = 1.0 / (1.0 + (-gate).exp());
            activated[lane] = gate * sigmoid * ups[lane];
        }
        unsafe {
            *(y.as_mut_ptr() as *mut u64).add(i) = bf16_quad_bits(activated);
        }
    }

    /// Fused [`swiglu_backward_gate`] + [`swiglu_backward_up`]: reads the
    /// interleaved `[rows, 2, ff]` gate/up panel once and writes both halves
    /// of the interleaved `[rows, 2, ff]` gradient, so the two separate
    /// gradient buffers and the join pass that merged them never exist.
    #[kernel]
    #[launch_bounds(256, 4)]
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

    /// [`swiglu_backward_interleaved`] reading a packed gate/up panel and
    /// storing packed pairs (every reader of the gate/up gradient panel is a
    /// tcgen05 operand, #59, and its saved activation is now packed too).
    ///
    /// Two 8-byte loads for the panel, one 16-byte load for the still-fp32
    /// downstream gradient, and two 8-byte packed stores per thread; `ff` must
    /// be a multiple of [`QUAD_LANES`]. Gate and up are the same fp32 values
    /// the forward stored, so the recomputed sigmoid is bit-identical to the
    /// one the packed forward used.
    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn swiglu_backward_interleaved_packed(
        gate_up: &[u32],
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
        let dy_base = row * ff + QUAD_LANES * quad;
        // Panel and gradient share a layout: this thread's two gate words start
        // at `gate_word` and its two up words `ff / 2` words later, in both.
        let gate_word = row * ff + 2 * quad;
        if gate_word + ff / 2 + 1 >= gate_up.len()
            || dy_base + QUAD_LANES > dy.len()
            || gate_word + ff / 2 + 1 >= d_gate_up.len()
        {
            return;
        }
        // SAFETY: `ff` is a multiple of `QUAD_LANES`, so `gate_word` and
        // `gate_word + ff / 2` are even and the 8-byte accesses are aligned,
        // and `dy_base` is a multiple of `QUAD_LANES` so the 16-byte load is;
        // bounds were checked above and this thread owns both word pairs.
        let gates = bf16_quad(unsafe { *(gate_up.as_ptr().add(gate_word) as *const u64) });
        let ups = bf16_quad(unsafe { *(gate_up.as_ptr().add(gate_word + ff / 2) as *const u64) });
        let grads = quad_lanes(unsafe { *(dy.as_ptr().add(dy_base) as *const u128) });
        let mut dgate = [0.0f32; QUAD_LANES];
        let mut dup = [0.0f32; QUAD_LANES];
        for lane in 0..QUAD_LANES {
            let gate = gates[lane];
            let grad = grads[lane];
            let sigmoid = 1.0 / (1.0 + (-gate).exp());
            let dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid));
            dgate[lane] = grad * ups[lane] * dsilu;
            dup[lane] = grad * gate * sigmoid;
        }
        unsafe {
            *(d_gate_up.as_mut_ptr().add(gate_word) as *mut u64) = bf16_quad_bits(dgate);
            *(d_gate_up.as_mut_ptr().add(gate_word + ff / 2) as *mut u64) = bf16_quad_bits(dup);
        }
    }

    #[kernel]
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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
    /// One thread owns one rotated pair. A [`QUAD_LANES`] arm was written for
    /// this kernel and **measured worse twice**, and the note is worth more
    /// than the arm was.
    ///
    /// The arithmetic that motivated it is real: the index math here is five
    /// runtime 64-bit divisions — [`rope_angle`]'s three and this kernel's own
    /// row split — for twelve bytes of output, and four pairs a thread pays
    /// each of them once for four while widening every access from eight bytes
    /// to sixteen. What it cost is the register file. The wide arm holds six
    /// vectors of gradient and three of packed output live at once: 40
    /// registers against 32, **176 bytes of local frame**, and six resident
    /// blocks an SM against eight. At 2.9 TB/s over 1.36 GB a launch this
    /// kernel is latency-bound with the same bytes either way, so the threads
    /// hiding that latency were worth more than the divisions saved —
    /// ferro-kittens#222's pole exactly, reached without a tile in sight.
    ///
    /// Measured `backward.qkv_proj.join`, against `main` in each arm's own
    /// container: **+3.0% and +0.6%** relative to the untouched-span drift.
    /// The lead that is still open is the divisions themselves: they are
    /// `usize` and every operand fits `u32`, which is a third of the
    /// instructions at no register cost at all.
    ///
    /// # Safety
    ///
    /// `dq`, `dk` and `dv` are `[N, heads * head_dim]` and `output` holds at
    /// least `N * 3 * heads * head_dim / 2` words.
    #[kernel]
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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

    /// Whole [`QUAD_LANES`] vectors in a classifier row of `stride_words`
    /// packed words, or zero when the row stride does not divide into them.
    ///
    /// A row base is `row * stride_words`, so an odd stride puts every other
    /// row's base off a 16-byte boundary. Rather than carry a per-row head
    /// alignment, a stride that does not divide gives the vector walk no work
    /// and the scalar walk beside it covers the whole row — the arm every
    /// shape but the padded vocabulary takes, and the one both kernels
    /// already had.
    #[inline(always)]
    fn classifier_row_quads(stride_words: usize) -> usize {
        if stride_words.is_multiple_of(QUAD_LANES) {
            stride_words / QUAD_LANES
        } else {
            0
        }
    }

    /// [`fused_classifier_forward`] over packed-bf16 logits rows.
    ///
    /// Rows are `padded_classes` elements wide (packed two per word) but the
    /// softmax and loss only see the first `classes` columns; the padded tail
    /// holds the lm-head's zero-weight vocabulary columns.
    ///
    /// The row walk is a [`QUAD_LANES`] vector at a time because that is the
    /// only thing putting more than four bytes in flight per lane: the trip
    /// count is a runtime value NVVM will not unroll, and the online
    /// `(max, sum)` is a serial dependence across the whole row, so a lane has
    /// exactly one access outstanding and it is worth four times as much. The
    /// scalar walk stays as the tail — and as the whole of the row when the
    /// stride is not a whole number of vectors, which is where a row base
    /// stops being 16-byte aligned.
    #[kernel]
    #[launch_bounds(256, 4)]
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
        let stride_words = padded_classes as usize / 2;
        let base = row * stride_words;
        let quads = classifier_row_quads(stride_words);
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut quad = tid;
        while quad < quads {
            // SAFETY: the row belongs to this block, `classifier_row_quads`
            // is what makes `base + QUAD_LANES * quad` 16-byte aligned, and
            // striding by the block width gives this lane the vector alone.
            let words = quad_words(unsafe {
                *(logits.as_ptr().add(base + QUAD_LANES * quad) as *const u128)
            });
            let mut lane = 0usize;
            while lane < QUAD_LANES {
                let mut half = 0;
                while half < 2 {
                    let col = 2 * (QUAD_LANES * quad + lane) + half;
                    if col < c {
                        let value = bf16_bits_to_f32((words[lane] >> (16 * half)) as u16);
                        let next_max = running_max.max(value);
                        running_sum =
                            running_sum * (running_max - next_max).exp() + (value - next_max).exp();
                        running_max = next_max;
                    }
                    half += 1;
                }
                lane += 1;
            }
            quad += CLASSIFIER_THREADS;
        }
        let mut pair = QUAD_LANES * quads + tid;
        while pair < stride_words {
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
    ///
    /// Both passes walk the row a [`QUAD_LANES`] vector at a time, for
    /// [`fused_classifier_forward_bf16`]'s reason: the write-back's
    /// read-modify-write had two four-byte accesses in flight per lane where
    /// it can have two sixteen-byte ones, and this kernel reads the row twice.
    #[kernel]
    #[launch_bounds(256, 4)]
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
        let quads = classifier_row_quads(stride_words);
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut quad = tid;
        while quad < quads {
            // SAFETY: the row belongs to this block, `classifier_row_quads`
            // is what makes `base + QUAD_LANES * quad` 16-byte aligned, and
            // striding by the block width gives this lane the vector alone.
            let words = quad_words(unsafe {
                *(logits.as_mut_ptr().add(base + QUAD_LANES * quad) as *const u128)
            });
            let mut lane = 0usize;
            while lane < QUAD_LANES {
                let mut half = 0;
                while half < 2 {
                    let col = 2 * (QUAD_LANES * quad + lane) + half;
                    if col < c {
                        let value = bf16_bits_to_f32((words[lane] >> (16 * half)) as u16);
                        let next_max = running_max.max(value);
                        running_sum =
                            running_sum * (running_max - next_max).exp() + (value - next_max).exp();
                        running_max = next_max;
                    }
                    half += 1;
                }
                lane += 1;
            }
            quad += CLASSIFIER_THREADS;
        }
        let mut pair = QUAD_LANES * quads + tid;
        while pair < stride_words {
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
        let mut quad = tid;
        while quad < quads {
            // SAFETY: this lane exclusively owns the vector for both the read
            // and the in-place gradient write, at the alignment
            // `classifier_row_quads` established.
            let slot = unsafe { logits.as_mut_ptr().add(base + QUAD_LANES * quad) as *mut u128 };
            let words = quad_words(unsafe { *slot });
            let mut packed = [0u32; QUAD_LANES];
            let mut lane = 0usize;
            while lane < QUAD_LANES {
                let mut half = 0;
                while half < 2 {
                    let col = 2 * (QUAD_LANES * quad + lane) + half;
                    if col < c {
                        let value = bf16_bits_to_f32((words[lane] >> (16 * half)) as u16);
                        let probability = (value - row_max).exp() * inverse_sum;
                        let indicator = if col == target { 1.0 } else { 0.0 };
                        let gradient = scale * (probability - indicator);
                        packed[lane] |= (f32_to_bf16_bits(gradient) as u32) << (16 * half);
                    }
                    half += 1;
                }
                lane += 1;
            }
            unsafe {
                *slot = quad_word_bits(packed);
            }
            quad += CLASSIFIER_THREADS;
        }
        let mut pair = QUAD_LANES * quads + tid;
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
    #[launch_bounds(256, 4)]
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

    // The five naive attention kernels below belong in `reference.rs` with the
    // rest of the baseline: no training step launches them, and what reads them
    // is `gpu/flash-attn`'s gate. They stay here only because moving them means
    // editing that crate's `main.rs` and `bin/bench.rs`, and those are being
    // changed on another branch. Move them when it lands.

    /// Materialize causal softmax probabilities as `[N,H,T]`.
    #[kernel]
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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

    /// [`router_logits`] over a packed-bf16 token stream. The router weight
    /// and the logits stay fp32: the router is an fp32 parameter by design.
    ///
    /// A block owns a `[ROUTER_GEMM_BM, ROUTER_GEMM_BN]` logit tile and a lane
    /// owns one element of it, which is the only mapping that fills the device:
    /// the output is `N * E` elements and `E` is 8, so a lane per *token* would
    /// leave an SM 166 threads and a lane per token pair would need a
    /// cross-lane reduction of the `dim` axis.
    ///
    /// What the reference blocking gets wrong is not that mapping but
    /// [`ROUTER_LOGITS_BK`]: at 16 a lane stages two `x` elements, waits on a
    /// barrier, does sixteen FMAs and waits again, 192 times over. Eight bytes
    /// in flight per lane is what 453 GB/s looks like. Here a lane stages
    /// sixteen elements as four [`QUAD_LANES`] vectors and the block pays 48
    /// barriers instead of 384.
    ///
    /// Both tiles are staged `f32`: `x` widens once per block on the way in
    /// rather than once per lane that reads it, and the weight lands
    /// *transposed*, so a lane's expert column is contiguous and both operands
    /// of the inner product are one vector load. Neither staging load is
    /// predicated — the address is clamped and the *value* is selected — so the
    /// four of them issue back to back instead of down a chain of branches.
    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn router_logits_bf16(
        x: &[u32],
        weight: &[f32],
        dim: u32,
        experts: u32,
        mut logits: DisjointSlice<f32>,
    ) {
        static mut TILE_X: SharedArray<f32, { ROUTER_GEMM_BM * ROUTER_LOGITS_STRIDE }, 16> =
            SharedArray::UNINIT;
        static mut TILE_WEIGHT: SharedArray<
            f32,
            { ROUTER_MAX_EXPERTS * ROUTER_LOGITS_STRIDE },
            16,
        > = SharedArray::UNINIT;

        let d = dim as usize;
        let e = experts as usize;
        let tid = thread::threadIdx_x() as usize;
        // Every bound the walks below rely on, paid once: the tile reads are
        // whole `QUAD_LANES` vectors, so a partial vector must not exist.
        if thread::blockDim_x() as usize != ROUTER_GEMM_THREADS
            || d == 0
            || e == 0
            || e > ROUTER_MAX_EXPERTS
            || !d.is_multiple_of(QUAD_LANES)
            || !(x.len() * 2).is_multiple_of(d)
            || d * e > weight.len()
        {
            return;
        }
        let n = x.len() * 2 / d;
        let block_row = thread::blockIdx_y() as usize * ROUTER_GEMM_BM;
        if block_row >= n || n * e > logits.len() {
            return;
        }

        let thread_row = tid / ROUTER_GEMM_BN;
        let thread_col = tid % ROUTER_GEMM_BN;
        let stage_row = tid / (ROUTER_LOGITS_BK / QUAD_LANES);
        let stage_column = tid % (ROUTER_LOGITS_BK / QUAD_LANES) * QUAD_LANES;
        let weight_depth = tid / ROUTER_MAX_EXPERTS;
        let weight_expert = tid % ROUTER_MAX_EXPERTS;

        let mut accumulator = 0.0f32;
        let mut base = 0usize;
        while base < d {
            let mut stage = 0usize;
            while stage < ROUTER_LOGITS_STAGES {
                let tile_row = stage * ROUTER_LOGITS_STAGE_ROWS + stage_row;
                let row = block_row + tile_row;
                let column = base + stage_column;
                let inside = row < n && column < d;
                let word = if inside { (row * d + column) / 2 } else { 0 };
                let quad = bf16_quad(unsafe { *(x.as_ptr().add(word) as *const u64) });
                unsafe {
                    *((&raw mut TILE_X as *mut f32)
                        .add(tile_row * ROUTER_LOGITS_STRIDE + stage_column)
                        as *mut u128) = quad_bits(if inside { quad } else { [0.0; QUAD_LANES] });
                }
                stage += 1;
            }

            let mut stage = 0usize;
            while stage < ROUTER_LOGITS_WEIGHT_STAGES {
                let depth = stage * (ROUTER_GEMM_THREADS / ROUTER_MAX_EXPERTS) + weight_depth;
                let column = base + depth;
                let inside = column < d && weight_expert < e;
                let index = if inside {
                    column * e + weight_expert
                } else {
                    0
                };
                let value = unsafe { *weight.as_ptr().add(index) };
                unsafe {
                    TILE_WEIGHT[weight_expert * ROUTER_LOGITS_STRIDE + depth] =
                        if inside { value } else { 0.0 };
                }
                stage += 1;
            }
            thread::sync_threads();

            let mut inner = 0usize;
            while inner < ROUTER_LOGITS_BK {
                let xs = quad_lanes(unsafe {
                    *((&raw const TILE_X as *const f32)
                        .add(thread_row * ROUTER_LOGITS_STRIDE + inner)
                        as *const u128)
                });
                let ws = quad_lanes(unsafe {
                    *((&raw const TILE_WEIGHT as *const f32)
                        .add(thread_col * ROUTER_LOGITS_STRIDE + inner)
                        as *const u128)
                });
                let mut lane = 0usize;
                while lane < QUAD_LANES {
                    accumulator += xs[lane] * ws[lane];
                    lane += 1;
                }
                inner += QUAD_LANES;
            }
            thread::sync_threads();
            base += ROUTER_LOGITS_BK;
        }

        let row = block_row + thread_row;
        if row < n && thread_col < e {
            unsafe {
                *logits.get_unchecked_mut(row * e + thread_col) = accumulator;
            }
        }
    }

    /// Per-token softmax, deterministic top-k, and selected-probability
    /// renormalization. Ties select the lower expert index.
    #[kernel]
    #[launch_bounds(256, 4)]
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

    /// Block-parallel deterministic capacity assignment.
    ///
    /// One block owns an expert and partitions the flattened token/rank order
    /// into contiguous lane ranges. The exclusive prefix of each range's match
    /// count is its first slot, preserving the serial kernel's exact ordering,
    /// capacity-drop behavior, and assignment counts without atomic claims.
    #[kernel]
    #[launch_bounds(256, 4)]
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

    /// One expert's addend to one layer's Switch-style load-balancing loss:
    /// `E · assignment_fraction · mean_probability`, so a layer's whole
    /// auxiliary loss is the sum of its `E` terms and the model's is the sum
    /// of the whole `[L, E]` buffer.
    ///
    /// One block owns an expert, and so owns that expert's slot outright: lanes
    /// accumulate strided token slices, a tree reduction combines them, and
    /// lane 0 weights the mean and stores. Accumulating the lanes with
    /// same-address atomics instead needed the buffer pre-zeroed by a `fill`
    /// launch per layer, serialized [`MOE_AUX_TERMS_THREADS`] adds on one word,
    /// and left the summation order — and so the reported auxiliary loss —
    /// dependent on their arrival order (#99). Leaving a term per
    /// `(layer, expert)` keeps that property while letting one launch at the
    /// end of the forward fold every layer's loss into the scalar at once. The
    /// auxiliary *gradient* never reads this: `router_backward` derives it from
    /// the assignment counts.
    #[kernel]
    #[launch_bounds(1024, 1)]
    pub unsafe fn moe_aux_terms(
        probabilities: &[f32],
        assignment_counts: &[u32],
        tokens: u32,
        experts: u32,
        top_k: u32,
        layer: u32,
        mut aux_terms: DisjointSlice<f32>,
    ) {
        static mut SUMS: SharedArray<f32, MOE_AUX_TERMS_THREADS> = SharedArray::UNINIT;

        let expert = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let n = tokens as usize;
        let e = experts as usize;
        let k = top_k as usize;
        let term = layer as usize * e + expert;
        // Uniform over the block, so no lane reaches a barrier the rest skip.
        if expert >= e
            || n == 0
            || k == 0
            || term >= aux_terms.len()
            || expert >= assignment_counts.len()
            || probabilities.len() < n * e
        {
            return;
        }
        let mut sum = 0.0f32;
        let mut token = lane;
        while token < n {
            sum += probabilities[token * e + expert];
            token += MOE_AUX_TERMS_THREADS;
        }

        // SAFETY: each lane owns its own slot of the block's scratch.
        unsafe {
            SUMS[lane] = sum;
        }
        thread::sync_threads();
        let mut stride = MOE_AUX_TERMS_THREADS / 2;
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
            let assignment_fraction = assignment_counts[expert] as f32 / (n * k) as f32;
            // SAFETY: this block exclusively owns `(layer, expert)`.
            unsafe {
                *aux_terms.get_unchecked_mut(term) =
                    e as f32 * assignment_fraction * (SUMS[0] / n as f32);
            }
        }
    }

    /// The scalar training loss: mean per-token cross entropy plus every
    /// layer's weighted auxiliary loss.
    ///
    /// One block. Lanes reduce the per-token losses grid-stride, a tree
    /// reduction combines them, and lane 0 folds in the terms
    /// [`moe_aux_terms`] left behind, summed in layer-major order so the
    /// reported loss does not depend on which block wrote which term. This is
    /// the whole loss tail: it replaces a `sum`, a `scale` and one
    /// single-threaded auxiliary launch per layer — kernels whose cost was
    /// almost entirely their launch. A model without experts leaves `aux_terms`
    /// zero and gets the plain mean.
    #[kernel]
    #[launch_bounds(1024, 1)]
    pub unsafe fn loss_mean_with_aux(
        losses: &[f32],
        tokens: u32,
        aux_terms: &[f32],
        coefficient: f32,
        mut loss: DisjointSlice<f32>,
    ) {
        static mut PARTIALS: SharedArray<f32, LOSS_TAIL_THREADS> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x() as usize;
        let n = tokens as usize;
        // Uniform over the block, so no lane reaches a barrier the rest skip.
        if thread::blockDim_x() as usize != LOSS_TAIL_THREADS
            || n == 0
            || losses.len() < n
            || loss.is_empty()
        {
            return;
        }
        let mut partial = 0.0f32;
        let mut token = lane;
        while token < n {
            partial += losses[token];
            token += LOSS_TAIL_THREADS;
        }

        // SAFETY: each lane owns its own slot of the block's scratch.
        unsafe {
            PARTIALS[lane] = partial;
        }
        thread::sync_threads();
        let mut stride = LOSS_TAIL_THREADS / 2;
        while stride > 0 {
            if lane < stride {
                // SAFETY: the surviving half of the lanes own disjoint slots.
                unsafe {
                    PARTIALS[lane] += PARTIALS[lane + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            let mut auxiliary = 0.0f32;
            for term in 0..aux_terms.len() {
                auxiliary += aux_terms[term];
            }
            // SAFETY: the single surviving lane of the single block stores.
            unsafe {
                *loss.get_unchecked_mut(0) = PARTIALS[0] / n as f32 + coefficient * auxiliary;
            }
        }
    }

    /// [`moe_scatter_bf16_quad`] with a packed source.
    ///
    /// The token stream is already the dtype the bins hold, so the scatter is
    /// one 8-byte copy per thread: it reads half the bytes of the fp32 twin
    /// and rounds nothing at all. `dim` must be a multiple of [`QUAD_LANES`].
    #[kernel]
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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

    /// [`moe_gather_combine_add`] on a packed residual stream, one word per
    /// thread. The arm for a `dim` the quad twin's 16-byte accesses cannot
    /// cover; `dim` must still be even.
    #[kernel]
    #[launch_bounds(256, 4)]
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

    /// [`moe_gather_combine_add_quad_bf16`] with the expert bins packed too —
    /// every operand this kernel touches is bf16 in memory.
    ///
    /// The down projection's epilogue writes its bins packed
    /// (`Tcgen05Gemm::store_at`), and a packed bin panel only exists where the
    /// experts are tcgen05-aligned, which forces `dim` to a multiple of
    /// `TC_N_TILE` and so of [`QUAD_LANES`]: there is no word-wise twin of this
    /// kernel because no shape can reach one. The combine still sums in fp32
    /// registers and the block boundary still rounds exactly once.
    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn moe_gather_combine_add_quad_packed(
        expert_output: &[u32],
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
        // are aligned, and every bin row's word offset is a multiple of
        // `QUAD_LANES / 2` so the 8-byte expert reads are; bounds were checked
        // above and this thread exclusively owns its output quad.
        let mut value = bf16_quad(unsafe { *(residual.as_ptr().add(base) as *const u64) });
        for rank in 0..k {
            let pair = token * k + rank;
            let slot = slots[pair];
            if slot != MOE_DROPPED_SLOT {
                let expert = selected_experts[pair] as usize;
                let input = (expert * c + slot as usize) * (d / 2) + 2 * quad;
                if input + 1 >= expert_output.len() {
                    return;
                }
                let gate = gate_weights[pair];
                let outputs =
                    bf16_quad(unsafe { *(expert_output.as_ptr().add(input) as *const u64) });
                for lane in 0..QUAD_LANES {
                    value[lane] += gate * outputs[lane];
                }
            }
        }
        unsafe {
            *(output.as_mut_ptr().add(base) as *mut u64) = bf16_quad_bits(value);
        }
    }

    /// [`moe_gather_combine_add_quad`] on a packed residual stream.
    ///
    /// The expert outputs stay fp32 here — this is the arm the non-aligned
    /// fp32 oracle takes, whose register-tiled GEMM writes a wide bin. The
    /// aligned path's twin is [`moe_gather_combine_add_quad_packed`]. The
    /// combine sums in fp32 registers either way, so the block boundary rounds
    /// exactly once.
    #[kernel]
    #[launch_bounds(256, 4)]
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
        let mut value = bf16_quad(unsafe { *(residual.as_ptr().add(base) as *const u64) });
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
            *(output.as_mut_ptr().add(base) as *mut u64) = bf16_quad_bits(value);
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
    #[launch_bounds(256, 4)]
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

    /// [`moe_scatter_dy`] on packed bins: a packed saved output read for the
    /// gate dot product, and a packed bin gradient stored.
    ///
    /// Both readers of the gradient panel are bf16 tcgen05 operands (#59), and
    /// the saved output is now the down projection's packed epilogue target, so
    /// neither side of this kernel is fp32 in memory. The dot product still
    /// accumulates in fp32 over the same quads in the same lane order; only the
    /// stored and loaded values narrow.
    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn moe_scatter_dy_packed(
        expert_output: &[u32],
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
        if (bin_base + d) / 2 > expert_output.len()
            || (bin_base + d) / 2 > expert_output_gradient.len()
            || token_base + d > dy.len()
        {
            return;
        }

        let gate = gate_weights[pair];
        let mut dot = 0.0f32;
        if d.is_multiple_of(QUAD_LANES) {
            // SAFETY: `bin_base` and `token_base` are multiples of `dim`, hence
            // of `QUAD_LANES`, so the fp32 gradient row is 16-byte aligned and
            // both packed rows' `bin_base / 2` bases are 8-byte aligned; the
            // row bounds were checked above. Each lane owns distinct quads.
            let dy_row = unsafe { dy.as_ptr().add(token_base) as *const u128 };
            let output_row = unsafe { expert_output.as_ptr().add(bin_base / 2) as *const u64 };
            let gradient_row =
                unsafe { expert_output_gradient.as_mut_ptr().add(bin_base / 2) as *mut u64 };
            let mut quad = tid;
            while quad < d / QUAD_LANES {
                let grad = quad_lanes(unsafe { *dy_row.add(quad) });
                let output = bf16_quad(unsafe { *output_row.add(quad) });
                let mut scaled = [0.0f32; QUAD_LANES];
                for lane in 0..QUAD_LANES {
                    dot += output[lane] * grad[lane];
                    scaled[lane] = gate * grad[lane];
                }
                unsafe {
                    *gradient_row.add(quad) = bf16_quad_bits(scaled);
                }
                quad += MOE_SCATTER_DY_THREADS;
            }
        } else {
            let mut word = tid;
            while word < d / 2 {
                let column = 2 * word;
                let low = dy[token_base + column];
                let high = dy[token_base + column + 1];
                let output = bf16_halves(expert_output[bin_base / 2 + word]);
                dot += output[0] * low + output[1] * high;
                // SAFETY: each lane owns distinct words of this block's bin row.
                unsafe {
                    *expert_output_gradient.get_unchecked_mut(bin_base / 2 + word) =
                        bf16_pair(gate * low, gate * high);
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
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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

    /// [`moe_gather_dx`] with the router input-gradient add folded in: each
    /// output element is `router_dx + Σ_k expert_input_gradient`, so the
    /// separate `[N, D]` combine pass and the intermediate it read never run.
    #[kernel]
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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
    #[launch_bounds(256, 4)]
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

    /// [`router_backward_weight_split`] reading its saved input packed.
    ///
    /// The pair of rows a lane owns is the pair a packed word already holds, so
    /// the widening is free and the load is the same one instruction.
    ///
    /// # Safety
    ///
    /// Same contract as [`router_backward_weight_split`], with `x` holding
    /// `tokens * dim / 2` packed words.
    /// **The one entry point in this module that declares no block**, and the
    /// measurement is the reason. `.maxntid` is an input to ptxas' heuristics
    /// and not only the register budget's divisor, and this kernel reads those
    /// heuristics harder than any other here: the note above is that the
    /// staged token depot is *supposed* to be a `.local` one, L1-resident, and
    /// that the registers it saves are what keep the warps hiding the loads
    /// (#111). Declaring the block moves that balance, and every target a
    /// 256-thread block has moves it the wrong way.
    ///
    /// `backward.router.weight`, same container, against `main`'s derived
    /// allocation, both passes of `BASELINE_REF=main ./run.sh model profile`:
    ///
    /// | declaration | regs | frame | blocks/SM | span |
    /// |---|---:|---:|---:|---:|
    /// | none (ships) | 32 | 0 | 8 | — |
    /// | `(256)` | 93 | 0 | **2** | +52% |
    /// | `(256, 4)` | 64 | 192 B | 4 | +8.1% / +8.6% |
    /// | `(256, 6)` | 40 | 408 B | 6 | +126% / +120% |
    /// | `(256, 8)` | 32 | 480 B | 8 | +189% |
    ///
    /// Four targets, four losses, and the two that keep the occupancy are the
    /// two that pay for it in frame. What #122 buys — a declared block instead
    /// of one the driver derives — is not worth 8% of this span, and the
    /// derived 1024 is in no danger of being narrower than the 256 it is
    /// launched in. Pinned when there is a target that does not cost anything.
    #[kernel]
    pub unsafe fn router_backward_weight_split_bf16(
        x: &[u32],
        dlogits: &[f32],
        tokens: u32,
        experts: u32,
        dim: u32,
        mut partials: DisjointSlice<f32>,
    ) {
        let n = tokens as usize;
        let e = experts as usize;
        let d = dim as usize;
        if thread::blockDim_x() as usize != ROUTER_WGRAD_THREADS
            || e == 0
            || e > ROUTER_MAX_EXPERTS
            || !d.is_multiple_of(ROUTER_WGRAD_ROWS)
            || n * d > x.len() * 2
            || n * e > dlogits.len()
            || ROUTER_WGRAD_SPLITS * e * d > partials.len()
        {
            return;
        }
        let row = thread::blockIdx_x() as usize * ROUTER_WGRAD_BM
            + thread::threadIdx_x() as usize * ROUTER_WGRAD_ROWS;
        if row >= d {
            return;
        }

        let partition = n.div_ceil(ROUTER_WGRAD_SPLITS);
        let split = thread::blockIdx_y() as usize;
        let token_end = ((split + 1) * partition).min(n);
        let mut token = (split * partition).min(n);

        let mut low = [0.0f32; ROUTER_MAX_EXPERTS];
        let mut high = [0.0f32; ROUTER_MAX_EXPERTS];
        while token < token_end {
            let mut lows = [0.0f32; ROUTER_WGRAD_TOKENS];
            let mut highs = [0.0f32; ROUTER_WGRAD_TOKENS];
            let mut step = 0usize;
            while step < ROUTER_WGRAD_TOKENS {
                let index = (token + step).min(token_end - 1);
                let keep = if token + step < token_end { 1.0 } else { 0.0 };
                let halves = bf16_halves(unsafe { *x.get_unchecked((index * d + row) / 2) });
                lows[step] = halves[0] * keep;
                highs[step] = halves[1] * keep;
                step += 1;
            }

            let mut step = 0usize;
            while step < ROUTER_WGRAD_TOKENS {
                let gate_base = (token + step).min(token_end - 1) * e;
                let mut expert = 0usize;
                while expert < ROUTER_MAX_EXPERTS {
                    let held = expert < e;
                    let gate = unsafe {
                        *dlogits.get_unchecked(gate_base + if held { expert } else { 0 })
                    };
                    let gate = if held { gate } else { 0.0 };
                    low[expert] += lows[step] * gate;
                    high[expert] += highs[step] * gate;
                    expert += 1;
                }
                step += 1;
            }
            token += ROUTER_WGRAD_TOKENS;
        }

        // The lane's two rows are adjacent in `partials`, so each expert's
        // pair of stores is one contiguous run per warp.
        let base = split * e * d + row;
        let mut expert = 0usize;
        while expert < ROUTER_MAX_EXPERTS {
            if expert < e {
                unsafe {
                    *partials.get_unchecked_mut(base + expert * d) = low[expert];
                    *partials.get_unchecked_mut(base + expert * d + 1) = high[expert];
                }
            }
            expert += 1;
        }
    }

    /// Sum the router weight-gradient token partitions in ascending order and
    /// accumulate into `dweight[D,E]`. Threads walk `partials` expert-major so
    /// the wide read is coalesced; the narrow `dweight` update is not, and does
    /// not matter at `D*E` elements.
    #[kernel]
    #[launch_bounds(256, 4)]
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
