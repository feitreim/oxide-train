//! Fused fp32 causal attention kernels.
//!
//! Forward applies an online softmax while streaming over K/V, so its only
//! output-sized storage is `[B*T,H,HD]`; it never materializes `[B*T,H,T]`
//! probabilities. Forward saves only one log-sum-exp scalar per row/head.
//! Backward recomputes probabilities with query-parallel dQ blocks and
//! key-parallel dK/dV blocks, making all gradient writes disjoint.
//!
//! Two kernel generations coexist (7e7):
//! - The original per-row kernels (`flash_attention_*`) give each `(row, head)`
//!   one block of `head_dim` lanes that scans keys serially, reducing every
//!   score through shared memory. They are the parity oracles.
//! - The FlashAttention-2 style kernels (`*_tiled`) stage query/key/value
//!   blocks through shared memory and compute whole `BQ x BK` score tiles as
//!   register-tiled fragments, with parallelism over query blocks (forward,
//!   dQ) or key blocks (dK/dV). They specialize on `TILE_HD` and use the
//!   `*_config` helpers below as their launch contract. The tiled backward
//!   reads per-row `dy . y` dots staged once by `flash_attention_backward_dot`.
//!
//! The per-row launch contract is:
//! - `head_dim` is a power of two and at most `MAX_HEAD_DIM`;
//! - `block_dim.x == head_dim`, with all other block/grid dimensions equal to 1;
//! - forward launches `rows * heads` blocks;
//! - tiled backward launches `rows * heads` blocks for each of dQ and dK/dV.

use cuda_core::LaunchConfig;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

// Host-only tcgen05 support (flash.ptx loader, TMA maps); no device code, so
// including it never affects an artifact. Not every including binary uses it.
#[allow(dead_code)]
pub mod host;

/// Maximum supported head width. This bounds the statically allocated shared
/// reduction buffer; actual launches use exactly `head_dim` threads.
pub const MAX_HEAD_DIM: usize = 256;

/// Head width the tiled kernels specialize on; shared tiles are sized by it
/// at compile time. Launches assert the model's `HD` matches.
pub const TILE_HD: usize = 128;

/// Tiled-forward query rows per block. Rewritten by the repository's `SWEEP`
/// harness. Halved from the HD=64 era so the doubled head width keeps the
/// static shared tiles under the 48 KiB limit.
pub const FWD_BQ: usize = 32;
/// Tiled-forward key rows per staged tile. Rewritten by `SWEEP`.
pub const FWD_BK: usize = 16;
/// Tiled-forward query rows per thread in the score fragment.
pub const FWD_TM: usize = 4;
/// Tiled-forward key columns per thread in the score fragment.
pub const FWD_TN: usize = 4;

const FWD_ROW_THREADS: usize = FWD_BQ / FWD_TM;
const FWD_COL_THREADS: usize = FWD_BK / FWD_TN;
const FWD_THREADS: usize = FWD_ROW_THREADS * FWD_COL_THREADS;
const FWD_TD: usize = TILE_HD / FWD_COL_THREADS;

/// Tiled dQ query rows per block. Rewritten by `SWEEP`.
pub const BWQ_BQ: usize = 16;
/// Tiled dQ key rows per staged tile. Rewritten by `SWEEP`.
pub const BWQ_BK: usize = 16;
/// Tiled dQ query rows per thread in the score fragment.
pub const BWQ_TM: usize = 4;
/// Tiled dQ key columns per thread in the score fragment.
pub const BWQ_TN: usize = 4;

const BWQ_ROW_THREADS: usize = BWQ_BQ / BWQ_TM;
const BWQ_COL_THREADS: usize = BWQ_BK / BWQ_TN;
const BWQ_THREADS: usize = BWQ_ROW_THREADS * BWQ_COL_THREADS;
const BWQ_TD: usize = TILE_HD / BWQ_COL_THREADS;

/// Tiled dK/dV key rows per block. Rewritten by `SWEEP`.
pub const KV_BK: usize = 16;
/// Tiled dK/dV query rows per staged tile. Rewritten by `SWEEP`.
pub const KV_BQ: usize = 16;
/// Tiled dK/dV query rows per thread in the score fragment.
pub const KV_TM: usize = 4;
/// Tiled dK/dV key columns per thread in the score fragment.
pub const KV_TN: usize = 4;

const KV_ROW_THREADS: usize = KV_BQ / KV_TM;
const KV_COL_THREADS: usize = KV_BK / KV_TN;
const KV_THREADS: usize = KV_ROW_THREADS * KV_COL_THREADS;
const KV_ACC_M: usize = KV_BK / KV_ROW_THREADS;
const KV_ACC_D: usize = TILE_HD / KV_COL_THREADS;

/// Finite stand-in for "masked": far enough below any real scaled score that
/// `exp(MASKED_SCORE - max)` flushes to zero, while keeping the online-softmax
/// recurrence free of `-inf - -inf` NaNs on rows past the sequence end.
const MASKED_SCORE: f32 = f32::MIN;

const STATIC_SHARED_LIMIT: usize = 48 * 1024;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[inline(always)]
    fn f32_to_bf16_rne(value: f32) -> u16 {
        let bits = value.to_bits();
        let round = 0x7fffu32 + ((bits >> 16) & 1);
        (bits.wrapping_add(round) >> 16) as u16
    }

    /// De-interleave fp32 `[B*T, H*64]` activations into the packed-bf16
    /// head panels `[B*H, T, 64]` the tcgen05 forward streams with TMA,
    /// folding `scale` into the quantization (`1.0` for K/V;
    /// `softmax_scale * log2(e)` for Q, which makes downstream softmax math
    /// base-2 native). One thread per packed output pair; launch with
    /// [`stage_heads_config`].
    #[kernel]
    pub fn stage_attention_heads_bf16(
        input: &[f32],
        sequence_length: u32,
        heads: u32,
        scale: f32,
        mut output: DisjointSlice<u32>,
    ) {
        const PAIRS_PER_ROW: usize = TILE_HD / 2;
        let index = thread::index_1d();
        let word = index.get();
        let t = sequence_length as usize;
        let h = heads as usize;
        let pair = word % PAIRS_PER_ROW;
        let token = (word / PAIRS_PER_ROW) % t;
        let plane = word / (PAIRS_PER_ROW * t);
        let batch = plane / h;
        let head = plane % h;
        let base = ((batch * t + token) * h + head) * TILE_HD + pair * 2;
        if let Some(slot) = output.get_mut(index) {
            let low = f32_to_bf16_rne(input[base] * scale) as u32;
            let high = f32_to_bf16_rne(input[base + 1] * scale) as u32;
            *slot = low | (high << 16);
        }
    }

    /// Flash-style causal attention forward using an online softmax.
    #[kernel]
    pub fn flash_attention_forward(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut output: DisjointSlice<f32>,
        mut logsumexp: DisjointSlice<f32>,
    ) {
        static mut PARTIALS: SharedArray<f32, MAX_HEAD_DIM> = SharedArray::UNINIT;
        static mut SCORE: SharedArray<f32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let hd = head_dim as usize;
        if hd == 0
            || hd > MAX_HEAD_DIM
            || !hd.is_power_of_two()
            || thread::blockDim_x() as usize != hd
        {
            return;
        }

        let t = sequence_length as usize;
        let h = heads as usize;
        let query_head = thread::blockIdx_x() as usize;
        let query_row = query_head / h;
        let head = query_head % h;
        let query_position = query_row % t;
        let sequence_start = query_row - query_position;
        let query_base = (query_row * h + head) * hd;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut accumulator = 0.0f32;

        for key_position in 0..=query_position {
            let key_row = sequence_start + key_position;
            let key_base = (key_row * h + head) * hd;
            unsafe {
                PARTIALS[tid] = q[query_base + tid] * k[key_base + tid];
            }
            thread::sync_threads();

            let mut stride = hd / 2;
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
                    SCORE[0] = PARTIALS[0] * scale;
                }
            }
            thread::sync_threads();

            let score = unsafe { SCORE[0] };
            let next_max = running_max.max(score);
            let previous_weight = (running_max - next_max).exp();
            let current_weight = (score - next_max).exp();
            running_sum = previous_weight * running_sum + current_weight;
            accumulator = previous_weight * accumulator + current_weight * v[key_base + tid];
            running_max = next_max;
        }

        // SAFETY: `(blockIdx.x, threadIdx.x)` uniquely identifies one output
        // element because the launch contract uses exactly `head_dim` lanes.
        unsafe {
            *output.get_unchecked_mut(query_base + tid) = accumulator / running_sum;
        }
        if tid == 0 {
            // SAFETY: one block owns each `query_head`, and only lane zero
            // writes its row statistic.
            unsafe {
                *logsumexp.get_unchecked_mut(query_head) = running_max + running_sum.ln();
            }
        }
    }

    /// Recompute-softmax backward, fused across dQ, dK, and dV.
    ///
    /// A block owns one `(batch, head)`. Lane `d` exclusively owns feature `d`
    /// for every token in that sequence, so updates to dK/dV need no atomics.
    #[kernel]
    pub fn flash_attention_backward(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        dy: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut dq: DisjointSlice<f32>,
        mut dk: DisjointSlice<f32>,
        mut dv: DisjointSlice<f32>,
    ) {
        static mut PARTIALS: SharedArray<f32, MAX_HEAD_DIM> = SharedArray::UNINIT;
        static mut SCALAR: SharedArray<f32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let hd = head_dim as usize;
        if hd == 0
            || hd > MAX_HEAD_DIM
            || !hd.is_power_of_two()
            || thread::blockDim_x() as usize != hd
        {
            return;
        }

        let t = sequence_length as usize;
        let h = heads as usize;
        let batch_head = thread::blockIdx_x() as usize;
        let batch = batch_head / h;
        let head = batch_head % h;
        let sequence_start = batch * t;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // This block owns every dK/dV feature in its `(batch, head)` slice.
        for position in 0..t {
            let index = ((sequence_start + position) * h + head) * hd + tid;
            // SAFETY: blocks have disjoint `(batch, head)` slices and lanes
            // have disjoint feature indices.
            unsafe {
                *dk.get_unchecked_mut(index) = 0.0;
                *dv.get_unchecked_mut(index) = 0.0;
            }
        }

        for query_position in 0..t {
            let query_row = sequence_start + query_position;
            let query_base = (query_row * h + head) * hd;
            let mut max_score = f32::NEG_INFINITY;

            // Pass 1: row maximum for stable softmax.
            for key_position in 0..=query_position {
                let key_row = sequence_start + key_position;
                let key_base = (key_row * h + head) * hd;
                unsafe {
                    PARTIALS[tid] = q[query_base + tid] * k[key_base + tid];
                }
                thread::sync_threads();
                let mut stride = hd / 2;
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
                        SCALAR[0] = PARTIALS[0] * scale;
                    }
                }
                thread::sync_threads();
                max_score = max_score.max(unsafe { SCALAR[0] });
            }

            // Pass 2: denominator and sum_j p_j * dP_j. Keeping both
            // unnormalized lets us divide only once after the pass.
            let mut denominator = 0.0f32;
            let mut softmax_dot_numerator = 0.0f32;
            for key_position in 0..=query_position {
                let key_row = sequence_start + key_position;
                let key_base = (key_row * h + head) * hd;
                unsafe {
                    PARTIALS[tid] = q[query_base + tid] * k[key_base + tid];
                }
                thread::sync_threads();
                let mut stride = hd / 2;
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
                        SCALAR[0] = PARTIALS[0] * scale;
                    }
                }
                thread::sync_threads();
                let exponential = (unsafe { SCALAR[0] } - max_score).exp();

                unsafe {
                    PARTIALS[tid] = dy[query_base + tid] * v[key_base + tid];
                }
                thread::sync_threads();
                let mut stride = hd / 2;
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
                        SCALAR[0] = PARTIALS[0];
                    }
                }
                thread::sync_threads();
                denominator += exponential;
                softmax_dot_numerator += exponential * unsafe { SCALAR[0] };
            }
            let softmax_dot = softmax_dot_numerator / denominator;

            // Pass 3: regenerate each probability and dP, then update all
            // three gradients while the row is resident.
            let mut dq_value = 0.0f32;
            for key_position in 0..=query_position {
                let key_row = sequence_start + key_position;
                let key_base = (key_row * h + head) * hd;
                unsafe {
                    PARTIALS[tid] = q[query_base + tid] * k[key_base + tid];
                }
                thread::sync_threads();
                let mut stride = hd / 2;
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
                        SCALAR[0] = PARTIALS[0] * scale;
                    }
                }
                thread::sync_threads();
                let probability = (unsafe { SCALAR[0] } - max_score).exp() / denominator;

                unsafe {
                    PARTIALS[tid] = dy[query_base + tid] * v[key_base + tid];
                }
                thread::sync_threads();
                let mut stride = hd / 2;
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
                        SCALAR[0] = PARTIALS[0];
                    }
                }
                thread::sync_threads();

                let dscore = probability * (unsafe { SCALAR[0] } - softmax_dot) * scale;
                dq_value += dscore * k[key_base + tid];
                // SAFETY: this block/lane exclusively owns `key_base + tid`.
                unsafe {
                    *dk.get_unchecked_mut(key_base + tid) += dscore * q[query_base + tid];
                    *dv.get_unchecked_mut(key_base + tid) += probability * dy[query_base + tid];
                }
            }

            // SAFETY: this block/lane exclusively owns `query_base + tid`.
            unsafe {
                *dq.get_unchecked_mut(query_base + tid) = dq_value;
            }
        }
    }

    /// Query-parallel flash-attention backward for dQ.
    ///
    /// One block owns one `(query row, head)`, so each lane writes one unique
    /// dQ feature. Probabilities are regenerated from the saved row log-sum-exp.
    #[kernel]
    pub fn flash_attention_backward_q(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &[f32],
        dy: &[f32],
        logsumexp: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut dq: DisjointSlice<f32>,
    ) {
        static mut PARTIALS: SharedArray<f32, MAX_HEAD_DIM> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let hd = head_dim as usize;
        if hd == 0
            || hd > MAX_HEAD_DIM
            || !hd.is_power_of_two()
            || thread::blockDim_x() as usize != hd
        {
            return;
        }

        let t = sequence_length as usize;
        let h = heads as usize;
        let query_head = thread::blockIdx_x() as usize;
        let query_row = query_head / h;
        let head = query_head % h;
        let query_position = query_row % t;
        let sequence_start = query_row - query_position;
        let query_base = (query_row * h + head) * hd;
        let scale = 1.0 / (head_dim as f32).sqrt();

        unsafe {
            PARTIALS[tid] = dy[query_base + tid] * output[query_base + tid];
        }
        thread::sync_threads();
        let mut stride = hd / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIALS[tid] += PARTIALS[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        let softmax_dot = unsafe { PARTIALS[0] };
        let row_logsumexp = logsumexp[query_head];
        let mut dq_value = 0.0f32;

        for key_position in 0..=query_position {
            let key_row = sequence_start + key_position;
            let key_base = (key_row * h + head) * hd;
            // Every lane must finish its broadcast read of PARTIALS[0] above
            // before lane zero overwrites it for the next reduction.
            thread::sync_threads();
            unsafe {
                PARTIALS[tid] = q[query_base + tid] * k[key_base + tid];
            }
            thread::sync_threads();
            let mut stride = hd / 2;
            while stride > 0 {
                if tid < stride {
                    unsafe {
                        PARTIALS[tid] += PARTIALS[tid + stride];
                    }
                }
                thread::sync_threads();
                stride /= 2;
            }
            let probability = (unsafe { PARTIALS[0] } * scale - row_logsumexp).exp();

            // Same broadcast-read ordering as above.
            thread::sync_threads();
            unsafe {
                PARTIALS[tid] = dy[query_base + tid] * v[key_base + tid];
            }
            thread::sync_threads();
            let mut stride = hd / 2;
            while stride > 0 {
                if tid < stride {
                    unsafe {
                        PARTIALS[tid] += PARTIALS[tid + stride];
                    }
                }
                thread::sync_threads();
                stride /= 2;
            }
            let dscore = probability * (unsafe { PARTIALS[0] } - softmax_dot) * scale;
            dq_value += dscore * k[key_base + tid];
        }

        // SAFETY: one block owns `query_head`, and each lane owns one feature.
        unsafe {
            *dq.get_unchecked_mut(query_base + tid) = dq_value;
        }
    }

    /// Key-parallel flash-attention backward for dK and dV.
    ///
    /// One block owns one `(key row, head)` and scans only queries that can
    /// attend to that key. This exposes `rows * heads` blocks while avoiding
    /// atomics because every dK/dV feature has one owning block and lane.
    #[kernel]
    pub fn flash_attention_backward_kv(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &[f32],
        dy: &[f32],
        logsumexp: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut dk: DisjointSlice<f32>,
        mut dv: DisjointSlice<f32>,
    ) {
        static mut SCORE_PARTIALS: SharedArray<f32, MAX_HEAD_DIM> = SharedArray::UNINIT;
        static mut DOT_PARTIALS: SharedArray<f32, MAX_HEAD_DIM> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let hd = head_dim as usize;
        if hd == 0
            || hd > MAX_HEAD_DIM
            || !hd.is_power_of_two()
            || thread::blockDim_x() as usize != hd
        {
            return;
        }

        let t = sequence_length as usize;
        let h = heads as usize;
        let key_head = thread::blockIdx_x() as usize;
        let key_row = key_head / h;
        let head = key_head % h;
        let key_position = key_row % t;
        let sequence_start = key_row - key_position;
        let key_base = (key_row * h + head) * hd;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut dk_value = 0.0f32;
        let mut dv_value = 0.0f32;

        for query_position in key_position..t {
            let query_row = sequence_start + query_position;
            let query_head = query_row * h + head;
            let query_base = query_head * hd;
            // Every lane must finish its broadcast reads of the partials
            // below before lane zero overwrites them for this iteration.
            thread::sync_threads();
            unsafe {
                SCORE_PARTIALS[tid] = q[query_base + tid] * k[key_base + tid];
            }
            thread::sync_threads();
            let mut stride = hd / 2;
            while stride > 0 {
                if tid < stride {
                    unsafe {
                        SCORE_PARTIALS[tid] += SCORE_PARTIALS[tid + stride];
                    }
                }
                thread::sync_threads();
                stride /= 2;
            }
            let probability = (unsafe { SCORE_PARTIALS[0] } * scale - logsumexp[query_head]).exp();

            // Same broadcast-read ordering as above.
            thread::sync_threads();
            unsafe {
                SCORE_PARTIALS[tid] = dy[query_base + tid] * output[query_base + tid];
                DOT_PARTIALS[tid] = dy[query_base + tid] * v[key_base + tid];
            }
            thread::sync_threads();
            let mut stride = hd / 2;
            while stride > 0 {
                if tid < stride {
                    unsafe {
                        SCORE_PARTIALS[tid] += SCORE_PARTIALS[tid + stride];
                        DOT_PARTIALS[tid] += DOT_PARTIALS[tid + stride];
                    }
                }
                thread::sync_threads();
                stride /= 2;
            }

            let dscore = probability * (unsafe { DOT_PARTIALS[0] - SCORE_PARTIALS[0] }) * scale;
            dk_value += dscore * q[query_base + tid];
            dv_value += probability * dy[query_base + tid];
        }

        // SAFETY: one block owns `key_head`, and each lane owns one feature.
        unsafe {
            *dk.get_unchecked_mut(key_base + tid) = dk_value;
            *dv.get_unchecked_mut(key_base + tid) = dv_value;
        }
    }

    /// FlashAttention-2 style tiled causal forward.
    ///
    /// One block owns `(sequence, head, query block)` and streams key/value
    /// tiles through shared memory, computing each `FWD_BQ x FWD_BK` score
    /// tile as register fragments and folding it into the online softmax.
    /// Launch with [`tiled_forward_config`].
    #[kernel]
    pub fn flash_attention_forward_tiled(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sequence_length: u32,
        heads: u32,
        mut output: DisjointSlice<f32>,
        mut logsumexp: DisjointSlice<f32>,
    ) {
        static mut Q_TILE: SharedArray<f32, { FWD_BQ * TILE_HD }> = SharedArray::UNINIT;
        static mut K_TILE: SharedArray<f32, { FWD_BK * TILE_HD }> = SharedArray::UNINIT;
        static mut V_TILE: SharedArray<f32, { FWD_BK * TILE_HD }> = SharedArray::UNINIT;
        static mut P_TILE: SharedArray<f32, { FWD_BQ * FWD_BK }> = SharedArray::UNINIT;
        static mut ROW_MAX: SharedArray<f32, FWD_BQ> = SharedArray::UNINIT;
        static mut ROW_SUM: SharedArray<f32, FWD_BQ> = SharedArray::UNINIT;
        static mut ROW_RESCALE: SharedArray<f32, FWD_BQ> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != FWD_THREADS {
            return;
        }
        let t = sequence_length as usize;
        let h = heads as usize;
        let head = thread::blockIdx_y() as usize;
        let sequence_start = thread::blockIdx_z() as usize * t;
        let query_start = thread::blockIdx_x() as usize * FWD_BQ;
        if query_start >= t {
            return;
        }
        let scale = 1.0 / (TILE_HD as f32).sqrt();

        // Stage this block's query tile once; rows past the sequence read zero
        // and are masked out of every store below.
        let mut index = tid;
        while index < FWD_BQ * TILE_HD {
            let row = index / TILE_HD;
            let feature = index % TILE_HD;
            let query_row = query_start + row;
            unsafe {
                Q_TILE[index] = if query_row < t {
                    q[((sequence_start + query_row) * h + head) * TILE_HD + feature]
                } else {
                    0.0
                };
            }
            index += FWD_THREADS;
        }
        index = tid;
        while index < FWD_BQ {
            unsafe {
                ROW_MAX[index] = MASKED_SCORE;
                ROW_SUM[index] = 0.0;
            }
            index += FWD_THREADS;
        }

        let fragment_row = tid / FWD_COL_THREADS;
        let fragment_col = tid % FWD_COL_THREADS;
        let mut out_acc = [[0.0f32; FWD_TD]; FWD_TM];

        let last_query = if query_start + FWD_BQ < t {
            query_start + FWD_BQ - 1
        } else {
            t - 1
        };
        let key_tiles = last_query / FWD_BK + 1;
        for key_tile in 0..key_tiles {
            let key_start = key_tile * FWD_BK;
            index = tid;
            while index < FWD_BK * TILE_HD {
                let row = index / TILE_HD;
                let feature = index % TILE_HD;
                let key_row = key_start + row;
                unsafe {
                    K_TILE[index] = if key_row < t {
                        k[((sequence_start + key_row) * h + head) * TILE_HD + feature]
                    } else {
                        0.0
                    };
                }
                index += FWD_THREADS;
            }
            thread::sync_threads();

            let mut scores = [[0.0f32; FWD_TN]; FWD_TM];
            for feature in 0..TILE_HD {
                for i in 0..FWD_TM {
                    let query_value =
                        unsafe { Q_TILE[(fragment_row * FWD_TM + i) * TILE_HD + feature] };
                    for j in 0..FWD_TN {
                        scores[i][j] += query_value
                            * unsafe { K_TILE[(fragment_col * FWD_TN + j) * TILE_HD + feature] };
                    }
                }
            }
            for i in 0..FWD_TM {
                let row = fragment_row * FWD_TM + i;
                let query_row = query_start + row;
                for j in 0..FWD_TN {
                    let col = fragment_col * FWD_TN + j;
                    let key_row = key_start + col;
                    unsafe {
                        P_TILE[row * FWD_BK + col] = if key_row <= query_row && key_row < t {
                            scores[i][j] * scale
                        } else {
                            MASKED_SCORE
                        };
                    }
                }
            }
            thread::sync_threads();

            // Online-softmax row statistics for this tile, while the value
            // tile (only needed after the probabilities) streams in.
            index = tid;
            while index < FWD_BQ {
                unsafe {
                    let mut tile_max = MASKED_SCORE;
                    for col in 0..FWD_BK {
                        tile_max = tile_max.max(P_TILE[index * FWD_BK + col]);
                    }
                    let next_max = ROW_MAX[index].max(tile_max);
                    ROW_RESCALE[index] = (ROW_MAX[index] - next_max).exp();
                    ROW_MAX[index] = next_max;
                }
                index += FWD_THREADS;
            }
            index = tid;
            while index < FWD_BK * TILE_HD {
                let row = index / TILE_HD;
                let feature = index % TILE_HD;
                let key_row = key_start + row;
                unsafe {
                    V_TILE[index] = if key_row < t {
                        v[((sequence_start + key_row) * h + head) * TILE_HD + feature]
                    } else {
                        0.0
                    };
                }
                index += FWD_THREADS;
            }
            thread::sync_threads();

            index = tid;
            while index < FWD_BQ * FWD_BK {
                let row = index / FWD_BK;
                unsafe {
                    P_TILE[index] = (P_TILE[index] - ROW_MAX[row]).exp();
                }
                index += FWD_THREADS;
            }
            thread::sync_threads();

            index = tid;
            while index < FWD_BQ {
                unsafe {
                    let mut tile_sum = 0.0f32;
                    for col in 0..FWD_BK {
                        tile_sum += P_TILE[index * FWD_BK + col];
                    }
                    ROW_SUM[index] = ROW_RESCALE[index] * ROW_SUM[index] + tile_sum;
                }
                index += FWD_THREADS;
            }
            for i in 0..FWD_TM {
                let rescale = unsafe { ROW_RESCALE[fragment_row * FWD_TM + i] };
                for d in 0..FWD_TD {
                    out_acc[i][d] *= rescale;
                }
            }
            for col in 0..FWD_BK {
                let mut value = [0.0f32; FWD_TD];
                for d in 0..FWD_TD {
                    value[d] = unsafe { V_TILE[col * TILE_HD + fragment_col * FWD_TD + d] };
                }
                for i in 0..FWD_TM {
                    let probability = unsafe { P_TILE[(fragment_row * FWD_TM + i) * FWD_BK + col] };
                    for d in 0..FWD_TD {
                        out_acc[i][d] += probability * value[d];
                    }
                }
            }
            thread::sync_threads();
        }

        for i in 0..FWD_TM {
            let row = fragment_row * FWD_TM + i;
            let query_row = query_start + row;
            if query_row < t {
                let inverse_sum = 1.0 / unsafe { ROW_SUM[row] };
                for d in 0..FWD_TD {
                    let feature = fragment_col * FWD_TD + d;
                    // SAFETY: `(sequence, head, query block)` blocks are
                    // disjoint and each thread owns disjoint (row, feature)
                    // fragments within the block.
                    unsafe {
                        *output.get_unchecked_mut(
                            ((sequence_start + query_row) * h + head) * TILE_HD + feature,
                        ) = out_acc[i][d] * inverse_sum;
                    }
                }
            }
        }
        index = tid;
        while index < FWD_BQ {
            let query_row = query_start + index;
            if query_row < t {
                // SAFETY: one block owns each of its query rows' statistics.
                unsafe {
                    *logsumexp.get_unchecked_mut((sequence_start + query_row) * h + head) =
                        ROW_MAX[index] + ROW_SUM[index].ln();
                }
            }
            index += FWD_THREADS;
        }
    }

    /// Per-`(row, head)` softmax dot `sum_d dy * y`, staged once so both tiled
    /// backward kernels read it instead of re-reducing it per tile.
    #[kernel]
    pub fn flash_attention_backward_dot(
        dy: &[f32],
        output: &[f32],
        head_dim: u32,
        mut dot: DisjointSlice<f32>,
    ) {
        static mut PARTIALS: SharedArray<f32, MAX_HEAD_DIM> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let hd = head_dim as usize;
        if hd == 0
            || hd > MAX_HEAD_DIM
            || !hd.is_power_of_two()
            || thread::blockDim_x() as usize != hd
        {
            return;
        }
        let row_head = thread::blockIdx_x() as usize;
        let base = row_head * hd;
        unsafe {
            PARTIALS[tid] = dy[base + tid] * output[base + tid];
        }
        thread::sync_threads();
        let mut stride = hd / 2;
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
            // SAFETY: one block owns each `row_head` dot.
            unsafe {
                *dot.get_unchecked_mut(row_head) = PARTIALS[0];
            }
        }
    }

    /// FlashAttention-2 style tiled dQ.
    ///
    /// One block owns `(sequence, head, query block)`; probabilities are
    /// regenerated from the saved log-sum-exp and the pre-staged `dy . y`
    /// dots, so no `[N,H,T]` state exists. Launch with
    /// [`tiled_backward_q_config`].
    #[kernel]
    pub fn flash_attention_backward_q_tiled(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        dy: &[f32],
        logsumexp: &[f32],
        dot: &[f32],
        sequence_length: u32,
        heads: u32,
        mut dq: DisjointSlice<f32>,
    ) {
        static mut Q_TILE: SharedArray<f32, { BWQ_BQ * TILE_HD }> = SharedArray::UNINIT;
        static mut DY_TILE: SharedArray<f32, { BWQ_BQ * TILE_HD }> = SharedArray::UNINIT;
        static mut K_TILE: SharedArray<f32, { BWQ_BK * TILE_HD }> = SharedArray::UNINIT;
        static mut V_TILE: SharedArray<f32, { BWQ_BK * TILE_HD }> = SharedArray::UNINIT;
        static mut DS_TILE: SharedArray<f32, { BWQ_BQ * BWQ_BK }> = SharedArray::UNINIT;
        static mut ROW_LSE: SharedArray<f32, BWQ_BQ> = SharedArray::UNINIT;
        static mut ROW_DOT: SharedArray<f32, BWQ_BQ> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != BWQ_THREADS {
            return;
        }
        let t = sequence_length as usize;
        let h = heads as usize;
        let head = thread::blockIdx_y() as usize;
        let sequence_start = thread::blockIdx_z() as usize * t;
        let query_start = thread::blockIdx_x() as usize * BWQ_BQ;
        if query_start >= t {
            return;
        }
        let scale = 1.0 / (TILE_HD as f32).sqrt();

        let mut index = tid;
        while index < BWQ_BQ * TILE_HD {
            let row = index / TILE_HD;
            let feature = index % TILE_HD;
            let query_row = query_start + row;
            unsafe {
                if query_row < t {
                    let base = ((sequence_start + query_row) * h + head) * TILE_HD;
                    Q_TILE[index] = q[base + feature];
                    DY_TILE[index] = dy[base + feature];
                } else {
                    Q_TILE[index] = 0.0;
                    DY_TILE[index] = 0.0;
                }
            }
            index += BWQ_THREADS;
        }
        index = tid;
        while index < BWQ_BQ {
            let query_row = query_start + index;
            unsafe {
                if query_row < t {
                    let row_head = (sequence_start + query_row) * h + head;
                    ROW_LSE[index] = logsumexp[row_head];
                    ROW_DOT[index] = dot[row_head];
                } else {
                    ROW_LSE[index] = 0.0;
                    ROW_DOT[index] = 0.0;
                }
            }
            index += BWQ_THREADS;
        }

        let fragment_row = tid / BWQ_COL_THREADS;
        let fragment_col = tid % BWQ_COL_THREADS;
        let mut dq_acc = [[0.0f32; BWQ_TD]; BWQ_TM];

        let last_query = if query_start + BWQ_BQ < t {
            query_start + BWQ_BQ - 1
        } else {
            t - 1
        };
        let key_tiles = last_query / BWQ_BK + 1;
        for key_tile in 0..key_tiles {
            let key_start = key_tile * BWQ_BK;
            index = tid;
            while index < BWQ_BK * TILE_HD {
                let row = index / TILE_HD;
                let feature = index % TILE_HD;
                let key_row = key_start + row;
                unsafe {
                    if key_row < t {
                        let base = ((sequence_start + key_row) * h + head) * TILE_HD;
                        K_TILE[index] = k[base + feature];
                        V_TILE[index] = v[base + feature];
                    } else {
                        K_TILE[index] = 0.0;
                        V_TILE[index] = 0.0;
                    }
                }
                index += BWQ_THREADS;
            }
            thread::sync_threads();

            let mut scores = [[0.0f32; BWQ_TN]; BWQ_TM];
            let mut dp = [[0.0f32; BWQ_TN]; BWQ_TM];
            for feature in 0..TILE_HD {
                for i in 0..BWQ_TM {
                    let row_offset = (fragment_row * BWQ_TM + i) * TILE_HD + feature;
                    let query_value = unsafe { Q_TILE[row_offset] };
                    let dy_value = unsafe { DY_TILE[row_offset] };
                    for j in 0..BWQ_TN {
                        let col_offset = (fragment_col * BWQ_TN + j) * TILE_HD + feature;
                        unsafe {
                            scores[i][j] += query_value * K_TILE[col_offset];
                            dp[i][j] += dy_value * V_TILE[col_offset];
                        }
                    }
                }
            }
            for i in 0..BWQ_TM {
                let row = fragment_row * BWQ_TM + i;
                let query_row = query_start + row;
                for j in 0..BWQ_TN {
                    let col = fragment_col * BWQ_TN + j;
                    let key_row = key_start + col;
                    let dscore = if key_row <= query_row && key_row < t && query_row < t {
                        let probability = unsafe { (scores[i][j] * scale - ROW_LSE[row]).exp() };
                        probability * (dp[i][j] - unsafe { ROW_DOT[row] }) * scale
                    } else {
                        0.0
                    };
                    unsafe {
                        DS_TILE[row * BWQ_BK + col] = dscore;
                    }
                }
            }
            thread::sync_threads();

            for col in 0..BWQ_BK {
                let mut key_value = [0.0f32; BWQ_TD];
                for d in 0..BWQ_TD {
                    key_value[d] = unsafe { K_TILE[col * TILE_HD + fragment_col * BWQ_TD + d] };
                }
                for i in 0..BWQ_TM {
                    let dscore = unsafe { DS_TILE[(fragment_row * BWQ_TM + i) * BWQ_BK + col] };
                    for d in 0..BWQ_TD {
                        dq_acc[i][d] += dscore * key_value[d];
                    }
                }
            }
            thread::sync_threads();
        }

        for i in 0..BWQ_TM {
            let row = fragment_row * BWQ_TM + i;
            let query_row = query_start + row;
            if query_row < t {
                for d in 0..BWQ_TD {
                    let feature = fragment_col * BWQ_TD + d;
                    // SAFETY: blocks own disjoint query tiles and each thread
                    // owns disjoint (row, feature) fragments within its block.
                    unsafe {
                        *dq.get_unchecked_mut(
                            ((sequence_start + query_row) * h + head) * TILE_HD + feature,
                        ) = dq_acc[i][d];
                    }
                }
            }
        }
    }

    /// FlashAttention-2 style tiled dK/dV.
    ///
    /// One block owns `(sequence, head, key block)` and scans only the query
    /// tiles that can attend to its keys, accumulating both gradients in
    /// registers so no atomics are needed. Launch with
    /// [`tiled_backward_kv_config`].
    #[kernel]
    pub fn flash_attention_backward_kv_tiled(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        dy: &[f32],
        logsumexp: &[f32],
        dot: &[f32],
        sequence_length: u32,
        heads: u32,
        mut dk: DisjointSlice<f32>,
        mut dv: DisjointSlice<f32>,
    ) {
        static mut K_TILE: SharedArray<f32, { KV_BK * TILE_HD }> = SharedArray::UNINIT;
        static mut V_TILE: SharedArray<f32, { KV_BK * TILE_HD }> = SharedArray::UNINIT;
        static mut Q_TILE: SharedArray<f32, { KV_BQ * TILE_HD }> = SharedArray::UNINIT;
        static mut DY_TILE: SharedArray<f32, { KV_BQ * TILE_HD }> = SharedArray::UNINIT;
        static mut P_TILE: SharedArray<f32, { KV_BQ * KV_BK }> = SharedArray::UNINIT;
        static mut DS_TILE: SharedArray<f32, { KV_BQ * KV_BK }> = SharedArray::UNINIT;
        static mut ROW_LSE: SharedArray<f32, KV_BQ> = SharedArray::UNINIT;
        static mut ROW_DOT: SharedArray<f32, KV_BQ> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        if thread::blockDim_x() as usize != KV_THREADS {
            return;
        }
        let t = sequence_length as usize;
        let h = heads as usize;
        let head = thread::blockIdx_y() as usize;
        let sequence_start = thread::blockIdx_z() as usize * t;
        let key_start = thread::blockIdx_x() as usize * KV_BK;
        if key_start >= t {
            return;
        }
        let scale = 1.0 / (TILE_HD as f32).sqrt();

        let mut index = tid;
        while index < KV_BK * TILE_HD {
            let row = index / TILE_HD;
            let feature = index % TILE_HD;
            let key_row = key_start + row;
            unsafe {
                if key_row < t {
                    let base = ((sequence_start + key_row) * h + head) * TILE_HD;
                    K_TILE[index] = k[base + feature];
                    V_TILE[index] = v[base + feature];
                } else {
                    K_TILE[index] = 0.0;
                    V_TILE[index] = 0.0;
                }
            }
            index += KV_THREADS;
        }

        let fragment_row = tid / KV_COL_THREADS;
        let fragment_col = tid % KV_COL_THREADS;
        let mut dk_acc = [[0.0f32; KV_ACC_D]; KV_ACC_M];
        let mut dv_acc = [[0.0f32; KV_ACC_D]; KV_ACC_M];

        let query_tiles = (t + KV_BQ - 1) / KV_BQ;
        for query_tile in (key_start / KV_BQ)..query_tiles {
            let tile_query_start = query_tile * KV_BQ;
            index = tid;
            while index < KV_BQ * TILE_HD {
                let row = index / TILE_HD;
                let feature = index % TILE_HD;
                let query_row = tile_query_start + row;
                unsafe {
                    if query_row < t {
                        let base = ((sequence_start + query_row) * h + head) * TILE_HD;
                        Q_TILE[index] = q[base + feature];
                        DY_TILE[index] = dy[base + feature];
                    } else {
                        Q_TILE[index] = 0.0;
                        DY_TILE[index] = 0.0;
                    }
                }
                index += KV_THREADS;
            }
            index = tid;
            while index < KV_BQ {
                let query_row = tile_query_start + index;
                unsafe {
                    if query_row < t {
                        let row_head = (sequence_start + query_row) * h + head;
                        ROW_LSE[index] = logsumexp[row_head];
                        ROW_DOT[index] = dot[row_head];
                    } else {
                        ROW_LSE[index] = 0.0;
                        ROW_DOT[index] = 0.0;
                    }
                }
                index += KV_THREADS;
            }
            thread::sync_threads();

            let mut scores = [[0.0f32; KV_TN]; KV_TM];
            let mut dp = [[0.0f32; KV_TN]; KV_TM];
            for feature in 0..TILE_HD {
                for i in 0..KV_TM {
                    let row_offset = (fragment_row * KV_TM + i) * TILE_HD + feature;
                    let query_value = unsafe { Q_TILE[row_offset] };
                    let dy_value = unsafe { DY_TILE[row_offset] };
                    for j in 0..KV_TN {
                        let col_offset = (fragment_col * KV_TN + j) * TILE_HD + feature;
                        unsafe {
                            scores[i][j] += query_value * K_TILE[col_offset];
                            dp[i][j] += dy_value * V_TILE[col_offset];
                        }
                    }
                }
            }
            for i in 0..KV_TM {
                let row = fragment_row * KV_TM + i;
                let query_row = tile_query_start + row;
                for j in 0..KV_TN {
                    let col = fragment_col * KV_TN + j;
                    let key_row = key_start + col;
                    let mut probability = 0.0f32;
                    let mut dscore = 0.0f32;
                    if key_row <= query_row && key_row < t && query_row < t {
                        probability = unsafe { (scores[i][j] * scale - ROW_LSE[row]).exp() };
                        dscore = probability * (dp[i][j] - unsafe { ROW_DOT[row] }) * scale;
                    }
                    unsafe {
                        P_TILE[row * KV_BK + col] = probability;
                        DS_TILE[row * KV_BK + col] = dscore;
                    }
                }
            }
            thread::sync_threads();

            for row in 0..KV_BQ {
                let mut query_value = [0.0f32; KV_ACC_D];
                let mut dy_value = [0.0f32; KV_ACC_D];
                for d in 0..KV_ACC_D {
                    let offset = row * TILE_HD + fragment_col * KV_ACC_D + d;
                    query_value[d] = unsafe { Q_TILE[offset] };
                    dy_value[d] = unsafe { DY_TILE[offset] };
                }
                for i in 0..KV_ACC_M {
                    let col = fragment_row * KV_ACC_M + i;
                    let probability = unsafe { P_TILE[row * KV_BK + col] };
                    let dscore = unsafe { DS_TILE[row * KV_BK + col] };
                    for d in 0..KV_ACC_D {
                        dv_acc[i][d] += probability * dy_value[d];
                        dk_acc[i][d] += dscore * query_value[d];
                    }
                }
            }
            thread::sync_threads();
        }

        for i in 0..KV_ACC_M {
            let key_row = key_start + fragment_row * KV_ACC_M + i;
            if key_row < t {
                for d in 0..KV_ACC_D {
                    let feature = fragment_col * KV_ACC_D + d;
                    let global = ((sequence_start + key_row) * h + head) * TILE_HD + feature;
                    // SAFETY: blocks own disjoint key tiles and each thread
                    // owns disjoint (key row, feature) fragments within its
                    // block.
                    unsafe {
                        *dk.get_unchecked_mut(global) = dk_acc[i][d];
                        *dv.get_unchecked_mut(global) = dv_acc[i][d];
                    }
                }
            }
        }
    }
}

/// Shared launch checks for one tiled kernel family.
fn tiled_config(
    query_or_key_blocks: usize,
    sequences: usize,
    heads: usize,
    head_dim: usize,
    threads: usize,
    shared_floats: usize,
) -> LaunchConfig {
    assert_eq!(head_dim, TILE_HD, "tiled kernels specialize on TILE_HD");
    assert!(threads > 0 && threads <= 1024);
    assert!(shared_floats * size_of::<f32>() <= STATIC_SHARED_LIMIT);
    assert!(query_or_key_blocks <= u32::MAX as usize);
    assert!(heads <= u16::MAX as usize && sequences <= u16::MAX as usize);
    LaunchConfig {
        grid_dim: (query_or_key_blocks as u32, heads as u32, sequences as u32),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch shape for [`kernels::flash_attention_forward_tiled`] over
/// `sequences` packed sequences of `sequence_length` rows.
pub fn tiled_forward_config(
    sequences: usize,
    sequence_length: usize,
    heads: usize,
    head_dim: usize,
) -> LaunchConfig {
    assert!(FWD_BQ.is_multiple_of(FWD_TM) && FWD_BK.is_multiple_of(FWD_TN));
    assert!(TILE_HD.is_multiple_of(FWD_COL_THREADS));
    tiled_config(
        sequence_length.div_ceil(FWD_BQ),
        sequences,
        heads,
        head_dim,
        FWD_THREADS,
        FWD_BQ * TILE_HD + 2 * FWD_BK * TILE_HD + FWD_BQ * FWD_BK + 3 * FWD_BQ,
    )
}

/// Launch shape for [`kernels::flash_attention_backward_dot`]: one block of
/// `head_dim` lanes per `(row, head)`.
pub fn dot_config(rows: usize, heads: usize, head_dim: usize) -> LaunchConfig {
    assert!(head_dim.is_power_of_two() && head_dim <= MAX_HEAD_DIM);
    assert!(rows * heads <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: ((rows * heads) as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch shape for [`kernels::flash_attention_backward_q_tiled`].
pub fn tiled_backward_q_config(
    sequences: usize,
    sequence_length: usize,
    heads: usize,
    head_dim: usize,
) -> LaunchConfig {
    assert!(BWQ_BQ.is_multiple_of(BWQ_TM) && BWQ_BK.is_multiple_of(BWQ_TN));
    assert!(TILE_HD.is_multiple_of(BWQ_COL_THREADS));
    tiled_config(
        sequence_length.div_ceil(BWQ_BQ),
        sequences,
        heads,
        head_dim,
        BWQ_THREADS,
        2 * BWQ_BQ * TILE_HD + 2 * BWQ_BK * TILE_HD + BWQ_BQ * BWQ_BK + 2 * BWQ_BQ,
    )
}

/// Launch shape for [`kernels::stage_attention_heads_bf16`]: one thread per
/// packed bf16 output pair.
pub fn stage_heads_config(rows: usize, heads: usize, head_dim: usize) -> LaunchConfig {
    assert_eq!(head_dim, TILE_HD, "staging specializes on TILE_HD");
    let words = rows * heads * head_dim / 2;
    assert!(words <= u32::MAX as usize);
    LaunchConfig::for_num_elems(words as u32)
}

/// Launch shape for [`kernels::flash_attention_backward_kv_tiled`].
pub fn tiled_backward_kv_config(
    sequences: usize,
    sequence_length: usize,
    heads: usize,
    head_dim: usize,
) -> LaunchConfig {
    assert!(KV_BQ.is_multiple_of(KV_TM) && KV_BK.is_multiple_of(KV_TN));
    assert!(TILE_HD.is_multiple_of(KV_COL_THREADS));
    assert!(KV_BK.is_multiple_of(KV_ROW_THREADS));
    tiled_config(
        sequence_length.div_ceil(KV_BK),
        sequences,
        heads,
        head_dim,
        KV_THREADS,
        2 * KV_BK * TILE_HD + 2 * KV_BQ * TILE_HD + 2 * KV_BQ * KV_BK + 2 * KV_BQ,
    )
}
