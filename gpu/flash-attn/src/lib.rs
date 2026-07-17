//! Fused fp32 causal attention kernels.
//!
//! Forward applies an online softmax while streaming over K/V, so its only
//! output-sized storage is `[B*T,H,HD]`; it never materializes `[B*T,H,T]`
//! probabilities. Forward saves only one log-sum-exp scalar per row/head.
//! Backward recomputes probabilities with query-parallel dQ blocks and
//! key-parallel dK/dV blocks, making all gradient writes disjoint.
//!
//! The launch contract is:
//! - `head_dim` is a power of two and at most `MAX_HEAD_DIM`;
//! - `block_dim.x == head_dim`, with all other block/grid dimensions equal to 1;
//! - forward launches `rows * heads` blocks;
//! - tiled backward launches `rows * heads` blocks for each of dQ and dK/dV.

use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

/// Maximum supported head width. This bounds the statically allocated shared
/// reduction buffer; actual launches use exactly `head_dim` threads.
pub const MAX_HEAD_DIM: usize = 256;

#[cuda_module]
pub mod kernels {
    use super::*;

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
}
