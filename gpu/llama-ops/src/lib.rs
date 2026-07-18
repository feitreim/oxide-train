//! Reference CUDA kernels for the first Llama modules.
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

/// Sentinel written by deterministic MoE binning for a capacity-dropped pair.
pub const MOE_DROPPED_SLOT: u32 = u32::MAX;

#[cuda_module]
pub mod kernels {
    use super::*;

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
        let row = i / d;
        let base = row * d;
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
        let row = i / d;
        let base = row * d;
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

    #[kernel]
    pub fn swiglu_forward(gate: &[f32], up: &[f32], mut y: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = y.get_mut(index) {
            let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
            *slot = gate[i] * sigmoid * up[i];
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

    #[kernel]
    pub fn embedding_forward(weight: &[f32], tokens: &[u32], dim: u32, mut y: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = y.get_mut(index) {
            let d = dim as usize;
            let row = i / d;
            let col = i % d;
            *slot = weight[tokens[row] as usize * d + col];
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
        let row = i / c;
        let base = row * c;
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

    #[kernel]
    pub fn rope_forward(
        x: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = y.get_mut(index) {
            let hd = head_dim as usize;
            let col = i % hd;
            let row = i / (heads as usize * hd);
            let position = row % sequence_length as usize;
            let pair = col / 2;
            let frequency = 10_000.0f32.powf(-((2 * pair) as f32) / head_dim as f32);
            let angle = position as f32 * frequency;
            let sin = angle.sin();
            let cos = angle.cos();
            let base = i - col % 2;
            *slot = if col % 2 == 0 {
                x[base] * cos - x[base + 1] * sin
            } else {
                x[base] * sin + x[base + 1] * cos
            };
        }
    }

    #[kernel]
    pub fn rope_backward(
        dy: &[f32],
        sequence_length: u32,
        heads: u32,
        head_dim: u32,
        mut dx: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = dx.get_mut(index) {
            let hd = head_dim as usize;
            let col = i % hd;
            let row = i / (heads as usize * hd);
            let position = row % sequence_length as usize;
            let pair = col / 2;
            let frequency = 10_000.0f32.powf(-((2 * pair) as f32) / head_dim as f32);
            let angle = position as f32 * frequency;
            let sin = angle.sin();
            let cos = angle.cos();
            let base = i - col % 2;
            *slot = if col % 2 == 0 {
                dy[base] * cos + dy[base + 1] * sin
            } else {
                -dy[base] * sin + dy[base + 1] * cos
            };
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

    /// Router logits for a skinny `[N,D] x [D,E]` fp32 matrix multiply.
    ///
    /// One block owns a token row and one lane owns each (small) expert.
    #[kernel]
    pub fn router_logits(
        x: &[f32],
        weight: &[f32],
        dim: u32,
        experts: u32,
        mut logits: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x() as usize;
        let expert = thread::threadIdx_x() as usize;
        let d = dim as usize;
        let e = experts as usize;
        if expert >= e || token * e + expert >= logits.len() {
            return;
        }
        let mut value = 0.0f32;
        for column in 0..d {
            value += x[token * d + column] * weight[column * e + expert];
        }
        if let Some(slot) = logits.get_mut(thread::index_1d()) {
            *slot = value;
        }
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

    /// Scatter `gate * dy` to expert-output order and compute one gate gradient
    /// dot product per accepted pair.
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
        let pair = thread::index_1d().get();
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
        let mut gate_gradient = 0.0f32;
        if slot != MOE_DROPPED_SLOT {
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
            for column in 0..d {
                gate_gradient += expert_output[bin_base + column] * dy[token_base + column];
                unsafe {
                    *expert_output_gradient.get_unchecked_mut(bin_base + column) =
                        gate_weights[pair] * dy[token_base + column];
                }
            }
        }
        unsafe {
            *gate_gradients.get_unchecked_mut(pair) = gate_gradient;
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

    /// Router linear backward with respect to its input.
    #[kernel]
    pub fn router_backward_input(
        dlogits: &[f32],
        weight: &[f32],
        experts: u32,
        mut dx: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        let e = experts as usize;
        if i >= dx.len() || e == 0 {
            return;
        }
        let d = weight.len() / e;
        let token = i / d;
        let column = i % d;
        let mut value = 0.0f32;
        for expert in 0..e {
            value += dlogits[token * e + expert] * weight[column * e + expert];
        }
        if let Some(slot) = dx.get_mut(index) {
            *slot = value;
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
            *slot = value;
        }
    }
}
