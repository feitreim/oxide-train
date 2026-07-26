//! Static-shape GPU tensor storage and foundational fp32 kernels.
//!
//! `GpuTensor<E, S>` deliberately wraps only storage. Operations remain
//! inherent methods and take an explicit stream and loaded kernel module; no
//! device dispatch or implicit synchronization is hidden behind `Tensor`.

use std::marker::PhantomData;

use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use tensor_core::{Element, Rank1, Rank2, Shape, Tensor, bf16};
use tensor_cpu::CpuTensor;

/// GEMM tile edge and launch block dimensions. This is intentionally public so
/// the repository sweep harness can rewrite it.
pub const TILE: usize = 16;
/// Threads in the single-block reduction kernels. Must remain a power of two.
pub const REDUCE_THREADS: usize = 256;
/// `rounding` selector for the fused bf16-master kernels: round to nearest even.
pub const MASTER_ROUNDING_NEAREST: u32 = 0;
/// `rounding` selector for the fused bf16-master kernels: stochastic rounding.
pub const MASTER_ROUNDING_STOCHASTIC: u32 = 1;
const TILE_ELEMENTS: usize = TILE * TILE;
/// Square element-tile edge of the packed-bf16 transpose kernel. Both matrix
/// dimensions must be multiples of this.
pub const TRANSPOSE_TILE: usize = 64;
const TRANSPOSE_THREADS: usize = 256;
const TRANSPOSE_WORDS: usize = TRANSPOSE_TILE * TRANSPOSE_TILE / 2;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = out.get_mut(index) {
            *slot = a[i] + b[i];
        }
    }

    #[kernel]
    pub fn mul(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = out.get_mut(index) {
            *slot = a[i] * b[i];
        }
    }

    #[kernel]
    pub fn scale(a: &[f32], factor: f32, mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = out.get_mut(index) {
            *slot = a[i] * factor;
        }
    }

    /// Fill an existing buffer without allocating replacement storage.
    #[kernel]
    pub fn fill(value: f32, mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        if let Some(slot) = out.get_mut(index) {
            *slot = value;
        }
    }

    /// `dst += factor * src`, used by gradient accumulation and optimizers.
    #[kernel]
    pub fn add_scaled(src: &[f32], factor: f32, mut dst: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = dst.get_mut(index) {
            *slot += factor * src[i];
        }
    }

    /// Fused decoupled AdamW update over one flat parameter buffer.
    #[kernel]
    pub fn adamw(
        gradient: &[f32],
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        first_correction: f32,
        second_correction: f32,
        mut parameter: DisjointSlice<f32>,
        mut first: DisjointSlice<f32>,
        mut second: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let Some(parameter) = parameter.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(first) = first.get_mut(thread::index_1d()) else {
            return;
        };
        let Some(second) = second.get_mut(thread::index_1d()) else {
            return;
        };

        *first = beta1 * *first + (1.0 - beta1) * gradient[i];
        *second = beta2 * *second + (1.0 - beta2) * gradient[i] * gradient[i];
        let first_hat = *first * first_correction;
        let second_hat = *second * second_correction;
        let update = first_hat / (second_hat.sqrt() + epsilon) + weight_decay * *parameter;
        *parameter -= learning_rate * update;
    }

    /// Muon's EMA momentum update: `m = beta * m + (1 - beta) * g`.
    #[kernel]
    pub fn ema_momentum(gradient: &[f32], beta: f32, mut momentum: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = momentum.get_mut(index) {
            *slot = beta * *slot + (1.0 - beta) * gradient[i];
        }
    }

    /// `out = alpha * a + beta * b` over the first `len` elements.
    ///
    /// All three buffers may be longer than `len` (Muon scratch prefixes);
    /// elements past `len` are untouched.
    #[kernel]
    pub fn scaled_sum(
        alpha: f32,
        a: &[f32],
        beta: f32,
        b: &[f32],
        len: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= len as usize {
            return;
        }
        if let Some(slot) = out.get_mut(index) {
            *slot = alpha * a[i] + beta * b[i];
        }
    }

    /// Sum of squares of the first `len` elements, accumulated in f64 to
    /// match the CPU reference's Frobenius-norm accumulator.
    #[kernel]
    pub fn sum_squares(a: &[f32], len: u32, mut out: DisjointSlice<f32>) {
        static mut PARTIALS: SharedArray<f64, REDUCE_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let mut i = tid;
        let mut partial = 0.0f64;
        while i < len as usize {
            let value = a[i] as f64;
            partial += value * value;
            i += REDUCE_THREADS;
        }
        unsafe {
            PARTIALS[tid] = partial;
        }
        thread::sync_threads();

        let mut stride = REDUCE_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIALS[tid] += PARTIALS[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        let index = thread::index_1d();
        if tid == 0
            && let Some(slot) = out.get_mut(index)
        {
            unsafe {
                *slot = PARTIALS[0] as f32;
            }
        }
    }

    /// `out = input / (sqrt(sum_squares[0]) + epsilon)` over the first `len`
    /// elements: Muon's Newton–Schulz pre-normalization, with the norm left
    /// device-resident so no step synchronizes the stream.
    #[kernel]
    pub fn scale_by_inv_norm(
        input: &[f32],
        sum_squares: &[f32],
        epsilon: f32,
        len: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= len as usize {
            return;
        }
        let scale = 1.0 / (sum_squares[0].sqrt() + epsilon);
        if let Some(slot) = out.get_mut(index) {
            *slot = input[i] * scale;
        }
    }

    /// Copy group `group` of a `[rows, groups, width]` tensor into a dense
    /// `[rows, width]` prefix of `out`. `len` is `rows * width`. With
    /// `groups = 1` this is a plain device copy.
    #[kernel]
    pub fn gather_group(
        input: &[f32],
        groups: u32,
        group: u32,
        width: u32,
        len: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= len as usize {
            return;
        }
        let row = i / width as usize;
        let column = i % width as usize;
        if let Some(slot) = out.get_mut(index) {
            *slot = input[(row * groups as usize + group as usize) * width as usize + column];
        }
    }

    /// Write a dense `[rows, width]` matrix back into group `group` of a
    /// `[rows, groups, width]` buffer. Inverse of [`gather_group`]; `len` is
    /// `rows * width`.
    #[kernel]
    pub unsafe fn scatter_group(
        input: &[f32],
        groups: u32,
        group: u32,
        width: u32,
        len: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= len as usize {
            return;
        }
        let row = i / width as usize;
        let column = i % width as usize;
        let target = (row * groups as usize + group as usize) * width as usize + column;
        // SAFETY: distinct `i` map to distinct `target` for a fixed `group`,
        // and the caller launches one group at a time.
        unsafe {
            *out.get_unchecked_mut(target) = input[i];
        }
    }

    /// Muon's fused decay-and-apply over a packed-bf16 master:
    /// `p = decay * p - scale * update`, one thread per element pair.
    ///
    /// Runs over the whole `[rows, groups, width]` parameter at once, against
    /// an `update` buffer each group has already scattered its orthogonalized
    /// result into. Per-group application is not an option here: `width` may be
    /// odd, so a packed word can straddle two groups and two group launches
    /// would race for it.
    #[kernel]
    pub fn muon_apply_bf16(
        update: &[f32],
        decay: f32,
        scale: f32,
        rounding: u32,
        seed: u64,
        mut parameter: DisjointSlice<u32>,
    ) {
        let index = thread::index_1d();
        let pair = index.get();
        let Some(word) = parameter.get_mut(index) else {
            return;
        };
        let stored = *word;

        let mut packed = 0u32;
        let mut half = 0;
        while half < 2 {
            let element = 2 * pair + half;
            let weight = bf16_bits_to_f32((stored >> (16 * half)) as u16);
            let updated = decay * weight - scale * update[element];
            packed |= (round_master(updated, rounding, seed, element) as u32) << (16 * half);
            half += 1;
        }
        *word = packed;
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

    /// splitmix64's mixing round, ported verbatim from
    /// `tensor_core::rng::splitmix64`. cuda-oxide collects device functions per
    /// artifact, so device code cannot call across crates; that function is the
    /// definition this one must match.
    #[inline(always)]
    fn splitmix64(state: u64) -> u64 {
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Round one updated master weight to bf16.
    ///
    /// `rounding` is [`MASTER_ROUNDING_NEAREST`] or
    /// [`MASTER_ROUNDING_STOCHASTIC`]. Stochastic adds a uniform draw to the
    /// mantissa bits bf16 discards before truncating, so the probability of
    /// rounding up equals the discarded fraction and updates below one ulp
    /// survive in expectation. The draw comes from `seed` — the host's
    /// `tensor_core::rng::stream_seed(step, parameter_id)` — mixed with the
    /// element index, never from runtime entropy, so a rerun and a resume
    /// reproduce the same weights bit for bit.
    #[inline(always)]
    fn round_master(value: f32, rounding: u32, seed: u64, element: usize) -> u16 {
        if rounding == MASTER_ROUNDING_NEAREST {
            f32_to_bf16_bits(value)
        } else {
            let draw = (splitmix64(seed.wrapping_add(element as u64)) >> 32) as u32;
            (value.to_bits().wrapping_add(draw & 0xffff) >> 16) as u16
        }
    }

    /// [`fill`] for packed storage, used to zero packed-bf16 gradients.
    #[kernel]
    pub fn fill_u32(value: u32, mut out: DisjointSlice<u32>) {
        let index = thread::index_1d();
        if let Some(slot) = out.get_mut(index) {
            *slot = value;
        }
    }

    /// Round two adjacent f32s into one packed bf16 pair per thread.
    ///
    /// `output` may be longer than `input / 2`; trailing words (padding rows)
    /// are left untouched.
    #[kernel]
    pub fn convert_f32_to_bf16_pairs(input: &[f32], mut output: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let pair = index.get();
        if 2 * pair + 1 >= input.len() {
            return;
        }
        if let Some(slot) = output.get_mut(index) {
            *slot = f32_to_bf16_bits(input[2 * pair]) as u32
                | ((f32_to_bf16_bits(input[2 * pair + 1]) as u32) << 16);
        }
    }

    /// Widen packed bf16 pairs to f32, one output element per thread.
    ///
    /// `input` may be longer than `output / 2`; trailing words (padding rows)
    /// are ignored.
    #[kernel]
    pub fn convert_bf16_pairs_to_f32(input: &[u32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = output.get_mut(index) {
            let word = input[i / 2];
            let bits = (if i % 2 == 0 { word } else { word >> 16 }) as u16;
            *slot = bf16_bits_to_f32(bits);
        }
    }

    /// [`convert_bf16_pairs_to_f32`] with explicit bounds: widen `len` elements
    /// starting at element `offset` into a dense fp32 prefix of `output`.
    ///
    /// Both ends of the fp32 oracle path need this rather than the whole-buffer
    /// convert — experts read one expert out of the middle of a stacked master,
    /// and block linears widen into staging sized for the largest of them, so
    /// neither `input` nor `output` bounds the work on its own.
    #[kernel]
    pub fn widen_bf16_region(input: &[u32], offset: u32, len: u32, mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= len as usize {
            return;
        }
        if let Some(slot) = output.get_mut(index) {
            let element = offset as usize + i;
            let word = input[element / 2];
            let bits = (if element % 2 == 0 { word } else { word >> 16 }) as u16;
            *slot = bf16_bits_to_f32(bits);
        }
    }

    /// Element-level transpose of a packed-bf16 `[rows, cols]` matrix into
    /// `[cols, rows]`, staged through a shared tile so both global sides stay
    /// coalesced. Launch with [`transpose_pairs_config`]; both dimensions must
    /// be multiples of `TRANSPOSE_TILE`.
    #[kernel]
    pub unsafe fn transpose_bf16_pairs(
        input: &[u32],
        rows: u32,
        cols: u32,
        mut output: DisjointSlice<u32>,
    ) {
        // One u16 value per slot, zero-extended; +1 padding column so the
        // column-major reads of the store phase spread across banks.
        static mut VALUES: SharedArray<u32, { TRANSPOSE_TILE * (TRANSPOSE_TILE + 1) }> =
            SharedArray::UNINIT;
        const TILE_WORDS_WIDE: usize = TRANSPOSE_TILE / 2;

        let tid = thread::threadIdx_x() as usize;
        let tile_row = thread::blockIdx_y() as usize * TRANSPOSE_TILE;
        let tile_col = thread::blockIdx_x() as usize * TRANSPOSE_TILE;
        let source_words_per_row = cols as usize / 2;
        let output_words_per_row = rows as usize / 2;

        let mut local = tid;
        while local < TRANSPOSE_WORDS {
            let row = local / TILE_WORDS_WIDE;
            let word_column = local % TILE_WORDS_WIDE;
            let word = input[(tile_row + row) * source_words_per_row + tile_col / 2 + word_column];
            unsafe {
                VALUES[row * (TRANSPOSE_TILE + 1) + 2 * word_column] = word & 0xffff;
                VALUES[row * (TRANSPOSE_TILE + 1) + 2 * word_column + 1] = word >> 16;
            }
            local += TRANSPOSE_THREADS;
        }
        thread::sync_threads();

        let mut local = tid;
        while local < TRANSPOSE_WORDS {
            // Output word [c, p] packs source elements [2p, c] and [2p+1, c].
            let output_row = local / TILE_WORDS_WIDE;
            let word_column = local % TILE_WORDS_WIDE;
            let (low, high) = unsafe {
                (
                    VALUES[(2 * word_column) * (TRANSPOSE_TILE + 1) + output_row],
                    VALUES[(2 * word_column + 1) * (TRANSPOSE_TILE + 1) + output_row],
                )
            };
            let global =
                (tile_col + output_row) * output_words_per_row + tile_row / 2 + word_column;
            // SAFETY: each (tile, local) pair maps to a unique output word.
            unsafe {
                *output.get_unchecked_mut(global) = low | (high << 16);
            }
            local += TRANSPOSE_THREADS;
        }
    }

    /// Fused quantize-and-transpose: read an fp32 `[rows, cols]` matrix, round
    /// each element to bf16, and store the packed-bf16 transpose `[cols, rows]`.
    ///
    /// Folds `convert_f32_to_bf16_pairs` + `transpose_bf16_pairs` into a single
    /// global pass for weight-gradient operand staging, halving that operand's
    /// staging traffic (one fp32 read + one bf16 write, versus an extra
    /// bf16 round-trip through the intermediate). Launch with
    /// [`transpose_pairs_config`]; both dimensions must be multiples of
    /// `TRANSPOSE_TILE`.
    #[kernel]
    pub unsafe fn convert_f32_transpose_bf16_pairs(
        input: &[f32],
        rows: u32,
        cols: u32,
        mut output: DisjointSlice<u32>,
    ) {
        // Mirror `transpose_bf16_pairs`' shared staging: one bf16 value per slot
        // with a +1 padding column so the store phase's column-major reads
        // spread across banks. The load phase differs only in rounding fp32
        // source elements to bf16 on the way in.
        static mut VALUES: SharedArray<u32, { TRANSPOSE_TILE * (TRANSPOSE_TILE + 1) }> =
            SharedArray::UNINIT;
        const TILE_WORDS_WIDE: usize = TRANSPOSE_TILE / 2;

        let tid = thread::threadIdx_x() as usize;
        let tile_row = thread::blockIdx_y() as usize * TRANSPOSE_TILE;
        let tile_col = thread::blockIdx_x() as usize * TRANSPOSE_TILE;
        let source_cols = cols as usize;
        let output_words_per_row = rows as usize / 2;

        let mut local = tid;
        while local < TRANSPOSE_WORDS {
            let row = local / TILE_WORDS_WIDE;
            let word_column = local % TILE_WORDS_WIDE;
            let source = (tile_row + row) * source_cols + tile_col + 2 * word_column;
            unsafe {
                VALUES[row * (TRANSPOSE_TILE + 1) + 2 * word_column] =
                    f32_to_bf16_bits(input[source]) as u32;
                VALUES[row * (TRANSPOSE_TILE + 1) + 2 * word_column + 1] =
                    f32_to_bf16_bits(input[source + 1]) as u32;
            }
            local += TRANSPOSE_THREADS;
        }
        thread::sync_threads();

        let mut local = tid;
        while local < TRANSPOSE_WORDS {
            // Output word [c, p] packs source elements [2p, c] and [2p+1, c].
            let output_row = local / TILE_WORDS_WIDE;
            let word_column = local % TILE_WORDS_WIDE;
            let (low, high) = unsafe {
                (
                    VALUES[(2 * word_column) * (TRANSPOSE_TILE + 1) + output_row],
                    VALUES[(2 * word_column + 1) * (TRANSPOSE_TILE + 1) + output_row],
                )
            };
            let global =
                (tile_col + output_row) * output_words_per_row + tile_row / 2 + word_column;
            // SAFETY: each (tile, local) pair maps to a unique output word.
            unsafe {
                *output.get_unchecked_mut(global) = low | (high << 16);
            }
            local += TRANSPOSE_THREADS;
        }
    }

    /// Fused quantize that emits both the packed-bf16 `[rows, cols]` matrix and
    /// its packed-bf16 transpose `[cols, rows]` from one fp32 read.
    ///
    /// The weight-gradient path needs the quantized output gradient in both
    /// layouts: row-major for the input GEMM and transposed for the weight GEMM.
    /// Reading fp32 once and writing both packed views folds the standalone
    /// transpose pass into the already-required convert. Launch with
    /// [`transpose_pairs_config`]; both dimensions must be multiples of
    /// `TRANSPOSE_TILE`.
    #[kernel]
    pub unsafe fn convert_f32_to_bf16_pairs_and_transpose(
        input: &[f32],
        rows: u32,
        cols: u32,
        mut normal: DisjointSlice<u32>,
        mut transposed: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<u32, { TRANSPOSE_TILE * (TRANSPOSE_TILE + 1) }> =
            SharedArray::UNINIT;
        const TILE_WORDS_WIDE: usize = TRANSPOSE_TILE / 2;

        let tid = thread::threadIdx_x() as usize;
        let tile_row = thread::blockIdx_y() as usize * TRANSPOSE_TILE;
        let tile_col = thread::blockIdx_x() as usize * TRANSPOSE_TILE;
        let source_cols = cols as usize;
        let normal_words_per_row = cols as usize / 2;
        let transposed_words_per_row = rows as usize / 2;

        let mut local = tid;
        while local < TRANSPOSE_WORDS {
            let row = local / TILE_WORDS_WIDE;
            let word_column = local % TILE_WORDS_WIDE;
            let source = (tile_row + row) * source_cols + tile_col + 2 * word_column;
            let low = f32_to_bf16_bits(input[source]);
            let high = f32_to_bf16_bits(input[source + 1]);
            // Row-major output word [row, wc] packs the two adjacent elements.
            let normal_global =
                (tile_row + row) * normal_words_per_row + tile_col / 2 + word_column;
            // SAFETY: each (tile, local) pair maps to a unique output word.
            unsafe {
                *normal.get_unchecked_mut(normal_global) = low as u32 | ((high as u32) << 16);
                VALUES[row * (TRANSPOSE_TILE + 1) + 2 * word_column] = low as u32;
                VALUES[row * (TRANSPOSE_TILE + 1) + 2 * word_column + 1] = high as u32;
            }
            local += TRANSPOSE_THREADS;
        }
        thread::sync_threads();

        let mut local = tid;
        while local < TRANSPOSE_WORDS {
            // Transposed word [c, p] packs source elements [2p, c] and [2p+1, c].
            let output_row = local / TILE_WORDS_WIDE;
            let word_column = local % TILE_WORDS_WIDE;
            let (low, high) = unsafe {
                (
                    VALUES[(2 * word_column) * (TRANSPOSE_TILE + 1) + output_row],
                    VALUES[(2 * word_column + 1) * (TRANSPOSE_TILE + 1) + output_row],
                )
            };
            let global =
                (tile_col + output_row) * transposed_words_per_row + tile_row / 2 + word_column;
            // SAFETY: each (tile, local) pair maps to a unique output word.
            unsafe {
                *transposed.get_unchecked_mut(global) = low | (high << 16);
            }
            local += TRANSPOSE_THREADS;
        }
    }

    /// Fused decoupled AdamW over a packed-bf16 master with an fp32 gradient:
    /// one thread owns one pair.
    ///
    /// Moments stay fp32 and the whole update is computed in fp32 — bf16 only
    /// ever appears at the write-back, where [`round_master`] decides how the
    /// discarded mantissa bits are handled. Beside that rounding the arithmetic
    /// is [`adamw`] exactly.
    #[kernel]
    pub fn adamw_bf16_master(
        gradient: &[f32],
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        first_correction: f32,
        second_correction: f32,
        rounding: u32,
        seed: u64,
        mut master: DisjointSlice<u32>,
        mut first: DisjointSlice<f32>,
        mut second: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let pair = index.get();
        let Some(word) = master.get_mut(index) else {
            return;
        };
        let stored = *word;

        let mut packed = 0u32;
        let mut half = 0;
        while half < 2 {
            let element = 2 * pair + half;
            let g = gradient[element];
            let mut weight = bf16_bits_to_f32((stored >> (16 * half)) as u16);
            // SAFETY: this thread exclusively owns elements 2*pair and
            // 2*pair+1 of every per-element buffer.
            unsafe {
                let first = first.get_unchecked_mut(element);
                let second = second.get_unchecked_mut(element);
                *first = beta1 * *first + (1.0 - beta1) * g;
                *second = beta2 * *second + (1.0 - beta2) * g * g;
                let first_hat = *first * first_correction;
                let second_hat = *second * second_correction;
                let update = first_hat / (second_hat.sqrt() + epsilon) + weight_decay * weight;
                weight -= learning_rate * update;
            }
            packed |= (round_master(weight, rounding, seed, element) as u32) << (16 * half);
            half += 1;
        }
        *word = packed;
    }

    /// [`adamw_bf16_master`] for the lm-head, whose weight gradient is produced
    /// in packed bf16 straight out of the tcgen05 accumulate epilogue.
    #[kernel]
    pub fn adamw_bf16_master_packed_grad(
        gradient: &[u32],
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        first_correction: f32,
        second_correction: f32,
        rounding: u32,
        seed: u64,
        mut master: DisjointSlice<u32>,
        mut first: DisjointSlice<f32>,
        mut second: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let pair = index.get();
        let Some(word) = master.get_mut(index) else {
            return;
        };
        let stored = *word;
        let gradient = gradient[pair];

        let mut packed = 0u32;
        let mut half = 0;
        while half < 2 {
            let element = 2 * pair + half;
            let g = bf16_bits_to_f32((gradient >> (16 * half)) as u16);
            let mut weight = bf16_bits_to_f32((stored >> (16 * half)) as u16);
            // SAFETY: this thread exclusively owns elements 2*pair and
            // 2*pair+1 of every per-element buffer.
            unsafe {
                let first = first.get_unchecked_mut(element);
                let second = second.get_unchecked_mut(element);
                *first = beta1 * *first + (1.0 - beta1) * g;
                *second = beta2 * *second + (1.0 - beta2) * g * g;
                let first_hat = *first * first_correction;
                let second_hat = *second * second_correction;
                let update = first_hat / (second_hat.sqrt() + epsilon) + weight_decay * weight;
                weight -= learning_rate * update;
            }
            packed |= (round_master(weight, rounding, seed, element) as u32) << (16 * half);
            half += 1;
        }
        *word = packed;
    }

    /// One-block reduction. Threads accumulate grid-stride partial sums before
    /// a standard shared-memory tree reduction.
    #[kernel]
    pub fn sum(a: &[f32], len: u32, mut out: DisjointSlice<f32>) {
        static mut PARTIALS: SharedArray<f32, REDUCE_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let mut i = tid;
        let mut partial = 0.0f32;
        while i < len as usize {
            partial += a[i];
            i += REDUCE_THREADS;
        }
        unsafe {
            PARTIALS[tid] = partial;
        }
        thread::sync_threads();

        let mut stride = REDUCE_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIALS[tid] += PARTIALS[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        let index = thread::index_1d();
        if tid == 0
            && let Some(slot) = out.get_mut(index)
        {
            unsafe {
                *slot = PARTIALS[0];
            }
        }
    }

    #[kernel]
    pub fn dot(a: &[f32], b: &[f32], len: u32, mut out: DisjointSlice<f32>) {
        static mut PARTIALS: SharedArray<f32, REDUCE_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let mut i = tid;
        let mut partial = 0.0f32;
        while i < len as usize {
            partial += a[i] * b[i];
            i += REDUCE_THREADS;
        }
        unsafe {
            PARTIALS[tid] = partial;
        }
        thread::sync_threads();

        let mut stride = REDUCE_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIALS[tid] += PARTIALS[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        let index = thread::index_1d();
        if tid == 0
            && let Some(slot) = out.get_mut(index)
        {
            unsafe {
                *slot = PARTIALS[0];
            }
        }
    }

    /// Auditable baseline: one output element per thread, reading both
    /// operands directly from global memory.
    #[kernel]
    pub fn gemm_naive(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::blockIdx_y() as usize * thread::blockDim_y() as usize
            + thread::threadIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        if row >= m as usize || col >= n as usize {
            return;
        }

        let mut acc = 0.0f32;
        for inner in 0..k as usize {
            acc += a[row * k as usize + inner] * b[inner * n as usize + col];
        }
        if let Some(index) = unsafe { thread::index_2d_runtime(n as usize) }
            && let Some(slot) = c.get_mut(index)
        {
            *slot = acc;
        }
    }

    /// Shared-memory tiled GEMM. Bounds checks make it valid for dimensions
    /// that are not multiples of `TILE`.
    #[kernel]
    pub fn gemm_tiled(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f32, TILE_ELEMENTS> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, TILE_ELEMENTS> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * TILE + ty;
        let col = thread::blockIdx_x() as usize * TILE + tx;
        let shared_index = ty * TILE + tx;
        let mut acc = 0.0f32;
        let tiles = (k as usize).div_ceil(TILE);

        for tile in 0..tiles {
            let a_col = tile * TILE + tx;
            let b_row = tile * TILE + ty;
            unsafe {
                TILE_A[shared_index] = if row < m as usize && a_col < k as usize {
                    a[row * k as usize + a_col]
                } else {
                    0.0
                };
                TILE_B[shared_index] = if b_row < k as usize && col < n as usize {
                    b[b_row * n as usize + col]
                } else {
                    0.0
                };
            }
            thread::sync_threads();

            for inner in 0..TILE {
                unsafe {
                    acc += TILE_A[ty * TILE + inner] * TILE_B[inner * TILE + tx];
                }
            }
            thread::sync_threads();
        }

        if row < m as usize
            && col < n as usize
            && let Some(index) = unsafe { thread::index_2d_runtime(n as usize) }
            && let Some(slot) = c.get_mut(index)
        {
            *slot = acc;
        }
    }

    /// `C = A^T . B`: `[M,K]^T x [M,N] -> [K,N]`.
    #[kernel]
    pub fn gemm_tn(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::blockIdx_y() as usize * thread::blockDim_y() as usize
            + thread::threadIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        if row >= k as usize || col >= n as usize {
            return;
        }
        let mut acc = 0.0f32;
        for inner in 0..m as usize {
            acc += a[inner * k as usize + row] * b[inner * n as usize + col];
        }
        if let Some(index) = unsafe { thread::index_2d_runtime(n as usize) }
            && let Some(slot) = c.get_mut(index)
        {
            *slot = acc;
        }
    }

    /// `C += A^T . B`: the accumulating counterpart to [`gemm_tn`].
    #[kernel]
    pub fn gemm_tn_accumulate(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::blockIdx_y() as usize * thread::blockDim_y() as usize
            + thread::threadIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        if row >= k as usize || col >= n as usize {
            return;
        }
        let mut acc = 0.0f32;
        for inner in 0..m as usize {
            acc += a[inner * k as usize + row] * b[inner * n as usize + col];
        }
        if let Some(index) = unsafe { thread::index_2d_runtime(n as usize) }
            && let Some(slot) = c.get_mut(index)
        {
            *slot += acc;
        }
    }

    /// `C = A . B^T`: `[M,K] x [N,K]^T -> [M,N]`.
    #[kernel]
    pub fn gemm_nt(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::blockIdx_y() as usize * thread::blockDim_y() as usize
            + thread::threadIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        if row >= m as usize || col >= n as usize {
            return;
        }
        let mut acc = 0.0f32;
        for inner in 0..k as usize {
            acc += a[row * k as usize + inner] * b[col * k as usize + inner];
        }
        if let Some(index) = unsafe { thread::index_2d_runtime(n as usize) }
            && let Some(slot) = c.get_mut(index)
        {
            *slot = acc;
        }
    }
}

/// Owning, contiguous device tensor. Shape information is zero-sized and
/// exists only in `S`; the allocation contains exactly `S::NUM_ELEMENTS`.
pub struct GpuTensor<E: Element, S: Shape> {
    data: DeviceBuffer<E>,
    _shape: PhantomData<S>,
}

/// Pack fp32 values into the device's packed-bf16 pair layout (one `u32` per
/// two adjacent row elements), rounding to nearest even.
pub fn pack_bf16_pairs(values: &[f32]) -> Vec<u32> {
    assert!(values.len().is_multiple_of(2), "packed bf16 needs pairs");
    values
        .chunks_exact(2)
        .map(|pair| {
            bf16::from_f32(pair[0]).to_bits() as u32
                | ((bf16::from_f32(pair[1]).to_bits() as u32) << 16)
        })
        .collect()
}

/// Inverse of [`pack_bf16_pairs`].
pub fn unpack_bf16_pairs(words: &[u32]) -> Vec<f32> {
    let mut values = Vec::with_capacity(2 * words.len());
    for &word in words {
        values.push(bf16::from_bits(word as u16).to_f32());
        values.push(bf16::from_bits((word >> 16) as u16).to_f32());
    }
    values
}

/// Owning, contiguous bf16 master weights in the device's packed-pair layout.
///
/// Masters moved from fp32 to bf16 in #57 (SPEC §7, decision #8 successor):
/// the update is still computed in fp32 against fp32 moments, and bf16 appears
/// only at the write-back. Storage is packed `u32` words rather than a bf16
/// element buffer because that is the layout every compute copy, TMA
/// descriptor, and tcgen05 operand in the tree already speaks.
///
/// Deliberately carries no arithmetic: the fused optimizer kernels widen,
/// compute, and round in one pass, and the non-tcgen05 oracle paths widen into
/// scratch.
pub struct GpuBf16Tensor<S: Shape> {
    words: DeviceBuffer<u32>,
    _shape: PhantomData<S>,
}

impl<S: Shape> Tensor for GpuBf16Tensor<S> {
    type Elem = tensor_core::bf16;
    type Shape = S;
}

impl<S: Shape> GpuBf16Tensor<S> {
    pub const LEN: usize = S::NUM_ELEMENTS;
    /// Packed words backing the tensor. Every master dimension is even, so the
    /// element count always is too.
    pub const WORDS: usize = {
        assert!(S::NUM_ELEMENTS.is_multiple_of(2));
        S::NUM_ELEMENTS / 2
    };

    pub fn zeros(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            words: DeviceBuffer::zeroed(stream, Self::WORDS)?,
            _shape: PhantomData,
        })
    }

    /// Round fp32 host values to bf16 and upload them.
    pub fn from_f32_host(stream: &CudaStream, values: &[f32]) -> Result<Self, DriverError> {
        assert_eq!(values.len(), Self::LEN, "slice length != shape volume");
        Ok(Self {
            words: DeviceBuffer::from_host(stream, &pack_bf16_pairs(values))?,
            _shape: PhantomData,
        })
    }

    /// Overwrite the master in place from rounded fp32 host values.
    ///
    /// Checkpoint resume must refill a master rather than replace it: TMA
    /// descriptors are encoded against the words' device address (#58), so a
    /// fresh allocation would leave every compute map pointing at freed memory.
    pub fn load_f32_host(
        &mut self,
        stream: &CudaStream,
        values: &[f32],
    ) -> Result<(), DriverError> {
        assert_eq!(values.len(), Self::LEN, "slice length != shape volume");
        let packed = pack_bf16_pairs(values);
        // SAFETY: `packed` is exactly the buffer's own word count, and the
        // synchronize keeps the host slice alive until the copy retires.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                self.words.cu_deviceptr(),
                packed.as_ptr(),
                std::mem::size_of_val(packed.as_slice()),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()
    }

    /// Download and widen. Every value is exactly bf16-representable.
    pub fn to_f32_host(&self, stream: &CudaStream) -> Result<Vec<f32>, DriverError> {
        Ok(unpack_bf16_pairs(&self.words.to_host_vec(stream)?))
    }

    pub fn as_words(&self) -> &DeviceBuffer<u32> {
        &self.words
    }

    pub fn as_words_mut(&mut self) -> &mut DeviceBuffer<u32> {
        &mut self.words
    }

    /// Widen the whole master into an fp32 buffer, for the register-tiled
    /// oracle GEMMs that read fp32 operands.
    ///
    /// `out` may be longer than the master — it is shared staging sized for the
    /// largest parameter that uses it — so this goes through the explicitly
    /// length-bounded region kernel rather than the whole-buffer one.
    pub fn widen_into(
        &self,
        out: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        assert!(out.len() >= Self::LEN);
        // SAFETY: the assert above bounds the written region, and the launch
        // covers exactly the master's own LEN values.
        unsafe {
            module.widen_bf16_region(
                stream,
                pairs_config(Self::LEN),
                &self.words,
                0,
                Self::LEN as u32,
                out,
            )
        }
    }

    /// One fused AdamW step: fp32 gradient and moments in, one rounded bf16
    /// write-back out. `seed` is `tensor_core::rng::stream_seed(step, id)`.
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step(
        &mut self,
        gradient: &GpuTensor<f32, S>,
        moments: &mut GpuAdamWMoments<S>,
        config: MasterAdamW,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        // SAFETY: the packed master, its fp32 gradient, and both moments all
        // describe S, and the launch covers one thread per packed pair.
        unsafe {
            module.adamw_bf16_master(
                stream,
                pairs_config(Self::WORDS),
                gradient.as_device_buffer(),
                config.learning_rate,
                config.beta1,
                config.beta2,
                config.epsilon,
                config.weight_decay,
                config.first_correction,
                config.second_correction,
                config.rounding,
                config.seed,
                &mut self.words,
                moments.first.as_device_buffer_mut(),
                moments.second.as_device_buffer_mut(),
            )
        }
    }
}

/// Everything the fused bf16-master AdamW kernels need beyond the buffers.
///
/// Bundled because the write-back adds a rounding mode and a noise seed to an
/// already long argument list, and every call site sets them the same way.
#[derive(Clone, Copy, Debug)]
pub struct MasterAdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub first_correction: f32,
    pub second_correction: f32,
    pub rounding: u32,
    pub seed: u64,
}

/// GPU-resident first and second AdamW moments for one parameter tensor.
///
/// Both stay fp32 even beside a bf16 master: the second moment sits inside a
/// square root in the denominator, where bf16 is not safe.
pub struct GpuAdamWMoments<S: Shape> {
    pub first: GpuTensor<f32, S>,
    pub second: GpuTensor<f32, S>,
}

impl<S: Shape> GpuAdamWMoments<S> {
    pub fn zeros(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            first: GpuTensor::zeros(stream)?,
            second: GpuTensor::zeros(stream)?,
        })
    }
}

/// GPU-resident Muon momentum buffer for one parameter tensor.
pub struct GpuMuonMomentum<S: Shape> {
    pub momentum: GpuTensor<f32, S>,
}

impl<S: Shape> GpuMuonMomentum<S> {
    pub fn zeros(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            momentum: GpuTensor::zeros(stream)?,
        })
    }
}

impl<E: Element, S: Shape> Tensor for GpuTensor<E, S> {
    type Elem = E;
    type Shape = S;
}

impl<E: Element + DeviceCopy, S: Shape> GpuTensor<E, S> {
    pub const LEN: usize = S::NUM_ELEMENTS;

    pub fn zeros(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            data: DeviceBuffer::zeroed(stream, S::NUM_ELEMENTS)?,
            _shape: PhantomData,
        })
    }

    pub fn from_host(stream: &CudaStream, src: &[E]) -> Result<Self, DriverError> {
        assert_eq!(src.len(), S::NUM_ELEMENTS, "slice length != shape volume");
        Ok(Self {
            data: DeviceBuffer::from_host(stream, src)?,
            _shape: PhantomData,
        })
    }

    pub fn from_cpu(stream: &CudaStream, src: &CpuTensor<E, S>) -> Result<Self, DriverError> {
        Self::from_host(stream, src.as_slice())
    }

    pub fn to_host(&self, stream: &CudaStream) -> Result<Vec<E>, DriverError> {
        self.data.to_host_vec(stream)
    }

    pub fn to_cpu(&self, stream: &CudaStream) -> Result<CpuTensor<E, S>, DriverError> {
        Ok(CpuTensor::from_slice(&self.to_host(stream)?))
    }

    pub fn as_device_buffer(&self) -> &DeviceBuffer<E> {
        &self.data
    }

    pub fn as_device_buffer_mut(&mut self) -> &mut DeviceBuffer<E> {
        &mut self.data
    }
}

fn elementwise_config<S: Shape>() -> LaunchConfig {
    assert!(S::NUM_ELEMENTS <= u32::MAX as usize);
    LaunchConfig::for_num_elems(S::NUM_ELEMENTS as u32)
}

fn reduction_config() -> LaunchConfig {
    assert!(REDUCE_THREADS.is_power_of_two());
    assert!(REDUCE_THREADS <= 1024);
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (REDUCE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One thread per packed-bf16 word (or per element of any flat buffer).
pub fn pairs_config(elements: usize) -> LaunchConfig {
    assert!(elements <= u32::MAX as usize);
    LaunchConfig::for_num_elems(elements as u32)
}

/// Validate dimensions and build the packed-bf16 transpose launch.
pub fn transpose_pairs_config(rows: usize, cols: usize) -> LaunchConfig {
    assert!(rows.is_multiple_of(TRANSPOSE_TILE) && cols.is_multiple_of(TRANSPOSE_TILE));
    assert!(rows <= u32::MAX as usize && cols <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            (cols / TRANSPOSE_TILE) as u32,
            (rows / TRANSPOSE_TILE) as u32,
            1,
        ),
        block_dim: (TRANSPOSE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn gemm_config<const M: usize, const N: usize>() -> LaunchConfig {
    assert!(TILE * TILE <= 1024);
    assert!(M <= u32::MAX as usize && N <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            (N as u32).div_ceil(TILE as u32),
            (M as u32).div_ceil(TILE as u32),
            1,
        ),
        block_dim: (TILE as u32, TILE as u32, 1),
        shared_mem_bytes: 0,
    }
}

impl<S: Shape> GpuTensor<f32, S> {
    pub fn add(
        &self,
        rhs: &Self,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<Self, DriverError> {
        let mut out = Self::zeros(stream)?;
        // SAFETY: typed tensors guarantee equally sized input/output buffers.
        unsafe {
            module.add(
                stream,
                elementwise_config::<S>(),
                &self.data,
                &rhs.data,
                &mut out.data,
            )
        }?;
        Ok(out)
    }

    pub fn mul(
        &self,
        rhs: &Self,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<Self, DriverError> {
        let mut out = Self::zeros(stream)?;
        // SAFETY: typed tensors guarantee equally sized input/output buffers.
        unsafe {
            module.mul(
                stream,
                elementwise_config::<S>(),
                &self.data,
                &rhs.data,
                &mut out.data,
            )
        }?;
        Ok(out)
    }

    pub fn scale(
        &self,
        factor: f32,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<Self, DriverError> {
        let mut out = Self::zeros(stream)?;
        // SAFETY: typed tensors guarantee equally sized input/output buffers.
        unsafe {
            module.scale(
                stream,
                elementwise_config::<S>(),
                &self.data,
                factor,
                &mut out.data,
            )
        }?;
        Ok(out)
    }

    pub fn add_scaled_assign(
        &mut self,
        factor: f32,
        rhs: &Self,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        // SAFETY: typed tensors guarantee equally sized input/output buffers.
        unsafe {
            module.add_scaled(
                stream,
                elementwise_config::<S>(),
                &rhs.data,
                factor,
                &mut self.data,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step(
        &mut self,
        gradient: &Self,
        moments: &mut GpuAdamWMoments<S>,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        first_correction: f32,
        second_correction: f32,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        // SAFETY: the parameter, gradient, and both moment tensors share S.
        unsafe {
            module.adamw(
                stream,
                elementwise_config::<S>(),
                gradient.as_device_buffer(),
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                first_correction,
                second_correction,
                self.as_device_buffer_mut(),
                moments.first.as_device_buffer_mut(),
                moments.second.as_device_buffer_mut(),
            )
        }
    }

    pub fn sum(
        &self,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank1<1>>, DriverError> {
        assert!(S::NUM_ELEMENTS <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        // SAFETY: n is the input length and out has one accumulator element.
        unsafe {
            module.sum(
                stream,
                reduction_config(),
                &self.data,
                S::NUM_ELEMENTS as u32,
                &mut out.data,
            )
        }?;
        Ok(out)
    }

    pub fn dot(
        &self,
        rhs: &Self,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank1<1>>, DriverError> {
        assert!(S::NUM_ELEMENTS <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        // SAFETY: both typed inputs have n elements; out has one accumulator.
        unsafe {
            module.dot(
                stream,
                reduction_config(),
                &self.data,
                &rhs.data,
                S::NUM_ELEMENTS as u32,
                &mut out.data,
            )
        }?;
        Ok(out)
    }
}

impl<const M: usize, const K: usize> GpuTensor<f32, Rank2<M, K>> {
    pub fn matmul_naive<const N: usize>(
        &self,
        rhs: &GpuTensor<f32, Rank2<K, N>>,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<M, N>>, DriverError> {
        assert!(K <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        // SAFETY: Rank2 types guarantee the MxK, KxN, and MxN buffer sizes.
        unsafe {
            module.gemm_naive(
                stream,
                gemm_config::<M, N>(),
                M as u32,
                N as u32,
                K as u32,
                &self.data,
                &rhs.data,
                &mut out.data,
            )
        }?;
        Ok(out)
    }

    /// Default fp32 GEMM: shared-memory tiled `[M,K] x [K,N] -> [M,N]`.
    pub fn matmul<const N: usize>(
        &self,
        rhs: &GpuTensor<f32, Rank2<K, N>>,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<M, N>>, DriverError> {
        assert!(K <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        // SAFETY: Rank2 types guarantee the MxK, KxN, and MxN buffer sizes.
        unsafe {
            module.gemm_tiled(
                stream,
                gemm_config::<M, N>(),
                M as u32,
                N as u32,
                K as u32,
                &self.data,
                &rhs.data,
                &mut out.data,
            )
        }?;
        Ok(out)
    }

    pub fn matmul_tn<const N: usize>(
        &self,
        rhs: &GpuTensor<f32, Rank2<M, N>>,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<K, N>>, DriverError> {
        assert!(M <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        // SAFETY: Rank2 types guarantee the MxK, MxN, and KxN buffer sizes.
        unsafe {
            module.gemm_tn(
                stream,
                gemm_config::<K, N>(),
                M as u32,
                N as u32,
                K as u32,
                &self.data,
                &rhs.data,
                &mut out.data,
            )
        }?;
        Ok(out)
    }

    pub fn matmul_nt<const N: usize>(
        &self,
        rhs: &GpuTensor<f32, Rank2<N, K>>,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<M, N>>, DriverError> {
        assert!(K <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        // SAFETY: Rank2 types guarantee the MxK, NxK, and MxN buffer sizes.
        unsafe {
            module.gemm_nt(
                stream,
                gemm_config::<M, N>(),
                M as u32,
                N as u32,
                K as u32,
                &self.data,
                &rhs.data,
                &mut out.data,
            )
        }?;
        Ok(out)
    }
}
