//! The correctness baseline the training kernels are checked against.
//!
//! `lib.rs` keeps the kernels a training step launches; everything the gates
//! compare those against lives here. The split is which file a kernel is in,
//! not what it does: these are the same definitions, moved.
//!
//! They are a second `#[cuda_module]` because a `#[cuda_module]` collects only
//! the kernels whose tokens it can see, and an attribute macro receives a
//! `mod reference;` declaration rather than this file's contents. So the gates
//! load two modules and reach these launchers through this one, the way
//! `gemm::fp32` and `flash_attn::tcgen05` are already reached.
//!
//! The launch-shape constants and the tile aliases stay `lib.rs`'s and are
//! read through `use super::*`. The four `#[inline(always)]` scalars below are
//! spelled twice instead, because a `#[cuda_module]` codegens the device
//! functions its own kernels call — the same duplication `flash_attn::tcgen05`
//! and `tensor_gpu` already carry for `pack_bf16_pair`.

use super::*;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// [`super::super::kernels::quad_lanes`], for this module's artifact.
    #[inline(always)]
    fn quad_lanes(bits: u128) -> [f32; QUAD_LANES] {
        [
            f32::from_bits(bits as u32),
            f32::from_bits((bits >> 32) as u32),
            f32::from_bits((bits >> 64) as u32),
            f32::from_bits((bits >> 96) as u32),
        ]
    }

    /// [`super::super::kernels::quad_bits`], for this module's artifact.
    #[inline(always)]
    fn quad_bits(lanes: [f32; QUAD_LANES]) -> u128 {
        (lanes[0].to_bits() as u128)
            | ((lanes[1].to_bits() as u128) << 32)
            | ((lanes[2].to_bits() as u128) << 64)
            | ((lanes[3].to_bits() as u128) << 96)
    }

    /// [`super::super::kernels::f32_to_bf16_bits`], for this module's artifact.
    #[inline(always)]
    fn f32_to_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let round = 0x7fffu32 + ((bits >> 16) & 1);
        (bits.wrapping_add(round) >> 16) as u16
    }

    /// [`super::super::kernels::rope_angle`], for this module's artifact.
    #[inline(always)]
    fn rope_angle(pair: usize, sequence_length: u32, heads: u32, head_dim: u32) -> usize {
        let half = head_dim as usize / 2;
        let position = (pair / (heads as usize * half)) % sequence_length as usize;
        2 * (position * half + pair % half)
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
    /// `dweight[D,E] = x[N,D]ᵀ·dlogits[N,E]`, written to
    /// `partials[SPLITS, E, D]` for `router_backward_weight_merge` to sum.
    ///
    /// `D*E` is far too few outputs to fill the machine on its own, so the
    /// token dimension is split across blocks: a block owns `ROUTER_WGRAD_BM`
    /// model rows of one partition, lane-major so each `x` read is a full
    /// coalesced sector, and each lane keeps the `[E]` accumulators of the two
    /// rows it owns in registers — one load feeds `2 * E` FMAs against a gate
    /// row the whole warp shares.
    ///
    /// Two things keep the token walk at memory rate, and both are about loads
    /// sharing a basic block: the shape is validated once up front so the walk
    /// indexes unchecked, and `ROUTER_WGRAD_TOKENS` of them are issued before
    /// any is multiplied. Guarding each load instead — a bounds check, an
    /// `expert < e` test — puts every one in a block of its own, which is one
    /// load in flight per warp and was a tenth of the achievable bandwidth.
    ///
    /// The reduction order is fixed: a lane owns its outputs alone and sums its
    /// partition in ascending token order, and the merge sums partitions in
    /// ascending order. No lane, block, or launch ordering can perturb it,
    /// which is what the checkpoint-resume gate holds the trajectory to.
    ///
    /// # Safety
    ///
    /// `x` must hold `tokens * dim` elements, `dlogits` `tokens * experts` and
    /// `partials` `SPLITS * experts * dim`; `dim` must be even. Each is
    /// checked, and a launch that fails any of them writes nothing.
    #[kernel]
    pub unsafe fn router_backward_weight_split(
        x: &[f32],
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
            || n * d > x.len()
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
                // The tail rereads the partition's last token and multiplies
                // it by zero. Clamping costs one select; branching would cost
                // the load its place in this block.
                let index = (token + step).min(token_end - 1);
                let keep = if token + step < token_end { 1.0 } else { 0.0 };
                let base = index * d + row;
                unsafe {
                    lows[step] = *x.get_unchecked(base) * keep;
                    highs[step] = *x.get_unchecked(base + 1) * keep;
                }
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

    /// [`swiglu_forward`] storing packed-bf16 pairs, one word per thread.
    ///
    /// The expert `activated` panel is only ever read as a bf16 tcgen05
    /// operand, so it is rounded once here instead of stored wide and
    /// quantized again by each of the two GEMMs that read it (#59).
    #[kernel]
    pub fn swiglu_forward_bf16(gate: &[f32], up: &[f32], mut y: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let pair = index.get();
        let elements = gate.len().min(up.len());
        if 2 * pair >= elements {
            return;
        }
        if let Some(slot) = y.get_mut(index) {
            let mut packed = 0u32;
            // An odd element count leaves the last word half real; its high
            // half stays zero and no reader looks past `elements`. Bailing on
            // the whole word instead, as this once did, dropped that element.
            for half in 0..2 {
                let i = 2 * pair + half;
                if i >= elements {
                    break;
                }
                let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
                packed |= (f32_to_bf16_bits(gate[i] * sigmoid * up[i]) as u32) << (16 * half);
            }
            *slot = packed;
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

    #[kernel]
    pub fn swiglu_backward_up(gate: &[f32], dy: &[f32], mut dup: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(up_slot) = dup.get_mut(index) {
            let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
            *up_slot = dy[i] * gate[i] * sigmoid;
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
}
