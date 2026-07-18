//! CPU simulation of the GPU bf16 lm-head pipeline on the tiny overfit batch.
//!
//! Quantization points mirror gpu/model exactly: head input, compute
//! weights (rounded from an fp32 master), stored logits, dlogits, dw, and dx
//! are all bf16-rounded; every accumulation is fp32; the master update sees
//! the bf16-rounded gradients.

use nn::{
    CausalAttention, Dense, Module, Rope, SoftmaxCrossEntropy, SoftmaxCrossEntropyInput, SwiGlu,
};
use optim::{AdamWConfig, AdamWMoments, DenseAdamW, adamw_step};
use tensor_core::{Rank2, Shape};
use tensor_cpu::CpuTensor;

const N: usize = 4;
const T: usize = 4;
const VOCAB: usize = 4;
const D: usize = 128;
const H: usize = 2;
const HD: usize = 64;
const FF: usize = 12;

fn quantize<S: Shape>(tensor: &CpuTensor<f32, S>) -> CpuTensor<f32, S> {
    tensor.to_bf16().to_f32()
}

fn main() {
    let tokens = [0, 1, 2, 3];
    let targets = [1, 2, 3, 0];
    let mut model = Dense::<N, T, VOCAB, D, H, HD, FF>::new(100);
    let config = AdamWConfig {
        learning_rate: 0.03,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut optimizer = DenseAdamW::new(config);
    let mut master: CpuTensor<f32, Rank2<D, VOCAB>> = model.lm_head.w.clone();
    let mut master_moments = AdamWMoments::<Rank2<D, VOCAB>>::zeros();
    let mut step_count = 0u64;

    for step in 0..=420 {
        model.zero_grad();

        // Forward, mirroring Dense::forward with a bf16 head.
        let (x, embedding_ctx) = model.embedding.forward(tokens);
        let attention_residual = x.clone();
        let (normalized, attention_norm_ctx) = model.attention_norm.forward(x);
        let (q, q_ctx) = model.q_proj.forward(normalized.clone());
        let (k, k_ctx) = model.k_proj.forward(normalized.clone());
        let (v, v_ctx) = model.v_proj.forward(normalized);
        let (q, ()) = Rope::<N, T, D, H, HD>.forward(q);
        let (k, ()) = Rope::<N, T, D, H, HD>.forward(k);
        let (attended, attention_ctx) = CausalAttention::<N, T, D, H, HD>.forward((q, k, v));
        let (attention_output, o_ctx) = model.o_proj.forward(attended);
        let x = attention_residual.add(&attention_output);

        let ffn_residual = x.clone();
        let (normalized, ffn_norm_ctx) = model.ffn_norm.forward(x);
        let (gate, gate_ctx) = model.gate_proj.forward(normalized.clone());
        let (up, up_ctx) = model.up_proj.forward(normalized);
        let (activated, swiglu_ctx) = SwiGlu::<N, FF>.forward((gate, up));
        let (ffn_output, down_ctx) = model.down_proj.forward(activated);
        let x = ffn_residual.add(&ffn_output);

        let (normalized, final_norm_ctx) = model.final_norm.forward(x);
        let head_input = quantize(&normalized);
        let compute_w = quantize(&master);
        let logits = quantize(&head_input.matmul(&compute_w));
        let (loss, loss_ctx) = SoftmaxCrossEntropy::<N, VOCAB>.forward(SoftmaxCrossEntropyInput {
            logits: logits.clone(),
            targets,
        });

        if step % 60 == 0 || step == 420 {
            let mut target_probabilities = [0.0f32; N];
            for row in 0..N {
                let row_logits = &logits.as_slice()[row * VOCAB..(row + 1) * VOCAB];
                let max = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row_logits.iter().map(|&l| (l - max).exp()).sum();
                target_probabilities[row] = (row_logits[targets[row]] - max).exp() / sum;
            }
            println!(
                "step={step} loss={:.6} p_target={target_probabilities:.4?}",
                loss.as_slice()[0]
            );
        }

        // Backward, mirroring Dense::backward with the bf16 head.
        let mut loss_module = SoftmaxCrossEntropy::<N, VOCAB>;
        let dlogits = loss_module
            .backward(loss_ctx, CpuTensor::from_slice(&[1.0]))
            .logits;
        let dlogits = quantize(&dlogits);
        let dw = quantize(&head_input.matmul_tn(&dlogits));
        let dx = quantize(&dlogits.matmul_nt(&compute_w));
        let dx = model.final_norm.backward(final_norm_ctx, dx);

        let dactivated = model.down_proj.backward(down_ctx, dx.clone());
        let (dgate, dup) = SwiGlu::<N, FF>.backward(swiglu_ctx, dactivated);
        let dnormalized = model
            .gate_proj
            .backward(gate_ctx, dgate)
            .add(&model.up_proj.backward(up_ctx, dup));
        let dffn_input = model.ffn_norm.backward(ffn_norm_ctx, dnormalized);
        let dx = dx.add(&dffn_input);

        let dattended = model.o_proj.backward(o_ctx, dx.clone());
        let (dq, dk, dv) = CausalAttention::<N, T, D, H, HD>.backward(attention_ctx, dattended);
        let dq = Rope::<N, T, D, H, HD>.backward((), dq);
        let dk = Rope::<N, T, D, H, HD>.backward((), dk);
        let dnormalized = model
            .q_proj
            .backward(q_ctx, dq)
            .add(&model.k_proj.backward(k_ctx, dk))
            .add(&model.v_proj.backward(v_ctx, dv));
        let dattn_input = model
            .attention_norm
            .backward(attention_norm_ctx, dnormalized);
        let dx = dx.add(&dattn_input);
        model.embedding.backward(embedding_ctx, dx);

        // Optimizer: fp32 master fed the bf16-rounded head gradients; the
        // model's own (zero-grad) lm_head is untouched because decay is zero.
        step_count += 1;
        adamw_step(&mut master, &dw, &mut master_moments, config, step_count);
        optimizer.update(&mut model);
    }
}
