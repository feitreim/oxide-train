//! An `L`-block Dense CPU reference with [`MoeFfn`] substituted for its FFNs.

use tensor_core::{Rank1, Rank2, Shape};
use tensor_cpu::CpuTensor;

use crate::attention::CausalAttentionCtx;
use crate::cross_entropy::SoftmaxCrossEntropyCtx;
use crate::moe::MoeFfnCtx;
use crate::rms_norm::RmsNormCtx;
use crate::{
    CausalAttention, Embedding, Linear, Module, MoeFfn, RmsNorm, Rope, SoftmaxCrossEntropy,
    SoftmaxCrossEntropyInput, TokenIds,
};

/// One pre-norm decoder block whose feed-forward branch is a mixture of experts.
pub struct MoeBlock<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> {
    pub attention_norm: RmsNorm<N, D>,
    pub q_proj: Linear<N, D, D>,
    pub k_proj: Linear<N, D, D>,
    pub v_proj: Linear<N, D, D>,
    pub o_proj: Linear<N, D, D>,
    pub ffn_norm: RmsNorm<N, D>,
    pub ffn: MoeFfn<N, D, FF, E, K, C>,
}

pub struct MoeBlockCtx<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> {
    attention_norm: RmsNormCtx<N, D>,
    q_proj: CpuTensor<f32, Rank2<N, D>>,
    k_proj: CpuTensor<f32, Rank2<N, D>>,
    v_proj: CpuTensor<f32, Rank2<N, D>>,
    attention: CausalAttentionCtx<N, T, D, H>,
    o_proj: CpuTensor<f32, Rank2<N, D>>,
    ffn_norm: RmsNormCtx<N, D>,
    ffn: MoeFfnCtx<N, D, FF, E, K, C>,
}

fn initialized<S: Shape>(seed: u64, scale: f32) -> CpuTensor<f32, S> {
    CpuTensor::uniform(seed).scale(scale)
}

fn sgd<S: Shape>(parameter: &mut CpuTensor<f32, S>, gradient: &CpuTensor<f32, S>, lr: f32) {
    parameter.add_scaled_assign(-lr, gradient);
}

impl<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> MoeBlock<N, T, D, H, HD, FF, E, K, C>
{
    /// Deterministic scaled initialization consuming seeds in field order, so
    /// per-block construction reproduces a whole-model seed sequence exactly.
    pub fn new(take_seed: &mut impl FnMut() -> u64, aux_coefficient: f32) -> Self {
        assert!(D > 0, "model width must be non-zero");
        assert!(H > 0, "head count must be non-zero");
        assert!(HD > 0, "head dimension must be non-zero");
        assert_eq!(HD % 2, 0, "RoPE head dimension must be even");
        assert!(FF > 0, "FFN width must be non-zero");
        assert_eq!(N % T, 0, "model rows must contain whole sequences");
        assert_eq!(D, H * HD, "model requires D == H * HD");
        let hidden_scale = (D as f32).sqrt().recip();
        Self {
            attention_norm: RmsNorm::ones(1e-5),
            q_proj: Linear::new(initialized(take_seed(), hidden_scale)),
            k_proj: Linear::new(initialized(take_seed(), hidden_scale)),
            v_proj: Linear::new(initialized(take_seed(), hidden_scale)),
            o_proj: Linear::new(initialized(take_seed(), hidden_scale)),
            ffn_norm: RmsNorm::ones(1e-5),
            ffn: MoeFfn::initialized(take_seed(), aux_coefficient),
        }
    }

    pub fn forward(
        &self,
        x: CpuTensor<f32, Rank2<N, D>>,
    ) -> (
        CpuTensor<f32, Rank2<N, D>>,
        MoeBlockCtx<N, T, D, H, FF, E, K, C>,
    ) {
        let attention_residual = x.clone();
        let (normalized, attention_norm) = self.attention_norm.forward(x);
        let (q, q_proj) = self.q_proj.forward(normalized.clone());
        let (k, k_proj) = self.k_proj.forward(normalized.clone());
        let (v, v_proj) = self.v_proj.forward(normalized);
        let (q, ()) = Rope::<N, T, D, H, HD>.forward(q);
        let (k, ()) = Rope::<N, T, D, H, HD>.forward(k);
        let (attended, attention) = CausalAttention::<N, T, D, H, HD>.forward((q, k, v));
        let (attention_output, o_proj) = self.o_proj.forward(attended);
        let x = attention_residual.add(&attention_output);

        let ffn_residual = x.clone();
        let (normalized, ffn_norm) = self.ffn_norm.forward(x);
        let (ffn_output, ffn) = self.ffn.forward(normalized);
        let x = ffn_residual.add(&ffn_output);

        (
            x,
            MoeBlockCtx {
                attention_norm,
                q_proj,
                k_proj,
                v_proj,
                attention,
                o_proj,
                ffn_norm,
                ffn,
            },
        )
    }

    pub fn backward(
        &mut self,
        ctx: MoeBlockCtx<N, T, D, H, FF, E, K, C>,
        dx: CpuTensor<f32, Rank2<N, D>>,
    ) -> CpuTensor<f32, Rank2<N, D>> {
        let dnormalized = self.ffn.backward(ctx.ffn, dx.clone());
        let dffn_input = self.ffn_norm.backward(ctx.ffn_norm, dnormalized);
        let dx = dx.add(&dffn_input);

        let dattended = self.o_proj.backward(ctx.o_proj, dx.clone());
        let (dq, dk, dv) = CausalAttention::<N, T, D, H, HD>.backward(ctx.attention, dattended);
        let dq = Rope::<N, T, D, H, HD>.backward((), dq);
        let dk = Rope::<N, T, D, H, HD>.backward((), dk);
        let dnormalized = self
            .q_proj
            .backward(ctx.q_proj, dq)
            .add(&self.k_proj.backward(ctx.k_proj, dk))
            .add(&self.v_proj.backward(ctx.v_proj, dv));
        let dattn_input = self
            .attention_norm
            .backward(ctx.attention_norm, dnormalized);
        dx.add(&dattn_input)
    }

    pub fn zero_grad(&mut self) {
        self.attention_norm.zero_grad();
        self.q_proj.zero_grad();
        self.k_proj.zero_grad();
        self.v_proj.zero_grad();
        self.o_proj.zero_grad();
        self.ffn_norm.zero_grad();
        self.ffn.zero_grad();
    }

    pub fn sgd_step(&mut self, learning_rate: f32) {
        sgd(
            &mut self.attention_norm.w,
            &self.attention_norm.dw,
            learning_rate,
        );
        sgd(&mut self.q_proj.w, &self.q_proj.dw, learning_rate);
        sgd(&mut self.k_proj.w, &self.k_proj.dw, learning_rate);
        sgd(&mut self.v_proj.w, &self.v_proj.dw, learning_rate);
        sgd(&mut self.o_proj.w, &self.o_proj.dw, learning_rate);
        sgd(&mut self.ffn_norm.w, &self.ffn_norm.dw, learning_rate);
        self.ffn.sgd_step(learning_rate);
    }
}

/// An `L`-deep stack of [`MoeBlock`]s between a token embedding and an lm-head.
pub struct MoeDense<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
    const L: usize = 1,
> {
    pub embedding: Embedding<N, VOCAB, D>,
    pub blocks: Vec<MoeBlock<N, T, D, H, HD, FF, E, K, C>>,
    pub final_norm: RmsNorm<N, D>,
    pub lm_head: Linear<N, D, VOCAB>,
}

pub struct MoeDenseCtx<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> {
    embedding: TokenIds<N>,
    blocks: Vec<MoeBlockCtx<N, T, D, H, FF, E, K, C>>,
    final_norm: RmsNormCtx<N, D>,
    lm_head: CpuTensor<f32, Rank2<N, D>>,
    loss: SoftmaxCrossEntropyCtx<N, VOCAB>,
}

impl<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
    const L: usize,
> MoeDense<N, T, VOCAB, D, H, HD, FF, E, K, C, L>
{
    /// Deterministic scaled initialization suitable for small CPU tests.
    pub fn new(seed: u64, aux_coefficient: f32) -> Self {
        assert!(T > 0, "sequence length must be non-zero");
        assert!(VOCAB > 0, "vocabulary must be non-zero");
        assert!(L > 0, "block count must be non-zero");
        let hidden_scale = (D as f32).sqrt().recip();
        let mut next_seed = seed;
        let mut take_seed = || {
            let current = next_seed;
            next_seed += 1;
            current
        };

        Self {
            embedding: Embedding::new(initialized(take_seed(), hidden_scale)),
            blocks: (0..L)
                .map(|_| MoeBlock::new(&mut take_seed, aux_coefficient))
                .collect(),
            final_norm: RmsNorm::ones(1e-5),
            lm_head: Linear::new(initialized(take_seed(), hidden_scale)),
        }
    }

    /// Sum of every block's coefficient-weighted auxiliary loss from the most
    /// recent forward pass.
    pub fn aux_loss(&self) -> f32 {
        self.blocks
            .iter()
            .map(|block| block.ffn.aux_coefficient * block.ffn.last_aux_loss.get())
            .sum()
    }

    pub fn forward(
        &self,
        tokens: TokenIds<N>,
        targets: TokenIds<N>,
    ) -> (
        CpuTensor<f32, Rank1<1>>,
        MoeDenseCtx<N, T, VOCAB, D, H, FF, E, K, C>,
    ) {
        let (mut x, embedding) = self.embedding.forward(tokens);

        let mut blocks = Vec::with_capacity(L);
        for block in &self.blocks {
            let (next, ctx) = block.forward(x);
            x = next;
            blocks.push(ctx);
        }

        let (normalized, final_norm) = self.final_norm.forward(x);
        let (logits, lm_head) = self.lm_head.forward(normalized);
        let (mut loss, loss_ctx) =
            SoftmaxCrossEntropy::<N, VOCAB>.forward(SoftmaxCrossEntropyInput { logits, targets });
        loss.as_mut_slice()[0] += self.aux_loss();

        (
            loss,
            MoeDenseCtx {
                embedding,
                blocks,
                final_norm,
                lm_head,
                loss: loss_ctx,
            },
        )
    }

    pub fn backward(&mut self, ctx: MoeDenseCtx<N, T, VOCAB, D, H, FF, E, K, C>) {
        let mut loss = SoftmaxCrossEntropy::<N, VOCAB>;
        let dloss = CpuTensor::from_slice(&[1.0]);
        let dlogits = loss.backward(ctx.loss, dloss).logits;
        let dx = self.lm_head.backward(ctx.lm_head, dlogits);
        let mut dx = self.final_norm.backward(ctx.final_norm, dx);

        for (block, block_ctx) in self.blocks.iter_mut().zip(ctx.blocks).rev() {
            dx = block.backward(block_ctx, dx);
        }

        self.embedding.backward(ctx.embedding, dx);
    }

    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        for block in &mut self.blocks {
            block.zero_grad();
        }
        self.final_norm.zero_grad();
        self.lm_head.zero_grad();
    }

    /// Plain SGD is intentionally enough for the tiny-batch correctness gate.
    pub fn sgd_step(&mut self, learning_rate: f32) {
        sgd(&mut self.embedding.w, &self.embedding.dw, learning_rate);
        for block in &mut self.blocks {
            block.sgd_step(learning_rate);
        }
        sgd(&mut self.final_norm.w, &self.final_norm.dw, learning_rate);
        sgd(&mut self.lm_head.w, &self.lm_head.dw, learning_rate);
    }
}
