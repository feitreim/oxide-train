//! Naive causal multi-head scaled dot-product attention.

use tensor_core::{Rank2, Rank3};
use tensor_cpu::CpuTensor;

use crate::Module;

pub type AttentionInput<const N: usize, const D: usize> = (
    CpuTensor<f32, Rank2<N, D>>,
    CpuTensor<f32, Rank2<N, D>>,
    CpuTensor<f32, Rank2<N, D>>,
);

pub struct CausalAttentionCtx<const N: usize, const T: usize, const D: usize, const H: usize> {
    q: CpuTensor<f32, Rank2<N, D>>,
    k: CpuTensor<f32, Rank2<N, D>>,
    v: CpuTensor<f32, Rank2<N, D>>,
    probabilities: CpuTensor<f32, Rank3<N, H, T>>,
}

/// Parameter-free attention over projected Q/K/V tensors.
///
/// Rows are flattened batches of contiguous `T`-token sequences. Attention is
/// causal within each sequence and cannot cross a sequence boundary.
#[derive(Copy, Clone, Debug, Default)]
pub struct CausalAttention<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
>;

impl<const N: usize, const T: usize, const D: usize, const H: usize, const HD: usize>
    CausalAttention<N, T, D, H, HD>
{
    fn validate() {
        assert!(T > 0, "attention sequence length must be non-zero");
        assert_eq!(N % T, 0, "attention rows must contain whole sequences");
        assert_eq!(D, H * HD, "attention requires D == H * HD");
        assert!(HD > 0, "attention head dimension must be non-zero");
    }

    fn value_index(row: usize, head: usize, dim: usize) -> usize {
        row * D + head * HD + dim
    }

    fn probability_index(row: usize, head: usize, key_position: usize) -> usize {
        (row * H + head) * T + key_position
    }
}

impl<const N: usize, const T: usize, const D: usize, const H: usize, const HD: usize> Module
    for CausalAttention<N, T, D, H, HD>
{
    type Input = AttentionInput<N, D>;
    type Output = CpuTensor<f32, Rank2<N, D>>;
    type Ctx = CausalAttentionCtx<N, T, D, H>;

    fn forward(&self, (q, k, v): Self::Input) -> (Self::Output, Self::Ctx) {
        Self::validate();
        let scale = (HD as f32).sqrt().recip();
        let mut probabilities = CpuTensor::<f32, Rank3<N, H, T>>::zeros();
        let mut output = Self::Output::zeros();

        for query_row in 0..N {
            let query_position = query_row % T;
            let sequence_start = query_row - query_position;
            for head in 0..H {
                let mut max_score = f32::NEG_INFINITY;
                for key_position in 0..=query_position {
                    let key_row = sequence_start + key_position;
                    let mut dot = 0.0f64;
                    for dim in 0..HD {
                        dot += q.as_slice()[Self::value_index(query_row, head, dim)] as f64
                            * k.as_slice()[Self::value_index(key_row, head, dim)] as f64;
                    }
                    let score = dot as f32 * scale;
                    let probability_index = Self::probability_index(query_row, head, key_position);
                    probabilities.as_mut_slice()[probability_index] = score;
                    max_score = max_score.max(score);
                }

                let mut denominator = 0.0f64;
                for key_position in 0..=query_position {
                    let index = Self::probability_index(query_row, head, key_position);
                    let exponential = (probabilities.as_slice()[index] - max_score).exp();
                    probabilities.as_mut_slice()[index] = exponential;
                    denominator += exponential as f64;
                }
                for key_position in 0..=query_position {
                    let index = Self::probability_index(query_row, head, key_position);
                    probabilities.as_mut_slice()[index] /= denominator as f32;
                }

                for dim in 0..HD {
                    let mut value = 0.0f64;
                    for key_position in 0..=query_position {
                        let key_row = sequence_start + key_position;
                        let probability = probabilities.as_slice()
                            [Self::probability_index(query_row, head, key_position)];
                        value += probability as f64
                            * v.as_slice()[Self::value_index(key_row, head, dim)] as f64;
                    }
                    output.as_mut_slice()[Self::value_index(query_row, head, dim)] = value as f32;
                }
            }
        }

        (
            output,
            CausalAttentionCtx {
                q,
                k,
                v,
                probabilities,
            },
        )
    }

    fn backward(&mut self, ctx: Self::Ctx, dy: Self::Output) -> Self::Input {
        Self::validate();
        let scale = (HD as f32).sqrt().recip();
        let mut dq = Self::Output::zeros();
        let mut dk = Self::Output::zeros();
        let mut dv = Self::Output::zeros();

        for query_row in 0..N {
            let query_position = query_row % T;
            let sequence_start = query_row - query_position;
            for head in 0..H {
                let mut probability_gradient = vec![0.0f32; query_position + 1];
                for (key_position, dp_slot) in probability_gradient.iter_mut().enumerate() {
                    let key_row = sequence_start + key_position;
                    let probability = ctx.probabilities.as_slice()
                        [Self::probability_index(query_row, head, key_position)];
                    let mut dp = 0.0f64;
                    for dim in 0..HD {
                        let output_index = Self::value_index(query_row, head, dim);
                        let value_index = Self::value_index(key_row, head, dim);
                        dp += dy.as_slice()[output_index] as f64
                            * ctx.v.as_slice()[value_index] as f64;
                        dv.as_mut_slice()[value_index] += probability * dy.as_slice()[output_index];
                    }
                    *dp_slot = dp as f32;
                }

                let softmax_dot = (0..=query_position)
                    .map(|key_position| {
                        ctx.probabilities.as_slice()
                            [Self::probability_index(query_row, head, key_position)]
                            as f64
                            * probability_gradient[key_position] as f64
                    })
                    .sum::<f64>() as f32;

                for (key_position, &dp) in probability_gradient.iter().enumerate() {
                    let key_row = sequence_start + key_position;
                    let probability = ctx.probabilities.as_slice()
                        [Self::probability_index(query_row, head, key_position)];
                    let ds = probability * (dp - softmax_dot) * scale;
                    for dim in 0..HD {
                        let query_index = Self::value_index(query_row, head, dim);
                        let key_index = Self::value_index(key_row, head, dim);
                        dq.as_mut_slice()[query_index] += ds * ctx.k.as_slice()[key_index];
                        dk.as_mut_slice()[key_index] += ds * ctx.q.as_slice()[query_index];
                    }
                }
            }
        }

        (dq, dk, dv)
    }
}
