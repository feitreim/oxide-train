//! Fused numerically-stable softmax cross entropy.

use tensor_core::{Rank1, Rank2};
use tensor_cpu::CpuTensor;

use crate::Module;

pub struct SoftmaxCrossEntropyInput<const N: usize, const C: usize> {
    pub logits: CpuTensor<f32, Rank2<N, C>>,
    pub targets: [usize; N],
}

pub struct SoftmaxCrossEntropyCtx<const N: usize, const C: usize> {
    probabilities: CpuTensor<f32, Rank2<N, C>>,
    targets: [usize; N],
}

/// Mean cross entropy over rows. Backward returns logits gradients and the
/// unchanged targets, preserving the `Input` type required by `Module`.
#[derive(Copy, Clone, Debug, Default)]
pub struct SoftmaxCrossEntropy<const N: usize, const C: usize>;

impl<const N: usize, const C: usize> Module for SoftmaxCrossEntropy<N, C> {
    type Input = SoftmaxCrossEntropyInput<N, C>;
    type Output = CpuTensor<f32, Rank1<1>>;
    type Ctx = SoftmaxCrossEntropyCtx<N, C>;

    fn forward(&self, input: Self::Input) -> (Self::Output, Self::Ctx) {
        assert!(N > 0, "cross entropy batch must be non-empty");
        assert!(C > 0, "cross entropy class dimension must be non-empty");

        let logits = input.logits.as_slice();
        let mut probabilities = CpuTensor::<f32, Rank2<N, C>>::zeros();
        let mut loss = 0.0f64;

        for i in 0..N {
            let target = input.targets[i];
            assert!(target < C, "target {target} is outside class count {C}");
            let row = &logits[i * C..(i + 1) * C];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f64;
            for (j, &logit) in row.iter().enumerate() {
                let exp = (logit - max).exp();
                probabilities.as_mut_slice()[i * C + j] = exp;
                sum_exp += exp as f64;
            }
            for j in 0..C {
                probabilities.as_mut_slice()[i * C + j] /= sum_exp as f32;
            }
            loss += max as f64 + sum_exp.ln() - row[target] as f64;
        }

        (
            CpuTensor::from_slice(&[(loss / N as f64) as f32]),
            SoftmaxCrossEntropyCtx {
                probabilities,
                targets: input.targets,
            },
        )
    }

    fn backward(&mut self, ctx: Self::Ctx, dy: Self::Output) -> Self::Input {
        let scale = dy.as_slice()[0] / N as f32;
        let mut dlogits = ctx.probabilities.scale(scale);
        for i in 0..N {
            dlogits.as_mut_slice()[i * C + ctx.targets[i]] -= scale;
        }
        SoftmaxCrossEntropyInput {
            logits: dlogits,
            targets: ctx.targets,
        }
    }
}
