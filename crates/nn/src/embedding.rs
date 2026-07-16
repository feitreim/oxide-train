//! Token embedding lookup.

use tensor_core::Rank2;
use tensor_cpu::CpuTensor;

use crate::Module;

/// Integer token IDs are deliberately not tensors: they are non-differentiable
/// model inputs and backward returns `()` while accumulating into `dw`.
pub type TokenIds<const N: usize> = [usize; N];

pub struct Embedding<const N: usize, const VOCAB: usize, const D: usize> {
    pub w: CpuTensor<f32, Rank2<VOCAB, D>>,
    pub dw: CpuTensor<f32, Rank2<VOCAB, D>>,
}

impl<const N: usize, const VOCAB: usize, const D: usize> Embedding<N, VOCAB, D> {
    pub fn new(w: CpuTensor<f32, Rank2<VOCAB, D>>) -> Self {
        Self {
            w,
            dw: CpuTensor::zeros(),
        }
    }

    pub fn uniform(seed: u64) -> Self {
        Self::new(CpuTensor::uniform(seed))
    }
}

impl<const N: usize, const VOCAB: usize, const D: usize> Module for Embedding<N, VOCAB, D> {
    type Input = TokenIds<N>;
    type Output = CpuTensor<f32, Rank2<N, D>>;
    type Ctx = TokenIds<N>;

    fn forward(&self, tokens: Self::Input) -> (Self::Output, Self::Ctx) {
        let y = CpuTensor::from_fn(|index| {
            let row = index / D;
            let col = index % D;
            let token = tokens[row];
            assert!(
                token < VOCAB,
                "token ID {token} is outside vocabulary {VOCAB}"
            );
            self.w.as_slice()[token * D + col]
        });
        (y, tokens)
    }

    fn backward(&mut self, tokens: Self::Ctx, dy: Self::Output) -> Self::Input {
        for (row, &token) in tokens.iter().enumerate() {
            for col in 0..D {
                self.dw.as_mut_slice()[token * D + col] += dy.as_slice()[row * D + col];
            }
        }
        tokens
    }

    fn zero_grad(&mut self) {
        self.dw = CpuTensor::zeros();
    }
}
