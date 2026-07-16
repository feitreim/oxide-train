//! Row-wise RMS normalization with a learned scale.

use tensor_core::{Rank1, Rank2};
use tensor_cpu::CpuTensor;

use crate::Module;

/// `y[i,j] = x[i,j] * w[j] / sqrt(mean_j(x[i,j]^2) + eps)`.
pub struct RmsNorm<const N: usize, const D: usize> {
    pub w: CpuTensor<f32, Rank1<D>>,
    pub dw: CpuTensor<f32, Rank1<D>>,
    pub eps: f32,
}

pub struct RmsNormCtx<const N: usize, const D: usize> {
    x: CpuTensor<f32, Rank2<N, D>>,
    inv_rms: Box<[f32]>,
}

impl<const N: usize, const D: usize> RmsNorm<N, D> {
    pub fn new(w: CpuTensor<f32, Rank1<D>>, eps: f32) -> Self {
        assert!(D > 0, "RmsNorm feature dimension must be non-zero");
        assert!(eps >= 0.0, "RmsNorm epsilon must be non-negative");
        Self {
            w,
            dw: CpuTensor::zeros(),
            eps,
        }
    }

    pub fn ones(eps: f32) -> Self {
        Self::new(CpuTensor::from_fn(|_| 1.0), eps)
    }
}

impl<const N: usize, const D: usize> Module for RmsNorm<N, D> {
    type Input = CpuTensor<f32, Rank2<N, D>>;
    type Output = CpuTensor<f32, Rank2<N, D>>;
    type Ctx = RmsNormCtx<N, D>;

    fn forward(&self, x: Self::Input) -> (Self::Output, Self::Ctx) {
        let xs = x.as_slice();
        let ws = self.w.as_slice();
        let mut inv_rms = vec![0.0; N].into_boxed_slice();
        let mut y = Self::Output::zeros();

        for i in 0..N {
            let mut sum_sq = 0.0f64;
            for j in 0..D {
                let value = xs[i * D + j] as f64;
                sum_sq += value * value;
            }
            let inv = (sum_sq as f32 / D as f32 + self.eps).sqrt().recip();
            inv_rms[i] = inv;
            for j in 0..D {
                y.as_mut_slice()[i * D + j] = xs[i * D + j] * inv * ws[j];
            }
        }

        (y, RmsNormCtx { x, inv_rms })
    }

    fn backward(&mut self, ctx: Self::Ctx, dy: Self::Output) -> Self::Input {
        let xs = ctx.x.as_slice();
        let dys = dy.as_slice();
        let ws = self.w.as_slice();
        let mut dx = Self::Input::zeros();

        for i in 0..N {
            let inv = ctx.inv_rms[i];
            let mut dot = 0.0f64;
            for j in 0..D {
                dot += dys[i * D + j] as f64 * ws[j] as f64 * xs[i * D + j] as f64;
                self.dw.as_mut_slice()[j] += dys[i * D + j] * xs[i * D + j] * inv;
            }
            let correction = inv * inv * inv * dot as f32 / D as f32;
            for j in 0..D {
                dx.as_mut_slice()[i * D + j] =
                    dys[i * D + j] * ws[j] * inv - xs[i * D + j] * correction;
            }
        }

        dx
    }

    fn zero_grad(&mut self) {
        self.dw = CpuTensor::zeros();
    }
}
