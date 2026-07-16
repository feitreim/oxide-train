//! Parameter-free SwiGLU activation.

use tensor_core::Rank2;
use tensor_cpu::CpuTensor;

use crate::Module;

/// Applies `silu(gate) * up` to a pair of equally-shaped tensors.
#[derive(Copy, Clone, Debug, Default)]
pub struct SwiGlu<const N: usize, const D: usize>;

pub struct SwiGluCtx<const N: usize, const D: usize> {
    gate: CpuTensor<f32, Rank2<N, D>>,
    up: CpuTensor<f32, Rank2<N, D>>,
}

impl<const N: usize, const D: usize> Module for SwiGlu<N, D> {
    type Input = (CpuTensor<f32, Rank2<N, D>>, CpuTensor<f32, Rank2<N, D>>);
    type Output = CpuTensor<f32, Rank2<N, D>>;
    type Ctx = SwiGluCtx<N, D>;

    fn forward(&self, (gate, up): Self::Input) -> (Self::Output, Self::Ctx) {
        let y = gate.zip(&up, |g, u| {
            let sigmoid = 1.0 / (1.0 + (-g).exp());
            g * sigmoid * u
        });
        (y, SwiGluCtx { gate, up })
    }

    fn backward(&mut self, ctx: Self::Ctx, dy: Self::Output) -> Self::Input {
        let gate = ctx.gate.as_slice();
        let up = ctx.up.as_slice();
        let dys = dy.as_slice();
        let mut dgate = Self::Output::zeros();
        let mut dup = Self::Output::zeros();

        for i in 0..gate.len() {
            let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
            let silu = gate[i] * sigmoid;
            let dsilu = sigmoid * (1.0 + gate[i] * (1.0 - sigmoid));
            dgate.as_mut_slice()[i] = dys[i] * up[i] * dsilu;
            dup.as_mut_slice()[i] = dys[i] * silu;
        }
        (dgate, dup)
    }
}
