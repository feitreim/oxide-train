//! Bias-free linear layer (Llama-style): `y = x . W`.
//!
//! The first leaf module, and the template for all the others: params + grad
//! buffers live in the module, `forward` moves its input into `Ctx`, and
//! `backward` accumulates `dW` and returns `dx` using the fused transposed
//! matmuls (never materializing a transpose).

use tensor_core::Rank2;
use tensor_cpu::CpuTensor;

use crate::Module;

/// `x: [N, IN] -> y: [N, OUT]` with weight `W: [IN, OUT]`.
///
/// `N` is the (compile-time) number of rows flowing through — for an LLM,
/// `B * T` flattened.
pub struct Linear<const N: usize, const IN: usize, const OUT: usize> {
    pub w: CpuTensor<f32, Rank2<IN, OUT>>,
    pub dw: CpuTensor<f32, Rank2<IN, OUT>>,
}

impl<const N: usize, const IN: usize, const OUT: usize> Linear<N, IN, OUT> {
    pub fn new(w: CpuTensor<f32, Rank2<IN, OUT>>) -> Self {
        Self {
            w,
            dw: CpuTensor::zeros(),
        }
    }

    /// Deterministic uniform init (placeholder until proper scaled init).
    pub fn uniform(seed: u64) -> Self {
        Self::new(CpuTensor::uniform(seed))
    }
}

impl<const N: usize, const IN: usize, const OUT: usize> Module for Linear<N, IN, OUT> {
    type Input = CpuTensor<f32, Rank2<N, IN>>;
    type Output = CpuTensor<f32, Rank2<N, OUT>>;
    /// Backward needs `x` (for `dW = x^T . dy`), so forward moves it here.
    type Ctx = CpuTensor<f32, Rank2<N, IN>>;

    fn forward(&self, x: Self::Input) -> (Self::Output, Self::Ctx) {
        (x.matmul(&self.w), x)
    }

    fn backward(&mut self, x: Self::Ctx, dy: Self::Output) -> Self::Input {
        self.dw.add_assign(&x.matmul_tn(&dy)); // dW += x^T . dy
        dy.matmul_nt(&self.w) // dx  = dy . W^T
    }

    fn zero_grad(&mut self) {
        self.dw = CpuTensor::zeros();
    }
}
