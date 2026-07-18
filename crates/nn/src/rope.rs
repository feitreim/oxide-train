//! Rotary position embeddings (RoPE).

use tensor_core::Rank2;
use tensor_cpu::CpuTensor;

use crate::Module;

/// Applies Dense-style rotary position embeddings independently to every head.
///
/// `N` is the flattened batch/sequence dimension. Rows are interpreted as
/// contiguous sequences of length `T`; `D == H * HD` and `HD` must be even.
#[derive(Copy, Clone, Debug, Default)]
pub struct Rope<const N: usize, const T: usize, const D: usize, const H: usize, const HD: usize>;

impl<const N: usize, const T: usize, const D: usize, const H: usize, const HD: usize>
    Rope<N, T, D, H, HD>
{
    fn validate() {
        assert!(T > 0, "RoPE sequence length must be non-zero");
        assert_eq!(N % T, 0, "RoPE rows must contain whole sequences");
        assert_eq!(D, H * HD, "RoPE requires D == H * HD");
        assert_eq!(HD % 2, 0, "RoPE head dimension must be even");
    }

    fn sin_cos(position: usize, pair: usize) -> (f32, f32) {
        let frequency = 10_000.0f32.powf(-((2 * pair) as f32) / HD as f32);
        (position as f32 * frequency).sin_cos()
    }
}

impl<const N: usize, const T: usize, const D: usize, const H: usize, const HD: usize> Module
    for Rope<N, T, D, H, HD>
{
    type Input = CpuTensor<f32, Rank2<N, D>>;
    type Output = CpuTensor<f32, Rank2<N, D>>;
    type Ctx = ();

    fn forward(&self, x: Self::Input) -> (Self::Output, Self::Ctx) {
        Self::validate();
        let mut y = Self::Output::zeros();
        for row in 0..N {
            let position = row % T;
            for head in 0..H {
                for pair in 0..HD / 2 {
                    let (sin, cos) = Self::sin_cos(position, pair);
                    let base = row * D + head * HD + 2 * pair;
                    let x0 = x.as_slice()[base];
                    let x1 = x.as_slice()[base + 1];
                    y.as_mut_slice()[base] = x0 * cos - x1 * sin;
                    y.as_mut_slice()[base + 1] = x0 * sin + x1 * cos;
                }
            }
        }
        (y, ())
    }

    fn backward(&mut self, (): Self::Ctx, dy: Self::Output) -> Self::Input {
        Self::validate();
        let mut dx = Self::Input::zeros();
        for row in 0..N {
            let position = row % T;
            for head in 0..H {
                for pair in 0..HD / 2 {
                    let (sin, cos) = Self::sin_cos(position, pair);
                    let base = row * D + head * HD + 2 * pair;
                    let dy0 = dy.as_slice()[base];
                    let dy1 = dy.as_slice()[base + 1];
                    dx.as_mut_slice()[base] = dy0 * cos + dy1 * sin;
                    dx.as_mut_slice()[base + 1] = -dy0 * sin + dy1 * cos;
                }
            }
        }
        dx
    }
}
