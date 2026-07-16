//! Reference ops for `CpuTensor<f32, _>`.
//!
//! Written for clarity: naive loops, no blocking, no SIMD. Shape agreement is
//! enforced by const generics — note how `matmul` shares `K` between its two
//! argument types and the transposed variants (`matmul_tn`, `matmul_nt`)
//! exist so backward passes never materialize a transposed matrix.

use tensor_core::{Rank2, Shape};

use crate::CpuTensor;

// ---------------------------------------------------------------------------
// Elementwise (any shape)
// ---------------------------------------------------------------------------

impl<S: Shape> CpuTensor<f32, S> {
    pub fn add(&self, rhs: &Self) -> Self {
        self.zip(rhs, |a, b| a + b)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.zip(rhs, |a, b| a - b)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        self.zip(rhs, |a, b| a * b)
    }

    pub fn scale(&self, s: f32) -> Self {
        self.map(|a| a * s)
    }

    /// `self += rhs` — gradient accumulation.
    pub fn add_assign(&mut self, rhs: &Self) {
        for (a, b) in self.as_mut_slice().iter_mut().zip(rhs.as_slice()) {
            *a += b;
        }
    }

    /// `self += s * rhs` — axpy; optimizers and gradcheck live on this.
    pub fn add_scaled_assign(&mut self, s: f32, rhs: &Self) {
        for (a, b) in self.as_mut_slice().iter_mut().zip(rhs.as_slice()) {
            *a += s * b;
        }
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self::from_slice(&self.as_slice().iter().map(|&a| f(a)).collect::<Vec<_>>())
    }

    pub fn zip(&self, rhs: &Self, f: impl Fn(f32, f32) -> f32) -> Self {
        Self::from_slice(
            &self
                .as_slice()
                .iter()
                .zip(rhs.as_slice())
                .map(|(&a, &b)| f(a, b))
                .collect::<Vec<_>>(),
        )
    }

    pub fn sum(&self) -> f32 {
        // f64 accumulator: the reference must not lose to the thing it checks.
        self.as_slice().iter().map(|&a| a as f64).sum::<f64>() as f32
    }

    pub fn mean(&self) -> f32 {
        self.sum() / S::NUM_ELEMENTS as f32
    }

    /// Sum of elementwise products — the flat inner product `<self, rhs>`.
    pub fn dot(&self, rhs: &Self) -> f32 {
        self.as_slice()
            .iter()
            .zip(rhs.as_slice())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum::<f64>() as f32
    }
}

// ---------------------------------------------------------------------------
// Matmul family (Rank2)
// ---------------------------------------------------------------------------

impl<const M: usize, const K: usize> CpuTensor<f32, Rank2<M, K>> {
    /// `C = self . rhs` : `[M,K] x [K,N] -> [M,N]`.
    pub fn matmul<const N: usize>(
        &self,
        rhs: &CpuTensor<f32, Rank2<K, N>>,
    ) -> CpuTensor<f32, Rank2<M, N>> {
        let (a, b) = (self.as_slice(), rhs.as_slice());
        let mut c = CpuTensor::<f32, Rank2<M, N>>::zeros();
        let cs = c.as_mut_slice();
        // i-k-j ordering: the inner loop walks b and c contiguously.
        for i in 0..M {
            for k in 0..K {
                let aik = a[i * K + k];
                for j in 0..N {
                    cs[i * N + j] += aik * b[k * N + j];
                }
            }
        }
        c
    }

    /// `C = self^T . rhs` : `[M,K]^T x [M,N] -> [K,N]`.
    ///
    /// Backward of a linear layer w.r.t. its weight: `dW = x^T . dy`.
    pub fn matmul_tn<const N: usize>(
        &self,
        rhs: &CpuTensor<f32, Rank2<M, N>>,
    ) -> CpuTensor<f32, Rank2<K, N>> {
        let (a, b) = (self.as_slice(), rhs.as_slice());
        let mut c = CpuTensor::<f32, Rank2<K, N>>::zeros();
        let cs = c.as_mut_slice();
        for i in 0..M {
            for k in 0..K {
                let aik = a[i * K + k];
                for j in 0..N {
                    cs[k * N + j] += aik * b[i * N + j];
                }
            }
        }
        c
    }

    /// `C = self . rhs^T` : `[M,K] x [N,K]^T -> [M,N]`.
    ///
    /// Backward of a linear layer w.r.t. its input: `dx = dy . W^T`.
    pub fn matmul_nt<const N: usize>(
        &self,
        rhs: &CpuTensor<f32, Rank2<N, K>>,
    ) -> CpuTensor<f32, Rank2<M, N>> {
        let (a, b) = (self.as_slice(), rhs.as_slice());
        let mut c = CpuTensor::<f32, Rank2<M, N>>::zeros();
        let cs = c.as_mut_slice();
        for i in 0..M {
            for j in 0..N {
                let mut acc = 0.0f32;
                for k in 0..K {
                    acc += a[i * K + k] * b[j * K + k];
                }
                cs[i * N + j] = acc;
            }
        }
        c
    }

    /// Explicit transpose: `[M,K] -> [K,M]`. Tests use it to cross-check the
    /// fused `_tn`/`_nt` variants; real code should prefer those.
    pub fn transpose(&self) -> CpuTensor<f32, Rank2<K, M>> {
        let a = self.as_slice();
        CpuTensor::from_fn(|idx| {
            let (k, i) = (idx / M, idx % M);
            a[i * K + k]
        })
    }
}

// ---------------------------------------------------------------------------
// Row-wise ops (Rank2): rows are the natural axis for attention & losses
// ---------------------------------------------------------------------------

impl<const M: usize, const N: usize> CpuTensor<f32, Rank2<M, N>> {
    /// Numerically-stable softmax over each row.
    pub fn softmax_rows(&self) -> Self {
        let a = self.as_slice();
        let mut out = Self::zeros();
        let o = out.as_mut_slice();
        for i in 0..M {
            let row = &a[i * N..(i + 1) * N];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f64;
            for j in 0..N {
                let e = (row[j] - max).exp();
                o[i * N + j] = e;
                denom += e as f64;
            }
            for j in 0..N {
                o[i * N + j] /= denom as f32;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use tensor_core::{Rank1, Rank2};

    use crate::CpuTensor;

    #[test]
    fn matmul_known_values() {
        // [1 2; 3 4] . [5 6; 7 8] = [19 22; 43 50]
        let a = CpuTensor::<f32, Rank2<2, 2>>::from_slice(&[1., 2., 3., 4.]);
        let b = CpuTensor::<f32, Rank2<2, 2>>::from_slice(&[5., 6., 7., 8.]);
        assert_eq!(a.matmul(&b).as_slice(), &[19., 22., 43., 50.]);
    }

    #[test]
    fn matmul_rectangular_shapes_compose() {
        // The point of const-generic shapes: this compiles only if K matches.
        let a = CpuTensor::<f32, Rank2<3, 5>>::uniform(1);
        let b = CpuTensor::<f32, Rank2<5, 4>>::uniform(2);
        let c: CpuTensor<f32, Rank2<3, 4>> = a.matmul(&b);
        assert_eq!(c.as_slice().len(), 12);
    }

    #[test]
    fn fused_transposed_matmuls_match_explicit_transpose() {
        let a = CpuTensor::<f32, Rank2<4, 3>>::uniform(1);
        let b = CpuTensor::<f32, Rank2<4, 5>>::uniform(2);
        let fused = a.matmul_tn(&b); // A^T . B : [3,5]
        let explicit = a.transpose().matmul(&b);
        for (x, y) in fused.as_slice().iter().zip(explicit.as_slice()) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }

        let c = CpuTensor::<f32, Rank2<5, 3>>::uniform(3);
        let fused = a.matmul_nt(&c); // A . C^T : [4,5]
        let explicit = a.matmul(&c.transpose());
        for (x, y) in fused.as_slice().iter().zip(explicit.as_slice()) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }

    #[test]
    fn softmax_rows_are_distributions() {
        let x = CpuTensor::<f32, Rank2<6, 17>>::uniform(7);
        let s = x.softmax_rows();
        for i in 0..6 {
            let row = &s.as_slice()[i * 17..(i + 1) * 17];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
            assert!(row.iter().all(|&p| p > 0.0));
        }
    }

    #[test]
    fn elementwise_and_reductions() {
        let a = CpuTensor::<f32, Rank1<8>>::from_slice(&[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = a.scale(2.0);
        assert_eq!(b.sum(), 72.0);
        assert_eq!(a.add(&a).as_slice(), b.as_slice());
        assert_eq!(a.dot(&a), 204.0);
        let mut c = a.clone();
        c.add_scaled_assign(-1.0, &a);
        assert_eq!(c.sum(), 0.0);
    }
}
