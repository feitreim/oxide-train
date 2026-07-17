//! `CpuTensor`: the reference tensor.
//!
//! Its job is *correctness, not speed*: every GPU kernel is validated against
//! these implementations, and every backward pass is finite-difference-checked
//! against them. Ops are written as plainly as possible — if a CPU op here is
//! clever, it's wrong.
//!
//! Storage is a heap `Box<[E]>` in row-major order; the shape lives entirely
//! in the type (`CpuTensor<f32, Rank2<M, N>>`), so shape errors are compile
//! errors and there are no runtime stride/shape structs.

mod ops;

use std::marker::PhantomData;

use tensor_core::rng::SplitMix64;
use tensor_core::{Element, Rank2, Shape, Tensor, bf16};

#[derive(Clone, Debug, PartialEq)]
pub struct CpuTensor<E: Element, S: Shape> {
    data: Box<[E]>,
    _shape: PhantomData<S>,
}

impl<E: Element, S: Shape> Tensor for CpuTensor<E, S> {
    type Elem = E;
    type Shape = S;
}

impl<E: Element, S: Shape> CpuTensor<E, S> {
    pub const LEN: usize = S::NUM_ELEMENTS;

    pub fn zeros() -> Self {
        Self {
            data: vec![E::ZERO; S::NUM_ELEMENTS].into_boxed_slice(),
            _shape: PhantomData,
        }
    }

    /// Build from a function of the flat (row-major) index.
    pub fn from_fn(f: impl FnMut(usize) -> E) -> Self {
        Self {
            data: (0..S::NUM_ELEMENTS).map(f).collect(),
            _shape: PhantomData,
        }
    }

    /// Copy from a slice; length is checked against the static shape.
    pub fn from_slice(src: &[E]) -> Self {
        assert_eq!(src.len(), S::NUM_ELEMENTS, "slice length != shape volume");
        Self {
            data: src.into(),
            _shape: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[E] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [E] {
        &mut self.data
    }
}

impl<S: Shape> CpuTensor<f32, S> {
    /// Deterministic uniform init in `[-1, 1)`; same generator as the GPU test
    /// harness, so parity tests can reproduce inputs on both sides from a seed.
    pub fn uniform(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        Self::from_fn(|_| rng.next_uniform())
    }

    /// Round an fp32 tensor to bf16 while preserving its static shape.
    pub fn to_bf16(&self) -> CpuTensor<bf16, S> {
        CpuTensor::from_fn(|i| bf16::from_f32(self.data[i]))
    }
}

impl<S: Shape> CpuTensor<bf16, S> {
    /// Widen a bf16 tensor to fp32 exactly while preserving its static shape.
    pub fn to_f32(&self) -> CpuTensor<f32, S> {
        CpuTensor::from_fn(|i| self.data[i].to_f32())
    }
}

impl<E: Element, const M: usize, const N: usize> CpuTensor<E, Rank2<M, N>> {
    /// Row-major 2-D accessor (debug/test convenience; ops index flatly).
    pub fn at(&self, i: usize, j: usize) -> E {
        debug_assert!(i < M && j < N);
        self.data[i * N + j]
    }
}

#[cfg(test)]
mod tests {
    use tensor_core::{Rank1, bf16};

    use super::CpuTensor;

    #[test]
    fn f32_bf16_conversions_preserve_shape_and_expected_values() {
        let fp32 =
            CpuTensor::<f32, Rank1<4>>::from_slice(&[0.0, 1.0, -2.5, f32::from_bits(0x3f80_8000)]);
        let compute: CpuTensor<bf16, Rank1<4>> = fp32.to_bf16();

        assert_eq!(
            compute.as_slice(),
            &[
                bf16::from_f32(0.0),
                bf16::from_f32(1.0),
                bf16::from_f32(-2.5),
                bf16::from_f32(1.0),
            ]
        );
        assert_eq!(compute.to_f32().as_slice(), &[0.0, 1.0, -2.5, 1.0]);
    }
}
