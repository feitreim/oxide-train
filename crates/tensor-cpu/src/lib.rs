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
use tensor_core::{Element, Rank2, Shape, Tensor};

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
}

impl<E: Element, const M: usize, const N: usize> CpuTensor<E, Rank2<M, N>> {
    /// Row-major 2-D accessor (debug/test convenience; ops index flatly).
    pub fn at(&self, i: usize, j: usize) -> E {
        debug_assert!(i < M && j < N);
        self.data[i * N + j]
    }
}
