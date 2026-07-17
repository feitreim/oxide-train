//! Storage-agnostic tensor foundations: compile-time shapes, element types,
//! and the deterministic RNG shared by CPU reference code and GPU test
//! harnesses.
//!
//! Design invariant: **every shape in the training loop is a compile-time
//! constant.** Shapes are zero-sized marker types carrying const generics
//! (`Rank2<M, N>`), so a tensor's shape is part of its *type* and shape
//! mismatches are compile errors, not runtime panics. There is deliberately no
//! runtime `Vec<usize>` shape anywhere.

pub mod element;
pub mod rng;
pub mod shape;

pub use element::{DType, Element, bf16};
pub use shape::{Rank1, Rank2, Rank3, Rank4, Shape};

/// The common surface shared by `CpuTensor` and `GpuTensor`.
///
/// Intentionally minimal: it exists so tests and generic plumbing can be
/// written once over both backends, *not* to unify their op sets. Ops live as
/// inherent methods on the concrete tensor types (GPU ops need streams and
/// launch configs; CPU ops don't), so there is no dynamic dispatch and no
/// lowest-common-denominator API tax.
pub trait Tensor {
    type Elem: Element;
    type Shape: Shape;

    /// Total number of elements (known at compile time).
    const LEN: usize = <Self::Shape as Shape>::NUM_ELEMENTS;
}
