//! Compile-time tensor shapes.
//!
//! A shape is a zero-sized marker type (`Rank2<M, N>`) implementing [`Shape`].
//! Element counts and dims are associated consts, so kernels and reference
//! code can rely on them being folded at compile time — and on the GPU side,
//! each instantiation monomorphizes into shape-specialized PTX.
//!
//! Rust cannot do arithmetic on const generics in *type position* on this
//! toolchain without `generic_const_exprs`, so the op set is designed to never
//! need it: matmul, attention, norms etc. only ever *share* const parameters
//! between input and output types. Arithmetic in *const item* position (e.g.
//! `NUM_ELEMENTS = A * B`) is fine and used freely.

/// A static tensor shape. Implemented only by the `RankN` marker types.
pub trait Shape: 'static + Copy + Clone + Default + Send + Sync + core::fmt::Debug {
    const RANK: usize;
    const NUM_ELEMENTS: usize;

    /// Fixed-size dims array, e.g. `[M, N]` for `Rank2<M, N>`.
    type Dims: AsRef<[usize]> + Copy + Send + Sync + core::fmt::Debug;
    const DIMS: Self::Dims;
}

/// Shape of a vector of length `A`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rank1<const A: usize>;

/// Shape of an `A x B` matrix (row-major: `B` is contiguous).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rank2<const A: usize, const B: usize>;

/// Shape of an `A x B x C` tensor (row-major: `C` is contiguous).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rank3<const A: usize, const B: usize, const C: usize>;

/// Shape of an `A x B x C x D` tensor (row-major: `D` is contiguous).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rank4<const A: usize, const B: usize, const C: usize, const D: usize>;

impl<const A: usize> Shape for Rank1<A> {
    const RANK: usize = 1;
    const NUM_ELEMENTS: usize = A;
    type Dims = [usize; 1];
    const DIMS: [usize; 1] = [A];
}

impl<const A: usize, const B: usize> Shape for Rank2<A, B> {
    const RANK: usize = 2;
    const NUM_ELEMENTS: usize = A * B;
    type Dims = [usize; 2];
    const DIMS: [usize; 2] = [A, B];
}

impl<const A: usize, const B: usize, const C: usize> Shape for Rank3<A, B, C> {
    const RANK: usize = 3;
    const NUM_ELEMENTS: usize = A * B * C;
    type Dims = [usize; 3];
    const DIMS: [usize; 3] = [A, B, C];
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> Shape for Rank4<A, B, C, D> {
    const RANK: usize = 4;
    const NUM_ELEMENTS: usize = A * B * C * D;
    type Dims = [usize; 4];
    const DIMS: [usize; 4] = [A, B, C, D];
}
