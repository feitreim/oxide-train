//! Tensor element (dtype) types.

/// Runtime dtype tag, mostly for debug output and shard-file headers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    /// Token ids. r50k_base has 50,257 tokens, so ids fit in `u16`.
    U16,
    U32,
}

/// A type that can be stored in a tensor.
///
/// Deliberately does *not* carry arithmetic: CPU reference ops are written
/// concretely for `f32` (clarity over genericity), and GPU kernels get their
/// arithmetic from the cuda-oxide device crates. `bf16` joins this list when
/// the mixed-precision phase starts.
pub trait Element:
    Copy + Clone + Default + PartialEq + Send + Sync + core::fmt::Debug + 'static
{
    const DTYPE: DType;
    const ZERO: Self;
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
    const ZERO: Self = 0.0;
}

impl Element for u16 {
    const DTYPE: DType = DType::U16;
    const ZERO: Self = 0;
}

impl Element for u32 {
    const DTYPE: DType = DType::U32;
    const ZERO: Self = 0;
}
