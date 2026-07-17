//! Tensor element (dtype) types.

pub use half::bf16;

/// Runtime dtype tag, mostly for debug output and shard-file headers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
    /// Token ids. r50k_base has 50,257 tokens, so ids fit in `u16`.
    U16,
    U32,
}

/// A type that can be stored in a tensor.
///
/// Deliberately does *not* carry arithmetic: CPU reference ops are written
/// concretely for `f32` (clarity over genericity), and GPU kernels get their
/// arithmetic from the cuda-oxide device crates.
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

impl Element for bf16 {
    const DTYPE: DType = DType::BF16;
    const ZERO: Self = bf16::ZERO;
}

impl Element for u16 {
    const DTYPE: DType = DType::U16;
    const ZERO: Self = 0;
}

impl Element for u32 {
    const DTYPE: DType = DType::U32;
    const ZERO: Self = 0;
}

#[cfg(test)]
mod tests {
    use super::{DType, Element, bf16};

    #[test]
    fn bf16_is_a_two_byte_tensor_element() {
        assert_eq!(size_of::<bf16>(), 2);
        assert_eq!(bf16::DTYPE, DType::BF16);
        assert_eq!(bf16::ZERO, bf16::from_f32(0.0));
    }

    #[test]
    fn bf16_uses_round_to_nearest_even_conversion() {
        assert_eq!(bf16::from_f32(1.0).to_bits(), 0x3f80);
        assert_eq!(bf16::from_f32(-2.5).to_f32(), -2.5);

        // Exactly halfway between 1.0 and the next bf16 value. The low bf16
        // mantissa is even, so ties-to-even rounds down to 1.0.
        let halfway = f32::from_bits(0x3f80_8000);
        assert_eq!(bf16::from_f32(halfway), bf16::from_f32(1.0));
    }
}
