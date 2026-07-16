//! Slicing `(inputs, targets)` batches out of a flat token stream.
//!
//! Next-token prediction: for a window starting at `pos`, inputs are
//! `tokens[pos .. pos+B*T]` reshaped to `[B, T]` row-major, and targets are
//! the same window shifted one token right. Consecutive batches advance by
//! `B*T`, so one pass over the iterator is one epoch over the shard.
//!
//! Deliberately dumb and deterministic — document boundaries are handled by
//! `<|endoftext|>` separators in the stream itself, exactly like llm.c.

use tensor_core::Rank2;
use tensor_cpu::CpuTensor;

/// Iterator of `(inputs, targets)` pairs over a token slice.
///
/// `B` and `T` are const generics: the batch shape is part of the type, so a
/// loader/model shape mismatch is a compile error.
pub struct Batches<'a, const B: usize, const T: usize> {
    tokens: &'a [u16],
    pos: usize,
}

impl<'a, const B: usize, const T: usize> Batches<'a, B, T> {
    pub fn new(tokens: &'a [u16]) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Batches in one epoch: floor((len - 1) / (B*T)).
    pub fn remaining(&self) -> usize {
        (self.tokens.len().saturating_sub(self.pos + 1)) / (B * T)
    }
}

impl<const B: usize, const T: usize> Iterator for Batches<'_, B, T> {
    type Item = (
        CpuTensor<u16, Rank2<B, T>>, // inputs
        CpuTensor<u16, Rank2<B, T>>, // targets (inputs shifted by one)
    );

    fn next(&mut self) -> Option<Self::Item> {
        // Need B*T inputs plus one lookahead token for the last target.
        let end = self.pos + B * T;
        if end + 1 > self.tokens.len() {
            return None;
        }
        let inputs = CpuTensor::from_slice(&self.tokens[self.pos..end]);
        let targets = CpuTensor::from_slice(&self.tokens[self.pos + 1..end + 1]);
        self.pos = end;
        Some((inputs, targets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_shift_and_advance() {
        // tokens = 0, 1, 2, ... so values equal positions.
        let tokens: Vec<u16> = (0..100).collect();
        let mut it = Batches::<2, 8>::new(&tokens);
        assert_eq!(it.remaining(), 6); // floor(99 / 16)

        let (x0, y0) = it.next().unwrap();
        assert_eq!(x0.as_slice()[0], 0);
        assert_eq!(y0.as_slice()[0], 1); // shifted by one
        assert_eq!(x0.at(1, 0), 8); // row-major: row 1 starts at position T
        assert_eq!(y0.at(1, 7), 16); // last target of row 1 looks ahead

        let (x1, _) = it.next().unwrap();
        assert_eq!(x1.as_slice()[0], 16); // advanced by B*T, not B*T+1

        assert_eq!(it.count(), 4); // 6 total, 2 consumed
    }

    #[test]
    fn too_short_stream_yields_nothing() {
        let tokens: Vec<u16> = (0..16).collect();
        // 2*8 = 16 inputs need a 17th lookahead token: not present.
        assert!(Batches::<2, 8>::new(&tokens).next().is_none());
    }
}
