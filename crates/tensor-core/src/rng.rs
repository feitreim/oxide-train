//! Deterministic RNG shared by CPU reference code and GPU test harnesses.
//!
//! CPU/GPU parity tests recompute expected results on the host, so the exact
//! same inputs must appear on both sides. This is the same splitmix64 +
//! top-24-bits scheme as cuda-learning's bench-util: every draw is exactly
//! representable in an `f32` mantissa, so host and device see identical bits.

/// Constant folded into the state before every mixing round.
pub const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// One splitmix64 mixing round over an already-advanced state.
///
/// Exposed on its own because bf16 stochastic rounding keys a draw directly
/// from a coordinate rather than stepping a stateful generator. The GPU
/// optimizer kernels re-implement this expression verbatim in device code:
/// cuda-oxide collects device functions per artifact, so they cannot call
/// across crates, and this is the definition they must match.
pub const fn splitmix64(state: u64) -> u64 {
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Host half of a deterministic optimizer noise stream: fold the step and a
/// parameter id into one seed so the device kernel only mixes in the element
/// index. Never seeded from runtime entropy — reruns and checkpoint resumes
/// must reproduce the same rounding.
pub const fn stream_seed(step: u64, parameter: u64) -> u64 {
    splitmix64(splitmix64(step ^ SPLITMIX64_GAMMA).wrapping_add(parameter))
}

/// Draw for one element of the stream identified by [`stream_seed`].
pub const fn stream_draw(seed: u64, element: u64) -> u32 {
    (splitmix64(seed.wrapping_add(element)) >> 32) as u32
}

/// splitmix64: tiny, seedable, high-quality-enough. Not for cryptography.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
        splitmix64(self.state)
    }

    /// Uniform `f32` in `[-1, 1)`, exactly representable (top 24 bits only).
    pub fn next_uniform(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32; // [0, 1)
        unit * 2.0 - 1.0
    }
}

/// `n` uniform-random `f32` samples in `[-1, 1)` from a deterministic PRNG.
pub fn uniform_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| rng.next_uniform()).collect()
}

#[cfg(test)]
mod tests {
    use super::{SplitMix64, splitmix64, stream_draw, stream_seed};

    #[test]
    fn the_stateless_mixer_is_the_generator_step() {
        let mut rng = SplitMix64::new(7);
        assert_eq!(rng.next_u64(), splitmix64(7u64.wrapping_add(super::SPLITMIX64_GAMMA)));
    }

    #[test]
    fn keyed_draws_depend_on_every_coordinate() {
        let base = stream_draw(stream_seed(3, 5), 11);
        assert_ne!(base, stream_draw(stream_seed(4, 5), 11));
        assert_ne!(base, stream_draw(stream_seed(3, 6), 11));
        assert_ne!(base, stream_draw(stream_seed(3, 5), 12));
        assert_eq!(base, stream_draw(stream_seed(3, 5), 11));
    }
}
