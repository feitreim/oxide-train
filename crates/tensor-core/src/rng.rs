//! Deterministic RNG shared by CPU reference code and GPU test harnesses.
//!
//! CPU/GPU parity tests recompute expected results on the host, so the exact
//! same inputs must appear on both sides. This is the same splitmix64 +
//! top-24-bits scheme as cuda-learning's bench-util: every draw is exactly
//! representable in an `f32` mantissa, so host and device see identical bits.

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
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
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
