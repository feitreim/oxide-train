//! Register vectors/tiles over the tcgen05 16x256b fragment map, plus the
//! pure-PTX-safe scalar maps they compose with.
//!
//! The fragment ownership contract every type here assumes (base-LDTM
//! 16x256b, the only drain shape the validated kernels use): within each
//! 16-row block of its warp's 32 TMEM rows, a thread owns rows `lane/4` and
//! `lane/4 + 8`, and columns `2*(lane%4)` and `+1` of each 8-column half. A
//! thread therefore holds 4 rows of a 64-row warpgroup tile — *slots*
//! `2*row_block + {0,1}` — and per row, 4 values per 16-column block at
//! offsets `{0, 1, 8, 9}`. Row statistics live once per owned row,
//! replicated across the 4 lanes of a quad by shuffle reductions.
//!
//! Scalar-map discipline: `f32::max/min/exp/ln/sqrt/floor` lower to libdevice
//! and would silently force an artifact off the pure-PTX path, so everything
//! here is comparison+select or explicit bit math. `exp2` exists twice —
//! [`exp2_approx`] (the FMA polynomial, bit-identical to what the flash
//! kernels shipped with) and [`exp2_hw`] (one `ex2.approx` SFU instruction,
//! also pure-PTX-safe post-#56). Ports that must hold "same SASS" keep the
//! polynomial; swapping to the SFU is a separate, measured change.

use cuda_device::warp;

/// NaN-free float max: comparison + select stays native PTX where
/// `f32::max` lowers to libdevice `__nv_fmaxf`. Callers guarantee finite
/// inputs (the kernels' masked sentinel is finite).
#[inline(always)]
pub fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// NaN-free float min; see [`fmax`].
#[inline(always)]
pub fn fmin(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `2^x` on FMA units: round-to-nearest split via the 1.5·2²³ shift trick,
/// exponent-bit insertion for the integer part, and a degree-3 minimax
/// polynomial (max relative error 7.5e-5 on the reduced range) for the
/// fraction. The clamp keeps the exponent field in the normal range and
/// flushes masked-sentinel inputs to a harmless ~2^-125.
#[inline(always)]
pub fn exp2_approx(x: f32) -> f32 {
    const SHIFT: f32 = 12582912.0; // 1.5 * 2^23
    const C0: f32 = 0.999_928_07;
    const C1: f32 = 0.693_260_99;
    const C2: f32 = 0.242_611_12;
    const C3: f32 = 0.055_171_67;
    let x = fmin(fmax(x, -125.0), 125.0);
    let shifted = x + SHIFT;
    let integer = (shifted.to_bits() as i32).wrapping_sub(0x4b40_0000);
    let fraction = x - (shifted - SHIFT);
    let poly = C0 + fraction * (C1 + fraction * (C2 + fraction * C3));
    f32::from_bits((poly.to_bits() as i32).wrapping_add(integer << 23) as u32)
}

/// `2^x` as one `ex2.approx.f32` SFU instruction — FA4's SFU offload.
/// Different rounding than [`exp2_approx`]; adopting it in a gated kernel is
/// a numerics change, not a refactor.
#[inline(always)]
pub fn exp2_hw(x: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(x)
}

/// `log2(x)` for positive normal `x`: exponent extraction, mantissa
/// renormalized to `[√½, √2]`, then the atanh series in `t = (m-1)/(m+1)`
/// (four terms; |error| < 5e-8 on the reduced range). The coefficient
/// literals are bit-exact copies of the validated kernel's.
#[allow(clippy::excessive_precision)]
#[inline(always)]
pub fn log2_approx(x: f32) -> f32 {
    const C0: f32 = 2.885_390_1;
    const C1: f32 = 0.961_796_7;
    const C2: f32 = 0.577_078_02;
    const C3: f32 = 0.412_198_58;
    let bits = x.to_bits();
    let mut exponent = ((bits >> 23) as i32) - 127;
    let mut mantissa = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if mantissa > core::f32::consts::SQRT_2 {
        mantissa *= 0.5;
        exponent += 1;
    }
    let t = (mantissa - 1.0) / (mantissa + 1.0);
    let t2 = t * t;
    exponent as f32 + t * (C0 + t2 * (C1 + t2 * (C2 + t2 * C3)))
}

/// Max across the 4 lanes of a quad — how a fragment row's statistic
/// becomes whole-row (each quad's lanes hold disjoint columns of one row).
#[inline(always)]
pub fn quad_max(value: f32) -> f32 {
    let value = fmax(value, warp::shuffle_xor_f32(value, 1));
    fmax(value, warp::shuffle_xor_f32(value, 2))
}

/// Sum across the 4 lanes of a quad; see [`quad_max`].
#[inline(always)]
pub fn quad_sum(value: f32) -> f32 {
    let value = value + warp::shuffle_xor_f32(value, 1);
    value + warp::shuffle_xor_f32(value, 2)
}

/// Per-thread row statistics of a fragment-mapped tile: one `f32` per owned
/// row (slot), replicated across each quad. `N` is 4 for the 64-row
/// warpgroup tiles (2 slots per 16-row block × 2 blocks per warp).
///
/// Every op is a compile-time-length loop over the slot array — plain
/// straight-line FMA/select code after inlining, nothing the register
/// allocator can see through less clearly than the hand-written form.
#[derive(Clone, Copy)]
pub struct RegVec<const N: usize>(pub [f32; N]);

impl<const N: usize> RegVec<N> {
    /// All slots set to `value` (e.g. a masked-score sentinel, or zero).
    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        Self([value; N])
    }

    /// Slotwise max with `other`.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < N {
            out.0[slot] = fmax(self.0[slot], other.0[slot]);
            slot += 1;
        }
        out
    }

    /// Slotwise `self - other`. A plain method rather than `ops::Sub` so
    /// every op the device code takes stays a direct `#[inline(always)]`
    /// call.
    #[allow(clippy::should_implement_trait)]
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < N {
            out.0[slot] = self.0[slot] - other.0[slot];
            slot += 1;
        }
        out
    }

    /// Slotwise software `2^x` ([`exp2_approx`]).
    #[inline(always)]
    pub fn exp2(self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < N {
            out.0[slot] = exp2_approx(self.0[slot]);
            slot += 1;
        }
        out
    }

    /// Slotwise `self *= other`.
    #[inline(always)]
    pub fn mul_assign(&mut self, other: Self) {
        let mut slot = 0;
        while slot < N {
            self.0[slot] *= other.0[slot];
            slot += 1;
        }
    }

    /// Slotwise `self += other`.
    #[inline(always)]
    pub fn add_assign(&mut self, other: Self) {
        let mut slot = 0;
        while slot < N {
            self.0[slot] += other.0[slot];
            slot += 1;
        }
    }

    /// True if any slot exceeds `reference + slack` — the correction-vote
    /// predicate (this lane's vote only; the warp/warpgroup OR is the
    /// caller's collective step).
    #[inline(always)]
    pub fn any_exceeds(self, reference: Self, slack: f32) -> bool {
        let mut exceed = false;
        let mut slot = 0;
        while slot < N {
            exceed = exceed || self.0[slot] > reference.0[slot] + slack;
            slot += 1;
        }
        exceed
    }

    /// Quad-reduce each slot's lane-local max into a whole-row max.
    #[inline(always)]
    pub fn quad_max(self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < N {
            out.0[slot] = quad_max(self.0[slot]);
            slot += 1;
        }
        out
    }

    /// Quad-reduce each slot's lane-local partial sum into a whole-row sum.
    #[inline(always)]
    pub fn quad_sum(self) -> Self {
        let mut out = self;
        let mut slot = 0;
        while slot < N {
            out.0[slot] = quad_sum(self.0[slot]);
            slot += 1;
        }
        out
    }
}

/// Per-thread fragment of a fragment-mapped `[16*SLOTS-ish, 4*VALUES]` fp32
/// tile: `SLOTS` owned rows × `VALUES` owned values per row (value `v` sits
/// in 16-column block `v/4` at offset `{0,1,8,9}[v%4]` from the lane's
/// column pair). The flash output accumulator is `RegTile<4, 32>` — 4 rows ×
/// 128 columns per thread-slice.
#[derive(Clone, Copy)]
pub struct RegTile<const SLOTS: usize, const VALUES: usize>(pub [[f32; VALUES]; SLOTS]);

/// The `[2, 4]` piece of a fragment-mapped tile one [`crate::tmem::TmemTile`]
/// drain returns: the two rows a thread owns in a 16-row block, times the four
/// values it owns in a 16-column block. Every register pass over a TMEM
/// accumulator is a loop over these.
pub type Fragment = RegTile<2, 4>;

/// Column of value `v` relative to the lane's own column pair: values come in
/// fours at offsets `{0, 1, 8, 9}` of successive 16-column blocks. The
/// coordinate a masking or per-column-statistic pass needs, and the inverse of
/// the packing [`crate::ldst::store_fragment_bf16`] undoes.
#[inline(always)]
pub const fn value_column(value: usize) -> usize {
    (value / 4) * 16 + [0, 1, 8, 9][value % 4]
}

impl<const SLOTS: usize, const VALUES: usize> RegTile<SLOTS, VALUES> {
    /// The additive identity — a fresh accumulator.
    #[inline(always)]
    pub fn zero() -> Self {
        Self([[0.0; VALUES]; SLOTS])
    }

    /// Scale every value in row-slot `s` by `factors` slot `s` — the
    /// running-max rescale of an online-softmax accumulator.
    #[inline(always)]
    pub fn scale_rows(&mut self, factors: RegVec<SLOTS>) {
        let mut slot = 0;
        while slot < SLOTS {
            let mut value = 0;
            while value < VALUES {
                self.0[slot][value] *= factors.0[slot];
                value += 1;
            }
            slot += 1;
        }
    }
}

/// One correction step of the online softmax, in the exact per-slot order of
/// the hand-written kernels: advance `m_ref` to cover `row_max`, and rescale
/// `running_sum` and `out_acc` into the new reference. Fused on purpose —
/// one scalar `next`/`factor` live at a time, each row's values rescaled
/// before the next row's factor is formed. The unfused form (`max`/`sub`/
/// `exp2`/`scale_rows`) keeps two full vectors live across the accumulator
/// scaling and measurably costs registers in register-tight kernels
/// (persistent forward: 206 → 212 regs/thread on B200).
#[inline(always)]
pub fn online_rescale<const S: usize, const V: usize>(
    m_ref: &mut RegVec<S>,
    row_max: RegVec<S>,
    running_sum: &mut RegVec<S>,
    out_acc: &mut RegTile<S, V>,
) {
    let mut slot = 0;
    while slot < S {
        let next = fmax(m_ref.0[slot], row_max.0[slot]);
        let factor = exp2_approx(m_ref.0[slot] - next);
        m_ref.0[slot] = next;
        running_sum.0[slot] *= factor;
        let mut value = 0;
        while value < V {
            out_acc.0[slot][value] *= factor;
            value += 1;
        }
        slot += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp2_polynomial_stays_inside_its_error_bound() {
        // The 7.5e-5 relative-error claim, plus the clamp semantics the
        // masked-score sentinel depends on.
        let mut x = -125.0f32;
        while x <= 125.0 {
            let approx = exp2_approx(x) as f64;
            let exact = (x as f64).exp2();
            assert!(
                ((approx - exact) / exact).abs() < 1.0e-4,
                "exp2_approx({x}) = {approx}, expected {exact}"
            );
            x += 0.137;
        }
        assert!(exp2_approx(-1.0e30) <= 2.0f32.powi(-124));
    }

    #[test]
    fn log2_series_stays_inside_its_error_bound() {
        let mut x = 1.0e-3f32;
        while x < 1.0e6 {
            let approx = log2_approx(x) as f64;
            let exact = (x as f64).log2();
            assert!(
                (approx - exact).abs() < 1.0e-6 + 1.0e-7 * exact.abs(),
                "log2_approx({x}) = {approx}, expected {exact}"
            );
            x *= 1.7;
        }
    }

    #[test]
    fn regvec_ops_match_the_hand_written_recurrence() {
        // The correction rescale from softmax_tile, replayed on both forms.
        let m_ref = RegVec([-3.0f32, 0.5, 2.0, -1.0e30]);
        let row_max = RegVec([1.0f32, 0.25, 9.0, -2.0]);
        let next = m_ref.max(row_max);
        let factor = m_ref.sub(next).exp2();
        for slot in 0..4 {
            let expected_next = if m_ref.0[slot] > row_max.0[slot] {
                m_ref.0[slot]
            } else {
                row_max.0[slot]
            };
            assert_eq!(next.0[slot], expected_next);
            assert_eq!(factor.0[slot], exp2_approx(m_ref.0[slot] - expected_next));
        }
        assert!(row_max.any_exceeds(m_ref, 8.0));
        assert!(!RegVec::<4>::splat(0.0).any_exceeds(RegVec::splat(0.0), 8.0));

        let mut tile = RegTile::<4, 8>::zero();
        tile.0[2][5] = 4.0;
        tile.scale_rows(factor);
        assert_eq!(tile.0[2][5], 4.0 * factor.0[2]);
    }

    #[test]
    fn value_columns_are_the_fragment_maps_offsets() {
        // The {0,1,8,9}-per-16-block pattern every drain loop spells by hand.
        assert_eq!(
            (0..8).map(value_column).collect::<Vec<_>>(),
            [0, 1, 8, 9, 16, 17, 24, 25]
        );
        // A RegTile<4, 32>'s 32 values cover 128 columns exactly once.
        let mut columns = (0..32).map(value_column).collect::<Vec<_>>();
        columns.sort();
        columns.dedup();
        assert_eq!(columns.len(), 32);
        assert_eq!(columns[31], 121);
    }

    #[test]
    fn online_rescale_matches_the_unfused_ops() {
        let mut m_ref = RegVec([-3.0f32, 0.5, 2.0, -1.0e30]);
        let row_max = RegVec([1.0f32, 0.25, 9.0, -2.0]);
        let mut running_sum = RegVec([1.0f32, 2.0, 3.0, 0.0]);
        let mut tile = RegTile::<4, 8>::zero();
        for slot in 0..4 {
            tile.0[slot][3] = 1.0 + slot as f32;
        }

        let mut m_unfused = m_ref;
        let mut sum_unfused = running_sum;
        let mut tile_unfused = tile;
        let next = m_unfused.max(row_max);
        let factor = m_unfused.sub(next).exp2();
        m_unfused = next;
        sum_unfused.mul_assign(factor);
        tile_unfused.scale_rows(factor);

        online_rescale(&mut m_ref, row_max, &mut running_sum, &mut tile);
        for slot in 0..4 {
            assert_eq!(m_ref.0[slot], m_unfused.0[slot]);
            assert_eq!(running_sum.0[slot], sum_unfused.0[slot]);
            assert_eq!(tile.0[slot][3], tile_unfused.0[slot][3]);
        }
    }
}
