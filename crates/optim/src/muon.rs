//! CPU reference implementation of Muon.
//!
//! Muon applies SGD momentum to a hidden matrix's gradient, approximately
//! orthogonalizes that update with a quintic Newton--Schulz iteration, and then
//! performs a decoupled weight-decay update. Embeddings, normalization gains,
//! and classifier heads are intentionally outside this module's matrix step;
//! [`crate::LlamaMuon`] routes those parameters to AdamW.

use tensor_core::{Rank2, Shape};
use tensor_cpu::CpuTensor;

/// Quintic iteration coefficients, shared with the GPU implementation.
pub const NEWTON_SCHULZ_A: f32 = 3.4445;
pub const NEWTON_SCHULZ_B: f32 = -4.7750;
pub const NEWTON_SCHULZ_C: f32 = 2.0315;
pub const NEWTON_SCHULZ_EPSILON: f32 = 1e-7;

/// Hyperparameters for Muon updates over hidden matrices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuonConfig {
    /// Learning rate in spectral-norm units.
    pub learning_rate: f32,
    /// EMA coefficient for the SGD momentum buffer.
    pub momentum: f32,
    /// AdamW-style decoupled weight decay.
    pub weight_decay: f32,
    /// Whether to use the gradient/momentum Nesterov interpolation.
    pub nesterov: bool,
    /// Number of quintic Newton--Schulz iterations.
    pub newton_schulz_steps: usize,
}

impl MuonConfig {
    pub fn validate(self) {
        assert!(
            self.learning_rate.is_finite() && self.learning_rate >= 0.0,
            "learning rate must be finite and non-negative"
        );
        assert!(
            self.momentum.is_finite() && (0.0..1.0).contains(&self.momentum),
            "momentum must be in [0, 1)"
        );
        assert!(
            self.weight_decay.is_finite() && self.weight_decay >= 0.0,
            "weight decay must be finite and non-negative"
        );
        assert!(
            self.newton_schulz_steps < 100,
            "Newton-Schulz steps must be less than 100"
        );
    }
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.02,
            momentum: 0.95,
            weight_decay: 0.0,
            nesterov: true,
            newton_schulz_steps: 5,
        }
    }
}

/// Muon's momentum buffer for one statically shaped parameter.
pub struct MuonMomentum<S: Shape> {
    pub momentum: CpuTensor<f32, S>,
}

impl<S: Shape> MuonMomentum<S> {
    pub fn zeros() -> Self {
        Self {
            momentum: CpuTensor::zeros(),
        }
    }
}

/// Approximate the polar factor (matrix zeroth power) with Newton--Schulz.
///
/// The smaller matrix dimension is used for the Gram matrix. Tall inputs are
/// therefore transposed for the iteration and transposed back before return.
/// The all-zero matrix remains all zero.
pub fn zeroth_power_via_newton_schulz<const ROWS: usize, const COLS: usize>(
    matrix: &CpuTensor<f32, Rank2<ROWS, COLS>>,
    steps: usize,
) -> CpuTensor<f32, Rank2<ROWS, COLS>> {
    assert!(steps < 100, "Newton-Schulz steps must be less than 100");
    assert!(ROWS > 0 && COLS > 0, "Muon matrices must be non-empty");

    if ROWS <= COLS {
        newton_schulz_wide(matrix, steps)
    } else {
        newton_schulz_wide(&matrix.transpose(), steps).transpose()
    }
}

/// Newton--Schulz for a matrix whose row count is no larger than its column
/// count. Keeping the Gram matrix on the smaller axis is important for the GPU
/// implementation this reference will validate.
fn newton_schulz_wide<const ROWS: usize, const COLS: usize>(
    matrix: &CpuTensor<f32, Rank2<ROWS, COLS>>,
    steps: usize,
) -> CpuTensor<f32, Rank2<ROWS, COLS>> {
    debug_assert!(ROWS <= COLS);

    let frobenius_norm = matrix
        .as_slice()
        .iter()
        .map(|&value| {
            let value = value as f64;
            value * value
        })
        .sum::<f64>()
        .sqrt() as f32;
    let mut x = matrix.scale(1.0 / (frobenius_norm + NEWTON_SCHULZ_EPSILON));

    for _ in 0..steps {
        // A = X X^T
        let gram = x.matmul_nt(&x);
        // B = b A + c A^2
        let gram_squared = gram.matmul(&gram);
        let polynomial = gram
            .scale(NEWTON_SCHULZ_B)
            .add(&gram_squared.scale(NEWTON_SCHULZ_C));
        // X = a X + B X
        x = x.scale(NEWTON_SCHULZ_A).add(&polynomial.matmul(&x));
    }

    x
}

/// Apply one reference Muon update to a hidden 2D parameter.
///
/// Momentum follows the canonical EMA form:
///
/// `m = momentum * m + (1 - momentum) * gradient`
///
/// With Nesterov enabled, the matrix passed to Newton--Schulz is
/// `(1 - momentum) * gradient + momentum * m`. The parameter update is:
///
/// `p = (1 - lr * weight_decay) * p - adjusted_lr * orthogonalized_update`
///
/// where `adjusted_lr = lr * sqrt(max(1, ROWS / COLS))`.
pub fn muon_step<const ROWS: usize, const COLS: usize>(
    parameter: &mut CpuTensor<f32, Rank2<ROWS, COLS>>,
    gradient: &CpuTensor<f32, Rank2<ROWS, COLS>>,
    state: &mut MuonMomentum<Rank2<ROWS, COLS>>,
    config: MuonConfig,
) {
    config.validate();

    for (momentum, &gradient) in state
        .momentum
        .as_mut_slice()
        .iter_mut()
        .zip(gradient.as_slice())
    {
        *momentum = config.momentum * *momentum + (1.0 - config.momentum) * gradient;
    }

    let update = if config.nesterov {
        gradient
            .scale(1.0 - config.momentum)
            .add(&state.momentum.scale(config.momentum))
    } else {
        state.momentum.clone()
    };
    let update = zeroth_power_via_newton_schulz(&update, config.newton_schulz_steps);
    let aspect_ratio_scale = ((ROWS as f32 / COLS as f32).max(1.0)).sqrt();
    let decay = 1.0 - config.learning_rate * config.weight_decay;
    let update_scale = config.learning_rate * aspect_ratio_scale;

    for (parameter, &update) in parameter.as_mut_slice().iter_mut().zip(update.as_slice()) {
        *parameter = decay * *parameter - update_scale * update;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_cosine<const ROWS: usize, const COLS: usize>(
        matrix: &CpuTensor<f32, Rank2<ROWS, COLS>>,
        lhs: usize,
        rhs: usize,
    ) -> f32 {
        let values = matrix.as_slice();
        let lhs = &values[lhs * COLS..(lhs + 1) * COLS];
        let rhs = &values[rhs * COLS..(rhs + 1) * COLS];
        let dot = lhs
            .iter()
            .zip(rhs)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum::<f64>();
        let lhs_norm = lhs
            .iter()
            .map(|&value| (value as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let rhs_norm = rhs
            .iter()
            .map(|&value| (value as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        (dot / (lhs_norm * rhs_norm)) as f32
    }

    #[test]
    fn newton_schulz_reduces_row_correlation() {
        let matrix = CpuTensor::<f32, Rank2<2, 3>>::from_slice(&[
            2.0, 1.0, 0.0, //
            1.0, 2.0, 1.0,
        ]);

        let orthogonalized = zeroth_power_via_newton_schulz(&matrix, 5);

        // The high-slope quintic intentionally does not converge all singular
        // values exactly to one, but it should make these rows substantially
        // less correlated.
        assert!(row_cosine(&orthogonalized, 0, 1).abs() < row_cosine(&matrix, 0, 1).abs() * 0.5);
        assert!(
            orthogonalized
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn one_step_matches_the_quintic_polynomial() {
        let matrix = CpuTensor::<f32, Rank2<2, 2>>::from_slice(&[2.0, 0.0, 0.0, 1.0]);
        let actual = zeroth_power_via_newton_schulz(&matrix, 1);
        let norm = 5.0f32.sqrt() + NEWTON_SCHULZ_EPSILON;
        let polynomial = |value: f32| {
            let x = value / norm;
            NEWTON_SCHULZ_A * x + NEWTON_SCHULZ_B * x.powi(3) + NEWTON_SCHULZ_C * x.powi(5)
        };
        let expected = [polynomial(2.0), 0.0, 0.0, polynomial(1.0)];

        for (&actual, &expected) in actual.as_slice().iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn tall_and_transposed_wide_iterations_agree() {
        let tall = CpuTensor::<f32, Rank2<3, 2>>::from_slice(&[
            1.0, 2.0, //
            -3.0, 0.5, //
            2.5, -1.0,
        ]);

        let actual = zeroth_power_via_newton_schulz(&tall, 5);
        let expected = zeroth_power_via_newton_schulz(&tall.transpose(), 5).transpose();

        for (&actual, &expected) in actual.as_slice().iter().zip(expected.as_slice()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_update_stays_zero() {
        let zero = CpuTensor::<f32, Rank2<2, 3>>::zeros();
        assert_eq!(
            zeroth_power_via_newton_schulz(&zero, 5).as_slice(),
            zero.as_slice()
        );
    }

    #[test]
    fn muon_updates_momentum_and_applies_decoupled_decay() {
        let mut parameter = CpuTensor::<f32, Rank2<2, 2>>::from_slice(&[1.0, 0.0, 0.0, 1.0]);
        let gradient = CpuTensor::<f32, Rank2<2, 2>>::from_slice(&[2.0, 0.0, 0.0, 1.0]);
        let mut state = MuonMomentum::zeros();
        let config = MuonConfig {
            learning_rate: 0.1,
            momentum: 0.5,
            weight_decay: 0.2,
            nesterov: false,
            newton_schulz_steps: 5,
        };

        muon_step(&mut parameter, &gradient, &mut state, config);

        assert_eq!(state.momentum.as_slice(), &[1.0, 0.0, 0.0, 0.5]);
        let orthogonalized = zeroth_power_via_newton_schulz(&state.momentum, 5);
        for ((&actual, &initial), &update) in parameter
            .as_slice()
            .iter()
            .zip([1.0, 0.0, 0.0, 1.0].iter())
            .zip(orthogonalized.as_slice())
        {
            let expected = 0.98 * initial - 0.1 * update;
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
