//! Finite-difference gradient checking.
//!
//! The correctness backbone of the whole trainer: every hand-written
//! `Module::backward` must agree with a central-difference estimate of the
//! same loss. Runs on the CPU reference tensors only.

use tensor_core::Shape;
use tensor_cpu::CpuTensor;

/// Central-difference gradient of a scalar function `f` at `x`:
/// `g[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)`.
///
/// O(len(x)) evaluations of `f` — strictly for tests on small shapes.
pub fn numeric_grad<S: Shape>(
    x: &CpuTensor<f32, S>,
    mut f: impl FnMut(&CpuTensor<f32, S>) -> f32,
    eps: f32,
) -> CpuTensor<f32, S> {
    let mut g = CpuTensor::<f32, S>::zeros();
    let mut probe = x.clone();
    for i in 0..S::NUM_ELEMENTS {
        let orig = probe.as_slice()[i];
        probe.as_mut_slice()[i] = orig + eps;
        let plus = f(&probe);
        probe.as_mut_slice()[i] = orig - eps;
        let minus = f(&probe);
        probe.as_mut_slice()[i] = orig;
        g.as_mut_slice()[i] = (plus - minus) / (2.0 * eps);
    }
    g
}

/// Assert two gradients agree elementwise within `atol + rtol * |analytic|`.
pub fn assert_close<S: Shape>(
    analytic: &CpuTensor<f32, S>,
    numeric: &CpuTensor<f32, S>,
    atol: f32,
    rtol: f32,
) {
    for (i, (&a, &n)) in analytic
        .as_slice()
        .iter()
        .zip(numeric.as_slice())
        .enumerate()
    {
        let tol = atol + rtol * a.abs();
        assert!(
            (a - n).abs() <= tol,
            "grad mismatch at flat index {i}: analytic={a}, numeric={n} (tol={tol})"
        );
    }
}
