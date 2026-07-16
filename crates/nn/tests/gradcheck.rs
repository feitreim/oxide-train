//! Finite-difference checks for the first modules.
//!
//! The loss used everywhere is `L(y) = <y, R>` for a fixed random `R`, so
//! `dL/dy = R` exactly, and — because `L` is linear in both `x` and `W` — the
//! central difference is exact up to f32 roundoff. Later, nonlinear modules
//! (softmax, rmsnorm, swiglu) get checked the same way with looser tolerances.

use nn::gradcheck::{assert_close, numeric_grad};
use nn::{Chain, Linear, Module, chain};
use tensor_core::Rank2;
use tensor_cpu::CpuTensor;

const ATOL: f32 = 1e-3;
const RTOL: f32 = 1e-2;
const EPS: f32 = 1e-2;

#[test]
fn linear_backward_matches_finite_differences() {
    type X = CpuTensor<f32, Rank2<4, 3>>;
    type R = CpuTensor<f32, Rank2<4, 2>>;

    let x = X::uniform(1);
    let r = R::uniform(2);
    let mut lin = Linear::<4, 3, 2>::uniform(3);

    // Analytic: forward, then backward with dy = R.
    let (_, ctx) = lin.forward(x.clone());
    let dx = lin.backward(ctx, r.clone());

    // Numeric dL/dx.
    let ndx = numeric_grad(&x, |x| lin.forward(x.clone()).0.dot(&r), EPS);
    assert_close(&dx, &ndx, ATOL, RTOL);

    // Numeric dL/dW: rebuild the layer around each perturbed W.
    let ndw = numeric_grad(
        &lin.w,
        |w| {
            Linear::<4, 3, 2>::new(w.clone())
                .forward(x.clone())
                .0
                .dot(&r)
        },
        EPS,
    );
    assert_close(&lin.dw, &ndw, ATOL, RTOL);
}

#[test]
fn backward_accumulates_across_calls() {
    let x = CpuTensor::<f32, Rank2<4, 3>>::uniform(1);
    let r = CpuTensor::<f32, Rank2<4, 2>>::uniform(2);
    let mut lin = Linear::<4, 3, 2>::uniform(3);

    let (_, ctx) = lin.forward(x.clone());
    lin.backward(ctx, r.clone());
    let once = lin.dw.clone();
    let (_, ctx) = lin.forward(x.clone());
    lin.backward(ctx, r.clone());
    assert_close(&lin.dw, &once.scale(2.0), 1e-6, 1e-6);

    lin.zero_grad();
    assert_eq!(lin.dw.sum(), 0.0);
}

#[test]
fn chained_linears_backprop_through_composition() {
    type X = CpuTensor<f32, Rank2<4, 3>>;
    type R = CpuTensor<f32, Rank2<4, 5>>;

    let x = X::uniform(1);
    let r = R::uniform(2);
    // [4,3] -> [4,6] -> [4,2] -> [4,5]: Chain is the typed chain rule.
    let mut net = chain!(
        Linear::<4, 3, 6>::uniform(3),
        Linear::<4, 6, 2>::uniform(4),
        Linear::<4, 2, 5>::uniform(5),
    );

    let (_, ctx) = net.forward(x.clone());
    let dx = net.backward(ctx, r.clone());

    let ndx = numeric_grad(&x, |x| net.forward(x.clone()).0.dot(&r), EPS);
    assert_close(&dx, &ndx, ATOL, RTOL);

    // Innermost weight gradient, through two downstream layers.
    let w0 = net.a.w.clone();
    let ndw0 = numeric_grad(
        &w0,
        |w| {
            let probe: Chain<_, _> = chain!(
                Linear::<4, 3, 6>::new(w.clone()),
                Linear::<4, 6, 2>::new(net.b.a.w.clone()),
                Linear::<4, 2, 5>::new(net.b.b.w.clone()),
            );
            probe.forward(x.clone()).0.dot(&r)
        },
        EPS,
    );
    assert_close(&net.a.dw, &ndw0, ATOL, RTOL);
}
