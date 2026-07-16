use nn::gradcheck::{assert_close, numeric_grad};
use nn::{CausalAttention, Module, Rope};
use tensor_core::Rank2;
use tensor_cpu::CpuTensor;

const EPS: f32 = 1e-3;
const ATOL: f32 = 4e-3;
const RTOL: f32 = 3e-2;

#[test]
fn rope_gradchecks_input_and_resets_position_for_each_sequence() {
    type TestRope = Rope<6, 3, 8, 2, 4>;
    let x = CpuTensor::<f32, Rank2<6, 8>>::uniform(20);
    let upstream = CpuTensor::<f32, Rank2<6, 8>>::uniform(21);
    let mut rope = TestRope::default();

    let (y, ctx) = rope.forward(x.clone());
    let dx = rope.backward(ctx, upstream.clone());
    let ndx = numeric_grad(
        &x,
        |probe| rope.forward(probe.clone()).0.dot(&upstream),
        EPS,
    );
    assert_close(&dx, &ndx, ATOL, RTOL);

    // Position zero is the identity, including at the next batch boundary.
    assert_eq!(&y.as_slice()[0..8], &x.as_slice()[0..8]);
    assert_eq!(&y.as_slice()[3 * 8..4 * 8], &x.as_slice()[3 * 8..4 * 8]);
}

#[test]
fn causal_attention_gradchecks_q_k_and_v() {
    type Attention = CausalAttention<6, 3, 8, 2, 4>;
    let q = CpuTensor::<f32, Rank2<6, 8>>::uniform(22).scale(0.5);
    let k = CpuTensor::<f32, Rank2<6, 8>>::uniform(23).scale(0.5);
    let v = CpuTensor::<f32, Rank2<6, 8>>::uniform(24).scale(0.5);
    let upstream = CpuTensor::<f32, Rank2<6, 8>>::uniform(25);
    let mut attention = Attention::default();

    let (_, ctx) = attention.forward((q.clone(), k.clone(), v.clone()));
    let (dq, dk, dv) = attention.backward(ctx, upstream.clone());

    let ndq = numeric_grad(
        &q,
        |probe| {
            attention
                .forward((probe.clone(), k.clone(), v.clone()))
                .0
                .dot(&upstream)
        },
        EPS,
    );
    assert_close(&dq, &ndq, ATOL, RTOL);

    let ndk = numeric_grad(
        &k,
        |probe| {
            attention
                .forward((q.clone(), probe.clone(), v.clone()))
                .0
                .dot(&upstream)
        },
        EPS,
    );
    assert_close(&dk, &ndk, ATOL, RTOL);

    let ndv = numeric_grad(
        &v,
        |probe| {
            attention
                .forward((q.clone(), k.clone(), probe.clone()))
                .0
                .dot(&upstream)
        },
        EPS,
    );
    assert_close(&dv, &ndv, ATOL, RTOL);
}

#[test]
fn causal_attention_cannot_see_future_tokens_or_other_sequences() {
    type Attention = CausalAttention<6, 3, 8, 2, 4>;
    let q = CpuTensor::<f32, Rank2<6, 8>>::uniform(26);
    let k = CpuTensor::<f32, Rank2<6, 8>>::uniform(27);
    let v = CpuTensor::<f32, Rank2<6, 8>>::uniform(28);
    let baseline = Attention::default()
        .forward((q.clone(), k.clone(), v.clone()))
        .0;

    let mut changed_k = k;
    let mut changed_v = v;
    // Change future rows in sequence one and every row in sequence two.
    for row in 1..6 {
        for col in 0..8 {
            changed_k.as_mut_slice()[row * 8 + col] += 100.0;
            changed_v.as_mut_slice()[row * 8 + col] -= 100.0;
        }
    }
    let changed = Attention::default().forward((q, changed_k, changed_v)).0;

    // Query row zero can only attend to key/value row zero.
    assert_eq!(&baseline.as_slice()[0..8], &changed.as_slice()[0..8]);
}
