use nn::gradcheck::{assert_close, numeric_grad};
use nn::{Embedding, Module, RmsNorm, SoftmaxCrossEntropy, SoftmaxCrossEntropyInput, SwiGlu};
use tensor_core::{Rank1, Rank2};
use tensor_cpu::CpuTensor;

const EPS: f32 = 1e-3;
const ATOL: f32 = 3e-3;
const RTOL: f32 = 2e-2;

#[test]
fn rms_norm_gradchecks_input_and_weight() {
    let x = CpuTensor::<f32, Rank2<3, 5>>::uniform(1);
    let r = CpuTensor::<f32, Rank2<3, 5>>::uniform(2);
    let w = CpuTensor::<f32, Rank1<5>>::uniform(3).map(|v| v + 1.25);
    let mut norm = RmsNorm::<3, 5>::new(w, 1e-5);

    let (_, ctx) = norm.forward(x.clone());
    let dx = norm.backward(ctx, r.clone());
    let ndx = numeric_grad(&x, |probe| norm.forward(probe.clone()).0.dot(&r), EPS);
    assert_close(&dx, &ndx, ATOL, RTOL);

    let ndw = numeric_grad(
        &norm.w,
        |probe| {
            RmsNorm::<3, 5>::new(probe.clone(), 1e-5)
                .forward(x.clone())
                .0
                .dot(&r)
        },
        EPS,
    );
    assert_close(&norm.dw, &ndw, ATOL, RTOL);
}

#[test]
fn swiglu_gradchecks_both_inputs() {
    let gate = CpuTensor::<f32, Rank2<3, 4>>::uniform(4);
    let up = CpuTensor::<f32, Rank2<3, 4>>::uniform(5);
    let r = CpuTensor::<f32, Rank2<3, 4>>::uniform(6);
    let mut swiglu = SwiGlu::<3, 4>;

    let (_, ctx) = swiglu.forward((gate.clone(), up.clone()));
    let (dgate, dup) = swiglu.backward(ctx, r.clone());

    let ndgate = numeric_grad(
        &gate,
        |probe| swiglu.forward((probe.clone(), up.clone())).0.dot(&r),
        EPS,
    );
    assert_close(&dgate, &ndgate, ATOL, RTOL);

    let ndup = numeric_grad(
        &up,
        |probe| swiglu.forward((gate.clone(), probe.clone())).0.dot(&r),
        EPS,
    );
    assert_close(&dup, &ndup, ATOL, RTOL);
}

#[test]
fn embedding_gradchecks_weights_and_accumulates_repeated_tokens() {
    let tokens = [2, 1, 2, 4];
    let r = CpuTensor::<f32, Rank2<4, 3>>::uniform(7);
    let mut embedding = Embedding::<4, 5, 3>::uniform(8);

    let (_, ctx) = embedding.forward(tokens);
    embedding.backward(ctx, r.clone());

    let ndw = numeric_grad(
        &embedding.w,
        |probe| {
            Embedding::<4, 5, 3>::new(probe.clone())
                .forward(tokens)
                .0
                .dot(&r)
        },
        EPS,
    );
    assert_close(&embedding.dw, &ndw, ATOL, RTOL);

    for col in 0..3 {
        let expected = r.as_slice()[col] + r.as_slice()[2 * 3 + col];
        assert!((embedding.dw.as_slice()[2 * 3 + col] - expected).abs() < 1e-6);
    }
}

#[test]
fn fused_softmax_cross_entropy_is_stable_and_gradchecks_logits() {
    let logits = CpuTensor::<f32, Rank2<4, 5>>::uniform(9).scale(4.0);
    let targets = [0, 3, 1, 4];
    let mut loss = SoftmaxCrossEntropy::<4, 5>;

    let (value, ctx) = loss.forward(SoftmaxCrossEntropyInput {
        logits: logits.clone(),
        targets,
    });
    assert!(value.as_slice()[0].is_finite());

    let dinput = loss.backward(ctx, CpuTensor::from_slice(&[1.0]));
    let ndlogits = numeric_grad(
        &logits,
        |probe| {
            loss.forward(SoftmaxCrossEntropyInput {
                logits: probe.clone(),
                targets,
            })
            .0
            .as_slice()[0]
        },
        EPS,
    );
    assert_close(&dinput.logits, &ndlogits, ATOL, RTOL);

    for row in dinput.logits.as_slice().chunks_exact(5) {
        assert!(row.iter().sum::<f32>().abs() < 1e-6);
    }

    let huge = CpuTensor::<f32, Rank2<1, 3>>::from_slice(&[10_000.0, 9_999.0, -10_000.0]);
    let stable = SoftmaxCrossEntropy::<1, 3>
        .forward(SoftmaxCrossEntropyInput {
            logits: huge,
            targets: [0],
        })
        .0;
    assert!(stable.as_slice()[0].is_finite());
}
