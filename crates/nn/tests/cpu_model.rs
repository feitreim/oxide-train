use nn::{Dense, MoeDense};

#[test]
fn dense_cpu_overfits_a_tiny_batch() {
    type TinyDense = Dense<4, 4, 4, 8, 2, 4, 12>;
    let tokens = [0, 1, 2, 3];
    let targets = [1, 2, 3, 0];
    let mut model = TinyDense::new(100);

    let initial_loss = model.forward(tokens, targets).0.as_slice()[0];
    let mut final_loss = initial_loss;
    for _ in 0..300 {
        model.zero_grad();
        let (loss, ctx) = model.forward(tokens, targets);
        final_loss = loss.as_slice()[0];
        model.backward(ctx);
        model.sgd_step(0.1);
    }

    assert!(
        final_loss < 0.05,
        "tiny batch did not overfit: initial loss={initial_loss}, final loss={final_loss}"
    );
    assert!(
        final_loss < initial_loss * 0.05,
        "loss did not fall enough: initial loss={initial_loss}, final loss={final_loss}"
    );
}

#[test]
fn moe_dense_cpu_overfits_a_tiny_batch_with_auxiliary_loss() {
    type TinyMoeDense = MoeDense<4, 4, 4, 8, 2, 4, 12, 2, 1, 4>;
    let tokens = [0, 1, 2, 3];
    let targets = [1, 2, 3, 0];
    let mut model = TinyMoeDense::new(100, 0.01);

    let initial_loss = model.forward(tokens, targets).0.as_slice()[0];
    let mut final_loss = initial_loss;
    for _ in 0..400 {
        model.zero_grad();
        let (loss, ctx) = model.forward(tokens, targets);
        final_loss = loss.as_slice()[0];
        model.backward(ctx);
        model.sgd_step(0.1);
    }

    assert!(
        final_loss < 0.08,
        "tiny MoE batch did not overfit: initial loss={initial_loss}, final loss={final_loss}"
    );
    assert!(
        final_loss < initial_loss * 0.08,
        "MoE loss did not fall enough: initial loss={initial_loss}, final loss={final_loss}"
    );
    assert!(model.aux_loss().is_finite());
}

#[test]
fn deep_moe_dense_cpu_overfits_a_tiny_batch() {
    type TinyDeepMoeDense = MoeDense<4, 4, 4, 8, 2, 4, 12, 2, 1, 4, 2>;
    let tokens = [0, 1, 2, 3];
    let targets = [1, 2, 3, 0];
    let mut model = TinyDeepMoeDense::new(100, 0.01);

    let initial_loss = model.forward(tokens, targets).0.as_slice()[0];
    let mut final_loss = initial_loss;
    for _ in 0..400 {
        model.zero_grad();
        let (loss, ctx) = model.forward(tokens, targets);
        final_loss = loss.as_slice()[0];
        model.backward(ctx);
        model.sgd_step(0.1);
    }

    assert!(
        final_loss < 0.08,
        "tiny deep MoE batch did not overfit: initial loss={initial_loss}, final loss={final_loss}"
    );
    assert!(
        final_loss < initial_loss * 0.08,
        "deep MoE loss did not fall enough: initial loss={initial_loss}, final loss={final_loss}"
    );
    assert!(model.aux_loss().is_finite());
}
