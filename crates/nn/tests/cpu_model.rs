use nn::Llama;

#[test]
fn llama_cpu_overfits_a_tiny_batch() {
    type TinyLlama = Llama<4, 4, 4, 8, 2, 4, 12>;
    let tokens = [0, 1, 2, 3];
    let targets = [1, 2, 3, 0];
    let mut model = TinyLlama::new(100);

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
