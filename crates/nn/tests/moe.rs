use nn::gradcheck::{assert_close, numeric_grad};
use nn::{ExpertFfn, Linear, Module, MoeFfn};
use tensor_core::Rank2;
use tensor_cpu::CpuTensor;

const EPS: f32 = 1e-3;
const ATOL: f32 = 8e-3;
const RTOL: f32 = 5e-2;

fn model<const K: usize, const C: usize>(aux_coefficient: f32) -> MoeFfn<4, 2, 3, 3, K, C> {
    let router = Linear::new(CpuTensor::from_slice(&[
        1.2, -0.8, 0.2, //
        0.1, 0.9, -1.0,
    ]));
    let experts = std::array::from_fn(|expert| ExpertFfn::initialized(20 + expert as u64 * 3));
    MoeFfn::new(router, experts, aux_coefficient)
}

fn mixed_input() -> CpuTensor<f32, Rank2<4, 2>> {
    CpuTensor::from_slice(&[
        1.0, 0.2, //
        0.8, 0.4, //
        -0.7, 1.0, //
        -1.0, 0.6,
    ])
}

fn underfull_input() -> CpuTensor<f32, Rank2<4, 2>> {
    CpuTensor::from_slice(&[
        1.0, 0.2, //
        0.9, 0.3, //
        1.1, 0.1, //
        0.8, 0.25,
    ])
}

fn objective<const K: usize, const C: usize>(
    model: &MoeFfn<4, 2, 3, 3, K, C>,
    x: &CpuTensor<f32, Rank2<4, 2>>,
    upstream: &CpuTensor<f32, Rank2<4, 2>>,
) -> f32 {
    let output = model.forward(x.clone()).0;
    output.dot(upstream) + model.aux_coefficient * model.last_aux_loss.get()
}

fn assert_router_margin<const K: usize, const C: usize>(
    model: &MoeFfn<4, 2, 3, 3, K, C>,
    x: &CpuTensor<f32, Rank2<4, 2>>,
) {
    let logits = x.matmul(&model.router.w);
    for token in 0..4 {
        let mut row = logits.as_slice()[token * 3..(token + 1) * 3].to_vec();
        row.sort_unstable_by(|a, b| b.total_cmp(a));
        let margin = row[K - 1] - row[K];
        assert!(
            margin > 20.0 * EPS,
            "token {token} top-k margin {margin} is too close to gradcheck epsilon"
        );
    }
}

fn gradcheck_configuration<const K: usize, const C: usize>(
    x: CpuTensor<f32, Rank2<4, 2>>,
    aux_coefficient: f32,
    expect_drop: bool,
    expect_underfull: bool,
) {
    let upstream = CpuTensor::<f32, Rank2<4, 2>>::uniform(91);
    let mut moe = model::<K, C>(aux_coefficient);
    assert_router_margin(&moe, &x);

    let (_, inspection_ctx) = moe.forward(x.clone());
    assert_eq!(
        inspection_ctx.routing.slots.iter().any(Option::is_none),
        expect_drop
    );
    assert_eq!(
        inspection_ctx.routing.accepted_counts.contains(&0),
        expect_underfull
    );

    moe.zero_grad();
    let (_, ctx) = moe.forward(x.clone());
    let dx = moe.backward(ctx, upstream.clone());
    let numeric_dx = numeric_grad(&x, |probe| objective(&moe, probe, &upstream), EPS);
    assert_close(&dx, &numeric_dx, ATOL, RTOL);

    let router_weight = moe.router.w.clone();
    let analytic_router = moe.router.dw.clone();
    let numeric_router = numeric_grad(
        &router_weight,
        |probe| {
            moe.router.w = probe.clone();
            objective(&moe, &x, &upstream)
        },
        EPS,
    );
    moe.router.w = router_weight;
    assert_close(&analytic_router, &numeric_router, ATOL, RTOL);

    for expert in 0..3 {
        let weight = moe.experts[expert].gate_proj.w.clone();
        let analytic = moe.experts[expert].gate_proj.dw.clone();
        let numeric = numeric_grad(
            &weight,
            |probe| {
                moe.experts[expert].gate_proj.w = probe.clone();
                objective(&moe, &x, &upstream)
            },
            EPS,
        );
        moe.experts[expert].gate_proj.w = weight;
        assert_close(&analytic, &numeric, ATOL, RTOL);

        let weight = moe.experts[expert].up_proj.w.clone();
        let analytic = moe.experts[expert].up_proj.dw.clone();
        let numeric = numeric_grad(
            &weight,
            |probe| {
                moe.experts[expert].up_proj.w = probe.clone();
                objective(&moe, &x, &upstream)
            },
            EPS,
        );
        moe.experts[expert].up_proj.w = weight;
        assert_close(&analytic, &numeric, ATOL, RTOL);

        let weight = moe.experts[expert].down_proj.w.clone();
        let analytic = moe.experts[expert].down_proj.dw.clone();
        let numeric = numeric_grad(
            &weight,
            |probe| {
                moe.experts[expert].down_proj.w = probe.clone();
                objective(&moe, &x, &upstream)
            },
            EPS,
        );
        moe.experts[expert].down_proj.w = weight;
        assert_close(&analytic, &numeric, ATOL, RTOL);
    }
}

#[test]
fn moe_gradchecks_every_path_without_drops() {
    gradcheck_configuration::<2, 4>(mixed_input(), 0.17, false, false);
}

#[test]
fn moe_gradchecks_every_path_with_forced_drops() {
    gradcheck_configuration::<2, 1>(mixed_input(), 0.17, true, false);
}

#[test]
fn moe_gradchecks_every_path_with_underfull_experts() {
    gradcheck_configuration::<1, 4>(underfull_input(), 0.17, false, true);
}

#[test]
fn router_gate_path_gradchecks_with_auxiliary_loss_disabled() {
    let x = mixed_input();
    let upstream = CpuTensor::<f32, Rank2<4, 2>>::uniform(92);
    let mut moe = model::<2, 4>(0.0);
    assert_router_margin(&moe, &x);

    let (_, ctx) = moe.forward(x.clone());
    moe.backward(ctx, upstream.clone());
    let weight = moe.router.w.clone();
    let analytic = moe.router.dw.clone();
    let numeric = numeric_grad(
        &weight,
        |probe| {
            moe.router.w = probe.clone();
            objective(&moe, &x, &upstream)
        },
        EPS,
    );
    moe.router.w = weight;
    assert_close(&analytic, &numeric, ATOL, RTOL);
}

#[test]
fn routing_is_deterministic_and_ties_choose_lower_expert_indices() {
    let router = Linear::new(CpuTensor::<f32, Rank2<2, 3>>::zeros());
    let experts = std::array::from_fn(|expert| ExpertFfn::initialized(50 + expert as u64 * 3));
    let moe = MoeFfn::<3, 2, 3, 3, 2, 2>::new(router, experts, 0.0);
    let x = CpuTensor::<f32, Rank2<3, 2>>::uniform(60);

    let (_, first) = moe.forward(x.clone());
    let (_, second) = moe.forward(x);
    assert_eq!(first.routing, second.routing);
    assert_eq!(first.routing.selected_experts.as_ref(), &[0, 1, 0, 1, 0, 1]);
    assert_eq!(
        first.routing.slots.as_ref(),
        &[Some(0), Some(0), Some(1), Some(1), None, None]
    );
    assert_eq!(first.routing.gate_weights.as_ref(), &[0.5; 6]);
}

#[test]
fn auxiliary_loss_is_one_at_uniform_routing() {
    let router = Linear::new(CpuTensor::<f32, Rank2<2, 2>>::zeros());
    let experts = std::array::from_fn(|expert| ExpertFfn::initialized(70 + expert as u64 * 3));
    let moe = MoeFfn::<4, 2, 3, 2, 2, 4>::new(router, experts, 0.1);
    moe.forward(mixed_input());
    assert!((moe.last_aux_loss.get() - 1.0).abs() < 1e-6);
}

#[test]
fn auxiliary_loss_flattens_a_strongly_imbalanced_router() {
    type BalancingMoe = MoeFfn<8, 2, 3, 2, 1, 8>;
    let inputs = CpuTensor::<f32, Rank2<8, 2>>::from_slice(&[
        1.0, -1.0, //
        1.0, -0.7, //
        1.0, -0.4, //
        1.0, -0.1, //
        1.0, 0.2, //
        1.0, 0.5, //
        1.0, 0.8, //
        1.0, 1.1,
    ]);
    let router_weight = CpuTensor::from_slice(&[
        1.5, -1.5, //
        0.2, -0.2,
    ]);
    let make_model = |coefficient| {
        BalancingMoe::new(
            Linear::new(router_weight.clone()),
            std::array::from_fn(|expert| ExpertFfn::initialized(100 + expert as u64 * 3)),
            coefficient,
        )
    };
    let mut without_aux = make_model(0.0);
    let mut with_aux = make_model(1.0);
    let zero_output_gradient = CpuTensor::<f32, Rank2<8, 2>>::zeros();

    for _ in 0..200 {
        for model in [&mut without_aux, &mut with_aux] {
            model.zero_grad();
            let (_, ctx) = model.forward(inputs.clone());
            model.backward(ctx, zero_output_gradient.clone());
            model.sgd_step(0.1);
        }
    }

    let (_, without_ctx) = without_aux.forward(inputs.clone());
    let (_, with_ctx) = with_aux.forward(inputs);
    let without = BalancingMoe::assignment_fractions(&without_ctx);
    let with = BalancingMoe::assignment_fractions(&with_ctx);
    let without_imbalance = (without[0] - without[1]).abs();
    let with_imbalance = (with[0] - with[1]).abs();
    assert_eq!(without, [1.0, 0.0]);
    assert!(
        with_imbalance + 0.25 <= without_imbalance,
        "auxiliary loss did not flatten routing: without={without:?}, with={with:?}"
    );
}
