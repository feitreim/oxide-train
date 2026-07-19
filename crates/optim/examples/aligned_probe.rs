//! Realization-sampling CPU probe for the *aligned* tcgen05 overfit gates.
//!
//! gpu/model's aligned gates (`aligned_tcgen05_linears`, `aligned_moe_overfit`)
//! train tiny models whose block linears run the bf16 tcgen05 path. Their
//! trajectories are violently sensitive to sub-ulp rounding differences (the
//! 7e7 lesson): toolchain changes (FMA contraction, llc codegen) and GPU
//! atomics ordering move which side of a bf16 tie each rounding lands on, and
//! a realization either converges to ~1e-5 or parks on a two-logit-tie
//! plateau above the gate bar.
//!
//! This probe mirrors the aligned pipelines on CPU — quantization points
//! match gpu/model exactly (GEMM operands and epilogues bf16-rounded, fp32
//! accumulation, fp32 masters fed bf16-valued gradients; the MoE router stays
//! fp32 per SPEC decision 22) — and injects seeded ±1-ulp-scale relative
//! noise at every point where the GPU's summation order is not contractual:
//! GEMM results before their bf16 rounding, attention outputs and input
//! gradients, norm weight gradients, embedding gradients, and router logits.
//! Each noise seed is one "realization"; sweeping seeds samples the family of
//! trajectories the same source can produce across toolchains and runs.
//!
//! Use it to pick gate hyperparameters that make *every* sampled realization
//! converge with margin, before spending B200 time:
//!
//! ```text
//! cargo run --release -p optim --example aligned_probe -- dense \
//!     --lrs 0.01,0.02,0.03 --steps 600 --realizations 16 --noise 1.0
//! cargo run --release -p optim --example aligned_probe -- moe \
//!     --lrs 0.02 --steps 1200 --realizations 16 --noise 1.0
//! ```
//!
//! `--noise` is in units of 2^-23 (one f32 ulp of relative error);
//! `--noise 0` gives the noiseless trajectory.

use nn::{
    CausalAttention, Dense, Module, MoeDense, Rope, SoftmaxCrossEntropy, SoftmaxCrossEntropyInput,
    SwiGlu,
};
use optim::{AdamWConfig, AdamWMoments, AuxLossSchedule, DenseAdamW, adamw_step};
use tensor_core::rng::SplitMix64;
use tensor_core::{Rank1, Rank2, Shape};
use tensor_cpu::CpuTensor;

// ============================================================================
// Gate configurations (mirror gpu/model/src/main.rs exactly)
// ============================================================================

// aligned_tcgen05_linears
const NA: usize = 128;
const TA: usize = 4;
const VA: usize = 17;
const DA: usize = 128;
const HA: usize = 2;
const HD: usize = 64;
const FFA: usize = 128;
const DENSE_SEED: u64 = 7;

// aligned_moe_overfit
const ON: usize = 128;
const OT: usize = 4;
const OV: usize = 17;
const OD: usize = 128;
const OH: usize = 2;
const OFF: usize = 128;
const OE: usize = 2;
const OK: usize = 1;
const OC: usize = 128;
const MOE_SEED: u64 = 97;
const MOE_AUX_BASE: f32 = 0.01;
const MOE_AUX_HORIZON: f32 = 1_200.0;

const GATE_BAR: f32 = 0.05;

// ============================================================================
// Quantization + noise
// ============================================================================

fn q<S: Shape>(tensor: &CpuTensor<f32, S>) -> CpuTensor<f32, S> {
    tensor.to_bf16().to_f32()
}

/// Seeded ±ulp-scale relative perturbation. Zeros stay exactly zero, so the
/// padding-inertness the aligned path relies on is preserved.
struct Noise {
    rng: SplitMix64,
    scale: f32,
}

impl Noise {
    fn new(realization: u64, ulps: f32) -> Self {
        Self {
            rng: SplitMix64::new(0x51_7e_ed ^ realization.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            scale: ulps * (-23f32).exp2(),
        }
    }

    fn perturb<S: Shape>(&mut self, tensor: &mut CpuTensor<f32, S>) {
        if self.scale == 0.0 {
            return;
        }
        for value in tensor.as_mut_slice() {
            let unit = 2.0 * self.rng.next_uniform() - 1.0;
            *value += *value * unit * self.scale;
        }
    }
}

/// Forward of one aligned block linear: `bf16round(bf16(x) · bf16(w))` with
/// fp32 accumulation, the accumulation-order freedom modelled as noise before
/// the epilogue rounding.
fn qlin_forward<const R: usize, const IN: usize, const OUT: usize>(
    x: &CpuTensor<f32, Rank2<R, IN>>,
    w: &CpuTensor<f32, Rank2<IN, OUT>>,
    noise: &mut Noise,
) -> CpuTensor<f32, Rank2<R, OUT>> {
    let mut y = q(x).matmul(&q(w));
    noise.perturb(&mut y);
    q(&y)
}

/// Backward of one aligned block linear: `dw += bf16round(bf16(x)ᵀ·bf16(dy))`,
/// `dx = bf16round(bf16(dy)·bf16(w)ᵀ)`, mirroring the two tcgen05 launches.
fn qlin_backward<const R: usize, const IN: usize, const OUT: usize>(
    x: &CpuTensor<f32, Rank2<R, IN>>,
    dy: &CpuTensor<f32, Rank2<R, OUT>>,
    w: &CpuTensor<f32, Rank2<IN, OUT>>,
    dw: &mut CpuTensor<f32, Rank2<IN, OUT>>,
    noise: &mut Noise,
) -> CpuTensor<f32, Rank2<R, IN>> {
    let mut dw_value = q(x).matmul_tn(&q(dy));
    noise.perturb(&mut dw_value);
    dw.add_assign(&q(&dw_value));
    let mut dx = q(dy).matmul_nt(&q(w));
    noise.perturb(&mut dx);
    q(&dx)
}

// ============================================================================
// Trajectory record
// ============================================================================

struct Trace {
    initial: f32,
    final_loss: f32,
    first_pass: Option<usize>,
    bounces: usize,
    tail_max: f32,
}

impl Trace {
    fn passes(&self) -> bool {
        self.final_loss < GATE_BAR && self.final_loss < self.initial * GATE_BAR
    }
}

struct Recorder {
    steps: usize,
    initial: Option<f32>,
    first_pass: Option<usize>,
    below: bool,
    bounces: usize,
    tail_max: f32,
}

impl Recorder {
    fn new(steps: usize) -> Self {
        Self {
            steps,
            initial: None,
            first_pass: None,
            below: false,
            bounces: 0,
            tail_max: 0.0,
        }
    }

    fn record(&mut self, step: usize, loss: f32) {
        if self.initial.is_none() {
            self.initial = Some(loss);
        }
        if loss < GATE_BAR {
            if self.first_pass.is_none() {
                self.first_pass = Some(step);
            }
            self.below = true;
        } else if self.below {
            self.bounces += 1;
            self.below = false;
        }
        if step >= self.steps.saturating_sub(self.steps / 4) {
            self.tail_max = self.tail_max.max(loss);
        }
    }

    fn finish(self, final_loss: f32) -> Trace {
        Trace {
            initial: self.initial.expect("at least one step recorded"),
            final_loss,
            first_pass: self.first_pass,
            bounces: self.bounces,
            tail_max: self.tail_max,
        }
    }
}

// ============================================================================
// Dense gate (aligned_tcgen05_linears)
// ============================================================================

fn run_dense(lr: f32, steps: usize, noise_ulps: f32, realization: u64) -> Trace {
    let tokens: [usize; NA] = std::array::from_fn(|i| (i * 7 + 3) % VA);
    let targets: [usize; NA] = std::array::from_fn(|i| (tokens[i] + 1) % VA);
    let mut model = Dense::<NA, TA, VA, DA, HA, HD, FFA>::new(DENSE_SEED);
    let config = AdamWConfig {
        learning_rate: lr,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut optimizer = DenseAdamW::new(config);
    let mut noise = Noise::new(realization, noise_ulps);
    let mut recorder = Recorder::new(steps);

    let forward_loss =
        |model: &mut Dense<NA, TA, VA, DA, HA, HD, FFA>, noise: &mut Noise, train: bool| -> f32 {
            model.zero_grad();
            let (x, embedding_ctx) = model.embedding.forward(tokens);
            let attention_residual = x.clone();
            let (normalized, attention_norm_ctx) = model.attention_norm.forward(x);
            let q_out = qlin_forward(&normalized, &model.q_proj.w, noise);
            let k_out = qlin_forward(&normalized, &model.k_proj.w, noise);
            let v_out = qlin_forward(&normalized, &model.v_proj.w, noise);
            let (q_rot, ()) = Rope::<NA, TA, DA, HA, HD>.forward(q_out);
            let (k_rot, ()) = Rope::<NA, TA, DA, HA, HD>.forward(k_out);
            let (mut attended, attention_ctx) =
                CausalAttention::<NA, TA, DA, HA, HD>.forward((q_rot, k_rot, v_out.clone()));
            noise.perturb(&mut attended);
            let attention_output = qlin_forward(&attended, &model.o_proj.w, noise);
            let x = attention_residual.add(&attention_output);

            let ffn_residual = x.clone();
            let (ffn_normalized, ffn_norm_ctx) = model.ffn_norm.forward(x);
            let gate = qlin_forward(&ffn_normalized, &model.gate_proj.w, noise);
            let up = qlin_forward(&ffn_normalized, &model.up_proj.w, noise);
            let (activated, swiglu_ctx) = SwiGlu::<NA, FFA>.forward((gate, up));
            let ffn_output = qlin_forward(&activated, &model.down_proj.w, noise);
            let x = ffn_residual.add(&ffn_output);

            let (head_input, final_norm_ctx) = model.final_norm.forward(x);
            let logits = qlin_forward(&head_input, &model.lm_head.w, noise);
            let (loss, loss_ctx) =
                SoftmaxCrossEntropy::<NA, VA>.forward(SoftmaxCrossEntropyInput { logits, targets });
            let loss_value = loss.as_slice()[0];
            if !train {
                return loss_value;
            }

            let mut loss_module = SoftmaxCrossEntropy::<NA, VA>;
            let mut dlogits = loss_module
                .backward(loss_ctx, CpuTensor::from_slice(&[1.0]))
                .logits;
            noise.perturb(&mut dlogits);
            let dlogits = q(&dlogits);
            let dx = qlin_backward(
                &head_input,
                &dlogits,
                &model.lm_head.w.clone(),
                &mut model.lm_head.dw,
                noise,
            );
            let dx = model.final_norm.backward(final_norm_ctx, dx);

            let dactivated = qlin_backward(
                &activated,
                &dx,
                &model.down_proj.w.clone(),
                &mut model.down_proj.dw,
                noise,
            );
            let (dgate, dup) = SwiGlu::<NA, FFA>.backward(swiglu_ctx, dactivated);
            let dnormalized = qlin_backward(
                &ffn_normalized,
                &dgate,
                &model.gate_proj.w.clone(),
                &mut model.gate_proj.dw,
                noise,
            )
            .add(&qlin_backward(
                &ffn_normalized,
                &dup,
                &model.up_proj.w.clone(),
                &mut model.up_proj.dw,
                noise,
            ));
            let dffn_input = model.ffn_norm.backward(ffn_norm_ctx, dnormalized);
            let dx = dx.add(&dffn_input);

            let dattended = qlin_backward(
                &attended,
                &dx,
                &model.o_proj.w.clone(),
                &mut model.o_proj.dw,
                noise,
            );
            let (mut dq, mut dk, mut dv) =
                CausalAttention::<NA, TA, DA, HA, HD>.backward(attention_ctx, dattended);
            noise.perturb(&mut dq);
            noise.perturb(&mut dk);
            noise.perturb(&mut dv);
            let dq = Rope::<NA, TA, DA, HA, HD>.backward((), dq);
            let dk = Rope::<NA, TA, DA, HA, HD>.backward((), dk);
            let dnormalized = qlin_backward(
                &normalized,
                &dq,
                &model.q_proj.w.clone(),
                &mut model.q_proj.dw,
                noise,
            )
            .add(&qlin_backward(
                &normalized,
                &dk,
                &model.k_proj.w.clone(),
                &mut model.k_proj.dw,
                noise,
            ))
            .add(&qlin_backward(
                &normalized,
                &dv,
                &model.v_proj.w.clone(),
                &mut model.v_proj.dw,
                noise,
            ));
            let dattn_input = model
                .attention_norm
                .backward(attention_norm_ctx, dnormalized);
            let dx = dx.add(&dattn_input);
            model.embedding.backward(embedding_ctx, dx);

            // Atomic-accumulation noise on the fp32 norm/embedding gradients.
            noise.perturb(&mut model.attention_norm.dw);
            noise.perturb(&mut model.ffn_norm.dw);
            noise.perturb(&mut model.final_norm.dw);
            noise.perturb(&mut model.embedding.dw);
            loss_value
        };

    for step in 0..steps {
        let loss = forward_loss(&mut model, &mut noise, true);
        recorder.record(step, loss);
        optimizer.update(&mut model);
    }
    let final_loss = forward_loss(&mut model, &mut noise, false);
    recorder.finish(final_loss)
}

// ============================================================================
// MoE gate (aligned_moe_overfit)
// ============================================================================

type MoeModel = MoeDense<ON, OT, OV, OD, OH, HD, OFF, OE, OK, OC>;

struct MoeOptimizer {
    step: u64,
    config: AdamWConfig,
    embedding: AdamWMoments<Rank2<OV, OD>>,
    attention_norm: AdamWMoments<Rank1<OD>>,
    q_proj: AdamWMoments<Rank2<OD, OD>>,
    k_proj: AdamWMoments<Rank2<OD, OD>>,
    v_proj: AdamWMoments<Rank2<OD, OD>>,
    o_proj: AdamWMoments<Rank2<OD, OD>>,
    ffn_norm: AdamWMoments<Rank1<OD>>,
    router: AdamWMoments<Rank2<OD, OE>>,
    expert_gate: [AdamWMoments<Rank2<OD, OFF>>; OE],
    expert_up: [AdamWMoments<Rank2<OD, OFF>>; OE],
    expert_down: [AdamWMoments<Rank2<OFF, OD>>; OE],
    final_norm: AdamWMoments<Rank1<OD>>,
    lm_head: AdamWMoments<Rank2<OD, OV>>,
}

impl MoeOptimizer {
    fn new(config: AdamWConfig) -> Self {
        Self {
            step: 0,
            config,
            embedding: AdamWMoments::zeros(),
            attention_norm: AdamWMoments::zeros(),
            q_proj: AdamWMoments::zeros(),
            k_proj: AdamWMoments::zeros(),
            v_proj: AdamWMoments::zeros(),
            o_proj: AdamWMoments::zeros(),
            ffn_norm: AdamWMoments::zeros(),
            router: AdamWMoments::zeros(),
            expert_gate: std::array::from_fn(|_| AdamWMoments::zeros()),
            expert_up: std::array::from_fn(|_| AdamWMoments::zeros()),
            expert_down: std::array::from_fn(|_| AdamWMoments::zeros()),
            final_norm: AdamWMoments::zeros(),
            lm_head: AdamWMoments::zeros(),
        }
    }

    fn update(&mut self, model: &mut MoeModel) {
        self.step += 1;
        let step = self.step;
        let config = self.config;
        adamw_step(
            &mut model.embedding.w,
            &model.embedding.dw,
            &mut self.embedding,
            config,
            step,
        );
        adamw_step(
            &mut model.attention_norm.w,
            &model.attention_norm.dw,
            &mut self.attention_norm,
            config,
            step,
        );
        adamw_step(
            &mut model.q_proj.w,
            &model.q_proj.dw,
            &mut self.q_proj,
            config,
            step,
        );
        adamw_step(
            &mut model.k_proj.w,
            &model.k_proj.dw,
            &mut self.k_proj,
            config,
            step,
        );
        adamw_step(
            &mut model.v_proj.w,
            &model.v_proj.dw,
            &mut self.v_proj,
            config,
            step,
        );
        adamw_step(
            &mut model.o_proj.w,
            &model.o_proj.dw,
            &mut self.o_proj,
            config,
            step,
        );
        adamw_step(
            &mut model.ffn_norm.w,
            &model.ffn_norm.dw,
            &mut self.ffn_norm,
            config,
            step,
        );
        adamw_step(
            &mut model.ffn.router.w,
            &model.ffn.router.dw,
            &mut self.router,
            config,
            step,
        );
        for expert in 0..OE {
            adamw_step(
                &mut model.ffn.experts[expert].gate_proj.w,
                &model.ffn.experts[expert].gate_proj.dw,
                &mut self.expert_gate[expert],
                config,
                step,
            );
            adamw_step(
                &mut model.ffn.experts[expert].up_proj.w,
                &model.ffn.experts[expert].up_proj.dw,
                &mut self.expert_up[expert],
                config,
                step,
            );
            adamw_step(
                &mut model.ffn.experts[expert].down_proj.w,
                &model.ffn.experts[expert].down_proj.dw,
                &mut self.expert_down[expert],
                config,
                step,
            );
        }
        adamw_step(
            &mut model.final_norm.w,
            &model.final_norm.dw,
            &mut self.final_norm,
            config,
            step,
        );
        adamw_step(
            &mut model.lm_head.w,
            &model.lm_head.dw,
            &mut self.lm_head,
            config,
            step,
        );
    }
}

/// Deterministic top-k + capacity assignment, mirroring `MoeFfn::route`.
struct Routing {
    selected: Vec<usize>,
    gates: Vec<f32>,
    slots: Vec<Option<usize>>,
    assignment_counts: [usize; OE],
}

fn route(probabilities: &CpuTensor<f32, Rank2<ON, OE>>) -> Routing {
    let mut selected = vec![0usize; ON * OK];
    let mut gates = vec![0.0f32; ON * OK];
    let mut slots = vec![None; ON * OK];
    let mut assignment_counts = [0usize; OE];
    let mut accepted = [0usize; OE];
    for token in 0..ON {
        let mut order: Vec<usize> = (0..OE).collect();
        order.sort_unstable_by(|&a, &b| {
            probabilities.as_slice()[token * OE + b]
                .total_cmp(&probabilities.as_slice()[token * OE + a])
                .then_with(|| a.cmp(&b))
        });
        let denominator = order[..OK]
            .iter()
            .map(|&expert| probabilities.as_slice()[token * OE + expert] as f64)
            .sum::<f64>() as f32;
        for (rank, &expert) in order.iter().take(OK).enumerate() {
            let pair = token * OK + rank;
            selected[pair] = expert;
            gates[pair] = probabilities.as_slice()[token * OE + expert] / denominator;
            assignment_counts[expert] += 1;
            if accepted[expert] < OC {
                slots[pair] = Some(accepted[expert]);
                accepted[expert] += 1;
            }
        }
    }
    Routing {
        selected,
        gates,
        slots,
        assignment_counts,
    }
}

fn run_moe(lr: f32, steps: usize, noise_ulps: f32, realization: u64) -> Trace {
    let schedule = AuxLossSchedule {
        base_coefficient: MOE_AUX_BASE,
        decay_horizon: MOE_AUX_HORIZON,
    };
    let tokens: [usize; ON] = std::array::from_fn(|i| (i * 7 + 3) % OV);
    let targets: [usize; ON] = std::array::from_fn(|i| (tokens[i] + 1) % OV);
    let mut model = MoeModel::new(MOE_SEED, schedule.base_coefficient);
    let config = AdamWConfig {
        learning_rate: lr,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut optimizer = MoeOptimizer::new(config);
    let mut noise = Noise::new(realization, noise_ulps);
    let mut recorder = Recorder::new(steps);

    let forward_loss =
        |model: &mut MoeModel, noise: &mut Noise, coefficient: f32, train: bool| -> f32 {
            model.zero_grad();
            let (x, embedding_ctx) = model.embedding.forward(tokens);
            let attention_residual = x.clone();
            let (normalized, attention_norm_ctx) = model.attention_norm.forward(x);
            let q_out = qlin_forward(&normalized, &model.q_proj.w, noise);
            let k_out = qlin_forward(&normalized, &model.k_proj.w, noise);
            let v_out = qlin_forward(&normalized, &model.v_proj.w, noise);
            let (q_rot, ()) = Rope::<ON, OT, OD, OH, HD>.forward(q_out);
            let (k_rot, ()) = Rope::<ON, OT, OD, OH, HD>.forward(k_out);
            let (mut attended, attention_ctx) =
                CausalAttention::<ON, OT, OD, OH, HD>.forward((q_rot, k_rot, v_out.clone()));
            noise.perturb(&mut attended);
            let attention_output = qlin_forward(&attended, &model.o_proj.w, noise);
            let x = attention_residual.add(&attention_output);

            let ffn_residual = x.clone();
            let (ffn_normalized, ffn_norm_ctx) = model.ffn_norm.forward(x);

            // --- MoE FFN forward (router fp32, experts bf16) ---
            let mut router_logits = ffn_normalized.matmul(&model.ffn.router.w);
            noise.perturb(&mut router_logits);
            let probabilities = router_logits.softmax_rows();
            let routing = route(&probabilities);
            let mut expert_inputs: [CpuTensor<f32, Rank2<OC, OD>>; OE] =
                std::array::from_fn(|_| CpuTensor::zeros());
            for token in 0..ON {
                for rank in 0..OK {
                    let pair = token * OK + rank;
                    let Some(slot) = routing.slots[pair] else {
                        continue;
                    };
                    let expert = routing.selected[pair];
                    expert_inputs[expert].as_mut_slice()[slot * OD..(slot + 1) * OD]
                        .copy_from_slice(&ffn_normalized.as_slice()[token * OD..(token + 1) * OD]);
                }
            }
            let mut expert_activated: Vec<CpuTensor<f32, Rank2<OC, OFF>>> = Vec::with_capacity(OE);
            let mut expert_swiglu = Vec::with_capacity(OE);
            let mut expert_outputs: Vec<CpuTensor<f32, Rank2<OC, OD>>> = Vec::with_capacity(OE);
            for expert in 0..OE {
                let gate = qlin_forward(
                    &expert_inputs[expert],
                    &model.ffn.experts[expert].gate_proj.w,
                    noise,
                );
                let up = qlin_forward(
                    &expert_inputs[expert],
                    &model.ffn.experts[expert].up_proj.w,
                    noise,
                );
                let (activated, swiglu_ctx) = SwiGlu::<OC, OFF>.forward((gate, up));
                let output =
                    qlin_forward(&activated, &model.ffn.experts[expert].down_proj.w, noise);
                expert_activated.push(activated);
                expert_swiglu.push(swiglu_ctx);
                expert_outputs.push(output);
            }
            let mut ffn_output = CpuTensor::<f32, Rank2<ON, OD>>::zeros();
            for token in 0..ON {
                for rank in 0..OK {
                    let pair = token * OK + rank;
                    let Some(slot) = routing.slots[pair] else {
                        continue;
                    };
                    let expert = routing.selected[pair];
                    let gate = routing.gates[pair];
                    for dim in 0..OD {
                        ffn_output.as_mut_slice()[token * OD + dim] +=
                            gate * expert_outputs[expert].as_slice()[slot * OD + dim];
                    }
                }
            }
            let mean_probabilities: [f32; OE] = std::array::from_fn(|expert| {
                (0..ON)
                    .map(|token| probabilities.as_slice()[token * OE + expert] as f64)
                    .sum::<f64>() as f32
                    / ON as f32
            });
            let assignment_fractions: [f32; OE] = std::array::from_fn(|expert| {
                routing.assignment_counts[expert] as f32 / (ON * OK) as f32
            });
            let aux_loss = OE as f32
                * (0..OE)
                    .map(|expert| {
                        assignment_fractions[expert] as f64 * mean_probabilities[expert] as f64
                    })
                    .sum::<f64>() as f32;
            // --- end MoE FFN forward ---

            let x = ffn_residual.add(&ffn_output);
            let (head_input, final_norm_ctx) = model.final_norm.forward(x);
            let logits = qlin_forward(&head_input, &model.lm_head.w, noise);
            let (loss, loss_ctx) =
                SoftmaxCrossEntropy::<ON, OV>.forward(SoftmaxCrossEntropyInput { logits, targets });
            let loss_value = loss.as_slice()[0] + coefficient * aux_loss;
            if !train {
                return loss_value;
            }

            let mut loss_module = SoftmaxCrossEntropy::<ON, OV>;
            let mut dlogits = loss_module
                .backward(loss_ctx, CpuTensor::from_slice(&[1.0]))
                .logits;
            noise.perturb(&mut dlogits);
            let dlogits = q(&dlogits);
            let dx = qlin_backward(
                &head_input,
                &dlogits,
                &model.lm_head.w.clone(),
                &mut model.lm_head.dw,
                noise,
            );
            let dx = model.final_norm.backward(final_norm_ctx, dx);

            // --- MoE FFN backward ---
            let dy = &dx;
            let mut expert_output_gradients: [CpuTensor<f32, Rank2<OC, OD>>; OE] =
                std::array::from_fn(|_| CpuTensor::zeros());
            let mut gate_gradients = vec![0.0f32; ON * OK];
            for token in 0..ON {
                for rank in 0..OK {
                    let pair = token * OK + rank;
                    let Some(slot) = routing.slots[pair] else {
                        continue;
                    };
                    let expert = routing.selected[pair];
                    let gate = routing.gates[pair];
                    let mut gate_gradient = 0.0f64;
                    for dim in 0..OD {
                        let token_index = token * OD + dim;
                        let expert_index = slot * OD + dim;
                        expert_output_gradients[expert].as_mut_slice()[expert_index] +=
                            gate * dy.as_slice()[token_index];
                        gate_gradient += expert_outputs[expert].as_slice()[expert_index] as f64
                            * dy.as_slice()[token_index] as f64;
                    }
                    gate_gradients[pair] = gate_gradient as f32;
                }
            }
            let mut expert_input_gradients: Vec<CpuTensor<f32, Rank2<OC, OD>>> =
                Vec::with_capacity(OE);
            for expert in 0..OE {
                let dactivated = qlin_backward(
                    &expert_activated[expert],
                    &expert_output_gradients[expert],
                    &model.ffn.experts[expert].down_proj.w.clone(),
                    &mut model.ffn.experts[expert].down_proj.dw,
                    noise,
                );
                let (dgate, dup) = SwiGlu::<OC, OFF>.backward(expert_swiglu.remove(0), dactivated);
                let dinput = qlin_backward(
                    &expert_inputs[expert],
                    &dgate,
                    &model.ffn.experts[expert].gate_proj.w.clone(),
                    &mut model.ffn.experts[expert].gate_proj.dw,
                    noise,
                )
                .add(&qlin_backward(
                    &expert_inputs[expert],
                    &dup,
                    &model.ffn.experts[expert].up_proj.w.clone(),
                    &mut model.ffn.experts[expert].up_proj.dw,
                    noise,
                ));
                expert_input_gradients.push(dinput);
            }
            let mut dffn = CpuTensor::<f32, Rank2<ON, OD>>::zeros();
            for token in 0..ON {
                for rank in 0..OK {
                    let pair = token * OK + rank;
                    let Some(slot) = routing.slots[pair] else {
                        continue;
                    };
                    let expert = routing.selected[pair];
                    for dim in 0..OD {
                        dffn.as_mut_slice()[token * OD + dim] +=
                            expert_input_gradients[expert].as_slice()[slot * OD + dim];
                    }
                }
            }
            let mut probability_gradients = CpuTensor::<f32, Rank2<ON, OE>>::zeros();
            for token in 0..ON {
                let weighted_gate_gradient = (0..OK)
                    .map(|rank| {
                        let pair = token * OK + rank;
                        gate_gradients[pair] as f64 * routing.gates[pair] as f64
                    })
                    .sum::<f64>() as f32;
                let selected_probability_sum = (0..OK)
                    .map(|rank| {
                        let expert = routing.selected[token * OK + rank];
                        probabilities.as_slice()[token * OE + expert] as f64
                    })
                    .sum::<f64>() as f32;
                for rank in 0..OK {
                    let pair = token * OK + rank;
                    let expert = routing.selected[pair];
                    probability_gradients.as_mut_slice()[token * OE + expert] +=
                        (gate_gradients[pair] - weighted_gate_gradient) / selected_probability_sum;
                }
                for (expert, &fraction) in assignment_fractions.iter().enumerate() {
                    probability_gradients.as_mut_slice()[token * OE + expert] +=
                        coefficient * OE as f32 * fraction / ON as f32;
                }
            }
            let mut router_dlogits = CpuTensor::<f32, Rank2<ON, OE>>::zeros();
            for token in 0..ON {
                let softmax_dot = (0..OE)
                    .map(|expert| {
                        probabilities.as_slice()[token * OE + expert] as f64
                            * probability_gradients.as_slice()[token * OE + expert] as f64
                    })
                    .sum::<f64>() as f32;
                for expert in 0..OE {
                    let index = token * OE + expert;
                    router_dlogits.as_mut_slice()[index] = probabilities.as_slice()[index]
                        * (probability_gradients.as_slice()[index] - softmax_dot);
                }
            }
            // Router stays fp32 (SPEC decision 22); its GEMMs still carry
            // toolchain-order noise.
            let mut router_dw = ffn_normalized.matmul_tn(&router_dlogits);
            noise.perturb(&mut router_dw);
            model.ffn.router.dw.add_assign(&router_dw);
            let mut router_dx = router_dlogits.matmul_nt(&model.ffn.router.w);
            noise.perturb(&mut router_dx);
            dffn.add_assign(&router_dx);
            // --- end MoE FFN backward ---

            let dffn_input = model.ffn_norm.backward(ffn_norm_ctx, dffn);
            let dx = dx.add(&dffn_input);

            let dattended = qlin_backward(
                &attended,
                &dx,
                &model.o_proj.w.clone(),
                &mut model.o_proj.dw,
                noise,
            );
            let (mut dq, mut dk, mut dv) =
                CausalAttention::<ON, OT, OD, OH, HD>.backward(attention_ctx, dattended);
            noise.perturb(&mut dq);
            noise.perturb(&mut dk);
            noise.perturb(&mut dv);
            let dq = Rope::<ON, OT, OD, OH, HD>.backward((), dq);
            let dk = Rope::<ON, OT, OD, OH, HD>.backward((), dk);
            let dnormalized = qlin_backward(
                &normalized,
                &dq,
                &model.q_proj.w.clone(),
                &mut model.q_proj.dw,
                noise,
            )
            .add(&qlin_backward(
                &normalized,
                &dk,
                &model.k_proj.w.clone(),
                &mut model.k_proj.dw,
                noise,
            ))
            .add(&qlin_backward(
                &normalized,
                &dv,
                &model.v_proj.w.clone(),
                &mut model.v_proj.dw,
                noise,
            ));
            let dattn_input = model
                .attention_norm
                .backward(attention_norm_ctx, dnormalized);
            let dx = dx.add(&dattn_input);
            model.embedding.backward(embedding_ctx, dx);

            noise.perturb(&mut model.attention_norm.dw);
            noise.perturb(&mut model.ffn_norm.dw);
            noise.perturb(&mut model.final_norm.dw);
            noise.perturb(&mut model.embedding.dw);
            loss_value
        };

    for step in 0..steps {
        let coefficient = schedule.coefficient(step as u64);
        let loss = forward_loss(&mut model, &mut noise, coefficient, true);
        recorder.record(step, loss);
        optimizer.update(&mut model);
    }
    let final_coefficient = schedule.coefficient(steps as u64);
    let final_loss = forward_loss(&mut model, &mut noise, final_coefficient, false);
    recorder.finish(final_loss)
}

// ============================================================================
// Sweep driver
// ============================================================================

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gate = args.get(1).cloned().unwrap_or_else(|| "dense".to_string());
    let lrs: Vec<f32> = parse_flag(&args, "--lrs")
        .unwrap_or_else(|| "0.02".to_string())
        .split(',')
        .map(|s| s.parse().expect("bad --lrs"))
        .collect();
    let steps: usize = parse_flag(&args, "--steps")
        .unwrap_or_else(|| if gate == "moe" { "1200" } else { "600" }.to_string())
        .parse()
        .expect("bad --steps");
    let realizations: u64 = parse_flag(&args, "--realizations")
        .unwrap_or_else(|| "16".to_string())
        .parse()
        .expect("bad --realizations");
    let noise: f32 = parse_flag(&args, "--noise")
        .unwrap_or_else(|| "1.0".to_string())
        .parse()
        .expect("bad --noise");

    println!(
        "gate={gate} steps={steps} realizations={realizations} noise={noise} ulp \
         bar: final<{GATE_BAR} && final<{GATE_BAR}*initial"
    );
    for &lr in &lrs {
        let mut passes = 0u64;
        let mut worst: Option<(u64, f32)> = None;
        println!("--- lr {lr}");
        let traces: Vec<(u64, Trace)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..realizations)
                .map(|r| {
                    let gate = gate.clone();
                    scope.spawn(move || {
                        let trace = if gate == "moe" {
                            run_moe(lr, steps, noise, r)
                        } else {
                            run_dense(lr, steps, noise, r)
                        };
                        (r, trace)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (r, trace) in traces {
            let pass = trace.passes();
            passes += pass as u64;
            if worst.is_none() || trace.final_loss > worst.unwrap().1 {
                worst = Some((r, trace.final_loss));
            }
            println!(
                "  r{r:<3} initial={:.6} final={:<12.6} first<bar={:<5} bounces={} \
                 tail_max={:.6} {}",
                trace.initial,
                trace.final_loss,
                trace
                    .first_pass
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "never".to_string()),
                trace.bounces,
                trace.tail_max,
                if pass { "PASS" } else { "FAIL" },
            );
        }
        let (worst_r, worst_final) = worst.unwrap();
        println!(
            "  lr {lr}: {passes}/{realizations} pass; worst r{worst_r} final={worst_final:.6}"
        );
    }
}
