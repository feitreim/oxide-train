//! CPU reference mixture-of-experts SwiGLU feed-forward module.
//!
//! Routing is dynamic, but every buffer shape is fixed by `N`, `D`, `FF`,
//! expert count `E`, top-k `K`, and per-expert capacity `C`.  This is the
//! correctness oracle for later GPU routing and expert kernels.

use std::cell::Cell;

use tensor_core::Rank2;
use tensor_cpu::CpuTensor;

use crate::swiglu::SwiGluCtx;
use crate::{Linear, Module, SwiGlu};

const DROPPED: usize = usize::MAX;

/// One capacity-padded SwiGLU expert.
pub struct ExpertFfn<const C: usize, const D: usize, const FF: usize> {
    pub gate_proj: Linear<C, D, FF>,
    pub up_proj: Linear<C, D, FF>,
    pub down_proj: Linear<C, FF, D>,
}

/// Saved activations for one expert.
pub struct ExpertFfnCtx<const C: usize, const D: usize, const FF: usize> {
    gate_proj: CpuTensor<f32, Rank2<C, D>>,
    up_proj: CpuTensor<f32, Rank2<C, D>>,
    swiglu: SwiGluCtx<C, FF>,
    down_proj: CpuTensor<f32, Rank2<C, FF>>,
}

impl<const C: usize, const D: usize, const FF: usize> ExpertFfn<C, D, FF> {
    pub fn new(
        gate_proj: CpuTensor<f32, Rank2<D, FF>>,
        up_proj: CpuTensor<f32, Rank2<D, FF>>,
        down_proj: CpuTensor<f32, Rank2<FF, D>>,
    ) -> Self {
        Self {
            gate_proj: Linear::new(gate_proj),
            up_proj: Linear::new(up_proj),
            down_proj: Linear::new(down_proj),
        }
    }

    /// Deterministic scaled initialization suitable for CPU tests.
    pub fn initialized(seed: u64) -> Self {
        assert!(D > 0, "expert input width must be non-zero");
        assert!(FF > 0, "expert hidden width must be non-zero");
        let hidden_scale = (D as f32).sqrt().recip();
        let ffn_scale = (FF as f32).sqrt().recip();
        Self::new(
            CpuTensor::uniform(seed).scale(hidden_scale),
            CpuTensor::uniform(seed + 1).scale(hidden_scale),
            CpuTensor::uniform(seed + 2).scale(ffn_scale),
        )
    }

    pub fn sgd_step(&mut self, learning_rate: f32) {
        self.gate_proj
            .w
            .add_scaled_assign(-learning_rate, &self.gate_proj.dw);
        self.up_proj
            .w
            .add_scaled_assign(-learning_rate, &self.up_proj.dw);
        self.down_proj
            .w
            .add_scaled_assign(-learning_rate, &self.down_proj.dw);
    }
}

impl<const C: usize, const D: usize, const FF: usize> Module for ExpertFfn<C, D, FF> {
    type Input = CpuTensor<f32, Rank2<C, D>>;
    type Output = CpuTensor<f32, Rank2<C, D>>;
    type Ctx = ExpertFfnCtx<C, D, FF>;

    fn forward(&self, x: Self::Input) -> (Self::Output, Self::Ctx) {
        let (gate, gate_proj) = self.gate_proj.forward(x.clone());
        let (up, up_proj) = self.up_proj.forward(x);
        let (activated, swiglu) = SwiGlu::<C, FF>.forward((gate, up));
        let (output, down_proj) = self.down_proj.forward(activated);
        (
            output,
            ExpertFfnCtx {
                gate_proj,
                up_proj,
                swiglu,
                down_proj,
            },
        )
    }

    fn backward(&mut self, ctx: Self::Ctx, dy: Self::Output) -> Self::Input {
        let dactivated = self.down_proj.backward(ctx.down_proj, dy);
        let (dgate, dup) = SwiGlu::<C, FF>.backward(ctx.swiglu, dactivated);
        self.gate_proj
            .backward(ctx.gate_proj, dgate)
            .add(&self.up_proj.backward(ctx.up_proj, dup))
    }

    fn zero_grad(&mut self) {
        self.gate_proj.zero_grad();
        self.up_proj.zero_grad();
        self.down_proj.zero_grad();
    }
}

/// Deterministic top-k and capacity assignment saved by a forward pass.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeRouting<const E: usize> {
    /// Expert indices in token-major, rank-major order (`token * K + rank`).
    pub selected_experts: Box<[usize]>,
    /// Renormalized combine weights in the same order.
    pub gate_weights: Box<[f32]>,
    /// Capacity slot for each pair, or `None` when the pair overflowed.
    pub slots: Box<[Option<usize>]>,
    /// Number of pre-capacity top-k assignments to each expert.
    pub assignment_counts: [usize; E],
    /// Number of accepted assignments in each expert's capacity bin.
    pub accepted_counts: [usize; E],
}

/// Backward context for [`MoeFfn`].
pub struct MoeFfnCtx<
    const N: usize,
    const D: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> {
    router: CpuTensor<f32, Rank2<N, D>>,
    probabilities: CpuTensor<f32, Rank2<N, E>>,
    pub mean_probabilities: [f32; E],
    pub routing: MoeRouting<E>,
    expert_outputs: [CpuTensor<f32, Rank2<C, D>>; E],
    expert_contexts: [ExpertFfnCtx<C, D, FF>; E],
}

/// A type-compatible replacement for a dense FFN block.
///
/// `last_aux_loss` uses interior mutability because the project-wide
/// [`Module::forward`] contract takes `&self`.  Read it with
/// `last_aux_loss.get()` after forward.
pub struct MoeFfn<
    const N: usize,
    const D: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> {
    pub router: Linear<N, D, E>,
    pub experts: [ExpertFfn<C, D, FF>; E],
    pub aux_coefficient: f32,
    pub last_aux_loss: Cell<f32>,
}

impl<
    const N: usize,
    const D: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> MoeFfn<N, D, FF, E, K, C>
{
    pub fn new(
        router: Linear<N, D, E>,
        experts: [ExpertFfn<C, D, FF>; E],
        aux_coefficient: f32,
    ) -> Self {
        Self::validate(aux_coefficient);
        Self {
            router,
            experts,
            aux_coefficient,
            last_aux_loss: Cell::new(0.0),
        }
    }

    /// Deterministic scaled initialization. Each parameter gets its own seed.
    pub fn initialized(seed: u64, aux_coefficient: f32) -> Self {
        let router_scale = (D as f32).sqrt().recip();
        Self::new(
            Linear::new(CpuTensor::uniform(seed).scale(router_scale)),
            std::array::from_fn(|expert| ExpertFfn::initialized(seed + 1 + 3 * expert as u64)),
            aux_coefficient,
        )
    }

    fn validate(aux_coefficient: f32) {
        assert!(N > 0, "MoE token count must be non-zero");
        assert!(D > 0, "MoE model width must be non-zero");
        assert!(FF > 0, "MoE hidden width must be non-zero");
        assert!(E > 0, "MoE must contain at least one expert");
        assert!(K > 0, "MoE top-k must be non-zero");
        assert!(K <= E, "MoE top-k cannot exceed expert count");
        assert!(C > 0, "MoE expert capacity must be non-zero");
        assert!(
            aux_coefficient.is_finite() && aux_coefficient >= 0.0,
            "MoE auxiliary coefficient must be finite and non-negative"
        );
    }

    fn route(
        probabilities: &CpuTensor<f32, Rank2<N, E>>,
    ) -> (MoeRouting<E>, [CpuTensor<f32, Rank2<C, D>>; E]) {
        let mut selected_experts = vec![0; N * K].into_boxed_slice();
        let mut gate_weights = vec![0.0; N * K].into_boxed_slice();
        let mut slot_indices = vec![DROPPED; N * K].into_boxed_slice();
        let mut assignment_counts = [0; E];
        let mut accepted_counts = [0; E];

        for token in 0..N {
            let mut order: Vec<usize> = (0..E).collect();
            order.sort_unstable_by(|&a, &b| {
                probabilities.as_slice()[token * E + b]
                    .total_cmp(&probabilities.as_slice()[token * E + a])
                    .then_with(|| a.cmp(&b))
            });

            let denominator = order[..K]
                .iter()
                .map(|&expert| probabilities.as_slice()[token * E + expert] as f64)
                .sum::<f64>() as f32;
            debug_assert!(denominator > 0.0);

            for (rank, &expert) in order.iter().take(K).enumerate() {
                let pair = token * K + rank;
                selected_experts[pair] = expert;
                gate_weights[pair] = probabilities.as_slice()[token * E + expert] / denominator;
                assignment_counts[expert] += 1;
                if accepted_counts[expert] < C {
                    slot_indices[pair] = accepted_counts[expert];
                    accepted_counts[expert] += 1;
                }
            }
        }

        let slots = slot_indices
            .iter()
            .map(|&slot| (slot != DROPPED).then_some(slot))
            .collect();
        (
            MoeRouting {
                selected_experts,
                gate_weights,
                slots,
                assignment_counts,
                accepted_counts,
            },
            std::array::from_fn(|_| CpuTensor::zeros()),
        )
    }

    /// Apply an SGD update to router and expert parameters.
    pub fn sgd_step(&mut self, learning_rate: f32) {
        self.router
            .w
            .add_scaled_assign(-learning_rate, &self.router.dw);
        for expert in &mut self.experts {
            expert.sgd_step(learning_rate);
        }
    }

    /// Fraction of top-k assignments received by each expert.
    pub fn assignment_fractions(ctx: &MoeFfnCtx<N, D, FF, E, K, C>) -> [f32; E] {
        std::array::from_fn(|expert| ctx.routing.assignment_counts[expert] as f32 / (N * K) as f32)
    }
}

impl<
    const N: usize,
    const D: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> Module for MoeFfn<N, D, FF, E, K, C>
{
    type Input = CpuTensor<f32, Rank2<N, D>>;
    type Output = CpuTensor<f32, Rank2<N, D>>;
    type Ctx = MoeFfnCtx<N, D, FF, E, K, C>;

    fn forward(&self, x: Self::Input) -> (Self::Output, Self::Ctx) {
        let (logits, router) = self.router.forward(x.clone());
        let probabilities = logits.softmax_rows();
        let (routing, mut expert_inputs) = Self::route(&probabilities);

        for token in 0..N {
            for rank in 0..K {
                let pair = token * K + rank;
                let Some(slot) = routing.slots[pair] else {
                    continue;
                };
                let expert = routing.selected_experts[pair];
                expert_inputs[expert].as_mut_slice()[slot * D..(slot + 1) * D]
                    .copy_from_slice(&x.as_slice()[token * D..(token + 1) * D]);
            }
        }

        let expert_pairs: [(CpuTensor<f32, Rank2<C, D>>, ExpertFfnCtx<C, D, FF>); E] =
            std::array::from_fn(|expert| {
                self.experts[expert].forward(expert_inputs[expert].clone())
            });
        let expert_outputs = std::array::from_fn(|expert| expert_pairs[expert].0.clone());
        let expert_contexts = expert_pairs.map(|(_, ctx)| ctx);

        let mut output = Self::Output::zeros();
        for token in 0..N {
            for rank in 0..K {
                let pair = token * K + rank;
                let Some(slot) = routing.slots[pair] else {
                    continue;
                };
                let expert = routing.selected_experts[pair];
                let gate = routing.gate_weights[pair];
                for dim in 0..D {
                    output.as_mut_slice()[token * D + dim] +=
                        gate * expert_outputs[expert].as_slice()[slot * D + dim];
                }
            }
        }

        let mean_probabilities = std::array::from_fn(|expert| {
            (0..N)
                .map(|token| probabilities.as_slice()[token * E + expert] as f64)
                .sum::<f64>() as f32
                / N as f32
        });
        let assignment_fractions: [f32; E] =
            std::array::from_fn(|expert| routing.assignment_counts[expert] as f32 / (N * K) as f32);
        let aux_loss = E as f32
            * (0..E)
                .map(|expert| {
                    assignment_fractions[expert] as f64 * mean_probabilities[expert] as f64
                })
                .sum::<f64>() as f32;
        self.last_aux_loss.set(aux_loss);

        (
            output,
            MoeFfnCtx {
                router,
                probabilities,
                mean_probabilities,
                routing,
                expert_outputs,
                expert_contexts,
            },
        )
    }

    fn backward(&mut self, ctx: Self::Ctx, dy: Self::Output) -> Self::Input {
        let mut expert_output_gradients: [CpuTensor<f32, Rank2<C, D>>; E] =
            std::array::from_fn(|_| CpuTensor::zeros());
        let mut gate_gradients = vec![0.0f32; N * K];

        for token in 0..N {
            for rank in 0..K {
                let pair = token * K + rank;
                let Some(slot) = ctx.routing.slots[pair] else {
                    continue;
                };
                let expert = ctx.routing.selected_experts[pair];
                let gate = ctx.routing.gate_weights[pair];
                let mut gate_gradient = 0.0f64;
                for dim in 0..D {
                    let token_index = token * D + dim;
                    let expert_index = slot * D + dim;
                    expert_output_gradients[expert].as_mut_slice()[expert_index] +=
                        gate * dy.as_slice()[token_index];
                    gate_gradient += ctx.expert_outputs[expert].as_slice()[expert_index] as f64
                        * dy.as_slice()[token_index] as f64;
                }
                gate_gradients[pair] = gate_gradient as f32;
            }
        }

        let mut expert_contexts = ctx.expert_contexts.into_iter();
        let mut expert_output_gradients = expert_output_gradients.into_iter();
        let expert_input_gradients: [CpuTensor<f32, Rank2<C, D>>; E] =
            std::array::from_fn(|expert| {
                self.experts[expert].backward(
                    expert_contexts.next().unwrap(),
                    expert_output_gradients.next().unwrap(),
                )
            });

        let mut dx = Self::Input::zeros();
        for token in 0..N {
            for rank in 0..K {
                let pair = token * K + rank;
                let Some(slot) = ctx.routing.slots[pair] else {
                    continue;
                };
                let expert = ctx.routing.selected_experts[pair];
                for dim in 0..D {
                    dx.as_mut_slice()[token * D + dim] +=
                        expert_input_gradients[expert].as_slice()[slot * D + dim];
                }
            }
        }

        let assignment_fractions: [f32; E] = std::array::from_fn(|expert| {
            ctx.routing.assignment_counts[expert] as f32 / (N * K) as f32
        });
        let mut probability_gradients = CpuTensor::<f32, Rank2<N, E>>::zeros();

        for token in 0..N {
            let weighted_gate_gradient = (0..K)
                .map(|rank| {
                    let pair = token * K + rank;
                    gate_gradients[pair] as f64 * ctx.routing.gate_weights[pair] as f64
                })
                .sum::<f64>() as f32;
            let selected_probability_sum = (0..K)
                .map(|rank| {
                    let expert = ctx.routing.selected_experts[token * K + rank];
                    ctx.probabilities.as_slice()[token * E + expert] as f64
                })
                .sum::<f64>() as f32;

            for rank in 0..K {
                let pair = token * K + rank;
                let expert = ctx.routing.selected_experts[pair];
                probability_gradients.as_mut_slice()[token * E + expert] +=
                    (gate_gradients[pair] - weighted_gate_gradient) / selected_probability_sum;
            }
            for (expert, &fraction) in assignment_fractions.iter().enumerate() {
                probability_gradients.as_mut_slice()[token * E + expert] +=
                    self.aux_coefficient * E as f32 * fraction / N as f32;
            }
        }

        let mut dlogits = CpuTensor::<f32, Rank2<N, E>>::zeros();
        for token in 0..N {
            let softmax_dot = (0..E)
                .map(|expert| {
                    ctx.probabilities.as_slice()[token * E + expert] as f64
                        * probability_gradients.as_slice()[token * E + expert] as f64
                })
                .sum::<f64>() as f32;
            for expert in 0..E {
                let index = token * E + expert;
                dlogits.as_mut_slice()[index] = ctx.probabilities.as_slice()[index]
                    * (probability_gradients.as_slice()[index] - softmax_dot);
            }
        }
        dx.add_assign(&self.router.backward(ctx.router, dlogits));
        dx
    }

    fn zero_grad(&mut self) {
        self.router.zero_grad();
        for expert in &mut self.experts {
            expert.zero_grad();
        }
    }
}
