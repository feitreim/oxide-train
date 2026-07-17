//! Optimizers and statically typed optimizer state.
//!
//! The CPU implementation is the numerical reference for GPU optimizer
//! kernels. `LlamaAdamW` mirrors the model's parameter structure, preserving
//! each parameter shape in the type system without a type-erased parameter
//! registry.

use nn::Llama;
use tensor_core::{Rank1, Rank2, Shape};
use tensor_cpu::CpuTensor;

/// Hyperparameters for decoupled AdamW.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl AdamWConfig {
    pub fn validate(self) {
        assert!(
            self.learning_rate.is_finite() && self.learning_rate >= 0.0,
            "learning rate must be finite and non-negative"
        );
        assert!(
            self.beta1.is_finite() && (0.0..1.0).contains(&self.beta1),
            "beta1 must be in [0, 1)"
        );
        assert!(
            self.beta2.is_finite() && (0.0..1.0).contains(&self.beta2),
            "beta2 must be in [0, 1)"
        );
        assert!(
            self.epsilon.is_finite() && self.epsilon > 0.0,
            "epsilon must be finite and positive"
        );
        assert!(
            self.weight_decay.is_finite() && self.weight_decay >= 0.0,
            "weight decay must be finite and non-negative"
        );
    }

    pub fn without_weight_decay(self) -> Self {
        Self {
            weight_decay: 0.0,
            ..self
        }
    }

    /// Multipliers applied to the first and second moments for bias correction.
    pub fn bias_correction(self, step: u64) -> (f32, f32) {
        assert!(step > 0, "AdamW steps are one-indexed");
        let step = i32::try_from(step).unwrap_or(i32::MAX);
        (
            1.0 / (1.0 - self.beta1.powi(step)),
            1.0 / (1.0 - self.beta2.powi(step)),
        )
    }
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// AdamW's first and second moments for one statically shaped parameter.
pub struct AdamWMoments<S: Shape> {
    pub first: CpuTensor<f32, S>,
    pub second: CpuTensor<f32, S>,
}

impl<S: Shape> AdamWMoments<S> {
    pub fn zeros() -> Self {
        Self {
            first: CpuTensor::zeros(),
            second: CpuTensor::zeros(),
        }
    }
}

/// Apply one reference AdamW update.
///
/// Weight decay is decoupled from the gradient moments:
/// `p -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)`.
pub fn adamw_step<S: Shape>(
    parameter: &mut CpuTensor<f32, S>,
    gradient: &CpuTensor<f32, S>,
    moments: &mut AdamWMoments<S>,
    config: AdamWConfig,
    step: u64,
) {
    config.validate();
    let (first_correction, second_correction) = config.bias_correction(step);

    for (((parameter, &gradient), first), second) in parameter
        .as_mut_slice()
        .iter_mut()
        .zip(gradient.as_slice())
        .zip(moments.first.as_mut_slice())
        .zip(moments.second.as_mut_slice())
    {
        *first = config.beta1 * *first + (1.0 - config.beta1) * gradient;
        *second = config.beta2 * *second + (1.0 - config.beta2) * gradient * gradient;
        let first_hat = *first * first_correction;
        let second_hat = *second * second_correction;
        let update =
            first_hat / (second_hat.sqrt() + config.epsilon) + config.weight_decay * *parameter;
        *parameter -= config.learning_rate * update;
    }
}

/// Parameter categories used for optimizer routing and checkpoint metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterKind {
    Embedding,
    Norm,
    Matrix,
    Head,
}

/// A shape-preserving visitor over CPU parameter/gradient pairs.
pub trait CpuParameterVisitor {
    fn visit<S: Shape>(
        &mut self,
        name: &'static str,
        kind: ParameterKind,
        parameter: &mut CpuTensor<f32, S>,
        gradient: &CpuTensor<f32, S>,
    );
}

pub trait VisitCpuParameters {
    fn visit_cpu_parameters<V: CpuParameterVisitor>(&mut self, visitor: &mut V);
}

impl<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> VisitCpuParameters for Llama<N, T, VOCAB, D, H, HD, FF>
{
    fn visit_cpu_parameters<V: CpuParameterVisitor>(&mut self, visitor: &mut V) {
        macro_rules! visit {
            ($field:ident, $kind:ident) => {
                visitor.visit(
                    stringify!($field),
                    ParameterKind::$kind,
                    &mut self.$field.w,
                    &self.$field.dw,
                );
            };
        }

        visit!(embedding, Embedding);
        visit!(attention_norm, Norm);
        visit!(q_proj, Matrix);
        visit!(k_proj, Matrix);
        visit!(v_proj, Matrix);
        visit!(o_proj, Matrix);
        visit!(ffn_norm, Norm);
        visit!(gate_proj, Matrix);
        visit!(up_proj, Matrix);
        visit!(down_proj, Matrix);
        visit!(final_norm, Norm);
        visit!(lm_head, Head);
    }
}

/// AdamW state for the single-block reference Llama.
pub struct LlamaAdamW<const VOCAB: usize, const D: usize, const FF: usize> {
    config: AdamWConfig,
    step: u64,
    pub embedding: AdamWMoments<Rank2<VOCAB, D>>,
    pub attention_norm: AdamWMoments<Rank1<D>>,
    pub q_proj: AdamWMoments<Rank2<D, D>>,
    pub k_proj: AdamWMoments<Rank2<D, D>>,
    pub v_proj: AdamWMoments<Rank2<D, D>>,
    pub o_proj: AdamWMoments<Rank2<D, D>>,
    pub ffn_norm: AdamWMoments<Rank1<D>>,
    pub gate_proj: AdamWMoments<Rank2<D, FF>>,
    pub up_proj: AdamWMoments<Rank2<D, FF>>,
    pub down_proj: AdamWMoments<Rank2<FF, D>>,
    pub final_norm: AdamWMoments<Rank1<D>>,
    pub lm_head: AdamWMoments<Rank2<D, VOCAB>>,
}

impl<const VOCAB: usize, const D: usize, const FF: usize> LlamaAdamW<VOCAB, D, FF> {
    pub fn new(config: AdamWConfig) -> Self {
        config.validate();
        Self {
            config,
            step: 0,
            embedding: AdamWMoments::zeros(),
            attention_norm: AdamWMoments::zeros(),
            q_proj: AdamWMoments::zeros(),
            k_proj: AdamWMoments::zeros(),
            v_proj: AdamWMoments::zeros(),
            o_proj: AdamWMoments::zeros(),
            ffn_norm: AdamWMoments::zeros(),
            gate_proj: AdamWMoments::zeros(),
            up_proj: AdamWMoments::zeros(),
            down_proj: AdamWMoments::zeros(),
            final_norm: AdamWMoments::zeros(),
            lm_head: AdamWMoments::zeros(),
        }
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn update<const N: usize, const T: usize, const H: usize, const HD: usize>(
        &mut self,
        model: &mut Llama<N, T, VOCAB, D, H, HD, FF>,
    ) {
        self.step = self.step.checked_add(1).expect("AdamW step overflow");
        let step = self.step;
        let decay = self.config;
        let no_decay = self.config.without_weight_decay();

        macro_rules! update {
            ($field:ident, $config:expr) => {
                adamw_step(
                    &mut model.$field.w,
                    &model.$field.dw,
                    &mut self.$field,
                    $config,
                    step,
                );
            };
        }

        update!(embedding, decay);
        update!(attention_norm, no_decay);
        update!(q_proj, decay);
        update!(k_proj, decay);
        update!(v_proj, decay);
        update!(o_proj, decay);
        update!(ffn_norm, no_decay);
        update!(gate_proj, decay);
        update!(up_proj, decay);
        update!(down_proj, decay);
        update!(final_norm, no_decay);
        update!(lm_head, decay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_matches_closed_form_adamw() {
        let config = AdamWConfig {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.2,
        };
        let mut parameter = CpuTensor::<f32, Rank1<2>>::from_slice(&[2.0, -3.0]);
        let gradient = CpuTensor::from_slice(&[0.5, -0.25]);
        let mut moments = AdamWMoments::zeros();

        adamw_step(&mut parameter, &gradient, &mut moments, config, 1);

        // On step one, bias-corrected moments are g and g^2.
        let expected = [
            2.0 - 0.1 * (1.0 + 0.2 * 2.0),
            -3.0 - 0.1 * (-1.0 + 0.2 * -3.0),
        ];
        for (&actual, expected) in parameter.as_slice().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn llama_visitor_reports_all_parameters_and_kinds() {
        struct Inventory {
            names: Vec<&'static str>,
            norm_elements: usize,
        }

        impl CpuParameterVisitor for Inventory {
            fn visit<S: Shape>(
                &mut self,
                name: &'static str,
                kind: ParameterKind,
                _parameter: &mut CpuTensor<f32, S>,
                _gradient: &CpuTensor<f32, S>,
            ) {
                self.names.push(name);
                if kind == ParameterKind::Norm {
                    self.norm_elements += S::NUM_ELEMENTS;
                }
            }
        }

        let mut model = Llama::<4, 4, 7, 8, 2, 4, 12>::new(7);
        let mut inventory = Inventory {
            names: Vec::new(),
            norm_elements: 0,
        };
        model.visit_cpu_parameters(&mut inventory);

        assert_eq!(inventory.names.len(), 12);
        assert_eq!(inventory.norm_elements, 3 * 8);
        assert_eq!(inventory.names[0], "embedding");
        assert_eq!(inventory.names[11], "lm_head");
    }

    #[test]
    fn adamw_overfits_the_tiny_llama_batch() {
        type TinyLlama = Llama<4, 4, 4, 8, 2, 4, 12>;
        let tokens = [0, 1, 2, 3];
        let targets = [1, 2, 3, 0];
        let mut model = TinyLlama::new(100);
        let mut optimizer = LlamaAdamW::new(AdamWConfig {
            learning_rate: 0.03,
            weight_decay: 0.0,
            ..AdamWConfig::default()
        });
        let initial_loss = model.forward(tokens, targets).0.as_slice()[0];

        for _ in 0..200 {
            model.zero_grad();
            let (_, ctx) = model.forward(tokens, targets);
            model.backward(ctx);
            optimizer.update(&mut model);
        }
        let final_loss = model.forward(tokens, targets).0.as_slice()[0];

        assert!(
            final_loss < 0.05,
            "tiny batch did not overfit: initial={initial_loss}, final={final_loss}"
        );
        assert!(final_loss < initial_loss * 0.05);
    }
}
