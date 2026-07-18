//! Optimizers and statically typed optimizer state.
//!
//! The CPU implementation is the numerical reference for GPU optimizer
//! kernels. `LlamaAdamW` and `LlamaMuon` mirror the model's parameter
//! structure, preserving each parameter shape in the type system without a
//! type-erased parameter registry.

mod muon;

pub use muon::{MuonConfig, MuonMomentum, muon_step, zeroth_power_via_newton_schulz};

use nn::Llama;
use tensor_core::{Rank1, Rank2, Shape, bf16};
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

/// Mixed Muon/AdamW state for the single-block reference Llama.
///
/// Hidden projection matrices use Muon. Embeddings, normalization gains, and
/// the classifier head use AdamW, matching the routing prescribed by Muon.
pub struct LlamaMuon<const VOCAB: usize, const D: usize, const FF: usize> {
    muon_config: MuonConfig,
    adamw_config: AdamWConfig,
    step: u64,
    pub embedding: AdamWMoments<Rank2<VOCAB, D>>,
    pub attention_norm: AdamWMoments<Rank1<D>>,
    pub q_proj: MuonMomentum<Rank2<D, D>>,
    pub k_proj: MuonMomentum<Rank2<D, D>>,
    pub v_proj: MuonMomentum<Rank2<D, D>>,
    pub o_proj: MuonMomentum<Rank2<D, D>>,
    pub ffn_norm: AdamWMoments<Rank1<D>>,
    pub gate_proj: MuonMomentum<Rank2<D, FF>>,
    pub up_proj: MuonMomentum<Rank2<D, FF>>,
    pub down_proj: MuonMomentum<Rank2<FF, D>>,
    pub final_norm: AdamWMoments<Rank1<D>>,
    pub lm_head: AdamWMoments<Rank2<D, VOCAB>>,
}

impl<const VOCAB: usize, const D: usize, const FF: usize> LlamaMuon<VOCAB, D, FF> {
    pub fn new(muon_config: MuonConfig, adamw_config: AdamWConfig) -> Self {
        muon_config.validate();
        adamw_config.validate();
        Self {
            muon_config,
            adamw_config,
            step: 0,
            embedding: AdamWMoments::zeros(),
            attention_norm: AdamWMoments::zeros(),
            q_proj: MuonMomentum::zeros(),
            k_proj: MuonMomentum::zeros(),
            v_proj: MuonMomentum::zeros(),
            o_proj: MuonMomentum::zeros(),
            ffn_norm: AdamWMoments::zeros(),
            gate_proj: MuonMomentum::zeros(),
            up_proj: MuonMomentum::zeros(),
            down_proj: MuonMomentum::zeros(),
            final_norm: AdamWMoments::zeros(),
            lm_head: AdamWMoments::zeros(),
        }
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn muon_config(&self) -> MuonConfig {
        self.muon_config
    }

    pub fn adamw_config(&self) -> AdamWConfig {
        self.adamw_config
    }

    pub fn update<const N: usize, const T: usize, const H: usize, const HD: usize>(
        &mut self,
        model: &mut Llama<N, T, VOCAB, D, H, HD, FF>,
    ) {
        self.step = self.step.checked_add(1).expect("Muon step overflow");
        let step = self.step;
        let decay = self.adamw_config;
        let no_decay = self.adamw_config.without_weight_decay();

        macro_rules! adamw {
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
        macro_rules! muon {
            ($field:ident) => {
                muon_step(
                    &mut model.$field.w,
                    &model.$field.dw,
                    &mut self.$field,
                    self.muon_config,
                );
            };
        }

        adamw!(embedding, decay);
        adamw!(attention_norm, no_decay);
        muon!(q_proj);
        muon!(k_proj);
        muon!(v_proj);
        muon!(o_proj);
        adamw!(ffn_norm, no_decay);
        muon!(gate_proj);
        muon!(up_proj);
        muon!(down_proj);
        adamw!(final_norm, no_decay);
        adamw!(lm_head, decay);
    }
}

/// Full-precision source of truth for a bf16 compute parameter.
///
/// `S` is shared by the master and compute tensors, so synchronizing tensors
/// with different static shapes is a compile-time error.
#[derive(Clone, Debug, PartialEq)]
pub struct Fp32MasterWeights<S: Shape> {
    values: CpuTensor<f32, S>,
}

impl<S: Shape> Fp32MasterWeights<S> {
    /// Preserve full-precision initialization values as the master copy.
    pub fn new(values: CpuTensor<f32, S>) -> Self {
        Self { values }
    }

    /// Reconstruct a master copy from bf16 weights, for example when importing
    /// a bf16-only checkpoint. Precision discarded by that checkpoint cannot
    /// be recovered.
    pub fn from_compute(compute: &CpuTensor<bf16, S>) -> Self {
        Self::new(compute.to_f32())
    }

    /// Read the fp32 source of truth.
    pub fn values(&self) -> &CpuTensor<f32, S> {
        &self.values
    }

    /// Create a rounded bf16 compute copy.
    pub fn to_compute(&self) -> CpuTensor<bf16, S> {
        self.values.to_bf16()
    }

    /// Refresh an existing bf16 compute copy without changing its allocation.
    pub fn sync_compute(&self, compute: &mut CpuTensor<bf16, S>) {
        for (dst, &src) in compute
            .as_mut_slice()
            .iter_mut()
            .zip(self.values.as_slice())
        {
            *dst = bf16::from_f32(src);
        }
    }

    /// Apply an fp32 additive update and then refresh the bf16 compute copy.
    ///
    /// The update is retained even when it is too small to change bf16 in a
    /// single step. Optimizers should pass their signed update here (for
    /// gradient descent, this is normally negative).
    pub fn apply_update(&mut self, update: &CpuTensor<f32, S>, compute: &mut CpuTensor<bf16, S>) {
        self.values.add_assign(update);
        self.sync_compute(compute);
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
    fn llama_muon_routes_hidden_matrices_and_auxiliary_parameters() {
        let mut model = Llama::<4, 4, 7, 8, 2, 4, 12>::new(7);
        model.embedding.dw.as_mut_slice().fill(1.0);
        model.attention_norm.dw.as_mut_slice().fill(1.0);
        model.q_proj.dw.as_mut_slice().fill(1.0);
        model.lm_head.dw.as_mut_slice().fill(1.0);
        let mut optimizer = LlamaMuon::new(MuonConfig::default(), AdamWConfig::default());

        optimizer.update(&mut model);

        assert_eq!(optimizer.step(), 1);
        assert!(
            optimizer
                .q_proj
                .momentum
                .as_slice()
                .iter()
                .all(|&value| (value - 0.05).abs() < 1e-6)
        );
        assert!(
            optimizer
                .embedding
                .first
                .as_slice()
                .iter()
                .all(|&value| (value - 0.1).abs() < 1e-6)
        );
        assert!(
            optimizer
                .attention_norm
                .second
                .as_slice()
                .iter()
                .all(|&value| (value - 0.001).abs() < 1e-6)
        );
        assert!(
            optimizer
                .lm_head
                .first
                .as_slice()
                .iter()
                .all(|&value| (value - 0.1).abs() < 1e-6)
        );
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

    #[test]
    fn muon_overfits_the_tiny_llama_batch() {
        type TinyLlama = Llama<4, 4, 4, 8, 2, 4, 12>;
        let tokens = [0, 1, 2, 3];
        let targets = [1, 2, 3, 0];
        let mut model = TinyLlama::new(100);
        let mut optimizer = LlamaMuon::new(
            MuonConfig {
                learning_rate: 0.02,
                ..MuonConfig::default()
            },
            AdamWConfig {
                learning_rate: 0.03,
                weight_decay: 0.0,
                ..AdamWConfig::default()
            },
        );
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
            "tiny batch did not overfit with Muon: initial={initial_loss}, final={final_loss}"
        );
        assert!(final_loss < initial_loss * 0.05);
    }
}

#[cfg(test)]
mod master_weights_tests {
    use tensor_core::{Rank1, bf16};
    use tensor_cpu::CpuTensor;

    use super::Fp32MasterWeights;

    #[test]
    fn initialization_keeps_unrounded_master_values() {
        let initial = CpuTensor::<f32, Rank1<2>>::from_slice(&[1.001, -2.003]);
        let master = Fp32MasterWeights::new(initial.clone());
        let compute = master.to_compute();

        assert_eq!(master.values(), &initial);
        assert_eq!(
            compute.as_slice(),
            &[bf16::from_f32(1.001), bf16::from_f32(-2.003)]
        );
    }

    #[test]
    fn sub_bf16_updates_accumulate_in_master_weights() {
        let mut master = Fp32MasterWeights::new(CpuTensor::<f32, Rank1<1>>::from_slice(&[1.0]));
        let mut compute = master.to_compute();
        let update = CpuTensor::<f32, Rank1<1>>::from_slice(&[0.001]);

        for _ in 0..3 {
            master.apply_update(&update, &mut compute);
        }
        assert_eq!(compute.as_slice(), &[bf16::from_f32(1.0)]);

        master.apply_update(&update, &mut compute);
        assert_eq!(master.values().as_slice(), &[1.0040002]);
        assert_eq!(compute.as_slice(), &[bf16::from_f32(1.0040002)]);
        assert_ne!(compute.as_slice(), &[bf16::from_f32(1.0)]);
    }

    #[test]
    fn bf16_checkpoint_can_seed_master_weights() {
        let compute =
            CpuTensor::<bf16, Rank1<2>>::from_slice(&[bf16::from_f32(0.25), bf16::from_f32(-4.5)]);
        let master = Fp32MasterWeights::from_compute(&compute);

        assert_eq!(master.values().as_slice(), &[0.25, -4.5]);
        assert_eq!(master.to_compute(), compute);
    }
}
