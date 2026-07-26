//! Optimizers and statically typed optimizer state.
//!
//! The CPU implementation is the numerical reference for GPU optimizer
//! kernels. `DenseAdamW` and `DenseMuon` mirror the model's parameter
//! structure, preserving each parameter shape in the type system without a
//! type-erased parameter registry.

mod muon;

pub use muon::{
    MuonConfig, MuonMomentum, NEWTON_SCHULZ_A, NEWTON_SCHULZ_B, NEWTON_SCHULZ_C,
    NEWTON_SCHULZ_EPSILON, muon_step, zeroth_power_via_newton_schulz,
};

use nn::{Dense, MoeDense};
use tensor_core::{Rank1, Rank2, Shape, bf16, bf16_stochastic, rng};
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
    /// The single source of truth for both [`Self::is_valid`] and
    /// [`Self::validate`].
    fn checks(self) -> [(bool, &'static str); 5] {
        [
            (
                self.learning_rate.is_finite() && self.learning_rate >= 0.0,
                "learning rate must be finite and non-negative",
            ),
            (
                self.beta1.is_finite() && (0.0..1.0).contains(&self.beta1),
                "beta1 must be in [0, 1)",
            ),
            (
                self.beta2.is_finite() && (0.0..1.0).contains(&self.beta2),
                "beta2 must be in [0, 1)",
            ),
            (
                self.epsilon.is_finite() && self.epsilon > 0.0,
                "epsilon must be finite and positive",
            ),
            (
                self.weight_decay.is_finite() && self.weight_decay >= 0.0,
                "weight decay must be finite and non-negative",
            ),
        ]
    }

    pub fn is_valid(self) -> bool {
        self.checks().iter().all(|&(ok, _)| ok)
    }

    pub fn validate(self) {
        for (ok, message) in self.checks() {
            assert!(ok, "{message}");
        }
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

/// Host-side schedule for the MoE load-balancing loss coefficient.
///
/// The coefficient decays linearly from `base_coefficient` at step zero to
/// zero at `decay_horizon`. Both parameters are checkpointed so resuming can
/// re-evaluate the schedule from the restored global step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxLossSchedule {
    pub base_coefficient: f32,
    pub decay_horizon: f32,
}

impl AuxLossSchedule {
    /// The single source of truth for both [`Self::is_valid`] and
    /// [`Self::validate`].
    fn checks(self) -> [(bool, &'static str); 2] {
        [
            (
                self.base_coefficient.is_finite() && self.base_coefficient >= 0.0,
                "auxiliary loss base coefficient must be finite and non-negative",
            ),
            (
                self.decay_horizon.is_finite() && self.decay_horizon > 0.0,
                "auxiliary loss decay horizon must be finite and positive",
            ),
        ]
    }

    pub fn is_valid(self) -> bool {
        self.checks().iter().all(|&(ok, _)| ok)
    }

    pub fn validate(self) {
        for (ok, message) in self.checks() {
            assert!(ok, "{message}");
        }
    }

    pub fn coefficient(self, step: u64) -> f32 {
        self.validate();
        self.base_coefficient * (1.0 - step as f32 / self.decay_horizon).max(0.0)
    }
}

impl Default for AuxLossSchedule {
    fn default() -> Self {
        Self {
            base_coefficient: 1e-2,
            decay_horizon: 10_000.0,
        }
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

/// How an fp32 update is committed into a bf16 master weight.
///
/// bf16 carries eight mantissa bits, so at learning-rate scale a single step's
/// update is regularly smaller than half an ulp of the weight it lands on.
/// Round-to-nearest drops those updates outright — the plateau
/// `examples/overfit_probe.rs` reproduces — while stochastic rounding keeps
/// them in expectation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MasterRounding {
    /// Round to nearest even, the `bf16::from_f32` conversion.
    Nearest,
    /// Round up with probability equal to the discarded mantissa fraction,
    /// drawn from a splitmix64 stream keyed on `(step, parameter id, element
    /// index)`. Deterministic by construction: no runtime entropy, so reruns
    /// and checkpoint resumes stay bit-identical.
    Stochastic,
}

/// Storage dtype of the parameter an optimizer writes back into.
///
/// The GPU keeps bf16 masters for every matrix-shaped parameter (SPEC §7,
/// decision #8 successor); norms, the router, and all optimizer moments stay
/// fp32. The CPU reference keeps `f32` storage throughout but snaps a bf16
/// master onto the bf16 grid after each update, so both sides walk the same
/// grid and a parity comparison compares the same trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MasterStorage {
    Fp32,
    Bf16 {
        rounding: MasterRounding,
        /// Keys the stochastic stream; ignored by [`MasterRounding::Nearest`].
        parameter_id: u64,
    },
}

impl MasterStorage {
    /// Round one already-updated parameter value into its storage dtype.
    pub(crate) fn commit(self, value: f32, step: u64, element: usize) -> f32 {
        match self {
            Self::Fp32 => value,
            Self::Bf16 {
                rounding: MasterRounding::Nearest,
                ..
            } => bf16::from_f32(value).to_f32(),
            Self::Bf16 {
                rounding: MasterRounding::Stochastic,
                parameter_id,
            } => {
                let seed = rng::stream_seed(step, parameter_id);
                bf16_stochastic(value, rng::stream_draw(seed, element as u64)).to_f32()
            }
        }
    }
}

/// Snap every element of a parameter onto the bf16 grid.
///
/// Initialization runs in fp32 on both sides; the GPU rounds when it uploads
/// the master, so the CPU reference has to round too or the two models start
/// one rounding apart and parity measures that instead of the update.
pub fn round_to_bf16_master<S: Shape>(parameter: &mut CpuTensor<f32, S>) {
    for value in parameter.as_mut_slice() {
        *value = bf16::from_f32(*value).to_f32();
    }
}

/// Whether the GPU stores this parameter kind as a bf16 master.
///
/// Norms are `2·L+1` vectors of `D` with no memory leverage and high
/// sensitivity; the router is fp32 end-to-end because rounding near a top-k
/// boundary reassigns tokens rather than perturbing outputs (decision #22).
pub fn kind_has_bf16_master(kind: ParameterKind) -> bool {
    match kind {
        ParameterKind::Embedding | ParameterKind::Matrix | ParameterKind::Head => true,
        ParameterKind::Norm | ParameterKind::Router => false,
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
/// `p -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)`. Moments are
/// always fp32; `storage` decides only how the new parameter value is stored.
pub fn adamw_step<S: Shape>(
    parameter: &mut CpuTensor<f32, S>,
    gradient: &CpuTensor<f32, S>,
    moments: &mut AdamWMoments<S>,
    config: AdamWConfig,
    step: u64,
    storage: MasterStorage,
) {
    config.validate();
    let (first_correction, second_correction) = config.bias_correction(step);

    for (element, (((parameter, &gradient), first), second)) in parameter
        .as_mut_slice()
        .iter_mut()
        .zip(gradient.as_slice())
        .zip(moments.first.as_mut_slice())
        .zip(moments.second.as_mut_slice())
        .enumerate()
    {
        *first = config.beta1 * *first + (1.0 - config.beta1) * gradient;
        *second = config.beta2 * *second + (1.0 - config.beta2) * gradient * gradient;
        let first_hat = *first * first_correction;
        let second_hat = *second * second_correction;
        let update =
            first_hat / (second_hat.sqrt() + config.epsilon) + config.weight_decay * *parameter;
        *parameter = storage.commit(*parameter - config.learning_rate * update, step, element);
    }
}

/// Parameter categories used for optimizer routing and checkpoint metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterKind {
    Embedding,
    Norm,
    Matrix,
    /// Skinny fp32 MoE routing matrix; intentionally remains on AdamW.
    Router,
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

/// Snap every bf16-master parameter of a CPU model onto the bf16 grid.
///
/// The GPU rounds when it uploads a model, so a reference built from the same
/// seed starts one rounding away unless it does this too.
pub struct Bf16MasterInit;

impl CpuParameterVisitor for Bf16MasterInit {
    fn visit<S: Shape>(
        &mut self,
        _name: &'static str,
        kind: ParameterKind,
        parameter: &mut CpuTensor<f32, S>,
        _gradient: &CpuTensor<f32, S>,
    ) {
        if kind_has_bf16_master(kind) {
            round_to_bf16_master(parameter);
        }
    }
}

impl<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> VisitCpuParameters for Dense<N, T, VOCAB, D, H, HD, FF>
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

impl<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
    const L: usize,
> VisitCpuParameters for MoeDense<N, T, VOCAB, D, H, HD, FF, E, K, C, L>
{
    fn visit_cpu_parameters<V: CpuParameterVisitor>(&mut self, visitor: &mut V) {
        macro_rules! visit {
            ($name:literal, $parameter:expr, $gradient:expr, $kind:ident) => {
                visitor.visit($name, ParameterKind::$kind, $parameter, $gradient);
            };
        }

        visit!(
            "embedding",
            &mut self.embedding.w,
            &self.embedding.dw,
            Embedding
        );
        for block in &mut self.blocks {
            visit!(
                "attention_norm",
                &mut block.attention_norm.w,
                &block.attention_norm.dw,
                Norm
            );
            visit!("q_proj", &mut block.q_proj.w, &block.q_proj.dw, Matrix);
            visit!("k_proj", &mut block.k_proj.w, &block.k_proj.dw, Matrix);
            visit!("v_proj", &mut block.v_proj.w, &block.v_proj.dw, Matrix);
            visit!("o_proj", &mut block.o_proj.w, &block.o_proj.dw, Matrix);
            visit!("ffn_norm", &mut block.ffn_norm.w, &block.ffn_norm.dw, Norm);
            visit!(
                "ffn.router",
                &mut block.ffn.router.w,
                &block.ffn.router.dw,
                Router
            );
            for expert in &mut block.ffn.experts {
                visit!(
                    "ffn.expert.gate_proj",
                    &mut expert.gate_proj.w,
                    &expert.gate_proj.dw,
                    Matrix
                );
                visit!(
                    "ffn.expert.up_proj",
                    &mut expert.up_proj.w,
                    &expert.up_proj.dw,
                    Matrix
                );
                visit!(
                    "ffn.expert.down_proj",
                    &mut expert.down_proj.w,
                    &expert.down_proj.dw,
                    Matrix
                );
            }
        }
        visit!(
            "final_norm",
            &mut self.final_norm.w,
            &self.final_norm.dw,
            Norm
        );
        visit!("lm_head", &mut self.lm_head.w, &self.lm_head.dw, Head);
    }
}

/// AdamW state for the single-block reference Dense.
///
/// Mirrors the GPU's storage split: embeddings, hidden matrices, and the head
/// are bf16 masters; norms stay fp32.
pub struct DenseAdamW<const VOCAB: usize, const D: usize, const FF: usize> {
    config: AdamWConfig,
    master_rounding: MasterRounding,
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

impl<const VOCAB: usize, const D: usize, const FF: usize> DenseAdamW<VOCAB, D, FF> {
    pub fn new(config: AdamWConfig) -> Self {
        Self::with_master_rounding(config, MasterRounding::Nearest)
    }

    pub fn with_master_rounding(config: AdamWConfig, master_rounding: MasterRounding) -> Self {
        config.validate();
        Self {
            config,
            master_rounding,
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
        model: &mut Dense<N, T, VOCAB, D, H, HD, FF>,
    ) {
        self.step = self.step.checked_add(1).expect("AdamW step overflow");
        let step = self.step;
        let decay = self.config;
        let no_decay = self.config.without_weight_decay();
        let rounding = self.master_rounding;

        macro_rules! update {
            ($field:ident, $config:expr, $storage:expr) => {
                adamw_step(
                    &mut model.$field.w,
                    &model.$field.dw,
                    &mut self.$field,
                    $config,
                    step,
                    $storage,
                );
            };
        }
        // Parameter ids are the parameter's position in
        // `visit_cpu_parameters` order: structural, so they survive a
        // checkpoint round-trip and keep the noise stream reproducible.
        macro_rules! master {
            ($id:literal) => {
                MasterStorage::Bf16 {
                    rounding,
                    parameter_id: $id,
                }
            };
        }

        update!(embedding, decay, master!(0));
        update!(attention_norm, no_decay, MasterStorage::Fp32);
        update!(q_proj, decay, master!(2));
        update!(k_proj, decay, master!(3));
        update!(v_proj, decay, master!(4));
        update!(o_proj, decay, master!(5));
        update!(ffn_norm, no_decay, MasterStorage::Fp32);
        update!(gate_proj, decay, master!(7));
        update!(up_proj, decay, master!(8));
        update!(down_proj, decay, master!(9));
        update!(final_norm, no_decay, MasterStorage::Fp32);
        update!(lm_head, decay, master!(11));
    }
}

/// Mixed Muon/AdamW state for the single-block reference Dense.
///
/// Hidden projection matrices use Muon. Embeddings, normalization gains, and
/// the classifier head use AdamW, matching the routing prescribed by Muon.
pub struct DenseMuon<const VOCAB: usize, const D: usize, const FF: usize> {
    muon_config: MuonConfig,
    adamw_config: AdamWConfig,
    master_rounding: MasterRounding,
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

impl<const VOCAB: usize, const D: usize, const FF: usize> DenseMuon<VOCAB, D, FF> {
    pub fn new(muon_config: MuonConfig, adamw_config: AdamWConfig) -> Self {
        Self::with_master_rounding(muon_config, adamw_config, MasterRounding::Nearest)
    }

    pub fn with_master_rounding(
        muon_config: MuonConfig,
        adamw_config: AdamWConfig,
        master_rounding: MasterRounding,
    ) -> Self {
        muon_config.validate();
        adamw_config.validate();
        Self {
            muon_config,
            adamw_config,
            master_rounding,
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
        model: &mut Dense<N, T, VOCAB, D, H, HD, FF>,
    ) {
        self.step = self.step.checked_add(1).expect("Muon step overflow");
        let step = self.step;
        let decay = self.adamw_config;
        let no_decay = self.adamw_config.without_weight_decay();

        let rounding = self.master_rounding;

        macro_rules! adamw {
            ($field:ident, $config:expr, $storage:expr) => {
                adamw_step(
                    &mut model.$field.w,
                    &model.$field.dw,
                    &mut self.$field,
                    $config,
                    step,
                    $storage,
                );
            };
        }
        macro_rules! muon {
            ($field:ident, $id:literal) => {
                muon_step(
                    &mut model.$field.w,
                    &model.$field.dw,
                    &mut self.$field,
                    self.muon_config,
                    step,
                    master!($id),
                );
            };
        }
        macro_rules! master {
            ($id:literal) => {
                MasterStorage::Bf16 {
                    rounding,
                    parameter_id: $id,
                }
            };
        }

        adamw!(embedding, decay, master!(0));
        adamw!(attention_norm, no_decay, MasterStorage::Fp32);
        muon!(q_proj, 2);
        muon!(k_proj, 3);
        muon!(v_proj, 4);
        muon!(o_proj, 5);
        adamw!(ffn_norm, no_decay, MasterStorage::Fp32);
        muon!(gate_proj, 7);
        muon!(up_proj, 8);
        muon!(down_proj, 9);
        adamw!(final_norm, no_decay, MasterStorage::Fp32);
        adamw!(lm_head, decay, master!(11));
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

        adamw_step(
            &mut parameter,
            &gradient,
            &mut moments,
            config,
            1,
            MasterStorage::Fp32,
        );

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
    fn dense_visitor_reports_all_parameters_and_kinds() {
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

        let mut model = Dense::<4, 4, 7, 8, 2, 4, 12>::new(7);
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
    fn moe_visitor_keeps_router_out_of_hidden_matrices() {
        struct Inventory {
            routers: usize,
            matrices: usize,
        }

        impl CpuParameterVisitor for Inventory {
            fn visit<S: Shape>(
                &mut self,
                _name: &'static str,
                kind: ParameterKind,
                _parameter: &mut CpuTensor<f32, S>,
                _gradient: &CpuTensor<f32, S>,
            ) {
                self.routers += usize::from(kind == ParameterKind::Router);
                self.matrices += usize::from(kind == ParameterKind::Matrix);
            }
        }

        let mut model = MoeDense::<4, 4, 7, 8, 2, 4, 6, 3, 2, 3>::new(7, 0.01);
        let mut inventory = Inventory {
            routers: 0,
            matrices: 0,
        };
        model.visit_cpu_parameters(&mut inventory);

        assert_eq!(inventory.routers, 1);
        assert_eq!(inventory.matrices, 4 + 3 * 3);
    }

    #[test]
    fn aux_loss_schedule_decays_from_global_step() {
        let schedule = AuxLossSchedule {
            base_coefficient: 0.2,
            decay_horizon: 100.0,
        };
        assert_eq!(schedule.coefficient(0), 0.2);
        assert!((schedule.coefficient(25) - 0.15).abs() < 1e-7);
        assert_eq!(schedule.coefficient(100), 0.0);
        assert_eq!(schedule.coefficient(101), 0.0);
    }

    #[test]
    fn dense_muon_routes_hidden_matrices_and_auxiliary_parameters() {
        let mut model = Dense::<4, 4, 7, 8, 2, 4, 12>::new(7);
        model.embedding.dw.as_mut_slice().fill(1.0);
        model.attention_norm.dw.as_mut_slice().fill(1.0);
        model.q_proj.dw.as_mut_slice().fill(1.0);
        model.lm_head.dw.as_mut_slice().fill(1.0);
        let mut optimizer = DenseMuon::new(MuonConfig::default(), AdamWConfig::default());

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
    fn adamw_overfits_the_tiny_dense_batch() {
        type TinyDense = Dense<4, 4, 4, 8, 2, 4, 12>;
        let tokens = [0, 1, 2, 3];
        let targets = [1, 2, 3, 0];
        let mut model = TinyDense::new(100);
        let mut optimizer = DenseAdamW::new(AdamWConfig {
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
    fn muon_overfits_the_tiny_dense_batch() {
        type TinyDense = Dense<4, 4, 4, 8, 2, 4, 12>;
        let tokens = [0, 1, 2, 3];
        let targets = [1, 2, 3, 0];
        let mut model = TinyDense::new(100);
        let mut optimizer = DenseMuon::new(
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
