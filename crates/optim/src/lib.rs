//! CPU reference plumbing shared by mixed-precision optimizers.
//!
//! bf16 is a compute/storage format, not the source of truth for parameter
//! updates. [`Fp32MasterWeights`] retains every update in fp32 and refreshes a
//! shape-identical bf16 compute tensor after the update. AdamW and Muon can
//! build their optimizer-specific state and update rules on top of this type.

use tensor_core::{Shape, bf16};
use tensor_cpu::CpuTensor;

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
