//! Full fp32 GPU forward and backward for the single-block reference Llama.
//!
//! Parameters, gradients, and saved activations remain GPU-resident. The
//! implementation mirrors `nn::Llama` explicitly so residual splits and the
//! ownership of every backward context stay visible.

use bench_util::{KernelProfiler, NoopProfiler};
use cuda_core::{CudaStream, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, kernel};
use cuda_host::cuda_module;
use nn::Llama;
use optim::AdamWConfig;
use tensor_core::{Rank1, Rank2, Rank3, Shape};

// cuda-oxide collects kernels from the selected binary target. The binary
// includes this file as a module, which in turn includes each canonical kernel
// source here instead of copying definitions or relying on dependency PTX.
#[path = "../../llama-ops/src/lib.rs"]
mod llama_device;
#[path = "../../tensor-gpu/src/lib.rs"]
#[allow(dead_code)]
pub mod tensor_device;

pub use llama_device::kernels as llama_kernels;
pub use tensor_device::kernels as tensor_kernels;
use tensor_device::{GpuAdamWMoments, GpuTensor};

pub mod checkpoint;

/// Static row-major `[D, 3 * D]` storage for horizontally packed Q/K/V weights.
///
/// A dedicated marker avoids `generic_const_exprs` in type position while
/// keeping the packed dimensions part of the tensor's type.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct QkvWeightShape<const D: usize>;

impl<const D: usize> Shape for QkvWeightShape<D> {
    const RANK: usize = 2;
    const NUM_ELEMENTS: usize = D * 3 * D;
    type Dims = [usize; 2];
    const DIMS: Self::Dims = [D, 3 * D];
}

/// Static row-major `[N, 3 * D]` storage for packed Q/K/V activations.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct QkvActivationShape<const N: usize, const D: usize>;

impl<const N: usize, const D: usize> Shape for QkvActivationShape<N, D> {
    const RANK: usize = 2;
    const NUM_ELEMENTS: usize = N * 3 * D;
    type Dims = [usize; 2];
    const DIMS: Self::Dims = [N, 3 * D];
}

#[cuda_module]
pub mod fusion_kernels {
    use super::*;

    /// Split row-major `[N, 3D]` into three row-major `[N, D]` tensors.
    ///
    /// # Safety
    ///
    /// `q`, `k`, and `v` must each contain exactly one third as many elements
    /// as `packed`, and `d` must be their non-zero row width. The launch must
    /// assign at most one thread to each output index.
    #[kernel]
    pub unsafe fn qkv_unpack(
        packed: &[f32],
        d: usize,
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        mut v: DisjointSlice<f32>,
    ) {
        let index = cuda_device::thread::index_1d().get();
        if index < q.len() {
            let row = index / d;
            let col = index % d;
            let packed_row = row * 3 * d;
            unsafe {
                *q.get_unchecked_mut(index) = packed[packed_row + col];
                *k.get_unchecked_mut(index) = packed[packed_row + d + col];
                *v.get_unchecked_mut(index) = packed[packed_row + 2 * d + col];
            }
        }
    }

    /// Pack three row-major `[N, D]` tensors into row-major `[N, 3D]`.
    ///
    /// # Safety
    ///
    /// `q`, `k`, and `v` must have equal lengths, `d` must be their non-zero
    /// row width, and `packed` must have three times their length. The launch
    /// must assign at most one thread to each input index.
    #[kernel]
    pub unsafe fn qkv_pack(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d: usize,
        mut packed: DisjointSlice<f32>,
    ) {
        let index = cuda_device::thread::index_1d().get();
        if index < q.len() {
            let row = index / d;
            let col = index % d;
            let packed_row = row * 3 * d;
            unsafe {
                *packed.get_unchecked_mut(packed_row + col) = q[index];
                *packed.get_unchecked_mut(packed_row + d + col) = k[index];
                *packed.get_unchecked_mut(packed_row + 2 * d + col) = v[index];
            }
        }
    }
}

fn pack_qkv_rows<const D: usize>(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    assert_eq!(q.len(), k.len());
    assert_eq!(q.len(), v.len());
    assert_eq!(q.len() % D, 0);
    let rows = q.len() / D;
    let mut packed = vec![0.0; rows * 3 * D];
    for row in 0..rows {
        let source = row * D;
        let destination = row * 3 * D;
        packed[destination..destination + D].copy_from_slice(&q[source..source + D]);
        packed[destination + D..destination + 2 * D].copy_from_slice(&k[source..source + D]);
        packed[destination + 2 * D..destination + 3 * D].copy_from_slice(&v[source..source + D]);
    }
    packed
}

fn unpack_qkv_rows<const D: usize>(packed: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(packed.len() % (3 * D), 0);
    let rows = packed.len() / (3 * D);
    let mut q = vec![0.0; rows * D];
    let mut k = vec![0.0; rows * D];
    let mut v = vec![0.0; rows * D];
    for row in 0..rows {
        let source = row * 3 * D;
        let destination = row * D;
        q[destination..destination + D].copy_from_slice(&packed[source..source + D]);
        k[destination..destination + D].copy_from_slice(&packed[source + D..source + 2 * D]);
        v[destination..destination + D].copy_from_slice(&packed[source + 2 * D..source + 3 * D]);
    }
    (q, k, v)
}

fn elementwise_config<S: Shape>() -> LaunchConfig {
    assert!(S::NUM_ELEMENTS <= u32::MAX as usize);
    LaunchConfig::for_num_elems(S::NUM_ELEMENTS as u32)
}

fn reduction_config() -> LaunchConfig {
    assert!(tensor_device::REDUCE_THREADS.is_power_of_two());
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (tensor_device::REDUCE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn gemm_config<const M: usize, const N: usize>() -> LaunchConfig {
    gemm_config_dims(M, N)
}

fn gemm_config_dims(m: usize, n: usize) -> LaunchConfig {
    assert!(tensor_device::TILE * tensor_device::TILE <= 1024);
    LaunchConfig {
        grid_dim: (
            (n as u32).div_ceil(tensor_device::TILE as u32),
            (m as u32).div_ceil(tensor_device::TILE as u32),
            1,
        ),
        block_dim: (tensor_device::TILE as u32, tensor_device::TILE as u32, 1),
        shared_mem_bytes: 0,
    }
}

fn add<S: Shape, P: KernelProfiler>(
    lhs: &GpuTensor<f32, S>,
    rhs: &GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<GpuTensor<f32, S>, DriverError> {
    let mut output = GpuTensor::zeros(stream)?;
    profiler.measure(stream, name, || {
        kernels.add(
            stream,
            elementwise_config::<S>(),
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })?;
    Ok(output)
}

fn add_scaled_assign<S: Shape, P: KernelProfiler>(
    dst: &mut GpuTensor<f32, S>,
    src: &GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.add_scaled(
            stream,
            elementwise_config::<S>(),
            src.as_device_buffer(),
            1.0,
            dst.as_device_buffer_mut(),
        )
    })
}

fn sum<S: Shape, P: KernelProfiler>(
    input: &GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<GpuTensor<f32, Rank1<1>>, DriverError> {
    let mut output = GpuTensor::zeros(stream)?;
    profiler.measure(stream, name, || {
        kernels.sum(
            stream,
            reduction_config(),
            input.as_device_buffer(),
            S::NUM_ELEMENTS as u32,
            output.as_device_buffer_mut(),
        )
    })?;
    Ok(output)
}

fn scale<S: Shape, P: KernelProfiler>(
    input: &GpuTensor<f32, S>,
    factor: f32,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<GpuTensor<f32, S>, DriverError> {
    let mut output = GpuTensor::zeros(stream)?;
    profiler.measure(stream, name, || {
        kernels.scale(
            stream,
            elementwise_config::<S>(),
            input.as_device_buffer(),
            factor,
            output.as_device_buffer_mut(),
        )
    })?;
    Ok(output)
}

fn gemm<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &GpuTensor<f32, Rank2<K, N>>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<GpuTensor<f32, Rank2<M, N>>, DriverError> {
    let mut output = GpuTensor::zeros(stream)?;
    profiler.measure(stream, name, || {
        kernels.gemm_tiled(
            stream,
            gemm_config::<M, N>(),
            M as u32,
            N as u32,
            K as u32,
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })?;
    Ok(output)
}

fn gemm_tn<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &GpuTensor<f32, Rank2<M, N>>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<GpuTensor<f32, Rank2<K, N>>, DriverError> {
    let mut output = GpuTensor::zeros(stream)?;
    profiler.measure(stream, name, || {
        kernels.gemm_tn(
            stream,
            gemm_config::<K, N>(),
            M as u32,
            N as u32,
            K as u32,
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })?;
    Ok(output)
}

fn gemm_nt<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &GpuTensor<f32, Rank2<N, K>>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<GpuTensor<f32, Rank2<M, N>>, DriverError> {
    let mut output = GpuTensor::zeros(stream)?;
    profiler.measure(stream, name, || {
        kernels.gemm_nt(
            stream,
            gemm_config::<M, N>(),
            M as u32,
            N as u32,
            K as u32,
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })?;
    Ok(output)
}

pub struct GpuLinear<const IN: usize, const OUT: usize> {
    pub w: GpuTensor<f32, Rank2<IN, OUT>>,
    pub dw: GpuTensor<f32, Rank2<IN, OUT>>,
}

impl<const IN: usize, const OUT: usize> GpuLinear<IN, OUT> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layer: &nn::Linear<N, IN, OUT>,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            w: GpuTensor::from_cpu(stream, &layer.w)?,
            dw: GpuTensor::zeros(stream)?,
        })
    }

    fn forward<const N: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<GpuTensor<f32, Rank2<N, OUT>>, DriverError> {
        gemm(x, &self.w, stream, kernels, profiler, name)
    }

    fn backward<const N: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        dy: &GpuTensor<f32, Rank2<N, OUT>>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
        profiler: &mut P,
        names: [&'static str; 3],
    ) -> Result<GpuTensor<f32, Rank2<N, IN>>, DriverError> {
        let dw = gemm_tn(x, dy, stream, kernels, profiler, names[0])?;
        add_scaled_assign(&mut self.dw, &dw, stream, kernels, profiler, names[1])?;
        gemm_nt(dy, &self.w, stream, kernels, profiler, names[2])
    }
}

/// Selects the retained three-GEMM oracle or the horizontally fused QKV path.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum QkvMode {
    Unfused,
    #[default]
    Fused,
}

enum GpuQkvStorage<const D: usize> {
    Unfused {
        q: GpuLinear<D, D>,
        k: GpuLinear<D, D>,
        v: GpuLinear<D, D>,
    },
    Fused {
        w: GpuTensor<f32, QkvWeightShape<D>>,
        dw: GpuTensor<f32, QkvWeightShape<D>>,
    },
}

/// Q/K/V projection with either the retained oracle layout or packed `[D,3D]`.
pub struct GpuQkvProjection<const D: usize> {
    storage: GpuQkvStorage<D>,
}

impl<const D: usize> GpuQkvProjection<D> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        q: &nn::Linear<N, D, D>,
        k: &nn::Linear<N, D, D>,
        v: &nn::Linear<N, D, D>,
        mode: QkvMode,
    ) -> Result<Self, DriverError> {
        let storage = match mode {
            QkvMode::Unfused => GpuQkvStorage::Unfused {
                q: GpuLinear::from_cpu(stream, q)?,
                k: GpuLinear::from_cpu(stream, k)?,
                v: GpuLinear::from_cpu(stream, v)?,
            },
            QkvMode::Fused => {
                let packed = pack_qkv_rows::<D>(q.w.as_slice(), k.w.as_slice(), v.w.as_slice());
                GpuQkvStorage::Fused {
                    w: GpuTensor::from_host(stream, &packed)?,
                    dw: GpuTensor::zeros(stream)?,
                }
            }
        };
        Ok(Self { storage })
    }

    fn forward<const N: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fusion: &fusion_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<
        (
            GpuTensor<f32, Rank2<N, D>>,
            GpuTensor<f32, Rank2<N, D>>,
            GpuTensor<f32, Rank2<N, D>>,
        ),
        DriverError,
    > {
        match &self.storage {
            GpuQkvStorage::Unfused { q, k, v } => Ok((
                q.forward(x, stream, tensor, profiler, "forward.q_proj.gemm")?,
                k.forward(x, stream, tensor, profiler, "forward.k_proj.gemm")?,
                v.forward(x, stream, tensor, profiler, "forward.v_proj.gemm")?,
            )),
            GpuQkvStorage::Fused { w, .. } => {
                let mut packed = GpuTensor::<f32, QkvActivationShape<N, D>>::zeros(stream)?;
                profiler.measure(stream, "forward.qkv.gemm", || {
                    tensor.gemm_tiled(
                        stream,
                        gemm_config_dims(N, 3 * D),
                        N as u32,
                        (3 * D) as u32,
                        D as u32,
                        x.as_device_buffer(),
                        w.as_device_buffer(),
                        packed.as_device_buffer_mut(),
                    )
                })?;

                let mut q = GpuTensor::zeros(stream)?;
                let mut k = GpuTensor::zeros(stream)?;
                let mut v = GpuTensor::zeros(stream)?;
                profiler.measure(stream, "forward.qkv.unpack", || unsafe {
                    fusion.qkv_unpack(
                        stream,
                        LaunchConfig::for_num_elems((N * D) as u32),
                        packed.as_device_buffer(),
                        D,
                        q.as_device_buffer_mut(),
                        k.as_device_buffer_mut(),
                        v.as_device_buffer_mut(),
                    )
                })?;
                Ok((q, k, v))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn backward<const N: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        dq: &GpuTensor<f32, Rank2<N, D>>,
        dk: &GpuTensor<f32, Rank2<N, D>>,
        dv: &GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fusion: &fusion_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<GpuTensor<f32, Rank2<N, D>>, DriverError> {
        match &mut self.storage {
            GpuQkvStorage::Unfused { q, k, v } => {
                let dq_input = q.backward(
                    x,
                    dq,
                    stream,
                    tensor,
                    profiler,
                    [
                        "backward.q_proj.weight_gemm",
                        "backward.q_proj.grad_accumulate",
                        "backward.q_proj.input_gemm",
                    ],
                )?;
                let dk_input = k.backward(
                    x,
                    dk,
                    stream,
                    tensor,
                    profiler,
                    [
                        "backward.k_proj.weight_gemm",
                        "backward.k_proj.grad_accumulate",
                        "backward.k_proj.input_gemm",
                    ],
                )?;
                let dv_input = v.backward(
                    x,
                    dv,
                    stream,
                    tensor,
                    profiler,
                    [
                        "backward.v_proj.weight_gemm",
                        "backward.v_proj.grad_accumulate",
                        "backward.v_proj.input_gemm",
                    ],
                )?;
                let dqk = add(
                    &dq_input,
                    &dk_input,
                    stream,
                    tensor,
                    profiler,
                    "backward.qk_projection_sum",
                )?;
                add(
                    &dqk,
                    &dv_input,
                    stream,
                    tensor,
                    profiler,
                    "backward.qkv_projection_sum",
                )
            }
            GpuQkvStorage::Fused { w, dw } => {
                let mut packed = GpuTensor::<f32, QkvActivationShape<N, D>>::zeros(stream)?;
                profiler.measure(stream, "backward.qkv.pack", || unsafe {
                    fusion.qkv_pack(
                        stream,
                        LaunchConfig::for_num_elems((N * D) as u32),
                        dq.as_device_buffer(),
                        dk.as_device_buffer(),
                        dv.as_device_buffer(),
                        D,
                        packed.as_device_buffer_mut(),
                    )
                })?;

                let mut gradient = GpuTensor::<f32, QkvWeightShape<D>>::zeros(stream)?;
                profiler.measure(stream, "backward.qkv.weight_gemm", || {
                    tensor.gemm_tn(
                        stream,
                        gemm_config_dims(D, 3 * D),
                        N as u32,
                        (3 * D) as u32,
                        D as u32,
                        x.as_device_buffer(),
                        packed.as_device_buffer(),
                        gradient.as_device_buffer_mut(),
                    )
                })?;
                add_scaled_assign(
                    dw,
                    &gradient,
                    stream,
                    tensor,
                    profiler,
                    "backward.qkv.grad_accumulate",
                )?;

                let mut dx = GpuTensor::zeros(stream)?;
                profiler.measure(stream, "backward.qkv.input_gemm", || {
                    tensor.gemm_nt(
                        stream,
                        gemm_config_dims(N, D),
                        N as u32,
                        D as u32,
                        (3 * D) as u32,
                        packed.as_device_buffer(),
                        w.as_device_buffer(),
                        dx.as_device_buffer_mut(),
                    )
                })?;
                Ok(dx)
            }
        }
    }

    pub fn weights_to_host(
        &self,
        stream: &CudaStream,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), DriverError> {
        match &self.storage {
            GpuQkvStorage::Unfused { q, k, v } => Ok((
                q.w.to_host(stream)?,
                k.w.to_host(stream)?,
                v.w.to_host(stream)?,
            )),
            GpuQkvStorage::Fused { w, .. } => Ok(unpack_qkv_rows::<D>(&w.to_host(stream)?)),
        }
    }

    pub fn gradients_to_host(
        &self,
        stream: &CudaStream,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), DriverError> {
        match &self.storage {
            GpuQkvStorage::Unfused { q, k, v } => Ok((
                q.dw.to_host(stream)?,
                k.dw.to_host(stream)?,
                v.dw.to_host(stream)?,
            )),
            GpuQkvStorage::Fused { dw, .. } => Ok(unpack_qkv_rows::<D>(&dw.to_host(stream)?)),
        }
    }

    fn replace_weights(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        stream: &CudaStream,
    ) -> Result<(), DriverError> {
        match &mut self.storage {
            GpuQkvStorage::Unfused {
                q: q_projection,
                k: k_projection,
                v: v_projection,
            } => {
                q_projection.w = GpuTensor::from_host(stream, q)?;
                k_projection.w = GpuTensor::from_host(stream, k)?;
                v_projection.w = GpuTensor::from_host(stream, v)?;
            }
            GpuQkvStorage::Fused { w, .. } => {
                *w = GpuTensor::from_host(stream, &pack_qkv_rows::<D>(q, k, v))?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        match &mut self.storage {
            GpuQkvStorage::Unfused { q, k, v } => {
                q.dw = GpuTensor::zeros(stream)?;
                k.dw = GpuTensor::zeros(stream)?;
                v.dw = GpuTensor::zeros(stream)?;
            }
            GpuQkvStorage::Fused { dw, .. } => *dw = GpuTensor::zeros(stream)?,
        }
        Ok(())
    }
}

pub struct GpuRmsNorm<const D: usize> {
    pub w: GpuTensor<f32, Rank1<D>>,
    pub dw: GpuTensor<f32, Rank1<D>>,
    eps: f32,
}

impl<const D: usize> GpuRmsNorm<D> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layer: &nn::RmsNorm<N, D>,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            w: GpuTensor::from_cpu(stream, &layer.w)?,
            dw: GpuTensor::zeros(stream)?,
            eps: layer.eps,
        })
    }

    fn forward<const N: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<GpuTensor<f32, Rank2<N, D>>, DriverError> {
        let mut y = GpuTensor::zeros(stream)?;
        profiler.measure(stream, name, || {
            kernels.rms_norm_forward(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                x.as_device_buffer(),
                self.w.as_device_buffer(),
                self.eps,
                D as u32,
                y.as_device_buffer_mut(),
            )
        })?;
        Ok(y)
    }

    fn backward<const N: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        dy: &GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<GpuTensor<f32, Rank2<N, D>>, DriverError> {
        let mut dx = GpuTensor::zeros(stream)?;
        profiler.measure(stream, names[0], || {
            kernels.rms_norm_backward_x(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                x.as_device_buffer(),
                self.w.as_device_buffer(),
                dy.as_device_buffer(),
                self.eps,
                D as u32,
                dx.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, names[1], || {
            kernels.rms_norm_backward_weight(
                stream,
                LaunchConfig::for_num_elems(D as u32),
                x.as_device_buffer(),
                dy.as_device_buffer(),
                self.eps,
                N as u32,
                D as u32,
                self.dw.as_device_buffer_mut(),
            )
        })?;
        Ok(dx)
    }
}

pub struct GpuEmbedding<const VOCAB: usize, const D: usize> {
    pub w: GpuTensor<f32, Rank2<VOCAB, D>>,
    pub dw: GpuTensor<f32, Rank2<VOCAB, D>>,
}

impl<const VOCAB: usize, const D: usize> GpuEmbedding<VOCAB, D> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layer: &nn::Embedding<N, VOCAB, D>,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            w: GpuTensor::from_cpu(stream, &layer.w)?,
            dw: GpuTensor::zeros(stream)?,
        })
    }

    fn forward<const N: usize, P: KernelProfiler>(
        &self,
        tokens: &GpuTensor<u32, Rank1<N>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<GpuTensor<f32, Rank2<N, D>>, DriverError> {
        let mut y = GpuTensor::zeros(stream)?;
        profiler.measure(stream, name, || {
            kernels.embedding_forward(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                self.w.as_device_buffer(),
                tokens.as_device_buffer(),
                D as u32,
                y.as_device_buffer_mut(),
            )
        })?;
        Ok(y)
    }

    fn backward<const N: usize, P: KernelProfiler>(
        &mut self,
        tokens: &GpuTensor<u32, Rank1<N>>,
        dy: &GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || {
            kernels.embedding_backward(
                stream,
                LaunchConfig::for_num_elems((VOCAB * D) as u32),
                tokens.as_device_buffer(),
                dy.as_device_buffer(),
                N as u32,
                D as u32,
                self.dw.as_device_buffer_mut(),
            )
        })
    }
}

pub struct GpuLlama<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> {
    pub embedding: GpuEmbedding<VOCAB, D>,
    pub attention_norm: GpuRmsNorm<D>,
    pub qkv: GpuQkvProjection<D>,
    pub o_proj: GpuLinear<D, D>,
    pub ffn_norm: GpuRmsNorm<D>,
    pub gate_proj: GpuLinear<D, FF>,
    pub up_proj: GpuLinear<D, FF>,
    pub down_proj: GpuLinear<FF, D>,
    pub final_norm: GpuRmsNorm<D>,
    pub lm_head: GpuLinear<D, VOCAB>,
}

enum GpuQkvAdamW<const D: usize> {
    Unfused {
        q: GpuAdamWMoments<Rank2<D, D>>,
        k: GpuAdamWMoments<Rank2<D, D>>,
        v: GpuAdamWMoments<Rank2<D, D>>,
    },
    Fused(GpuAdamWMoments<QkvWeightShape<D>>),
}

impl<const D: usize> GpuQkvAdamW<D> {
    fn zeros(stream: &CudaStream, mode: QkvMode) -> Result<Self, DriverError> {
        match mode {
            QkvMode::Unfused => Ok(Self::Unfused {
                q: GpuAdamWMoments::zeros(stream)?,
                k: GpuAdamWMoments::zeros(stream)?,
                v: GpuAdamWMoments::zeros(stream)?,
            }),
            QkvMode::Fused => Ok(Self::Fused(GpuAdamWMoments::zeros(stream)?)),
        }
    }

    pub(crate) fn moments_to_host(
        &self,
        stream: &CudaStream,
    ) -> Result<
        (
            (Vec<f32>, Vec<f32>, Vec<f32>),
            (Vec<f32>, Vec<f32>, Vec<f32>),
        ),
        DriverError,
    > {
        match self {
            Self::Unfused { q, k, v } => Ok((
                (
                    q.first.to_host(stream)?,
                    k.first.to_host(stream)?,
                    v.first.to_host(stream)?,
                ),
                (
                    q.second.to_host(stream)?,
                    k.second.to_host(stream)?,
                    v.second.to_host(stream)?,
                ),
            )),
            Self::Fused(moments) => Ok((
                unpack_qkv_rows::<D>(&moments.first.to_host(stream)?),
                unpack_qkv_rows::<D>(&moments.second.to_host(stream)?),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn replace_moments(
        &mut self,
        q_first: &[f32],
        k_first: &[f32],
        v_first: &[f32],
        q_second: &[f32],
        k_second: &[f32],
        v_second: &[f32],
        stream: &CudaStream,
    ) -> Result<(), DriverError> {
        match self {
            Self::Unfused { q, k, v } => {
                q.first = GpuTensor::from_host(stream, q_first)?;
                k.first = GpuTensor::from_host(stream, k_first)?;
                v.first = GpuTensor::from_host(stream, v_first)?;
                q.second = GpuTensor::from_host(stream, q_second)?;
                k.second = GpuTensor::from_host(stream, k_second)?;
                v.second = GpuTensor::from_host(stream, v_second)?;
            }
            Self::Fused(moments) => {
                moments.first =
                    GpuTensor::from_host(stream, &pack_qkv_rows::<D>(q_first, k_first, v_first))?;
                moments.second = GpuTensor::from_host(
                    stream,
                    &pack_qkv_rows::<D>(q_second, k_second, v_second),
                )?;
            }
        }
        Ok(())
    }
}

/// GPU-resident AdamW state mirroring every model parameter.
pub struct GpuLlamaAdamW<const VOCAB: usize, const D: usize, const FF: usize> {
    config: AdamWConfig,
    step: u64,
    pub embedding: GpuAdamWMoments<Rank2<VOCAB, D>>,
    pub attention_norm: GpuAdamWMoments<Rank1<D>>,
    qkv: GpuQkvAdamW<D>,
    pub o_proj: GpuAdamWMoments<Rank2<D, D>>,
    pub ffn_norm: GpuAdamWMoments<Rank1<D>>,
    pub gate_proj: GpuAdamWMoments<Rank2<D, FF>>,
    pub up_proj: GpuAdamWMoments<Rank2<D, FF>>,
    pub down_proj: GpuAdamWMoments<Rank2<FF, D>>,
    pub final_norm: GpuAdamWMoments<Rank1<D>>,
    pub lm_head: GpuAdamWMoments<Rank2<D, VOCAB>>,
}

impl<const VOCAB: usize, const D: usize, const FF: usize> GpuLlamaAdamW<VOCAB, D, FF> {
    pub fn new(stream: &CudaStream, config: AdamWConfig) -> Result<Self, DriverError> {
        Self::new_with_qkv_mode(stream, config, QkvMode::Fused)
    }

    pub fn new_with_qkv_mode(
        stream: &CudaStream,
        config: AdamWConfig,
        qkv_mode: QkvMode,
    ) -> Result<Self, DriverError> {
        config.validate();
        Ok(Self {
            config,
            step: 0,
            embedding: GpuAdamWMoments::zeros(stream)?,
            attention_norm: GpuAdamWMoments::zeros(stream)?,
            qkv: GpuQkvAdamW::zeros(stream, qkv_mode)?,
            o_proj: GpuAdamWMoments::zeros(stream)?,
            ffn_norm: GpuAdamWMoments::zeros(stream)?,
            gate_proj: GpuAdamWMoments::zeros(stream)?,
            up_proj: GpuAdamWMoments::zeros(stream)?,
            down_proj: GpuAdamWMoments::zeros(stream)?,
            final_norm: GpuAdamWMoments::zeros(stream)?,
            lm_head: GpuAdamWMoments::zeros(stream)?,
        })
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    pub(crate) fn restore_step(&mut self, step: u64) {
        self.step = step;
    }

    pub fn update<const N: usize, const T: usize, const H: usize, const HD: usize>(
        &mut self,
        model: &mut GpuLlama<N, T, VOCAB, D, H, HD, FF>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.update_profiled(model, stream, kernels, &mut profiler)
    }

    pub fn update_profiled<
        const N: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        P: KernelProfiler,
    >(
        &mut self,
        model: &mut GpuLlama<N, T, VOCAB, D, H, HD, FF>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        self.step = self.step.checked_add(1).expect("AdamW step overflow");
        let (first_correction, second_correction) = self.config.bias_correction(self.step);

        macro_rules! update {
            ($field:ident, $weight_decay:expr) => {
                profiler.measure(
                    stream,
                    concat!("optimizer.", stringify!($field), ".adamw"),
                    || {
                        model.$field.w.adamw_step(
                            &model.$field.dw,
                            &mut self.$field,
                            self.config.learning_rate,
                            self.config.beta1,
                            self.config.beta2,
                            self.config.epsilon,
                            $weight_decay,
                            first_correction,
                            second_correction,
                            stream,
                            kernels,
                        )
                    },
                )?;
            };
        }

        update!(embedding, self.config.weight_decay);
        update!(attention_norm, 0.0);
        match (&mut model.qkv.storage, &mut self.qkv) {
            (
                GpuQkvStorage::Unfused { q, k, v },
                GpuQkvAdamW::Unfused {
                    q: q_moments,
                    k: k_moments,
                    v: v_moments,
                },
            ) => {
                macro_rules! update_projection {
                    ($name:literal, $projection:expr, $moments:expr) => {
                        profiler.measure(stream, $name, || {
                            $projection.w.adamw_step(
                                &$projection.dw,
                                $moments,
                                self.config.learning_rate,
                                self.config.beta1,
                                self.config.beta2,
                                self.config.epsilon,
                                self.config.weight_decay,
                                first_correction,
                                second_correction,
                                stream,
                                kernels,
                            )
                        })?;
                    };
                }
                update_projection!("optimizer.q_proj.adamw", q, q_moments);
                update_projection!("optimizer.k_proj.adamw", k, k_moments);
                update_projection!("optimizer.v_proj.adamw", v, v_moments);
            }
            (GpuQkvStorage::Fused { w, dw }, GpuQkvAdamW::Fused(moments)) => {
                profiler.measure(stream, "optimizer.qkv.adamw", || {
                    w.adamw_step(
                        dw,
                        moments,
                        self.config.learning_rate,
                        self.config.beta1,
                        self.config.beta2,
                        self.config.epsilon,
                        self.config.weight_decay,
                        first_correction,
                        second_correction,
                        stream,
                        kernels,
                    )
                })?;
            }
            _ => panic!("QKV projection and optimizer modes must match"),
        }
        update!(o_proj, self.config.weight_decay);
        update!(ffn_norm, 0.0);
        update!(gate_proj, self.config.weight_decay);
        update!(up_proj, self.config.weight_decay);
        update!(down_proj, self.config.weight_decay);
        update!(final_norm, 0.0);
        update!(lm_head, self.config.weight_decay);
        Ok(())
    }
}

pub struct GpuLlamaCtx<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
> {
    tokens: GpuTensor<u32, Rank1<N>>,
    targets: GpuTensor<u32, Rank1<N>>,
    attention_input: GpuTensor<f32, Rank2<N, D>>,
    attention_normalized: GpuTensor<f32, Rank2<N, D>>,
    q: GpuTensor<f32, Rank2<N, D>>,
    k: GpuTensor<f32, Rank2<N, D>>,
    v: GpuTensor<f32, Rank2<N, D>>,
    probabilities: GpuTensor<f32, Rank3<N, H, T>>,
    attended: GpuTensor<f32, Rank2<N, D>>,
    ffn_input: GpuTensor<f32, Rank2<N, D>>,
    ffn_normalized: GpuTensor<f32, Rank2<N, D>>,
    gate: GpuTensor<f32, Rank2<N, FF>>,
    up: GpuTensor<f32, Rank2<N, FF>>,
    activated: GpuTensor<f32, Rank2<N, FF>>,
    final_input: GpuTensor<f32, Rank2<N, D>>,
    final_normalized: GpuTensor<f32, Rank2<N, D>>,
    loss_probabilities: GpuTensor<f32, Rank2<N, VOCAB>>,
}

impl<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> GpuLlama<N, T, VOCAB, D, H, HD, FF>
{
    pub fn from_cpu(
        stream: &CudaStream,
        model: &Llama<N, T, VOCAB, D, H, HD, FF>,
    ) -> Result<Self, DriverError> {
        Self::from_cpu_with_qkv_mode(stream, model, QkvMode::Fused)
    }

    pub fn from_cpu_with_qkv_mode(
        stream: &CudaStream,
        model: &Llama<N, T, VOCAB, D, H, HD, FF>,
        qkv_mode: QkvMode,
    ) -> Result<Self, DriverError> {
        assert!(N <= u32::MAX as usize);
        assert!(N * H * T <= u32::MAX as usize);
        assert!(D > 0);
        assert!(D <= u32::MAX as usize / 3);
        assert_eq!(N % T, 0);
        assert_eq!(D, H * HD);
        Ok(Self {
            embedding: GpuEmbedding::from_cpu(stream, &model.embedding)?,
            attention_norm: GpuRmsNorm::from_cpu(stream, &model.attention_norm)?,
            qkv: GpuQkvProjection::from_cpu(
                stream,
                &model.q_proj,
                &model.k_proj,
                &model.v_proj,
                qkv_mode,
            )?,
            o_proj: GpuLinear::from_cpu(stream, &model.o_proj)?,
            ffn_norm: GpuRmsNorm::from_cpu(stream, &model.ffn_norm)?,
            gate_proj: GpuLinear::from_cpu(stream, &model.gate_proj)?,
            up_proj: GpuLinear::from_cpu(stream, &model.up_proj)?,
            down_proj: GpuLinear::from_cpu(stream, &model.down_proj)?,
            final_norm: GpuRmsNorm::from_cpu(stream, &model.final_norm)?,
            lm_head: GpuLinear::from_cpu(stream, &model.lm_head)?,
        })
    }

    pub fn forward(
        &self,
        tokens: [usize; N],
        targets: [usize; N],
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        fusion: &fusion_kernels::LoadedModule,
    ) -> Result<(GpuTensor<f32, Rank1<1>>, GpuLlamaCtx<N, T, VOCAB, D, H, FF>), DriverError> {
        let mut profiler = NoopProfiler;
        self.forward_profiled(
            tokens,
            targets,
            stream,
            tensor,
            llama,
            fusion,
            &mut profiler,
        )
    }

    pub fn forward_profiled<P: KernelProfiler>(
        &self,
        tokens: [usize; N],
        targets: [usize; N],
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        fusion: &fusion_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(GpuTensor<f32, Rank1<1>>, GpuLlamaCtx<N, T, VOCAB, D, H, FF>), DriverError> {
        let token_u32 = tokens.map(|token| {
            assert!(token < VOCAB);
            token as u32
        });
        let target_u32 = targets.map(|target| {
            assert!(target < VOCAB);
            target as u32
        });
        let tokens = GpuTensor::from_host(stream, &token_u32)?;
        let targets = GpuTensor::from_host(stream, &target_u32)?;
        let attention_input =
            self.embedding
                .forward(&tokens, stream, llama, profiler, "forward.embedding")?;
        let attention_normalized = self.attention_norm.forward(
            &attention_input,
            stream,
            llama,
            profiler,
            "forward.attention_norm",
        )?;
        let (q, k, v) =
            self.qkv
                .forward(&attention_normalized, stream, tensor, fusion, profiler)?;
        let q = rope::<N, T, D, H, HD, P>(&q, false, stream, llama, profiler, "forward.q_rope")?;
        let k = rope::<N, T, D, H, HD, P>(&k, false, stream, llama, profiler, "forward.k_rope")?;
        let (attended, probabilities) =
            attention_forward::<N, T, D, H, HD, P>(&q, &k, &v, stream, llama, profiler)?;
        let attention_output =
            self.o_proj
                .forward(&attended, stream, tensor, profiler, "forward.o_proj.gemm")?;
        let ffn_input = add(
            &attention_input,
            &attention_output,
            stream,
            tensor,
            profiler,
            "forward.attention_residual",
        )?;

        let ffn_normalized =
            self.ffn_norm
                .forward(&ffn_input, stream, llama, profiler, "forward.ffn_norm")?;
        let gate = self.gate_proj.forward(
            &ffn_normalized,
            stream,
            tensor,
            profiler,
            "forward.gate_proj.gemm",
        )?;
        let up = self.up_proj.forward(
            &ffn_normalized,
            stream,
            tensor,
            profiler,
            "forward.up_proj.gemm",
        )?;
        let activated = swiglu(&gate, &up, stream, llama, profiler, "forward.swiglu")?;
        let ffn_output = self.down_proj.forward(
            &activated,
            stream,
            tensor,
            profiler,
            "forward.down_proj.gemm",
        )?;
        let final_input = add(
            &ffn_input,
            &ffn_output,
            stream,
            tensor,
            profiler,
            "forward.ffn_residual",
        )?;

        let final_normalized =
            self.final_norm
                .forward(&final_input, stream, llama, profiler, "forward.final_norm")?;
        let logits = self.lm_head.forward(
            &final_normalized,
            stream,
            tensor,
            profiler,
            "forward.lm_head.gemm",
        )?;
        let (loss, loss_probabilities) =
            cross_entropy(&logits, &targets, stream, tensor, llama, profiler)?;
        Ok((
            loss,
            GpuLlamaCtx {
                tokens,
                targets,
                attention_input,
                attention_normalized,
                q,
                k,
                v,
                probabilities,
                attended,
                ffn_input,
                ffn_normalized,
                gate,
                up,
                activated,
                final_input,
                final_normalized,
                loss_probabilities,
            },
        ))
    }

    pub fn backward(
        &mut self,
        ctx: GpuLlamaCtx<N, T, VOCAB, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        fusion: &fusion_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.backward_profiled(ctx, stream, tensor, llama, fusion, &mut profiler)
    }

    pub fn backward_profiled<P: KernelProfiler>(
        &mut self,
        ctx: GpuLlamaCtx<N, T, VOCAB, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        fusion: &fusion_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        let dlogits = cross_entropy_backward(
            &ctx.loss_probabilities,
            &ctx.targets,
            stream,
            llama,
            profiler,
        )?;
        let dx = self.lm_head.backward(
            &ctx.final_normalized,
            &dlogits,
            stream,
            tensor,
            profiler,
            [
                "backward.lm_head.weight_gemm",
                "backward.lm_head.grad_accumulate",
                "backward.lm_head.input_gemm",
            ],
        )?;
        let dx = self.final_norm.backward(
            &ctx.final_input,
            &dx,
            stream,
            llama,
            profiler,
            ["backward.final_norm.input", "backward.final_norm.weight"],
        )?;

        let dactivated = self.down_proj.backward(
            &ctx.activated,
            &dx,
            stream,
            tensor,
            profiler,
            [
                "backward.down_proj.weight_gemm",
                "backward.down_proj.grad_accumulate",
                "backward.down_proj.input_gemm",
            ],
        )?;
        let (dgate, dup) =
            swiglu_backward(&ctx.gate, &ctx.up, &dactivated, stream, llama, profiler)?;
        let dgate_input = self.gate_proj.backward(
            &ctx.ffn_normalized,
            &dgate,
            stream,
            tensor,
            profiler,
            [
                "backward.gate_proj.weight_gemm",
                "backward.gate_proj.grad_accumulate",
                "backward.gate_proj.input_gemm",
            ],
        )?;
        let dup_input = self.up_proj.backward(
            &ctx.ffn_normalized,
            &dup,
            stream,
            tensor,
            profiler,
            [
                "backward.up_proj.weight_gemm",
                "backward.up_proj.grad_accumulate",
                "backward.up_proj.input_gemm",
            ],
        )?;
        let dnormalized = add(
            &dgate_input,
            &dup_input,
            stream,
            tensor,
            profiler,
            "backward.ffn_projection_sum",
        )?;
        let dffn_input = self.ffn_norm.backward(
            &ctx.ffn_input,
            &dnormalized,
            stream,
            llama,
            profiler,
            ["backward.ffn_norm.input", "backward.ffn_norm.weight"],
        )?;
        let dx = add(
            &dx,
            &dffn_input,
            stream,
            tensor,
            profiler,
            "backward.ffn_residual",
        )?;

        let dattended = self.o_proj.backward(
            &ctx.attended,
            &dx,
            stream,
            tensor,
            profiler,
            [
                "backward.o_proj.weight_gemm",
                "backward.o_proj.grad_accumulate",
                "backward.o_proj.input_gemm",
            ],
        )?;
        let (dq, dk, dv) = attention_backward::<N, T, D, H, HD, P>(
            &ctx.q,
            &ctx.k,
            &ctx.v,
            &ctx.probabilities,
            &dattended,
            stream,
            llama,
            profiler,
        )?;
        let dq = rope::<N, T, D, H, HD, P>(&dq, true, stream, llama, profiler, "backward.q_rope")?;
        let dk = rope::<N, T, D, H, HD, P>(&dk, true, stream, llama, profiler, "backward.k_rope")?;
        let dnormalized = self.qkv.backward(
            &ctx.attention_normalized,
            &dq,
            &dk,
            &dv,
            stream,
            tensor,
            fusion,
            profiler,
        )?;
        let dattn_input = self.attention_norm.backward(
            &ctx.attention_input,
            &dnormalized,
            stream,
            llama,
            profiler,
            [
                "backward.attention_norm.input",
                "backward.attention_norm.weight",
            ],
        )?;
        let dx = add(
            &dx,
            &dattn_input,
            stream,
            tensor,
            profiler,
            "backward.attention_residual",
        )?;
        self.embedding.backward(
            &ctx.tokens,
            &dx,
            stream,
            llama,
            profiler,
            "backward.embedding",
        )
    }

    pub fn zero_grad(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        self.embedding.dw = GpuTensor::zeros(stream)?;
        self.attention_norm.dw = GpuTensor::zeros(stream)?;
        self.qkv.zero_grad(stream)?;
        self.o_proj.dw = GpuTensor::zeros(stream)?;
        self.ffn_norm.dw = GpuTensor::zeros(stream)?;
        self.gate_proj.dw = GpuTensor::zeros(stream)?;
        self.up_proj.dw = GpuTensor::zeros(stream)?;
        self.down_proj.dw = GpuTensor::zeros(stream)?;
        self.final_norm.dw = GpuTensor::zeros(stream)?;
        self.lm_head.dw = GpuTensor::zeros(stream)?;
        Ok(())
    }
}

fn rope<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    x: &GpuTensor<f32, Rank2<N, D>>,
    backward: bool,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<GpuTensor<f32, Rank2<N, D>>, DriverError> {
    let mut y = GpuTensor::zeros(stream)?;
    if backward {
        profiler.measure(stream, name, || {
            kernels.rope_backward(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                x.as_device_buffer(),
                T as u32,
                H as u32,
                HD as u32,
                y.as_device_buffer_mut(),
            )
        })?;
    } else {
        profiler.measure(stream, name, || {
            kernels.rope_forward(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                x.as_device_buffer(),
                T as u32,
                H as u32,
                HD as u32,
                y.as_device_buffer_mut(),
            )
        })?;
    }
    Ok(y)
}

fn attention_forward<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    q: &GpuTensor<f32, Rank2<N, D>>,
    k: &GpuTensor<f32, Rank2<N, D>>,
    v: &GpuTensor<f32, Rank2<N, D>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(GpuTensor<f32, Rank2<N, D>>, GpuTensor<f32, Rank3<N, H, T>>), DriverError> {
    let mut probabilities = GpuTensor::zeros(stream)?;
    let mut output = GpuTensor::zeros(stream)?;
    profiler.measure(stream, "forward.attention.probabilities", || {
        kernels.attention_probabilities(
            stream,
            LaunchConfig::for_num_elems((N * H * T) as u32),
            q.as_device_buffer(),
            k.as_device_buffer(),
            T as u32,
            H as u32,
            HD as u32,
            probabilities.as_device_buffer_mut(),
        )
    })?;
    profiler.measure(stream, "forward.attention.output", || {
        kernels.attention_output(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            probabilities.as_device_buffer(),
            v.as_device_buffer(),
            T as u32,
            H as u32,
            HD as u32,
            output.as_device_buffer_mut(),
        )
    })?;
    Ok((output, probabilities))
}

fn attention_backward<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    q: &GpuTensor<f32, Rank2<N, D>>,
    k: &GpuTensor<f32, Rank2<N, D>>,
    v: &GpuTensor<f32, Rank2<N, D>>,
    probabilities: &GpuTensor<f32, Rank3<N, H, T>>,
    dy: &GpuTensor<f32, Rank2<N, D>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<
    (
        GpuTensor<f32, Rank2<N, D>>,
        GpuTensor<f32, Rank2<N, D>>,
        GpuTensor<f32, Rank2<N, D>>,
    ),
    DriverError,
> {
    let mut dq = GpuTensor::zeros(stream)?;
    let mut dk = GpuTensor::zeros(stream)?;
    let mut dv = GpuTensor::zeros(stream)?;
    let config = LaunchConfig::for_num_elems((N * D) as u32);
    profiler.measure(stream, "backward.attention.q", || {
        kernels.attention_backward_q(
            stream,
            config,
            q.as_device_buffer(),
            k.as_device_buffer(),
            v.as_device_buffer(),
            probabilities.as_device_buffer(),
            dy.as_device_buffer(),
            T as u32,
            H as u32,
            HD as u32,
            dq.as_device_buffer_mut(),
        )
    })?;
    profiler.measure(stream, "backward.attention.k", || {
        kernels.attention_backward_k(
            stream,
            config,
            q.as_device_buffer(),
            v.as_device_buffer(),
            probabilities.as_device_buffer(),
            dy.as_device_buffer(),
            T as u32,
            H as u32,
            HD as u32,
            dk.as_device_buffer_mut(),
        )
    })?;
    profiler.measure(stream, "backward.attention.v", || {
        kernels.attention_backward_v(
            stream,
            config,
            probabilities.as_device_buffer(),
            dy.as_device_buffer(),
            T as u32,
            H as u32,
            HD as u32,
            dv.as_device_buffer_mut(),
        )
    })?;
    Ok((dq, dk, dv))
}

fn swiglu<const N: usize, const FF: usize, P: KernelProfiler>(
    gate: &GpuTensor<f32, Rank2<N, FF>>,
    up: &GpuTensor<f32, Rank2<N, FF>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<GpuTensor<f32, Rank2<N, FF>>, DriverError> {
    let mut output = GpuTensor::zeros(stream)?;
    profiler.measure(stream, name, || {
        kernels.swiglu_forward(
            stream,
            LaunchConfig::for_num_elems((N * FF) as u32),
            gate.as_device_buffer(),
            up.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })?;
    Ok(output)
}

fn swiglu_backward<const N: usize, const FF: usize, P: KernelProfiler>(
    gate: &GpuTensor<f32, Rank2<N, FF>>,
    up: &GpuTensor<f32, Rank2<N, FF>>,
    dy: &GpuTensor<f32, Rank2<N, FF>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(GpuTensor<f32, Rank2<N, FF>>, GpuTensor<f32, Rank2<N, FF>>), DriverError> {
    let mut dgate = GpuTensor::zeros(stream)?;
    let mut dup = GpuTensor::zeros(stream)?;
    let config = LaunchConfig::for_num_elems((N * FF) as u32);
    profiler.measure(stream, "backward.swiglu.gate", || {
        kernels.swiglu_backward_gate(
            stream,
            config,
            gate.as_device_buffer(),
            up.as_device_buffer(),
            dy.as_device_buffer(),
            dgate.as_device_buffer_mut(),
        )
    })?;
    profiler.measure(stream, "backward.swiglu.up", || {
        kernels.swiglu_backward_up(
            stream,
            config,
            gate.as_device_buffer(),
            dy.as_device_buffer(),
            dup.as_device_buffer_mut(),
        )
    })?;
    Ok((dgate, dup))
}

fn cross_entropy<const N: usize, const VOCAB: usize, P: KernelProfiler>(
    logits: &GpuTensor<f32, Rank2<N, VOCAB>>,
    targets: &GpuTensor<u32, Rank1<N>>,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    llama: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(GpuTensor<f32, Rank1<1>>, GpuTensor<f32, Rank2<N, VOCAB>>), DriverError> {
    let mut probabilities = GpuTensor::zeros(stream)?;
    let mut losses = GpuTensor::<f32, Rank1<N>>::zeros(stream)?;
    profiler.measure(stream, "forward.loss.softmax", || {
        llama.softmax_forward(
            stream,
            LaunchConfig::for_num_elems((N * VOCAB) as u32),
            logits.as_device_buffer(),
            VOCAB as u32,
            probabilities.as_device_buffer_mut(),
        )
    })?;
    profiler.measure(stream, "forward.loss.cross_entropy", || {
        llama.cross_entropy_loss(
            stream,
            LaunchConfig::for_num_elems(N as u32),
            logits.as_device_buffer(),
            targets.as_device_buffer(),
            N as u32,
            VOCAB as u32,
            losses.as_device_buffer_mut(),
        )
    })?;
    let loss_sum = sum(&losses, stream, tensor, profiler, "forward.loss.reduction")?;
    let loss = scale(
        &loss_sum,
        1.0 / N as f32,
        stream,
        tensor,
        profiler,
        "forward.loss.mean",
    )?;
    Ok((loss, probabilities))
}

fn cross_entropy_backward<const N: usize, const VOCAB: usize, P: KernelProfiler>(
    probabilities: &GpuTensor<f32, Rank2<N, VOCAB>>,
    targets: &GpuTensor<u32, Rank1<N>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<GpuTensor<f32, Rank2<N, VOCAB>>, DriverError> {
    let mut dlogits = GpuTensor::zeros(stream)?;
    profiler.measure(stream, "backward.loss.softmax_cross_entropy", || {
        kernels.softmax_cross_entropy_backward(
            stream,
            LaunchConfig::for_num_elems((N * VOCAB) as u32),
            probabilities.as_device_buffer(),
            targets.as_device_buffer(),
            1.0,
            N as u32,
            VOCAB as u32,
            dlogits.as_device_buffer_mut(),
        )
    })?;
    Ok(dlogits)
}
