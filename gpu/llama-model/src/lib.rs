//! GPU forward and backward for the single-block reference Llama: fp32
//! everywhere except the lm-head, which runs bf16 tcgen05 GEMMs against fp32
//! master weights (§7 phase 2, adopted head-first per the 7e5 profile).
//!
//! Parameters, gradients, and saved activations remain GPU-resident. The
//! implementation mirrors `nn::Llama` explicitly so residual splits and the
//! aliasing story stay visible. Since 7e2, activations and scratch live in a
//! persistent `GpuLlamaWorkspace` reused across steps; safety comes from
//! disjoint workspace fields (each saved activation has a dedicated buffer),
//! not from the CPU reference's by-value Ctx ownership.
//!
//! Two padded dimensions keep every head GEMM inside the tcgen05 tile
//! contract without touching the tuned kernel:
//! - `VP` pads the vocabulary (50,257 -> 50,304). The padded weight columns
//!   are zero at initialization and stay zero: the classifier backward writes
//!   exact zeros there, so their gradients, moments, and decayed masters never
//!   move. Checkpoints store the unpadded columns only.
//! - `NP` pads the token rows to the 128-row tile. Padded rows of the head
//!   input are zeroed once and never written, so they contribute exactly
//!   nothing to any product (including the `K = NP` weight-gradient GEMM).

use std::error::Error;

use bench_util::{KernelProfiler, NoopProfiler};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, DriverError, LaunchConfig, PinnedHostBuffer};
use nn::Llama;
use optim::AdamWConfig;
use tensor_core::{Rank1, Rank2, Rank3, Shape, bf16};

// cuda-oxide collects kernels from the selected binary target. The binary
// includes this file as a module, which in turn includes each canonical kernel
// source here instead of copying definitions or relying on dependency PTX.
//
// The tcgen05 GEMM kernels are the one exception: this binary's kernels use
// libdevice math (`exp`/`ln`/`sqrt`), which forces its device artifact
// through libNVVM, and libNVVM rejects tcgen05 lowerings. Only gpu/gemm's
// host-side support is included here; the kernels themselves load at runtime
// from the pure-PTX `gemm.ptx` that `cargo oxide build gemm` produces
// (modal_app.py prebuilds it for llama-model runs).
#[path = "../../flash-attn/src/lib.rs"]
mod flash_device;
#[path = "../../gemm/src/fp32.rs"]
mod gemm_device;
#[path = "../../gemm/src/host.rs"]
#[allow(dead_code)]
mod gemm_host;
#[path = "../../llama-ops/src/lib.rs"]
mod llama_device;
#[path = "../../tensor-gpu/src/lib.rs"]
#[allow(dead_code)]
pub mod tensor_device;

pub use flash_device::kernels as flash_kernels;
pub use gemm_device::kernels as gemm_kernels;
pub use gemm_host::Tcgen05Gemm;
pub use llama_device::kernels as llama_kernels;
pub use tensor_device::kernels as tensor_kernels;

use gemm_device::launch_config as fp32_launch_config;
use gemm_host::{Bf16PairsTmaMap, TC_TILE, create_bf16_pairs_tma_map, tcgen05_launch_config};
use tensor_device::{GpuAdamWMoments, GpuTensor, transpose_pairs_config};

pub mod checkpoint;

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

fn classifier_config<const N: usize>() -> LaunchConfig {
    assert!(llama_device::CLASSIFIER_THREADS.is_power_of_two());
    assert!(N <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (llama_device::CLASSIFIER_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn norm_config<const N: usize>() -> LaunchConfig {
    assert!(llama_device::NORM_THREADS.is_power_of_two());
    assert!(N <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (llama_device::NORM_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn norm_weight_config<const N: usize, const D: usize>() -> LaunchConfig {
    let threads = llama_device::NORM_THREADS;
    let rows_per_block = llama_device::NORM_WEIGHT_ROWS_PER_BLOCK;
    assert!(threads.is_power_of_two());
    assert!(N <= u32::MAX as usize && D <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            D.div_ceil(threads) as u32,
            N.div_ceil(rows_per_block) as u32,
            1,
        ),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn pairs_config(words: usize) -> LaunchConfig {
    assert!(words <= u32::MAX as usize);
    LaunchConfig::for_num_elems(words as u32)
}

fn flash_forward_config<const N: usize, const T: usize, const H: usize, const HD: usize>()
-> LaunchConfig {
    assert_eq!(N % T, 0);
    flash_device::tiled_forward_config(N / T, T, H, HD)
}

fn flash_dot_config<const N: usize, const H: usize, const HD: usize>() -> LaunchConfig {
    flash_device::dot_config(N, H, HD)
}

fn flash_backward_q_config<const N: usize, const T: usize, const H: usize, const HD: usize>()
-> LaunchConfig {
    assert_eq!(N % T, 0);
    flash_device::tiled_backward_q_config(N / T, T, H, HD)
}

fn flash_backward_kv_config<const N: usize, const T: usize, const H: usize, const HD: usize>()
-> LaunchConfig {
    assert_eq!(N % T, 0);
    flash_device::tiled_backward_kv_config(N / T, T, H, HD)
}

fn add_into<S: Shape, P: KernelProfiler>(
    lhs: &GpuTensor<f32, S>,
    rhs: &GpuTensor<f32, S>,
    output: &mut GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.add(
            stream,
            elementwise_config::<S>(),
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })
}

fn fill_zero<S: Shape, P: KernelProfiler>(
    output: &mut GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.fill(
            stream,
            elementwise_config::<S>(),
            0.0,
            output.as_device_buffer_mut(),
        )
    })
}

fn sum_into<S: Shape, P: KernelProfiler>(
    input: &GpuTensor<f32, S>,
    output: &mut GpuTensor<f32, Rank1<1>>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.sum(
            stream,
            reduction_config(),
            input.as_device_buffer(),
            S::NUM_ELEMENTS as u32,
            output.as_device_buffer_mut(),
        )
    })
}

fn scale_into<S: Shape, P: KernelProfiler>(
    input: &GpuTensor<f32, S>,
    factor: f32,
    output: &mut GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.scale(
            stream,
            elementwise_config::<S>(),
            input.as_device_buffer(),
            factor,
            output.as_device_buffer_mut(),
        )
    })
}

fn gemm_into<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &GpuTensor<f32, Rank2<K, N>>,
    output: &mut GpuTensor<f32, Rank2<M, N>>,
    stream: &CudaStream,
    kernels: &gemm_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || unsafe {
        kernels.register_gemm_store(
            stream,
            fp32_launch_config(M, N),
            M,
            N,
            K,
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })
}

fn gemm_tn_accumulate_into<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &GpuTensor<f32, Rank2<M, N>>,
    output: &mut GpuTensor<f32, Rank2<K, N>>,
    stream: &CudaStream,
    kernels: &gemm_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || unsafe {
        kernels.register_gemm_tn_accumulate(
            stream,
            fp32_launch_config(K, N),
            K,
            N,
            M,
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })
}

fn gemm_nt_into<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &GpuTensor<f32, Rank2<N, K>>,
    output: &mut GpuTensor<f32, Rank2<M, N>>,
    stream: &CudaStream,
    kernels: &gemm_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || unsafe {
        kernels.register_gemm_nt_store(
            stream,
            fp32_launch_config(M, N),
            M,
            N,
            K,
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })
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

    fn forward_into<const N: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        output: &mut GpuTensor<f32, Rank2<N, OUT>>,
        stream: &CudaStream,
        kernels: &gemm_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        gemm_into(x, &self.w, output, stream, kernels, profiler, name)
    }

    fn backward_into<const N: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        dy: &GpuTensor<f32, Rank2<N, OUT>>,
        dx: &mut GpuTensor<f32, Rank2<N, IN>>,
        stream: &CudaStream,
        kernels: &gemm_kernels::LoadedModule,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<(), DriverError> {
        gemm_tn_accumulate_into(x, dy, &mut self.dw, stream, kernels, profiler, names[0])?;
        gemm_nt_into(dy, &self.w, dx, stream, kernels, profiler, names[1])
    }
}

pub struct GpuGroupedLinear<const IN: usize, const GROUPS: usize, const OUT: usize> {
    pub w: GpuTensor<f32, Rank3<IN, GROUPS, OUT>>,
    pub dw: GpuTensor<f32, Rank3<IN, GROUPS, OUT>>,
}

impl<const IN: usize, const GROUPS: usize, const OUT: usize> GpuGroupedLinear<IN, GROUPS, OUT> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layers: [&nn::Linear<N, IN, OUT>; GROUPS],
    ) -> Result<Self, DriverError> {
        let mut weights = vec![0.0; IN * GROUPS * OUT];
        for input in 0..IN {
            for (group, layer) in layers.iter().enumerate() {
                let source = &layer.w.as_slice()[input * OUT..(input + 1) * OUT];
                let destination = (input * GROUPS + group) * OUT;
                weights[destination..destination + OUT].copy_from_slice(source);
            }
        }
        Ok(Self {
            w: GpuTensor::from_host(stream, &weights)?,
            dw: GpuTensor::zeros(stream)?,
        })
    }

    fn forward_into<const N: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        output: &mut GpuTensor<f32, Rank3<N, GROUPS, OUT>>,
        stream: &CudaStream,
        kernels: &gemm_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.register_gemm_store(
                stream,
                fp32_launch_config(N, GROUPS * OUT),
                N,
                GROUPS * OUT,
                IN,
                x.as_device_buffer(),
                self.w.as_device_buffer(),
                output.as_device_buffer_mut(),
            )
        })
    }

    fn backward_into<const N: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        dy: &GpuTensor<f32, Rank3<N, GROUPS, OUT>>,
        dx: &mut GpuTensor<f32, Rank2<N, IN>>,
        stream: &CudaStream,
        kernels: &gemm_kernels::LoadedModule,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<(), DriverError> {
        profiler.measure(stream, names[0], || unsafe {
            kernels.register_gemm_tn_accumulate(
                stream,
                fp32_launch_config(IN, GROUPS * OUT),
                IN,
                GROUPS * OUT,
                N,
                x.as_device_buffer(),
                dy.as_device_buffer(),
                self.dw.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, names[1], || unsafe {
            kernels.register_gemm_nt_store(
                stream,
                fp32_launch_config(N, IN),
                N,
                IN,
                GROUPS * OUT,
                dy.as_device_buffer(),
                self.w.as_device_buffer(),
                dx.as_device_buffer_mut(),
            )
        })
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

    fn forward_into<const N: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        y: &mut GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || {
            kernels.rms_norm_forward_fast(
                stream,
                norm_config::<N>(),
                x.as_device_buffer(),
                self.w.as_device_buffer(),
                self.eps,
                D as u32,
                y.as_device_buffer_mut(),
            )
        })
    }

    fn backward_into<const N: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        dy: &GpuTensor<f32, Rank2<N, D>>,
        dx: &mut GpuTensor<f32, Rank2<N, D>>,
        inv: &mut GpuTensor<f32, Rank1<N>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<(), DriverError> {
        profiler.measure(stream, names[0], || {
            kernels.rms_norm_backward_x_fast(
                stream,
                norm_config::<N>(),
                x.as_device_buffer(),
                self.w.as_device_buffer(),
                dy.as_device_buffer(),
                self.eps,
                D as u32,
                dx.as_device_buffer_mut(),
                inv.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, names[1], || unsafe {
            kernels.rms_norm_backward_weight_fast(
                stream,
                norm_weight_config::<N, D>(),
                x.as_device_buffer(),
                dy.as_device_buffer(),
                inv.as_device_buffer(),
                N as u32,
                D as u32,
                self.dw.as_device_buffer_mut(),
            )
        })
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

    fn forward_into<const N: usize, P: KernelProfiler>(
        &self,
        tokens: &GpuTensor<u32, Rank1<N>>,
        y: &mut GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || {
            kernels.embedding_forward(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                self.w.as_device_buffer(),
                tokens.as_device_buffer(),
                D as u32,
                y.as_device_buffer_mut(),
            )
        })
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
        profiler.measure(stream, name, || unsafe {
            kernels.embedding_backward_scatter(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                tokens.as_device_buffer(),
                dy.as_device_buffer(),
                D as u32,
                self.dw.as_device_buffer_mut(),
            )
        })
    }
}

/// bf16 lm-head with fp32 master weights (§7 phase 2).
///
/// The compute weight exists in both layouts the tcgen05 `C = A B^T` form
/// needs as a K-contiguous `B` operand: `w` is `[D, VP]` (consumed by the
/// input-gradient GEMM) and `w_t` is `[VP, D]` (consumed by the forward
/// GEMM). `master` is the fp32 source of truth; the optimizer updates it and
/// re-rounds both compute copies every step. `dw` accumulates in packed bf16,
/// produced directly by the tcgen05 accumulate epilogue.
pub struct GpuBf16Head<const D: usize, const VP: usize> {
    pub master: GpuTensor<f32, Rank2<D, VP>>,
    w: DeviceBuffer<u32>,
    w_t: DeviceBuffer<u32>,
    dw: DeviceBuffer<u32>,
    w_tma: Bf16PairsTmaMap,
    w_t_tma: Bf16PairsTmaMap,
}

impl<const D: usize, const VP: usize> GpuBf16Head<D, VP> {
    fn from_cpu<const N: usize, const VOCAB: usize>(
        stream: &CudaStream,
        layer: &nn::Linear<N, D, VOCAB>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(VP >= VOCAB);
        let mut padded = vec![0.0f32; D * VP];
        for row in 0..D {
            padded[row * VP..row * VP + VOCAB]
                .copy_from_slice(&layer.w.as_slice()[row * VOCAB..(row + 1) * VOCAB]);
        }
        Self::from_master_values(stream, &padded)
    }

    /// Rebuild the head from padded `[D, VP]` fp32 master values, rounding
    /// both packed compute copies on the host.
    pub(crate) fn from_master_values(
        stream: &CudaStream,
        values: &[f32],
    ) -> Result<Self, Box<dyn Error>> {
        assert_eq!(values.len(), D * VP);
        let pack = |low: f32, high: f32| {
            bf16::from_f32(low).to_bits() as u32 | ((bf16::from_f32(high).to_bits() as u32) << 16)
        };
        let compute: Vec<u32> = values
            .chunks_exact(2)
            .map(|pair| pack(pair[0], pair[1]))
            .collect();
        let mut transposed = vec![0u32; VP * D / 2];
        for column in 0..VP {
            for pair in 0..D / 2 {
                transposed[column * D / 2 + pair] = pack(
                    values[2 * pair * VP + column],
                    values[(2 * pair + 1) * VP + column],
                );
            }
        }

        let master = GpuTensor::from_host(stream, values)?;
        let w = DeviceBuffer::from_host(stream, &compute)?;
        let w_t = DeviceBuffer::from_host(stream, &transposed)?;
        let dw = DeviceBuffer::zeroed(stream, D * VP / 2)?;
        // SAFETY: `w` and `w_t` live in this struct beside their maps and are
        // never reallocated.
        let w_tma = unsafe { create_bf16_pairs_tma_map(stream, &w, VP, D)? };
        let w_t_tma = unsafe { create_bf16_pairs_tma_map(stream, &w_t, D, VP)? };
        Ok(Self {
            master,
            w,
            w_t,
            dw,
            w_tma,
            w_t_tma,
        })
    }

    /// Packed-bf16 weight gradient. Parity-test accessor: binaries other than
    /// the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn dw_words(&self) -> &DeviceBuffer<u32> {
        &self.dw
    }

    /// Packed-bf16 `[D, VP]` compute weights. Parity-test accessor: binaries
    /// other than the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn w_words(&self) -> &DeviceBuffer<u32> {
        &self.w
    }

    /// Packed-bf16 `[VP, D]` transposed compute weights. Parity-test accessor:
    /// binaries other than the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn w_t_words(&self) -> &DeviceBuffer<u32> {
        &self.w_t
    }

    fn zero_grad<P: KernelProfiler>(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || {
            kernels.fill_u32(stream, pairs_config(D * VP / 2), 0, &mut self.dw)
        })
    }

    fn forward_into<const NP: usize, P: KernelProfiler>(
        &self,
        x_tma: &Bf16PairsTmaMap,
        logits: &mut DeviceBuffer<u32>,
        stream: &CudaStream,
        kernels: &Tcgen05Gemm,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.store(
                stream,
                tcgen05_launch_config(NP, VP, D),
                x_tma.as_ptr(),
                self.w_t_tma.as_ptr(),
                logits,
                VP as u32,
                D as u32,
            )
        })
    }

    /// `dw += x^T dlogits` from the transposed operands staged in the
    /// workspace; padded token rows and vocabulary columns contribute zeros.
    fn backward_weight<const NP: usize, P: KernelProfiler>(
        &mut self,
        x_t_tma: &Bf16PairsTmaMap,
        dlogits_t_tma: &Bf16PairsTmaMap,
        stream: &CudaStream,
        kernels: &Tcgen05Gemm,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.accumulate(
                stream,
                tcgen05_launch_config(D, VP, NP),
                x_t_tma.as_ptr(),
                dlogits_t_tma.as_ptr(),
                &mut self.dw,
                VP as u32,
                NP as u32,
            )
        })
    }

    fn backward_input<const NP: usize, P: KernelProfiler>(
        &self,
        dlogits_tma: &Bf16PairsTmaMap,
        dx: &mut DeviceBuffer<u32>,
        stream: &CudaStream,
        kernels: &Tcgen05Gemm,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.store(
                stream,
                tcgen05_launch_config(NP, D, VP),
                dlogits_tma.as_ptr(),
                self.w_tma.as_ptr(),
                dx,
                D as u32,
                VP as u32,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        &mut self,
        moments: &mut GpuAdamWMoments<Rank2<D, VP>>,
        config: AdamWConfig,
        first_correction: f32,
        second_correction: f32,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        kernels.adamw_master_bf16(
            stream,
            pairs_config(D * VP / 2),
            &self.dw,
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.epsilon,
            config.weight_decay,
            first_correction,
            second_correction,
            self.master.as_device_buffer_mut(),
            moments.first.as_device_buffer_mut(),
            moments.second.as_device_buffer_mut(),
            &mut self.w,
        )
    }

    /// Refresh `w_t` from `w` after an optimizer step.
    fn sync_transposed(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        unsafe {
            kernels.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(D, VP),
                &self.w,
                D as u32,
                VP as u32,
                &mut self.w_t,
            )
        }
    }
}

pub struct GpuLlama<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> {
    pub embedding: GpuEmbedding<VOCAB, D>,
    pub attention_norm: GpuRmsNorm<D>,
    pub qkv_proj: GpuGroupedLinear<D, 3, D>,
    pub o_proj: GpuLinear<D, D>,
    pub ffn_norm: GpuRmsNorm<D>,
    pub gate_up_proj: GpuGroupedLinear<D, 2, FF>,
    pub down_proj: GpuLinear<FF, D>,
    pub final_norm: GpuRmsNorm<D>,
    pub lm_head: GpuBf16Head<D, VP>,
}

/// GPU-resident AdamW state mirroring every model parameter.
///
/// The lm-head moments span the padded `[D, VP]` master; padded columns hold
/// zeros forever because their gradients are exactly zero.
pub struct GpuLlamaAdamW<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize> {
    config: AdamWConfig,
    step: u64,
    pub embedding: GpuAdamWMoments<Rank2<VOCAB, D>>,
    pub attention_norm: GpuAdamWMoments<Rank1<D>>,
    pub qkv_proj: GpuAdamWMoments<Rank3<D, 3, D>>,
    pub o_proj: GpuAdamWMoments<Rank2<D, D>>,
    pub ffn_norm: GpuAdamWMoments<Rank1<D>>,
    pub gate_up_proj: GpuAdamWMoments<Rank3<D, 2, FF>>,
    pub down_proj: GpuAdamWMoments<Rank2<FF, D>>,
    pub final_norm: GpuAdamWMoments<Rank1<D>>,
    pub lm_head: GpuAdamWMoments<Rank2<D, VP>>,
}

impl<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize>
    GpuLlamaAdamW<VOCAB, VP, D, FF>
{
    pub fn new(stream: &CudaStream, config: AdamWConfig) -> Result<Self, DriverError> {
        config.validate();
        Ok(Self {
            config,
            step: 0,
            embedding: GpuAdamWMoments::zeros(stream)?,
            attention_norm: GpuAdamWMoments::zeros(stream)?,
            qkv_proj: GpuAdamWMoments::zeros(stream)?,
            o_proj: GpuAdamWMoments::zeros(stream)?,
            ffn_norm: GpuAdamWMoments::zeros(stream)?,
            gate_up_proj: GpuAdamWMoments::zeros(stream)?,
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

    pub fn update<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
    >(
        &mut self,
        model: &mut GpuLlama<N, NP, T, VOCAB, VP, D, H, HD, FF>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.update_profiled(model, stream, kernels, &mut profiler)
    }

    pub fn update_profiled<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        P: KernelProfiler,
    >(
        &mut self,
        model: &mut GpuLlama<N, NP, T, VOCAB, VP, D, H, HD, FF>,
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
        update!(qkv_proj, self.config.weight_decay);
        update!(o_proj, self.config.weight_decay);
        update!(ffn_norm, 0.0);
        update!(gate_up_proj, self.config.weight_decay);
        update!(down_proj, self.config.weight_decay);
        update!(final_norm, 0.0);
        profiler.measure(stream, "optimizer.lm_head.adamw", || {
            model.lm_head.adamw_step(
                &mut self.lm_head,
                self.config,
                first_correction,
                second_correction,
                stream,
                kernels,
            )
        })?;
        profiler.measure(stream, "optimizer.lm_head.sync_w_t", || {
            model.lm_head.sync_transposed(stream, kernels)
        })?;
        Ok(())
    }
}

struct InputStaging<const N: usize> {
    tokens: PinnedHostBuffer<u32>,
    targets: PinnedHostBuffer<u32>,
    copied: CudaEvent,
    pending: bool,
}

impl<const N: usize> InputStaging<N> {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            tokens: PinnedHostBuffer::zeroed(stream.context(), N)?,
            targets: PinnedHostBuffer::zeroed(stream.context(), N)?,
            copied: stream.context().new_event(None)?,
            pending: false,
        })
    }
}

/// Persistent device and pinned-host storage for one model's training steps.
///
/// Create this once and pass it to every forward/backward call. All operator
/// outputs are written into these allocations, so a steady-state step performs
/// no device allocation or synchronous device free.
///
/// The packed lm-head buffers are `NP` rows tall. Rows `N..NP` of
/// `head_input` are zeroed at allocation and never written afterwards
/// (`convert_f32_to_bf16_pairs` stops at the input length), and the same rows
/// of `logits` always hold zeros: the forward GEMM computes them from the
/// zero input rows and the classifier backward never touches them.
pub struct GpuLlamaWorkspace<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
> {
    tokens: GpuTensor<u32, Rank1<N>>,
    targets: GpuTensor<u32, Rank1<N>>,
    staging: [InputStaging<N>; 2],
    next_staging: usize,
    attention_input: GpuTensor<f32, Rank2<N, D>>,
    attention_normalized: GpuTensor<f32, Rank2<N, D>>,
    qkv: GpuTensor<f32, Rank3<N, 3, D>>,
    q: GpuTensor<f32, Rank2<N, D>>,
    k: GpuTensor<f32, Rank2<N, D>>,
    v: GpuTensor<f32, Rank2<N, D>>,
    attended: GpuTensor<f32, Rank2<N, D>>,
    attention_logsumexp: GpuTensor<f32, Rank2<N, H>>,
    attention_dot: GpuTensor<f32, Rank2<N, H>>,
    ffn_input: GpuTensor<f32, Rank2<N, D>>,
    ffn_normalized: GpuTensor<f32, Rank2<N, D>>,
    gate_up: GpuTensor<f32, Rank3<N, 2, FF>>,
    gate: GpuTensor<f32, Rank2<N, FF>>,
    up: GpuTensor<f32, Rank2<N, FF>>,
    activated: GpuTensor<f32, Rank2<N, FF>>,
    final_input: GpuTensor<f32, Rank2<N, D>>,
    final_normalized: GpuTensor<f32, Rank2<N, D>>,
    projection_output: GpuTensor<f32, Rank2<N, D>>,
    head_input: DeviceBuffer<u32>,
    head_input_t: DeviceBuffer<u32>,
    logits: DeviceBuffer<u32>,
    dlogits_t: DeviceBuffer<u32>,
    d_head_input: DeviceBuffer<u32>,
    head_input_tma: Bf16PairsTmaMap,
    head_input_t_tma: Bf16PairsTmaMap,
    logits_tma: Bf16PairsTmaMap,
    dlogits_t_tma: Bf16PairsTmaMap,
    norm_backward_inv: GpuTensor<f32, Rank1<N>>,
    losses: GpuTensor<f32, Rank1<N>>,
    loss_sum: GpuTensor<f32, Rank1<1>>,
    loss: GpuTensor<f32, Rank1<1>>,
    d_model_0: GpuTensor<f32, Rank2<N, D>>,
    d_model_1: GpuTensor<f32, Rank2<N, D>>,
    d_model_2: GpuTensor<f32, Rank2<N, D>>,
    d_model_3: GpuTensor<f32, Rank2<N, D>>,
    d_model_4: GpuTensor<f32, Rank2<N, D>>,
    d_ff_0: GpuTensor<f32, Rank2<N, FF>>,
    d_ff_1: GpuTensor<f32, Rank2<N, FF>>,
    d_ff_2: GpuTensor<f32, Rank2<N, FF>>,
}

impl<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
> GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF>
{
    pub fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        let head_input = DeviceBuffer::zeroed(stream, NP * D / 2)?;
        let head_input_t = DeviceBuffer::zeroed(stream, D * NP / 2)?;
        let logits = DeviceBuffer::zeroed(stream, NP * VP / 2)?;
        let dlogits_t = DeviceBuffer::zeroed(stream, VP * NP / 2)?;
        // SAFETY: the mapped buffers live in this workspace beside their maps
        // and are never reallocated.
        let head_input_tma = unsafe { create_bf16_pairs_tma_map(stream, &head_input, D, NP)? };
        let head_input_t_tma = unsafe { create_bf16_pairs_tma_map(stream, &head_input_t, NP, D)? };
        let logits_tma = unsafe { create_bf16_pairs_tma_map(stream, &logits, VP, NP)? };
        let dlogits_t_tma = unsafe { create_bf16_pairs_tma_map(stream, &dlogits_t, NP, VP)? };
        Ok(Self {
            tokens: GpuTensor::zeros(stream)?,
            targets: GpuTensor::zeros(stream)?,
            staging: [InputStaging::new(stream)?, InputStaging::new(stream)?],
            next_staging: 0,
            attention_input: GpuTensor::zeros(stream)?,
            attention_normalized: GpuTensor::zeros(stream)?,
            qkv: GpuTensor::zeros(stream)?,
            q: GpuTensor::zeros(stream)?,
            k: GpuTensor::zeros(stream)?,
            v: GpuTensor::zeros(stream)?,
            attended: GpuTensor::zeros(stream)?,
            attention_logsumexp: GpuTensor::zeros(stream)?,
            attention_dot: GpuTensor::zeros(stream)?,
            ffn_input: GpuTensor::zeros(stream)?,
            ffn_normalized: GpuTensor::zeros(stream)?,
            gate_up: GpuTensor::zeros(stream)?,
            gate: GpuTensor::zeros(stream)?,
            up: GpuTensor::zeros(stream)?,
            activated: GpuTensor::zeros(stream)?,
            final_input: GpuTensor::zeros(stream)?,
            final_normalized: GpuTensor::zeros(stream)?,
            projection_output: GpuTensor::zeros(stream)?,
            head_input,
            head_input_t,
            logits,
            dlogits_t,
            d_head_input: DeviceBuffer::zeroed(stream, NP * D / 2)?,
            head_input_tma,
            head_input_t_tma,
            logits_tma,
            dlogits_t_tma,
            norm_backward_inv: GpuTensor::zeros(stream)?,
            losses: GpuTensor::zeros(stream)?,
            loss_sum: GpuTensor::zeros(stream)?,
            loss: GpuTensor::zeros(stream)?,
            d_model_0: GpuTensor::zeros(stream)?,
            d_model_1: GpuTensor::zeros(stream)?,
            d_model_2: GpuTensor::zeros(stream)?,
            d_model_3: GpuTensor::zeros(stream)?,
            d_model_4: GpuTensor::zeros(stream)?,
            d_ff_0: GpuTensor::zeros(stream)?,
            d_ff_1: GpuTensor::zeros(stream)?,
            d_ff_2: GpuTensor::zeros(stream)?,
        })
    }

    /// Packed-bf16 logits (dlogits after a backward pass). Parity-test
    /// accessor: binaries other than the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn logits_words(&self) -> &DeviceBuffer<u32> {
        &self.logits
    }

    pub fn loss(&self) -> &GpuTensor<f32, Rank1<1>> {
        &self.loss
    }

    /// Host readback of one packed-bf16 logits row, widened to f32.
    ///
    /// Sampling and debugging only: this synchronizes the stream after copying
    /// only the requested row.
    pub fn logits_row(&self, row: usize, stream: &CudaStream) -> Result<Vec<f32>, DriverError> {
        assert!(row < NP);
        let stride = VP / 2;
        let byte_offset = row
            .checked_mul(stride)
            .and_then(|offset| offset.checked_mul(std::mem::size_of::<u32>()))
            .expect("logits row byte offset overflow");
        let source = self
            .logits
            .cu_deviceptr()
            .checked_add(byte_offset as u64)
            .expect("logits row device pointer overflow");
        let mut words = vec![0u32; stride];
        // SAFETY: `row < NP` and the workspace allocation of `NP * VP / 2`
        // words guarantee that `source` has `words.len()` readable elements.
        // The initialized host vector remains live until stream synchronization.
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                words.as_mut_ptr(),
                source,
                std::mem::size_of_val(words.as_slice()),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        Ok(words
            .iter()
            .flat_map(|&word| {
                [
                    f32::from_bits((word & 0xFFFF) << 16),
                    f32::from_bits((word >> 16) << 16),
                ]
            })
            .collect())
    }

    fn upload_inputs(
        &mut self,
        tokens: &[usize; N],
        targets: &[usize; N],
        stream: &CudaStream,
    ) -> Result<(), DriverError> {
        let slot = &mut self.staging[self.next_staging];
        if slot.pending {
            slot.copied.synchronize()?;
        }
        for i in 0..N {
            assert!(tokens[i] < VOCAB);
            assert!(targets[i] < VOCAB);
            slot.tokens[i] = tokens[i] as u32;
            slot.targets[i] = targets[i] as u32;
        }

        // SAFETY: the staging slot remains owned by this workspace and is not
        // read, mutated, or dropped until `copied` has synchronized before its
        // next reuse. The event is recorded after both copies on this stream.
        unsafe {
            self.tokens
                .as_device_buffer_mut()
                .copy_from_pinned_host_async(stream, &slot.tokens)?;
            self.targets
                .as_device_buffer_mut()
                .copy_from_pinned_host_async(stream, &slot.targets)?;
        }
        slot.copied.record(stream)?;
        slot.pending = true;
        self.next_staging ^= 1;
        Ok(())
    }
}

impl<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> GpuLlama<N, NP, T, VOCAB, VP, D, H, HD, FF>
{
    pub fn from_cpu(
        stream: &CudaStream,
        model: &Llama<N, T, VOCAB, D, H, HD, FF>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(N <= u32::MAX as usize);
        assert_eq!(N % T, 0);
        assert_eq!(D, H * HD);
        // tcgen05 head contract: padded tokens and vocabulary are tile
        // multiples, and D serves as both an output dimension (input-gradient
        // N, weight-gradient M) and a reduction width.
        assert_eq!(NP, N.next_multiple_of(TC_TILE));
        assert!(VP >= VOCAB);
        assert_eq!(VP % TC_TILE, 0);
        assert_eq!(D % TC_TILE, 0);
        Ok(Self {
            embedding: GpuEmbedding::from_cpu(stream, &model.embedding)?,
            attention_norm: GpuRmsNorm::from_cpu(stream, &model.attention_norm)?,
            qkv_proj: GpuGroupedLinear::from_cpu(
                stream,
                [&model.q_proj, &model.k_proj, &model.v_proj],
            )?,
            o_proj: GpuLinear::from_cpu(stream, &model.o_proj)?,
            ffn_norm: GpuRmsNorm::from_cpu(stream, &model.ffn_norm)?,
            gate_up_proj: GpuGroupedLinear::from_cpu(stream, [&model.gate_proj, &model.up_proj])?,
            down_proj: GpuLinear::from_cpu(stream, &model.down_proj)?,
            final_norm: GpuRmsNorm::from_cpu(stream, &model.final_norm)?,
            lm_head: GpuBf16Head::from_cpu(stream, &model.lm_head)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: &[usize; N],
        targets: &[usize; N],
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.forward_profiled(
            tokens,
            targets,
            workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            llama,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_profiled<P: KernelProfiler>(
        &self,
        tokens: &[usize; N],
        targets: &[usize; N],
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        workspace.upload_inputs(tokens, targets, stream)?;
        self.embedding.forward_into(
            &workspace.tokens,
            &mut workspace.attention_input,
            stream,
            llama,
            profiler,
            "forward.embedding",
        )?;
        self.attention_norm.forward_into(
            &workspace.attention_input,
            &mut workspace.attention_normalized,
            stream,
            llama,
            profiler,
            "forward.attention_norm",
        )?;
        self.qkv_proj.forward_into(
            &workspace.attention_normalized,
            &mut workspace.qkv,
            stream,
            gemm,
            profiler,
            "forward.qkv_proj.gemm",
        )?;
        profiler.measure(stream, "forward.qkv_proj.split", || {
            llama.split_group3(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                workspace.qkv.as_device_buffer(),
                D as u32,
                workspace.q.as_device_buffer_mut(),
                workspace.k.as_device_buffer_mut(),
                workspace.v.as_device_buffer_mut(),
            )
        })?;
        rope_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &mut workspace.d_model_0,
            false,
            stream,
            llama,
            profiler,
            "forward.q_rope",
        )?;
        std::mem::swap(&mut workspace.q, &mut workspace.d_model_0);
        rope_into::<N, T, D, H, HD, P>(
            &workspace.k,
            &mut workspace.d_model_0,
            false,
            stream,
            llama,
            profiler,
            "forward.k_rope",
        )?;
        std::mem::swap(&mut workspace.k, &mut workspace.d_model_0);
        flash_attention_forward_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &mut workspace.attended,
            &mut workspace.attention_logsumexp,
            stream,
            flash,
            profiler,
        )?;
        self.o_proj.forward_into(
            &workspace.attended,
            &mut workspace.projection_output,
            stream,
            gemm,
            profiler,
            "forward.o_proj.gemm",
        )?;
        add_into(
            &workspace.attention_input,
            &workspace.projection_output,
            &mut workspace.ffn_input,
            stream,
            tensor,
            profiler,
            "forward.attention_residual",
        )?;

        self.ffn_norm.forward_into(
            &workspace.ffn_input,
            &mut workspace.ffn_normalized,
            stream,
            llama,
            profiler,
            "forward.ffn_norm",
        )?;
        self.gate_up_proj.forward_into(
            &workspace.ffn_normalized,
            &mut workspace.gate_up,
            stream,
            gemm,
            profiler,
            "forward.gate_up_proj.gemm",
        )?;
        profiler.measure(stream, "forward.gate_up_proj.split", || {
            llama.split_group2(
                stream,
                LaunchConfig::for_num_elems((N * FF) as u32),
                workspace.gate_up.as_device_buffer(),
                FF as u32,
                workspace.gate.as_device_buffer_mut(),
                workspace.up.as_device_buffer_mut(),
            )
        })?;
        swiglu_into(
            &workspace.gate,
            &workspace.up,
            &mut workspace.activated,
            stream,
            llama,
            profiler,
            "forward.swiglu",
        )?;
        self.down_proj.forward_into(
            &workspace.activated,
            &mut workspace.projection_output,
            stream,
            gemm,
            profiler,
            "forward.down_proj.gemm",
        )?;
        add_into(
            &workspace.ffn_input,
            &workspace.projection_output,
            &mut workspace.final_input,
            stream,
            tensor,
            profiler,
            "forward.ffn_residual",
        )?;

        self.final_norm.forward_into(
            &workspace.final_input,
            &mut workspace.final_normalized,
            stream,
            llama,
            profiler,
            "forward.final_norm",
        )?;
        // Rows N..NP of head_input were zeroed at allocation and the convert
        // stops at the fp32 input's length, so they stay zero.
        profiler.measure(stream, "forward.lm_head.quantize", || {
            tensor.convert_f32_to_bf16_pairs(
                stream,
                pairs_config(N * D / 2),
                workspace.final_normalized.as_device_buffer(),
                &mut workspace.head_input,
            )
        })?;
        self.lm_head.forward_into::<NP, P>(
            &workspace.head_input_tma,
            &mut workspace.logits,
            stream,
            gemm_bf16,
            profiler,
            "forward.lm_head.gemm",
        )?;
        cross_entropy_into::<N, VOCAB, VP, P>(
            &workspace.logits,
            &workspace.targets,
            &mut workspace.losses,
            &mut workspace.loss_sum,
            &mut workspace.loss,
            stream,
            tensor,
            llama,
            profiler,
        )
    }

    pub fn backward(
        &mut self,
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.backward_profiled(
            workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            llama,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_profiled<P: KernelProfiler>(
        &mut self,
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        cross_entropy_backward_into::<N, VOCAB, VP, P>(
            &workspace.targets,
            &mut workspace.logits,
            stream,
            llama,
            profiler,
        )?;
        // Rows N..NP of logits hold zeros (forward computed them from the
        // zero-padded head input and the classifier backward skips them), so
        // the transposed operands feed exact zeros into the weight GEMM's
        // padded reduction slice.
        profiler.measure(stream, "backward.lm_head.transpose_input", || unsafe {
            tensor.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(NP, D),
                &workspace.head_input,
                NP as u32,
                D as u32,
                &mut workspace.head_input_t,
            )
        })?;
        profiler.measure(stream, "backward.lm_head.transpose_dlogits", || unsafe {
            tensor.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(NP, VP),
                &workspace.logits,
                NP as u32,
                VP as u32,
                &mut workspace.dlogits_t,
            )
        })?;
        self.lm_head.backward_weight::<NP, P>(
            &workspace.head_input_t_tma,
            &workspace.dlogits_t_tma,
            stream,
            gemm_bf16,
            profiler,
            "backward.lm_head.weight_gemm",
        )?;
        self.lm_head.backward_input::<NP, P>(
            &workspace.logits_tma,
            &mut workspace.d_head_input,
            stream,
            gemm_bf16,
            profiler,
            "backward.lm_head.input_gemm",
        )?;
        profiler.measure(stream, "backward.lm_head.dequantize", || {
            tensor.convert_bf16_pairs_to_f32(
                stream,
                elementwise_config::<Rank2<N, D>>(),
                &workspace.d_head_input,
                workspace.d_model_0.as_device_buffer_mut(),
            )
        })?;
        self.final_norm.backward_into(
            &workspace.final_input,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
            &mut workspace.norm_backward_inv,
            stream,
            llama,
            profiler,
            ["backward.final_norm.input", "backward.final_norm.weight"],
        )?;

        self.down_proj.backward_into(
            &workspace.activated,
            &workspace.d_model_1,
            &mut workspace.d_ff_0,
            stream,
            gemm,
            profiler,
            [
                "backward.down_proj.weight_gemm",
                "backward.down_proj.input_gemm",
            ],
        )?;
        swiglu_backward_into(
            &workspace.gate,
            &workspace.up,
            &workspace.d_ff_0,
            &mut workspace.d_ff_1,
            &mut workspace.d_ff_2,
            stream,
            llama,
            profiler,
        )?;
        profiler.measure(stream, "backward.gate_up_proj.join", || unsafe {
            llama.join_group2(
                stream,
                LaunchConfig::for_num_elems((N * FF) as u32),
                workspace.d_ff_1.as_device_buffer(),
                workspace.d_ff_2.as_device_buffer(),
                FF as u32,
                workspace.gate_up.as_device_buffer_mut(),
            )
        })?;
        self.gate_up_proj.backward_into(
            &workspace.ffn_normalized,
            &workspace.gate_up,
            &mut workspace.d_model_3,
            stream,
            gemm,
            profiler,
            [
                "backward.gate_up_proj.weight_gemm",
                "backward.gate_up_proj.input_gemm",
            ],
        )?;
        self.ffn_norm.backward_into(
            &workspace.ffn_input,
            &workspace.d_model_3,
            &mut workspace.d_model_0,
            &mut workspace.norm_backward_inv,
            stream,
            llama,
            profiler,
            ["backward.ffn_norm.input", "backward.ffn_norm.weight"],
        )?;
        add_into(
            &workspace.d_model_1,
            &workspace.d_model_0,
            &mut workspace.d_model_2,
            stream,
            tensor,
            profiler,
            "backward.ffn_residual",
        )?;

        self.o_proj.backward_into(
            &workspace.attended,
            &workspace.d_model_2,
            &mut workspace.d_model_0,
            stream,
            gemm,
            profiler,
            ["backward.o_proj.weight_gemm", "backward.o_proj.input_gemm"],
        )?;
        flash_attention_backward_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.attended,
            &workspace.attention_logsumexp,
            &mut workspace.attention_dot,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
            &mut workspace.d_model_3,
            &mut workspace.d_model_4,
            stream,
            flash,
            profiler,
        )?;
        rope_into::<N, T, D, H, HD, P>(
            &workspace.d_model_1,
            &mut workspace.d_model_0,
            true,
            stream,
            llama,
            profiler,
            "backward.q_rope",
        )?;
        rope_into::<N, T, D, H, HD, P>(
            &workspace.d_model_3,
            &mut workspace.d_model_1,
            true,
            stream,
            llama,
            profiler,
            "backward.k_rope",
        )?;
        profiler.measure(stream, "backward.qkv_proj.join", || unsafe {
            llama.join_group3(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                workspace.d_model_0.as_device_buffer(),
                workspace.d_model_1.as_device_buffer(),
                workspace.d_model_4.as_device_buffer(),
                D as u32,
                workspace.qkv.as_device_buffer_mut(),
            )
        })?;
        self.qkv_proj.backward_into(
            &workspace.attention_normalized,
            &workspace.qkv,
            &mut workspace.d_model_3,
            stream,
            gemm,
            profiler,
            [
                "backward.qkv_proj.weight_gemm",
                "backward.qkv_proj.input_gemm",
            ],
        )?;
        self.attention_norm.backward_into(
            &workspace.attention_input,
            &workspace.d_model_3,
            &mut workspace.d_model_0,
            &mut workspace.norm_backward_inv,
            stream,
            llama,
            profiler,
            [
                "backward.attention_norm.input",
                "backward.attention_norm.weight",
            ],
        )?;
        add_into(
            &workspace.d_model_2,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
            stream,
            tensor,
            profiler,
            "backward.attention_residual",
        )?;
        self.embedding.backward(
            &workspace.tokens,
            &workspace.d_model_1,
            stream,
            llama,
            profiler,
            "backward.embedding",
        )
    }

    pub fn zero_grad(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.zero_grad_profiled(stream, tensor, &mut profiler)
    }

    pub fn zero_grad_profiled<P: KernelProfiler>(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        macro_rules! zero {
            ($field:ident) => {
                fill_zero(
                    &mut self.$field.dw,
                    stream,
                    tensor,
                    profiler,
                    concat!("zero_grad.", stringify!($field)),
                )?;
            };
        }
        zero!(embedding);
        zero!(attention_norm);
        zero!(qkv_proj);
        zero!(o_proj);
        zero!(ffn_norm);
        zero!(gate_up_proj);
        zero!(down_proj);
        zero!(final_norm);
        self.lm_head
            .zero_grad(stream, tensor, profiler, "zero_grad.lm_head")?;
        Ok(())
    }
}

fn rope_into<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    x: &GpuTensor<f32, Rank2<N, D>>,
    y: &mut GpuTensor<f32, Rank2<N, D>>,
    backward: bool,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
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
    Ok(())
}

fn flash_attention_forward_into<
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
    output: &mut GpuTensor<f32, Rank2<N, D>>,
    logsumexp: &mut GpuTensor<f32, Rank2<N, H>>,
    stream: &CudaStream,
    kernels: &flash_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    profiler.measure(stream, "forward.attention.flash", || {
        kernels.flash_attention_forward_tiled(
            stream,
            flash_forward_config::<N, T, H, HD>(),
            q.as_device_buffer(),
            k.as_device_buffer(),
            v.as_device_buffer(),
            T as u32,
            H as u32,
            output.as_device_buffer_mut(),
            logsumexp.as_device_buffer_mut(),
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn flash_attention_backward_into<
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
    output: &GpuTensor<f32, Rank2<N, D>>,
    logsumexp: &GpuTensor<f32, Rank2<N, H>>,
    softmax_dot: &mut GpuTensor<f32, Rank2<N, H>>,
    dy: &GpuTensor<f32, Rank2<N, D>>,
    dq: &mut GpuTensor<f32, Rank2<N, D>>,
    dk: &mut GpuTensor<f32, Rank2<N, D>>,
    dv: &mut GpuTensor<f32, Rank2<N, D>>,
    stream: &CudaStream,
    kernels: &flash_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    profiler.measure(stream, "backward.attention.flash_dot", || {
        kernels.flash_attention_backward_dot(
            stream,
            flash_dot_config::<N, H, HD>(),
            dy.as_device_buffer(),
            output.as_device_buffer(),
            HD as u32,
            softmax_dot.as_device_buffer_mut(),
        )
    })?;
    profiler.measure(stream, "backward.attention.flash_q", || {
        kernels.flash_attention_backward_q_tiled(
            stream,
            flash_backward_q_config::<N, T, H, HD>(),
            q.as_device_buffer(),
            k.as_device_buffer(),
            v.as_device_buffer(),
            dy.as_device_buffer(),
            logsumexp.as_device_buffer(),
            softmax_dot.as_device_buffer(),
            T as u32,
            H as u32,
            dq.as_device_buffer_mut(),
        )
    })?;
    profiler.measure(stream, "backward.attention.flash_kv", || {
        kernels.flash_attention_backward_kv_tiled(
            stream,
            flash_backward_kv_config::<N, T, H, HD>(),
            q.as_device_buffer(),
            k.as_device_buffer(),
            v.as_device_buffer(),
            dy.as_device_buffer(),
            logsumexp.as_device_buffer(),
            softmax_dot.as_device_buffer(),
            T as u32,
            H as u32,
            dk.as_device_buffer_mut(),
            dv.as_device_buffer_mut(),
        )
    })?;
    Ok(())
}

fn swiglu_into<const N: usize, const FF: usize, P: KernelProfiler>(
    gate: &GpuTensor<f32, Rank2<N, FF>>,
    up: &GpuTensor<f32, Rank2<N, FF>>,
    output: &mut GpuTensor<f32, Rank2<N, FF>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.swiglu_forward(
            stream,
            LaunchConfig::for_num_elems((N * FF) as u32),
            gate.as_device_buffer(),
            up.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })?;
    Ok(())
}

fn swiglu_backward_into<const N: usize, const FF: usize, P: KernelProfiler>(
    gate: &GpuTensor<f32, Rank2<N, FF>>,
    up: &GpuTensor<f32, Rank2<N, FF>>,
    dy: &GpuTensor<f32, Rank2<N, FF>>,
    dgate: &mut GpuTensor<f32, Rank2<N, FF>>,
    dup: &mut GpuTensor<f32, Rank2<N, FF>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
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
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cross_entropy_into<const N: usize, const VOCAB: usize, const VP: usize, P: KernelProfiler>(
    logits: &DeviceBuffer<u32>,
    targets: &GpuTensor<u32, Rank1<N>>,
    losses: &mut GpuTensor<f32, Rank1<N>>,
    loss_sum: &mut GpuTensor<f32, Rank1<1>>,
    loss: &mut GpuTensor<f32, Rank1<1>>,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    llama: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    profiler.measure(stream, "forward.loss.fused_classifier", || {
        llama.fused_classifier_forward_bf16(
            stream,
            classifier_config::<N>(),
            logits,
            targets.as_device_buffer(),
            N as u32,
            VOCAB as u32,
            VP as u32,
            losses.as_device_buffer_mut(),
        )
    })?;
    sum_into(
        losses,
        loss_sum,
        stream,
        tensor,
        profiler,
        "forward.loss.reduction",
    )?;
    scale_into(
        loss_sum,
        1.0 / N as f32,
        loss,
        stream,
        tensor,
        profiler,
        "forward.loss.mean",
    )
}

fn cross_entropy_backward_into<
    const N: usize,
    const VOCAB: usize,
    const VP: usize,
    P: KernelProfiler,
>(
    targets: &GpuTensor<u32, Rank1<N>>,
    dlogits: &mut DeviceBuffer<u32>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    profiler.measure(stream, "backward.loss.fused_classifier", || {
        kernels.fused_classifier_backward_in_place_bf16(
            stream,
            classifier_config::<N>(),
            targets.as_device_buffer(),
            1.0,
            N as u32,
            VOCAB as u32,
            VP as u32,
            dlogits,
        )
    })
}
