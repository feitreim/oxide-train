//! Full fp32 GPU forward and backward for the single-block reference Llama.
//!
//! Parameters, gradients, and saved activations remain GPU-resident. The
//! implementation mirrors `nn::Llama` explicitly so residual splits and the
//! ownership of every backward context stay visible.

use bench_util::{KernelProfiler, NoopProfiler};
use cuda_core::{CudaEvent, CudaStream, DriverError, LaunchConfig, PinnedHostBuffer};
use nn::Llama;
use optim::AdamWConfig;
use tensor_core::{Rank1, Rank2, Rank3, Shape};

// cuda-oxide collects kernels from the selected binary target. The binary
// includes this file as a module, which in turn includes each canonical kernel
// source here instead of copying definitions or relying on dependency PTX.
#[path = "../../gemm/src/fp32.rs"]
mod gemm_device;
#[path = "../../llama-ops/src/lib.rs"]
mod llama_device;
#[path = "../../tensor-gpu/src/lib.rs"]
#[allow(dead_code)]
pub mod tensor_device;

pub use gemm_device::kernels as gemm_kernels;
pub use llama_device::kernels as llama_kernels;
pub use tensor_device::kernels as tensor_kernels;
use tensor_device::{GpuAdamWMoments, GpuTensor};

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
            gemm_device::launch_config(M, N),
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
            gemm_device::launch_config(K, N),
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
            gemm_device::launch_config(M, N),
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
                gemm_device::launch_config(N, GROUPS * OUT),
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
                gemm_device::launch_config(IN, GROUPS * OUT),
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
                gemm_device::launch_config(N, IN),
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
            kernels.rms_norm_forward(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
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
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<(), DriverError> {
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
    pub qkv_proj: GpuGroupedLinear<D, 3, D>,
    pub o_proj: GpuLinear<D, D>,
    pub ffn_norm: GpuRmsNorm<D>,
    pub gate_up_proj: GpuGroupedLinear<D, 2, FF>,
    pub down_proj: GpuLinear<FF, D>,
    pub final_norm: GpuRmsNorm<D>,
    pub lm_head: GpuLinear<D, VOCAB>,
}

/// GPU-resident AdamW state mirroring every model parameter.
pub struct GpuLlamaAdamW<const VOCAB: usize, const D: usize, const FF: usize> {
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
    pub lm_head: GpuAdamWMoments<Rank2<D, VOCAB>>,
}

impl<const VOCAB: usize, const D: usize, const FF: usize> GpuLlamaAdamW<VOCAB, D, FF> {
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
        update!(qkv_proj, self.config.weight_decay);
        update!(o_proj, self.config.weight_decay);
        update!(ffn_norm, 0.0);
        update!(gate_up_proj, self.config.weight_decay);
        update!(down_proj, self.config.weight_decay);
        update!(final_norm, 0.0);
        update!(lm_head, self.config.weight_decay);
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
pub struct GpuLlamaWorkspace<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
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
    probabilities: GpuTensor<f32, Rank3<N, H, T>>,
    attended: GpuTensor<f32, Rank2<N, D>>,
    ffn_input: GpuTensor<f32, Rank2<N, D>>,
    ffn_normalized: GpuTensor<f32, Rank2<N, D>>,
    gate_up: GpuTensor<f32, Rank3<N, 2, FF>>,
    gate: GpuTensor<f32, Rank2<N, FF>>,
    up: GpuTensor<f32, Rank2<N, FF>>,
    activated: GpuTensor<f32, Rank2<N, FF>>,
    final_input: GpuTensor<f32, Rank2<N, D>>,
    final_normalized: GpuTensor<f32, Rank2<N, D>>,
    loss_probabilities: GpuTensor<f32, Rank2<N, VOCAB>>,
    projection_output: GpuTensor<f32, Rank2<N, D>>,
    logits: GpuTensor<f32, Rank2<N, VOCAB>>,
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
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
> GpuLlamaWorkspace<N, T, VOCAB, D, H, FF>
{
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
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
            probabilities: GpuTensor::zeros(stream)?,
            attended: GpuTensor::zeros(stream)?,
            ffn_input: GpuTensor::zeros(stream)?,
            ffn_normalized: GpuTensor::zeros(stream)?,
            gate_up: GpuTensor::zeros(stream)?,
            gate: GpuTensor::zeros(stream)?,
            up: GpuTensor::zeros(stream)?,
            activated: GpuTensor::zeros(stream)?,
            final_input: GpuTensor::zeros(stream)?,
            final_normalized: GpuTensor::zeros(stream)?,
            loss_probabilities: GpuTensor::zeros(stream)?,
            projection_output: GpuTensor::zeros(stream)?,
            logits: GpuTensor::zeros(stream)?,
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

    pub fn loss(&self) -> &GpuTensor<f32, Rank1<1>> {
        &self.loss
    }

    fn upload_inputs(
        &mut self,
        tokens: [usize; N],
        targets: [usize; N],
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
        assert!(N <= u32::MAX as usize);
        assert!(N * H * T <= u32::MAX as usize);
        assert_eq!(N % T, 0);
        assert_eq!(D, H * HD);
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
            lm_head: GpuLinear::from_cpu(stream, &model.lm_head)?,
        })
    }

    pub fn forward(
        &self,
        tokens: [usize; N],
        targets: [usize; N],
        workspace: &mut GpuLlamaWorkspace<N, T, VOCAB, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
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
            llama,
            &mut profiler,
        )
    }

    pub fn forward_profiled<P: KernelProfiler>(
        &self,
        tokens: [usize; N],
        targets: [usize; N],
        workspace: &mut GpuLlamaWorkspace<N, T, VOCAB, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
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
        attention_forward_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &mut workspace.attended,
            &mut workspace.probabilities,
            stream,
            llama,
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
        self.lm_head.forward_into(
            &workspace.final_normalized,
            &mut workspace.logits,
            stream,
            gemm,
            profiler,
            "forward.lm_head.gemm",
        )?;
        cross_entropy_into(
            &workspace.logits,
            &workspace.targets,
            &mut workspace.loss_probabilities,
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
        workspace: &mut GpuLlamaWorkspace<N, T, VOCAB, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.backward_profiled(workspace, stream, tensor, gemm, llama, &mut profiler)
    }

    pub fn backward_profiled<P: KernelProfiler>(
        &mut self,
        workspace: &mut GpuLlamaWorkspace<N, T, VOCAB, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        cross_entropy_backward_into(
            &workspace.loss_probabilities,
            &workspace.targets,
            &mut workspace.logits,
            stream,
            llama,
            profiler,
        )?;
        self.lm_head.backward_into(
            &workspace.final_normalized,
            &workspace.logits,
            &mut workspace.d_model_0,
            stream,
            gemm,
            profiler,
            [
                "backward.lm_head.weight_gemm",
                "backward.lm_head.input_gemm",
            ],
        )?;
        self.final_norm.backward_into(
            &workspace.final_input,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
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
        attention_backward_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.probabilities,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
            &mut workspace.d_model_3,
            &mut workspace.d_model_4,
            stream,
            llama,
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
        zero!(lm_head);
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

fn attention_forward_into<
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
    probabilities: &mut GpuTensor<f32, Rank3<N, H, T>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
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
    Ok(())
}

fn attention_backward_into<
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
    dq: &mut GpuTensor<f32, Rank2<N, D>>,
    dk: &mut GpuTensor<f32, Rank2<N, D>>,
    dv: &mut GpuTensor<f32, Rank2<N, D>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
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

fn cross_entropy_into<const N: usize, const VOCAB: usize, P: KernelProfiler>(
    logits: &GpuTensor<f32, Rank2<N, VOCAB>>,
    targets: &GpuTensor<u32, Rank1<N>>,
    probabilities: &mut GpuTensor<f32, Rank2<N, VOCAB>>,
    losses: &mut GpuTensor<f32, Rank1<N>>,
    loss_sum: &mut GpuTensor<f32, Rank1<1>>,
    loss: &mut GpuTensor<f32, Rank1<1>>,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    llama: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
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

fn cross_entropy_backward_into<const N: usize, const VOCAB: usize, P: KernelProfiler>(
    probabilities: &GpuTensor<f32, Rank2<N, VOCAB>>,
    targets: &GpuTensor<u32, Rank1<N>>,
    dlogits: &mut GpuTensor<f32, Rank2<N, VOCAB>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
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
    })
}
