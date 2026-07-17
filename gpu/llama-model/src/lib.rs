//! Full fp32 GPU forward and backward for the single-block reference Llama.
//!
//! Parameters, gradients, and saved activations remain GPU-resident. The
//! implementation mirrors `nn::Llama` explicitly so residual splits and the
//! ownership of every backward context stay visible.

use bench_util::{KernelProfiler, NoopProfiler};
use cuda_core::{CudaStream, DriverError, LaunchConfig};
use nn::Llama;
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
use tensor_device::GpuTensor;
pub use tensor_device::kernels as tensor_kernels;

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
    assert!(tensor_device::TILE * tensor_device::TILE <= 1024);
    LaunchConfig {
        grid_dim: (
            (N as u32).div_ceil(tensor_device::TILE as u32),
            (M as u32).div_ceil(tensor_device::TILE as u32),
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
    pub q_proj: GpuLinear<D, D>,
    pub k_proj: GpuLinear<D, D>,
    pub v_proj: GpuLinear<D, D>,
    pub o_proj: GpuLinear<D, D>,
    pub ffn_norm: GpuRmsNorm<D>,
    pub gate_proj: GpuLinear<D, FF>,
    pub up_proj: GpuLinear<D, FF>,
    pub down_proj: GpuLinear<FF, D>,
    pub final_norm: GpuRmsNorm<D>,
    pub lm_head: GpuLinear<D, VOCAB>,
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
        assert!(N <= u32::MAX as usize);
        assert!(N * H * T <= u32::MAX as usize);
        assert_eq!(N % T, 0);
        assert_eq!(D, H * HD);
        Ok(Self {
            embedding: GpuEmbedding::from_cpu(stream, &model.embedding)?,
            attention_norm: GpuRmsNorm::from_cpu(stream, &model.attention_norm)?,
            q_proj: GpuLinear::from_cpu(stream, &model.q_proj)?,
            k_proj: GpuLinear::from_cpu(stream, &model.k_proj)?,
            v_proj: GpuLinear::from_cpu(stream, &model.v_proj)?,
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
    ) -> Result<(GpuTensor<f32, Rank1<1>>, GpuLlamaCtx<N, T, VOCAB, D, H, FF>), DriverError> {
        let mut profiler = NoopProfiler;
        self.forward_profiled(tokens, targets, stream, tensor, llama, &mut profiler)
    }

    pub fn forward_profiled<P: KernelProfiler>(
        &self,
        tokens: [usize; N],
        targets: [usize; N],
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
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
        let q = self.q_proj.forward(
            &attention_normalized,
            stream,
            tensor,
            profiler,
            "forward.q_proj.gemm",
        )?;
        let k = self.k_proj.forward(
            &attention_normalized,
            stream,
            tensor,
            profiler,
            "forward.k_proj.gemm",
        )?;
        let v = self.v_proj.forward(
            &attention_normalized,
            stream,
            tensor,
            profiler,
            "forward.v_proj.gemm",
        )?;
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
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.backward_profiled(ctx, stream, tensor, llama, &mut profiler)
    }

    pub fn backward_profiled<P: KernelProfiler>(
        &mut self,
        ctx: GpuLlamaCtx<N, T, VOCAB, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
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
        let dq_input = self.q_proj.backward(
            &ctx.attention_normalized,
            &dq,
            stream,
            tensor,
            profiler,
            [
                "backward.q_proj.weight_gemm",
                "backward.q_proj.grad_accumulate",
                "backward.q_proj.input_gemm",
            ],
        )?;
        let dk_input = self.k_proj.backward(
            &ctx.attention_normalized,
            &dk,
            stream,
            tensor,
            profiler,
            [
                "backward.k_proj.weight_gemm",
                "backward.k_proj.grad_accumulate",
                "backward.k_proj.input_gemm",
            ],
        )?;
        let dv_input = self.v_proj.backward(
            &ctx.attention_normalized,
            &dv,
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
        let dnormalized = add(
            &dqk,
            &dv_input,
            stream,
            tensor,
            profiler,
            "backward.qkv_projection_sum",
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
        self.q_proj.dw = GpuTensor::zeros(stream)?;
        self.k_proj.dw = GpuTensor::zeros(stream)?;
        self.v_proj.dw = GpuTensor::zeros(stream)?;
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
