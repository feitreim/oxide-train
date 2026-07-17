//! Full fp32 GPU forward and backward for the single-block reference Llama.
//!
//! Parameters, gradients, and saved activations remain GPU-resident. The
//! implementation mirrors `nn::Llama` explicitly so residual splits and the
//! ownership of every backward context stay visible.

use cuda_core::{CudaStream, DriverError, LaunchConfig};
use nn::Llama;
use optim::AdamWConfig;
use tensor_core::{Rank1, Rank2, Rank3};

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

    fn forward<const N: usize>(
        &self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<N, OUT>>, DriverError> {
        x.matmul(&self.w, stream, kernels)
    }

    fn backward<const N: usize>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        dy: &GpuTensor<f32, Rank2<N, OUT>>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<N, IN>>, DriverError> {
        let dw = x.matmul_tn(dy, stream, kernels)?;
        self.dw.add_scaled_assign(1.0, &dw, stream, kernels)?;
        dy.matmul_nt(&self.w, stream, kernels)
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

    fn forward<const N: usize>(
        &self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<N, D>>, DriverError> {
        let mut y = GpuTensor::zeros(stream)?;
        kernels.rms_norm_forward(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            x.as_device_buffer(),
            self.w.as_device_buffer(),
            self.eps,
            D as u32,
            y.as_device_buffer_mut(),
        )?;
        Ok(y)
    }

    fn backward<const N: usize>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        dy: &GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<N, D>>, DriverError> {
        let mut dx = GpuTensor::zeros(stream)?;
        kernels.rms_norm_backward_x(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            x.as_device_buffer(),
            self.w.as_device_buffer(),
            dy.as_device_buffer(),
            self.eps,
            D as u32,
            dx.as_device_buffer_mut(),
        )?;
        kernels.rms_norm_backward_weight(
            stream,
            LaunchConfig::for_num_elems(D as u32),
            x.as_device_buffer(),
            dy.as_device_buffer(),
            self.eps,
            N as u32,
            D as u32,
            self.dw.as_device_buffer_mut(),
        )?;
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

    fn forward<const N: usize>(
        &self,
        tokens: &GpuTensor<u32, Rank1<N>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<N, D>>, DriverError> {
        let mut y = GpuTensor::zeros(stream)?;
        kernels.embedding_forward(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            self.w.as_device_buffer(),
            tokens.as_device_buffer(),
            D as u32,
            y.as_device_buffer_mut(),
        )?;
        Ok(y)
    }

    fn backward<const N: usize>(
        &mut self,
        tokens: &GpuTensor<u32, Rank1<N>>,
        dy: &GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        kernels.embedding_backward(
            stream,
            LaunchConfig::for_num_elems((VOCAB * D) as u32),
            tokens.as_device_buffer(),
            dy.as_device_buffer(),
            N as u32,
            D as u32,
            self.dw.as_device_buffer_mut(),
        )
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

/// GPU-resident AdamW state mirroring every model parameter.
pub struct GpuLlamaAdamW<const VOCAB: usize, const D: usize, const FF: usize> {
    config: AdamWConfig,
    step: u64,
    pub embedding: GpuAdamWMoments<Rank2<VOCAB, D>>,
    pub attention_norm: GpuAdamWMoments<Rank1<D>>,
    pub q_proj: GpuAdamWMoments<Rank2<D, D>>,
    pub k_proj: GpuAdamWMoments<Rank2<D, D>>,
    pub v_proj: GpuAdamWMoments<Rank2<D, D>>,
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
        config.validate();
        Ok(Self {
            config,
            step: 0,
            embedding: GpuAdamWMoments::zeros(stream)?,
            attention_norm: GpuAdamWMoments::zeros(stream)?,
            q_proj: GpuAdamWMoments::zeros(stream)?,
            k_proj: GpuAdamWMoments::zeros(stream)?,
            v_proj: GpuAdamWMoments::zeros(stream)?,
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
        self.step = self.step.checked_add(1).expect("AdamW step overflow");
        let (first_correction, second_correction) = self.config.bias_correction(self.step);

        macro_rules! update {
            ($field:ident, $weight_decay:expr) => {
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
                )?;
            };
        }

        update!(embedding, self.config.weight_decay);
        update!(attention_norm, 0.0);
        update!(q_proj, self.config.weight_decay);
        update!(k_proj, self.config.weight_decay);
        update!(v_proj, self.config.weight_decay);
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
        let attention_input = self.embedding.forward(&tokens, stream, llama)?;
        let attention_normalized = self
            .attention_norm
            .forward(&attention_input, stream, llama)?;
        let q = self.q_proj.forward(&attention_normalized, stream, tensor)?;
        let k = self.k_proj.forward(&attention_normalized, stream, tensor)?;
        let v = self.v_proj.forward(&attention_normalized, stream, tensor)?;
        let q = rope::<N, T, D, H, HD>(&q, false, stream, llama)?;
        let k = rope::<N, T, D, H, HD>(&k, false, stream, llama)?;
        let (attended, probabilities) =
            attention_forward::<N, T, D, H, HD>(&q, &k, &v, stream, llama)?;
        let attention_output = self.o_proj.forward(&attended, stream, tensor)?;
        let ffn_input = attention_input.add(&attention_output, stream, tensor)?;

        let ffn_normalized = self.ffn_norm.forward(&ffn_input, stream, llama)?;
        let gate = self.gate_proj.forward(&ffn_normalized, stream, tensor)?;
        let up = self.up_proj.forward(&ffn_normalized, stream, tensor)?;
        let activated = swiglu(&gate, &up, stream, llama)?;
        let ffn_output = self.down_proj.forward(&activated, stream, tensor)?;
        let final_input = ffn_input.add(&ffn_output, stream, tensor)?;

        let final_normalized = self.final_norm.forward(&final_input, stream, llama)?;
        let logits = self.lm_head.forward(&final_normalized, stream, tensor)?;
        let (loss, loss_probabilities) = cross_entropy(&logits, &targets, stream, tensor, llama)?;
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
        let dlogits = cross_entropy_backward(&ctx.loss_probabilities, &ctx.targets, stream, llama)?;
        let dx = self
            .lm_head
            .backward(&ctx.final_normalized, &dlogits, stream, tensor)?;
        let dx = self
            .final_norm
            .backward(&ctx.final_input, &dx, stream, llama)?;

        let dactivated = self
            .down_proj
            .backward(&ctx.activated, &dx, stream, tensor)?;
        let (dgate, dup) = swiglu_backward(&ctx.gate, &ctx.up, &dactivated, stream, llama)?;
        let dgate_input = self
            .gate_proj
            .backward(&ctx.ffn_normalized, &dgate, stream, tensor)?;
        let dup_input = self
            .up_proj
            .backward(&ctx.ffn_normalized, &dup, stream, tensor)?;
        let dnormalized = dgate_input.add(&dup_input, stream, tensor)?;
        let dffn_input = self
            .ffn_norm
            .backward(&ctx.ffn_input, &dnormalized, stream, llama)?;
        let dx = dx.add(&dffn_input, stream, tensor)?;

        let dattended = self.o_proj.backward(&ctx.attended, &dx, stream, tensor)?;
        let (dq, dk, dv) = attention_backward::<N, T, D, H, HD>(
            &ctx.q,
            &ctx.k,
            &ctx.v,
            &ctx.probabilities,
            &dattended,
            stream,
            llama,
        )?;
        let dq = rope::<N, T, D, H, HD>(&dq, true, stream, llama)?;
        let dk = rope::<N, T, D, H, HD>(&dk, true, stream, llama)?;
        let dq_input = self
            .q_proj
            .backward(&ctx.attention_normalized, &dq, stream, tensor)?;
        let dk_input = self
            .k_proj
            .backward(&ctx.attention_normalized, &dk, stream, tensor)?;
        let dv_input = self
            .v_proj
            .backward(&ctx.attention_normalized, &dv, stream, tensor)?;
        let dnormalized = dq_input
            .add(&dk_input, stream, tensor)?
            .add(&dv_input, stream, tensor)?;
        let dattn_input =
            self.attention_norm
                .backward(&ctx.attention_input, &dnormalized, stream, llama)?;
        let dx = dx.add(&dattn_input, stream, tensor)?;
        self.embedding.backward(&ctx.tokens, &dx, stream, llama)
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

fn rope<const N: usize, const T: usize, const D: usize, const H: usize, const HD: usize>(
    x: &GpuTensor<f32, Rank2<N, D>>,
    backward: bool,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
) -> Result<GpuTensor<f32, Rank2<N, D>>, DriverError> {
    let mut y = GpuTensor::zeros(stream)?;
    if backward {
        kernels.rope_backward(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            x.as_device_buffer(),
            T as u32,
            H as u32,
            HD as u32,
            y.as_device_buffer_mut(),
        )?;
    } else {
        kernels.rope_forward(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            x.as_device_buffer(),
            T as u32,
            H as u32,
            HD as u32,
            y.as_device_buffer_mut(),
        )?;
    }
    Ok(y)
}

fn attention_forward<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
>(
    q: &GpuTensor<f32, Rank2<N, D>>,
    k: &GpuTensor<f32, Rank2<N, D>>,
    v: &GpuTensor<f32, Rank2<N, D>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
) -> Result<(GpuTensor<f32, Rank2<N, D>>, GpuTensor<f32, Rank3<N, H, T>>), DriverError> {
    let mut probabilities = GpuTensor::zeros(stream)?;
    let mut output = GpuTensor::zeros(stream)?;
    kernels.attention_probabilities(
        stream,
        LaunchConfig::for_num_elems((N * H * T) as u32),
        q.as_device_buffer(),
        k.as_device_buffer(),
        T as u32,
        H as u32,
        HD as u32,
        probabilities.as_device_buffer_mut(),
    )?;
    kernels.attention_output(
        stream,
        LaunchConfig::for_num_elems((N * D) as u32),
        probabilities.as_device_buffer(),
        v.as_device_buffer(),
        T as u32,
        H as u32,
        HD as u32,
        output.as_device_buffer_mut(),
    )?;
    Ok((output, probabilities))
}

fn attention_backward<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
>(
    q: &GpuTensor<f32, Rank2<N, D>>,
    k: &GpuTensor<f32, Rank2<N, D>>,
    v: &GpuTensor<f32, Rank2<N, D>>,
    probabilities: &GpuTensor<f32, Rank3<N, H, T>>,
    dy: &GpuTensor<f32, Rank2<N, D>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
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
    )?;
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
    )?;
    kernels.attention_backward_v(
        stream,
        config,
        probabilities.as_device_buffer(),
        dy.as_device_buffer(),
        T as u32,
        H as u32,
        HD as u32,
        dv.as_device_buffer_mut(),
    )?;
    Ok((dq, dk, dv))
}

fn swiglu<const N: usize, const FF: usize>(
    gate: &GpuTensor<f32, Rank2<N, FF>>,
    up: &GpuTensor<f32, Rank2<N, FF>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
) -> Result<GpuTensor<f32, Rank2<N, FF>>, DriverError> {
    let mut output = GpuTensor::zeros(stream)?;
    kernels.swiglu_forward(
        stream,
        LaunchConfig::for_num_elems((N * FF) as u32),
        gate.as_device_buffer(),
        up.as_device_buffer(),
        output.as_device_buffer_mut(),
    )?;
    Ok(output)
}

fn swiglu_backward<const N: usize, const FF: usize>(
    gate: &GpuTensor<f32, Rank2<N, FF>>,
    up: &GpuTensor<f32, Rank2<N, FF>>,
    dy: &GpuTensor<f32, Rank2<N, FF>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
) -> Result<(GpuTensor<f32, Rank2<N, FF>>, GpuTensor<f32, Rank2<N, FF>>), DriverError> {
    let mut dgate = GpuTensor::zeros(stream)?;
    let mut dup = GpuTensor::zeros(stream)?;
    let config = LaunchConfig::for_num_elems((N * FF) as u32);
    kernels.swiglu_backward_gate(
        stream,
        config,
        gate.as_device_buffer(),
        up.as_device_buffer(),
        dy.as_device_buffer(),
        dgate.as_device_buffer_mut(),
    )?;
    kernels.swiglu_backward_up(
        stream,
        config,
        gate.as_device_buffer(),
        dy.as_device_buffer(),
        dup.as_device_buffer_mut(),
    )?;
    Ok((dgate, dup))
}

fn cross_entropy<const N: usize, const VOCAB: usize>(
    logits: &GpuTensor<f32, Rank2<N, VOCAB>>,
    targets: &GpuTensor<u32, Rank1<N>>,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    llama: &llama_kernels::LoadedModule,
) -> Result<(GpuTensor<f32, Rank1<1>>, GpuTensor<f32, Rank2<N, VOCAB>>), DriverError> {
    let mut probabilities = GpuTensor::zeros(stream)?;
    let mut losses = GpuTensor::<f32, Rank1<N>>::zeros(stream)?;
    llama.softmax_forward(
        stream,
        LaunchConfig::for_num_elems((N * VOCAB) as u32),
        logits.as_device_buffer(),
        VOCAB as u32,
        probabilities.as_device_buffer_mut(),
    )?;
    llama.cross_entropy_loss(
        stream,
        LaunchConfig::for_num_elems(N as u32),
        logits.as_device_buffer(),
        targets.as_device_buffer(),
        N as u32,
        VOCAB as u32,
        losses.as_device_buffer_mut(),
    )?;
    let loss = losses
        .sum(stream, tensor)?
        .scale(1.0 / N as f32, stream, tensor)?;
    Ok((loss, probabilities))
}

fn cross_entropy_backward<const N: usize, const VOCAB: usize>(
    probabilities: &GpuTensor<f32, Rank2<N, VOCAB>>,
    targets: &GpuTensor<u32, Rank1<N>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
) -> Result<GpuTensor<f32, Rank2<N, VOCAB>>, DriverError> {
    let mut dlogits = GpuTensor::zeros(stream)?;
    kernels.softmax_cross_entropy_backward(
        stream,
        LaunchConfig::for_num_elems((N * VOCAB) as u32),
        probabilities.as_device_buffer(),
        targets.as_device_buffer(),
        1.0,
        N as u32,
        VOCAB as u32,
        dlogits.as_device_buffer_mut(),
    )?;
    Ok(dlogits)
}
