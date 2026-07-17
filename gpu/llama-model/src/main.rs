//! End-to-end fp32 forward/backward parity against `nn::Llama`.

use cuda_core::CudaContext;
use nn::Llama;
use optim::{AdamWConfig, LlamaAdamW};
use tensor_core::Shape;
use tensor_cpu::CpuTensor;

#[path = "lib.rs"]
mod model;
use model::{GpuLlama, GpuLlamaAdamW, GpuLlamaWorkspace};

fn assert_close<S: Shape>(
    name: &str,
    gpu: &model::tensor_device::GpuTensor<f32, S>,
    cpu: &CpuTensor<f32, S>,
    stream: &cuda_core::CudaStream,
    atol: f32,
    rtol: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = gpu.to_host(stream)?;
    for (i, (&a, &e)) in actual.iter().zip(cpu.as_slice()).enumerate() {
        let tolerance = atol + rtol * e.abs();
        assert!(
            (a - e).abs() <= tolerance,
            "{name} mismatch at {i}: gpu={a}, cpu={e}, tolerance={tolerance}"
        );
    }
    Ok(())
}

fn overfit_tiny_batch(
    stream: &cuda_core::CudaStream,
    tensor: &model::tensor_kernels::LoadedModule,
    llama: &model::llama_kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    type TinyLlama = Llama<4, 4, 4, 8, 2, 4, 12>;
    let tokens = [0, 1, 2, 3];
    let targets = [1, 2, 3, 0];
    let cpu = TinyLlama::new(100);
    let mut gpu = GpuLlama::from_cpu(stream, &cpu)?;
    let config = AdamWConfig {
        learning_rate: 0.03,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut optimizer = GpuLlamaAdamW::new(stream, config)?;
    let mut workspace = GpuLlamaWorkspace::<4, 4, 4, 8, 2, 12>::new(stream)?;
    let mut initial_loss = None;

    for _ in 0..200 {
        gpu.zero_grad(stream, tensor)?;
        gpu.forward(tokens, targets, &mut workspace, stream, tensor, llama)?;
        if initial_loss.is_none() {
            initial_loss = Some(workspace.loss().to_host(stream)?[0]);
        }
        gpu.backward(&mut workspace, stream, tensor, llama)?;
        optimizer.update(&mut gpu, stream, tensor)?;
    }

    gpu.forward(tokens, targets, &mut workspace, stream, tensor, llama)?;
    let final_loss = workspace.loss().to_host(stream)?[0];
    let initial_loss = initial_loss.expect("training loop runs at least once");
    assert!(
        final_loss < 0.05,
        "GPU tiny batch did not overfit: initial={initial_loss}, final={final_loss}"
    );
    assert!(
        final_loss < initial_loss * 0.05,
        "GPU loss did not fall enough: initial={initial_loss}, final={final_loss}"
    );
    println!("✓ fused GPU AdamW overfits a tiny batch ({initial_loss:.6} -> {final_loss:.6})");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 8;
    const T: usize = 4;
    const VOCAB: usize = 17;
    const D: usize = 12;
    const H: usize = 3;
    const HD: usize = 4;
    const FF: usize = 19;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let tensor = model::tensor_kernels::load(&ctx)?;
    let llama = model::llama_kernels::load(&ctx)?;

    let mut cpu = Llama::<N, T, VOCAB, D, H, HD, FF>::new(42);
    let mut gpu = GpuLlama::from_cpu(&stream, &cpu)?;
    let mut workspace = GpuLlamaWorkspace::<N, T, VOCAB, D, H, FF>::new(&stream)?;
    let tokens = [1, 5, 5, 2, 9, 3, 16, 0];
    let targets = [5, 5, 2, 7, 3, 16, 0, 4];

    let (cpu_loss, cpu_ctx) = cpu.forward(tokens, targets);
    gpu.forward(tokens, targets, &mut workspace, &stream, &tensor, &llama)?;
    assert_close("loss", workspace.loss(), &cpu_loss, &stream, 5e-5, 5e-5)?;

    cpu.backward(cpu_ctx);
    gpu.backward(&mut workspace, &stream, &tensor, &llama)?;

    macro_rules! grad {
        ($field:ident, $tol:expr) => {
            assert_close(
                concat!(stringify!($field), ".dw"),
                &gpu.$field.dw,
                &cpu.$field.dw,
                &stream,
                $tol,
                $tol,
            )?;
        };
    }
    grad!(embedding, 2e-4);
    grad!(attention_norm, 2e-4);
    grad!(q_proj, 2e-4);
    grad!(k_proj, 2e-4);
    grad!(v_proj, 2e-4);
    grad!(o_proj, 2e-4);
    grad!(ffn_norm, 2e-4);
    grad!(gate_proj, 2e-4);
    grad!(up_proj, 2e-4);
    grad!(down_proj, 2e-4);
    grad!(final_norm, 2e-4);
    grad!(lm_head, 2e-4);

    // Second pass through the same workspace: identical weights and inputs
    // must reproduce identical loss and gradients. Catches state leaking
    // between steps via reused buffers, which single-pass parity cannot.
    gpu.zero_grad(&stream, &tensor)?;
    gpu.forward(tokens, targets, &mut workspace, &stream, &tensor, &llama)?;
    assert_close("loss (pass 2)", workspace.loss(), &cpu_loss, &stream, 5e-5, 5e-5)?;
    gpu.backward(&mut workspace, &stream, &tensor, &llama)?;
    grad!(embedding, 2e-4);
    grad!(attention_norm, 2e-4);
    grad!(q_proj, 2e-4);
    grad!(k_proj, 2e-4);
    grad!(v_proj, 2e-4);
    grad!(o_proj, 2e-4);
    grad!(ffn_norm, 2e-4);
    grad!(gate_proj, 2e-4);
    grad!(up_proj, 2e-4);
    grad!(down_proj, 2e-4);
    grad!(final_norm, 2e-4);
    grad!(lm_head, 2e-4);

    // Feed the exact GPU gradients to both optimizers so this comparison
    // isolates the fused update kernel from forward/backward rounding.
    macro_rules! copy_grad {
        ($field:ident) => {
            cpu.$field.dw = gpu.$field.dw.to_cpu(&stream)?;
        };
    }
    copy_grad!(embedding);
    copy_grad!(attention_norm);
    copy_grad!(q_proj);
    copy_grad!(k_proj);
    copy_grad!(v_proj);
    copy_grad!(o_proj);
    copy_grad!(ffn_norm);
    copy_grad!(gate_proj);
    copy_grad!(up_proj);
    copy_grad!(down_proj);
    copy_grad!(final_norm);
    copy_grad!(lm_head);

    let config = AdamWConfig {
        learning_rate: 0.01,
        weight_decay: 0.1,
        ..AdamWConfig::default()
    };
    let mut cpu_optimizer = LlamaAdamW::new(config);
    let mut gpu_optimizer = GpuLlamaAdamW::new(&stream, config)?;
    cpu_optimizer.update(&mut cpu);
    gpu_optimizer.update(&mut gpu, &stream, &tensor)?;

    macro_rules! weight {
        ($field:ident) => {
            assert_close(
                concat!(stringify!($field), ".w after AdamW"),
                &gpu.$field.w,
                &cpu.$field.w,
                &stream,
                2e-6,
                2e-6,
            )?;
        };
    }
    weight!(embedding);
    weight!(attention_norm);
    weight!(q_proj);
    weight!(k_proj);
    weight!(v_proj);
    weight!(o_proj);
    weight!(ffn_norm);
    weight!(gate_proj);
    weight!(up_proj);
    weight!(down_proj);
    weight!(final_norm);
    weight!(lm_head);
    gpu.zero_grad(&stream, &tensor)?;

    println!("✓ full fp32 GPU Llama forward/backward and AdamW match CPU");
    overfit_tiny_batch(&stream, &tensor, &llama)?;
    Ok(())
}
