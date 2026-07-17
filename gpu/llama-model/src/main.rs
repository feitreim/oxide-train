//! End-to-end fp32 forward/backward parity against `nn::Llama`.

use cuda_core::CudaContext;
use nn::Llama;
use tensor_core::Shape;
use tensor_cpu::CpuTensor;

#[path = "lib.rs"]
mod model;
use model::GpuLlama;

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
    let tokens = [1, 5, 5, 2, 9, 3, 16, 0];
    let targets = [5, 5, 2, 7, 3, 16, 0, 4];

    let (cpu_loss, cpu_ctx) = cpu.forward(tokens, targets);
    let (gpu_loss, gpu_ctx) = gpu.forward(tokens, targets, &stream, &tensor, &llama)?;
    assert_close("loss", &gpu_loss, &cpu_loss, &stream, 5e-5, 5e-5)?;

    cpu.backward(cpu_ctx);
    gpu.backward(gpu_ctx, &stream, &tensor, &llama)?;

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
    gpu.zero_grad(&stream)?;

    println!("✓ full fp32 GPU Llama forward/backward matches CPU");
    Ok(())
}
