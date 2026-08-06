//! Batch-size sweep for the Wikipedia training loop.
//!
//! `bin/train.rs` fixes `B` at compile time, so finding the largest batch the
//! B200 holds means recompiling per candidate. This binary compiles a handful
//! of candidates at once and picks one per process from `SWEEP_B`: a whole
//! sweep is one build, and a batch that exhausts HBM takes only its own
//! process down rather than poisoning the allocator for the next one.
//!
//! Every candidate runs the step the trainer runs -- forward, backward, AdamW
//! -- and reports free VRAM after each allocation phase alongside the
//! trainer's own throughput and MFU line.

use std::{env, error::Error, sync::Arc, time::Instant};

use cuda_core::CudaContext;
use data::{Batches, TokenFile};
use optim::{AdamWConfig, AuxLossSchedule};

#[path = "../lib.rs"]
mod model;
use model::{GpuDense, GpuDenseAdamW, GpuMoeWorkspace};

const T: usize = 2_048;
const VOCAB: usize = 50_257;
const VP: usize = 50_432;
const D: usize = 3_072;
const H: usize = 24;
const HD: usize = 128;
const FF: usize = 4_096;
const E: usize = 8;
const K: usize = 2;
const L: usize = 12;
const B200_BF16_PEAK_FLOPS: f64 = 2.25e15;

fn training_flops_per_token() -> f64 {
    let linear_parameters = D * VP + L * (4 * D * D + D * E + 3 * K * D * FF);
    let linear_flops = 6 * linear_parameters;
    let attention_flops = 12 * L * T * H * HD;
    (linear_flops + attention_flops) as f64
}

fn print_vram(label: &str) {
    let mut free = 0usize;
    let mut total = 0usize;
    let rc = unsafe { cuda_bindings::cuMemGetInfo_v2(&mut free, &mut total) };
    if rc == 0 {
        println!(
            "vram {label}: used={:.1}GiB free={:.1}GiB total={:.1}GiB",
            (total - free) as f64 / (1 << 30) as f64,
            free as f64 / (1 << 30) as f64,
            total as f64 / (1 << 30) as f64,
        );
    }
}

/// Time one candidate batch size. `N` is `B * T`, and `C` the per-expert
/// capacity at a capacity factor of one, `N * K / E`; `dispatch!` below is the
/// only place those relations are spelled, and the asserts here hold it to
/// them.
fn measure<const B: usize, const N: usize, const C: usize>(
    tokens: &[u16],
    steps: usize,
    cuda: &Arc<CudaContext>,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(N, B * T);
    assert_eq!(C, N * K / E);
    if Batches::<B, T>::new(tokens).remaining() == 0 {
        return Err(format!("the shard is too short for a [{B}, {T}] batch").into());
    }

    let stream = cuda.default_stream();
    let tensor = model::tensor_kernels::load(cuda)?;
    let gemm = model::gemm_kernels::load(cuda)?;
    let gemm_bf16 = model::Tcgen05Gemm::load(cuda)?;
    let flash_bf16 = model::Tcgen05Flash::load(cuda)?;
    let flash = model::flash_kernels::load(cuda)?;
    let dense = model::dense_kernels::load(cuda)?;
    print_vram("after kernel load");

    let aux_schedule = AuxLossSchedule::default();
    let mut gpu = GpuDense::<N, N, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>::initialized(
        &stream,
        42,
        aux_schedule.coefficient(0),
    )?;
    let mut optimizer = GpuDenseAdamW::new(&stream, AdamWConfig::default(), aux_schedule, L)?;
    print_vram("after model and optimizer");
    let mut workspace = GpuMoeWorkspace::<N, N, T, VOCAB, VP, D, H, FF, E, K, C, L>::new(&stream)?;
    print_vram("after workspace");

    let mut batches = Batches::<B, T>::new(tokens);
    let mut timing_start = None;
    for step in 1..=steps {
        let (inputs, targets) = match batches.next() {
            Some(batch) => batch,
            None => {
                batches = Batches::new(tokens);
                batches.next().expect("a shard with one batch has a first")
            }
        };
        let inputs: Vec<usize> = inputs.as_slice().iter().map(|&t| t as usize).collect();
        let targets: Vec<usize> = targets.as_slice().iter().map(|&t| t as usize).collect();
        let inputs: &[usize; N] = inputs.as_slice().try_into().expect("length N");
        let targets: &[usize; N] = targets.as_slice().try_into().expect("length N");

        // No gradient fill, matching the trainer: allocation zeroes the
        // gradients and every AdamW write-back clears the one it consumed.
        let aux_coefficient = optimizer.aux_coefficient();
        gpu.forward(
            inputs,
            targets,
            aux_coefficient,
            &mut workspace,
            &stream,
            &tensor,
            &gemm,
            &gemm_bf16,
            &flash,
            &flash_bf16,
            &dense,
        )?;
        gpu.backward(
            aux_coefficient,
            &mut workspace,
            &stream,
            &tensor,
            &gemm,
            &gemm_bf16,
            &flash,
            &flash_bf16,
            &dense,
        )?;
        optimizer.update(&mut gpu, &stream, &tensor)?;
        if step == 1 {
            let loss = workspace.loss().to_host(&stream)?[0];
            println!("B={B} step=1 loss={loss:.6}");
            if !loss.is_finite() {
                return Err(format!("non-finite loss at B={B}").into());
            }
            print_vram("after the first step");
            stream.synchronize()?;
            timing_start = Some(Instant::now());
        }
    }

    let start = timing_start.expect("the sweep runs at least one step");
    stream.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();
    let measured_steps = steps - 1;
    let tokens_per_second = (measured_steps * N) as f64 / elapsed;
    let mfu = tokens_per_second * training_flops_per_token() / B200_BF16_PEAK_FLOPS;
    println!(
        "B={B} tokens/batch={N} throughput={tokens_per_second:.1} tokens/s mfu={:.2}% measured_steps={measured_steps} elapsed={elapsed:.3}s",
        100.0 * mfu,
    );
    Ok(())
}

/// Instantiates one [`measure`] per compiled-in candidate and selects among
/// them at run time. `N` and `C` must be const arguments, so the macro derives
/// each candidate's from its `B`. No selection lists the candidates and
/// allocates nothing, which is how a caller gates a sweep on the build.
macro_rules! dispatch {
    ($selected:expr, $tokens:expr, $steps:expr, $cuda:expr; $($b:literal),* $(,)?) => {
        match $selected {
            None => {
                println!("bsweep candidates: {:?}", [$($b),*]);
                Ok(())
            }
            $(Some($b) => measure::<$b, { $b * T }, { $b * T * K / E }>($tokens, $steps, $cuda),)*
            Some(other) => Err(format!(
                "SWEEP_B={other} is not compiled in; the candidates are {:?}",
                [$($b),*]
            )
            .into()),
        }
    };
}

fn main() -> Result<(), Box<dyn Error>> {
    let shard_path =
        env::var("TRAIN_SHARD").unwrap_or_else(|_| "/data/wiki-val-00000.tok".to_owned());
    let batch: Option<usize> = env::var("SWEEP_B")
        .ok()
        .map(|value| value.parse().expect("SWEEP_B must be a batch size"));
    let steps: usize = env::var("TRAIN_STEPS")
        .map(|value| value.parse().expect("TRAIN_STEPS must be a step count"))
        .unwrap_or(12);
    assert!(steps > 1, "throughput needs a step after the warmup step");

    let shard = TokenFile::open(&shard_path)?;
    let cuda = CudaContext::new(0)?;
    dispatch!(batch, shard.tokens(), steps, &cuda; 12, 14, 16, 18, 20, 24)
}
