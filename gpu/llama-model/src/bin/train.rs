//! Minimal Wikipedia training loop.
//!
//! This is the milestone-6 correctness runner, not the eventual performance
//! configuration. It intentionally uses the auditable reference kernels and a
//! small single-block model while exercising real `TOK1` shards end to end.
//! `D` is the one-tile minimum the bf16 tcgen05 lm-head requires.

use std::env;

use cuda_core::CudaContext;
use data::{Batches, TokenFile};
use nn::Llama;
use optim::AdamWConfig;

#[path = "../lib.rs"]
mod model;
use model::{GpuLlama, GpuLlamaAdamW, GpuLlamaWorkspace};

const B: usize = 1;
const T: usize = 64;
const N: usize = 64;
const NP: usize = 128;
const VOCAB: usize = 50_257;
const VP: usize = 50_304;
const D: usize = 128;
const H: usize = 4;
const HD: usize = 32;
const FF: usize = 192;

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} has an invalid value"))
        })
        .unwrap_or(default)
}

fn env_flag(name: &str) -> bool {
    env::var(name).is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(N, B * T);
    let shard_path =
        env::var("TRAIN_SHARD").unwrap_or_else(|_| "/data/wiki-val-00000.tok".to_owned());
    let max_steps: usize = env_parse("TRAIN_STEPS", 100);
    let log_every: usize = env_parse("TRAIN_LOG_EVERY", 10);
    let checkpoint_every: usize = env_parse("TRAIN_CHECKPOINT_EVERY", 0);
    let checkpoint_path = env::var("TRAIN_CHECKPOINT").ok();
    let resume = env_flag("TRAIN_RESUME");
    assert!(max_steps > 0, "TRAIN_STEPS must be positive");
    assert!(log_every > 0, "TRAIN_LOG_EVERY must be positive");
    if checkpoint_every > 0 && checkpoint_path.is_none() {
        return Err("TRAIN_CHECKPOINT_EVERY requires TRAIN_CHECKPOINT".into());
    }
    if resume && checkpoint_path.is_none() {
        return Err("TRAIN_RESUME requires TRAIN_CHECKPOINT".into());
    }

    let shard = TokenFile::open(&shard_path)?;
    let batches_per_epoch = Batches::<B, T>::new(shard.tokens()).remaining();
    if batches_per_epoch == 0 {
        return Err(format!("shard {shard_path} is too short for a [{B}, {T}] batch").into());
    }

    let cuda = CudaContext::new(0)?;
    let stream = cuda.default_stream();
    let tensor = model::tensor_kernels::load(&cuda)?;
    let gemm = model::gemm_kernels::load(&cuda)?;
    let gemm_bf16 = model::gemm_bf16_kernels::load(&cuda)?;
    let flash = model::flash_kernels::load(&cuda)?;
    let llama = model::llama_kernels::load(&cuda)?;
    let config = AdamWConfig {
        learning_rate: env_parse("TRAIN_LEARNING_RATE", 3e-4),
        weight_decay: env_parse("TRAIN_WEIGHT_DECAY", 0.1),
        ..AdamWConfig::default()
    };
    let (mut gpu, mut optimizer, mut next_batch) = if resume {
        let checkpoint = model::checkpoint::load::<N, NP, T, VOCAB, VP, D, H, HD, FF>(
            checkpoint_path.as_deref().expect("validated above"),
            &stream,
        )?;
        if checkpoint.optimizer.config() != config {
            return Err(format!(
                "checkpoint optimizer config {:?} does not match requested {:?}",
                checkpoint.optimizer.config(),
                config
            )
            .into());
        }
        println!(
            "resumed {} at step={} next_batch={}",
            checkpoint_path.as_deref().expect("validated above"),
            checkpoint.optimizer.step(),
            checkpoint.next_batch
        );
        (
            checkpoint.model,
            checkpoint.optimizer,
            checkpoint.next_batch,
        )
    } else {
        let cpu = Llama::<N, T, VOCAB, D, H, HD, FF>::new(42);
        (
            GpuLlama::<N, NP, T, VOCAB, VP, D, H, HD, FF>::from_cpu(&stream, &cpu)?,
            GpuLlamaAdamW::new(&stream, config)?,
            0,
        )
    };
    let starting_step = optimizer.step() as usize;
    let mut workspace = GpuLlamaWorkspace::<N, NP, T, VOCAB, VP, D, H, FF>::new(&stream)?;
    if max_steps < starting_step {
        return Err(
            format!("TRAIN_STEPS={max_steps} is behind checkpoint step {starting_step}").into(),
        );
    }

    println!(
        "training shard={shard_path} tokens={} batches/epoch={batches_per_epoch} steps={starting_step}..{max_steps}",
        shard.len(),
    );
    let mut batches = Batches::<B, T>::new(shard.tokens());
    for _ in 0..next_batch % batches_per_epoch as u64 {
        let _ = batches.next();
    }
    for step in starting_step + 1..=max_steps {
        let (inputs, targets) = match batches.next() {
            Some(batch) => batch,
            None => {
                batches = Batches::new(shard.tokens());
                batches.next().expect("non-empty shard has a first batch")
            }
        };
        next_batch += 1;
        let inputs = std::array::from_fn(|i| inputs.as_slice()[i] as usize);
        let targets = std::array::from_fn(|i| targets.as_slice()[i] as usize);

        gpu.zero_grad(&stream, &tensor)?;
        gpu.forward(
            inputs,
            targets,
            &mut workspace,
            &stream,
            &tensor,
            &gemm,
            &gemm_bf16,
            &flash,
            &llama,
        )?;
        let should_log = step == 1 || step % log_every == 0 || step == max_steps;
        if should_log {
            let loss = workspace.loss().to_host(&stream)?[0];
            println!("step={step} loss={loss:.6}");
            if !loss.is_finite() {
                return Err(format!("non-finite loss at step {step}").into());
            }
        }
        gpu.backward(
            &mut workspace,
            &stream,
            &tensor,
            &gemm,
            &gemm_bf16,
            &flash,
            &llama,
        )?;
        optimizer.update(&mut gpu, &stream, &tensor)?;

        let periodic_checkpoint = checkpoint_every > 0 && step % checkpoint_every == 0;
        let final_checkpoint = step == max_steps;
        if (periodic_checkpoint || final_checkpoint)
            && let Some(path) = &checkpoint_path
        {
            model::checkpoint::save(path, &gpu, &optimizer, next_batch, &stream)?;
            println!("checkpoint={path} step={step} next_batch={next_batch}");
        }
    }

    println!("finished {} AdamW steps", optimizer.step());
    Ok(())
}
