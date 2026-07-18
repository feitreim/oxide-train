//! Minimal Wikipedia training loop.
//!
//! Runs the canonical 182.7M-parameter single-block profile configuration
//! (matching bin/profile.rs) on real `TOK1` shards end to end.

use std::env;

use cuda_core::CudaContext;
use data::{Batches, TokenFile};
use nn::MoeDense;
use optim::{AdamWConfig, AuxLossSchedule};

#[path = "../lib.rs"]
mod model;
use model::{GpuDense, GpuDenseAdamW, GpuDenseWorkspace};

const B: usize = 32;
const T: usize = 1_024;
const N: usize = 32_768;
const NP: usize = 32_768;
const VOCAB: usize = 50_257;
const VP: usize = 50_304;
const D: usize = 1_536;
const H: usize = 24;
const HD: usize = 64;
const FF: usize = 2_048;
const E: usize = 8;
const K: usize = 2;
const C: usize = 8_192;

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
    let gemm_bf16 = model::Tcgen05Gemm::load_from_ptx(&cuda, "gemm.ptx")?;
    let flash = model::flash_kernels::load(&cuda)?;
    let dense = model::dense_kernels::load(&cuda)?;
    let config = AdamWConfig {
        learning_rate: env_parse("TRAIN_LEARNING_RATE", 3e-4),
        weight_decay: env_parse("TRAIN_WEIGHT_DECAY", 0.1),
        ..AdamWConfig::default()
    };
    let aux_schedule = AuxLossSchedule {
        base_coefficient: env_parse("TRAIN_AUX_COEFFICIENT", 1e-2),
        decay_horizon: env_parse("TRAIN_AUX_DECAY_HORIZON", 10_000.0),
    };
    aux_schedule.validate();
    let (mut gpu, mut optimizer, mut next_batch) = if resume {
        let checkpoint = model::checkpoint::load::<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C>(
            checkpoint_path.as_deref().expect("validated above"),
            &stream,
            &tensor,
        )?;
        if checkpoint.optimizer.config() != config {
            return Err(format!(
                "checkpoint optimizer config {:?} does not match requested {:?}",
                checkpoint.optimizer.config(),
                config
            )
            .into());
        }
        if checkpoint.optimizer.aux_schedule() != aux_schedule {
            return Err(format!(
                "checkpoint aux-loss schedule {:?} does not match requested {:?}",
                checkpoint.optimizer.aux_schedule(),
                aux_schedule
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
        let cpu =
            MoeDense::<N, T, VOCAB, D, H, HD, FF, E, K, C>::new(42, aux_schedule.coefficient(0));
        (
            GpuDense::<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C>::from_cpu(&stream, &cpu)?,
            GpuDenseAdamW::new(&stream, config, aux_schedule)?,
            0,
        )
    };
    let starting_step = optimizer.step() as usize;
    let mut workspace = GpuDenseWorkspace::<N, NP, T, VOCAB, VP, D, H, FF, E, K, C>::new(&stream)?;
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
        let inputs: Vec<usize> = inputs.as_slice().iter().map(|&t| t as usize).collect();
        let targets: Vec<usize> = targets.as_slice().iter().map(|&t| t as usize).collect();
        let inputs: &[usize; N] = inputs.as_slice().try_into().expect("length N");
        let targets: &[usize; N] = targets.as_slice().try_into().expect("length N");

        gpu.zero_grad(&stream, &tensor)?;
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
            &dense,
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
            aux_coefficient,
            &mut workspace,
            &stream,
            &tensor,
            &gemm,
            &gemm_bf16,
            &flash,
            &dense,
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
