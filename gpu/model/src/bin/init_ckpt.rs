//! Save the step-0 checkpoint every measurement run would otherwise re-derive.
//!
//! Parameter initialization is deterministic — seed 42, a pure function of the
//! shape constants — and it is the longest silent window in a trainer process.
//! A `compare_train` A/B pays it once per arm's warmup and once per arm per
//! round, six times for the default two rounds, to arrive six times at the same
//! bytes. This writes those bytes once so `bin/train` can take them with
//! `TRAIN_RESUME=1`, which at step zero is the state fresh init reaches: the
//! optimizer step is zero, its moments are zero, and `next_batch` is zero, so
//! the resumed loop opens on the same batch with the same parameters.
//!
//! The shape constants below are imposed from `bin/train.rs` by
//! `modal_app.py::_impose_train_shape` before this is built, so the two cannot
//! drift apart in a run that fills the cache. If they ever do,
//! `checkpoint::load` compares N/T/VOCAB/D/H/HD/FF/E/K/C/L against the binary
//! reading the file and refuses the mismatch rather than resuming the wrong
//! model — a stale cache entry costs a failed run, never a wrong number.

use std::{env, time::Instant};

use cuda_core::CudaContext;
use optim::{AdamWConfig, AuxLossSchedule};

#[path = "../lib.rs"]
mod model;
use model::{GpuDense, GpuDenseAdamW};

// SHAPE: imposed from bin/train.rs. Keep these as plain `const NAME: usize =
// <expr>;` lines — `_impose_train_shape` rewrites them by name.
const B: usize = 32;
const T: usize = 2_048;
const N: usize = B * T;
const NP: usize = N;
const VOCAB: usize = 50_257;
const VP: usize = 50_432;
const D: usize = 3_072;
const H: usize = 24;
const HD: usize = 128;
const FF: usize = 4_096;
const E: usize = 8;
const K: usize = 2;
const C: usize = N * K / E;
const L: usize = 12;

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

/// The shape this binary was compiled for, in the order `checkpoint::load`
/// checks it. Printed rather than assumed: the constants are imposed from
/// `bin/train.rs` at build time, so the way to know the imposition took is to
/// ask the binary that came out.
fn shape() -> String {
    format!(
        "B={B} T={T} N={N} NP={NP} VOCAB={VOCAB} VP={VP} D={D} H={H} HD={HD} \
         FF={FF} E={E} K={K} C={C} L={L} seed=42"
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `shape` costs no GPU and no CUDA context: it is how a harness both
    // builds this binary and checks what it was built as. `load` exists so the
    // cache can be justified rather than assumed -- it prints what resuming a
    // file costs, against the `init_seconds` the save side prints, in one
    // container on one filesystem.
    let mode = env::var("INIT_CKPT_MODE").unwrap_or_else(|_| "save".to_owned());
    if mode == "shape" {
        println!("init_ckpt_shape {}", shape());
        return Ok(());
    }
    let path = env::var("TRAIN_CHECKPOINT")
        .map_err(|_| "init_ckpt needs TRAIN_CHECKPOINT: the path to write")?;

    // The same names and defaults bin/train.rs reads. A resume whose config
    // disagrees with the checkpoint's is refused by train.rs, so the two have
    // to be spelled once and identically.
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

    let cuda = CudaContext::new(0)?;
    let stream = cuda.default_stream();

    if mode == "load" {
        let tensor = model::tensor_kernels::load(&cuda)?;
        let loading = Instant::now();
        let loaded = model::checkpoint::load::<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>(
            &path, &stream, &tensor,
        )?;
        stream.synchronize()?;
        let seconds = loading.elapsed().as_secs_f64();
        let bytes = std::fs::metadata(&path)?.len();
        println!(
            "load_seconds={seconds:.3} bytes={bytes} step={} next_batch={} rate={:.1} MiB/s",
            loaded.optimizer.step(),
            loaded.next_batch,
            bytes as f64 / seconds / (1 << 20) as f64,
        );
        return Ok(());
    }

    let started = Instant::now();
    let model = GpuDense::<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>::initialized(
        &stream,
        42,
        aux_schedule.coefficient(0),
    )?;
    let optimizer = GpuDenseAdamW::new(&stream, config, aux_schedule, L)?;
    stream.synchronize()?;
    let initialized = started.elapsed().as_secs_f64();

    let saving = Instant::now();
    model::checkpoint::save(&path, &model, &optimizer, 0, &stream)?;
    let saved = saving.elapsed().as_secs_f64();
    let bytes = std::fs::metadata(&path)?.len();

    println!("init_checkpoint={path} bytes={bytes} {}", shape());
    println!("init_seconds={initialized:.3} save_seconds={saved:.3}");
    Ok(())
}
