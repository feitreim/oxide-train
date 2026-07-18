//! Sample continuations from a training checkpoint.
//!
//! Eyeball check of training progress, not an inference path: sequence 0 of
//! the static [B, T] window holds a shard-prefix prompt, every other position
//! is `<|endoftext|>`, and causal attention keeps the dead positions
//! invisible while the live prefix grows one sampled token per forward.
//! Prints r50k token ids; decode with `data::Tokenizer` (offline feature).

use std::env;

use cuda_core::CudaContext;
use data::TokenFile;

#[path = "../lib.rs"]
mod model;
use model::GpuDenseWorkspace;

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

const EOT: usize = 50_256;
const PROMPT_TOKENS: usize = 32;
const TOP_K: usize = 40;
const TEMPERATURE: f32 = 0.8;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn sample_top_k(logits: &[f32], rng: &mut u64) -> usize {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(TOP_K);
    let max = ranked[0].1;
    let weights: Vec<f32> = ranked
        .iter()
        .map(|&(_, logit)| ((logit - max) / TEMPERATURE).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    let mut draw = (splitmix64(rng) >> 40) as f32 / (1u64 << 24) as f32 * total;
    for (&(token, _), &weight) in ranked.iter().zip(&weights) {
        if draw < weight {
            return token;
        }
        draw -= weight;
    }
    ranked[0].0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(N, B * T);
    let checkpoint_path =
        env::var("TRAIN_CHECKPOINT").map_err(|_| "sampling requires TRAIN_CHECKPOINT")?;
    let shard_path =
        env::var("TRAIN_SHARD").unwrap_or_else(|_| "/data/wiki-val-00000.tok".to_owned());
    let generate: usize = env::var("TRAIN_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    assert!(PROMPT_TOKENS + generate <= T);

    let cuda = CudaContext::new(0)?;
    let stream = cuda.default_stream();
    let tensor = model::tensor_kernels::load(&cuda)?;
    let gemm = model::gemm_kernels::load(&cuda)?;
    let gemm_bf16 = model::Tcgen05Gemm::load_from_ptx(&cuda, "gemm.ptx")?;
    let flash = model::flash_kernels::load(&cuda)?;
    let dense = model::dense_kernels::load(&cuda)?;

    let checkpoint = model::checkpoint::load::<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C>(
        &checkpoint_path,
        &stream,
        &tensor,
    )?;
    let gpu = checkpoint.model;
    println!(
        "sampling checkpoint={checkpoint_path} step={} temperature={TEMPERATURE} top_k={TOP_K}",
        checkpoint.optimizer.step()
    );
    let mut workspace = GpuDenseWorkspace::<N, NP, T, VOCAB, VP, D, H, FF, E, K, C>::new(&stream)?;

    let shard = TokenFile::open(&shard_path)?;
    let mut rng = 0x5EED_5EED_5EED_5EEDu64;

    let zero_targets = vec![0usize; N];
    let zero_targets: &[usize; N] = zero_targets.as_slice().try_into().expect("length N");
    for (index, offset) in [0usize, 1_000_000, 5_000_000].into_iter().enumerate() {
        let mut window = vec![EOT; N];
        for (slot, &token) in window[..PROMPT_TOKENS]
            .iter_mut()
            .zip(&shard.tokens()[offset..offset + PROMPT_TOKENS])
        {
            *slot = token as usize;
        }
        let mut live = PROMPT_TOKENS;
        while live < PROMPT_TOKENS + generate {
            gpu.forward(
                window.as_slice().try_into().expect("length N"),
                zero_targets,
                0.0,
                &mut workspace,
                &stream,
                &tensor,
                &gemm,
                &gemm_bf16,
                &flash,
                &dense,
            )?;
            let logits = workspace.logits_row(live - 1, &stream)?;
            window[live] = sample_top_k(&logits[..VOCAB], &mut rng);
            live += 1;
        }
        let ids: Vec<String> = window[..live].iter().map(|t| t.to_string()).collect();
        println!(
            "prompt{index} prompt_len={PROMPT_TOKENS} ids: {}",
            ids.join(" ")
        );
    }
    Ok(())
}
