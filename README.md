# rust-trainer

A from-scratch LLM training engine in pure Rust, targeting a single NVIDIA
B200 (Blackwell), with GPU kernels written in Rust via
[cuda-oxide](https://github.com/NVlabs/cuda-oxide).

## Design pillars

- **Everything static.** All shapes in the training loop are const generics
  (`Rank2<M, N>` marker types); shape errors are compile errors, and every
  kernel instantiation is shape-specialized PTX.
- **Separate `CpuTensor` / `GpuTensor` types.** No device enum, no dynamic
  dispatch. The CPU tensor is the *reference*: every GPU kernel is validated
  against it, sharing a deterministic RNG so inputs match bit-for-bit.
- **Typed module-level reverse mode, no tape.** A model is a static
  composition of `Module`s; combinators like `Chain` derive the chain rule at
  the type level, and every hand-written backward is finite-difference checked
  (`nn::gradcheck`).
- **Kernels are benchmarked in isolation.** Each GPU kernel is a standalone
  crate under `gpu/` with a correctness binary and a CUDA-event benchmark;
  tuning constants sweep via Modal *outside* of training.

Plan: Dense-style model (RMSNorm, RoPE, SwiGLU, untied embeddings) at
~150-350M params, fp32 first then bf16 compute + fp32 master weights, AdamW
then Muon, trained on English Wikipedia tokenized with tiktoken `r50k_base`
(ids fit `u16`). MoE is next (SPEC milestone 8).

**See [SPEC.md](SPEC.md) for the full architecture, every design decision and
its rationale, and the milestone plan.**

## Layout

```
crates/                 CPU-side cargo workspace -- builds/tests on any machine
  tensor-core/          Shape (const-generic markers), Element, shared RNG
  tensor-cpu/           CpuTensor + naive reference ops
  nn/                   Module trait, Chain combinator, layers, gradcheck
  data/                 tiktoken r50k tokenizer, u16 token shards, mmap loader,
                        [B,T] batcher, prepare-wiki preprocessing binary
  optim/                AdamW + Muon CPU references, typed Dense state/visitor
gpu/                    standalone cuda-oxide crates -- built on Modal GPUs
  bench-util/           CUDA-event timing; re-exports the shared RNG
  vecadd/               toolchain smoke test (lib.rs kernel, main.rs check,
                        bin/bench.rs benchmark) -- the template for new kernels
  ops/                  auditable fp32 RMSNorm, SwiGLU, embedding, and fused
                        classifier parity kernels (f32 + packed-bf16 variants)
  gemm/                 register-tiled fp32 + Blackwell tcgen05 bf16 GEMMs
  flash-attn/           fused fp32 causal attention forward/backward
  tensor-gpu/           GpuTensor, elementwise/reduction/GEMM, fused AdamW,
                        Muon momentum/apply kernels, packed-bf16
                        converts/transpose + master-weight AdamW
  model/                full model parity (fp32 + bf16 tcgen05 lm-head and
                        block linears), Muon Newton–Schulz optimizer, tiny
                        overfit gates, shard trainer
modal_app.py            Modal image (CUDA 13 + LLVM 21 + pinned nightly +
                        cuda-oxide backend) and run/bench/sweep/sanitize entrypoints
run.sh                  thin wrapper over `modal run`
```

## CPU-side development (local)

```bash
cargo test          # tensor ops, gradchecks, shard/batcher/tokenizer; no GPU needed
```

## Data preparation (offline, once)

```bash
cargo run --release -p data --bin prepare_wiki -- --limit-files 1   # smoke test
cargo run --release -p data --bin prepare_wiki -- \
  --limit-files 1 --limit-articles 1000                             # bounded smoke shard
cargo run --release -p data --bin prepare_wiki                      # full run
```

Downloads `wikimedia/wikipedia` `20231101.en` parquet from the HF hub (cached
in `~/.cache/huggingface`), tokenizes with tiktoken `r50k_base` in parallel,
and writes `u16` token shards to `data/wiki/` (first 10M tokens to `wiki-val`,
the rest to 250M-token `wiki-train-*` shards).

The pinned nightly in `rust-toolchain.toml` matches the Modal image so local
tooling and GPU builds agree.

## GPU kernels (Modal)

```bash
pip install modal && modal setup        # once
modal run modal_app.py::doctor          # toolchain + GPU sanity check
./run.sh vecadd                         # correctness
./run.sh vecadd bench                   # throughput
./run.sh ops                            # Dense leaf-op CPU/GPU parity
./run.sh model profile                  # ~183M-param full-step CUDA-event profile
SWEEP="BM=128 BN=128,BM=256 BN=64" ./run.sh gemm   # tuning sweep (one container)
```

The first run builds the Modal image (the cuda-oxide backend build is the slow
part); later runs reuse it and only recompile the kernel. Default GPU is B200
(`GPU=H100 ./run.sh ...` to override).

### Full-step profiling

Run the dedicated profiler without a dataset shard:

```bash
./run.sh model profile
```

The binary uses a fixed, compile-time performance configuration: `B=32`, `T=1024`,
`VOCAB=50,257` (padded to 50,304 for the bf16 tcgen05 lm-head), `D=1536`,
`H=24`, `HD=64`, and `FF=4096` (about 182.7M parameters). It runs two complete
warmup steps, synchronizes the stream, and then measures one `zero_grad +
forward + backward + AdamW` step. Normal correctness and training binaries
retain the zero-event `NoopProfiler` path.

The report contains one CUDA-event duration per named kernel launch plus:

- `all kernels`: the sum of the individually measured launches;
- `unattributed`: device time inside the full-step events but outside a named
  kernel span, including input copies, allocations, gradient-buffer zero fills,
  and launch gaps;
- `full step`: the end-to-end device timeline used for performance comparisons.

Shard reading, checkpointing, and loss copies performed only for logging are
not part of this compute-step profile. Kernel names are prefixed with
`forward.`, `backward.`, or `optimizer.` so regressions can be assigned to a
training phase directly.

Use a single run to find hotspots or record the current baseline. For a
performance/fusion PR, run `BASELINE_REF=<git-ref> ./run.sh model
profile`: it builds the pushed baseline ref and the mounted candidate in one
container and profiles both back-to-back after equivalent warmups. Report both
full-step times and the changed kernel rows. Two separate `./run.sh`
invocations may land on different GPUs or clock states and do **not** satisfy
the same-container measurement gate in `SPEC.md`.

To add a kernel: copy `gpu/vecadd` to `gpu/<name>`, set `name` in its
`Cargo.toml`, write the `#[kernel]` in `src/lib.rs`, and give it a real
`bench.rs` figure of merit (GB/s if bandwidth-bound, TFLOP/s if compute-bound).
Expose tuning knobs as `pub const NAME: usize` in `lib.rs` so `SWEEP` can
rewrite them.

## GPU training smoke run

The milestone-6 trainer reads `TOK1` shards from the `rust-trainer-wiki` Modal
volume. Upload a prepared shard once, then launch the small reference
configuration (fp32 masters with the bf16 tcgen05 lm-head and block linears):

```bash
modal volume create rust-trainer-wiki
modal volume put rust-trainer-wiki \
  data/wiki/wiki-val-00000.tok /wiki-val-00000.tok

SHARD=/data/wiki-val-00000.tok STEPS=100 ./run.sh model train
LR=0.0003 WEIGHT_DECAY=0.1 LOG_EVERY=10 \
  SHARD=/data/wiki-val-00000.tok STEPS=1000 \
  CHECKPOINT=/data/checkpoints/wiki.ckpt CHECKPOINT_EVERY=100 \
  ./run.sh model train

# TRAIN_STEPS is the target global step when resuming.
SHARD=/data/wiki-val-00000.tok STEPS=2000 \
  CHECKPOINT=/data/checkpoints/wiki.ckpt RESUME=1 \
  ./run.sh model train
```

Model and batch shapes remain compile-time constants in
`gpu/model/src/bin/train.rs`. Runtime settings are limited to the shard,
step count, logging/checkpoint intervals, and AdamW scalars. Checkpoints include
all parameters, AdamW moments/configuration, the global step, static shape
metadata, and the next batch position; saves use atomic replacement.
