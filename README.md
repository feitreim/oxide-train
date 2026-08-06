# rust-trainer

A from-scratch LLM training engine in pure Rust, targeting a single NVIDIA
B200 (Blackwell), with GPU kernels written in Rust via
[cuda-oxide](https://github.com/NVlabs/cuda-oxide).

Train a small Dense or MoE model on a B200 in minutes. Currently setup to train
on wikipedia, going for max MFU on a single GPU, then going to expand to 8xB200.
right now model size is sitting around 4billion total 1billion active
parameters.

The kernels are written against
[ferro-kittens](https://github.com/feitreim/ferro-kittens), a thunderkittens
style abstraction for kernels that are fast and easy to write and understand.
It grew up here as `gpu/kittens` and now lives in its own repository.


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
FEATURES=cublas ./run.sh gemm model_shapes         # every step GEMM shape vs cuBLASLt
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
`VOCAB=50,257` (padded to 50,432 for the bf16 tcgen05 lm-head), `D=1536`,
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

A throughput claim is quoted from `bin/train`, not from the profile, and gets
its pair from `modal run modal_app.py::train_ab --ref <pushed-branch>`: two
worktrees of two pushed refs (`--baseline-ref`, default `main`), each built
once and then alternated for `--rounds` rounds in one container, reporting
tokens/s and MFU per run plus each arm's mean and spread.

The external pair is `modal run modal_app.py::versus --ref main --batch 32`,
which alternates `bin/train` against the PyTorch baseline of
`gpu/model/baselines/pytorch_baseline.py` the same way. It runs on `ab_image`
-- the kernel toolchain plus torch, the only image that can hold both -- and
takes the trainer from `--ref` and the script from `--torch-ref`, so the
baseline branch stays the source of truth for what PyTorch is asked to do. The
batch is imposed on both arms, torch's compile time lands in its own warmup
steps, an arm whose inductor recorded CUDA graphs is rejected (the trainer has
none), and the summary carries a 95% interval on the difference rather than
two point estimates.

At `B=32`, 30 steps, `main` (`c5a3b1a`) against `torch.compile()` with default
mode, three containers of alternating rounds:

| Container | rounds | trainer tokens/s | `torch.compile()` tokens/s | trainer/torch |
| --- | --- | --- | --- | --- |
| 1 | 3 | 92,517 (±0.13%) | 92,554 (±0.22%) | 0.9996 |
| 2 | 6 | 90,048 (±0.13%) | 91,637 (±0.43%) | 0.9827 |
| 3 | 6 | 90,716 (±0.38%) | 91,625 (±0.25%) | 0.9901 |

Over all fifteen pairs the trainer is at **0.9890 of `torch.compile()`, ±0.0042
at 95%** -- 1.1% behind, and the interval excludes parity. It is not ahead in
any container, so `mode="max-autotune"` stays unrun by the standing escalation
rule. The reason this needed one container is in the columns: each arm holds a
0.13-0.43% spread *within* a container while the trainer's mean moves 2.7%
*between* them. Pairing is what makes a 1% claim measurable at all.

To add a kernel: copy `gpu/vecadd` to `gpu/<name>`, set `name` in its
`Cargo.toml`, write the `#[kernel]` in `src/lib.rs`, and give it a real
`bench.rs` figure of merit (GB/s if bandwidth-bound, TFLOP/s if compute-bound).
Expose tuning knobs as `pub const NAME: usize` in `lib.rs` so `SWEEP` can
rewrite them.

## GPU training smoke run

The milestone-6 trainer reads `TOK1` shards from the `rust-trainer-wiki` Modal
volume. Upload a prepared shard once, then launch the small reference
configuration (bf16 masters with the bf16 tcgen05 lm-head and block linears):

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
