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

## PyTorch throughput baseline

`gpu/model/baselines/pytorch_baseline.py` is the trainer's external reference:
the same 4.39B-parameter model, the same tokens off the same shard, the same
twelve-steps-minus-one measurement window, and the same
`training_flops_per_token()` over the same 2.25 PFLOP/s denominator. It runs in
its own Modal image (torch, no rustc) beside the kernel image:

```bash
modal run modal_app.py::pytorch_baseline                             # eager, B=16
modal run modal_app.py::pytorch_baseline --tiers "default" --batches 16
modal run modal_app.py::pytorch_baseline --tiers "reduce-overhead" --batches 16
modal run modal_app.py::pytorch_baseline --masters fp32              # stock autocast
```

The script asserts its own parameter count against the Rust model's term by
term, and its optimizer's moment dtypes against the policy, before it reports a
number — so neither a drifted shape nor a quietly narrowed moment can become a
throughput win.

### Matched precision policy (`--masters bf16`, the default)

`optim`'s policy, reproduced: **bf16 masters** for every matrix-shaped
parameter, **fp32 for norms and the router**, **fp32 AdamW moments**, and the
update arithmetic done in fp32 on a widened master that rounds back to bf16 on
the way out. One B200, B=16, twelve steps with the first discarded,
`wiki-val-00000.tok`:

| Tier | tokens/s | MFU | vs trainer | peak VRAM | CUDA graphs | compile |
| --- | --- | --- | --- | --- | --- | --- |
| trainer (`bin/train.rs`) | 81,532 | 36.20% | — | — | no | — |
| PyTorch eager | 49,282 | 21.88% | −39.6% | 132.6 GiB | no | 30.9 s |
| `torch.compile()` | **91,669** | **40.71%** | **+12.4%** | **95.5 GiB** | no | 38.0 s |
| `torch.compile(mode="reduce-overhead")` | 91,289 | 40.54% | +12.0% | 95.5 GiB | **yes** | 23.1 s |

The measured split is 87 parameters: 50 bf16 masters, 37 fp32 norms and
routers, and 174 of 174 moments fp32.

`mode="reduce-overhead"` *is* CUDA graphs, which the trainer does not have, so
it is not the honest comparison — and here it is 0.4% *behind* plain
`torch.compile()`, i.e. inside the noise of an 11-step window on a 357ms step.
The script checks whether inductor actually recorded graphs rather than
trusting the mode name. The line to answer is the **+12.4%** one.

`mode="max-autotune"` was not run: the escalation only continues while the
trainer is still ahead, and it stopped being ahead at `torch.compile()`.

### Stock autocast policy (`--masters fp32`), kept for the record

fp32 masters under bf16 autocast with `AdamW(fused=True)` — the default a
PyTorch practitioner reaches for, and what the first pass at this baseline
measured. It carries a real handicap the trainer does not: a weight cast per
matmul and twice the parameter and gradient bytes.

| Tier | B | tokens/s | MFU | vs trainer @ B=16 | peak VRAM | CUDA graphs |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch eager | 16 | 44,635 | 19.82% | −45.3% | 148.7 GiB | no |
| PyTorch eager | 12 | 46,668 | 20.72% | −42.8% | 125.8 GiB | no |
| `torch.compile()` | 16 | 88,840 | 39.45% | +9.0% | 111.6 GiB | no |
| `torch.compile(mode="reduce-overhead")` | 16 | 89,541 | 39.76% | +9.8% | 111.5 GiB | yes |

Matching the policy is worth +3.2% and 16.1 GiB against that, which is the
whole of the handicap and not much more.

### Where the policies still cannot meet

- **Gradients.** The trainer keeps fp32 gradients for every parameter — its
  weight-gradient GEMMs write fp32 directly beside a bf16 master. PyTorch's
  autograd produces gradients in the parameter's dtype, so the 50 bf16 masters
  get bf16 gradients; the 37 fp32 norms and routers do match. This is not
  fixable without a custom autograd function behind every matmul, and it
  **favors PyTorch**: half the gradient bytes to write, read, and reduce. It
  costs accuracy rather than speed — most visibly in the embedding's
  scatter-add, which accumulates in bf16 where the trainer accumulates in fp32.
- **Optimizer fusion.** PyTorch ships no fused kernel for a mixed
  parameter/moment dtype, so the bf16-master step is `torch.compile`d instead.
  Every row above has a fused optimizer one way or the other — `fused=True`
  where the dtypes allow it. It is not a detail: the same update left
  uncompiled is eleven passes over 4.39B parameters, costs 77ms a step, and
  drags `torch.compile()` from 91,669 down to 75,490, which would have measured
  the baseline's optimizer rather than the trainer's kernels.
- **The classifier.** The trainer's fused classifier never materializes fp32
  logits over the padded vocabulary; `F.cross_entropy` does. **Favors the
  trainer**, in both memory and bandwidth.
- **Everything else matches.** Norms, router, moments, and the update math are
  fp32 on both sides; the update expression and its round-to-nearest commit are
  transcribed from `optim::adamw_step` and `MasterStorage::Bf16`. Under bf16
  masters the embedding lookup is already bf16, so the residual-stream cast that
  the fp32 policy needed is a no-op and the one concession PyTorch used to get
  is gone.
- **Routing is identical**, and with a capacity factor of one both sides compute
  `E * C` expert rows whatever the routing looks like — so load balance cannot
  move either number.
