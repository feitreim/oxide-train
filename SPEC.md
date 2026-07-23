# rust-trainer SPEC

Design decisions and architecture for a from-scratch LLM training engine in
pure Rust. This is the record of *what we decided and why*; the README covers
how to run things. Update this file when a decision changes — it should always
reflect current intent.

## 1. Goal

Train a Dense-style LLM on English Wikipedia, forwards + backwards + optimizer,
on a **single NVIDIA B200** (Blackwell, sm_100a), with every GPU kernel written
in Rust via [cuda-oxide](https://github.com/NVlabs/cuda-oxide). No PyTorch, no
cuBLAS host libraries, no `.cu` files.

Non-goals (for now): multi-GPU, inference serving, dynamic shapes, Python
bindings.

## 2. Toolchain & dev environment

- **cuda-oxide** (pinned `v0.2.1`, upstream stock backend) compiles `#[kernel]`
  Rust to PTX through a real rustc codegen backend. Kernels monomorphize like
  host Rust — const generics included — which is what makes the static-shape
  design (§3) real.
- **Pinned nightly** `nightly-2026-04-03` everywhere (rust-toolchain.toml =
  Modal image), because the codegen backend and cuda-oxide's proc macros
  require it. Bump both together with CUDA_OXIDE_REF.
- **Dev machines are GPU-less** (macOS). All CPU-side crates build/test
  locally; GPU work runs on **Modal B200** via `modal_app.py` (image: CUDA 13 +
  LLVM 21 + pinned nightly + prebaked cuda-oxide backend, adapted from the
  cuda-learning repo). `./run.sh <kernel> [bench]` is the loop.
- Blackwell-specific paths (tcgen05 tensor cores, TMA, cta_group::2) are fair
  game — B200 is the only serious target. `GPU=H100 ./run.sh` exists for
  debugging portable kernels only.

## 3. Static shapes (the core design bet)

**Every shape in the training loop is a compile-time constant.** Batch `B`,
sequence length `T`, model width `D`, heads `H`, FFN width, vocab — all const
generics. Fixed `T` with packed/padded sequences, standard for pretraining.

- Shapes are zero-sized marker types `Rank1<A>..Rank4<A,B,C,D>` implementing
  `Shape` (associated consts `RANK`, `NUM_ELEMENTS`, `DIMS`). A tensor's shape
  is part of its *type*; mismatches are compile errors. No runtime
  shape/stride structs anywhere.
- We avoid `generic_const_exprs` (incomplete even on nightly) by designing the
  op set to never need type-level shape arithmetic: matmul/attention/norms
  only *share* const params between types. Arithmetic in const-item position
  (`NUM_ELEMENTS = A * B`) is fine. If a future op truly needs it, a tiny,
  contained dose is acceptable — not a general dependency.
- Consequence to accept: each shape instantiation is separately-compiled
  (PTX included). One model config = one set of kernels; config sweeps pay in
  compile time.

## 4. Tensors: two types, no dispatch

- `CpuTensor<E, S>` (crates/tensor-cpu) and `GpuTensor<E, S>` (future,
  wrapping cuda-oxide `DeviceBuffer`) are **separate concrete types**. No
  device enum, no `dyn`, no PyTorch-style dispatcher. The `Tensor` trait in
  tensor-core is a minimal common surface for generic plumbing/tests only —
  ops are inherent methods on each concrete type (GPU ops take streams; CPU
  ops don't; unifying them would be a lie).
- **The CPU tensor is the reference, not a fast path.** Naive loops on
  purpose; if a CPU op is clever, it's wrong. Every GPU kernel is validated
  against it; every backward is finite-difference checked against it.
- **Parity via shared RNG**: one splitmix64 (top-24-bit f32 draws, exactly
  representable) lives in tensor-core and is re-exported by gpu/bench-util,
  so CPU and GPU tests reproduce identical inputs from a seed, bit-for-bit.
- Element types: `f32`, `bf16`, `u16` (token ids), `u32`.

## 5. Differentiation: typed module-level reverse mode (no tape)

Decided against a PyTorch-style runtime tape: with const-generic shapes every
tensor is a different type, so a tape needs type erasure — rebuilding dynamic
dispatch through the back door — and define-by-run buys nothing for a static
transformer. Instead (llm.c approach, made type-safe):

```rust
trait Module {
    type Input; type Output; type Ctx;
    fn forward(&self, x: Self::Input) -> (Self::Output, Self::Ctx);
    fn backward(&mut self, ctx: Self::Ctx, dy: Self::Output) -> Self::Input;
    fn zero_grad(&mut self);
}
```

- **Ownership contract**: `forward` takes input *by value*; a module that
  needs it for backward moves it into `Ctx`. No implicit clones — on GPU that
  means no implicit device copies. Values needed twice (residual streams) are
  duplicated by an explicit combinator that owns that policy. This by-value
  Ctx contract governs the CPU reference; the GPU model deliberately trades
  it for a persistent typed workspace (7e2) — aliasing safety there comes
  from disjoint workspace fields, verified by two-pass parity, rather than
  from ownership.
- **Accumulation contract**: `backward` *accumulates* (`+=`) into the module's
  own grad buffers — shared params and micro-batch gradient accumulation come
  free. `zero_grad` resets.
- Composition combinators derive the chain rule at the type level: `Chain<A,
  B: Module<Input = A::Output>>` (+ `chain!` macro) today; `Residual`, block
  repetition, and a duplication combinator when attention lands.
- ~8 hand-derived leaf backwards expected: matmul/linear, rmsnorm, rope,
  attention-softmax (flash-style later), swiglu, embedding, fused
  softmax-cross-entropy. Everything above them is mechanical.
- **Every leaf backward gets a gradcheck** (central differences vs analytic,
  `nn::gradcheck`) before it earns a GPU kernel.

## 6. Model

- Start: **Dense-style decoder** — RMSNorm (pre-norm), RoPE, SwiGLU FFN,
  untied embedding/lm-head, no biases. ~150–350M params first; scale after
  the loop is proven.
- Next (milestone 8): **MoE** (inkling-style thinking model is the eventual
  interest). Design rule that keeps it cheap: routing is a *runtime* decision
  inside a statically-shaped module — expert count/capacity are const
  generics, token assignment is data. FFN is shaped so SwiGLU → MoE is a
  type substitution.

## 7. Precision

1. **Phase 1: pure fp32** — gradcheck stays crisp, correctness first.
2. **Phase 2: bf16 compute + fp32 master weights and optimizer states**
   (tcgen05 GEMMs want bf16 inputs). Loss scaling not expected to be needed
   for bf16; revisit if grads underflow.

Phase 2 was adopted head-first (7e5), then extended to the block linears after
the post-7e7 profile put their fp32 GEMMs at 47.3% of the step (7e9). The
lm-head keeps packed-bf16 outputs/gradients; block linears use bf16 operands
with fp32 store/accumulate tcgen05 epilogues so activations, gradient
accumulation, AdamW state, and checkpoints remain fp32. Device-side bf16 is
stored as packed pairs (`u32` = two adjacent row elements). The master-weight
mechanism is observable: the tiny overfit gate plateaus while per-step updates
are below one bf16 ulp of the compute weights, then escapes once the fp32
master crosses a rounding boundary
(`crates/optim/examples/overfit_probe.rs` reproduces this on CPU).

CPU reference reductions accumulate in f64 so the reference never loses to
the thing it checks.

## 8. Optimizers

- **AdamW first**: one fused elementwise GPU kernel over params/grads/m/v.
  The CPU reference and GPU model keep shape-typed first/second moments; norm
  weights skip decay. A parameter visitor exposes static parameter kinds for
  later optimizer routing and checkpoint metadata.
- Training checkpoints use a versioned, little-endian `RTCKPT01` format with
  static model dimensions, AdamW config/step, next batch position, parameters,
  and both moments. Saves atomically replace the previous file; resume rejects
  shape or optimizer-config mismatches before training.
- **Muon after the first successful AdamW run**: Newton–Schulz
  orthogonalization (~5 matmuls per 2D weight per step) — cheap once GEMM
  works. Standard split: Muon for hidden 2D matrices; AdamW for embeddings,
  norms, lm-head. Parameter *kind* is statically known per module, which is
  what makes the routing trivial — optimizer state lives alongside params via
  a param-visitor trait (to be added with the optim crate).

## 9. Data pipeline

- **Dataset**: `wikimedia/wikipedia`, dump `20231101.en` (~6.4M articles),
  parquet from the HF hub.
- **Tokenizer**: tiktoken **`r50k_base`** (GPT-2 vocab, 50,257 tokens) via
  `tiktoken-rs` (embeds the vocab; no network at train time). Rationale:
  ids fit `u16` (halves shard size) and keeps embedding+head a sane fraction
  of a 150–350M model. Revisit when the model grows.
- **Offline preprocessing** (`prepare-wiki` binary, run once): download →
  parquet row-read `text` → rayon-parallel `encode_doc` (article tokens +
  `<|endoftext|>` = 50256 separator) → rolling shards. First 10M tokens →
  `wiki-val`, rest → `wiki-train-NNNNN.tok` (250M tokens ≈ 500MB each).
  Deterministic; no shuffling at this stage.
- **Shard format** `TOK1`: 24-byte LE header (magic/version/dtype/count) +
  flat `u16` ids. Header keeps the payload 2-byte aligned so `TokenFile`
  mmaps and hands out a zero-copy `&[u16]`. Little-endian targets only
  (compile-time asserted).
- **Batching**: `Batches<B, T>` slices `(inputs, targets)` as
  `CpuTensor<u16, Rank2<B, T>>`, targets = inputs shifted one token, windows
  advance by `B*T` (one pass = one epoch). Document boundaries are EOT tokens
  in-stream (llm.c-style); no attention masking across docs in v1. Sampling
  strategy (shuffled offsets, multi-shard) belongs to the loader, later.
- H2D upload of batches should overlap compute via pinned buffers +
  `cuda-async` once the GPU loop exists.

## 10. GPU kernels: benchmarking & sweeps

- Each kernel = **standalone crate under `gpu/`** (own `[workspace]`, built by
  `cargo oxide` on Modal): `src/lib.rs` (the `#[cuda_module]`), `main.rs`
  (correctness vs CPU reference), `src/bin/bench.rs` (CUDA-event timing via
  bench-util's `time_gpu_iters`; report GB/s if bandwidth-bound, TFLOP/s if
  compute-bound). Kernels are benchmarked and tuned *outside* training.
- **Tuning knobs are `pub const NAME: usize` in lib.rs** feeding const
  generics. `SWEEP="BM=128 BN=128,BM=256 BN=64" ./run.sh <kernel>` rewrites
  them per config and runs correctness + bench for each, all in one container
  so configs share a GPU and its clocks.
- CUDA C++ baselines live in `gpu/<kernel>/baselines/*.cu` (nvcc flags in a
  leading comment); `compute-sanitizer` and PTX dumps are wired in
  modal_app.py. GEMM work starts from cuda-oxide's `gemm_sol_final`
  (Blackwell SoL example) rather than from scratch.

### 10.1 Full-step profiler usage

- `./run.sh model profile` is the canonical hotspot report. It requires
  no shard and runs a fixed static configuration representative of the initial
  model scale: `B=1`, `T=64`, `VOCAB=50,257`, `D=1536`, `H=24`, `HD=64`, and
  `FF=4096` (182,705,664 parameters).
- The binary performs two untimed `zero_grad + forward + backward + AdamW`
  warmup steps, synchronizes, then records one complete step with CUDA events.
  Every explicit kernel launch is named by phase (`forward.*`, `backward.*`,
  `optimizer.*`). The normal training path uses `NoopProfiler`, so collecting
  events is opt-in.
- The full-step event interval includes gradient zeroing, allocations, and
  input H2D copies. Work not enclosed by a named kernel span is reported as
  `unattributed`; it must not be silently dropped when quoting step time.
  Dataset mmap/batching, checkpoint I/O, and optional loss D2H logging are
  outside the compute-step scope.
- A standalone profile run is suitable for hotspot discovery and baseline
  recording. A 7.x performance claim must execute the baseline and candidate
  back-to-back in the same container after equivalent warmups, and report
  both full-step totals plus affected kernel rows:
  `BASELINE_REF=<git-ref> ./run.sh model profile` builds the pushed
  baseline ref and the mounted candidate in one container and profiles both.
  This replaced the in-model naive-oracle plumbing (7e5): the retained naive
  kernels stay in their crates as parity oracles, but historical step
  configurations are reproduced from git rather than kept callable inside
  the model. Comparing separate `run.sh` invocations is invalid because
  Modal may assign different hardware or clock states.

## 11. Kernel fusion

- **Fusion is explicit substitution, never a compiler.** A fused kernel is a
  new typed module (or a new GEMM variant) with the *same* Input/Output types
  as the composition it replaces; adopting it is a type substitution — the
  same mechanism reserved for SwiGLU → MoE. No graph, no scheduler, no tape.
- **Three-tier oracle chain**: CPU reference (gradchecked) ← naive GPU
  kernels (parity vs CPU; milestones 4–5) ← fused/optimized kernels (parity
  vs naive). The naive kernels are never deleted — they are the spec the
  fused ones are tested against.
- **No epilogue framework.** GEMM fusion ships as hand-written variants:
  plain store, `+=` accumulate (for `dw`), `+residual` if a profile demands
  it. On Blackwell the epilogue is a pipeline stage (TMEM drain → smem
  swizzle → TMA store), not a scalar hook, so a shared epilogue abstraction
  may be *extracted* from ≥3 working variants but is never designed up front.
  Elementwise tails (e.g. the SwiGLU pair) stay separate kernels: after
  horizontal fusion, gate and up land in different output tiles, so a
  per-element hook couldn't fuse them anyway.
- **Horizontal fusion** (Q/K/V → one `[D,3D]` GEMM, gate+up → `[D,2FF]`) is a
  type-level shape change to existing modules — no new kernel machinery.
- **Measurement gate**: every perf/fusion PR must show full-step
  before/after numbers *from the same container* (~3× cross-container
  variance was observed on identical code). The step profiler (7a) exists to
  make this cheap.
- Activation checkpointing/recomputation: deferred to the scale milestone.
  Typed `Ctx` already makes each module's save-vs-recompute policy visible.

## 12. Repo layout

```
crates/            CPU-side workspace (builds/tests anywhere, no CUDA)
  tensor-core/     Shape markers, Element, shared RNG, Tensor trait
  tensor-cpu/      CpuTensor + naive reference ops
  nn/              Module trait, combinators, layers, gradcheck
  data/            tokenizer, shard format, mmap loader, prepare-wiki binary
  optim/           AdamW + Muon CPU references, typed Dense state, param
                   visitor, fp32 master weights
gpu/               standalone cuda-oxide kernel crates (Modal-built)
  bench-util/      CUDA-event timing + shared-RNG re-export
  vecadd/          toolchain smoke test; template for new kernels
  ops/             direct fp32 reference kernels + CPU/GPU parity for RMSNorm,
                   RoPE, causal attention, SwiGLU, embedding, and loss;
                   packed-bf16 fused classifier + block-parallel RMSNorm
  gemm/            register-tiled fp32 + Blackwell tcgen05 bf16 GEMMs,
                   packed-bf16 and fp32 store/accumulate variants; host-only
                   tcgen05 support (TMA maps, raw launchers) in src/host.rs
  flash-attn/      fused fp32 causal attention forward/backward, parity-tested
                   against ops without materialized probabilities;
                   FlashAttention-2 tiled kernels + per-row oracles
  tensor-gpu/      GpuTensor + elementwise/reduction kernels + naive/tiled
                   GEMM; packed-bf16 converts/transpose + master-weight AdamW
  model/           full GPU Dense forward/backward + CPU parity (fp32 with a
                   bf16 tcgen05 lm-head), fused AdamW, tiny overfit gate,
                   TOK1 shard trainer, checkpoints
modal_app.py       Modal image + run/bench/sweep/sanitize/baseline/ptx
```

## 13. Milestones

Each gated on tests; correctness before speed at every step.

1. ✅ Scaffold: workspace, tensor-core/tensor-cpu, Module/Chain/Linear +
   gradcheck, Modal harness, vecadd smoke crate
2. ✅ Data: shard format, tokenizer, batcher, prepare-wiki binary
3. ✅ CPU model forward+backward: RMSNorm, RoPE attention, SwiGLU, embedding,
   fused softmax-cross-entropy — all gradchecked; overfit a tiny batch on CPU
4. ✅ GPU foundation: tensor-gpu (GpuTensor), elementwise/reduction kernels,
   naive-then-tiled GEMM — all parity-tested vs CPU
5. ✅ GPU forward+backward of the full model; parity vs CPU at fp32
6. ✅ AdamW + GPU training loop; CPU/GPU update parity, tiny-batch overfit,
   deterministic checkpoint/resume, and a 100-step real-Wikipedia run
7. Perf — parallel tracks, each owning disjoint crates so PRs don't collide;
   integration is the one serialized step:
   - ✅ **7a step profiler** (bench-util + train): per-kernel CUDA-event
     breakdown of one full training step. Lands first — it gates every other
     7.x perf claim (see §11 measurement gate).
   - ✅ **7b GEMM ladder** (`gpu/gemm`, starting from cuda-oxide
     `gemm_sol_final`): register-tiled fp32 → tcgen05 bf16; store +
     accumulate variants; tuned via SWEEP.
   - ✅ **7c flash attention** (`gpu/flash-attn`): fused online-softmax fp32
     forward and recompute-softmax backward, parity-tested against ops'
     naive attention kernels without materializing the probability matrix.
   - ✅ **7d bf16 plumbing** (crates/tensor-core, tensor-cpu, optim): bf16
     `Element`, conversions, fp32 master weights — feeds 7b's tcgen05 phase.
   - **7e integration/fusion pass** (gpu/model): small serialized PRs,
     each gated on a §10.1 same-process before/after at the 182.7M profile
     config. Ordered by the first real-scale profile (2026-07-16: full step
     261 ms; loss softmax 59.6%, unattributed alloc/zero-fill/copy 24.8%,
     all other kernels ~15% combined); re-profile after each landing and
     reorder the remainder if the measured tail moves:
     - ✅ **7e1 fused classifier**: replace the naive softmax + cross-entropy
       pair with one row-parallel fused forward/backward (llm.c-style:
       block-per-row online reduction, dlogits produced in place, no
       `[N,VOCAB]` probability tensor saved in ctx). Motivation: the naive
       softmax recomputes the row max/denominator per element — O(V²) per
       row at V=50,257 — and alone measured 59.6–67.6% of the step. B200
       same-process result after 7e2: 196.57 → 38.34 ms full step (5.13×);
       classifier forward+backward 158.30 → 0.077 ms. The measured tail moved
       to naive GEMMs and attention backward, so 7e3 remains next.
     - ✅ **7e2 memory hygiene**: in-place `zero_grad` via a fill kernel
       (formerly twelve fresh `DeviceBuffer::zeroed` allocations per step);
       reuse activation/output buffers across steps instead of allocating
       per op; pinned-host staging for token H2D. Motivation: 24.8% of the
       step is unattributed allocation/zero-fill/copy time. B200 re-profile:
       196.67 ms full step, 0.226 ms (0.12%) unattributed; all twelve in-place
       gradient fills total 0.436 ms.
     - ✅ **7e3 GEMM integration**: swap model matmuls to gpu/gemm's
       register-tiled fp32 (store + accumulate variants — the accumulate
       path deletes the separate grad-accumulate launch per linear);
       horizontal QKV `[D,3D]` and gate+up `[D,2FF]` fusion. B200
       same-container re-profile against post-7e1 main: 37.95 → 36.23 ms
       full step (-4.5%); the same ~1.7 ms GEMM-row win measured 0.88%
       against the pre-7e1 196 ms step, before the fused classifier
       removed the softmax that buried it.
     - ✅ **7e4 flash-attention integration**: swap the naive attention
       kernels for gpu/flash-attn; re-tile its backward (key-block
       parallel, flash-2 style) if the profile shows the B·H-block launch
       becoming the tail at real T. The first integrated profile put the
       B·H-block backward at 4.29 ms, so backward was split into query-parallel
       dQ and key-parallel dK/dV using saved `[N,H]` log-sum-exp scalars. B200
       same-process result after 7e3: 36.35 → 29.79 ms full step (-18.0%);
       attention forward+backward 6.789 → 0.243 ms (27.9×), with no `[N,H,T]`
       probability matrix.
     - ✅ **7e5 bf16 compute + norm backward**: tcgen05 GEMMs + fp32 master
       weights/optimizer states (§7 phase 2, plumbing from 7d), adopted
       head-first — the lm-head is ~70% of this one-block model's GEMM
       FLOPs and the profile's named target; block linears stay fp32
       register-tiled until a profile moves them (rule 17). Vocab padded
       50,257 → 50,304 (393×128) and token rows padded to `NP` (128) with
       provably inert zeros, so the tuned tcgen05 kernel's M,N ≡ 0 (mod
       128) contract holds unmodified and checkpoints stay byte-compatible
       (masters stored without padding). Also replaced the naive RMSNorm
       weight-gradient reduction (per-column row-norm recomputation) with a
       block-per-row inverse pass + column-parallel reduce. B200
       same-container result vs post-7e4 main: 29.84 → 13.03 ms full step
       (-56.3%, 2.29×); `backward.lm_head.input_gemm` 9.05 → 0.88 ms,
       `forward.lm_head.gemm` 0.66 → 0.10 ms, the three
       `backward.*_norm.weight` rows 8.29 → 0.06 ms (incl. the new inverse
       pass), all new head plumbing (quantize/transposes/dequantize/w_t
       sync) ~0.19 ms combined. Residual+RMSNorm fusion measured below
       noise (residual adds ~0.007 ms) and was dropped per the
       conditional. The re-profiled tail is `backward.embedding` at
       3.48 ms (26.7%) — the token-scan reference kernel — followed by the
       remaining fp32 block GEMMs (`gate_up` input 1.35 ms, lm-head
       weight-GEMM 1.08 ms); 7e6 addresses the embedding tail.
     - ✅ **7e6 embedding backward**: at the training shape B=8 T=512
       (N=4096, first profile after moving off the correctness batch),
       `backward.embedding` is 214.3 ms of a 341.9 ms step (62.7%) — the
       token-scan kernel is O(V·D·N) so it grew linearly with batch while
       everything else scaled sublinearly. Replace with a scatter-add
       (fp32 atomics or token bucketing), naive kernel retained as the
       parity oracle. Implemented as one fp32 atomic add per upstream-gradient
       element (O(N·D)); repeated-token parity passes against the naive oracle.
       B200 same-process result: 341.59 → 127.32 ms full step (-62.7%, 2.68×);
       `backward.embedding` 214.272 → 0.027 ms (~7,966×). The measured tail is
       now flash attention at 36.9 ms combined (29.0%, quadratic-in-T forward
       + backward), followed by the three `*_norm.input` backwards at 20.6 ms
       combined (16.1%).
     - ✅ **7e7 flash-attention tiling**: the post-7e6 batch-shape
       sweep (B200, 182.7M) measured a flat 38.9 µs/token across
       B=32/64/128 at T=1024 (GPU saturates by N≈32k; B is an
       optimization knob, not throughput) and 56.5 µs/token at T=2048,
       with the flash rows at 45.8% of the step at T=1024 and 62.7% at
       T=2048. The per-(row, head) blocks scanned keys serially with HD
       lanes; re-tiled to the FlashAttention-2 block structure: query/key
       tiles staged through shared memory with register-tiled score
       fragments, forward and dQ parallel over query blocks, dK/dV over
       key blocks, and the per-row `dy·y` dots staged once by a new
       `backward_dot` kernel (tile sizes are SWEEP consts; kernels
       specialize on `TILE_HD` = 64). Naive and per-row flash kernels
       retained as oracles; parity vs naive at T=80 (partial tiles) and
       T=4 holds to ~1e-7. B200 same-container results — B=32 T=1024:
       1276.4 → 751.8 ms full step (-41.1%, 1.70×), flash rows 582.6 →
       58.0 ms (10.0×); B=64 T=2048: 7402.4 → 3199.6 ms (-56.8%, 2.31×),
       flash rows 4645.1 → 442.5 ms (10.5×, 62.7% → 13.8% of step),
       56.5 → 24.4 µs/token. The T=2048 gate ran at B=64, not B=128:
       at B=128 the packed-bf16 logits launches index `NP*VP/2` words,
       which overflows u32 — a real scale ceiling to lift when a shape
       needs it. Landing note: the tiny-overfit gate's lr 0.03 proved
       knife-edge — a CPU probe injecting ±1-ulp-scale noise into
       attention outputs/gradients (modelling summation-order changes)
       parked ~1 in 8 realizations on the bf16 two-logit tie, which the
       tiled kernels' rounding realized on GPU; the gate now runs lr
       0.02, where every sampled realization converges by ~step 60. The
       re-profiled tail at B=32 T=1024 is the RMSNorm family (three
       `*_norm.input` backwards 163.6 ms + three forwards 88.3 ms,
       ~33.5% combined) and the fp32 block GEMMs
       (`backward.gate_up_proj` pair 118.7 ms, 15.8%).
     - ✅ **7e8 block-parallel RMSNorm**: the forward and input-backward
       reference kernels launched one thread per output element and had every
       thread rescan all D features, making both O(N·D²). Replaced the model
       path with one 256-thread block per row: lanes cooperatively reduce
       `sum(x²)` (and `sum(dy·w·x)` backward), then write strided columns.
       Input-backward also writes the row inverse factors, deleting the
       standalone inverse pass. The weight gradient now tiles the row
       dimension into 256-row chunks and atomically contributes one partial
       per column/chunk, exposing 128× more row-grid parallelism at N=32,768;
       naive kernels remain parity oracles. B200 same-container result at
       B=32 T=1024: 751.93 → 485.19 ms full step (-35.5%, 1.55×); all norm
       rows combined 267.99 → 1.08 ms (~248×). The three forwards fell
       88.34 → 0.38 ms, input backward plus inverse 163.77 → 0.45 ms, and
       weight backward 15.88 → 0.25 ms.
     - ✅ **7e9 tcgen05 block linears**: add fp32-output store/accumulate
       epilogues to the bf16 tcgen05 kernel and use them for QKV, output,
       gate/up, and down projections. Persistent packed-bf16 compute
       weights are refreshed from fp32 masters after AdamW; three reusable
       workspace buffers (~512 MB each at N=32,768) hold quantized row
       operands and the two transposes required by weight gradients.
       Non-tile-aligned correctness shapes retain the fp32 register-tiled
       oracle; the aligned tcgen05 path is gated end-to-end by a second
       tile-aligned parity/overfit configuration in gpu/model
       (128-aligned CPU parity at bf16 tolerances plus an overfit run,
       3.080031 → 0.000008). B200 same-container result vs post-7e8 main
       at B=32 T=1024: 484.21 → 152.14 ms full step (-68.6%, 3.18×); the
       twelve block-linear GEMM rows, including conversion/transpose work,
       fell 355.89 → 23.81 ms (14.9×), and the four post-AdamW
       sync_compute rows total 0.11 ms. Combined with 7e8, the step is
       751.9 → 152.1 ms (~4.9×) since the post-7e7 profile. The measured
       tail is now flash attention (57.5 ms combined, 37.8%) and the
       lm-head GEMM trio (54.2 ms, 35.6%).
     - ✅ **7e10 FA4-shaped tcgen05 attention forward** (#35, phases 1–3):
       rebuild the attention forward around Blackwell's tensor cores,
       FlashAttention-4 style, and route the model's tile-aligned shapes
       through it. Q/K/V de-interleave to packed-bf16 `[B*H, T, 64]` head
       panels with `softmax_scale·log₂e` folded into Q, so the softmax is
       base-2 on a software `exp2` (no SFU, no libdevice — the kernels
       ship in a separately built pure-PTX flash.ptx, gemm.ptx-style,
       loaded by a host-only module). `S = Q·Kᵀ` and `O += P·V` are
       tcgen05 MMAs with fp32 TMEM accumulation; O accumulates in TMEM
       *segments* under a fixed per-row max reference, drained to
       registers only when a tile max exceeds the reference by 2⁸ (FA4's
       conditional correction adapted to the missing `tcgen05.st` as
       segment-restart; the vote is one `vote.any` per warp plus one
       128-count mbarrier phase). The production kernel is persistent:
       two softmax warpgroups ping-pong adjacent query tiles over one
       shared K/V TMA ring and MMA warp (TMEM 384/512 columns, one
       CTA/SM, 320 threads), with a static descending-cost work-item loop
       and per-item mbarrier re-initialization. The phase-1 sync and
       phase-2 pipelined kernels stay in the artifact as oracles; the
       fp32 tiled kernels remain the fallback for non-aligned shapes and
       still run the backward. Kernel bench (B=32, T=1024, H=24, one
       container): sync 1.481 → pipelined 1.089 → persistent 0.844 ms
       (137 TFLOP/s); parity vs the staged-bf16 CPU reference and both
       fp32 oracles is bit-identical across all three schedules, and the
       measured correction rate is 0% at bench distributions (the first
       tile's reference survives every stream). Model gates (tile-aligned
       parity, all overfits) passed with no re-tune. B200 same-container
       profile vs post-MoE main (239.3M, B=32 T=1024): 160.48 → 148.41 ms
       full step (-7.5%); forward.attention.flash 13.59 → 0.85 ms (16.0×;
       1.23 ms including the new bf16 staging row). The measured tail is
       now the fp32 attention backward (flash_q + flash_kv + dot, 44.5
       ms, 30.0% — #35 phase 4) and the lm-head GEMM trio (54.2 ms,
       36.5%).
     - ✅ **7e11 tcgen05 attention backward** (#35, phase 4): two
       synchronous tcgen05 gradient kernels in the same flash.ptx
       artifact, sharing the forward's swizzle-aware bf16 fragment writes,
       transposed-B O-MMA shape, and fp32 TMEM accumulation. Probabilities
       are recomputed base-2 straight from the saved LSE (`P = exp2(s −
       lse·log₂e)`, no running-max machinery) over the same packed-bf16
       Q/K/V/dY head panels, with `logsumexp` (natural log) and the fp32
       `Σ dy·y` dot fed as read-only device slices. `backward_q`
       (query-parallel) recomputes `S = Q·Kᵀ` and `dP = dY·Vᵀ` per key
       tile, forms the true `dS = P·(dP − D)·scale`, and accumulates
       `dQ += dS·K`; `backward_kv` (key-parallel) recomputes the
       transposed `Sᵀ`/`dPᵀ` per query tile and accumulates `dV += Pᵀ·dY`
       and `dK += dSᵀ·Q`. The operand scaling folds into the bf16
       conversions — `scale` into dS (K is staged unscaled), `ln2` into
       dSᵀ (the pre-scaled Q lands `scale` on dK since `ln2·scale·log₂e =
       scale`), dV needs none. Gradient writes are disjoint by query/key
       tile, so no atomics; the three-kernel split keeps `backward_dot`
       fp32 in lib.rs. Parity: staged-bf16 CPU reference (dq ≤6.9e-4,
       dk ≤6.4e-4, dv ≤3.8e-3) and the ops materialized-probability oracle
       at bf16 tolerances (dq ≤2.0e-3, dk ≤2.1e-3, dv ≤5.5e-3, T up to
       1024). Kernel bench (B=32 T=1024 H=24): combined dQ + dK/dV backward
       3.94 ms, 73.5 TFLOP/s. Model gate (aligned tcgen05 MoE parity,
       T=128) matches CPU and all overfits converge, no re-tune.
       B200 same-container profile vs main: 149.65 → 110.07 ms full step
       (-26.4%); backward.attention.flash_q 21.49 → 1.84 ms (11.7×),
       flash_kv 22.56 → 2.11 ms (10.7×), combined backward attention (dot
       + q + kv + the new bf16 staging row) 44.46 → 4.87 ms (9.1×). Shipped
       sync-only: with the backward attention now 4.4% of the step it is no
       longer a tail, so the pipelined/warp-specialized backward is a
       recorded follow-up (as is the fused single-pass backward and 2-CTA,
       deferred per the issue). compute-sanitizer is unavailable to gate it
       — every tool reports "Device not supported" on B200 (sm_100).
     - ✅ **7e12 clustered 256×256 tcgen05 GEMM** (#41): replace the
       single-CTA 128×128 tcgen05 kernel that backs every block linear and
       the lm-head with a Blackwell `cta_group::2` pair-UMMA kernel (adapted
       from cuda-oxide's `gemm_sol_final`): two CTAs — a cluster — cooperate
       on one M256×N256 output tile over a four-stage TMA pipeline and an
       fp32 TMEM accumulator, one cluster per tile. The eligibility contract
       tightens from 128 to 256: M, N, K must all be ≡ 0 (mod 256), and any
       GEMM that is 128- but not 256-aligned silently takes the fp32
       register-tiled fallback (all production dims qualify — D=1536,
       FF_expert=2048, 2·FF=4096, C=8192, VP, NP=32768). To keep the lm-head
       on the tiled path the padded vocabulary re-pads 50,304 (393×128) →
       50,432 (197×256), the added columns staying provably-inert zeros
       (7e5). **Root-caused deadlock**: the reference kernel schedules tiles
       persistently via CLC (`clc_try_cancel`), and that cross-cluster
       cancel/steal handshake deadlocks a fraction of launches at small
       grids — a fast cluster cancels a not-yet-launched peer and the cancel
       accounting stalls on the barrier, compounded by the multi-tile TMEM
       ACCUM ping-pong the steal path exposes. It was non-deterministic and
       slipped every gate: a standalone f32-store loop at the gate_up shape
       [8192,4096,1536] (512 tiles) hung after 5–25 iterations while
       [32768,·] never did, and 256³ parity is a single cluster / the ≤256
       model gates a single wave — neither reaches the CLC handoff. In the
       model the production forward froze in the first warmup expert GEMM.
       Fix: an exact-cover grid already gives every output tile an owning
       cluster, so work-stealing buys nothing and is removed — each cluster
       produces its one block-indexed tile and exits (the ACCUM double-buffer
       is retained but inert, `tile_iter` ≡ 0). 80× f32-store at the
       previously hanging shape, the down shape [8192,1536,2048], and 256³
       now pass. Kernel bench, single container: 4096³ store 1127 /
       accumulate 551 / f32-store 1054 / f32-accum 531 TFLOP/s; at the tall
       QKV shape M=32768 N=4608 K=1536 store 883 / accumulate 287 /
       f32-store 771 / f32-accum 273 — the packed-bf16 RMW accumulate
       epilogue is bandwidth-bound and falls to ~290 TFLOP/s at M=32768 (the
       weight-gradient GEMMs run this mode). B200 same-container profile vs
       main (239.3M, B=32 T=1024): 108.82 → 51.61 ms full step (−52.6%). Key
       GEMM rows (baseline → candidate ms): forward lm_head 18.41 → 5.50,
       backward lm_head weight 17.98 → 3.62 / input 17.79 → 3.37 (the lm-head
       trio 54.2 → 12.5 ms); forward qkv 1.87 → 0.70, o_proj 0.68 → 0.30,
       experts gate_up 2.98 → 1.28, down 1.72 → 0.78; backward experts
       gate_up weight 4.94 → 2.29 / input 3.35 → 0.83, down weight 3.29 →
       1.31 / input 1.48 → 0.65, qkv weight 2.22 → 0.94 / input 1.71 → 0.40,
       o_proj weight 0.86 → 0.47 / input 0.60 → 0.22. gemm 256³ parity (all
       four store/accumulate modes) and all 15 model gates pass; the branch
       is merged onto current main (post-#40).
     - ✅ **7e13 flash attention HD=128 conversion** (#42): the trained
       model's head width moved to 128 (§13.9), so every flash kernel that
       specialized on `HD == 64` now specializes on `HD == 128` — a single
       compile-time constant, no dual-head-width dispatch, the 64-wide fast
       paths deleted. Only the head-dim-generic per-row oracle kernels stay.
       **Tiling (issue §3 Option 1):** `TILE = 64` rows so `TILE_BYTES`
       returns to 16 KiB and all five SMEM plans fit the 227 KiB budget with
       the #35 pipeline/barrier structure intact. Each 128-wide operand is
       stored as **two stacked 64-wide (128-byte-row) `SWIZZLE_128B`
       subtiles**, so the swizzle phase still equals the row index inside each
       subtile — the coincidence HD=64 gave for free — and the manual
       `stmatrix` P/dS writes need no re-derivation (`swizzle_probe` confirms
       `chunk ^ ((row + phase) & 7)` empirically). HD-deep MMAs (`S = Q·Kᵀ`)
       walk 8 K=16 chunks across the two subtiles; HD-wide MMAs (`O = P·V`,
       the gradient MMAs) split into two N=64 accumulations, one per V/K
       subtile; the softmax scale is the `1/√128` literal (a `.sqrt()` would
       pull in libdevice and reject the pure-PTX path). **MMA shape:** the
       `M64` tcgen05 shape mis-pairs operand A/B K-indices for accumulator
       rows ≥16 across multi-chunk K accumulation (isolated with row-, column-
       and K-encoded TMEM probes; a single chunk or a uniform operand hides
       it). The fix keeps the proven `M128` shape over the 64-row tiles: the
       tensor core computes 128 M-rows, the unused rows 64..128 stream a
       `TILE_BYTES` phantom tail past each operand's base (one extra
       `TILE_BYTES` per SMEM plan keeps it in bounds; the garbage lands in
       accumulator rows never drained), and the drain reads rows 0..63 with
       the original `lane = row` fragment map. **Gates (B200):** flash-attn
       in-crate parity — sync + pipelined forward and both backward kernels vs
       the staged-bf16 CPU reference, over `T ∈ {128,256,384,512,1024}` and
       `B·H` up to 152 workstreams — pass (max abs: y 1.5e-3, lse 7.3e-5, dQ
       7.7e-4, dK 7.3e-4, dV 2.9e-3), and `T = 1024` exercises multiple key
       tiles and the correction path; pipelined 181 TFLOP/s at
       [32,1024,24,128]. gpu/model parity gates (dense fwd/bwd + AdamW, Muon,
       Newton–Schulz, expert and MoE substitution) match CPU at HD=128.
       Single-block profile at the §13.9 shape (B=12 T=2048 D=3072 H=24
       HD=128, tcgen05 pipelined): `forward.attention.flash` 2.98 ms,
       `backward.attention.flash_q` 5.52 ms, `flash_kv` 6.57 ms — ≈15 ms/block
       against the per-row oracle's ≈1.8 s/block (§1), a ~120× attention-kernel
       win; full step 81 ms/block. **Deferred:** the persistent (phase-3)
       two-Q-tile forward has a stream-B regression under this conversion, so
       the model runs on the pipelined (phase-2) forward meanwhile; the M64
       operand-pairing quirk is the reason the wasteful-but-correct M128 path
       is used.
     - ✅ **7e14 persistent forward stream-B fix** (#47): the deferred
       phase-3 regression was a TMEM **lane-access / warp-placement** bug, not
       an operand or column-overlap one. A `tcgen05.ld` warp reaches only TMEM
       lanes `(warp % 4) * 32 .. +32`, and a 64-row tile's `M128` accumulator
       keeps its real rows in lanes 0..63 (rows 64..127 are the undrained
       phantom tail). When the HD=128 conversion dropped `TILE` 128→64 each
       stream shrank from a 4-warp warpgroup to 2 warps, and the dispatch left
       stream B on warps 2–3 (warpgroup-0 positions 2–3 → hardware lanes
       64..127): its softmax drained the phantom rows, the inflated max
       underflowed every `P = exp2(s − m_ref)` to ~0, and `y` (and the LSE)
       came out garbage — stream A (warps 0–1 → lanes 0..63) was fine. The
       fix relabels the dispatch so stream B runs on warps 4–5 (warpgroup-1
       positions 0–1 → lanes 0..63) with the TMA/MMA warps on 2–3; block
       stays 192, every per-stream offset unchanged. Persistent is now green
       on both parity harnesses (flash-attn in-crate y 1.5e-3 / lse 7.3e-5;
       cross-oracle y 2.4e-3 / lse 1.2e-3) and **faster than pipelined**
       (kernel bench [32,1024,24,128]: 1.937 ms / 226 TFLOP/s vs pipelined
       2.425 ms / 181), so the model forward is switched back to
       `forward_persistent`. `BASELINE_REF=main` A/B at §13.9 (B=12 T=2048,
       single block): `forward.attention.flash` 3.07 → 2.45 ms (−20%), full
       step 81.70 → 80.51 ms; backward unchanged. **Still deferred:** the M64
       operand-K mispairing (issue #47 item 2) — a genuine, distinct operand
       addressing quirk — so the M128-over-64-row path stays.
     - ✅ **7e15 Design B tile-pairing for the backward kernels** (#47 item 2):
       `M64` is a proven dead end (the SS operand read broadcasts A-rows across
       subpartition pairs — only 32 distinct rows, SBO-invariant), so the wasted
       phantom half is recovered instead by PAIRING two adjacent 64-row tiles
       into every `M128_N64` MMA: operand A is stored as two stacked `[128, 64]`
       subtiles (tile A in accumulator rows 0..63, tile B in 64..127, both
       real), a single `score_mma_paired` walks A at the `TILE_BYTES` subtile
       stride against the shared unpaired B at `SUBTILE_BYTES`, and the drains
       run a 128-thread warpgroup (warps 0–1 → lanes 0..63, warps 2–3 →
       64..127). The causal edge compares the key/query column against the row
       **within its own tile** (`row & 63`), since the stacked rows 64..127 hold
       a second tile that restarts its causal count. Because paired tiles are
       adjacent their 128 rows are contiguous in the global tensors, so the
       existing `merge_output_tile`/`store_grad_tile` and LSE/dot staging reuse
       unchanged. Applied to **backward dQ** (pair two query tiles per CTA) and
       **backward dK/dV** (pair two key tiles); block grows 64→128, grid halves,
       and the operand plans drop `PHANTOM_PAD` (7·/8·`TILE_BYTES`).
       `tcgen05_attention_eligible` tightens to `T % 128 == 0` (odd-tile shapes
       fall to the per-row oracle; canonical T=2048 is unaffected). **Result:**
       gradients are bit-for-bit the same tolerance as the unpaired kernels
       (flash-attn in-crate dQ 7.7e-4, dK 7.3e-4, dV 2.9e-3 over
       `T ∈ {128,256,1024}`; all 15 model gates green, MoE overfit 2.94→1.1e-5),
       and the kernel bench `[32,1024,24,128]` backward goes **9.28 ms / 118
       TFLOP/s → 5.42 ms / 202 TFLOP/s (1.71×)** — the phantom-half recovery
       Design B promised. **Forward NOT paired:** the same pairing on the
       persistent forward is *correct* but *regresses* (kernel bench 226 →
       170 TFLOP/s) — fusing the tile pair into one warpgroup sacrifices the
       two-warpgroup ping-pong (7e14) that made the persistent forward fast
       (it is scheduling-bound, not MMA-bound), so the forward stays on the
       ping-pong scheme. The `BASELINE_REF=main` model A/B (baseline
       `backward.attention.flash_q` 66.5 ms, `flash_kv` 78.7 ms over 12 blocks;
       full step 748.8 ms) could not capture the candidate side: the Modal
       workspace was disabled (spend limit) mid-run.
   - **7f Muon**: ✅ CPU reference + orthogonality tests (`crates/optim`);
     ✅ GPU step (`GpuDenseMuon`): fp32 register-GEMM Newton–Schulz with
     per-group orthogonalization of the fused qkv/gate-up weights, gated on
     zeroth-power/optimizer parity vs the CPU reference plus tiny and
     tile-aligned Muon overfits. Train-loop/checkpoint wiring and a bf16
     tcgen05 Newton–Schulz remain follow-ups.

   Dependency shape: 7a/7b/7c/7d/7f can all run in parallel; 7e integrates
   their results into the model.
8. **MoE FFN** — next up. The §6 type substitution: swap the dense SwiGLU
   FFN for a mixture of experts, statically shaped (expert count `E`, top-k
   `K`, per-expert capacity `C` are const generics), with routing as runtime
   data. Same ladder as the dense model — correctness on CPU first, GPU
   parity second, speed last:
   - ✅ **8a CPU reference** (crates/nn): softmax top-k router + `E` expert
     SwiGLU FFNs behind the existing `Module` types, capacity-`C` dispatch
     with dropped-token passthrough, auxiliary load-balancing loss folded
     into the training loss; gradchecked (router included) and a tiny-batch
     CPU overfit gate. The aux-loss coefficient is runtime config, not a
     const generic: the host evaluates its schedule from the global step
     each iteration (balancing pressure wants to be strongest early,
     against routing collapse) and passes the scalar to the kernel like
     `learning_rate`; it rides the checkpoint header like `AdamWConfig` so
     resume can't silently change it. Only `E`/`K`/`C` are const generics —
     they determine shapes; the coefficient shapes nothing.
   - ✅ **8b GPU routing** (gpu/ops): top-k select + scatter/gather
     between token order and capacity-padded expert bins, CPU/GPU parity on
     shapes that force drops and underfull experts. The router is fp32
     end-to-end — logits GEMM, softmax, top-k, and gate weights — over
     bf16 experts: routing decisions are discrete, so rounding near a
     top-k boundary flips token assignment outright (the 7e7 two-logit
     tie was this failure mode in miniature), and the `[N,D]×[D,E]`
     router GEMM is too skinny for the tcgen05 tile contract anyway.
   - ✅ **8c GPU expert compute** (gpu/model): stacked fp32-master
     gate/up `[E,D,2FF]` and down `[E,FF,D]` weights with persistent packed-bf16
     compute copies; per-expert GEMM launches over capacity-padded bins on the
     7e9 tcgen05 fp32-store/accumulate path (`C` supplies tile alignment).
     One global transpose plus strided per-expert TMA maps keeps refresh and
     weight-gradient staging allocation-wide rather than per expert. The fp32
     register-tiled fallback remains the non-aligned oracle; both paths are
     gated against CPU expert forward/backward, zero-row inertness, repeated
     gradient accumulation, and post-AdamW compute-copy refresh.
   - ✅ **8d integration**: FFN swap in `GpuDense` behind identical types
     (dense retained as `GpuDenseDense`), gated like 7e — full parity with
     forced drops/underfull experts on both the fp32-oracle and tcgen05
     paths, aligned MoE overfit under a scheduled aux loss, checkpoint v3
     (`E`/`K`/`C` + schedule + router/expert state), and a §10.1
     same-container profile against the dense 152.3 ms baseline: 171.8 ms
     at matched active params (+12.7%; the win is params/FLOP, not step
     time). ✅ The first profile-ordered follow-up replaced the serial
     per-expert token scan with a deterministic block-parallel prefix
     assignment, preserving exact token/rank capacity order and tie behavior.
     B200 same-container result: 171.59 → 164.66 ms full step (-4.0%);
     `forward.router.assign` 7.009 → 0.083 ms (~85×). ✅ The second follow-up
     replaced the one-thread-per-output router weight scan with a deterministic
     tiled fp32 `Xᵀ·dlogits`: 2×8 output tiles, 16 fixed token partitions per
     output, and a fixed-order shared-memory reduction without atomics. B200
     same-container result against pre-follow-up main: 171.76 → 160.69 ms full
     step (-6.45%) for both routing changes; `backward.router.weight` 4.890 →
     0.916 ms (5.34×).
9. **Scale-up (in progress)** — depth + width toward MFU. Structural half
   done: `MoeDense`/`GpuDense` are `L`-block stacks (`MoeBlock`/`GpuBlock`);
   per-block workspaces split into saved activations (one set per block) vs
   backward scratch (`d_*` buffers, expert gradient bins, bf16 staging —
   one shared set), so activation memory scales as
   `L * saved + 1 * scratch`. Block backward keeps the residual-stream
   gradient resident in `d_model_1` across the reverse loop (no
   inter-block copies). Per-block aux losses accumulate into the same loss
   scalar; checkpoint v4 adds `L` and streams blocks; `GpuDense::initialized`
   builds one CPU block at a time so the 4.4B-param config never
   materializes host-side. Canonical config moved to
   `L=12, D=3072, H=24, HD=128, FF=4096, E=8, K=2` at `B=8, T=2048`
   (`C=4096` keeps `N·K == E·C`). Gates: CPU deep overfit, two-block
   CPU/GPU parity on both expert paths, checkpoint v4 bit-identical resume
   with `L`-mismatch rejection.
   - ⏳ **tcgen05 flash at HD=128** (#42): both flash generations specialize on
     `HD == 64` (`TILE_HD` fp32 tiles; the tcgen05 swizzle/SMEM plans bake
     `TILE_BYTES = 128·64·2`). `HD != 64` currently dispatches to the
     per-row oracle kernels — correct but serial over keys, so the big
     config trains at oracle-attention speed until the tcgen05 kernels
     learn 128 (two 64-wide panels or a 64-row tile re-tiling; the naive
     SMEM scaling of the persistent kernel would need 384 KiB > 227 KiB).
   - ✅ **`moe_scatter_dy` block-per-pair** (#45): the backward MoE scatter gave
     each `(token, slot)` pair a single thread that serially walked the whole
     `D=3072` gradient row — an uncoalesced copy fused with a serial gate dot,
     ~1% of HBM. Restructured to one block per pair: lanes stride the row for a
     fully coalesced `gate·dy` bin copy and the gate dot `Σ_d expert_output·dy`
     reduces in shared memory (fixed-order tree, no atomics; writes stay
     disjoint per surviving slot). Parity held with forced drops and underfull
     experts on both expert paths. B200 same-container A/B vs main at the §13.9
     shape (B=12 T=2048): `backward.router.scatter_dy` 30.77 → 3.66 ms (8.4×,
     4.11% → 0.51% of step); full step 748.81 → 721.19 ms (-3.7%). Follow-ups:
     float4-vectorize the row for the sub-1 ms floor, and fold the preceding
     `backward.router.zero_dy_bins` (3.74 ms `E·C·D` fill) into a dead-slot
     zeroing pass using the routing `assignment_counts`.
   - Then: activation checkpointing if B wants to grow past memory,
     (much later) multi-GPU

## 14. Decision log

| # | Decision | Why |
|---|----------|-----|
| 1 | cuda-oxide for all GPU code | Rust-native kernels; rustc monomorphization gives shape-specialized PTX; tcgen05/Blackwell support; accepted alpha-stage risk |
| 2 | All shapes const-generic | Compile-time shape safety; kernel specialization; no runtime shape machinery |
| 3 | Separate CpuTensor/GpuTensor | No dynamic dispatch; CPU is reference implementation, not a backend |
| 4 | Typed module reverse mode, no tape | Tape fights static types (type erasure); static transformer needs no define-by-run; less code, less memory |
| 5 | forward-by-value / Ctx-moves ownership | Zero implicit copies on GPU; duplication made explicit |
| 6 | Backward accumulates grads | Micro-batch grad accumulation + shared params for free |
| 7 | AdamW → Muon | AdamW trivial; Muon needs working GEMM; hidden/non-hidden routing is static |
| 8 | fp32 → bf16+fp32-master | Gradcheck clarity first; tcgen05 wants bf16 |
| 9 | Dense-style 150–350M first, MoE later | Proven architecture to validate engine; MoE = statically-shaped module w/ runtime routing |
| 10 | tiktoken r50k_base | u16 ids halve shard size; sane embed/head fraction at this scale; solved problem via tiktoken-rs |
| 11 | Offline tokenize → mmap u16 shards | Zero data-loading complexity in the hot loop (llm.c-style) |
| 12 | Per-kernel crates + const sweeps on Modal | Kernels tuned/benched in isolation, outside training; sweep = same mechanism as compile-time shapes |
| 13 | Shared splitmix64 for CPU/GPU parity | Bit-identical test inputs from a seed on both sides |
| 14 | B200 on Modal; Blackwell-only paths OK | The one real target; dev machines have no GPU |
| 15 | Fusion = typed substitution; no graph, no epilogue framework | Fused kernel = new module with identical types, parity vs the unfused path; Blackwell epilogues are pipeline stages, not scalar hooks — abstractions get extracted from ≥3 working variants, never designed first |
| 16 | Perf claims need same-container before/after | ~3× variance observed across Modal containers on identical code; sweeps already share one container for exactly this reason |
| 17 | Optimization backlog is profile-ordered | First real-scale profile contradicted intuition (naive loss softmax 60%+ of step, tcgen05 GEMM integration nowhere near top); 7e sub-milestones follow measured step share and get re-ordered after each landing |
| 18 | bf16 adopted head-first via padded NP/VP dims | lm-head ≈70% of the one-block model's GEMM FLOPs and the measured rock; zero-padding tokens→NP and vocab→VP keeps the tuned tcgen05 kernel's tile contract with provably inert padding (zero rows/columns never move), so no boundary-guard variants and byte-compatible checkpoints |
| 19 | tcgen05 kernels ship as a second, pure-PTX artifact | One embedded artifact per binary, and libdevice math (`exp`/`ln`/`sqrt`) forces it through libNVVM, which rejects tcgen05 lowerings; model loads `gemm.ptx` (prebuilt by gpu/gemm) through hand-written launchers in gemm/src/host.rs that mirror the generated marshalling |
| 20 | Block tcgen05 keeps fp32 model tensors | Quantize operands into persistent scratch and use concrete fp32-output store/accumulate epilogues; buffers, optimizer/checkpoint layout, and the naive fp32 fallback stay fp32, though epilogue values are bf16-rounded (the drain reuses the packed-bf16 shared-memory staging, so each GEMM result carries bf16 mantissa precision after full-K fp32 accumulation — doubling SMEM_OUT for true fp32 staging wasn't warranted) |
| 21 | MoE aux-loss coefficient is runtime config, not const | Const generics are reserved for values the compiler specializes on — `E`/`K`/`C` size buffers, bins, and launch grids; the coefficient is one scalar FMA that shapes nothing, needs a step schedule, and must be sweepable without a Modal rebuild (stable Rust also forbids f32 const generics). It flows host→kernel per step like `learning_rate` and is recorded in the checkpoint header like `AdamWConfig` |
| 22 | MoE router is fp32 over bf16 experts | Routing is discrete: bf16 rounding near a top-k boundary doesn't perturb the output, it reassigns the token (the 7e7 bf16 two-logit tie showed how violently trajectories react to that). The router GEMM is `[N,D]×[D,E]` — skinny, off the tcgen05 tile contract, and a rounding-error share of step FLOPs — so fp32 costs nothing measurable while keeping gate weights and the aux loss in the precision gradcheck trusts |
