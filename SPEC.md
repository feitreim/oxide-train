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

- **cuda-oxide** (pinned rev `2409204733c55b81435abf1db4e5fda8309edead`,
  upstream `main` 2026-07-18, upstream stock backend) compiles `#[kernel]`
  Rust to PTX through a real rustc codegen backend. Kernels monomorphize like
  host Rust — const generics included — which is what makes the static-shape
  design (§3) real. The pin moved off `v0.2.1` to pick up the
  generated-intrinsics infrastructure (upstream #406) that makes adding
  missing device intrinsics (e.g. `tcgen05.st` for FA4-style in-TMEM
  rescaling) substantially cheaper. Since upstream #318, every generated
  launcher is `unsafe fn` unless the kernel declares a `#[launch_contract]`;
  we absorb that with `unsafe { }` + SAFETY comments at each launch site
  rather than adopting contracts (our launch shapes are already asserted by
  the `*_config` helpers).
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
9. Scale/stretch: bigger model, activation checkpointing, (much later)
   multi-GPU

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
| 23 | cuda-oxide pinned to `main` rev `2409204` (2026-07-18), off `v0.2.1` | The FA4-shaped flash-attention plan needs a register→TMEM `tcgen05.st` the June tag lacks; this rev still lacks it but ships the generated-intrinsics path (upstream #406) that makes adding it — locally or upstream — cheap. Also picks up the rustc-independent PTX backend (#314), typed launch contracts (#318, absorbed as `unsafe` launch sites), and `--no-fmad` codegen fixes (#326). Toolchain nightly unchanged; `Cargo.toml` rev and `CUDA_OXIDE_REF` move together per §2 |
