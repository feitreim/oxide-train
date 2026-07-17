# rust-trainer SPEC

Design decisions and architecture for a from-scratch LLM training engine in
pure Rust. This is the record of *what we decided and why*; the README covers
how to run things. Update this file when a decision changes — it should always
reflect current intent.

## 1. Goal

Train a Llama-style LLM on English Wikipedia, forwards + backwards + optimizer,
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

- Start: **Llama-style decoder** — RMSNorm (pre-norm), RoPE, SwiGLU FFN,
  untied embedding/lm-head, no biases. ~150–350M params first; scale after
  the loop is proven.
- Later: **MoE** (inkling-style thinking model is the eventual interest).
  Design rule to keep that cheap: routing is a *runtime* decision inside a
  statically-shaped module — expert count/capacity are const generics, token
  assignment is data. FFN is shaped so SwiGLU → MoE is a type substitution.

## 7. Precision

1. **Phase 1: pure fp32** — gradcheck stays crisp, correctness first.
2. **Phase 2: bf16 compute + fp32 master weights and optimizer states**
   (tcgen05 GEMMs want bf16 inputs). Loss scaling not expected to be needed
   for bf16; revisit if grads underflow.

Phase 2 is adopted head-first (7e5): the lm-head — ~70% of this one-block
model's GEMM FLOPs and the profile's named bf16 target — runs bf16 tcgen05
against fp32 master weights, while the block linears stay fp32 until a
profile puts them above noise. Device-side bf16 is stored as packed pairs
(`u32` = two adjacent row elements), matching the tcgen05 epilogue. The
master-weight mechanism is observable: the tiny overfit gate plateaus while
per-step updates are below one bf16 ulp of the compute weights, then escapes
once the fp32 master crosses a rounding boundary
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

- `./run.sh llama-model profile` is the canonical hotspot report. It requires
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
  `BASELINE_REF=<git-ref> ./run.sh llama-model profile` builds the pushed
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
  optim/           AdamW CPU reference, typed Llama state, param visitor,
                   fp32 master weights (Muon planned)
gpu/               standalone cuda-oxide kernel crates (Modal-built)
  bench-util/      CUDA-event timing + shared-RNG re-export
  vecadd/          toolchain smoke test; template for new kernels
  llama-ops/       direct fp32 reference kernels + CPU/GPU parity for RMSNorm,
                   RoPE, causal attention, SwiGLU, embedding, and loss;
                   packed-bf16 fused classifier + fast norm weight-gradient
  gemm/            register-tiled fp32 + Blackwell tcgen05 bf16 GEMMs,
                   store/accumulate variants, sweep benchmarks; host-only
                   tcgen05 support (TMA maps, raw launchers) in src/host.rs
  flash-attn/      fused fp32 causal attention forward/backward, parity-tested
                   against llama-ops without materialized probabilities
  tensor-gpu/      GpuTensor + elementwise/reduction kernels + naive/tiled
                   GEMM; packed-bf16 converts/transpose + master-weight AdamW
  llama-model/     full GPU Llama forward/backward + CPU parity (fp32 with a
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
     forward and recompute-softmax backward, parity-tested against llama-ops'
     naive attention kernels without materializing the probability matrix.
   - ✅ **7d bf16 plumbing** (crates/tensor-core, tensor-cpu, optim): bf16
     `Element`, conversions, fp32 master weights — feeds 7b's tcgen05 phase.
   - **7e integration/fusion pass** (gpu/llama-model): small serialized PRs,
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
       weight-GEMM 1.08 ms); a future 7e6 should start there.
   - **7f Muon** (crates/optim): CPU reference + orthogonality tests any
     time after milestone 6; GPU Newton–Schulz step once 7b's GEMM is fast.

   Dependency shape: 7a/7b/7c/7d/7f can all run in parallel; 7e integrates
   their results into the model.
8. Scale/stretch: bigger model, MoE, activation checkpointing, (much later)
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
| 9 | Llama-style 150–350M first, MoE later | Proven architecture to validate engine; MoE = statically-shaped module w/ runtime routing |
| 10 | tiktoken r50k_base | u16 ids halve shard size; sane embed/head fraction at this scale; solved problem via tiktoken-rs |
| 11 | Offline tokenize → mmap u16 shards | Zero data-loading complexity in the hot loop (llm.c-style) |
| 12 | Per-kernel crates + const sweeps on Modal | Kernels tuned/benched in isolation, outside training; sweep = same mechanism as compile-time shapes |
| 13 | Shared splitmix64 for CPU/GPU parity | Bit-identical test inputs from a seed on both sides |
| 14 | B200 on Modal; Blackwell-only paths OK | The one real target; dev machines have no GPU |
| 15 | Fusion = typed substitution; no graph, no epilogue framework | Fused kernel = new module with identical types, parity vs the unfused path; Blackwell epilogues are pipeline stages, not scalar hooks — abstractions get extracted from ≥3 working variants, never designed first |
| 16 | Perf claims need same-container before/after | ~3× variance observed across Modal containers on identical code; sweeps already share one container for exactly this reason |
| 17 | Optimization backlog is profile-ordered | First real-scale profile contradicted intuition (naive loss softmax 60%+ of step, tcgen05 GEMM integration nowhere near top); 7e sub-milestones follow measured step share and get re-ordered after each landing |
| 18 | bf16 adopted head-first via padded NP/VP dims | lm-head ≈70% of the one-block model's GEMM FLOPs and the measured rock; zero-padding tokens→NP and vocab→VP keeps the tuned tcgen05 kernel's tile contract with provably inert padding (zero rows/columns never move), so no boundary-guard variants and byte-compatible checkpoints |
| 19 | tcgen05 kernels ship as a second, pure-PTX artifact | One embedded artifact per binary, and libdevice math (`exp`/`ln`/`sqrt`) forces it through libNVVM, which rejects tcgen05 lowerings; llama-model loads `gemm.ptx` (prebuilt by gpu/gemm) through hand-written launchers in gemm/src/host.rs that mirror the generated marshalling |
