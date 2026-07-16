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
- Element types: `f32`, `u16` (token ids), `u32`. `bf16` joins with the
  mixed-precision phase (§7).

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
  duplicated by an explicit combinator that owns that policy.
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

CPU reference reductions accumulate in f64 so the reference never loses to
the thing it checks.

## 8. Optimizers

- **AdamW first**: one fused elementwise GPU kernel over params/grads/m/v.
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

## 11. Repo layout

```
crates/            CPU-side workspace (builds/tests anywhere, no CUDA)
  tensor-core/     Shape markers, Element, shared RNG, Tensor trait
  tensor-cpu/      CpuTensor + naive reference ops
  nn/              Module trait, combinators, layers, gradcheck
  data/            tokenizer, shard format, mmap loader, prepare-wiki binary
  (planned) optim/ AdamW, Muon, param visitor
  (planned) train/ the training binary: config, loop, checkpoints, logging
gpu/               standalone cuda-oxide kernel crates (Modal-built)
  bench-util/      CUDA-event timing + shared-RNG re-export
  vecadd/          toolchain smoke test; template for new kernels
  (planned) tensor-gpu host-side GpuTensor wrapping DeviceBuffer + kernel launches
modal_app.py       Modal image + run/bench/sweep/sanitize/baseline/ptx
```

## 12. Milestones

Each gated on tests; correctness before speed at every step.

1. ✅ Scaffold: workspace, tensor-core/tensor-cpu, Module/Chain/Linear +
   gradcheck, Modal harness, vecadd smoke crate
2. ✅ Data: shard format, tokenizer, batcher, prepare-wiki binary
3. CPU model forward+backward: RMSNorm, RoPE attention, SwiGLU, embedding,
   fused softmax-cross-entropy — all gradchecked; overfit a tiny batch on CPU
4. GPU foundation: tensor-gpu (GpuTensor), elementwise/reduction kernels,
   naive-then-tiled GEMM — all parity-tested vs CPU
5. GPU forward+backward of the full model; parity vs CPU at fp32
6. AdamW + training loop on GPU; overfit tiny batch, then real wiki run
7. Perf: bf16 + fp32 master, kernel fusion, tcgen05 GEMM, flash-style
   attention, sweeps; Muon
8. Scale/stretch: bigger model, MoE, (much later) multi-GPU

## 13. Decision log

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
