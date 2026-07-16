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

Plan: Llama-style model (RMSNorm, RoPE, SwiGLU, untied embeddings) at
~150-350M params, fp32 first then bf16 compute + fp32 master weights, AdamW
then Muon, trained on English Wikipedia tokenized with tiktoken `r50k_base`
(ids fit `u16`). MoE later.

## Layout

```
crates/                 CPU-side cargo workspace -- builds/tests on any machine
  tensor-core/          Shape (const-generic markers), Element, shared RNG
  tensor-cpu/           CpuTensor + naive reference ops
  nn/                   Module trait, Chain combinator, layers, gradcheck
gpu/                    standalone cuda-oxide crates -- built on Modal GPUs
  bench-util/           CUDA-event timing; re-exports the shared RNG
  vecadd/               toolchain smoke test (lib.rs kernel, main.rs check,
                        bin/bench.rs benchmark) -- the template for new kernels
modal_app.py            Modal image (CUDA 13 + LLVM 21 + pinned nightly +
                        cuda-oxide backend) and run/bench/sweep/sanitize entrypoints
run.sh                  thin wrapper over `modal run`
```

## CPU-side development (local)

```bash
cargo test          # tensor ops + gradchecks; no GPU or CUDA needed
```

The pinned nightly in `rust-toolchain.toml` matches the Modal image so local
tooling and GPU builds agree.

## GPU kernels (Modal)

```bash
pip install modal && modal setup        # once
modal run modal_app.py::doctor          # toolchain + GPU sanity check
./run.sh vecadd                         # correctness
./run.sh vecadd bench                   # throughput
SWEEP="BM=128 BN=128,BM=256 BN=64" ./run.sh gemm   # tuning sweep (one container)
```

The first run builds the Modal image (the cuda-oxide backend build is the slow
part); later runs reuse it and only recompile the kernel. Default GPU is B200
(`GPU=H100 ./run.sh ...` to override).

To add a kernel: copy `gpu/vecadd` to `gpu/<name>`, set `name` in its
`Cargo.toml`, write the `#[kernel]` in `src/lib.rs`, and give it a real
`bench.rs` figure of merit (GB/s if bandwidth-bound, TFLOP/s if compute-bound).
Expose tuning knobs as `pub const NAME: usize` in `lib.rs` so `SWEEP` can
rewrite them.
