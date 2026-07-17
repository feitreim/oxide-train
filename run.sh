#!/usr/bin/env bash
# Build + run (or benchmark / sweep) a cuda-oxide kernel on a Modal GPU.
#
#   ./run.sh                          # vecadd correctness check
#   ./run.sh vecadd bench             # vecadd throughput benchmark
#   GPU=H100 ./run.sh vecadd bench    # pick a GPU (default: B200)
#   SWEEP="BM=128 BN=128,BM=256 BN=64" ./run.sh gemm   # tuning-const sweep
#   SANITIZE=synccheck ./run.sh gemm  # compute-sanitizer (memcheck/racecheck/synccheck/initcheck)
#   BASELINE=gemm_baseline ./run.sh gemm  # CUDA C++ baseline (gpu/<k>/baselines/<name>.cu)
#   PTX=1 ./run.sh gemm               # dump the generated PTX
set -euo pipefail
cd "$(dirname "$0")"

kernel="${1:-vecadd}"
bin="${2:-}"

args=(--kernel "$kernel")
[[ -n "$bin" ]] && args+=(--bin "$bin")
[[ -n "${GPU:-}" ]] && args+=(--gpu "$GPU")
[[ -n "${SWEEP:-}" ]] && args+=(--sweep "$SWEEP")
[[ -n "${SANITIZE:-}" ]] && args+=(--sanitize "$SANITIZE")
[[ -n "${BASELINE:-}" ]] && args+=(--baseline "$BASELINE")
[[ -n "${PTX:-}" ]] && args+=(--ptx)
[[ -n "${SHARD:-}" ]] && args+=(--shard "$SHARD")
[[ -n "${STEPS:-}" ]] && args+=(--steps "$STEPS")
[[ -n "${LR:-}" ]] && args+=(--learning-rate "$LR")
[[ -n "${WEIGHT_DECAY:-}" ]] && args+=(--weight-decay "$WEIGHT_DECAY")
[[ -n "${LOG_EVERY:-}" ]] && args+=(--log-every "$LOG_EVERY")
[[ -n "${CHECKPOINT:-}" ]] && args+=(--checkpoint "$CHECKPOINT")
[[ -n "${CHECKPOINT_EVERY:-}" ]] && args+=(--checkpoint-every "$CHECKPOINT_EVERY")
[[ -n "${RESUME:-}" ]] && args+=(--resume)

exec modal run modal_app.py::main "${args[@]}"
