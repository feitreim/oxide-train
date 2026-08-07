"""Build, run, benchmark, and sweep rust-trainer's cuda-oxide kernels on a Modal GPU.

cuda-oxide is a rustc codegen backend (Rust -> PTX). The only place the full
toolchain can live is a Linux box with an NVIDIA GPU + CUDA 13 + LLVM 21, so we
bake all of that into a Modal image once and reuse it. (Adapted from the
cuda-learning setup; the local-fork backend override was dropped -- this repo
uses the stock upstream backend. Re-add a CUDA_OXIDE_BACKEND layer if needed.)

Local usage (see also ./run.sh):
    modal run modal_app.py --kernel vecadd               # correctness (main.rs)
    modal run modal_app.py --kernel vecadd --bin bench   # benchmark (src/bin/bench.rs)
    modal run modal_app.py --kernel gemm --bin bench --features cublas  # + cuBLASLt
    modal run modal_app.py --kernel gemm --sweep "BM=128 BN=128,BM=256 BN=128"
    modal run modal_app.py --kernel gemm --sanitize synccheck   # compute-sanitizer
    modal run modal_app.py::doctor                        # env / GPU sanity check
"""

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

# Keep this revision in sync with the git deps in gpu/*/Cargo.toml: the codegen
# backend and the device/host/core crates must come from the same revision.
CUDA_OXIDE_REF = "20a56163f258e09f2c51e4c27ae4e4ff17582443"
RUST_TOOLCHAIN = "nightly-2026-04-03"
GIT_REPO = "https://github.com/NVlabs/cuda-oxide.git"
TRAINER_REPO = "https://github.com/feitreim/oxide-train.git"

DEFAULT_GPU = "B200"  # training target; kernels will use tcgen05 features.
PROJECT_DIR = "/root/project"  # local gpu/ + crates/ mounted here at run time

# Mirror of the dependency block in gpu/vecadd/Cargo.toml. Used only to warm
# the backend + git-dep caches into an image layer so per-run builds are fast.
WARMUP_CARGO_TOML = f"""
[package]
name = "warmup"
version = "0.1.0"
edition = "2024"
[workspace]
[dependencies]
cuda-device = {{ git = "{GIT_REPO}", rev = "{CUDA_OXIDE_REF}" }}
cuda-host = {{ git = "{GIT_REPO}", rev = "{CUDA_OXIDE_REF}" }}
cuda-core = {{ git = "{GIT_REPO}", rev = "{CUDA_OXIDE_REF}" }}
"""

WARMUP_MAIN_RS = """
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(e) = c.get_mut(idx) { *e = a[i] + b[i]; }
    }
}
fn main() { let _ = (CudaContext::new(0), LaunchConfig::for_num_elems(1)); }
"""

kernel_toolchain = (
    # CUDA 13 devel base -- same as cuda-oxide's own .devcontainer/Dockerfile.
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12"
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "CUDA_TOOLKIT_PATH": "/usr/local/cuda",
            "CUDA_OXIDE_LLC": "/usr/bin/llc-21",
            "LIBCLANG_PATH": "/usr/lib/llvm-21/lib",
            "LLVM_CONFIG_PATH": "/usr/bin/llvm-config-21",
            "PATH": (
                "/root/.cargo/bin:/usr/lib/llvm-21/bin:"
                "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN,
        }
    )
    .apt_install(
        "ca-certificates", "curl", "g++", "gcc", "git", "gnupg",
        "libc6-dev", "libssl-dev", "make", "pkg-config", "xz-utils",
    )
    # LLVM 21 toolchain (NVPTX target + clang headers for bindgen).
    .run_commands(
        "curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key "
        "| gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg",
        'echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] '
        'https://apt.llvm.org/noble/ llvm-toolchain-noble-21 main" '
        "> /etc/apt/sources.list.d/llvm-toolchain-noble-21.list",
        "apt-get update && apt-get install -y --no-install-recommends "
        "clang-21 libclang-common-21-dev lld-21 llvm-21 llvm-21-dev "
        "&& rm -rf /var/lib/apt/lists/*",
    )
    # Pinned nightly Rust with the components the codegen backend needs.
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --default-toolchain none --profile minimal",
        f"rustup toolchain install {RUST_TOOLCHAIN} --profile minimal "
        "-c rust-src -c rustc-dev -c llvm-tools",
        f"cargo +{RUST_TOOLCHAIN} install --git {GIT_REPO} --rev {CUDA_OXIDE_REF} cargo-oxide",
    )
    # Build the codegen backend (slow, one time; baked into this image layer)
    # and compile a trivial kernel end-to-end to prove the toolchain works.
    #
    # cargo-oxide links libcuda (the *driver*), which isn't present at build
    # time (no GPU). The toolkit ships a driver *stub* that satisfies the
    # linker; expose it ONLY here via an inline LD_LIBRARY_PATH so it never
    # shadows the real driver injected at run time.
    .run_commands(
        "mkdir -p /opt/warmup/src",
        f"cat > /opt/warmup/Cargo.toml <<'EOF'\n{WARMUP_CARGO_TOML}\nEOF",
        f"cat > /opt/warmup/src/main.rs <<'EOF'\n{WARMUP_MAIN_RS}\nEOF",
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1",
        # `cargo oxide build` bootstraps and caches the backend on first use.
        # Do not call `setup` from a standalone project: at this revision that
        # command tries to rebuild the project itself as a backend library.
        "cd /opt/warmup && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs cargo oxide build warmup",
    )
)

MOUNT_IGNORE = ["**/target", "**/target/**", "**/__pycache__", "**/__pycache__/**"]

image = (
    kernel_toolchain
    # Live mounts (re-read each run; edits need no image rebuild). crates/ is
    # mounted because gpu/bench-util path-depends on crates/tensor-core (shared
    # RNG for CPU/GPU parity).
    #
    # `ignore` is not an optimization. Nothing is excluded by default, so the
    # mount is otherwise a gigabyte of local `target/` -- artifacts the
    # container rebuilds anyway -- which the client must walk and hash on every
    # run, and a local `cargo build` that writes into that tree while the walk
    # is in progress aborts the run.
    .add_local_dir(str(Path(__file__).parent / "gpu"), f"{PROJECT_DIR}/gpu", ignore=MOUNT_IGNORE)
    .add_local_dir(str(Path(__file__).parent / "crates"), f"{PROJECT_DIR}/crates", ignore=MOUNT_IGNORE)
    # CPU reference crates inherit package metadata and workspace dependencies
    # from the root manifest. Mount it so GPU correctness binaries can depend
    # on `nn`/`tensor-cpu` while retaining standalone workspaces under gpu/.
    .add_local_file(str(Path(__file__).parent / "Cargo.toml"), f"{PROJECT_DIR}/Cargo.toml")
    .add_local_file(
        str(Path(__file__).parent / "rust-toolchain.toml"),
        f"{PROJECT_DIR}/rust-toolchain.toml",
    )
)

# The kernel toolchain plus PyTorch, for the one question neither image alone
# can answer: how the trainer and the PyTorch baseline compare on ONE GPU under
# ONE set of container conditions. Torch's several gigabytes stay off `image`,
# so kernel runs are unaffected; only `torch_versus_train` pays for them.
TORCH_VERSION = "2.13.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

ab_image = (
    kernel_toolchain.pip_install(f"torch=={TORCH_VERSION}", index_url=TORCH_INDEX)
    .pip_install("numpy")
    # Peak memory is close enough to the B200's that fragmentation decides
    # whether the batch fits.
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rust-trainer", image=image)
wiki_volume = modal.Volume.from_name("rust-trainer-wiki", create_if_missing=True)


def _warn_unless_detached(entrypoint: str) -> None:
    """Say, before the GPU is spent, that this run dies with its client.

    An attached `modal run` owns the container: when the client goes away the
    input is cancelled and the app stops wherever it was, which in a log looks
    like a round that started and never finished. The entrypoints that take
    tens of minutes are the ones that lose to it, so they check the flag they
    needed rather than leaving it to be remembered.
    """
    if not {"-d", "--detach"} & set(sys.argv):
        print(
            f"WARNING: run this detached -- `modal run --detach modal_app.py::{entrypoint} ...`.\n"
            "  Attached, this container stops the moment the local client does, and the app's\n"
            "  log ends mid-round with `Stopping app - local client disconnected`.\n"
            "  Detached, `modal app logs <app-id>` re-attaches to a run already in progress.",
            flush=True,
        )


HEARTBEAT_SECONDS = 30.0


def _run(cmd: list[str], cwd: str, stamp: bool = False) -> str:
    """Run `cmd` to completion, echoing its output as it arrives, and return it.

    Callers that parse a number out of a run take it from the return value
    rather than from `capture_output=True`, because a captured run says nothing
    at all until it exits: a training round holds its whole log in a pipe for
    the minutes it takes, and if the app stops in that window -- a local `modal
    run` client that dies takes the container with it -- the round's output is
    lost with it. Echoed, every line is in the app's log the moment it is
    printed, and the log outlives the client.

    The heartbeat covers the one silence no child can report: `bin/train`
    prints nothing between its first instruction and the end of parameter
    initialization, which is minutes. A quiet window with a pulse in it is a
    run still working; a quiet window without one is the app being gone.

    `stamp` puts the elapsed time on every echoed line, which is how a run's
    internal phases get attributed -- the moment `training shard=` appears is
    the moment initialization ended. The returned text is never stamped, so a
    parser still sees exactly what the child printed.
    """
    print(f"$ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    started = quiet_since = time.monotonic()
    process = subprocess.Popen(
        cmd, cwd=cwd, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    finished = threading.Event()

    def pulse() -> None:
        while not finished.wait(HEARTBEAT_SECONDS):
            now = time.monotonic()
            if now - quiet_since >= HEARTBEAT_SECONDS:
                print(f"[+{now - started:.0f}s] still running, no output for "
                      f"{now - quiet_since:.0f}s", flush=True)

    threading.Thread(target=pulse, daemon=True).start()
    lines = []
    for line in process.stdout:
        quiet_since = time.monotonic()
        lines.append(line)
        prefix = f"[+{quiet_since - started:.0f}s] " if stamp else ""
        print(prefix + line, end="", flush=True)
    finished.set()
    code = process.wait()
    print(f"[+{time.monotonic() - started:.0f}s] exit={code}", flush=True)
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)
    return "".join(lines)


def _proj(kernel: str) -> str:
    import os

    proj = f"{PROJECT_DIR}/gpu/{kernel}"
    if not os.path.isdir(proj):
        raise SystemExit(f"no kernel project at gpu/{kernel}")
    return proj


def _resolve_ref(clone: str, ref: str) -> str:
    """Name `ref` as the fresh `clone` can see it.

    A clone's only local branch is the one it checked out; every other pushed
    branch is a remote-tracking ref. `git worktree add --detach` skips the DWIM
    that would otherwise infer `origin/<ref>`, so spell it out here.
    """
    for candidate in (ref, f"origin/{ref}"):
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],
            cwd=clone, capture_output=True,
        )
        if probe.returncode == 0:
            return candidate
    raise SystemExit(f"ref {ref} is neither a ref nor a branch on origin")


def _sha(root: str) -> str:
    probe = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return probe.stdout.strip() if probe.returncode == 0 else "unknown"


def _built_here(root: str, binary: str, label: str) -> None:
    """Name the commit a round is about to measure, and refuse a stale binary.

    A harness that builds once and launches the binary many times is only
    honest while the binary is the one that ref's sources produce. Nothing in
    these entry points hands a binary between trees -- each arm builds in its
    own worktree -- but "the binary is current" is the assumption every reused
    launch rests on, so it is checked rather than assumed, and the commit is
    printed so a number in a log can be traced to a tree later.

    The check is the binary's mtime against the newest source under it. That
    catches the failure that matters -- a build that silently did not happen,
    leaving an older artifact in place -- which is exactly how a prebuilt-
    binary shortcut turns a gate green on the wrong tree.
    """
    import os

    path = os.path.join(root, binary)
    if not os.path.isfile(path):
        raise SystemExit(f"{label}: {binary} was never built")
    built = os.path.getmtime(path)
    newest, newest_at = None, 0.0
    for directory, subdirectories, names in os.walk(root):
        subdirectories[:] = [d for d in subdirectories if d not in ("target", ".git")]
        for name in names:
            if name.endswith((".rs", ".toml", ".lock")):
                source = os.path.join(directory, name)
                if (when := os.path.getmtime(source)) > newest_at:
                    newest, newest_at = source, when
    if newest_at > built:
        raise SystemExit(
            f"{label}: {binary} is older than {os.path.relpath(newest, root)} "
            f"({built:.0f} < {newest_at:.0f}) -- it is not this tree's binary"
        )
    print(f"provenance {label}: sha={_sha(root)} {binary} built after every source", flush=True)


def _prepare_gemm_ptx(root: str, oxide: list[str] | None = None) -> None:
    """Prebuild gpu/gemm and stage its pure-PTX artifact for model.

    model's own device artifact is NVVM IR (its kernels use libdevice
    math), which cannot also carry tcgen05 lowerings; the model loads the
    tcgen05 GEMMs from this separately built gemm.ptx instead.
    """
    import shutil

    gemm = f"{root}/gpu/gemm"
    _run([*(oxide or ["cargo", "oxide"]), "build", "gemm"], cwd=gemm)
    shutil.copy(f"{gemm}/gemm.ptx", f"{root}/gpu/model/gemm.ptx")


def _prepare_flash_ptx(root: str, oxide: list[str] | None = None) -> None:
    """Prebuild gpu/flash-attn's binaries so the pure-PTX tcgen05 attention
    artifact (flash.ptx, emitted by the `flash` bin target) exists for the
    parity harness, and stage a copy for model's phase-3 integration.

    Same artifact split as gemm.ptx: the harness and model device artifacts
    go through libNVVM (libdevice math), which rejects tcgen05 constructs.
    """
    import shutil

    flash = f"{root}/gpu/flash-attn"
    # `build` compiles every bin target and, unlike `run`, does not
    # auto-detect the GPU arch. The oracle binaries use libdevice math plus
    # device atomics, which legacy NVVM IR rejects; pin the Blackwell target
    # (tcgen05 requires sm_100a anyway) so they take the NVVM path that
    # `cargo oxide run` would pick on the B200.
    _run(
        [*(oxide or ["cargo", "oxide"]), "build", "flash-attn", "--arch", "sm_100a"],
        cwd=flash,
    )
    if not Path(flash, "flash.ptx").is_file():
        raise SystemExit(
            "gpu/flash-attn built but produced no flash.ptx: the tcgen05 "
            "module picked up a libdevice lowering (even f32::max counts) "
            "and silently switched to NVVM IR. Check src/tcgen05.rs for "
            "libdevice math."
        )
    shutil.copy(f"{flash}/flash.ptx", f"{root}/gpu/model/flash.ptx")


@app.function(
    gpu=DEFAULT_GPU,
    # Not right-sized, deliberately: `--steps` is open-ended, so this entry
    # point has no baseline to be three times. The unit is measured -- 0.72s a
    # step at B=32 behind a ~120s initialization -- and 4h is where that puts
    # the step budget at ~19,000.
    timeout=4 * 3600,
    volumes={"/data": wiki_volume},
)
def run_kernel(
    kernel: str,
    bin: str | None = None,
    features: str | None = None,
    shard: str | None = None,
    steps: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    log_every: int | None = None,
    checkpoint: str | None = None,
    checkpoint_every: int | None = None,
    resume: bool = False,
) -> None:
    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    proj = _proj(kernel)
    cmd = ["cargo", "oxide", "run", kernel]
    if bin:
        cmd += ["--bin", bin]
    # Cargo features reach the build through cargo-oxide unchanged; gemm's
    # `cublas` is the only one, and it is what links the benchmark's
    # denominator.
    if features:
        cmd += ["--features", features]
    env = []
    if shard:
        env.append(f"TRAIN_SHARD={shard}")
    if steps:
        env.append(f"TRAIN_STEPS={steps}")
    if learning_rate is not None:
        env.append(f"TRAIN_LEARNING_RATE={learning_rate}")
    if weight_decay is not None:
        env.append(f"TRAIN_WEIGHT_DECAY={weight_decay}")
    if log_every:
        env.append(f"TRAIN_LOG_EVERY={log_every}")
    if checkpoint:
        env.append(f"TRAIN_CHECKPOINT={checkpoint}")
    if checkpoint_every:
        env.append(f"TRAIN_CHECKPOINT_EVERY={checkpoint_every}")
    if resume:
        env.append("TRAIN_RESUME=1")
    if env:
        cmd = ["env", *env, *cmd]
    _run(cmd, cwd=proj)


def _set_train_batch(proj: str, batch: int) -> None:
    """Rewrite `bin/train.rs`'s compile-time batch size in the mounted copy."""
    import re

    source = Path(proj, "src/bin/train.rs")
    patched, count = re.subn(
        r"(?m)^const B: usize = \d+;$",
        f"const B: usize = {batch};",
        source.read_text(),
    )
    if count != 1:
        raise SystemExit(f"expected one `const B: usize` in train.rs, found {count}")
    source.write_text(patched)


@app.function(
    gpu=DEFAULT_GPU,
    # ~200s per batch measured (a ~60s build, then a trainer process whose
    # 123.9s at three steps is almost all initialization). `--batches` sets how
    # many, so the cap is on the list rather than on a baseline: 2h is ~36.
    timeout=2 * 3600,
    volumes={"/data": wiki_volume},
)
def batch_sweep(batches: str, steps: int = 12, shard: str | None = None) -> None:
    """Measure the trainer's throughput at several batch sizes, in one container.

    `B` is a compile-time constant, so each batch is its own build of
    `bin/train.rs`. `N`, `NP` and `C` derive from `B`, which makes the whole
    reconfiguration one line to rewrite, and a build is only ~25s beside the
    minutes each batch spends initializing parameters.

    Sweeping the trainer rather than a harness that carries every batch size at
    once is not fastidiousness. cuda-oxide collects kernels from the selected
    binary target, so such a harness compiles different device code: one read
    12 -> 20 as +5.5% where the trainer measures +3.2%, agreeing at the small
    batch it was checked against and diverging at the large one.
    """
    _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], cwd="/")
    proj = _proj("model")
    # A log every hundredth step keeps the loss readback out of the timed
    # window, which is the only place the trainer would otherwise synchronize.
    env = [f"TRAIN_STEPS={steps}", "TRAIN_LOG_EVERY=100"]
    if shard:
        env.append(f"TRAIN_SHARD={shard}")

    for batch in filter(None, (value.strip() for value in batches.split(","))):
        _set_train_batch(proj, int(batch))
        print(f"=== train B={batch} ===", flush=True)
        try:
            _run(["env", *env, "cargo", "oxide", "run", "model", "--bin", "train"], cwd=proj)
        except subprocess.CalledProcessError as e:
            # A batch too large for HBM fails here, which is how the sweep finds
            # the ceiling; the batches after it still run.
            print(f"train B={batch} failed: {e}", flush=True)


# --------------------------------------------------------------------------
# The step-0 init checkpoint cache: MEASURED AND REJECTED. Do not build it.
#
# Parameter initialization is deterministic (seed 42) and is the longest window
# in a trainer process -- 119.5s of the 123.9s a three-step run takes. A
# `compare_train` A/B pays it six times to reach the same bytes six times, so
# the arithmetic says to derive it once and have every later run resume it,
# and that arithmetic is wrong. `init_cache_probe` measured it, at B=32:
#
#              save        load     resume vs fresh init (123.9s)
#   volume    174.5s      237.8s      -30.3s per run
#   disk       35.4s      139.0s      -14.0s per run
#
# Resuming is slower than initializing on BOTH storage tiers, and the second
# row is why: the container's own disk takes the file at 1.18 GiB/s and gives
# it back at 300.8 MiB/s, so the read is not bound by the filesystem -- it is
# bound by what `checkpoint::load` does with the bytes. The artifact is simply
# bigger than the work it replaces. Of its 40.85 GiB, 80% is AdamW moments
# that are ZERO at step 0 and that a fresh init never materializes at all:
# `GpuTensor::zeros` fills them on the device, over no bus. So the cache trades
# ~119s of host RNG and an 8.8 GiB upload for a 40.85 GiB read, 4.4B host bf16
# widenings, and a 35 GiB upload of zeros.
#
# What would change the verdict is the format, not the storage: a step-0
# artifact that omitted the zero moments would be ~8.6 GiB. That is a
# checkpoint v6, and `init_cache_probe` is kept so the claim can be re-measured
# the day someone writes one.
INIT_CACHE_DIR = "/data/init-cache"

# The compile-time shape of a trainer binary. Read out of the arm's own
# `bin/train.rs`, imposed on `bin/init_ckpt.rs`, and hashed into the cache
# filename -- so what derives an entry and what consumes it are the same
# numbers, from the same file.
SHAPE_CONSTS = ("B", "T", "N", "NP", "VOCAB", "VP", "D", "H", "HD", "FF", "E", "K", "C", "L")
CONST_LINE = r"(?m)^const {}: usize = (.+);$"


def _train_shape(proj: str) -> dict[str, str]:
    source = Path(proj, "src/bin/train.rs").read_text()
    shape = {}
    for name in SHAPE_CONSTS:
        found = re.search(CONST_LINE.format(name), source)
        if found is None:
            raise SystemExit(f"{proj}/src/bin/train.rs has no `const {name}: usize`")
        shape[name] = found[1].strip()
    return shape


def _impose_train_shape(proj: str, shape: dict[str, str]) -> None:
    """Give `bin/init_ckpt.rs` the shape of `bin/train.rs`.

    The two carry the same constants and a copy is a thing that drifts, so the
    copy is made at build time from the file that owns them. The guard behind
    this one is the checkpoint header, which records N/T/VOCAB/D/H/HD/FF/E/K/
    C/L and refuses a resume that disagrees -- drift costs a failed run, never
    a wrong number.
    """
    source = Path(proj, "src/bin/init_ckpt.rs")
    text = source.read_text()
    for name, value in shape.items():
        text, count = re.subn(
            CONST_LINE.format(name), f"const {name}: usize = {value};", text
        )
        if count != 1:
            raise SystemExit(f"expected one `const {name}: usize` in init_ckpt.rs, found {count}")
    source.write_text(text)


# What the checkpoint's contents depend on beyond the shape: the seed is fixed
# at 42 in both binaries, and these are the AdamW scalars `train.rs` refuses to
# resume across. Keyed on the values the harness actually imposes, so adding a
# knob later cannot silently reuse an entry built without it.
INIT_CACHE_CONFIG = {
    "learning_rate": "3e-4",
    "weight_decay": "0.1",
    "aux_coefficient": "1e-2",
    "aux_decay_horizon": "10000.0",
}


def _init_cache_entry(shape: dict[str, str]) -> tuple[str, str]:
    """`(path, key)` for the cache entry a binary of this shape can resume."""
    import hashlib

    key = " ".join(
        f"{name}={value}"
        for name, value in [*shape.items(), *INIT_CACHE_CONFIG.items(), ("seed", "42")]
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    name = f"init-b{shape['B']}-t{shape['T']}-d{shape['D']}-l{shape['L']}-{digest}.ckpt"
    return f"{INIT_CACHE_DIR}/{name}", key



# 827s measured (ap-m8a6S3jJuk60uNdgIzCuPW): a clone, a baseline cargo-oxide
# install, two builds and two profiled runs.
@app.function(gpu=DEFAULT_GPU, timeout=45 * 60)
def compare_profile(kernel: str, baseline_ref: str) -> None:
    """Build a retained git baseline and the mounted candidate, then profile
    both back-to-back in one container after each binary's equivalent warmups.
    """
    import os
    import re

    baseline_root = "/tmp/rust-trainer-baseline"
    _run(["git", "clone", "--quiet", TRAINER_REPO, baseline_root], cwd="/tmp")
    _run(["git", "checkout", "--quiet", baseline_ref], cwd=baseline_root)

    baseline = f"{baseline_root}/gpu/{kernel}"
    candidate = _proj(kernel)
    baseline_manifest = Path(baseline, "Cargo.toml").read_text()
    oxide_ref = re.search(
        r'cuda-oxide\.git",\s*(tag|rev)\s*=\s*"([^"]+)"',
        baseline_manifest,
    )
    if oxide_ref is None:
        raise SystemExit("baseline manifest has no cuda-oxide tag/rev")
    ref_kind, ref_value = oxide_ref.groups()
    baseline_oxide_root = "/tmp/cargo-oxide-baseline"
    _run(
        [
            "cargo",
            "install",
            "--git",
            GIT_REPO,
            f"--{ref_kind}",
            ref_value,
            "--root",
            baseline_oxide_root,
            "cargo-oxide",
        ],
        cwd="/tmp",
    )
    baseline_oxide = [
        "env",
        f"PATH={baseline_oxide_root}/bin:{os.environ['PATH']}",
        "cargo",
        "oxide",
    ]
    if kernel == "model":
        # Historical model refs infer `src/lib.rs` as a library target even
        # though each executable includes it directly for cuda-oxide kernel
        # discovery. Current Cargo/cuda-oxide then links both copies of every
        # device symbol. This build-only manifest setting keeps old source refs
        # profileable without changing their model or kernel implementation.
        manifest = Path(baseline) / "Cargo.toml"
        contents = baseline_manifest
        if "autolib = false" not in contents:
            contents = contents.replace(
                "[package]\n",
                "[package]\nautolib = false\n",
                1,
            )
            manifest.write_text(contents)
        # Preserve the staged artifacts only for historical refs that load
        # them. The current candidate always uses one embedded artifact.
        baseline_source = "\n".join(
            source.read_text() for source in Path(baseline, "src").rglob("*.rs")
        )
        if '"gemm.ptx"' in baseline_source:
            _prepare_gemm_ptx(baseline_root, baseline_oxide)
        if (
            '"flash.ptx"' in baseline_source
            and Path(baseline_root, "gpu/flash-attn/src/bin/flash.rs").is_file()
        ):
            _prepare_flash_ptx(baseline_root, baseline_oxide)
    _run([*baseline_oxide, "run", kernel, "--bin", "profile"], cwd=baseline)
    _run(["cargo", "oxide", "run", kernel, "--bin", "profile"], cwd=candidate)

    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    print(f"=== baseline {baseline_ref} ===", flush=True)
    _run(["target/release/profile"], cwd=baseline)
    print("=== candidate ===", flush=True)
    _run(["target/release/profile"], cwd=candidate)


@app.function(
    gpu=DEFAULT_GPU,
    # 848s / 1332s / 1413s / 1509s measured over four healthy runs at the
    # default two rounds; 3x the slowest. The old 4h could not distinguish a
    # wedged container from a long one, which is the whole job of a timeout.
    timeout=80 * 60,
    volumes={"/data": wiki_volume},
)
def compare_train(
    ref: str,
    baseline_ref: str = "main",
    steps: int = 12,
    rounds: int = 2,
    shard: str | None = None,
) -> None:
    """Alternate `bin/train` between two pushed refs in ONE container.

    `compare_profile` gives a fusion claim its same-container pair, but a
    throughput claim is quoted from `bin/train.rs`, whose numbers move with the
    GPU it landed on and the clocks it found there. Both arms are worktrees of
    the same clone -- neither is the mounted tree, so what is measured is
    exactly two committed refs, and the candidate must be pushed. `B` is a
    compile-time constant of train.rs and is whatever each ref says it is.

    Each arm is built once (that build's run is the discarded warmup), then the
    arms alternate round by round so a drift in clocks lands on both.
    """
    import re
    import statistics

    clone = "/tmp/rust-trainer-refs"
    _run(["git", "clone", "--quiet", TRAINER_REPO, clone], cwd="/tmp")
    arms = []
    for name, arm_ref in (("baseline", baseline_ref), ("candidate", ref)):
        root = f"/tmp/train-{name}"
        resolved = _resolve_ref(clone, arm_ref)
        _run(["git", "worktree", "add", "--quiet", "--detach", root, resolved], cwd=clone)
        proj = f"{root}/gpu/model"
        if CUDA_OXIDE_REF not in Path(proj, "Cargo.toml").read_text():
            raise SystemExit(f"{name} ref {arm_ref} pins a cuda-oxide the image lacks")
        print(f"arm {name}: ref={arm_ref} resolved={resolved} sha={_sha(root)}", flush=True)
        arms.append((f"{name} {arm_ref}", root, proj))

    env = ["env", f"TRAIN_STEPS={steps}", "TRAIN_LOG_EVERY=100"]
    if shard:
        env.append(f"TRAIN_SHARD={shard}")
    for _, _, proj in arms:
        _run([*env, "cargo", "oxide", "run", "model", "--bin", "train"], cwd=proj)
    # Every round below launches `target/release/train` directly -- one build
    # per arm, then `2 * rounds` launches of it, and cargo is never in the loop.
    # These two lines are what makes that safe to say out loud.
    for label, root, _ in arms:
        _built_here(root, "gpu/model/target/release/train", label)

    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    measured: dict[str, list[tuple[float, float]]] = {label: [] for label, _, _ in arms}
    for round_index in range(rounds):
        for label, _, proj in arms:
            print(f"=== round {round_index} {label} ===", flush=True)
            output = _run([*env, "target/release/train"], cwd=proj)
            throughput = re.search(r"throughput=([\d.]+) tokens/s mfu=([\d.]+)%", output)
            if throughput is None:
                raise SystemExit(f"{label} round {round_index} reported no throughput")
            measured[label].append((float(throughput[1]), float(throughput[2])))

    print("=== summary ===", flush=True)
    means = []
    for label, _, _ in arms:
        tokens, mfu = zip(*measured[label])
        means.append(statistics.fmean(tokens))
        runs = " ".join(f"{value:.1f}" for value in tokens)
        print(
            f"{label}: mean={means[-1]:.1f} tokens/s "
            f"spread={(max(tokens) - min(tokens)) / means[-1]:.2%} "
            f"mfu={statistics.fmean(mfu):.2f}% runs=[{runs}]",
            flush=True,
        )
    print(f"candidate/baseline={means[1] / means[0]:.4f}", flush=True)


@app.function(
    gpu=DEFAULT_GPU,
    # Measured 1550s at the default arms; 3x, rounded.
    timeout=80 * 60,
    volumes={"/data": wiki_volume},
)
def init_cache_probe(steps: int = 3, shard: str | None = None, volume_arm: bool = False) -> None:
    """Is resuming a step-0 checkpoint cheaper than deriving it, and is it the
    same run?

    Parameter initialization is the longest silent window in a trainer process
    and a `compare_train` A/B pays it six times over to reach the same bytes
    every time, so caching it is the obvious economy. This is the measurement
    that says whether it is one, and the answer is in `README.md`'s spending
    section. Two things have to hold and neither is arithmetic:

    - **Cheaper.** A fresh init is host RNG plus an upload of the *masters*;
      the AdamW moments it also needs are `GpuTensor::zeros`, filled on the
      device and never sent over the bus at all. The checkpoint carries those
      moments -- 80% of its 40.85 GiB -- as literal zeros that have to be read
      back, widened, and uploaded. The cached artifact is therefore *larger
      than the work it replaces*, and whether it still wins is a fact about the
      filesystem it sits on. Hence two storage arms: the wiki volume (a network
      filesystem, `--volume-arm`) and the container's own disk.
    - **The same.** A resumed step 0 has to be fresh initialization, or every
      number taken through the cache is taken on a different model. Loss
      equality is the readable form of that -- but a trainer process is not
      bit-identical to *itself* across runs (the embedding gradient accumulates
      atomically, so its order varies), and a comparison without that control
      would blame the resume for the hardware. So fresh runs *twice*, and the
      resumed arm is judged against the agreement two fresh arms manage.

    Everything runs from the mounted tree, one build, one GPU: this measures
    the cache, not two refs.
    """
    import os
    import shutil

    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    proj = _proj("model")
    shape = _train_shape(proj)
    _impose_train_shape(proj, shape)

    env = ["env", f"TRAIN_STEPS={steps}", "TRAIN_LOG_EVERY=1"]
    if shard:
        env.append(f"TRAIN_SHARD={shard}")
    derive_env = [
        f"TRAIN_{name.upper()}={value}" for name, value in INIT_CACHE_CONFIG.items()
    ]

    def timed(label: str, cmd: list[str], cwd: str = proj) -> tuple[float, str]:
        print(f"=== {label} ===", flush=True)
        started = time.monotonic()
        output = _run(cmd, cwd, stamp=True)
        return time.monotonic() - started, output

    def losses(output: str) -> dict[int, str]:
        return {int(step): value for step, value in re.findall(r"step=(\d+) loss=([\d.]+)", output)}

    # One build for both binaries; `cargo oxide run`'s own run is discarded,
    # exactly as `compare_train`'s warmup is, and every timed arm below then
    # launches an already-built binary.
    _run([*env, "cargo", "oxide", "run", "model", "--bin", "train"], cwd=proj)
    _run(
        ["env", "INIT_CKPT_MODE=shape",
         "cargo", "oxide", "run", "model", "--bin", "init_ckpt", "--features", "init-ckpt"],
        cwd=proj,
    )

    cost: list[tuple[str, float, str]] = []
    arms: dict[str, dict[int, str]] = {}

    first_seconds, first = timed("fresh init, run 1", [*env, "target/release/train"])
    cost.append(("fresh init", first_seconds, "what the cache would replace"))
    arms["fresh 1"] = losses(first)
    second_seconds, second = timed("fresh init, run 2", [*env, "target/release/train"])
    cost.append(("fresh init again", second_seconds, "the same binary, the same input"))
    arms["fresh 2"] = losses(second)

    stores = [("container disk", "/tmp/init-cache")]
    if volume_arm:
        stores.append(("wiki volume", INIT_CACHE_DIR))
    for store, directory in stores:
        os.makedirs(directory, exist_ok=True)
        path, key = _init_cache_entry(shape)
        path = f"{directory}/{Path(path).name}"
        save_seconds, saved = timed(
            f"derive onto {store}",
            ["env", f"TRAIN_CHECKPOINT={path}", *derive_env, "target/release/init_ckpt"],
        )
        cost.append((f"derive onto {store}", save_seconds, "once, then reused"))
        for line in saved.splitlines():
            if line.startswith(("init_checkpoint=", "init_seconds=")):
                print(f"  {line}", flush=True)
        if directory == INIT_CACHE_DIR:
            Path(f"{path}.key").write_text(f"{key}\n")
            wiki_volume.commit()

        load_seconds, loaded = timed(
            f"load from {store}",
            ["env", f"TRAIN_CHECKPOINT={path}", "INIT_CKPT_MODE=load", "target/release/init_ckpt"],
        )
        cost.append((f"load from {store}", load_seconds, "per run, in place of init"))
        for line in loaded.splitlines():
            if line.startswith("load_seconds="):
                print(f"  {line}", flush=True)

        resume_seconds, resumed = timed(
            f"resume from {store}",
            [*env, f"TRAIN_CHECKPOINT_INIT={path}", "target/release/train"],
        )
        cost.append((f"resume from {store}", resume_seconds, f"against fresh init's {first_seconds:.0f}s"))
        arms[f"resumed ({store})"] = losses(resumed)
        if directory != INIT_CACHE_DIR:
            shutil.rmtree(directory, ignore_errors=True)

    print("=== cost, one trainer process ===", flush=True)
    for label, seconds, note in cost:
        print(f"  {label:<28} {seconds:8.1f}s   {note}", flush=True)
    for store, _ in stores:
        delta = first_seconds - dict((label, s) for label, s, _ in cost)[f"resume from {store}"]
        verdict = "a saving" if delta > 0 else "a LOSS -- do not cache here"
        print(f"  {store}: {delta:+.1f}s per run -- {verdict}", flush=True)

    print("=== is it the same run? ===", flush=True)
    names = list(arms)
    print("  step  " + "  ".join(f"{name:>22}" for name in names), flush=True)
    for step in sorted({step for values in arms.values() for step in values}):
        print(
            f"  {step:<5} " + "  ".join(f"{arms[name].get(step, '-'):>22}" for name in names),
            flush=True,
        )
    control = arms["fresh 1"] == arms["fresh 2"]
    print(
        f"  two fresh runs agree at every step: {control}",
        flush=True,
    )
    for name, values in arms.items():
        if name.startswith("resumed"):
            print(
                f"  {name} vs fresh 1: {'identical' if values == arms['fresh 1'] else 'differs'}"
                f" -- judge against the control above, not against zero",
                flush=True,
            )


# Two-sided 95% critical values of Student's t, indexed by degrees of freedom.
# Round counts here are small enough that the normal 1.96 would understate the
# interval by a factor of two.
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}
BASELINE_SCRIPT = "gpu/model/baselines/pytorch_baseline.py"


@app.function(
    gpu=DEFAULT_GPU,
    image=ab_image,
    # Derived, not measured -- no run of this one survives in Modal's log
    # retention. From measured parts at the default three rounds: a ~90s build,
    # a ~130s trainer warmup, torch's own compile warmup, and six rounds of
    # ~130s is ~21 min. 90 min is ~4x that.
    timeout=90 * 60,
    volumes={"/data": wiki_volume},
)
def torch_versus_train(
    ref: str = "main",
    torch_ref: str = "pytorch-throughput-baseline",
    batch: int = 32,
    steps: int = 30,
    rounds: int = 3,
    compile_mode: str = "default",
    shard: str | None = None,
) -> None:
    """Alternate `bin/train` and the PyTorch baseline in ONE container.

    The two have only ever been measured in different containers, where the
    1.4% container-to-container spread of the trainer alone is wider than the
    ~1% that separates them -- so the comparison was unresolved by construction,
    not by measurement. This runs both arms on one GPU, alternating round by
    round like `compare_train`, so a drift in clocks lands on both.

    Both arms come from pushed refs of this repo: the trainer from `ref`, the
    baseline script from `torch_ref`, which keeps the baseline branch the source
    of truth for what PyTorch is asked to do. `batch` is imposed on both -- on
    the script by argument and on train.rs by rewriting its compile-time `B` --
    so a matched batch is a property of the run rather than of the two refs.

    Each arm gets a discarded warmup: the trainer's is the run that follows its
    build, torch's is a whole extra process, which also fills the inductor cache
    so the timed rounds recompile from disk. Compile time never lands in a timed
    window either way -- the script's own warmup steps absorb it.
    """
    import re
    import statistics

    clone = "/tmp/rust-trainer-versus"
    _run(["git", "clone", "--quiet", TRAINER_REPO, clone], cwd="/tmp")
    roots = {}
    for name, arm_ref in (("train", ref), ("torch", torch_ref)):
        roots[name] = f"/tmp/versus-{name}"
        _run(
            ["git", "worktree", "add", "--quiet", "--detach", roots[name], _resolve_ref(clone, arm_ref)],
            cwd=clone,
        )

    proj = f"{roots['train']}/gpu/model"
    if CUDA_OXIDE_REF not in Path(proj, "Cargo.toml").read_text():
        raise SystemExit(f"ref {ref} pins a cuda-oxide the image lacks")
    _set_train_batch(proj, batch)
    script = f"{roots['torch']}/{BASELINE_SCRIPT}"
    if not Path(script).is_file():
        raise SystemExit(f"ref {torch_ref} carries no {BASELINE_SCRIPT}")

    env = ["env", f"TRAIN_STEPS={steps}", "TRAIN_LOG_EVERY=100"]
    if shard:
        env.append(f"TRAIN_SHARD={shard}")
    train_cmd = [*env, "target/release/train"]
    # The script discards its first measured step exactly as train.rs does, so
    # one `steps` gives both arms the same measured window.
    torch_cmd = ["python", script, "--batch", str(batch), "--steps", str(steps)]
    torch_cmd += ["--masters", "bf16", "--warmup", "3"]
    if compile_mode:
        torch_cmd += ["--compile", compile_mode]
    if shard:
        torch_cmd += ["--shard", shard]

    def measure(label: str, cmd: list[str], cwd: str) -> tuple[float, float]:
        print(f"=== {label} ===", flush=True)
        output = _run(cmd, cwd)
        reported = re.search(r"throughput=([\d.]+) tokens/s mfu=([\d.]+)%", output)
        if reported is None:
            raise SystemExit(f"{label} reported no throughput")
        graphs = re.search(r"cuda_graphs=(\S+)", output)
        if graphs and graphs[1] != "False":
            # The trainer has no CUDA graphs and cuda-oxide cannot give it any,
            # so a graphed torch arm is a different comparison than this one.
            raise SystemExit(f"{label} recorded CUDA graphs ({graphs[1]}); not a matched arm")
        return float(reported[1]), float(reported[2])

    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    _run([*env, "cargo", "oxide", "run", "model", "--bin", "train"], cwd=proj)
    # One build, then `rounds` direct launches of it -- and `_set_train_batch`
    # rewrote train.rs above, so "the binary postdates its sources" is a live
    # question here rather than a formality.
    _built_here(roots["train"], "gpu/model/target/release/train", f"train {ref}")
    measure("warmup torch", torch_cmd, "/root")

    arms = (("train", train_cmd, proj), ("torch", torch_cmd, "/root"))
    measured: dict[str, list[tuple[float, float]]] = {name: [] for name, _, _ in arms}
    for index in range(rounds):
        for name, cmd, cwd in arms:
            measured[name].append(measure(f"round {index} {name}", cmd, cwd))

    print("=== rounds ===", flush=True)
    print("round  " + "  ".join(f"{name:>22}" for name, _, _ in arms), flush=True)
    for index in range(rounds):
        cells = (f"{measured[name][index][0]:>13.1f} {measured[name][index][1]:>7.2f}%" for name, _, _ in arms)
        print(f"{index:<7}" + "  ".join(cells), flush=True)

    print("=== summary ===", flush=True)
    statistics_by_arm = {}
    for name, _, _ in arms:
        tokens, mfu = zip(*measured[name])
        mean = statistics.fmean(tokens)
        deviation = statistics.stdev(tokens) if len(tokens) > 1 else 0.0
        statistics_by_arm[name] = (mean, deviation, len(tokens))
        print(
            f"{name}: mean={mean:.1f} tokens/s sd={deviation:.1f} ({deviation / mean:.2%}) "
            f"spread={(max(tokens) - min(tokens)) / mean:.2%} mfu={statistics.fmean(mfu):.2f}%",
            flush=True,
        )

    (train_mean, train_sd, count), (torch_mean, torch_sd, _) = (
        statistics_by_arm["train"],
        statistics_by_arm["torch"],
    )
    # Welch's interval with the conservative degrees of freedom -- one arm's
    # worth minus one, rather than the Satterthwaite value it can only exceed.
    # A single round per arm has no interval at all, and says so.
    error = (
        T95.get(count - 1, 2.23) * ((train_sd**2 + torch_sd**2) / count) ** 0.5
        if count > 1
        else float("inf")
    )
    difference = train_mean - torch_mean
    print(
        f"train/torch={train_mean / torch_mean:.4f} "
        f"difference={difference:+.1f} +/- {error:.1f} tokens/s "
        f"({difference / torch_mean:+.2%} +/- {error / torch_mean:.2%})",
        flush=True,
    )
    verdict = "ahead" if difference > error else ("behind" if -difference > error else "unresolved")
    print(f"verdict: trainer is {verdict} at 95%", flush=True)


# No baseline: each config is a fresh shape-specialized compile plus a
# correctness run and a bench, and `--configs` sets how many. Left at 1h.
@app.function(gpu=DEFAULT_GPU, timeout=3600)
def run_sweep(kernel: str, configs: str) -> None:
    """Bench several tuning configs in ONE container so they share a GPU and
    its clocks.

    `configs` is comma-separated; each config is space-separated `NAME=VAL`
    pairs, e.g. "BM=128 BN=128,BM=256 BN=64". Each NAME must exist in exactly
    one of the kernel crate's src/*.rs files as `pub const NAME: usize = ...;` -- tuning
    consts feed const generics, so every config is a fresh shape-specialized
    compile. Correctness (main.rs) runs before each bench so a bad config
    fails loudly. Container-side edits never touch the local checkout.
    """
    import re

    proj = _proj(kernel)
    sources = sorted(Path(proj, "src").glob("*.rs"))
    for cfg in configs.split(","):
        contents = {source: source.read_text() for source in sources}
        for assign in cfg.split():
            name, val = assign.split("=")
            matches = 0
            for source, src in contents.items():
                contents[source], n = re.subn(
                    rf"(pub const {name}: usize = )\d+", rf"\g<1>{val}", src
                )
                matches += n
            if matches != 1:
                raise SystemExit(
                    f"expected exactly one `pub const {name}: usize` in "
                    f"gpu/{kernel}/src/*.rs, found {matches}"
                )
        for source, src in contents.items():
            source.write_text(src)
        print(f"=== config {cfg} ===", flush=True)
        if kernel == "flash-attn":
            # The standalone flash binary is its correctness gate and bench.
            commands = [["cargo", "oxide", "run", kernel, "--bin", "flash"]]
        else:
            commands = [
                ["cargo", "oxide", "run", kernel],
                ["cargo", "oxide", "run", kernel, "--bin", "bench"],
            ]
        for cmd in commands:
            try:
                _run(cmd, cwd=proj)
            except (subprocess.CalledProcessError, SystemExit) as e:
                print(f"config failed: {e}", flush=True)
                break


# One nvcc compile of a single .cu and one run of it.
@app.function(gpu=DEFAULT_GPU, timeout=600)
def run_baseline(kernel: str, name: str) -> None:
    """Compile and run a CUDA C++ baseline from gpu/<kernel>/baselines/.

    Default flags are `-O3 -arch=native` (compile for the card we run on); a
    baseline needing more declares it in a leading `// nvcc-flags: ...` comment
    (e.g. `-arch=sm_100a -lcuda` for tcgen05 + the tensor-map driver API).
    """
    import os

    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    src = f"{_proj(kernel)}/baselines/{name}.cu"
    if not os.path.isfile(src):
        raise SystemExit(f"no baseline at gpu/{kernel}/baselines/{name}.cu")
    flags = ["-arch=native"]
    with open(src) as f:
        first = f.readline().strip()
    if first.startswith("// nvcc-flags:"):
        flags = first.removeprefix("// nvcc-flags:").split()
    _run(["nvcc", "-O3", *flags, "-o", f"/tmp/{name}", src], cwd="/")
    _run([f"/tmp/{name}"], cwd="/")


# Left at 1h: compute-sanitizer's instrumentation costs one to two orders of
# magnitude over the uninstrumented binary, so a build's runtime says little
# about this one's.
@app.function(gpu=DEFAULT_GPU, timeout=3600)
def run_sanitizer(kernel: str, bin: str | None = None, tool: str = "memcheck") -> None:
    """Run a kernel binary under compute-sanitizer (memcheck / racecheck /
    synccheck / initcheck).

    `cargo oxide run` builds and launches in one step, so to interpose the
    sanitizer we build first, then find the host binary under target/ and
    launch it ourselves.
    """
    import os

    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    proj = _proj(kernel)
    name = bin or kernel
    # `build` (unlike `run`) does not auto-detect the GPU arch, and the
    # oracle binaries' libdevice math + device atomics are rejected by the
    # legacy NVVM IR path. Pin the container default (B200); revisit if a
    # sanitizer run ever targets another GPU class.
    _run(["cargo", "oxide", "build", kernel, "--arch", "sm_100a"], cwd=proj)
    candidates = []
    for root, _, files in os.walk(f"{proj}/target"):
        for f in files:
            path = os.path.join(root, f)
            if f == name and os.access(path, os.X_OK):
                candidates.append(path)
    if not candidates:
        raise SystemExit(f"no built binary named {name} under gpu/{kernel}/target")
    binary = max(candidates, key=os.path.getmtime)
    _run(["compute-sanitizer", "--tool", tool, binary], cwd=proj)


# A build and a file read. Builds measured 14-90s across these crates; 20 min.
@app.function(gpu=DEFAULT_GPU, timeout=20 * 60)
def dump_ptx(kernel: str) -> str:
    proj = _proj(kernel)
    # `build` does not auto-detect the GPU arch the way `run` does; without the
    # pin, bins carrying device atomics or libdevice math fall back to legacy
    # NVVM IR and fail to compile at all. Same pin `_prepare_flash_ptx` uses.
    _run(["cargo", "oxide", "build", kernel, "--arch", "sm_100a"], cwd=proj)
    dumps = [
        Path(root, f)
        for root, _, files in __import__("os").walk(proj)
        for f in sorted(files)
        if f.endswith(".ptx")
    ]
    if not dumps:
        raise SystemExit(f"no .ptx produced under {proj}")
    return "\n".join(f"===== {p.relative_to(proj)} =====\n{p.read_text()}" for p in dumps)


# ptxas' per-entry report. `-v` writes one block per entry point to stderr; the
# stack frame line is omitted entirely when the frame is zero.
_PTXAS_ENTRY = re.compile(r"Function properties for '?([A-Za-z0-9_$]+)'?")
_PTXAS_FRAME = re.compile(r"(\d+) bytes stack frame")
_PTXAS_REGS = re.compile(r"Used (\d+) registers")


def _ptxas_report(ptx: Path, arch: str) -> dict[str, tuple[int, int]]:
    """`{entry: (registers, stack frame bytes)}` for one PTX file.

    This is the same allocator the driver's JIT runs when `cuModuleLoadData`
    takes the PTX text a `#[cuda_module]` embeds, so it answers the register
    question without a GPU.
    """
    result = subprocess.run(
        ["ptxas", f"--gpu-name={arch}", "-O3", "-v", str(ptx), "-o", "/dev/null"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ptxas failed on {ptx}:\n{result.stderr}")
    entries: dict[str, tuple[int, int]] = {}
    current = None
    frame = 0
    for line in result.stderr.splitlines():
        if name := _PTXAS_ENTRY.search(line):
            current, frame = name.group(1), 0
        if current and (bytes_ := _PTXAS_FRAME.search(line)):
            frame = int(bytes_.group(1))
        if current and (regs := _PTXAS_REGS.search(line)):
            entries[current] = (int(regs.group(1)), frame)
            current = None
    return entries


def _ptx_functions(text: str) -> dict[str, str]:
    """`{name: body}` for every function in one PTX file.

    LLVM brackets each one with `// -- Begin function <name>` / `// -- End
    function`, which is the only structure a permutation preserves — comparing
    these instead of the whole file is what tells a reordering apart from a
    codegen difference.
    """
    functions: dict[str, str] = {}
    name, body = None, []
    for line in text.splitlines():
        if begin := re.search(r"// -- Begin function ([A-Za-z0-9_$.]+)", line):
            name, body = begin.group(1), []
        elif "// -- End function" in line and name:
            functions[name] = "\n".join(body)
            name = None
        elif name:
            body.append(line)
    return functions


# Derived: `rounds` clean builds (measured 48-90s for a crate of this size)
# plus a ptxas pass each, so ~15 min at the default six. 90 min is ~6x, which
# leaves room for a larger `--rounds` without leaving a wedged CPU container
# to run for four hours.
@app.function(cpu=16, memory=32 * 1024, timeout=90 * 60)
def rebuild_stability(kernel: str = "gemm", rounds: int = 6, arch: str = "sm_100a") -> None:
    """Build ONE commit `rounds` times and ask whether the toolchain agrees
    with itself (oxide-train#122).

    Register allocation was observed to move between builds of an identical
    tree — 140 registers and no frame twice, 182 and a 2176 B frame once — and
    at a wide enough block that is `max_active_blocks_per_multiprocessor = 0`
    and a refused launch, not a slow one. Two layers can be responsible and
    this separates them, because the PTX is the boundary between them:

    - **codegen** (rustc + the cuda-oxide backend + `llc`) turns the crate into
      PTX text. If the PTX *bytes* move across rounds, it is this layer.
    - **ptxas** turns PTX into SASS and assigns registers. If the bytes are
      identical and the register counts still move, it is this one.

    Each round builds in its own copy of the tree so nothing is incremental and
    every rustc process is fresh. No GPU: `cargo oxide` links the driver stub,
    and `ptxas` is the allocator the driver's JIT runs anyway.
    """
    import hashlib
    import os
    import shutil

    # No GPU here, so no driver: `cargo oxide` links libcuda and the toolkit's
    # stub is what satisfies it, exactly as the image's own warmup build does.
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64/stubs"

    source = f"{PROJECT_DIR}"
    hashes: dict[str, list[str]] = {}
    reports: list[dict[str, tuple[int, int]]] = []
    keep = Path("/tmp/rounds")
    keep.mkdir(exist_ok=True)

    # Every round builds at the SAME path. Cargo's `-C metadata` disambiguator
    # is derived from it and rides inside every mangled generic symbol, so a
    # per-round directory would make the PTX differ for a reason that has
    # nothing to do with the question.
    root = "/tmp/build"
    orders: list[list[str]] = []
    for round_index in range(rounds):
        shutil.rmtree(root, ignore_errors=True)
        shutil.copytree(source, root, ignore=shutil.ignore_patterns("target"))
        proj = f"{root}/gpu/{kernel}"
        _run(
            ["cargo", "oxide", "build", kernel, "--arch", arch],
            cwd=proj,
        )
        dumps = sorted(
            Path(dirpath, name)
            for dirpath, _, names in os.walk(proj)
            for name in names
            if name.endswith(".ptx")
        )
        if not dumps:
            raise SystemExit(f"round {round_index}: no .ptx under gpu/{kernel}")
        report: dict[str, tuple[int, int]] = {}
        order: list[str] = []
        for ptx in dumps:
            name = str(ptx.relative_to(proj))
            text = ptx.read_text()
            hashes.setdefault(name, []).append(
                hashlib.sha256(text.encode()).hexdigest()[:12]
            )
            order += re.findall(r"^\.visible \.entry ([A-Za-z0-9_$]+)", text, re.M)
            shutil.copy(ptx, keep / f"{name.replace('/', '_')}.{round_index}")
            report.update(_ptxas_report(ptx, arch))
        reports.append(report)
        orders.append(order)

    print(f"\n===== {rounds} builds of gpu/{kernel} at {arch} =====", flush=True)
    print("\nPTX identity (sha256, first 12) — codegen's output")
    codegen_stable = True
    for name, digests in hashes.items():
        distinct = sorted(set(digests))
        codegen_stable &= len(distinct) == 1
        print(f"  {name:<24} {' '.join(digests)}   {len(distinct)} distinct")

    print("\nEntry emission order — what a byte diff is mostly made of")
    distinct_orders = {tuple(order) for order in orders}
    print(f"  {len(distinct_orders)} distinct orders over {rounds} builds")
    if len(distinct_orders) > 1:
        for round_index, order in enumerate(orders):
            print(f"  round {round_index}: {' '.join(order)}")

    print("\nptxas allocation (registers / stack frame bytes) — the allocator's output")
    names = sorted({entry for report in reports for entry in report})
    ptxas_stable = True
    for entry in names:
        cells = [report.get(entry) for report in reports]
        rendered = " ".join("-" if c is None else f"{c[0]}/{c[1]}" for c in cells)
        distinct = sorted({c for c in cells if c is not None})
        ptxas_stable &= len(distinct) == 1
        flap = "" if len(distinct) == 1 else "   ✗ flaps"
        print(f"  {entry:<40} {rendered}{flap}")

    print("\nPer-function body digest — is the difference a permutation, or the code?")
    bodies: dict[str, list[str]] = {}
    for round_index in range(rounds):
        for dump in sorted(keep.glob(f"*.{round_index}")):
            for function, body in _ptx_functions(dump.read_text()).items():
                bodies.setdefault(function, []).append(
                    hashlib.sha256(body.encode()).hexdigest()[:12]
                )
    permutation = True
    for function, digests in sorted(bodies.items()):
        distinct = len(set(digests))
        permutation &= distinct == 1 and len(digests) == rounds
        mark = "" if distinct == 1 else f"   ✗ {distinct} distinct bodies"
        print(f"  {function:<40} {digests[0]}{mark}")

    print()
    if codegen_stable and ptxas_stable:
        print("stable: identical PTX and identical allocation across every build")
    elif not codegen_stable and permutation and ptxas_stable:
        print(
            "CODEGEN emits the same functions in a nondeterministic ORDER: the PTX "
            "bytes move every build, every function body is byte-identical, and "
            "ptxas allocates the same registers regardless. Reproducibility bug, "
            "not (here) a register bug."
        )
    elif not codegen_stable:
        print(
            "CODEGEN is nondeterministic in the function BODIES, not only their "
            "order. Any register movement is downstream of that."
        )
    else:
        print(
            "PTXAS is nondeterministic: byte-identical PTX allocated differently. "
            "Report it upstream against this image's ptxas."
        )
    subprocess.run("ptxas --version", shell=True)


# A build, a ptxas report and one run: ~3 min. 20 min.
@app.function(gpu=DEFAULT_GPU, timeout=20 * 60)
def jit_probe(kernel: str = "gemm") -> None:
    """The other half of [`rebuild_stability`]: what the *driver* does with the
    PTX a build produced.

    `cargo oxide` emits `<kernel>.ptx` and `load_kernel_module` hands the text
    to `cuModuleLoad`, so register allocation happens in the driver's JIT at
    load time — not at build time. That makes the driver a build input nobody
    pins: Modal hands out whatever host it has, and the register counts a
    kernel gets are a function of (PTX bytes, driver version).

    So this prints all three together — the PTX digest, the driver, and what
    `cuFuncGetAttribute` says the loaded kernels got. Run it a few times; a
    constant digest beside a moving register count is the driver, and both
    moving together is the build.
    """
    import hashlib

    _run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
        cwd="/",
    )
    proj = _proj(kernel)
    _run(["cargo", "oxide", "build", kernel, "--arch", "sm_100a"], cwd=proj)
    print("=== the image's ptxas ===", flush=True)
    subprocess.run("ptxas --version | tail -2", shell=True)
    for ptx in sorted(Path(proj).glob("*.ptx")):
        digest = hashlib.sha256(ptx.read_bytes()).hexdigest()[:12]
        print(f"{ptx.name}: sha256 {digest}, {ptx.stat().st_size} bytes", flush=True)
        for entry, (registers, frame) in sorted(
            _ptxas_report(ptx, "sm_100a").items()
        ):
            print(f"  {entry:<40} {registers:>3} regs, {frame:>4} frame bytes")
    # And now the driver's own copy of it, which is the one that counts: these
    # numbers come back from `cuFuncGetAttribute` after `cuModuleLoad` JITs the
    # same text above. A disagreement here is the whole bug.
    print("=== the driver's JIT, via cuFuncGetAttribute ===", flush=True)
    subprocess.run(["cargo", "oxide", "run", kernel], cwd=proj)


@app.local_entrypoint()
def rebuilds(kernel: str = "gemm", rounds: int = 6, arch: str = "sm_100a") -> None:
    """modal run modal_app.py::rebuilds --kernel gemm --rounds 6"""
    rebuild_stability.remote(kernel, rounds, arch)


@app.local_entrypoint()
def jit(kernel: str = "gemm", gpu: str = "") -> None:
    """modal run modal_app.py::jit --kernel gemm"""
    fn = jit_probe.with_options(gpu=gpu) if gpu else jit_probe
    fn.remote(kernel)


# Left at 1h: `ncu --set full` replays every kernel once per metric pass and
# this one also apt-installs Nsight Systems, so the profiled run is nothing
# like the unprofiled one it was built from.
@app.function(gpu=DEFAULT_GPU, timeout=3600)
def profile(kernel: str, bin: str | None = None, features: str | None = None) -> None:
    """Run a kernel binary under Nsight Compute.

    `ncu` ships in the CUDA devel image at /usr/local/cuda/bin. As with the
    sanitizer, `cargo oxide run` builds and launches in one step, so the binary
    is built first and launched under the profiler by hand.

    Two passes over the same binary: a per-kernel summary (cheap, names and
    durations, which is what decodes a closed library's tile) and the full
    metric set as CSV (occupancy, launch geometry including the cluster dims,
    and the memory workload).
    """
    import os

    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    proj = _proj(kernel)
    name = bin or kernel
    build = ["cargo", "oxide", "build", kernel, "--arch", "sm_100a"]
    if features:
        build += ["--features", features]
    _run(build, cwd=proj)
    candidates = [
        os.path.join(root, f)
        for root, _, files in os.walk(f"{proj}/target")
        for f in files
        if f == name and os.access(os.path.join(root, f), os.X_OK)
    ]
    if not candidates:
        raise SystemExit(f"no built binary named {name} under gpu/{kernel}/target")
    binary = max(candidates, key=os.path.getmtime)

    print("=== profiler environment ===", flush=True)
    for probe in [
        "ncu --version",
        "ls -l /usr/local/cuda/bin/ncu",
        "ldconfig -p | grep -Ei 'nvperf|cupti|nvidia-ml|libcuda'",
        "ls /usr/local/cuda/extras/CUPTI/lib64 2>/dev/null | head",
        "ls /opt/nvidia/nsight-compute/*/target/linux-desktop-glibc_2_11_3-x64 2>/dev/null | head -30",
        "cat /proc/driver/nvidia/params | grep -i restrict",
        "cat /proc/self/status | grep -i cap",
    ]:
        print(f"$ {probe}", flush=True)
        subprocess.run(probe, shell=True)

    common = ["ncu", "--target-processes", "all", "--kernel-name-base", "demangled"]
    print("=== ncu summary ===", flush=True)
    subprocess.run([*common, "--print-summary", "per-kernel", binary], cwd=proj)
    print("=== ncu full metrics ===", flush=True)
    subprocess.run(
        [*common, "--set", "full", "--csv", "--page", "raw", binary],
        cwd=proj,
    )

    # CUPTI's activity API needs no performance counters: it reports a kernel's
    # name, grid, block, cluster, register count and shared memory straight from
    # the driver's launch record. That is most of what decodes a closed
    # library's configuration, and it survives a container where the counter
    # library does not. Nsight Systems is the packaged form of it.
    print("=== nsys ===", flush=True)
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
        "nsight-systems-2025.1.1 || apt-get install -y -qq --no-install-recommends nsight-systems",
        shell=True,
    )
    trace = subprocess.run(
        "nsys profile --trace=cuda --cuda-graph-trace=node --force-overwrite true "
        f"-o /tmp/decode {binary}",
        shell=True,
        cwd=proj,
    )
    if trace.returncode == 0:
        subprocess.run(
            "nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_trace "
            "--format csv /tmp/decode.nsys-rep",
            shell=True,
        )


# ~40s measured. 600s.
@app.function(gpu=DEFAULT_GPU, timeout=600)
def doctor() -> None:
    _run(["nvidia-smi"], cwd="/")
    _run(["cargo", "oxide", "doctor"], cwd="/opt/warmup")


@app.function(
    cpu=32,
    memory=64 * 1024,
    # Not right-sized: `--limit-files` is open-ended and the full wikimedia
    # dump is the intended input. Left at 20h.
    timeout=20 * 3600,
    volumes={"/data": wiki_volume},
)
def prepare_data(limit_files: int = 0, limit_articles: int = 0) -> None:
    """Tokenize wikimedia/wikipedia into u16 shards directly on the volume."""
    cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "data",
        "--bin",
        "prepare_wiki",
        "--",
        "--out",
        "/data",
    ]
    if limit_files:
        cmd += ["--limit-files", str(limit_files)]
    if limit_articles:
        cmd += ["--limit-articles", str(limit_articles)]
    _run(cmd, f"{PROJECT_DIR}/crates")
    wiki_volume.commit()


@app.local_entrypoint()
def prepare(limit_files: int = 0, limit_articles: int = 0) -> None:
    prepare_data.remote(limit_files, limit_articles)


@app.local_entrypoint()
def sweep_batch(batches: str = "12,16,20", steps: int = 12, shard: str = "") -> None:
    """modal run modal_app.py::sweep_batch --batches 12,16,20"""
    batch_sweep.remote(batches, steps, shard or None)


@app.local_entrypoint()
def init_cache(steps: int = 3, shard: str = "", volume_arm: bool = False) -> None:
    """modal run --detach modal_app.py::init_cache

    What a step-0 init checkpoint costs to derive, to store, and to resume,
    against deriving it fresh -- and whether a resumed run is the run it
    replaces. `--volume-arm` adds the wiki volume beside the container disk.
    Pass `--detach`: this is six trainer-sized processes.
    """
    _warn_unless_detached("init_cache")
    init_cache_probe.remote(steps, shard or None, volume_arm)


@app.local_entrypoint()
def train_ab(
    ref: str, baseline_ref: str = "main", steps: int = 12, rounds: int = 2, shard: str = ""
) -> None:
    """modal run --detach modal_app.py::train_ab --ref <pushed-branch>

    Pass `--detach`. An A/B is a clone, two builds and `2 * rounds + 2` training
    runs, tens of minutes at the operating points worth quoting, and an attached
    `modal run` ties the container's life to the local client: when the client
    goes away -- killed, timed out, or a dropped connection -- Modal cancels the
    input and stops the app mid-round. What that leaves in the log is a banner
    with nothing after it, which reads exactly like a hang and is not one.
    Detached, the run finishes without a client and `modal app logs <app-id>`
    re-attaches to it; `--show-timestamps` puts a clock on every line.
    """
    _warn_unless_detached("train_ab")
    compare_train.remote(ref, baseline_ref, steps, rounds, shard or None)


@app.local_entrypoint()
def versus(
    ref: str = "main",
    torch_ref: str = "pytorch-throughput-baseline",
    batch: int = 32,
    steps: int = 30,
    rounds: int = 3,
    compile_mode: str = "default",
    shard: str = "",
) -> None:
    """modal run --detach modal_app.py::versus --ref main --batch 32

    The trainer against the PyTorch baseline in one container. `--compile-mode`
    is a `torch.compile` mode, empty meaning eager; the arm is rejected if
    inductor recorded CUDA graphs, since the trainer has none.

    Pass `--detach`, for the reason `train_ab` gives: this one is longer still.
    """
    _warn_unless_detached("versus")
    torch_versus_train.remote(ref, torch_ref, batch, steps, rounds, compile_mode, shard or None)


@app.local_entrypoint()
def main(
    kernel: str = "vecadd",
    bin: str = "",
    features: str = "",
    gpu: str = "",
    ptx: bool = False,
    sweep: str = "",
    sanitize: str = "",
    baseline: str = "",
    shard: str = "",
    steps: int = 0,
    learning_rate: float = 0.0,
    weight_decay: float = -1.0,
    log_every: int = 0,
    checkpoint: str = "",
    checkpoint_every: int = 0,
    resume: bool = False,
    baseline_ref: str = "",
    ncu: bool = False,
) -> None:
    if ncu:
        fn = profile.with_options(gpu=gpu) if gpu else profile
        fn.remote(kernel, bin or None, features or None)
        return
    if sanitize:
        fn = run_sanitizer.with_options(gpu=gpu) if gpu else run_sanitizer
        fn.remote(kernel, bin or None, sanitize)
        return
    if baseline:
        fn = run_baseline.with_options(gpu=gpu) if gpu else run_baseline
        fn.remote(kernel, baseline)
        return
    if ptx:
        fn = dump_ptx.with_options(gpu=gpu) if gpu else dump_ptx
        print(fn.remote(kernel))
        return
    if sweep:
        fn = run_sweep.with_options(gpu=gpu) if gpu else run_sweep
        fn.remote(kernel, sweep)
        return
    if baseline_ref:
        fn = compare_profile.with_options(gpu=gpu) if gpu else compare_profile
        fn.remote(kernel, baseline_ref)
        return
    fn = run_kernel.with_options(gpu=gpu) if gpu else run_kernel
    fn.remote(
        kernel,
        bin or None,
        features or None,
        shard or None,
        steps or None,
        learning_rate or None,
        weight_decay if weight_decay >= 0.0 else None,
        log_every or None,
        checkpoint or None,
        checkpoint_every or None,
        resume,
    )


# `which` over six tools, three directory listings and an apt-cache search.
@app.function(gpu=DEFAULT_GPU, timeout=600)
def tools_check() -> None:
    """Is there a profiler in this image, and will the driver let it count?"""
    import shutil
    for tool in ["ncu", "nsys", "nvprof", "compute-sanitizer", "cuobjdump", "nvdisasm"]:
        print(f"{tool}: {shutil.which(tool)}", flush=True)
    subprocess.run("ls /usr/local/cuda/bin", shell=True)
    subprocess.run("ls -d /opt/nvidia/nsight-compute* /usr/local/cuda/nsight-compute* 2>/dev/null", shell=True)
    subprocess.run("cat /proc/driver/nvidia/params 2>/dev/null | grep -i perf", shell=True)
    subprocess.run("apt-get update -qq && apt-cache search nsight | head -20", shell=True)
    subprocess.run("nvidia-smi", shell=True)
