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

import subprocess
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

image = (
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
    # Live mounts (re-read each run; edits need no image rebuild). crates/ is
    # mounted because gpu/bench-util path-depends on crates/tensor-core (shared
    # RNG for CPU/GPU parity).
    .add_local_dir(str(Path(__file__).parent / "gpu"), f"{PROJECT_DIR}/gpu")
    .add_local_dir(str(Path(__file__).parent / "crates"), f"{PROJECT_DIR}/crates")
    # CPU reference crates inherit package metadata and workspace dependencies
    # from the root manifest. Mount it so GPU correctness binaries can depend
    # on `nn`/`tensor-cpu` while retaining standalone workspaces under gpu/.
    .add_local_file(str(Path(__file__).parent / "Cargo.toml"), f"{PROJECT_DIR}/Cargo.toml")
    .add_local_file(
        str(Path(__file__).parent / "rust-toolchain.toml"),
        f"{PROJECT_DIR}/rust-toolchain.toml",
    )
)

app = modal.App("rust-trainer", image=image)
wiki_volume = modal.Volume.from_name("rust-trainer-wiki", create_if_missing=True)


def _run(cmd: list[str], cwd: str) -> None:
    print(f"$ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _proj(kernel: str) -> str:
    import os

    proj = f"{PROJECT_DIR}/gpu/{kernel}"
    if not os.path.isdir(proj):
        raise SystemExit(f"no kernel project at gpu/{kernel}")
    return proj


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
    timeout=4 * 3600,
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


@app.function(gpu=DEFAULT_GPU, timeout=3600)
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
    timeout=4 * 3600,
    volumes={"/data": wiki_volume},
)
def compare_train(
    baseline_ref: str,
    steps: int = 100,
    batch: int = 16,
    shard: str | None = None,
) -> None:
    """Train a retained git baseline and the mounted candidate back to back in
    one container.

    `compare_profile` does this for `bin/profile`, but a throughput claim is
    made with `bin/train.rs`, and SPEC §11 wants the before/after pair from one
    container. `B` is a compile-time constant, so both trees are rewritten to
    the same batch the way `batch_sweep` does.

    The baseline must pin the image's cuda-oxide revision; a ref that does not
    is `compare_profile`'s problem, which installs a matching toolchain.
    """
    baseline_root = "/tmp/rust-trainer-baseline"
    _run(["git", "clone", "--quiet", TRAINER_REPO, baseline_root], cwd="/tmp")
    _run(["git", "checkout", "--quiet", baseline_ref], cwd=baseline_root)
    baseline = f"{baseline_root}/gpu/model"
    if CUDA_OXIDE_REF not in Path(baseline, "Cargo.toml").read_text():
        raise SystemExit(f"baseline {baseline_ref} pins another cuda-oxide revision")

    env = ["env", f"TRAIN_STEPS={steps}", "TRAIN_LOG_EVERY=100"]
    if shard:
        env.append(f"TRAIN_SHARD={shard}")
    arms = [(f"baseline {baseline_ref}", baseline), ("candidate", _proj("model"))]
    for _, proj in arms:
        _set_train_batch(proj, batch)

    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    for name, proj in arms:
        print(f"=== {name} B={batch} steps={steps} ===", flush=True)
        _run([*env, "cargo", "oxide", "run", "model", "--bin", "train"], cwd=proj)


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


@app.function(gpu=DEFAULT_GPU, timeout=3600)
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


@app.function(gpu=DEFAULT_GPU, timeout=600)
def doctor() -> None:
    _run(["nvidia-smi"], cwd="/")
    _run(["cargo", "oxide", "doctor"], cwd="/opt/warmup")


@app.function(
    cpu=32,
    memory=64 * 1024,
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


@app.function(gpu=DEFAULT_GPU, timeout=1800)
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
