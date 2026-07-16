"""Build, run, benchmark, and sweep rust-trainer's cuda-oxide kernels on a Modal GPU.

cuda-oxide is a rustc codegen backend (Rust -> PTX). The only place the full
toolchain can live is a Linux box with an NVIDIA GPU + CUDA 13 + LLVM 21, so we
bake all of that into a Modal image once and reuse it. (Adapted from the
cuda-learning setup; the local-fork backend override was dropped -- this repo
uses the stock upstream backend. Re-add a CUDA_OXIDE_BACKEND layer if needed.)

Local usage (see also ./run.sh):
    modal run modal_app.py --kernel vecadd               # correctness (main.rs)
    modal run modal_app.py --kernel vecadd --bin bench   # benchmark (src/bin/bench.rs)
    modal run modal_app.py --kernel gemm --sweep "BM=128 BN=128,BM=256 BN=128"
    modal run modal_app.py --kernel gemm --sanitize synccheck   # compute-sanitizer
    modal run modal_app.py::doctor                        # env / GPU sanity check
"""

import subprocess
from pathlib import Path

import modal

# Keep this tag in sync with the git deps in gpu/*/Cargo.toml: the codegen
# backend and the device/host/core crates must come from the same revision.
CUDA_OXIDE_REF = "v0.2.1"
RUST_TOOLCHAIN = "nightly-2026-04-03"
GIT_REPO = "https://github.com/NVlabs/cuda-oxide.git"

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
cuda-device = {{ git = "{GIT_REPO}", tag = "{CUDA_OXIDE_REF}" }}
cuda-host = {{ git = "{GIT_REPO}", tag = "{CUDA_OXIDE_REF}" }}
cuda-core = {{ git = "{GIT_REPO}", tag = "{CUDA_OXIDE_REF}" }}
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
        "libc6-dev", "make", "pkg-config", "xz-utils",
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
        f"cargo +{RUST_TOOLCHAIN} install --git {GIT_REPO} --tag {CUDA_OXIDE_REF} cargo-oxide",
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
        "cd /opt/warmup && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs cargo oxide setup",
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


def _run(cmd: list[str], cwd: str) -> None:
    print(f"$ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _proj(kernel: str) -> str:
    import os

    proj = f"{PROJECT_DIR}/gpu/{kernel}"
    if not os.path.isdir(proj):
        raise SystemExit(f"no kernel project at gpu/{kernel}")
    return proj


@app.function(gpu=DEFAULT_GPU, timeout=3600)
def run_kernel(kernel: str, bin: str | None = None) -> None:
    _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd="/")
    proj = _proj(kernel)
    cmd = ["cargo", "oxide", "run", kernel]
    if bin:
        cmd += ["--bin", bin]
    _run(cmd, cwd=proj)


@app.function(gpu=DEFAULT_GPU, timeout=3600)
def run_sweep(kernel: str, configs: str) -> None:
    """Bench several tuning configs in ONE container so they share a GPU and
    its clocks.

    `configs` is comma-separated; each config is space-separated `NAME=VAL`
    pairs, e.g. "BM=128 BN=128,BM=256 BN=64". Each NAME must exist in the
    kernel crate's src/lib.rs as `pub const NAME: usize = ...;` -- tuning
    consts feed const generics, so every config is a fresh shape-specialized
    compile. Correctness (main.rs) runs before each bench so a bad config
    fails loudly. Container-side edits never touch the local checkout.
    """
    import re

    proj = _proj(kernel)
    lib = Path(proj, "src", "lib.rs")
    for cfg in configs.split(","):
        src = lib.read_text()
        for assign in cfg.split():
            name, val = assign.split("=")
            src, n = re.subn(
                rf"(pub const {name}: usize = )\d+", rf"\g<1>{val}", src
            )
            if n != 1:
                raise SystemExit(
                    f"expected exactly one `pub const {name}: usize` in "
                    f"gpu/{kernel}/src/lib.rs, found {n}"
                )
        lib.write_text(src)
        print(f"=== config {cfg} ===", flush=True)
        for cmd in (
            ["cargo", "oxide", "run", kernel],
            ["cargo", "oxide", "run", kernel, "--bin", "bench"],
        ):
            try:
                _run(cmd, cwd=proj)
            except subprocess.CalledProcessError as e:
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
    _run(["cargo", "oxide", "build", kernel], cwd=proj)
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
    _run(["cargo", "oxide", "build", kernel], cwd=proj)
    for root, _, files in __import__("os").walk(proj):
        for f in sorted(files):
            if f.endswith(".ptx"):
                return Path(root, f).read_text()
    raise SystemExit(f"no .ptx produced under {proj}")


@app.function(gpu=DEFAULT_GPU, timeout=600)
def doctor() -> None:
    _run(["nvidia-smi"], cwd="/")
    _run(["cargo", "oxide", "doctor"], cwd="/opt/warmup")


@app.local_entrypoint()
def main(
    kernel: str = "vecadd",
    bin: str = "",
    gpu: str = "",
    ptx: bool = False,
    sweep: str = "",
    sanitize: str = "",
    baseline: str = "",
) -> None:
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
    fn = run_kernel.with_options(gpu=gpu) if gpu else run_kernel
    fn.remote(kernel, bin or None)
