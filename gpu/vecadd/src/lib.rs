//! Toolchain smoke test: the canonical vecadd kernel.
//!
//! The `#[kernel]` function inside `#[cuda_module]` is compiled to PTX by the
//! cuda-oxide codegen backend and written to `vecadd.ptx` next to this crate.
//! Shared by `main.rs` (correctness) and `src/bin/bench.rs` (throughput) so
//! the kernel is defined exactly once.
//!
//! Each binary loads the PTX file directly (`ctx.load_module_from_file` +
//! `kernels::from_module`) rather than `kernels::load`: with the kernel in a
//! lib crate, the linker drops the embedded artifact from the binaries as
//! dead weight, so `kernels::load` would fail with `ModuleNotFound`.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}
