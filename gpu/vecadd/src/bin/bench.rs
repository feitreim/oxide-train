//! Vector addition — throughput microbenchmark.
//!
//! Times the kernel with CUDA events (device-side, not wall clock): WARMUP
//! launches to settle clocks/caches, then ITERS launches measured between two
//! recorded events. Vecadd is bandwidth-bound, so the figure of merit is GB/s.
//!
//! Run on a GPU (via Modal):  ./run.sh vecadd bench

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use vecadd::kernels;

const N: usize = 1 << 26; // 64M elements: big enough to saturate DRAM
const WARMUP: usize = 200;
const ITERS: usize = 1000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::from_module(ctx.load_module_from_file("vecadd.ptx")?)?;

    let a = DeviceBuffer::from_host(&stream, &uniform_vec(N, 1))?;
    let b = DeviceBuffer::from_host(&stream, &uniform_vec(N, 2))?;
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    let avg_ms = time_gpu_iters(&stream, WARMUP, ITERS, || {
        // SAFETY: the 1-D launch covers exactly N elements and all three
        // buffers hold N elements.
        unsafe {
            module.vecadd(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &a,
                &b,
                &mut c,
            )
        }
        .map_err(Into::into)
    })?;

    let secs = avg_ms / 1.0e3;
    // Traffic: read a, read b, write c — 3 x 4 bytes per element.
    let gbs = (3.0 * 4.0 * N as f64) / secs / 1.0e9;
    println!("vecadd  n={N}  avg={avg_ms:.4} ms  {gbs:.1} GB/s");

    // Copy one result down so a broken launch surfaces here, not silently.
    let c0 = c.to_host_vec(&stream)?[0];
    println!("\u{2713} result (c[0] = {c0})");
    Ok(())
}
