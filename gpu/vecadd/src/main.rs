//! Vector addition — correctness check.
//!
//! Run on a GPU (via Modal):  ./run.sh vecadd

use bench_util::uniform_vec;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use vecadd::kernels;

fn main() {
    const N: usize = 1 << 20; // 1M elements

    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();

    let a_host = uniform_vec(N, 1);
    let b_host = uniform_vec(N, 2);

    let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::from_module(
        ctx.load_module_from_file("vecadd.ptx")
            .expect("load vecadd.ptx"),
    )
    .expect("wrap loaded module");
    module
        .vecadd(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a,
            &b,
            &mut c,
        )
        .expect("kernel launch");

    let c_host = c.to_host_vec(&stream).unwrap();

    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();
    if errors == 0 {
        println!(
            "\u{2713} vecadd correct over {N} elements (c[..3] = {:?})",
            &c_host[..3]
        );
    } else {
        eprintln!("\u{2717} {errors} mismatches");
        std::process::exit(1);
    }
}
