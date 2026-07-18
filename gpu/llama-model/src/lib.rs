//! GPU forward and backward for the single-block reference Llama. Aligned
//! training shapes run the lm-head and block linears through bf16 tcgen05
//! against fp32 master weights/gradients; small parity shapes retain the fp32
//! register-tiled block-linear oracle.
//!
//! Parameters, gradients, and saved activations remain GPU-resident. The
//! implementation mirrors `nn::Llama` explicitly so residual splits and the
//! aliasing story stay visible. Since 7e2, activations and scratch live in a
//! persistent `GpuLlamaWorkspace` reused across steps; safety comes from
//! disjoint workspace fields (each saved activation has a dedicated buffer),
//! not from the CPU reference's by-value Ctx ownership.
//!
//! Two padded dimensions keep every head GEMM inside the tcgen05 tile
//! contract without touching the tuned kernel:
//! - `VP` pads the vocabulary (50,257 -> 50,304). The padded weight columns
//!   are zero at initialization and stay zero: the classifier backward writes
//!   exact zeros there, so their gradients, moments, and decayed masters never
//!   move. Checkpoints store the unpadded columns only.
//! - `NP` pads the token rows to the 128-row tile. Padded rows of the head
//!   input are zeroed once and never written, so they contribute exactly
//!   nothing to any product (including the `K = NP` weight-gradient GEMM).

use std::error::Error;

use bench_util::{KernelProfiler, NoopProfiler};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, DriverError, LaunchConfig, PinnedHostBuffer};
use nn::{Llama, MoeLlama};
use optim::{
    AdamWConfig, AuxLossSchedule, MuonConfig, NEWTON_SCHULZ_A, NEWTON_SCHULZ_B, NEWTON_SCHULZ_C,
    NEWTON_SCHULZ_EPSILON,
};
use tensor_core::{Rank1, Rank2, Rank3, Rank4, Shape, bf16};

// cuda-oxide collects kernels from the selected binary target. The binary
// includes this file as a module, which in turn includes each canonical kernel
// source here instead of copying definitions or relying on dependency PTX.
//
// The tcgen05 GEMM kernels are the one exception: this binary's kernels use
// libdevice math (`exp`/`ln`/`sqrt`), which forces its device artifact
// through libNVVM, and libNVVM rejects tcgen05 lowerings. Only gpu/gemm's
// host-side support is included here; the kernels themselves load at runtime
// from the pure-PTX `gemm.ptx` that `cargo oxide build gemm` produces
// (modal_app.py prebuilds it for llama-model runs).
#[path = "../../flash-attn/src/lib.rs"]
mod flash_device;
#[path = "../../gemm/src/fp32.rs"]
mod gemm_device;
#[path = "../../gemm/src/host.rs"]
#[allow(dead_code)]
mod gemm_host;
#[path = "../../llama-ops/src/lib.rs"]
mod llama_device;
#[path = "../../tensor-gpu/src/lib.rs"]
#[allow(dead_code)]
pub mod tensor_device;

pub use flash_device::kernels as flash_kernels;
pub use gemm_device::kernels as gemm_kernels;
pub use gemm_host::Tcgen05Gemm;
pub use llama_device::kernels as llama_kernels;
pub use tensor_device::kernels as tensor_kernels;

use gemm_device::launch_config as fp32_launch_config;
use gemm_host::{
    Bf16PairsTmaMap, TC_TILE, create_bf16_pairs_tma_map, create_bf16_pairs_tma_map_prefix,
    create_bf16_pairs_tma_map_region, tcgen05_launch_config,
};
use tensor_device::{GpuAdamWMoments, GpuMuonMomentum, GpuTensor, transpose_pairs_config};

pub mod checkpoint;

fn elementwise_config<S: Shape>() -> LaunchConfig {
    assert!(S::NUM_ELEMENTS <= u32::MAX as usize);
    LaunchConfig::for_num_elems(S::NUM_ELEMENTS as u32)
}

fn reduction_config() -> LaunchConfig {
    assert!(tensor_device::REDUCE_THREADS.is_power_of_two());
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (tensor_device::REDUCE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn classifier_config<const N: usize>() -> LaunchConfig {
    assert!(llama_device::CLASSIFIER_THREADS.is_power_of_two());
    assert!(N <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (llama_device::CLASSIFIER_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn norm_config<const N: usize>() -> LaunchConfig {
    assert!(llama_device::NORM_THREADS.is_power_of_two());
    assert!(N <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (llama_device::NORM_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn norm_weight_config<const N: usize, const D: usize>() -> LaunchConfig {
    let threads = llama_device::NORM_THREADS;
    let rows_per_block = llama_device::NORM_WEIGHT_ROWS_PER_BLOCK;
    assert!(threads.is_power_of_two());
    assert!(N <= u32::MAX as usize && D <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            D.div_ceil(threads) as u32,
            N.div_ceil(rows_per_block) as u32,
            1,
        ),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn pairs_config(words: usize) -> LaunchConfig {
    assert!(words <= u32::MAX as usize);
    LaunchConfig::for_num_elems(words as u32)
}

fn flash_forward_config<const N: usize, const T: usize, const H: usize, const HD: usize>()
-> LaunchConfig {
    assert_eq!(N % T, 0);
    flash_device::tiled_forward_config(N / T, T, H, HD)
}

fn flash_dot_config<const N: usize, const H: usize, const HD: usize>() -> LaunchConfig {
    flash_device::dot_config(N, H, HD)
}

fn flash_backward_q_config<const N: usize, const T: usize, const H: usize, const HD: usize>()
-> LaunchConfig {
    assert_eq!(N % T, 0);
    flash_device::tiled_backward_q_config(N / T, T, H, HD)
}

fn flash_backward_kv_config<const N: usize, const T: usize, const H: usize, const HD: usize>()
-> LaunchConfig {
    assert_eq!(N % T, 0);
    flash_device::tiled_backward_kv_config(N / T, T, H, HD)
}

fn add_into<S: Shape, P: KernelProfiler>(
    lhs: &GpuTensor<f32, S>,
    rhs: &GpuTensor<f32, S>,
    output: &mut GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.add(
            stream,
            elementwise_config::<S>(),
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })
}

fn fill_zero<S: Shape, P: KernelProfiler>(
    output: &mut GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.fill(
            stream,
            elementwise_config::<S>(),
            0.0,
            output.as_device_buffer_mut(),
        )
    })
}

fn sum_into<S: Shape, P: KernelProfiler>(
    input: &GpuTensor<f32, S>,
    output: &mut GpuTensor<f32, Rank1<1>>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.sum(
            stream,
            reduction_config(),
            input.as_device_buffer(),
            S::NUM_ELEMENTS as u32,
            output.as_device_buffer_mut(),
        )
    })
}

fn scale_into<S: Shape, P: KernelProfiler>(
    input: &GpuTensor<f32, S>,
    factor: f32,
    output: &mut GpuTensor<f32, S>,
    stream: &CudaStream,
    kernels: &tensor_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.scale(
            stream,
            elementwise_config::<S>(),
            input.as_device_buffer(),
            factor,
            output.as_device_buffer_mut(),
        )
    })
}

fn gemm_into<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &GpuTensor<f32, Rank2<K, N>>,
    output: &mut GpuTensor<f32, Rank2<M, N>>,
    stream: &CudaStream,
    kernels: &gemm_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || unsafe {
        kernels.register_gemm_store(
            stream,
            fp32_launch_config(M, N),
            M,
            N,
            K,
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })
}

fn gemm_tn_accumulate_into<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &GpuTensor<f32, Rank2<M, N>>,
    output: &mut GpuTensor<f32, Rank2<K, N>>,
    stream: &CudaStream,
    kernels: &gemm_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || unsafe {
        kernels.register_gemm_tn_accumulate(
            stream,
            fp32_launch_config(K, N),
            K,
            N,
            M,
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })
}

fn gemm_nt_into<const M: usize, const K: usize, const N: usize, P: KernelProfiler>(
    lhs: &GpuTensor<f32, Rank2<M, K>>,
    rhs: &GpuTensor<f32, Rank2<N, K>>,
    output: &mut GpuTensor<f32, Rank2<M, N>>,
    stream: &CudaStream,
    kernels: &gemm_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || unsafe {
        kernels.register_gemm_nt_store(
            stream,
            fp32_launch_config(M, N),
            M,
            N,
            K,
            lhs.as_device_buffer(),
            rhs.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })
}

fn copy_device_region(
    destination: &mut DeviceBuffer<f32>,
    destination_offset: usize,
    source: &DeviceBuffer<f32>,
    source_offset: usize,
    elements: usize,
    stream: &CudaStream,
) -> Result<(), DriverError> {
    let destination_end = destination_offset
        .checked_add(elements)
        .expect("device copy destination region overflow");
    let source_end = source_offset
        .checked_add(elements)
        .expect("device copy source region overflow");
    assert!(destination_end <= destination.len());
    assert!(source_end <= source.len());
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .expect("device copy byte count overflow");
    let destination_bytes = destination_offset
        .checked_mul(std::mem::size_of::<f32>())
        .expect("device copy destination byte offset overflow");
    let source_bytes = source_offset
        .checked_mul(std::mem::size_of::<f32>())
        .expect("device copy source byte offset overflow");
    let destination = destination
        .cu_deviceptr()
        .checked_add(destination_bytes as u64)
        .expect("device copy destination pointer overflow");
    let source = source
        .cu_deviceptr()
        .checked_add(source_bytes as u64)
        .expect("device copy source pointer overflow");
    // SAFETY: the checked element ranges above are within their allocations.
    // Expert staging always copies between distinct allocations.
    unsafe { cuda_core::memory::memcpy_dtod_async(destination, source, bytes, stream.cu_stream()) }
}

struct Bf16LinearWeights {
    normal: DeviceBuffer<u32>,
    transposed: DeviceBuffer<u32>,
    normal_tma: Bf16PairsTmaMap,
    transposed_tma: Bf16PairsTmaMap,
}

impl Bf16LinearWeights {
    fn new(
        stream: &CudaStream,
        values: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Self, Box<dyn Error>> {
        assert_eq!(values.len(), rows * columns);
        let pack = |low: f32, high: f32| {
            bf16::from_f32(low).to_bits() as u32 | ((bf16::from_f32(high).to_bits() as u32) << 16)
        };
        let packed: Vec<u32> = values
            .chunks_exact(2)
            .map(|pair| pack(pair[0], pair[1]))
            .collect();
        let mut packed_t = vec![0u32; rows * columns / 2];
        for column in 0..columns {
            for pair in 0..rows / 2 {
                packed_t[column * rows / 2 + pair] = pack(
                    values[2 * pair * columns + column],
                    values[(2 * pair + 1) * columns + column],
                );
            }
        }
        let normal = DeviceBuffer::from_host(stream, &packed)?;
        let transposed = DeviceBuffer::from_host(stream, &packed_t)?;
        // SAFETY: both allocations live beside their maps and are never
        // replaced. Optimizer refreshes mutate their contents in place.
        let normal_tma = unsafe { create_bf16_pairs_tma_map(stream, &normal, columns, rows)? };
        let transposed_tma =
            unsafe { create_bf16_pairs_tma_map(stream, &transposed, rows, columns)? };
        Ok(Self {
            normal,
            transposed,
            normal_tma,
            transposed_tma,
        })
    }

    fn sync_from_master(
        &mut self,
        master: &DeviceBuffer<f32>,
        rows: usize,
        columns: usize,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        kernels.convert_f32_to_bf16_pairs(
            stream,
            pairs_config(rows * columns / 2),
            master,
            &mut self.normal,
        )?;
        unsafe {
            kernels.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(rows, columns),
                &self.normal,
                rows as u32,
                columns as u32,
                &mut self.transposed,
            )
        }
    }
}

struct Bf16LinearMaps {
    d: Bf16PairsTmaMap,
    ff: Bf16PairsTmaMap,
    qkv: Bf16PairsTmaMap,
    gate_up: Bf16PairsTmaMap,
}

impl Bf16LinearMaps {
    fn get<const D: usize, const FF: usize>(&self, width: usize) -> &Bf16PairsTmaMap {
        if width == D {
            &self.d
        } else if width == FF {
            &self.ff
        } else if width == 3 * D {
            &self.qkv
        } else if width == 2 * FF {
            &self.gate_up
        } else {
            panic!("unsupported tcgen05 linear width {width}")
        }
    }
}

/// Reusable packed-bf16 operand storage for all block-linear GEMMs.
///
/// `rows` holds an `[N,width]` operand. `lhs_t` and `rhs_t` retain both
/// transposed operands for the fp32-accumulating weight-gradient launch.
struct Bf16LinearScratch<const N: usize, const D: usize, const FF: usize> {
    rows: DeviceBuffer<u32>,
    lhs_t: DeviceBuffer<u32>,
    rhs_t: DeviceBuffer<u32>,
    row_maps: Bf16LinearMaps,
    lhs_t_maps: Bf16LinearMaps,
    rhs_t_maps: Bf16LinearMaps,
}

impl<const N: usize, const D: usize, const FF: usize> Bf16LinearScratch<N, D, FF> {
    fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        let max_width = D.max(FF).max(3 * D).max(2 * FF);
        let rows = DeviceBuffer::zeroed(stream, N * max_width / 2)?;
        let lhs_t = DeviceBuffer::zeroed(stream, N * max_width / 2)?;
        let rhs_t = DeviceBuffer::zeroed(stream, N * max_width / 2)?;

        let row_maps = Self::maps(stream, &rows, false)?;
        let lhs_t_maps = Self::maps(stream, &lhs_t, true)?;
        let rhs_t_maps = Self::maps(stream, &rhs_t, true)?;
        Ok(Self {
            rows,
            lhs_t,
            rhs_t,
            row_maps,
            lhs_t_maps,
            rhs_t_maps,
        })
    }

    fn maps(
        stream: &CudaStream,
        buffer: &DeviceBuffer<u32>,
        transposed: bool,
    ) -> Result<Bf16LinearMaps, Box<dyn Error>> {
        let make = |width| unsafe {
            if transposed {
                create_bf16_pairs_tma_map_prefix(stream, buffer, N, width)
            } else {
                create_bf16_pairs_tma_map_prefix(stream, buffer, width, N)
            }
        };
        Ok(Bf16LinearMaps {
            d: make(D)?,
            ff: make(FF)?,
            qkv: make(3 * D)?,
            gate_up: make(2 * FF)?,
        })
    }
}

fn tcgen05_linear_eligible(m: usize, k: usize, n: usize) -> bool {
    m.is_multiple_of(TC_TILE) && k.is_multiple_of(TC_TILE) && n.is_multiple_of(TC_TILE)
}

pub struct GpuLinear<const IN: usize, const OUT: usize> {
    pub w: GpuTensor<f32, Rank2<IN, OUT>>,
    pub dw: GpuTensor<f32, Rank2<IN, OUT>>,
    compute: Option<Bf16LinearWeights>,
}

impl<const IN: usize, const OUT: usize> GpuLinear<IN, OUT> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layer: &nn::Linear<N, IN, OUT>,
    ) -> Result<Self, Box<dyn Error>> {
        let compute = if IN.is_multiple_of(TC_TILE) && OUT.is_multiple_of(TC_TILE) {
            Some(Bf16LinearWeights::new(stream, layer.w.as_slice(), IN, OUT)?)
        } else {
            None
        };
        Ok(Self {
            w: GpuTensor::from_cpu(stream, &layer.w)?,
            dw: GpuTensor::zeros(stream)?,
            compute,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_into<const N: usize, const D: usize, const FF: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        output: &mut GpuTensor<f32, Rank2<N, OUT>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        scratch: Option<&mut Bf16LinearScratch<N, D, FF>>,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        if let (Some(compute), Some(scratch)) = (&self.compute, scratch)
            && tcgen05_linear_eligible(N, IN, OUT)
        {
            profiler.measure(stream, name, || {
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * IN / 2),
                    x.as_device_buffer(),
                    &mut scratch.rows,
                )?;
                unsafe {
                    tcgen05.f32_store(
                        stream,
                        tcgen05_launch_config(N, OUT, IN),
                        scratch.row_maps.get::<D, FF>(IN).as_ptr(),
                        compute.transposed_tma.as_ptr(),
                        output.as_device_buffer_mut(),
                        OUT as u32,
                        IN as u32,
                    )
                }
            })
        } else {
            gemm_into(x, &self.w, output, stream, fp32, profiler, name)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_into<const N: usize, const D: usize, const FF: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        dy: &GpuTensor<f32, Rank2<N, OUT>>,
        dx: &mut GpuTensor<f32, Rank2<N, IN>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        scratch: Option<&mut Bf16LinearScratch<N, D, FF>>,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<(), DriverError> {
        if let (Some(compute), Some(scratch)) = (&self.compute, scratch)
            && tcgen05_linear_eligible(N, IN, OUT)
        {
            profiler.measure(stream, names[0], || {
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * IN / 2),
                    x.as_device_buffer(),
                    &mut scratch.rows,
                )?;
                unsafe {
                    tensor.transpose_bf16_pairs(
                        stream,
                        transpose_pairs_config(N, IN),
                        &scratch.rows,
                        N as u32,
                        IN as u32,
                        &mut scratch.lhs_t,
                    )?;
                }
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * OUT / 2),
                    dy.as_device_buffer(),
                    &mut scratch.rows,
                )?;
                unsafe {
                    tensor.transpose_bf16_pairs(
                        stream,
                        transpose_pairs_config(N, OUT),
                        &scratch.rows,
                        N as u32,
                        OUT as u32,
                        &mut scratch.rhs_t,
                    )?;
                    tcgen05.f32_accumulate(
                        stream,
                        tcgen05_launch_config(IN, OUT, N),
                        scratch.lhs_t_maps.get::<D, FF>(IN).as_ptr(),
                        scratch.rhs_t_maps.get::<D, FF>(OUT).as_ptr(),
                        self.dw.as_device_buffer_mut(),
                        OUT as u32,
                        N as u32,
                    )
                }
            })?;
            // `scratch.rows` still holds the quantized `dy` written by the
            // weight-gradient pass above; this launch consumes it as its row
            // operand, so nothing may overwrite `rows` between the two.
            profiler.measure(stream, names[1], || unsafe {
                tcgen05.f32_store(
                    stream,
                    tcgen05_launch_config(N, IN, OUT),
                    scratch.row_maps.get::<D, FF>(OUT).as_ptr(),
                    compute.normal_tma.as_ptr(),
                    dx.as_device_buffer_mut(),
                    IN as u32,
                    OUT as u32,
                )
            })
        } else {
            gemm_tn_accumulate_into(x, dy, &mut self.dw, stream, fp32, profiler, names[0])?;
            gemm_nt_into(dy, &self.w, dx, stream, fp32, profiler, names[1])
        }
    }

    fn sync_compute(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        if let Some(compute) = &mut self.compute {
            compute.sync_from_master(self.w.as_device_buffer(), IN, OUT, stream, kernels)?;
        }
        Ok(())
    }
}

pub struct GpuGroupedLinear<const IN: usize, const GROUPS: usize, const OUT: usize> {
    pub w: GpuTensor<f32, Rank3<IN, GROUPS, OUT>>,
    pub dw: GpuTensor<f32, Rank3<IN, GROUPS, OUT>>,
    compute: Option<Bf16LinearWeights>,
}

impl<const IN: usize, const GROUPS: usize, const OUT: usize> GpuGroupedLinear<IN, GROUPS, OUT> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layers: [&nn::Linear<N, IN, OUT>; GROUPS],
    ) -> Result<Self, Box<dyn Error>> {
        let mut weights = vec![0.0; IN * GROUPS * OUT];
        for input in 0..IN {
            for (group, layer) in layers.iter().enumerate() {
                let source = &layer.w.as_slice()[input * OUT..(input + 1) * OUT];
                let destination = (input * GROUPS + group) * OUT;
                weights[destination..destination + OUT].copy_from_slice(source);
            }
        }
        let compute = if IN.is_multiple_of(TC_TILE) && (GROUPS * OUT).is_multiple_of(TC_TILE) {
            Some(Bf16LinearWeights::new(stream, &weights, IN, GROUPS * OUT)?)
        } else {
            None
        };
        Ok(Self {
            w: GpuTensor::from_host(stream, &weights)?,
            dw: GpuTensor::zeros(stream)?,
            compute,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_into<const N: usize, const D: usize, const FF: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        output: &mut GpuTensor<f32, Rank3<N, GROUPS, OUT>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        scratch: Option<&mut Bf16LinearScratch<N, D, FF>>,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        let width = GROUPS * OUT;
        if let (Some(compute), Some(scratch)) = (&self.compute, scratch)
            && tcgen05_linear_eligible(N, IN, width)
        {
            profiler.measure(stream, name, || {
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * IN / 2),
                    x.as_device_buffer(),
                    &mut scratch.rows,
                )?;
                unsafe {
                    tcgen05.f32_store(
                        stream,
                        tcgen05_launch_config(N, width, IN),
                        scratch.row_maps.get::<D, FF>(IN).as_ptr(),
                        compute.transposed_tma.as_ptr(),
                        output.as_device_buffer_mut(),
                        width as u32,
                        IN as u32,
                    )
                }
            })
        } else {
            profiler.measure(stream, name, || unsafe {
                fp32.register_gemm_store(
                    stream,
                    fp32_launch_config(N, width),
                    N,
                    width,
                    IN,
                    x.as_device_buffer(),
                    self.w.as_device_buffer(),
                    output.as_device_buffer_mut(),
                )
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_into<const N: usize, const D: usize, const FF: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, IN>>,
        dy: &GpuTensor<f32, Rank3<N, GROUPS, OUT>>,
        dx: &mut GpuTensor<f32, Rank2<N, IN>>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        scratch: Option<&mut Bf16LinearScratch<N, D, FF>>,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<(), DriverError> {
        let width = GROUPS * OUT;
        if let (Some(compute), Some(scratch)) = (&self.compute, scratch)
            && tcgen05_linear_eligible(N, IN, width)
        {
            profiler.measure(stream, names[0], || {
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * IN / 2),
                    x.as_device_buffer(),
                    &mut scratch.rows,
                )?;
                unsafe {
                    tensor.transpose_bf16_pairs(
                        stream,
                        transpose_pairs_config(N, IN),
                        &scratch.rows,
                        N as u32,
                        IN as u32,
                        &mut scratch.lhs_t,
                    )?;
                }
                tensor.convert_f32_to_bf16_pairs(
                    stream,
                    pairs_config(N * width / 2),
                    dy.as_device_buffer(),
                    &mut scratch.rows,
                )?;
                unsafe {
                    tensor.transpose_bf16_pairs(
                        stream,
                        transpose_pairs_config(N, width),
                        &scratch.rows,
                        N as u32,
                        width as u32,
                        &mut scratch.rhs_t,
                    )?;
                    tcgen05.f32_accumulate(
                        stream,
                        tcgen05_launch_config(IN, width, N),
                        scratch.lhs_t_maps.get::<D, FF>(IN).as_ptr(),
                        scratch.rhs_t_maps.get::<D, FF>(width).as_ptr(),
                        self.dw.as_device_buffer_mut(),
                        width as u32,
                        N as u32,
                    )
                }
            })?;
            // `scratch.rows` still holds the quantized `dy` written by the
            // weight-gradient pass above; this launch consumes it as its row
            // operand, so nothing may overwrite `rows` between the two.
            profiler.measure(stream, names[1], || unsafe {
                tcgen05.f32_store(
                    stream,
                    tcgen05_launch_config(N, IN, width),
                    scratch.row_maps.get::<D, FF>(width).as_ptr(),
                    compute.normal_tma.as_ptr(),
                    dx.as_device_buffer_mut(),
                    IN as u32,
                    width as u32,
                )
            })
        } else {
            profiler.measure(stream, names[0], || unsafe {
                fp32.register_gemm_tn_accumulate(
                    stream,
                    fp32_launch_config(IN, width),
                    IN,
                    width,
                    N,
                    x.as_device_buffer(),
                    dy.as_device_buffer(),
                    self.dw.as_device_buffer_mut(),
                )
            })?;
            profiler.measure(stream, names[1], || unsafe {
                fp32.register_gemm_nt_store(
                    stream,
                    fp32_launch_config(N, IN),
                    N,
                    IN,
                    width,
                    dy.as_device_buffer(),
                    self.w.as_device_buffer(),
                    dx.as_device_buffer_mut(),
                )
            })
        }
    }

    fn sync_compute(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        if let Some(compute) = &mut self.compute {
            compute.sync_from_master(
                self.w.as_device_buffer(),
                IN,
                GROUPS * OUT,
                stream,
                kernels,
            )?;
        }
        Ok(())
    }
}

/// Packed-bf16 compute copies for `experts` row-major `[rows, columns]`
/// matrices held in one stacked allocation.
///
/// `normal` keeps experts contiguous. `transposed` is the transpose of the
/// global `[experts * rows, columns]` matrix; each expert is addressed through
/// a strided TMA descriptor, avoiding one allocation or transpose launch per
/// expert.
struct StackedBf16Weights {
    normal: DeviceBuffer<u32>,
    transposed: DeviceBuffer<u32>,
    normal_maps: Vec<Bf16PairsTmaMap>,
    transposed_maps: Vec<Bf16PairsTmaMap>,
    experts: usize,
    rows: usize,
    columns: usize,
}

impl StackedBf16Weights {
    fn new(
        stream: &CudaStream,
        values: &[f32],
        experts: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Self, Box<dyn Error>> {
        assert_eq!(values.len(), experts * rows * columns);
        let pack = |low: f32, high: f32| {
            bf16::from_f32(low).to_bits() as u32 | ((bf16::from_f32(high).to_bits() as u32) << 16)
        };
        let normal_values: Vec<u32> = values
            .chunks_exact(2)
            .map(|pair| pack(pair[0], pair[1]))
            .collect();
        let total_rows = experts * rows;
        let mut transposed_values = vec![0u32; values.len() / 2];
        for column in 0..columns {
            for pair in 0..total_rows / 2 {
                transposed_values[column * total_rows / 2 + pair] = pack(
                    values[2 * pair * columns + column],
                    values[(2 * pair + 1) * columns + column],
                );
            }
        }

        let normal = DeviceBuffer::from_host(stream, &normal_values)?;
        let transposed = DeviceBuffer::from_host(stream, &transposed_values)?;
        let normal_maps = (0..experts)
            .map(|expert| unsafe {
                create_bf16_pairs_tma_map_region(
                    stream,
                    &normal,
                    expert * rows * columns / 2,
                    columns,
                    rows,
                    columns,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let transposed_maps = (0..experts)
            .map(|expert| unsafe {
                create_bf16_pairs_tma_map_region(
                    stream,
                    &transposed,
                    expert * rows / 2,
                    rows,
                    columns,
                    total_rows,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            normal,
            transposed,
            normal_maps,
            transposed_maps,
            experts,
            rows,
            columns,
        })
    }

    fn sync_from_master(
        &mut self,
        master: &DeviceBuffer<f32>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        kernels.convert_f32_to_bf16_pairs(
            stream,
            pairs_config(master.len() / 2),
            master,
            &mut self.normal,
        )?;
        unsafe {
            kernels.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(self.experts * self.rows, self.columns),
                &self.normal,
                (self.experts * self.rows) as u32,
                self.columns as u32,
                &mut self.transposed,
            )
        }
    }
}

struct ExpertBf16MapSet {
    d: Vec<Bf16PairsTmaMap>,
    ff: Vec<Bf16PairsTmaMap>,
    gate_up: Vec<Bf16PairsTmaMap>,
}

impl ExpertBf16MapSet {
    fn get<const D: usize, const FF: usize>(&self, width: usize) -> &[Bf16PairsTmaMap] {
        if width == D {
            &self.d
        } else if width == FF {
            &self.ff
        } else if width == 2 * FF {
            &self.gate_up
        } else {
            panic!("unsupported expert tcgen05 width {width}")
        }
    }
}

/// Packed operand and transpose staging shared by every expert launch.
struct ExpertBf16Scratch<const E: usize, const C: usize, const D: usize, const FF: usize> {
    rows: DeviceBuffer<u32>,
    lhs_t: DeviceBuffer<u32>,
    rhs_t: DeviceBuffer<u32>,
    row_maps: ExpertBf16MapSet,
    lhs_t_maps: ExpertBf16MapSet,
    rhs_t_maps: ExpertBf16MapSet,
}

impl<const E: usize, const C: usize, const D: usize, const FF: usize>
    ExpertBf16Scratch<E, C, D, FF>
{
    fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        let max_width = D.max(FF).max(2 * FF);
        let words = E * C * max_width / 2;
        let rows = DeviceBuffer::zeroed(stream, words)?;
        let lhs_t = DeviceBuffer::zeroed(stream, words)?;
        let rhs_t = DeviceBuffer::zeroed(stream, words)?;
        let row_maps = Self::maps(stream, &rows, false)?;
        let lhs_t_maps = Self::maps(stream, &lhs_t, true)?;
        let rhs_t_maps = Self::maps(stream, &rhs_t, true)?;
        Ok(Self {
            rows,
            lhs_t,
            rhs_t,
            row_maps,
            lhs_t_maps,
            rhs_t_maps,
        })
    }

    fn maps(
        stream: &CudaStream,
        buffer: &DeviceBuffer<u32>,
        transposed: bool,
    ) -> Result<ExpertBf16MapSet, Box<dyn Error>> {
        fn make(
            stream: &CudaStream,
            buffer: &DeviceBuffer<u32>,
            experts: usize,
            capacity: usize,
            width: usize,
            transposed: bool,
        ) -> Result<Vec<Bf16PairsTmaMap>, Box<dyn Error>> {
            (0..experts)
                .map(|expert| unsafe {
                    if transposed {
                        create_bf16_pairs_tma_map_region(
                            stream,
                            buffer,
                            expert * capacity / 2,
                            capacity,
                            width,
                            experts * capacity,
                        )
                    } else {
                        create_bf16_pairs_tma_map_region(
                            stream,
                            buffer,
                            expert * capacity * width / 2,
                            width,
                            capacity,
                            width,
                        )
                    }
                })
                .collect()
        }

        Ok(ExpertBf16MapSet {
            d: make(stream, buffer, E, C, D, transposed)?,
            ff: make(stream, buffer, E, C, FF, transposed)?,
            gate_up: make(stream, buffer, E, C, 2 * FF, transposed)?,
        })
    }
}

/// Staging used only by the non-aligned fp32 oracle. One expert is copied into
/// these buffers and passed to the existing register-tiled GEMM launchers.
struct ExpertFp32Scratch {
    a: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    c: DeviceBuffer<f32>,
}

impl ExpertFp32Scratch {
    fn new<const C: usize, const D: usize, const FF: usize>(
        stream: &CudaStream,
    ) -> Result<Self, DriverError> {
        let max_width = D.max(FF).max(2 * FF);
        let max_elements = (C * max_width).max(D * 2 * FF).max(FF * D);
        Ok(Self {
            a: DeviceBuffer::zeroed(stream, max_elements)?,
            b: DeviceBuffer::zeroed(stream, max_elements)?,
            c: DeviceBuffer::zeroed(stream, max_elements)?,
        })
    }
}

struct ExpertLinearScratch<const E: usize, const C: usize, const D: usize, const FF: usize> {
    bf16: Option<ExpertBf16Scratch<E, C, D, FF>>,
    fp32: Option<ExpertFp32Scratch>,
}

#[allow(clippy::too_many_arguments)]
fn expert_linear_forward<
    const E: usize,
    const C: usize,
    const D: usize,
    const FF: usize,
    P: KernelProfiler,
>(
    input: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    compute: Option<&StackedBf16Weights>,
    output: &mut DeviceBuffer<f32>,
    input_width: usize,
    output_width: usize,
    scratch: &mut ExpertLinearScratch<E, C, D, FF>,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    fp32: &gemm_kernels::LoadedModule,
    tcgen05: &Tcgen05Gemm,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    if let (Some(compute), Some(bf16_scratch)) = (compute, scratch.bf16.as_mut())
        && tcgen05_linear_eligible(C, input_width, output_width)
    {
        profiler.measure(stream, name, || {
            tensor.convert_f32_to_bf16_pairs(
                stream,
                pairs_config(E * C * input_width / 2),
                input,
                &mut bf16_scratch.rows,
            )?;
            let input_maps = bf16_scratch.row_maps.get::<D, FF>(input_width);
            for expert in 0..E {
                unsafe {
                    tcgen05.f32_store_at(
                        stream,
                        tcgen05_launch_config(C, output_width, input_width),
                        input_maps[expert].as_ptr(),
                        compute.transposed_maps[expert].as_ptr(),
                        output,
                        expert * C * output_width,
                        C * output_width,
                        output_width as u32,
                        input_width as u32,
                    )?;
                }
            }
            Ok(())
        })
    } else {
        let fp32_scratch = scratch
            .fp32
            .as_mut()
            .expect("non-aligned experts must own fp32 staging");
        profiler.measure(stream, name, || {
            for expert in 0..E {
                copy_device_region(
                    &mut fp32_scratch.a,
                    0,
                    input,
                    expert * C * input_width,
                    C * input_width,
                    stream,
                )?;
                copy_device_region(
                    &mut fp32_scratch.b,
                    0,
                    weights,
                    expert * input_width * output_width,
                    input_width * output_width,
                    stream,
                )?;
                unsafe {
                    fp32.register_gemm_store(
                        stream,
                        fp32_launch_config(C, output_width),
                        C,
                        output_width,
                        input_width,
                        &fp32_scratch.a,
                        &fp32_scratch.b,
                        &mut fp32_scratch.c,
                    )?;
                }
                copy_device_region(
                    output,
                    expert * C * output_width,
                    &fp32_scratch.c,
                    0,
                    C * output_width,
                    stream,
                )?;
            }
            Ok(())
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn expert_linear_backward<
    const E: usize,
    const C: usize,
    const D: usize,
    const FF: usize,
    P: KernelProfiler,
>(
    input: &DeviceBuffer<f32>,
    output_gradient: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    weight_gradient: &mut DeviceBuffer<f32>,
    compute: Option<&StackedBf16Weights>,
    input_gradient: &mut DeviceBuffer<f32>,
    input_width: usize,
    output_width: usize,
    scratch: &mut ExpertLinearScratch<E, C, D, FF>,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    fp32: &gemm_kernels::LoadedModule,
    tcgen05: &Tcgen05Gemm,
    profiler: &mut P,
    names: [&'static str; 2],
) -> Result<(), DriverError> {
    if let (Some(compute), Some(bf16_scratch)) = (compute, scratch.bf16.as_mut())
        && tcgen05_linear_eligible(C, input_width, output_width)
    {
        profiler.measure(stream, names[0], || {
            tensor.convert_f32_to_bf16_pairs(
                stream,
                pairs_config(E * C * input_width / 2),
                input,
                &mut bf16_scratch.rows,
            )?;
            unsafe {
                tensor.transpose_bf16_pairs(
                    stream,
                    transpose_pairs_config(E * C, input_width),
                    &bf16_scratch.rows,
                    (E * C) as u32,
                    input_width as u32,
                    &mut bf16_scratch.lhs_t,
                )?;
            }
            tensor.convert_f32_to_bf16_pairs(
                stream,
                pairs_config(E * C * output_width / 2),
                output_gradient,
                &mut bf16_scratch.rows,
            )?;
            unsafe {
                tensor.transpose_bf16_pairs(
                    stream,
                    transpose_pairs_config(E * C, output_width),
                    &bf16_scratch.rows,
                    (E * C) as u32,
                    output_width as u32,
                    &mut bf16_scratch.rhs_t,
                )?;
            }
            let lhs_maps = bf16_scratch.lhs_t_maps.get::<D, FF>(input_width);
            let rhs_maps = bf16_scratch.rhs_t_maps.get::<D, FF>(output_width);
            for expert in 0..E {
                unsafe {
                    tcgen05.f32_accumulate_at(
                        stream,
                        tcgen05_launch_config(input_width, output_width, C),
                        lhs_maps[expert].as_ptr(),
                        rhs_maps[expert].as_ptr(),
                        weight_gradient,
                        expert * input_width * output_width,
                        input_width * output_width,
                        output_width as u32,
                        C as u32,
                    )?;
                }
            }
            Ok(())
        })?;
        // `rows` still contains the packed output gradient.
        profiler.measure(stream, names[1], || {
            let output_maps = bf16_scratch.row_maps.get::<D, FF>(output_width);
            for expert in 0..E {
                unsafe {
                    tcgen05.f32_store_at(
                        stream,
                        tcgen05_launch_config(C, input_width, output_width),
                        output_maps[expert].as_ptr(),
                        compute.normal_maps[expert].as_ptr(),
                        input_gradient,
                        expert * C * input_width,
                        C * input_width,
                        input_width as u32,
                        output_width as u32,
                    )?;
                }
            }
            Ok(())
        })
    } else {
        let fp32_scratch = scratch
            .fp32
            .as_mut()
            .expect("non-aligned experts must own fp32 staging");
        profiler.measure(stream, names[0], || {
            for expert in 0..E {
                copy_device_region(
                    &mut fp32_scratch.a,
                    0,
                    input,
                    expert * C * input_width,
                    C * input_width,
                    stream,
                )?;
                copy_device_region(
                    &mut fp32_scratch.b,
                    0,
                    output_gradient,
                    expert * C * output_width,
                    C * output_width,
                    stream,
                )?;
                copy_device_region(
                    &mut fp32_scratch.c,
                    0,
                    weight_gradient,
                    expert * input_width * output_width,
                    input_width * output_width,
                    stream,
                )?;
                unsafe {
                    fp32.register_gemm_tn_accumulate(
                        stream,
                        fp32_launch_config(input_width, output_width),
                        input_width,
                        output_width,
                        C,
                        &fp32_scratch.a,
                        &fp32_scratch.b,
                        &mut fp32_scratch.c,
                    )?;
                }
                copy_device_region(
                    weight_gradient,
                    expert * input_width * output_width,
                    &fp32_scratch.c,
                    0,
                    input_width * output_width,
                    stream,
                )?;
            }
            Ok(())
        })?;
        profiler.measure(stream, names[1], || {
            for expert in 0..E {
                copy_device_region(
                    &mut fp32_scratch.a,
                    0,
                    output_gradient,
                    expert * C * output_width,
                    C * output_width,
                    stream,
                )?;
                copy_device_region(
                    &mut fp32_scratch.b,
                    0,
                    weights,
                    expert * input_width * output_width,
                    input_width * output_width,
                    stream,
                )?;
                unsafe {
                    fp32.register_gemm_nt_store(
                        stream,
                        fp32_launch_config(C, input_width),
                        C,
                        input_width,
                        output_width,
                        &fp32_scratch.a,
                        &fp32_scratch.b,
                        &mut fp32_scratch.c,
                    )?;
                }
                copy_device_region(
                    input_gradient,
                    expert * C * input_width,
                    &fp32_scratch.c,
                    0,
                    C * input_width,
                    stream,
                )?;
            }
            Ok(())
        })
    }
}

/// Stacked GPU weights for `E` capacity-binned SwiGLU experts.
///
/// Gate and up projections share one `[E, D, 2, FF]` master/gradient entry;
/// down projections share one `[E, FF, D]` entry. Aligned shapes also own one
/// persistent packed-bf16 compute allocation per entry.
pub struct GpuExpertFfn<const E: usize, const D: usize, const FF: usize> {
    pub gate_up: GpuTensor<f32, Rank4<E, D, 2, FF>>,
    pub d_gate_up: GpuTensor<f32, Rank4<E, D, 2, FF>>,
    pub down: GpuTensor<f32, Rank3<E, FF, D>>,
    pub d_down: GpuTensor<f32, Rank3<E, FF, D>>,
    gate_up_compute: Option<StackedBf16Weights>,
    down_compute: Option<StackedBf16Weights>,
}

impl<const E: usize, const D: usize, const FF: usize> GpuExpertFfn<E, D, FF> {
    pub fn from_cpu<const C: usize>(
        stream: &CudaStream,
        experts: &[nn::ExpertFfn<C, D, FF>; E],
    ) -> Result<Self, Box<dyn Error>> {
        assert!(E > 0, "GPU expert count must be non-zero");
        let mut gate_up = vec![0.0; E * D * 2 * FF];
        let mut down = vec![0.0; E * FF * D];
        for (expert, source) in experts.iter().enumerate() {
            for input in 0..D {
                let destination = (expert * D + input) * 2 * FF;
                gate_up[destination..destination + FF]
                    .copy_from_slice(&source.gate_proj.w.as_slice()[input * FF..(input + 1) * FF]);
                gate_up[destination + FF..destination + 2 * FF]
                    .copy_from_slice(&source.up_proj.w.as_slice()[input * FF..(input + 1) * FF]);
            }
            down[expert * FF * D..(expert + 1) * FF * D]
                .copy_from_slice(source.down_proj.w.as_slice());
        }
        let aligned = D.is_multiple_of(TC_TILE) && FF.is_multiple_of(TC_TILE);
        Ok(Self {
            gate_up: GpuTensor::from_host(stream, &gate_up)?,
            d_gate_up: GpuTensor::zeros(stream)?,
            down: GpuTensor::from_host(stream, &down)?,
            d_down: GpuTensor::zeros(stream)?,
            gate_up_compute: aligned
                .then(|| StackedBf16Weights::new(stream, &gate_up, E, D, 2 * FF))
                .transpose()?,
            down_compute: aligned
                .then(|| StackedBf16Weights::new(stream, &down, E, FF, D))
                .transpose()?,
        })
    }

    pub fn forward<const C: usize>(
        &self,
        workspace: &mut GpuExpertWorkspace<E, C, D, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        llama: &llama_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.forward_profiled(
            workspace,
            stream,
            tensor,
            fp32,
            tcgen05,
            llama,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_profiled<const C: usize, P: KernelProfiler>(
        &self,
        workspace: &mut GpuExpertWorkspace<E, C, D, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        llama: &llama_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        expert_linear_forward(
            workspace.bin_input.as_device_buffer(),
            self.gate_up.as_device_buffer(),
            self.gate_up_compute.as_ref(),
            workspace.gate_up.as_device_buffer_mut(),
            D,
            2 * FF,
            &mut workspace.scratch,
            stream,
            tensor,
            fp32,
            tcgen05,
            profiler,
            "forward.experts.gate_up_gemm",
        )?;
        profiler.measure(stream, "forward.experts.gate_up_split", || {
            llama.split_group2(
                stream,
                LaunchConfig::for_num_elems((E * C * FF) as u32),
                workspace.gate_up.as_device_buffer(),
                FF as u32,
                workspace.gate.as_device_buffer_mut(),
                workspace.up.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "forward.experts.swiglu", || {
            llama.swiglu_forward(
                stream,
                LaunchConfig::for_num_elems((E * C * FF) as u32),
                workspace.gate.as_device_buffer(),
                workspace.up.as_device_buffer(),
                workspace.activated.as_device_buffer_mut(),
            )
        })?;
        expert_linear_forward(
            workspace.activated.as_device_buffer(),
            self.down.as_device_buffer(),
            self.down_compute.as_ref(),
            workspace.bin_output.as_device_buffer_mut(),
            FF,
            D,
            &mut workspace.scratch,
            stream,
            tensor,
            fp32,
            tcgen05,
            profiler,
            "forward.experts.down_gemm",
        )?;
        Ok(())
    }

    pub fn backward<const C: usize>(
        &mut self,
        workspace: &mut GpuExpertWorkspace<E, C, D, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        llama: &llama_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.backward_profiled(
            workspace,
            stream,
            tensor,
            fp32,
            tcgen05,
            llama,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_profiled<const C: usize, P: KernelProfiler>(
        &mut self,
        workspace: &mut GpuExpertWorkspace<E, C, D, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        fp32: &gemm_kernels::LoadedModule,
        tcgen05: &Tcgen05Gemm,
        llama: &llama_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        expert_linear_backward(
            workspace.activated.as_device_buffer(),
            workspace.d_bin_output.as_device_buffer(),
            self.down.as_device_buffer(),
            self.d_down.as_device_buffer_mut(),
            self.down_compute.as_ref(),
            workspace.d_activated.as_device_buffer_mut(),
            FF,
            D,
            &mut workspace.scratch,
            stream,
            tensor,
            fp32,
            tcgen05,
            profiler,
            [
                "backward.experts.down_weight_gemm",
                "backward.experts.down_input_gemm",
            ],
        )?;
        let elementwise = LaunchConfig::for_num_elems((E * C * FF) as u32);
        profiler.measure(stream, "backward.experts.swiglu_gate", || {
            llama.swiglu_backward_gate(
                stream,
                elementwise,
                workspace.gate.as_device_buffer(),
                workspace.up.as_device_buffer(),
                workspace.d_activated.as_device_buffer(),
                workspace.d_gate.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "backward.experts.swiglu_up", || {
            llama.swiglu_backward_up(
                stream,
                elementwise,
                workspace.gate.as_device_buffer(),
                workspace.d_activated.as_device_buffer(),
                workspace.d_up.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "backward.experts.gate_up_join", || unsafe {
            llama.join_group2(
                stream,
                elementwise,
                workspace.d_gate.as_device_buffer(),
                workspace.d_up.as_device_buffer(),
                FF as u32,
                workspace.d_gate_up.as_device_buffer_mut(),
            )
        })?;
        expert_linear_backward(
            workspace.bin_input.as_device_buffer(),
            workspace.d_gate_up.as_device_buffer(),
            self.gate_up.as_device_buffer(),
            self.d_gate_up.as_device_buffer_mut(),
            self.gate_up_compute.as_ref(),
            workspace.d_bin_input.as_device_buffer_mut(),
            D,
            2 * FF,
            &mut workspace.scratch,
            stream,
            tensor,
            fp32,
            tcgen05,
            profiler,
            [
                "backward.experts.gate_up_weight_gemm",
                "backward.experts.gate_up_input_gemm",
            ],
        )?;
        Ok(())
    }

    pub fn zero_grad(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        fill_zero(
            &mut self.d_gate_up,
            stream,
            tensor,
            &mut profiler,
            "zero_grad.experts.gate_up",
        )?;
        fill_zero(
            &mut self.d_down,
            stream,
            tensor,
            &mut profiler,
            "zero_grad.experts.down",
        )
    }

    pub fn sync_compute(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        if let Some(compute) = &mut self.gate_up_compute {
            compute.sync_from_master(self.gate_up.as_device_buffer(), stream, tensor)?;
        }
        if let Some(compute) = &mut self.down_compute {
            compute.sync_from_master(self.down.as_device_buffer(), stream, tensor)?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn gate_up_compute_words(&self) -> Option<(&DeviceBuffer<u32>, &DeviceBuffer<u32>)> {
        self.gate_up_compute
            .as_ref()
            .map(|weights| (&weights.normal, &weights.transposed))
    }

    #[allow(dead_code)]
    pub fn down_compute_words(&self) -> Option<(&DeviceBuffer<u32>, &DeviceBuffer<u32>)> {
        self.down_compute
            .as_ref()
            .map(|weights| (&weights.normal, &weights.transposed))
    }
}

/// Persistent capacity-bin activations and backward scratch for
/// [`GpuExpertFfn`].
///
/// The fp32 activation/gradient buffers occupy
/// `4 * E * C * (4 * D + 10 * FF)` bytes. Aligned tcgen05 shapes add three
/// packed-bf16 operand/transpose buffers totaling
/// `6 * E * C * max(D, FF, 2 * FF)` bytes; non-aligned oracle shapes allocate
/// three one-expert fp32 staging buffers instead.
pub struct GpuExpertWorkspace<const E: usize, const C: usize, const D: usize, const FF: usize> {
    pub bin_input: GpuTensor<f32, Rank3<E, C, D>>,
    gate_up: GpuTensor<f32, Rank4<E, C, 2, FF>>,
    gate: GpuTensor<f32, Rank3<E, C, FF>>,
    up: GpuTensor<f32, Rank3<E, C, FF>>,
    activated: GpuTensor<f32, Rank3<E, C, FF>>,
    pub bin_output: GpuTensor<f32, Rank3<E, C, D>>,
    pub d_bin_output: GpuTensor<f32, Rank3<E, C, D>>,
    d_activated: GpuTensor<f32, Rank3<E, C, FF>>,
    d_gate: GpuTensor<f32, Rank3<E, C, FF>>,
    d_up: GpuTensor<f32, Rank3<E, C, FF>>,
    d_gate_up: GpuTensor<f32, Rank4<E, C, 2, FF>>,
    pub d_bin_input: GpuTensor<f32, Rank3<E, C, D>>,
    scratch: ExpertLinearScratch<E, C, D, FF>,
}

impl<const E: usize, const C: usize, const D: usize, const FF: usize>
    GpuExpertWorkspace<E, C, D, FF>
{
    pub fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        assert!(E > 0 && C > 0 && D > 0 && FF > 0);
        assert!(E * C * D <= u32::MAX as usize);
        assert!(E * C * FF <= u32::MAX as usize);
        let aligned =
            C.is_multiple_of(TC_TILE) && D.is_multiple_of(TC_TILE) && FF.is_multiple_of(TC_TILE);
        Ok(Self {
            bin_input: GpuTensor::zeros(stream)?,
            gate_up: GpuTensor::zeros(stream)?,
            gate: GpuTensor::zeros(stream)?,
            up: GpuTensor::zeros(stream)?,
            activated: GpuTensor::zeros(stream)?,
            bin_output: GpuTensor::zeros(stream)?,
            d_bin_output: GpuTensor::zeros(stream)?,
            d_activated: GpuTensor::zeros(stream)?,
            d_gate: GpuTensor::zeros(stream)?,
            d_up: GpuTensor::zeros(stream)?,
            d_gate_up: GpuTensor::zeros(stream)?,
            d_bin_input: GpuTensor::zeros(stream)?,
            scratch: ExpertLinearScratch {
                bf16: aligned
                    .then(|| ExpertBf16Scratch::new(stream))
                    .transpose()?,
                fp32: (!aligned)
                    .then(|| ExpertFp32Scratch::new::<C, D, FF>(stream))
                    .transpose()?,
            },
        })
    }

    pub fn upload_bins(&mut self, values: &[f32], stream: &CudaStream) -> Result<(), DriverError> {
        assert_eq!(values.len(), E * C * D);
        self.bin_input = GpuTensor::from_host(stream, values)?;
        Ok(())
    }

    pub fn upload_output_gradient(
        &mut self,
        values: &[f32],
        stream: &CudaStream,
    ) -> Result<(), DriverError> {
        assert_eq!(values.len(), E * C * D);
        self.d_bin_output = GpuTensor::from_host(stream, values)?;
        Ok(())
    }

    pub fn tcgen05_active(&self) -> bool {
        self.scratch.bf16.is_some()
    }
}

/// GPU AdamW state for the two stacked expert parameter entries.
pub struct GpuExpertAdamW<const E: usize, const D: usize, const FF: usize> {
    config: AdamWConfig,
    step: u64,
    pub gate_up: GpuAdamWMoments<Rank4<E, D, 2, FF>>,
    pub down: GpuAdamWMoments<Rank3<E, FF, D>>,
}

impl<const E: usize, const D: usize, const FF: usize> GpuExpertAdamW<E, D, FF> {
    pub fn new(stream: &CudaStream, config: AdamWConfig) -> Result<Self, DriverError> {
        config.validate();
        Ok(Self {
            config,
            step: 0,
            gate_up: GpuAdamWMoments::zeros(stream)?,
            down: GpuAdamWMoments::zeros(stream)?,
        })
    }

    pub fn update(
        &mut self,
        experts: &mut GpuExpertFfn<E, D, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        self.step = self
            .step
            .checked_add(1)
            .expect("expert AdamW step overflow");
        let (first_correction, second_correction) = self.config.bias_correction(self.step);
        experts.gate_up.adamw_step(
            &experts.d_gate_up,
            &mut self.gate_up,
            self.config.learning_rate,
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon,
            self.config.weight_decay,
            first_correction,
            second_correction,
            stream,
            tensor,
        )?;
        experts.down.adamw_step(
            &experts.d_down,
            &mut self.down,
            self.config.learning_rate,
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon,
            self.config.weight_decay,
            first_correction,
            second_correction,
            stream,
            tensor,
        )?;
        experts.sync_compute(stream, tensor)
    }
}

pub struct GpuRmsNorm<const D: usize> {
    pub w: GpuTensor<f32, Rank1<D>>,
    pub dw: GpuTensor<f32, Rank1<D>>,
    eps: f32,
}

impl<const D: usize> GpuRmsNorm<D> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layer: &nn::RmsNorm<N, D>,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            w: GpuTensor::from_cpu(stream, &layer.w)?,
            dw: GpuTensor::zeros(stream)?,
            eps: layer.eps,
        })
    }

    fn forward_into<const N: usize, P: KernelProfiler>(
        &self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        y: &mut GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || {
            kernels.rms_norm_forward_fast(
                stream,
                norm_config::<N>(),
                x.as_device_buffer(),
                self.w.as_device_buffer(),
                self.eps,
                D as u32,
                y.as_device_buffer_mut(),
            )
        })
    }

    fn backward_into<const N: usize, P: KernelProfiler>(
        &mut self,
        x: &GpuTensor<f32, Rank2<N, D>>,
        dy: &GpuTensor<f32, Rank2<N, D>>,
        dx: &mut GpuTensor<f32, Rank2<N, D>>,
        inv: &mut GpuTensor<f32, Rank1<N>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        names: [&'static str; 2],
    ) -> Result<(), DriverError> {
        profiler.measure(stream, names[0], || {
            kernels.rms_norm_backward_x_fast(
                stream,
                norm_config::<N>(),
                x.as_device_buffer(),
                self.w.as_device_buffer(),
                dy.as_device_buffer(),
                self.eps,
                D as u32,
                dx.as_device_buffer_mut(),
                inv.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, names[1], || unsafe {
            kernels.rms_norm_backward_weight_fast(
                stream,
                norm_weight_config::<N, D>(),
                x.as_device_buffer(),
                dy.as_device_buffer(),
                inv.as_device_buffer(),
                N as u32,
                D as u32,
                self.dw.as_device_buffer_mut(),
            )
        })
    }
}

pub struct GpuEmbedding<const VOCAB: usize, const D: usize> {
    pub w: GpuTensor<f32, Rank2<VOCAB, D>>,
    pub dw: GpuTensor<f32, Rank2<VOCAB, D>>,
}

impl<const VOCAB: usize, const D: usize> GpuEmbedding<VOCAB, D> {
    fn from_cpu<const N: usize>(
        stream: &CudaStream,
        layer: &nn::Embedding<N, VOCAB, D>,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            w: GpuTensor::from_cpu(stream, &layer.w)?,
            dw: GpuTensor::zeros(stream)?,
        })
    }

    fn forward_into<const N: usize, P: KernelProfiler>(
        &self,
        tokens: &GpuTensor<u32, Rank1<N>>,
        y: &mut GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || {
            kernels.embedding_forward(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                self.w.as_device_buffer(),
                tokens.as_device_buffer(),
                D as u32,
                y.as_device_buffer_mut(),
            )
        })
    }

    fn backward<const N: usize, P: KernelProfiler>(
        &mut self,
        tokens: &GpuTensor<u32, Rank1<N>>,
        dy: &GpuTensor<f32, Rank2<N, D>>,
        stream: &CudaStream,
        kernels: &llama_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.embedding_backward_scatter(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                tokens.as_device_buffer(),
                dy.as_device_buffer(),
                D as u32,
                self.dw.as_device_buffer_mut(),
            )
        })
    }
}

/// bf16 lm-head with fp32 master weights (§7 phase 2).
///
/// The compute weight exists in both layouts the tcgen05 `C = A B^T` form
/// needs as a K-contiguous `B` operand: `w` is `[D, VP]` (consumed by the
/// input-gradient GEMM) and `w_t` is `[VP, D]` (consumed by the forward
/// GEMM). `master` is the fp32 source of truth; the optimizer updates it and
/// re-rounds both compute copies every step. `dw` accumulates in packed bf16,
/// produced directly by the tcgen05 accumulate epilogue.
pub struct GpuBf16Head<const D: usize, const VP: usize> {
    pub master: GpuTensor<f32, Rank2<D, VP>>,
    w: DeviceBuffer<u32>,
    w_t: DeviceBuffer<u32>,
    dw: DeviceBuffer<u32>,
    w_tma: Bf16PairsTmaMap,
    w_t_tma: Bf16PairsTmaMap,
}

impl<const D: usize, const VP: usize> GpuBf16Head<D, VP> {
    fn from_cpu<const N: usize, const VOCAB: usize>(
        stream: &CudaStream,
        layer: &nn::Linear<N, D, VOCAB>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(VP >= VOCAB);
        let mut padded = vec![0.0f32; D * VP];
        for row in 0..D {
            padded[row * VP..row * VP + VOCAB]
                .copy_from_slice(&layer.w.as_slice()[row * VOCAB..(row + 1) * VOCAB]);
        }
        Self::from_master_values(stream, &padded)
    }

    /// Rebuild the head from padded `[D, VP]` fp32 master values, rounding
    /// both packed compute copies on the host.
    pub(crate) fn from_master_values(
        stream: &CudaStream,
        values: &[f32],
    ) -> Result<Self, Box<dyn Error>> {
        assert_eq!(values.len(), D * VP);
        let pack = |low: f32, high: f32| {
            bf16::from_f32(low).to_bits() as u32 | ((bf16::from_f32(high).to_bits() as u32) << 16)
        };
        let compute: Vec<u32> = values
            .chunks_exact(2)
            .map(|pair| pack(pair[0], pair[1]))
            .collect();
        let mut transposed = vec![0u32; VP * D / 2];
        for column in 0..VP {
            for pair in 0..D / 2 {
                transposed[column * D / 2 + pair] = pack(
                    values[2 * pair * VP + column],
                    values[(2 * pair + 1) * VP + column],
                );
            }
        }

        let master = GpuTensor::from_host(stream, values)?;
        let w = DeviceBuffer::from_host(stream, &compute)?;
        let w_t = DeviceBuffer::from_host(stream, &transposed)?;
        let dw = DeviceBuffer::zeroed(stream, D * VP / 2)?;
        // SAFETY: `w` and `w_t` live in this struct beside their maps and are
        // never reallocated.
        let w_tma = unsafe { create_bf16_pairs_tma_map(stream, &w, VP, D)? };
        let w_t_tma = unsafe { create_bf16_pairs_tma_map(stream, &w_t, D, VP)? };
        Ok(Self {
            master,
            w,
            w_t,
            dw,
            w_tma,
            w_t_tma,
        })
    }

    /// Packed-bf16 weight gradient. Parity-test accessor: binaries other than
    /// the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn dw_words(&self) -> &DeviceBuffer<u32> {
        &self.dw
    }

    /// Packed-bf16 `[D, VP]` compute weights. Parity-test accessor: binaries
    /// other than the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn w_words(&self) -> &DeviceBuffer<u32> {
        &self.w
    }

    /// Packed-bf16 `[VP, D]` transposed compute weights. Parity-test accessor:
    /// binaries other than the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn w_t_words(&self) -> &DeviceBuffer<u32> {
        &self.w_t
    }

    fn zero_grad<P: KernelProfiler>(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || {
            kernels.fill_u32(stream, pairs_config(D * VP / 2), 0, &mut self.dw)
        })
    }

    fn forward_into<const NP: usize, P: KernelProfiler>(
        &self,
        x_tma: &Bf16PairsTmaMap,
        logits: &mut DeviceBuffer<u32>,
        stream: &CudaStream,
        kernels: &Tcgen05Gemm,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.store(
                stream,
                tcgen05_launch_config(NP, VP, D),
                x_tma.as_ptr(),
                self.w_t_tma.as_ptr(),
                logits,
                VP as u32,
                D as u32,
            )
        })
    }

    /// `dw += x^T dlogits` from the transposed operands staged in the
    /// workspace; padded token rows and vocabulary columns contribute zeros.
    fn backward_weight<const NP: usize, P: KernelProfiler>(
        &mut self,
        x_t_tma: &Bf16PairsTmaMap,
        dlogits_t_tma: &Bf16PairsTmaMap,
        stream: &CudaStream,
        kernels: &Tcgen05Gemm,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.accumulate(
                stream,
                tcgen05_launch_config(D, VP, NP),
                x_t_tma.as_ptr(),
                dlogits_t_tma.as_ptr(),
                &mut self.dw,
                VP as u32,
                NP as u32,
            )
        })
    }

    fn backward_input<const NP: usize, P: KernelProfiler>(
        &self,
        dlogits_tma: &Bf16PairsTmaMap,
        dx: &mut DeviceBuffer<u32>,
        stream: &CudaStream,
        kernels: &Tcgen05Gemm,
        profiler: &mut P,
        name: &'static str,
    ) -> Result<(), DriverError> {
        profiler.measure(stream, name, || unsafe {
            kernels.store(
                stream,
                tcgen05_launch_config(NP, D, VP),
                dlogits_tma.as_ptr(),
                self.w_tma.as_ptr(),
                dx,
                D as u32,
                VP as u32,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        &mut self,
        moments: &mut GpuAdamWMoments<Rank2<D, VP>>,
        config: AdamWConfig,
        first_correction: f32,
        second_correction: f32,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        kernels.adamw_master_bf16(
            stream,
            pairs_config(D * VP / 2),
            &self.dw,
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.epsilon,
            config.weight_decay,
            first_correction,
            second_correction,
            self.master.as_device_buffer_mut(),
            moments.first.as_device_buffer_mut(),
            moments.second.as_device_buffer_mut(),
            &mut self.w,
        )
    }

    /// Refresh `w_t` from `w` after an optimizer step.
    fn sync_transposed(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        unsafe {
            kernels.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(D, VP),
                &self.w,
                D as u32,
                VP as u32,
                &mut self.w_t,
            )
        }
    }
}

pub struct GpuDenseLlama<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> {
    pub embedding: GpuEmbedding<VOCAB, D>,
    pub attention_norm: GpuRmsNorm<D>,
    pub qkv_proj: GpuGroupedLinear<D, 3, D>,
    pub o_proj: GpuLinear<D, D>,
    pub ffn_norm: GpuRmsNorm<D>,
    pub gate_up_proj: GpuGroupedLinear<D, 2, FF>,
    pub down_proj: GpuLinear<FF, D>,
    pub final_norm: GpuRmsNorm<D>,
    pub lm_head: GpuBf16Head<D, VP>,
}

/// GPU-resident AdamW state mirroring every model parameter.
///
/// The lm-head moments span the padded `[D, VP]` master; padded columns hold
/// zeros forever because their gradients are exactly zero.
pub struct GpuDenseLlamaAdamW<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize>
{
    config: AdamWConfig,
    step: u64,
    pub embedding: GpuAdamWMoments<Rank2<VOCAB, D>>,
    pub attention_norm: GpuAdamWMoments<Rank1<D>>,
    pub qkv_proj: GpuAdamWMoments<Rank3<D, 3, D>>,
    pub o_proj: GpuAdamWMoments<Rank2<D, D>>,
    pub ffn_norm: GpuAdamWMoments<Rank1<D>>,
    pub gate_up_proj: GpuAdamWMoments<Rank3<D, 2, FF>>,
    pub down_proj: GpuAdamWMoments<Rank2<FF, D>>,
    pub final_norm: GpuAdamWMoments<Rank1<D>>,
    pub lm_head: GpuAdamWMoments<Rank2<D, VP>>,
}

impl<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize>
    GpuDenseLlamaAdamW<VOCAB, VP, D, FF>
{
    pub fn new(stream: &CudaStream, config: AdamWConfig) -> Result<Self, DriverError> {
        config.validate();
        Ok(Self {
            config,
            step: 0,
            embedding: GpuAdamWMoments::zeros(stream)?,
            attention_norm: GpuAdamWMoments::zeros(stream)?,
            qkv_proj: GpuAdamWMoments::zeros(stream)?,
            o_proj: GpuAdamWMoments::zeros(stream)?,
            ffn_norm: GpuAdamWMoments::zeros(stream)?,
            gate_up_proj: GpuAdamWMoments::zeros(stream)?,
            down_proj: GpuAdamWMoments::zeros(stream)?,
            final_norm: GpuAdamWMoments::zeros(stream)?,
            lm_head: GpuAdamWMoments::zeros(stream)?,
        })
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    pub(crate) fn restore_step(&mut self, step: u64) {
        self.step = step;
    }

    pub fn update<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
    >(
        &mut self,
        model: &mut GpuDenseLlama<N, NP, T, VOCAB, VP, D, H, HD, FF>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.update_profiled(model, stream, kernels, &mut profiler)
    }

    pub fn update_profiled<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        P: KernelProfiler,
    >(
        &mut self,
        model: &mut GpuDenseLlama<N, NP, T, VOCAB, VP, D, H, HD, FF>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        self.step = self.step.checked_add(1).expect("AdamW step overflow");
        let (first_correction, second_correction) = self.config.bias_correction(self.step);

        macro_rules! update {
            ($field:ident, $weight_decay:expr) => {
                profiler.measure(
                    stream,
                    concat!("optimizer.", stringify!($field), ".adamw"),
                    || {
                        model.$field.w.adamw_step(
                            &model.$field.dw,
                            &mut self.$field,
                            self.config.learning_rate,
                            self.config.beta1,
                            self.config.beta2,
                            self.config.epsilon,
                            $weight_decay,
                            first_correction,
                            second_correction,
                            stream,
                            kernels,
                        )
                    },
                )?;
            };
        }

        update!(embedding, self.config.weight_decay);
        update!(attention_norm, 0.0);
        update!(qkv_proj, self.config.weight_decay);
        update!(o_proj, self.config.weight_decay);
        update!(ffn_norm, 0.0);
        update!(gate_up_proj, self.config.weight_decay);
        update!(down_proj, self.config.weight_decay);
        update!(final_norm, 0.0);
        macro_rules! sync_compute {
            ($field:ident) => {
                profiler.measure(
                    stream,
                    concat!("optimizer.", stringify!($field), ".sync_compute"),
                    || model.$field.sync_compute(stream, kernels),
                )?;
            };
        }
        sync_compute!(qkv_proj);
        sync_compute!(o_proj);
        sync_compute!(gate_up_proj);
        sync_compute!(down_proj);
        profiler.measure(stream, "optimizer.lm_head.adamw", || {
            model.lm_head.adamw_step(
                &mut self.lm_head,
                self.config,
                first_correction,
                second_correction,
                stream,
                kernels,
            )
        })?;
        profiler.measure(stream, "optimizer.lm_head.sync_w_t", || {
            model.lm_head.sync_transposed(stream, kernels)
        })?;
        Ok(())
    }
}

/// Single-block Llama with the dense SwiGLU branch substituted by a statically
/// shaped mixture of experts. Routing remains runtime data.
pub struct GpuLlama<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> {
    pub embedding: GpuEmbedding<VOCAB, D>,
    pub attention_norm: GpuRmsNorm<D>,
    pub qkv_proj: GpuGroupedLinear<D, 3, D>,
    pub o_proj: GpuLinear<D, D>,
    pub ffn_norm: GpuRmsNorm<D>,
    pub router: GpuTensor<f32, Rank2<D, E>>,
    pub d_router: GpuTensor<f32, Rank2<D, E>>,
    pub experts: GpuExpertFfn<E, D, FF>,
    pub final_norm: GpuRmsNorm<D>,
    pub lm_head: GpuBf16Head<D, VP>,
}

/// AdamW state for every MoE model parameter. The router remains on AdamW
/// regardless of future hidden-matrix Muon routing.
pub struct GpuLlamaAdamW<
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const FF: usize,
    const E: usize,
> {
    config: AdamWConfig,
    aux_schedule: AuxLossSchedule,
    step: u64,
    pub embedding: GpuAdamWMoments<Rank2<VOCAB, D>>,
    pub attention_norm: GpuAdamWMoments<Rank1<D>>,
    pub qkv_proj: GpuAdamWMoments<Rank3<D, 3, D>>,
    pub o_proj: GpuAdamWMoments<Rank2<D, D>>,
    pub ffn_norm: GpuAdamWMoments<Rank1<D>>,
    pub router: GpuAdamWMoments<Rank2<D, E>>,
    pub expert_gate_up: GpuAdamWMoments<Rank4<E, D, 2, FF>>,
    pub expert_down: GpuAdamWMoments<Rank3<E, FF, D>>,
    pub final_norm: GpuAdamWMoments<Rank1<D>>,
    pub lm_head: GpuAdamWMoments<Rank2<D, VP>>,
}

impl<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize, const E: usize>
    GpuLlamaAdamW<VOCAB, VP, D, FF, E>
{
    pub fn new(
        stream: &CudaStream,
        config: AdamWConfig,
        aux_schedule: AuxLossSchedule,
    ) -> Result<Self, DriverError> {
        config.validate();
        aux_schedule.validate();
        Ok(Self {
            config,
            aux_schedule,
            step: 0,
            embedding: GpuAdamWMoments::zeros(stream)?,
            attention_norm: GpuAdamWMoments::zeros(stream)?,
            qkv_proj: GpuAdamWMoments::zeros(stream)?,
            o_proj: GpuAdamWMoments::zeros(stream)?,
            ffn_norm: GpuAdamWMoments::zeros(stream)?,
            router: GpuAdamWMoments::zeros(stream)?,
            expert_gate_up: GpuAdamWMoments::zeros(stream)?,
            expert_down: GpuAdamWMoments::zeros(stream)?,
            final_norm: GpuAdamWMoments::zeros(stream)?,
            lm_head: GpuAdamWMoments::zeros(stream)?,
        })
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    pub fn aux_schedule(&self) -> AuxLossSchedule {
        self.aux_schedule
    }

    pub fn aux_coefficient(&self) -> f32 {
        self.aux_schedule.coefficient(self.step)
    }

    pub(crate) fn restore_step(&mut self, step: u64) {
        self.step = step;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        const K: usize,
        const C: usize,
    >(
        &mut self,
        model: &mut GpuLlama<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.update_profiled(model, stream, kernels, &mut profiler)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_profiled<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
        const K: usize,
        const C: usize,
        P: KernelProfiler,
    >(
        &mut self,
        model: &mut GpuLlama<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C>,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        self.step = self.step.checked_add(1).expect("AdamW step overflow");
        let (first_correction, second_correction) = self.config.bias_correction(self.step);

        macro_rules! update {
            ($name:literal, $parameter:expr, $gradient:expr, $moments:expr, $decay:expr) => {
                profiler.measure(stream, $name, || {
                    $parameter.adamw_step(
                        $gradient,
                        $moments,
                        self.config.learning_rate,
                        self.config.beta1,
                        self.config.beta2,
                        self.config.epsilon,
                        $decay,
                        first_correction,
                        second_correction,
                        stream,
                        kernels,
                    )
                })?;
            };
        }

        update!(
            "optimizer.embedding.adamw",
            model.embedding.w,
            &model.embedding.dw,
            &mut self.embedding,
            self.config.weight_decay
        );
        update!(
            "optimizer.attention_norm.adamw",
            model.attention_norm.w,
            &model.attention_norm.dw,
            &mut self.attention_norm,
            0.0
        );
        update!(
            "optimizer.qkv_proj.adamw",
            model.qkv_proj.w,
            &model.qkv_proj.dw,
            &mut self.qkv_proj,
            self.config.weight_decay
        );
        update!(
            "optimizer.o_proj.adamw",
            model.o_proj.w,
            &model.o_proj.dw,
            &mut self.o_proj,
            self.config.weight_decay
        );
        update!(
            "optimizer.ffn_norm.adamw",
            model.ffn_norm.w,
            &model.ffn_norm.dw,
            &mut self.ffn_norm,
            0.0
        );
        update!(
            "optimizer.router.adamw",
            model.router,
            &model.d_router,
            &mut self.router,
            self.config.weight_decay
        );
        update!(
            "optimizer.experts.gate_up.adamw",
            model.experts.gate_up,
            &model.experts.d_gate_up,
            &mut self.expert_gate_up,
            self.config.weight_decay
        );
        update!(
            "optimizer.experts.down.adamw",
            model.experts.down,
            &model.experts.d_down,
            &mut self.expert_down,
            self.config.weight_decay
        );
        update!(
            "optimizer.final_norm.adamw",
            model.final_norm.w,
            &model.final_norm.dw,
            &mut self.final_norm,
            0.0
        );
        profiler.measure(stream, "optimizer.qkv_proj.sync_compute", || {
            model.qkv_proj.sync_compute(stream, kernels)
        })?;
        profiler.measure(stream, "optimizer.o_proj.sync_compute", || {
            model.o_proj.sync_compute(stream, kernels)
        })?;
        profiler.measure(stream, "optimizer.experts.sync_compute", || {
            model.experts.sync_compute(stream, kernels)
        })?;
        profiler.measure(stream, "optimizer.lm_head.adamw", || {
            model.lm_head.adamw_step(
                &mut self.lm_head,
                self.config,
                first_correction,
                second_correction,
                stream,
                kernels,
            )
        })?;
        profiler.measure(stream, "optimizer.lm_head.sync_w_t", || {
            model.lm_head.sync_transposed(stream, kernels)
        })
    }
}

/// Device scratch for Muon's Newton–Schulz orthogonalization.
///
/// Every buffer is sized once for the largest hidden matrix and reused for
/// all of them via prefixes, so a steady-state Muon step performs no device
/// allocation. Gram-side buffers hold `min(rows, cols)^2` elements, which the
/// model bounds by `D^2`.
pub struct GpuMuonScratch {
    update: DeviceBuffer<f32>,
    x: DeviceBuffer<f32>,
    x_next: DeviceBuffer<f32>,
    product: DeviceBuffer<f32>,
    gram: DeviceBuffer<f32>,
    gram_squared: DeviceBuffer<f32>,
    polynomial: DeviceBuffer<f32>,
    sum_squares: DeviceBuffer<f32>,
}

impl GpuMuonScratch {
    pub fn new(
        stream: &CudaStream,
        max_update_elements: usize,
        max_matrix_elements: usize,
        max_gram_side: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            update: DeviceBuffer::zeroed(stream, max_update_elements)?,
            x: DeviceBuffer::zeroed(stream, max_matrix_elements)?,
            x_next: DeviceBuffer::zeroed(stream, max_matrix_elements)?,
            product: DeviceBuffer::zeroed(stream, max_matrix_elements)?,
            gram: DeviceBuffer::zeroed(stream, max_gram_side * max_gram_side)?,
            gram_squared: DeviceBuffer::zeroed(stream, max_gram_side * max_gram_side)?,
            polynomial: DeviceBuffer::zeroed(stream, max_gram_side * max_gram_side)?,
            sum_squares: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    /// Test hook mirroring `optim::zeroth_power_via_newton_schulz`: copy
    /// `input` (a dense `[rows, cols]` matrix) through the iteration and read
    /// the result back. Parity-binary only; other binaries see dead code.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn zeroth_power(
        &mut self,
        input: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        steps: usize,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
    ) -> Result<Vec<f32>, DriverError> {
        let elements = rows * cols;
        tensor.gather_group(
            stream,
            pairs_config(elements),
            input,
            1,
            0,
            cols as u32,
            elements as u32,
            &mut self.x,
        )?;
        newton_schulz_orthogonalize(self, rows, cols, steps, stream, tensor, gemm)?;
        let mut values = self.x.to_host_vec(stream)?;
        values.truncate(elements);
        Ok(values)
    }
}

/// Orthogonalize the `[rows, cols]` prefix of `scratch.x` in place with the
/// quintic Newton–Schulz iteration, matching the CPU reference's math.
///
/// The Gram matrix always lives on the smaller axis. For wide matrices the
/// iteration is the reference's `X = aX + (bA + cA^2) X` with `A = X X^T`.
/// For tall matrices the reference transposes, iterates, and transposes back;
/// since `A = X^T X` and its polynomial `B` are symmetric, that whole
/// round-trip collapses to `X = aX + X B`, so no f32 transpose kernel exists.
fn newton_schulz_orthogonalize(
    scratch: &mut GpuMuonScratch,
    rows: usize,
    cols: usize,
    steps: usize,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    gemm: &gemm_kernels::LoadedModule,
) -> Result<(), DriverError> {
    assert!(steps < 100, "Newton-Schulz steps must be less than 100");
    assert!(rows > 0 && cols > 0, "Muon matrices must be non-empty");
    let elements = rows * cols;
    let gram_side = rows.min(cols);
    let gram_elements = gram_side * gram_side;

    tensor.sum_squares(
        stream,
        reduction_config(),
        &scratch.x,
        elements as u32,
        &mut scratch.sum_squares,
    )?;
    tensor.scale_by_inv_norm(
        stream,
        pairs_config(elements),
        &scratch.x,
        &scratch.sum_squares,
        NEWTON_SCHULZ_EPSILON,
        elements as u32,
        &mut scratch.x_next,
    )?;
    std::mem::swap(&mut scratch.x, &mut scratch.x_next);

    for _ in 0..steps {
        if rows <= cols {
            // A = X X^T
            unsafe {
                gemm.register_gemm_nt_store(
                    stream,
                    fp32_launch_config(rows, rows),
                    rows,
                    rows,
                    cols,
                    &scratch.x,
                    &scratch.x,
                    &mut scratch.gram,
                )?;
            }
        } else {
            // A = X^T X; the fp32 family has no TN store, so zero + accumulate.
            tensor.fill(stream, pairs_config(gram_elements), 0.0, &mut scratch.gram)?;
            unsafe {
                gemm.register_gemm_tn_accumulate(
                    stream,
                    fp32_launch_config(gram_side, gram_side),
                    gram_side,
                    gram_side,
                    rows,
                    &scratch.x,
                    &scratch.x,
                    &mut scratch.gram,
                )?;
            }
        }
        unsafe {
            gemm.register_gemm_store(
                stream,
                fp32_launch_config(gram_side, gram_side),
                gram_side,
                gram_side,
                gram_side,
                &scratch.gram,
                &scratch.gram,
                &mut scratch.gram_squared,
            )?;
        }
        // B = b A + c A^2
        tensor.scaled_sum(
            stream,
            pairs_config(gram_elements),
            NEWTON_SCHULZ_B,
            &scratch.gram,
            NEWTON_SCHULZ_C,
            &scratch.gram_squared,
            gram_elements as u32,
            &mut scratch.polynomial,
        )?;
        if rows <= cols {
            // X = a X + B X
            unsafe {
                gemm.register_gemm_store(
                    stream,
                    fp32_launch_config(rows, cols),
                    rows,
                    cols,
                    rows,
                    &scratch.polynomial,
                    &scratch.x,
                    &mut scratch.product,
                )?;
            }
        } else {
            // X = a X + X B
            unsafe {
                gemm.register_gemm_store(
                    stream,
                    fp32_launch_config(rows, cols),
                    rows,
                    cols,
                    cols,
                    &scratch.x,
                    &scratch.polynomial,
                    &mut scratch.product,
                )?;
            }
        }
        tensor.scaled_sum(
            stream,
            pairs_config(elements),
            NEWTON_SCHULZ_A,
            &scratch.x,
            1.0,
            &scratch.product,
            elements as u32,
            &mut scratch.x_next,
        )?;
        std::mem::swap(&mut scratch.x, &mut scratch.x_next);
    }
    Ok(())
}

/// One Muon update over a `[rows, groups, cols]` parameter whose groups are
/// independent `[rows, cols]` matrices (`groups = 1` for plain linears).
///
/// Momentum and the Nesterov interpolation are elementwise and run over the
/// whole interleaved buffer; orthogonalization and the fused decay/apply then
/// run per group so each projection is orthogonalized on its own, matching
/// the CPU reference's separate `q/k/v` and `gate/up` matrices.
#[allow(clippy::too_many_arguments)]
fn muon_step_raw(
    parameter: &mut DeviceBuffer<f32>,
    gradient: &DeviceBuffer<f32>,
    momentum: &mut DeviceBuffer<f32>,
    rows: usize,
    groups: usize,
    cols: usize,
    config: MuonConfig,
    scratch: &mut GpuMuonScratch,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    gemm: &gemm_kernels::LoadedModule,
) -> Result<(), DriverError> {
    let total = rows * groups * cols;
    let per_group = rows * cols;
    tensor.ema_momentum(
        stream,
        pairs_config(total),
        gradient,
        config.momentum,
        momentum,
    )?;
    let (gradient_weight, momentum_weight) = if config.nesterov {
        (1.0 - config.momentum, config.momentum)
    } else {
        (0.0, 1.0)
    };
    tensor.scaled_sum(
        stream,
        pairs_config(total),
        gradient_weight,
        gradient,
        momentum_weight,
        momentum,
        total as u32,
        &mut scratch.update,
    )?;

    let aspect_ratio_scale = ((rows as f32 / cols as f32).max(1.0)).sqrt();
    let decay = 1.0 - config.learning_rate * config.weight_decay;
    let update_scale = config.learning_rate * aspect_ratio_scale;
    for group in 0..groups {
        tensor.gather_group(
            stream,
            pairs_config(per_group),
            &scratch.update,
            groups as u32,
            group as u32,
            cols as u32,
            per_group as u32,
            &mut scratch.x,
        )?;
        newton_schulz_orthogonalize(
            scratch,
            rows,
            cols,
            config.newton_schulz_steps,
            stream,
            tensor,
            gemm,
        )?;
        unsafe {
            tensor.muon_apply_group(
                stream,
                pairs_config(per_group),
                &scratch.x,
                decay,
                update_scale,
                groups as u32,
                group as u32,
                cols as u32,
                per_group as u32,
                parameter,
            )?;
        }
    }
    Ok(())
}

/// GPU-resident mixed Muon/AdamW state mirroring `optim::LlamaMuon`'s
/// routing: hidden projection matrices take Muon, while the embedding,
/// norms, and lm-head keep AdamW (the lm-head over its padded `[D, VP]`
/// master, exactly as in [`GpuDenseLlamaAdamW`]).
pub struct GpuLlamaMuon<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize> {
    muon_config: MuonConfig,
    adamw_config: AdamWConfig,
    step: u64,
    scratch: GpuMuonScratch,
    pub embedding: GpuAdamWMoments<Rank2<VOCAB, D>>,
    pub attention_norm: GpuAdamWMoments<Rank1<D>>,
    pub qkv_proj: GpuMuonMomentum<Rank3<D, 3, D>>,
    pub o_proj: GpuMuonMomentum<Rank2<D, D>>,
    pub ffn_norm: GpuAdamWMoments<Rank1<D>>,
    pub gate_up_proj: GpuMuonMomentum<Rank3<D, 2, FF>>,
    pub down_proj: GpuMuonMomentum<Rank2<FF, D>>,
    pub final_norm: GpuAdamWMoments<Rank1<D>>,
    pub lm_head: GpuAdamWMoments<Rank2<D, VP>>,
}

impl<const VOCAB: usize, const VP: usize, const D: usize, const FF: usize>
    GpuLlamaMuon<VOCAB, VP, D, FF>
{
    pub fn new(
        stream: &CudaStream,
        muon_config: MuonConfig,
        adamw_config: AdamWConfig,
    ) -> Result<Self, DriverError> {
        muon_config.validate();
        adamw_config.validate();
        // Largest interleaved hidden parameter, largest single matrix, and
        // the Gram side (min dimension), which every hidden matrix bounds by D.
        let max_update_elements = (3 * D * D).max(2 * D * FF);
        let max_matrix_elements = (D * D).max(D * FF);
        Ok(Self {
            muon_config,
            adamw_config,
            step: 0,
            scratch: GpuMuonScratch::new(stream, max_update_elements, max_matrix_elements, D)?,
            embedding: GpuAdamWMoments::zeros(stream)?,
            attention_norm: GpuAdamWMoments::zeros(stream)?,
            qkv_proj: GpuMuonMomentum::zeros(stream)?,
            o_proj: GpuMuonMomentum::zeros(stream)?,
            ffn_norm: GpuAdamWMoments::zeros(stream)?,
            gate_up_proj: GpuMuonMomentum::zeros(stream)?,
            down_proj: GpuMuonMomentum::zeros(stream)?,
            final_norm: GpuAdamWMoments::zeros(stream)?,
            lm_head: GpuAdamWMoments::zeros(stream)?,
        })
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn muon_config(&self) -> MuonConfig {
        self.muon_config
    }

    pub fn adamw_config(&self) -> AdamWConfig {
        self.adamw_config
    }

    pub fn update<
        const N: usize,
        const NP: usize,
        const T: usize,
        const H: usize,
        const HD: usize,
    >(
        &mut self,
        model: &mut GpuDenseLlama<N, NP, T, VOCAB, VP, D, H, HD, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        self.step = self.step.checked_add(1).expect("Muon step overflow");
        let (first_correction, second_correction) = self.adamw_config.bias_correction(self.step);

        macro_rules! adamw {
            ($field:ident, $weight_decay:expr) => {
                model.$field.w.adamw_step(
                    &model.$field.dw,
                    &mut self.$field,
                    self.adamw_config.learning_rate,
                    self.adamw_config.beta1,
                    self.adamw_config.beta2,
                    self.adamw_config.epsilon,
                    $weight_decay,
                    first_correction,
                    second_correction,
                    stream,
                    tensor,
                )?;
            };
        }
        macro_rules! muon {
            ($field:ident, $rows:expr, $groups:expr, $cols:expr) => {
                muon_step_raw(
                    model.$field.w.as_device_buffer_mut(),
                    model.$field.dw.as_device_buffer(),
                    self.$field.momentum.as_device_buffer_mut(),
                    $rows,
                    $groups,
                    $cols,
                    self.muon_config,
                    &mut self.scratch,
                    stream,
                    tensor,
                    gemm,
                )?;
            };
        }

        adamw!(embedding, self.adamw_config.weight_decay);
        adamw!(attention_norm, 0.0);
        muon!(qkv_proj, D, 3, D);
        muon!(o_proj, D, 1, D);
        adamw!(ffn_norm, 0.0);
        muon!(gate_up_proj, D, 2, FF);
        muon!(down_proj, FF, 1, D);
        adamw!(final_norm, 0.0);
        model.sync_linear_compute(stream, tensor)?;
        model.lm_head.adamw_step(
            &mut self.lm_head,
            self.adamw_config,
            first_correction,
            second_correction,
            stream,
            tensor,
        )?;
        model.lm_head.sync_transposed(stream, tensor)
    }
}

struct InputStaging<const N: usize> {
    tokens: PinnedHostBuffer<u32>,
    targets: PinnedHostBuffer<u32>,
    copied: CudaEvent,
    pending: bool,
}

impl<const N: usize> InputStaging<N> {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            tokens: PinnedHostBuffer::zeroed(stream.context(), N)?,
            targets: PinnedHostBuffer::zeroed(stream.context(), N)?,
            copied: stream.context().new_event(None)?,
            pending: false,
        })
    }
}

/// Persistent device and pinned-host storage for one model's training steps.
///
/// Create this once and pass it to every forward/backward call. All operator
/// outputs are written into these allocations, so a steady-state step performs
/// no device allocation or synchronous device free.
///
/// The packed lm-head buffers are `NP` rows tall. Rows `N..NP` of
/// `head_input` are zeroed at allocation and never written afterwards
/// (`convert_f32_to_bf16_pairs` stops at the input length), and the same rows
/// of `logits` always hold zeros: the forward GEMM computes them from the
/// zero input rows and the classifier backward never touches them.
///
struct GpuMoeRoutingWorkspace<const N: usize, const D: usize, const E: usize, const K: usize> {
    logits: GpuTensor<f32, Rank2<N, E>>,
    probabilities: GpuTensor<f32, Rank2<N, E>>,
    selected_experts: GpuTensor<u32, Rank2<N, K>>,
    gate_weights: GpuTensor<f32, Rank2<N, K>>,
    slots: GpuTensor<u32, Rank2<N, K>>,
    assignment_counts: GpuTensor<u32, Rank1<E>>,
    probability_sums: GpuTensor<f32, Rank1<E>>,
    gate_gradients: GpuTensor<f32, Rank2<N, K>>,
    dlogits: GpuTensor<f32, Rank2<N, E>>,
    router_dx: GpuTensor<f32, Rank2<N, D>>,
}

impl<const N: usize, const D: usize, const E: usize, const K: usize>
    GpuMoeRoutingWorkspace<N, D, E, K>
{
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        assert!(E > 0, "MoE must have at least one expert");
        assert!(K > 0 && K <= E, "MoE top-k must be in 1..=E");
        Ok(Self {
            logits: GpuTensor::zeros(stream)?,
            probabilities: GpuTensor::zeros(stream)?,
            selected_experts: GpuTensor::zeros(stream)?,
            gate_weights: GpuTensor::zeros(stream)?,
            slots: GpuTensor::zeros(stream)?,
            assignment_counts: GpuTensor::zeros(stream)?,
            probability_sums: GpuTensor::zeros(stream)?,
            gate_gradients: GpuTensor::zeros(stream)?,
            dlogits: GpuTensor::zeros(stream)?,
            router_dx: GpuTensor::zeros(stream)?,
        })
    }
}

/// `E`, `K`, and `C` default to zero for the dense model. MoE workspaces set
/// all three and receive routing plus capacity-binned expert buffers.
pub struct GpuLlamaWorkspace<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
    const E: usize = 0,
    const K: usize = 0,
    const C: usize = 0,
> {
    tokens: GpuTensor<u32, Rank1<N>>,
    targets: GpuTensor<u32, Rank1<N>>,
    staging: [InputStaging<N>; 2],
    next_staging: usize,
    attention_input: GpuTensor<f32, Rank2<N, D>>,
    attention_normalized: GpuTensor<f32, Rank2<N, D>>,
    qkv: GpuTensor<f32, Rank3<N, 3, D>>,
    q: GpuTensor<f32, Rank2<N, D>>,
    k: GpuTensor<f32, Rank2<N, D>>,
    v: GpuTensor<f32, Rank2<N, D>>,
    attended: GpuTensor<f32, Rank2<N, D>>,
    attention_logsumexp: GpuTensor<f32, Rank2<N, H>>,
    attention_dot: GpuTensor<f32, Rank2<N, H>>,
    ffn_input: GpuTensor<f32, Rank2<N, D>>,
    ffn_normalized: GpuTensor<f32, Rank2<N, D>>,
    gate_up: GpuTensor<f32, Rank3<N, 2, FF>>,
    gate: GpuTensor<f32, Rank2<N, FF>>,
    up: GpuTensor<f32, Rank2<N, FF>>,
    activated: GpuTensor<f32, Rank2<N, FF>>,
    final_input: GpuTensor<f32, Rank2<N, D>>,
    final_normalized: GpuTensor<f32, Rank2<N, D>>,
    projection_output: GpuTensor<f32, Rank2<N, D>>,
    head_input: DeviceBuffer<u32>,
    head_input_t: DeviceBuffer<u32>,
    logits: DeviceBuffer<u32>,
    dlogits_t: DeviceBuffer<u32>,
    d_head_input: DeviceBuffer<u32>,
    head_input_tma: Bf16PairsTmaMap,
    head_input_t_tma: Bf16PairsTmaMap,
    logits_tma: Bf16PairsTmaMap,
    dlogits_t_tma: Bf16PairsTmaMap,
    linear_scratch: Option<Bf16LinearScratch<N, D, FF>>,
    norm_backward_inv: GpuTensor<f32, Rank1<N>>,
    losses: GpuTensor<f32, Rank1<N>>,
    loss_sum: GpuTensor<f32, Rank1<1>>,
    loss: GpuTensor<f32, Rank1<1>>,
    d_model_0: GpuTensor<f32, Rank2<N, D>>,
    d_model_1: GpuTensor<f32, Rank2<N, D>>,
    d_model_2: GpuTensor<f32, Rank2<N, D>>,
    d_model_3: GpuTensor<f32, Rank2<N, D>>,
    d_model_4: GpuTensor<f32, Rank2<N, D>>,
    d_ff_0: GpuTensor<f32, Rank2<N, FF>>,
    d_ff_1: GpuTensor<f32, Rank2<N, FF>>,
    d_ff_2: GpuTensor<f32, Rank2<N, FF>>,
    routing: Option<GpuMoeRoutingWorkspace<N, D, E, K>>,
    experts: Option<GpuExpertWorkspace<E, C, D, FF>>,
}

impl<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF, E, K, C>
{
    pub fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        let head_input = DeviceBuffer::zeroed(stream, NP * D / 2)?;
        let head_input_t = DeviceBuffer::zeroed(stream, D * NP / 2)?;
        let logits = DeviceBuffer::zeroed(stream, NP * VP / 2)?;
        let dlogits_t = DeviceBuffer::zeroed(stream, VP * NP / 2)?;
        // SAFETY: the mapped buffers live in this workspace beside their maps
        // and are never reallocated.
        let head_input_tma = unsafe { create_bf16_pairs_tma_map(stream, &head_input, D, NP)? };
        let head_input_t_tma = unsafe { create_bf16_pairs_tma_map(stream, &head_input_t, NP, D)? };
        let logits_tma = unsafe { create_bf16_pairs_tma_map(stream, &logits, VP, NP)? };
        let dlogits_t_tma = unsafe { create_bf16_pairs_tma_map(stream, &dlogits_t, NP, VP)? };
        Ok(Self {
            tokens: GpuTensor::zeros(stream)?,
            targets: GpuTensor::zeros(stream)?,
            staging: [InputStaging::new(stream)?, InputStaging::new(stream)?],
            next_staging: 0,
            attention_input: GpuTensor::zeros(stream)?,
            attention_normalized: GpuTensor::zeros(stream)?,
            qkv: GpuTensor::zeros(stream)?,
            q: GpuTensor::zeros(stream)?,
            k: GpuTensor::zeros(stream)?,
            v: GpuTensor::zeros(stream)?,
            attended: GpuTensor::zeros(stream)?,
            attention_logsumexp: GpuTensor::zeros(stream)?,
            attention_dot: GpuTensor::zeros(stream)?,
            ffn_input: GpuTensor::zeros(stream)?,
            ffn_normalized: GpuTensor::zeros(stream)?,
            gate_up: GpuTensor::zeros(stream)?,
            gate: GpuTensor::zeros(stream)?,
            up: GpuTensor::zeros(stream)?,
            activated: GpuTensor::zeros(stream)?,
            final_input: GpuTensor::zeros(stream)?,
            final_normalized: GpuTensor::zeros(stream)?,
            projection_output: GpuTensor::zeros(stream)?,
            head_input,
            head_input_t,
            logits,
            dlogits_t,
            d_head_input: DeviceBuffer::zeroed(stream, NP * D / 2)?,
            head_input_tma,
            head_input_t_tma,
            logits_tma,
            dlogits_t_tma,
            linear_scratch: if N.is_multiple_of(TC_TILE)
                && D.is_multiple_of(TC_TILE)
                && FF.is_multiple_of(TC_TILE)
            {
                Some(Bf16LinearScratch::new(stream)?)
            } else {
                None
            },
            norm_backward_inv: GpuTensor::zeros(stream)?,
            losses: GpuTensor::zeros(stream)?,
            loss_sum: GpuTensor::zeros(stream)?,
            loss: GpuTensor::zeros(stream)?,
            d_model_0: GpuTensor::zeros(stream)?,
            d_model_1: GpuTensor::zeros(stream)?,
            d_model_2: GpuTensor::zeros(stream)?,
            d_model_3: GpuTensor::zeros(stream)?,
            d_model_4: GpuTensor::zeros(stream)?,
            d_ff_0: GpuTensor::zeros(stream)?,
            d_ff_1: GpuTensor::zeros(stream)?,
            d_ff_2: GpuTensor::zeros(stream)?,
            routing: if E == 0 && K == 0 && C == 0 {
                None
            } else {
                assert!(E > 0 && K > 0 && C > 0, "MoE workspace requires E, K, C");
                Some(GpuMoeRoutingWorkspace::new(stream)?)
            },
            experts: if E == 0 && K == 0 && C == 0 {
                None
            } else {
                Some(GpuExpertWorkspace::new(stream)?)
            },
        })
    }

    /// Packed-bf16 logits (dlogits after a backward pass). Parity-test
    /// accessor: binaries other than the parity check see it as dead code.
    #[allow(dead_code)]
    pub fn logits_words(&self) -> &DeviceBuffer<u32> {
        &self.logits
    }

    pub fn loss(&self) -> &GpuTensor<f32, Rank1<1>> {
        &self.loss
    }

    /// Whether this workspace's shapes route the block linears through the
    /// bf16 tcgen05 path. Lets the aligned parity gate assert it is actually
    /// exercising that path rather than silently falling back to fp32.
    pub fn tcgen05_linears_active(&self) -> bool {
        self.linear_scratch.is_some()
    }

    pub fn expert_workspace(&self) -> Option<&GpuExpertWorkspace<E, C, D, FF>> {
        self.experts.as_ref()
    }

    pub fn expert_workspace_mut(&mut self) -> Option<&mut GpuExpertWorkspace<E, C, D, FF>> {
        self.experts.as_mut()
    }

    /// Host readback of one packed-bf16 logits row, widened to f32.
    ///
    /// Sampling and debugging only: this synchronizes the stream after copying
    /// only the requested row.
    pub fn logits_row(&self, row: usize, stream: &CudaStream) -> Result<Vec<f32>, DriverError> {
        assert!(row < NP);
        let stride = VP / 2;
        let byte_offset = row
            .checked_mul(stride)
            .and_then(|offset| offset.checked_mul(std::mem::size_of::<u32>()))
            .expect("logits row byte offset overflow");
        let source = self
            .logits
            .cu_deviceptr()
            .checked_add(byte_offset as u64)
            .expect("logits row device pointer overflow");
        let mut words = vec![0u32; stride];
        // SAFETY: `row < NP` and the workspace allocation of `NP * VP / 2`
        // words guarantee that `source` has `words.len()` readable elements.
        // The initialized host vector remains live until stream synchronization.
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                words.as_mut_ptr(),
                source,
                std::mem::size_of_val(words.as_slice()),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        Ok(words
            .iter()
            .flat_map(|&word| {
                [
                    f32::from_bits((word & 0xFFFF) << 16),
                    f32::from_bits((word >> 16) << 16),
                ]
            })
            .collect())
    }

    fn upload_inputs(
        &mut self,
        tokens: &[usize; N],
        targets: &[usize; N],
        stream: &CudaStream,
    ) -> Result<(), DriverError> {
        let slot = &mut self.staging[self.next_staging];
        if slot.pending {
            slot.copied.synchronize()?;
        }
        for i in 0..N {
            assert!(tokens[i] < VOCAB);
            assert!(targets[i] < VOCAB);
            slot.tokens[i] = tokens[i] as u32;
            slot.targets[i] = targets[i] as u32;
        }

        // SAFETY: the staging slot remains owned by this workspace and is not
        // read, mutated, or dropped until `copied` has synchronized before its
        // next reuse. The event is recorded after both copies on this stream.
        unsafe {
            self.tokens
                .as_device_buffer_mut()
                .copy_from_pinned_host_async(stream, &slot.tokens)?;
            self.targets
                .as_device_buffer_mut()
                .copy_from_pinned_host_async(stream, &slot.targets)?;
        }
        slot.copied.record(stream)?;
        slot.pending = true;
        self.next_staging ^= 1;
        Ok(())
    }
}

impl<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
> GpuLlama<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C>
{
    pub fn from_cpu(
        stream: &CudaStream,
        model: &MoeLlama<N, T, VOCAB, D, H, HD, FF, E, K, C>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(N <= u32::MAX as usize);
        assert_eq!(N % T, 0);
        assert_eq!(D, H * HD);
        assert_eq!(NP, N.next_multiple_of(TC_TILE));
        assert!(VP >= VOCAB);
        assert_eq!(VP % TC_TILE, 0);
        assert_eq!(D % TC_TILE, 0);
        assert!(E > 0 && K > 0 && K <= E && C > 0);
        Ok(Self {
            embedding: GpuEmbedding::from_cpu(stream, &model.embedding)?,
            attention_norm: GpuRmsNorm::from_cpu(stream, &model.attention_norm)?,
            qkv_proj: GpuGroupedLinear::from_cpu(
                stream,
                [&model.q_proj, &model.k_proj, &model.v_proj],
            )?,
            o_proj: GpuLinear::from_cpu(stream, &model.o_proj)?,
            ffn_norm: GpuRmsNorm::from_cpu(stream, &model.ffn_norm)?,
            router: GpuTensor::from_host(stream, model.ffn.router.w.as_slice())?,
            d_router: GpuTensor::zeros(stream)?,
            experts: GpuExpertFfn::from_cpu(stream, &model.ffn.experts)?,
            final_norm: GpuRmsNorm::from_cpu(stream, &model.final_norm)?,
            lm_head: GpuBf16Head::from_cpu(stream, &model.lm_head)?,
        })
    }

    pub(crate) fn sync_compute(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        self.qkv_proj.sync_compute(stream, kernels)?;
        self.o_proj.sync_compute(stream, kernels)?;
        self.experts.sync_compute(stream, kernels)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: &[usize; N],
        targets: &[usize; N],
        aux_coefficient: f32,
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF, E, K, C>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.forward_profiled(
            tokens,
            targets,
            aux_coefficient,
            workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            llama,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_profiled<P: KernelProfiler>(
        &self,
        tokens: &[usize; N],
        targets: &[usize; N],
        aux_coefficient: f32,
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF, E, K, C>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        assert!(aux_coefficient.is_finite() && aux_coefficient >= 0.0);
        workspace.upload_inputs(tokens, targets, stream)?;
        self.embedding.forward_into(
            &workspace.tokens,
            &mut workspace.attention_input,
            stream,
            llama,
            profiler,
            "forward.embedding",
        )?;
        self.attention_norm.forward_into(
            &workspace.attention_input,
            &mut workspace.attention_normalized,
            stream,
            llama,
            profiler,
            "forward.attention_norm",
        )?;
        self.qkv_proj.forward_into(
            &workspace.attention_normalized,
            &mut workspace.qkv,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            "forward.qkv_proj.gemm",
        )?;
        profiler.measure(stream, "forward.qkv_proj.split", || {
            llama.split_group3(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                workspace.qkv.as_device_buffer(),
                D as u32,
                workspace.q.as_device_buffer_mut(),
                workspace.k.as_device_buffer_mut(),
                workspace.v.as_device_buffer_mut(),
            )
        })?;
        rope_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &mut workspace.d_model_0,
            false,
            stream,
            llama,
            profiler,
            "forward.q_rope",
        )?;
        std::mem::swap(&mut workspace.q, &mut workspace.d_model_0);
        rope_into::<N, T, D, H, HD, P>(
            &workspace.k,
            &mut workspace.d_model_0,
            false,
            stream,
            llama,
            profiler,
            "forward.k_rope",
        )?;
        std::mem::swap(&mut workspace.k, &mut workspace.d_model_0);
        flash_attention_forward_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &mut workspace.attended,
            &mut workspace.attention_logsumexp,
            stream,
            flash,
            profiler,
        )?;
        self.o_proj.forward_into(
            &workspace.attended,
            &mut workspace.projection_output,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            "forward.o_proj.gemm",
        )?;
        add_into(
            &workspace.attention_input,
            &workspace.projection_output,
            &mut workspace.ffn_input,
            stream,
            tensor,
            profiler,
            "forward.attention_residual",
        )?;
        self.ffn_norm.forward_into(
            &workspace.ffn_input,
            &mut workspace.ffn_normalized,
            stream,
            llama,
            profiler,
            "forward.ffn_norm",
        )?;

        let routing = workspace.routing.as_mut().expect("MoE routing workspace");
        let experts = workspace.experts.as_mut().expect("MoE expert workspace");
        profiler.measure(stream, "forward.router.logits", || {
            llama.router_logits(
                stream,
                LaunchConfig {
                    grid_dim: (N as u32, 1, 1),
                    block_dim: (E as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                workspace.ffn_normalized.as_device_buffer(),
                self.router.as_device_buffer(),
                D as u32,
                E as u32,
                routing.logits.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "forward.router.topk", || unsafe {
            llama.router_softmax_topk(
                stream,
                LaunchConfig::for_num_elems(N as u32),
                routing.logits.as_device_buffer(),
                E as u32,
                K as u32,
                routing.probabilities.as_device_buffer_mut(),
                routing.selected_experts.as_device_buffer_mut(),
                routing.gate_weights.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "forward.router.assign", || unsafe {
            llama.moe_bin_assign(
                stream,
                LaunchConfig {
                    grid_dim: (E as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                routing.selected_experts.as_device_buffer(),
                N as u32,
                E as u32,
                K as u32,
                C as u32,
                routing.slots.as_device_buffer_mut(),
                routing.assignment_counts.as_device_buffer_mut(),
            )
        })?;
        fill_zero(
            &mut experts.bin_input,
            stream,
            tensor,
            profiler,
            "forward.router.zero_bins",
        )?;
        profiler.measure(stream, "forward.router.scatter", || unsafe {
            llama.moe_scatter(
                stream,
                LaunchConfig::for_num_elems((N * K * D) as u32),
                workspace.ffn_normalized.as_device_buffer(),
                routing.selected_experts.as_device_buffer(),
                routing.slots.as_device_buffer(),
                D as u32,
                K as u32,
                C as u32,
                experts.bin_input.as_device_buffer_mut(),
            )
        })?;
        self.experts
            .forward_profiled(experts, stream, tensor, gemm, gemm_bf16, llama, profiler)?;
        profiler.measure(stream, "forward.router.gather", || {
            llama.moe_gather_combine(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                experts.bin_output.as_device_buffer(),
                routing.selected_experts.as_device_buffer(),
                routing.gate_weights.as_device_buffer(),
                routing.slots.as_device_buffer(),
                D as u32,
                K as u32,
                C as u32,
                workspace.projection_output.as_device_buffer_mut(),
            )
        })?;
        add_into(
            &workspace.ffn_input,
            &workspace.projection_output,
            &mut workspace.final_input,
            stream,
            tensor,
            profiler,
            "forward.ffn_residual",
        )?;
        self.final_norm.forward_into(
            &workspace.final_input,
            &mut workspace.final_normalized,
            stream,
            llama,
            profiler,
            "forward.final_norm",
        )?;
        profiler.measure(stream, "forward.lm_head.quantize", || {
            tensor.convert_f32_to_bf16_pairs(
                stream,
                pairs_config(N * D / 2),
                workspace.final_normalized.as_device_buffer(),
                &mut workspace.head_input,
            )
        })?;
        self.lm_head.forward_into::<NP, P>(
            &workspace.head_input_tma,
            &mut workspace.logits,
            stream,
            gemm_bf16,
            profiler,
            "forward.lm_head.gemm",
        )?;
        cross_entropy_into::<N, VOCAB, VP, P>(
            &workspace.logits,
            &workspace.targets,
            &mut workspace.losses,
            &mut workspace.loss_sum,
            &mut workspace.loss,
            stream,
            tensor,
            llama,
            profiler,
        )?;
        fill_zero(
            &mut routing.probability_sums,
            stream,
            tensor,
            profiler,
            "forward.router.zero_probability_sums",
        )?;
        profiler.measure(stream, "forward.router.aux_probability_sums", || unsafe {
            llama.moe_probability_sums(
                stream,
                LaunchConfig {
                    grid_dim: (E as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                routing.probabilities.as_device_buffer(),
                N as u32,
                E as u32,
                routing.probability_sums.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "forward.router.aux_loss", || unsafe {
            llama.moe_aux_loss(
                stream,
                LaunchConfig::for_num_elems(1),
                routing.probability_sums.as_device_buffer(),
                routing.assignment_counts.as_device_buffer(),
                N as u32,
                E as u32,
                K as u32,
                aux_coefficient,
                workspace.loss.as_device_buffer_mut(),
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &mut self,
        aux_coefficient: f32,
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF, E, K, C>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.backward_profiled(
            aux_coefficient,
            workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            llama,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_profiled<P: KernelProfiler>(
        &mut self,
        aux_coefficient: f32,
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF, E, K, C>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        assert!(aux_coefficient.is_finite() && aux_coefficient >= 0.0);
        cross_entropy_backward_into::<N, VOCAB, VP, P>(
            &workspace.targets,
            &mut workspace.logits,
            stream,
            llama,
            profiler,
        )?;
        profiler.measure(stream, "backward.lm_head.transpose_input", || unsafe {
            tensor.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(NP, D),
                &workspace.head_input,
                NP as u32,
                D as u32,
                &mut workspace.head_input_t,
            )
        })?;
        profiler.measure(stream, "backward.lm_head.transpose_dlogits", || unsafe {
            tensor.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(NP, VP),
                &workspace.logits,
                NP as u32,
                VP as u32,
                &mut workspace.dlogits_t,
            )
        })?;
        self.lm_head.backward_weight::<NP, P>(
            &workspace.head_input_t_tma,
            &workspace.dlogits_t_tma,
            stream,
            gemm_bf16,
            profiler,
            "backward.lm_head.weight_gemm",
        )?;
        self.lm_head.backward_input::<NP, P>(
            &workspace.logits_tma,
            &mut workspace.d_head_input,
            stream,
            gemm_bf16,
            profiler,
            "backward.lm_head.input_gemm",
        )?;
        profiler.measure(stream, "backward.lm_head.dequantize", || {
            tensor.convert_bf16_pairs_to_f32(
                stream,
                elementwise_config::<Rank2<N, D>>(),
                &workspace.d_head_input,
                workspace.d_model_0.as_device_buffer_mut(),
            )
        })?;
        self.final_norm.backward_into(
            &workspace.final_input,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
            &mut workspace.norm_backward_inv,
            stream,
            llama,
            profiler,
            ["backward.final_norm.input", "backward.final_norm.weight"],
        )?;

        let routing = workspace.routing.as_mut().expect("MoE routing workspace");
        let experts = workspace.experts.as_mut().expect("MoE expert workspace");
        fill_zero(
            &mut experts.d_bin_output,
            stream,
            tensor,
            profiler,
            "backward.router.zero_dy_bins",
        )?;
        profiler.measure(stream, "backward.router.scatter_dy", || unsafe {
            llama.moe_scatter_dy(
                stream,
                LaunchConfig::for_num_elems((N * K) as u32),
                experts.bin_output.as_device_buffer(),
                workspace.d_model_1.as_device_buffer(),
                routing.selected_experts.as_device_buffer(),
                routing.gate_weights.as_device_buffer(),
                routing.slots.as_device_buffer(),
                D as u32,
                K as u32,
                C as u32,
                experts.d_bin_output.as_device_buffer_mut(),
                routing.gate_gradients.as_device_buffer_mut(),
            )
        })?;
        self.experts
            .backward_profiled(experts, stream, tensor, gemm, gemm_bf16, llama, profiler)?;
        profiler.measure(stream, "backward.router.gather_dx", || {
            llama.moe_gather_dx(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                experts.d_bin_input.as_device_buffer(),
                routing.selected_experts.as_device_buffer(),
                routing.slots.as_device_buffer(),
                D as u32,
                K as u32,
                C as u32,
                workspace.d_model_3.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "backward.router.softmax", || unsafe {
            llama.router_backward(
                stream,
                LaunchConfig::for_num_elems(N as u32),
                routing.probabilities.as_device_buffer(),
                routing.selected_experts.as_device_buffer(),
                routing.gate_weights.as_device_buffer(),
                routing.gate_gradients.as_device_buffer(),
                routing.assignment_counts.as_device_buffer(),
                N as u32,
                E as u32,
                K as u32,
                aux_coefficient,
                routing.dlogits.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "backward.router.input", || {
            llama.router_backward_input(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                routing.dlogits.as_device_buffer(),
                self.router.as_device_buffer(),
                E as u32,
                routing.router_dx.as_device_buffer_mut(),
            )
        })?;
        profiler.measure(stream, "backward.router.weight", || {
            llama.router_backward_weight(
                stream,
                LaunchConfig::for_num_elems((D * E) as u32),
                workspace.ffn_normalized.as_device_buffer(),
                routing.dlogits.as_device_buffer(),
                N as u32,
                E as u32,
                self.d_router.as_device_buffer_mut(),
            )
        })?;
        add_into(
            &workspace.d_model_3,
            &routing.router_dx,
            &mut workspace.d_model_4,
            stream,
            tensor,
            profiler,
            "backward.router.combine_dx",
        )?;
        self.ffn_norm.backward_into(
            &workspace.ffn_input,
            &workspace.d_model_4,
            &mut workspace.d_model_0,
            &mut workspace.norm_backward_inv,
            stream,
            llama,
            profiler,
            ["backward.ffn_norm.input", "backward.ffn_norm.weight"],
        )?;
        add_into(
            &workspace.d_model_1,
            &workspace.d_model_0,
            &mut workspace.d_model_2,
            stream,
            tensor,
            profiler,
            "backward.ffn_residual",
        )?;
        self.o_proj.backward_into(
            &workspace.attended,
            &workspace.d_model_2,
            &mut workspace.d_model_0,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            ["backward.o_proj.weight_gemm", "backward.o_proj.input_gemm"],
        )?;
        flash_attention_backward_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.attended,
            &workspace.attention_logsumexp,
            &mut workspace.attention_dot,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
            &mut workspace.d_model_3,
            &mut workspace.d_model_4,
            stream,
            flash,
            profiler,
        )?;
        rope_into::<N, T, D, H, HD, P>(
            &workspace.d_model_1,
            &mut workspace.d_model_0,
            true,
            stream,
            llama,
            profiler,
            "backward.q_rope",
        )?;
        rope_into::<N, T, D, H, HD, P>(
            &workspace.d_model_3,
            &mut workspace.d_model_1,
            true,
            stream,
            llama,
            profiler,
            "backward.k_rope",
        )?;
        profiler.measure(stream, "backward.qkv_proj.join", || unsafe {
            llama.join_group3(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                workspace.d_model_0.as_device_buffer(),
                workspace.d_model_1.as_device_buffer(),
                workspace.d_model_4.as_device_buffer(),
                D as u32,
                workspace.qkv.as_device_buffer_mut(),
            )
        })?;
        self.qkv_proj.backward_into(
            &workspace.attention_normalized,
            &workspace.qkv,
            &mut workspace.d_model_3,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            [
                "backward.qkv_proj.weight_gemm",
                "backward.qkv_proj.input_gemm",
            ],
        )?;
        self.attention_norm.backward_into(
            &workspace.attention_input,
            &workspace.d_model_3,
            &mut workspace.d_model_0,
            &mut workspace.norm_backward_inv,
            stream,
            llama,
            profiler,
            [
                "backward.attention_norm.input",
                "backward.attention_norm.weight",
            ],
        )?;
        add_into(
            &workspace.d_model_2,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
            stream,
            tensor,
            profiler,
            "backward.attention_residual",
        )?;
        self.embedding.backward(
            &workspace.tokens,
            &workspace.d_model_1,
            stream,
            llama,
            profiler,
            "backward.embedding",
        )
    }

    pub fn zero_grad(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.zero_grad_profiled(stream, tensor, &mut profiler)
    }

    pub fn zero_grad_profiled<P: KernelProfiler>(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        macro_rules! zero {
            ($name:literal, $gradient:expr) => {
                fill_zero($gradient, stream, tensor, profiler, $name)?;
            };
        }
        zero!("zero_grad.embedding", &mut self.embedding.dw);
        zero!("zero_grad.attention_norm", &mut self.attention_norm.dw);
        zero!("zero_grad.qkv_proj", &mut self.qkv_proj.dw);
        zero!("zero_grad.o_proj", &mut self.o_proj.dw);
        zero!("zero_grad.ffn_norm", &mut self.ffn_norm.dw);
        zero!("zero_grad.router", &mut self.d_router);
        zero!("zero_grad.experts.gate_up", &mut self.experts.d_gate_up);
        zero!("zero_grad.experts.down", &mut self.experts.d_down);
        zero!("zero_grad.final_norm", &mut self.final_norm.dw);
        self.lm_head
            .zero_grad(stream, tensor, profiler, "zero_grad.lm_head")
    }
}

impl<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> GpuDenseLlama<N, NP, T, VOCAB, VP, D, H, HD, FF>
{
    pub fn from_cpu(
        stream: &CudaStream,
        model: &Llama<N, T, VOCAB, D, H, HD, FF>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(N <= u32::MAX as usize);
        assert_eq!(N % T, 0);
        assert_eq!(D, H * HD);
        // tcgen05 head contract: padded tokens and vocabulary are tile
        // multiples, and D serves as both an output dimension (input-gradient
        // N, weight-gradient M) and a reduction width.
        assert_eq!(NP, N.next_multiple_of(TC_TILE));
        assert!(VP >= VOCAB);
        assert_eq!(VP % TC_TILE, 0);
        assert_eq!(D % TC_TILE, 0);
        Ok(Self {
            embedding: GpuEmbedding::from_cpu(stream, &model.embedding)?,
            attention_norm: GpuRmsNorm::from_cpu(stream, &model.attention_norm)?,
            qkv_proj: GpuGroupedLinear::from_cpu(
                stream,
                [&model.q_proj, &model.k_proj, &model.v_proj],
            )?,
            o_proj: GpuLinear::from_cpu(stream, &model.o_proj)?,
            ffn_norm: GpuRmsNorm::from_cpu(stream, &model.ffn_norm)?,
            gate_up_proj: GpuGroupedLinear::from_cpu(stream, [&model.gate_proj, &model.up_proj])?,
            down_proj: GpuLinear::from_cpu(stream, &model.down_proj)?,
            final_norm: GpuRmsNorm::from_cpu(stream, &model.final_norm)?,
            lm_head: GpuBf16Head::from_cpu(stream, &model.lm_head)?,
        })
    }

    pub(crate) fn sync_linear_compute(
        &mut self,
        stream: &CudaStream,
        kernels: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        self.qkv_proj.sync_compute(stream, kernels)?;
        self.o_proj.sync_compute(stream, kernels)?;
        self.gate_up_proj.sync_compute(stream, kernels)?;
        self.down_proj.sync_compute(stream, kernels)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: &[usize; N],
        targets: &[usize; N],
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.forward_profiled(
            tokens,
            targets,
            workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            llama,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_profiled<P: KernelProfiler>(
        &self,
        tokens: &[usize; N],
        targets: &[usize; N],
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        workspace.upload_inputs(tokens, targets, stream)?;
        self.embedding.forward_into(
            &workspace.tokens,
            &mut workspace.attention_input,
            stream,
            llama,
            profiler,
            "forward.embedding",
        )?;
        self.attention_norm.forward_into(
            &workspace.attention_input,
            &mut workspace.attention_normalized,
            stream,
            llama,
            profiler,
            "forward.attention_norm",
        )?;
        self.qkv_proj.forward_into(
            &workspace.attention_normalized,
            &mut workspace.qkv,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            "forward.qkv_proj.gemm",
        )?;
        profiler.measure(stream, "forward.qkv_proj.split", || {
            llama.split_group3(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                workspace.qkv.as_device_buffer(),
                D as u32,
                workspace.q.as_device_buffer_mut(),
                workspace.k.as_device_buffer_mut(),
                workspace.v.as_device_buffer_mut(),
            )
        })?;
        rope_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &mut workspace.d_model_0,
            false,
            stream,
            llama,
            profiler,
            "forward.q_rope",
        )?;
        std::mem::swap(&mut workspace.q, &mut workspace.d_model_0);
        rope_into::<N, T, D, H, HD, P>(
            &workspace.k,
            &mut workspace.d_model_0,
            false,
            stream,
            llama,
            profiler,
            "forward.k_rope",
        )?;
        std::mem::swap(&mut workspace.k, &mut workspace.d_model_0);
        flash_attention_forward_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &mut workspace.attended,
            &mut workspace.attention_logsumexp,
            stream,
            flash,
            profiler,
        )?;
        self.o_proj.forward_into(
            &workspace.attended,
            &mut workspace.projection_output,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            "forward.o_proj.gemm",
        )?;
        add_into(
            &workspace.attention_input,
            &workspace.projection_output,
            &mut workspace.ffn_input,
            stream,
            tensor,
            profiler,
            "forward.attention_residual",
        )?;

        self.ffn_norm.forward_into(
            &workspace.ffn_input,
            &mut workspace.ffn_normalized,
            stream,
            llama,
            profiler,
            "forward.ffn_norm",
        )?;
        self.gate_up_proj.forward_into(
            &workspace.ffn_normalized,
            &mut workspace.gate_up,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            "forward.gate_up_proj.gemm",
        )?;
        profiler.measure(stream, "forward.gate_up_proj.split", || {
            llama.split_group2(
                stream,
                LaunchConfig::for_num_elems((N * FF) as u32),
                workspace.gate_up.as_device_buffer(),
                FF as u32,
                workspace.gate.as_device_buffer_mut(),
                workspace.up.as_device_buffer_mut(),
            )
        })?;
        swiglu_into(
            &workspace.gate,
            &workspace.up,
            &mut workspace.activated,
            stream,
            llama,
            profiler,
            "forward.swiglu",
        )?;
        self.down_proj.forward_into(
            &workspace.activated,
            &mut workspace.projection_output,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            "forward.down_proj.gemm",
        )?;
        add_into(
            &workspace.ffn_input,
            &workspace.projection_output,
            &mut workspace.final_input,
            stream,
            tensor,
            profiler,
            "forward.ffn_residual",
        )?;

        self.final_norm.forward_into(
            &workspace.final_input,
            &mut workspace.final_normalized,
            stream,
            llama,
            profiler,
            "forward.final_norm",
        )?;
        // Rows N..NP of head_input were zeroed at allocation and the convert
        // stops at the fp32 input's length, so they stay zero.
        profiler.measure(stream, "forward.lm_head.quantize", || {
            tensor.convert_f32_to_bf16_pairs(
                stream,
                pairs_config(N * D / 2),
                workspace.final_normalized.as_device_buffer(),
                &mut workspace.head_input,
            )
        })?;
        self.lm_head.forward_into::<NP, P>(
            &workspace.head_input_tma,
            &mut workspace.logits,
            stream,
            gemm_bf16,
            profiler,
            "forward.lm_head.gemm",
        )?;
        cross_entropy_into::<N, VOCAB, VP, P>(
            &workspace.logits,
            &workspace.targets,
            &mut workspace.losses,
            &mut workspace.loss_sum,
            &mut workspace.loss,
            stream,
            tensor,
            llama,
            profiler,
        )
    }

    pub fn backward(
        &mut self,
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.backward_profiled(
            workspace,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            flash,
            llama,
            &mut profiler,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_profiled<P: KernelProfiler>(
        &mut self,
        workspace: &mut GpuLlamaWorkspace<N, NP, T, VOCAB, VP, D, H, FF>,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        gemm: &gemm_kernels::LoadedModule,
        gemm_bf16: &Tcgen05Gemm,
        flash: &flash_kernels::LoadedModule,
        llama: &llama_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        cross_entropy_backward_into::<N, VOCAB, VP, P>(
            &workspace.targets,
            &mut workspace.logits,
            stream,
            llama,
            profiler,
        )?;
        // Rows N..NP of logits hold zeros (forward computed them from the
        // zero-padded head input and the classifier backward skips them), so
        // the transposed operands feed exact zeros into the weight GEMM's
        // padded reduction slice.
        profiler.measure(stream, "backward.lm_head.transpose_input", || unsafe {
            tensor.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(NP, D),
                &workspace.head_input,
                NP as u32,
                D as u32,
                &mut workspace.head_input_t,
            )
        })?;
        profiler.measure(stream, "backward.lm_head.transpose_dlogits", || unsafe {
            tensor.transpose_bf16_pairs(
                stream,
                transpose_pairs_config(NP, VP),
                &workspace.logits,
                NP as u32,
                VP as u32,
                &mut workspace.dlogits_t,
            )
        })?;
        self.lm_head.backward_weight::<NP, P>(
            &workspace.head_input_t_tma,
            &workspace.dlogits_t_tma,
            stream,
            gemm_bf16,
            profiler,
            "backward.lm_head.weight_gemm",
        )?;
        self.lm_head.backward_input::<NP, P>(
            &workspace.logits_tma,
            &mut workspace.d_head_input,
            stream,
            gemm_bf16,
            profiler,
            "backward.lm_head.input_gemm",
        )?;
        profiler.measure(stream, "backward.lm_head.dequantize", || {
            tensor.convert_bf16_pairs_to_f32(
                stream,
                elementwise_config::<Rank2<N, D>>(),
                &workspace.d_head_input,
                workspace.d_model_0.as_device_buffer_mut(),
            )
        })?;
        self.final_norm.backward_into(
            &workspace.final_input,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
            &mut workspace.norm_backward_inv,
            stream,
            llama,
            profiler,
            ["backward.final_norm.input", "backward.final_norm.weight"],
        )?;

        self.down_proj.backward_into(
            &workspace.activated,
            &workspace.d_model_1,
            &mut workspace.d_ff_0,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            [
                "backward.down_proj.weight_gemm",
                "backward.down_proj.input_gemm",
            ],
        )?;
        swiglu_backward_into(
            &workspace.gate,
            &workspace.up,
            &workspace.d_ff_0,
            &mut workspace.d_ff_1,
            &mut workspace.d_ff_2,
            stream,
            llama,
            profiler,
        )?;
        profiler.measure(stream, "backward.gate_up_proj.join", || unsafe {
            llama.join_group2(
                stream,
                LaunchConfig::for_num_elems((N * FF) as u32),
                workspace.d_ff_1.as_device_buffer(),
                workspace.d_ff_2.as_device_buffer(),
                FF as u32,
                workspace.gate_up.as_device_buffer_mut(),
            )
        })?;
        self.gate_up_proj.backward_into(
            &workspace.ffn_normalized,
            &workspace.gate_up,
            &mut workspace.d_model_3,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            [
                "backward.gate_up_proj.weight_gemm",
                "backward.gate_up_proj.input_gemm",
            ],
        )?;
        self.ffn_norm.backward_into(
            &workspace.ffn_input,
            &workspace.d_model_3,
            &mut workspace.d_model_0,
            &mut workspace.norm_backward_inv,
            stream,
            llama,
            profiler,
            ["backward.ffn_norm.input", "backward.ffn_norm.weight"],
        )?;
        add_into(
            &workspace.d_model_1,
            &workspace.d_model_0,
            &mut workspace.d_model_2,
            stream,
            tensor,
            profiler,
            "backward.ffn_residual",
        )?;

        self.o_proj.backward_into(
            &workspace.attended,
            &workspace.d_model_2,
            &mut workspace.d_model_0,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            ["backward.o_proj.weight_gemm", "backward.o_proj.input_gemm"],
        )?;
        flash_attention_backward_into::<N, T, D, H, HD, P>(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.attended,
            &workspace.attention_logsumexp,
            &mut workspace.attention_dot,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
            &mut workspace.d_model_3,
            &mut workspace.d_model_4,
            stream,
            flash,
            profiler,
        )?;
        rope_into::<N, T, D, H, HD, P>(
            &workspace.d_model_1,
            &mut workspace.d_model_0,
            true,
            stream,
            llama,
            profiler,
            "backward.q_rope",
        )?;
        rope_into::<N, T, D, H, HD, P>(
            &workspace.d_model_3,
            &mut workspace.d_model_1,
            true,
            stream,
            llama,
            profiler,
            "backward.k_rope",
        )?;
        profiler.measure(stream, "backward.qkv_proj.join", || unsafe {
            llama.join_group3(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                workspace.d_model_0.as_device_buffer(),
                workspace.d_model_1.as_device_buffer(),
                workspace.d_model_4.as_device_buffer(),
                D as u32,
                workspace.qkv.as_device_buffer_mut(),
            )
        })?;
        self.qkv_proj.backward_into(
            &workspace.attention_normalized,
            &workspace.qkv,
            &mut workspace.d_model_3,
            stream,
            tensor,
            gemm,
            gemm_bf16,
            workspace.linear_scratch.as_mut(),
            profiler,
            [
                "backward.qkv_proj.weight_gemm",
                "backward.qkv_proj.input_gemm",
            ],
        )?;
        self.attention_norm.backward_into(
            &workspace.attention_input,
            &workspace.d_model_3,
            &mut workspace.d_model_0,
            &mut workspace.norm_backward_inv,
            stream,
            llama,
            profiler,
            [
                "backward.attention_norm.input",
                "backward.attention_norm.weight",
            ],
        )?;
        add_into(
            &workspace.d_model_2,
            &workspace.d_model_0,
            &mut workspace.d_model_1,
            stream,
            tensor,
            profiler,
            "backward.attention_residual",
        )?;
        self.embedding.backward(
            &workspace.tokens,
            &workspace.d_model_1,
            stream,
            llama,
            profiler,
            "backward.embedding",
        )
    }

    pub fn zero_grad(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        let mut profiler = NoopProfiler;
        self.zero_grad_profiled(stream, tensor, &mut profiler)
    }

    pub fn zero_grad_profiled<P: KernelProfiler>(
        &mut self,
        stream: &CudaStream,
        tensor: &tensor_kernels::LoadedModule,
        profiler: &mut P,
    ) -> Result<(), DriverError> {
        macro_rules! zero {
            ($field:ident) => {
                fill_zero(
                    &mut self.$field.dw,
                    stream,
                    tensor,
                    profiler,
                    concat!("zero_grad.", stringify!($field)),
                )?;
            };
        }
        zero!(embedding);
        zero!(attention_norm);
        zero!(qkv_proj);
        zero!(o_proj);
        zero!(ffn_norm);
        zero!(gate_up_proj);
        zero!(down_proj);
        zero!(final_norm);
        self.lm_head
            .zero_grad(stream, tensor, profiler, "zero_grad.lm_head")?;
        Ok(())
    }
}

fn rope_into<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    x: &GpuTensor<f32, Rank2<N, D>>,
    y: &mut GpuTensor<f32, Rank2<N, D>>,
    backward: bool,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    if backward {
        profiler.measure(stream, name, || {
            kernels.rope_backward(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                x.as_device_buffer(),
                T as u32,
                H as u32,
                HD as u32,
                y.as_device_buffer_mut(),
            )
        })?;
    } else {
        profiler.measure(stream, name, || {
            kernels.rope_forward(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                x.as_device_buffer(),
                T as u32,
                H as u32,
                HD as u32,
                y.as_device_buffer_mut(),
            )
        })?;
    }
    Ok(())
}

fn flash_attention_forward_into<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    q: &GpuTensor<f32, Rank2<N, D>>,
    k: &GpuTensor<f32, Rank2<N, D>>,
    v: &GpuTensor<f32, Rank2<N, D>>,
    output: &mut GpuTensor<f32, Rank2<N, D>>,
    logsumexp: &mut GpuTensor<f32, Rank2<N, H>>,
    stream: &CudaStream,
    kernels: &flash_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    profiler.measure(stream, "forward.attention.flash", || {
        kernels.flash_attention_forward_tiled(
            stream,
            flash_forward_config::<N, T, H, HD>(),
            q.as_device_buffer(),
            k.as_device_buffer(),
            v.as_device_buffer(),
            T as u32,
            H as u32,
            output.as_device_buffer_mut(),
            logsumexp.as_device_buffer_mut(),
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn flash_attention_backward_into<
    const N: usize,
    const T: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    P: KernelProfiler,
>(
    q: &GpuTensor<f32, Rank2<N, D>>,
    k: &GpuTensor<f32, Rank2<N, D>>,
    v: &GpuTensor<f32, Rank2<N, D>>,
    output: &GpuTensor<f32, Rank2<N, D>>,
    logsumexp: &GpuTensor<f32, Rank2<N, H>>,
    softmax_dot: &mut GpuTensor<f32, Rank2<N, H>>,
    dy: &GpuTensor<f32, Rank2<N, D>>,
    dq: &mut GpuTensor<f32, Rank2<N, D>>,
    dk: &mut GpuTensor<f32, Rank2<N, D>>,
    dv: &mut GpuTensor<f32, Rank2<N, D>>,
    stream: &CudaStream,
    kernels: &flash_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    profiler.measure(stream, "backward.attention.flash_dot", || {
        kernels.flash_attention_backward_dot(
            stream,
            flash_dot_config::<N, H, HD>(),
            dy.as_device_buffer(),
            output.as_device_buffer(),
            HD as u32,
            softmax_dot.as_device_buffer_mut(),
        )
    })?;
    profiler.measure(stream, "backward.attention.flash_q", || {
        kernels.flash_attention_backward_q_tiled(
            stream,
            flash_backward_q_config::<N, T, H, HD>(),
            q.as_device_buffer(),
            k.as_device_buffer(),
            v.as_device_buffer(),
            dy.as_device_buffer(),
            logsumexp.as_device_buffer(),
            softmax_dot.as_device_buffer(),
            T as u32,
            H as u32,
            dq.as_device_buffer_mut(),
        )
    })?;
    profiler.measure(stream, "backward.attention.flash_kv", || {
        kernels.flash_attention_backward_kv_tiled(
            stream,
            flash_backward_kv_config::<N, T, H, HD>(),
            q.as_device_buffer(),
            k.as_device_buffer(),
            v.as_device_buffer(),
            dy.as_device_buffer(),
            logsumexp.as_device_buffer(),
            softmax_dot.as_device_buffer(),
            T as u32,
            H as u32,
            dk.as_device_buffer_mut(),
            dv.as_device_buffer_mut(),
        )
    })?;
    Ok(())
}

fn swiglu_into<const N: usize, const FF: usize, P: KernelProfiler>(
    gate: &GpuTensor<f32, Rank2<N, FF>>,
    up: &GpuTensor<f32, Rank2<N, FF>>,
    output: &mut GpuTensor<f32, Rank2<N, FF>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
    name: &'static str,
) -> Result<(), DriverError> {
    profiler.measure(stream, name, || {
        kernels.swiglu_forward(
            stream,
            LaunchConfig::for_num_elems((N * FF) as u32),
            gate.as_device_buffer(),
            up.as_device_buffer(),
            output.as_device_buffer_mut(),
        )
    })?;
    Ok(())
}

fn swiglu_backward_into<const N: usize, const FF: usize, P: KernelProfiler>(
    gate: &GpuTensor<f32, Rank2<N, FF>>,
    up: &GpuTensor<f32, Rank2<N, FF>>,
    dy: &GpuTensor<f32, Rank2<N, FF>>,
    dgate: &mut GpuTensor<f32, Rank2<N, FF>>,
    dup: &mut GpuTensor<f32, Rank2<N, FF>>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    let config = LaunchConfig::for_num_elems((N * FF) as u32);
    profiler.measure(stream, "backward.swiglu.gate", || {
        kernels.swiglu_backward_gate(
            stream,
            config,
            gate.as_device_buffer(),
            up.as_device_buffer(),
            dy.as_device_buffer(),
            dgate.as_device_buffer_mut(),
        )
    })?;
    profiler.measure(stream, "backward.swiglu.up", || {
        kernels.swiglu_backward_up(
            stream,
            config,
            gate.as_device_buffer(),
            dy.as_device_buffer(),
            dup.as_device_buffer_mut(),
        )
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cross_entropy_into<const N: usize, const VOCAB: usize, const VP: usize, P: KernelProfiler>(
    logits: &DeviceBuffer<u32>,
    targets: &GpuTensor<u32, Rank1<N>>,
    losses: &mut GpuTensor<f32, Rank1<N>>,
    loss_sum: &mut GpuTensor<f32, Rank1<1>>,
    loss: &mut GpuTensor<f32, Rank1<1>>,
    stream: &CudaStream,
    tensor: &tensor_kernels::LoadedModule,
    llama: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    profiler.measure(stream, "forward.loss.fused_classifier", || {
        llama.fused_classifier_forward_bf16(
            stream,
            classifier_config::<N>(),
            logits,
            targets.as_device_buffer(),
            N as u32,
            VOCAB as u32,
            VP as u32,
            losses.as_device_buffer_mut(),
        )
    })?;
    sum_into(
        losses,
        loss_sum,
        stream,
        tensor,
        profiler,
        "forward.loss.reduction",
    )?;
    scale_into(
        loss_sum,
        1.0 / N as f32,
        loss,
        stream,
        tensor,
        profiler,
        "forward.loss.mean",
    )
}

fn cross_entropy_backward_into<
    const N: usize,
    const VOCAB: usize,
    const VP: usize,
    P: KernelProfiler,
>(
    targets: &GpuTensor<u32, Rank1<N>>,
    dlogits: &mut DeviceBuffer<u32>,
    stream: &CudaStream,
    kernels: &llama_kernels::LoadedModule,
    profiler: &mut P,
) -> Result<(), DriverError> {
    profiler.measure(stream, "backward.loss.fused_classifier", || {
        kernels.fused_classifier_backward_in_place_bf16(
            stream,
            classifier_config::<N>(),
            targets.as_device_buffer(),
            1.0,
            N as u32,
            VOCAB as u32,
            VP as u32,
            dlogits,
        )
    })
}
