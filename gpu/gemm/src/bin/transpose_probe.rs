//! Pure-PTX artifact binary and self-checking harness for the tcgen05
//! descriptor-transpose probe (issue #53).
//!
//! Run on B200 with `./run.sh gemm transpose_probe`.
//!
//! # What is being pinned down
//!
//! The weight-gradient GEMMs stage `dW += dYᵀ·X` by materializing both
//! transposes in global memory. Eliminating them needs the instruction
//! descriptor's `transpose_a`/`transpose_b` bits, which make the MMA read an
//! operand MN-major (`[K, MN]` row-major, MN contiguous) instead of the
//! default K-major (`[MN, K]` row-major, K contiguous).
//!
//! `transpose_b` already runs in production (flash-attn's `O = P·V` and the
//! gradient MMAs read a `[K, N]` V/K panel with LBO 16 / SBO 1024 / 128B
//! swizzle and a 2048-byte step per `K = 16` chunk). `transpose_a` has never
//! run here, and it collides with a geometry constraint: `SWIZZLE_128B` caps a
//! TMA box at 128 bytes, so an MN-major tile holds only 64 MN per row, while
//! the `M64` MMA shape is a proven dead end (SPEC 7e15: the SS operand read
//! broadcasts A-rows across subpartition pairs). An `M128` transposed A
//! therefore needs 128 M elements along the contiguous axis — a 256-byte row
//! pitch, which `SWIZZLE_128B` cannot describe.
//!
//! So the probe sweeps A geometries rather than guessing one:
//!
//! | layout                | tile                                   | LBO  | SBO  | swizzle | chunk step |
//! |-----------------------|----------------------------------------|------|------|---------|------------|
//! | `a_k_major`           | `[128 M, 64 K]`, 128B rows             | 16   | 1024 | 128B    | 32         |
//! | `a_mn_unswizzled`     | `[64 K, 128 M]`, 256B rows, no swizzle | 16   | 2048 | none    | 4096       |
//! | `a_mn_split`          | two stacked `[64 K, 64 M]` subtiles    | 8192 | 1024 | 128B    | 2048       |
//! | `a_mn_m64`            | one `[64 K, 64 M]` subtile, `M64_N64`  | 16   | 1024 | 128B    | 2048       |
//! | `a_k_major_m64`       | `[128 M, 64 K]`, `M64_N64`             | 16   | 1024 | 128B    | 32         |
//!
//! `a_k_major` is the control (today's proven path). `a_mn_unswizzled` is the
//! "LBO is always the 16-byte step along the contiguous axis, SBO is always
//! eight rows of pitch" reading, which is the only reading consistent with
//! both the K-major GEMM and flash-attn's transposed B. `a_mn_split` tests the
//! competing reading, that under 128B swizzle LBO instead jumps whole 64-wide
//! subtiles (which would keep a transposed A fully swizzled). `a_mn_m64` and
//! `a_k_major_m64` bracket whether the `M64` dead end is a property of the
//! shape or of the K-major operand read.
//!
//! Each geometry runs against both a K-major B and an MN-major B, and against
//! four operand encodings, because #47 showed that operands uniform along `K`
//! hide K-permutation bugs:
//!
//! - `random`   — the real correctness gate, varies along every axis.
//! - `outer`    — `A[m,0] = m+1`, `B[n,0] = n+1`, zero elsewhere, so
//!   `C[m,n] = (m+1)(n+1)`: an exact-integer map of the M and N axes.
//! - `a_walk`   — `A[m,k] = δ(k, m mod 64)`, `B[n,k] = k+1`, so
//!   `C[m,n] = (m mod 64)+1`: row `m` of the accumulator reads out *which* `K`
//!   index the A operand paired with, exactly the A/B K-mispairing #47 hit.
//! - `b_walk`   — mirror image: `A[m,k] = k+1`, `B[n,k] = δ(k, n)`, so
//!   `C[m,n] = n+1` reads out B's `K` indexing per output column.
//!
//! All three structured encodings are exact in bf16 (integers ≤ 128) and in
//! fp32 accumulation, so a mismatch is unambiguous rather than a tolerance
//! judgement, and a failure prints the observed readout.
//!
//! # What it found
//!
//! `a_mn_split` wins on every encoding, alone and with a transposed B: under
//! 128B swizzle the LBO of an MN-major operand *is* a subtile jump, so a
//! transposed `M128` A stays fully swizzled as two stacked 64-wide subtiles.
//! `a_mn_unswizzled` fails, and both `M64` cases fail identically, which puts
//! SPEC 7e15's `M64` dead end on the shape rather than the operand layout.
//! `optimized.rs`'s `transposed` path is that geometry; this binary stays as
//! the isolated regression gate for it.

use std::error::Error;
use std::io::Write;
use std::mem::MaybeUninit;

use bench_util::uniform_vec;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
};
use half::bf16;

#[path = "../transpose_probe.rs"]
mod device;

use device::{PROBE_K, PROBE_M, PROBE_N, kernels};

/// Row pitch in bytes of a `SWIZZLE_128B` subtile.
const SWIZZLED_PITCH: usize = 128;
/// MN elements in one such subtile (bf16).
const SUBTILE_MN: usize = SWIZZLED_PITCH / 2;
/// Descriptor swizzle field: 2 = `SWIZZLE_128B`, 0 = none.
const SWIZZLE_128B: u8 = 2;
const SWIZZLE_NONE: u8 = 0;

// ---------------------------------------------------------------------------
// Operand encodings
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Encoding {
    Random,
    Outer,
    AWalk,
    BWalk,
}

const ENCODINGS: [Encoding; 4] = [
    Encoding::Random,
    Encoding::Outer,
    Encoding::AWalk,
    Encoding::BWalk,
];

impl Encoding {
    fn name(self) -> &'static str {
        match self {
            Encoding::Random => "random",
            Encoding::Outer => "outer",
            Encoding::AWalk => "a_walk",
            Encoding::BWalk => "b_walk",
        }
    }

    /// Exact-integer encodings admit no rounding slack at all; the random one
    /// carries the usual bf16 GEMM tolerance.
    fn tolerance(self) -> (f32, f32) {
        match self {
            Encoding::Random => (0.02, 0.01),
            _ => (1e-3, 0.0),
        }
    }

    /// Logical `A[m, k]` and `B[n, k]`, before bf16 rounding.
    fn operands(self) -> (Vec<f32>, Vec<f32>) {
        if self == Encoding::Random {
            return (
                uniform_vec(PROBE_M * PROBE_K, 71),
                uniform_vec(PROBE_N * PROBE_K, 72),
            );
        }
        let mut a = vec![0.0f32; PROBE_M * PROBE_K];
        let mut b = vec![0.0f32; PROBE_N * PROBE_K];
        match self {
            Encoding::Random => unreachable!(),
            Encoding::Outer => {
                for m in 0..PROBE_M {
                    a[m * PROBE_K] = (m + 1) as f32;
                }
                for n in 0..PROBE_N {
                    b[n * PROBE_K] = (n + 1) as f32;
                }
            }
            Encoding::AWalk => {
                for m in 0..PROBE_M {
                    a[m * PROBE_K + m % PROBE_K] = 1.0;
                }
                for n in 0..PROBE_N {
                    for k in 0..PROBE_K {
                        b[n * PROBE_K + k] = (k + 1) as f32;
                    }
                }
            }
            Encoding::BWalk => {
                for m in 0..PROBE_M {
                    for k in 0..PROBE_K {
                        a[m * PROBE_K + k] = (k + 1) as f32;
                    }
                }
                for n in 0..PROBE_N {
                    b[n * PROBE_K + n] = 1.0;
                }
            }
        }
        (a, b)
    }
}

/// One encoding's operands in every storage order the sweep needs, plus the
/// oracle computed from the bf16-rounded values.
struct Operands {
    a_k_major: Vec<u16>,
    a_mn_major: Vec<u16>,
    b_k_major: Vec<u16>,
    b_mn_major: Vec<u16>,
    expected: Vec<f32>,
}

fn round_bf16(values: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = values
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect();
    let rounded = bits
        .iter()
        .map(|&bits| bf16::from_bits(bits).to_f32())
        .collect();
    (bits, rounded)
}

fn transpose(source: &[u16], rows: usize, columns: usize) -> Vec<u16> {
    let mut output = vec![0u16; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            output[column * rows + row] = source[row * columns + column];
        }
    }
    output
}

impl Operands {
    fn build(encoding: Encoding) -> Self {
        let (a, b) = encoding.operands();
        let (a_bits, a_values) = round_bf16(&a);
        let (b_bits, b_values) = round_bf16(&b);
        let mut expected = vec![0.0f32; PROBE_M * PROBE_N];
        for m in 0..PROBE_M {
            for n in 0..PROBE_N {
                let mut sum = 0.0f64;
                for k in 0..PROBE_K {
                    sum += a_values[m * PROBE_K + k] as f64 * b_values[n * PROBE_K + k] as f64;
                }
                expected[m * PROBE_N + n] = sum as f32;
            }
        }
        Self {
            a_mn_major: transpose(&a_bits, PROBE_M, PROBE_K),
            a_k_major: a_bits,
            b_mn_major: transpose(&b_bits, PROBE_N, PROBE_K),
            b_k_major: b_bits,
            expected,
        }
    }
}

// ---------------------------------------------------------------------------
// Operand geometry
// ---------------------------------------------------------------------------

/// Which staged matrix a layout reads, and with which TMA box/swizzle.
#[derive(Clone, Copy, PartialEq)]
enum Source {
    /// `[128 M, 64 K]` row-major, box `64 K x 128 M`, 128B swizzle.
    AKMajor,
    /// `[64 K, 128 M]` row-major, box `128 M x 64 K`, unswizzled (256-byte
    /// shared rows — the only way to put 128 M on the contiguous axis).
    AMnWide,
    /// `[64 K, 128 M]` row-major, box `64 M x 64 K`, 128B swizzle; one load
    /// per 64-M subtile.
    AMnSubtile,
    /// `[64 N, 64 K]` row-major, box `64 K x 64 N`, 128B swizzle.
    BKMajor,
    /// `[64 K, 64 N]` row-major, box `64 N x 64 K`, 128B swizzle.
    BMnMajor,
}

/// A complete operand plan: where the tile comes from and how the MMA walks it.
#[derive(Clone, Copy)]
struct Layout {
    name: &'static str,
    source: Source,
    transposed: bool,
    tiles: u32,
    tile_bytes: u32,
    chunk_step: u32,
    leading_bytes: u32,
    stride_bytes: u32,
    swizzle: u8,
}

impl Layout {
    /// The smem descriptor's constant bits; the kernel ORs in the address.
    fn word(&self) -> u64 {
        ((((self.leading_bytes >> 4) & 0x3fff) as u64) << 16)
            | ((((self.stride_bytes >> 4) & 0x3fff) as u64) << 32)
            | (1u64 << 46)
            | ((self.swizzle as u64) << 61)
    }
}

const A_TILE_BYTES: u32 = (PROBE_M * PROBE_K * 2) as u32;
const A_SUBTILE_BYTES: u32 = A_TILE_BYTES / 2;
const B_TILE_BYTES: u32 = (PROBE_N * PROBE_K * 2) as u32;
/// Eight rows of a 128-byte-pitch tile: the SBO of every swizzled subtile.
const SBO_128B: u32 = (8 * SWIZZLED_PITCH) as u32;
/// Eight rows of the 256-byte-pitch unswizzled A tile.
const SBO_256B: u32 = 2 * SBO_128B;

/// `(layout, instruction M)`.
const A_LAYOUTS: [(Layout, usize); 5] = [
    (
        Layout {
            name: "a_k_major",
            source: Source::AKMajor,
            transposed: false,
            tiles: 1,
            tile_bytes: A_TILE_BYTES,
            // K = 16 bf16 elements along a row.
            chunk_step: 32,
            leading_bytes: 16,
            stride_bytes: SBO_128B,
            swizzle: SWIZZLE_128B,
        },
        128,
    ),
    (
        Layout {
            name: "a_mn_unswizzled",
            source: Source::AMnWide,
            transposed: true,
            tiles: 1,
            tile_bytes: A_TILE_BYTES,
            // K = 16 rows of a 256-byte-pitch tile.
            chunk_step: 16 * 2 * SWIZZLED_PITCH as u32,
            leading_bytes: 16,
            stride_bytes: SBO_256B,
            swizzle: SWIZZLE_NONE,
        },
        128,
    ),
    (
        Layout {
            name: "a_mn_split",
            source: Source::AMnSubtile,
            transposed: true,
            tiles: 2,
            tile_bytes: A_SUBTILE_BYTES,
            // K = 16 rows of a 128-byte-pitch subtile.
            chunk_step: 16 * SWIZZLED_PITCH as u32,
            // Hypothesis: under 128B swizzle a transposed operand's LBO jumps
            // whole 64-MN subtiles rather than 8-MN core matrices.
            leading_bytes: A_SUBTILE_BYTES,
            stride_bytes: SBO_128B,
            swizzle: SWIZZLE_128B,
        },
        128,
    ),
    (
        Layout {
            name: "a_mn_m64",
            source: Source::AMnSubtile,
            transposed: true,
            tiles: 1,
            tile_bytes: A_SUBTILE_BYTES,
            chunk_step: 16 * SWIZZLED_PITCH as u32,
            leading_bytes: 16,
            stride_bytes: SBO_128B,
            swizzle: SWIZZLE_128B,
        },
        64,
    ),
    (
        Layout {
            name: "a_k_major_m64",
            source: Source::AKMajor,
            transposed: false,
            tiles: 1,
            tile_bytes: A_TILE_BYTES,
            chunk_step: 32,
            leading_bytes: 16,
            stride_bytes: SBO_128B,
            swizzle: SWIZZLE_128B,
        },
        64,
    ),
];

const B_LAYOUTS: [Layout; 2] = [
    Layout {
        name: "b_k_major",
        source: Source::BKMajor,
        transposed: false,
        tiles: 1,
        tile_bytes: B_TILE_BYTES,
        chunk_step: 32,
        leading_bytes: 16,
        stride_bytes: SBO_128B,
        swizzle: SWIZZLE_128B,
    },
    Layout {
        name: "b_mn_major",
        source: Source::BMnMajor,
        transposed: true,
        tiles: 1,
        tile_bytes: B_TILE_BYTES,
        chunk_step: 16 * SWIZZLED_PITCH as u32,
        leading_bytes: 16,
        stride_bytes: SBO_128B,
        swizzle: SWIZZLE_128B,
    },
];

// ---------------------------------------------------------------------------
// TMA maps
// ---------------------------------------------------------------------------

/// Encode a bf16 tensor map over a row-major `[height, width]` matrix with an
/// explicit box and swizzle mode. `gemm::host`'s encoder is fixed at the
/// 64x128 / 128B-swizzle GEMM tile; the probe needs the box to follow the
/// operand's contiguous axis.
fn encode_tma_map(
    stream: &CudaStream,
    base: u64,
    width: usize,
    height: usize,
    box_width: usize,
    box_height: usize,
    swizzle: u32,
) -> Result<DeviceBuffer<u64>, Box<dyn Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dimensions = [width as u64, height as u64];
    let global_strides = [(width * 2) as u64];
    let box_dimensions = [box_width as u32, box_height as u32];
    let element_strides = [1u32, 1u32];
    let status = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            base as *mut std::ffi::c_void,
            global_dimensions.as_ptr(),
            global_strides.as_ptr(),
            box_dimensions.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if status != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled failed: {status:?}").into());
    }
    let tensor_map = unsafe { tensor_map.assume_init() };
    Ok(DeviceBuffer::from_host(stream, &tensor_map.opaque)?)
}

/// The staged operands and one TMA map per [`Source`], rebuilt per encoding.
struct Staged {
    maps: Vec<DeviceBuffer<u64>>,
    _matrices: Vec<DeviceBuffer<u16>>,
    expected: Vec<f32>,
}

impl Staged {
    fn build(stream: &CudaStream, encoding: Encoding) -> Result<Self, Box<dyn Error>> {
        use cuda_core::sys::{
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
        };
        let operands = Operands::build(encoding);
        let a_k = DeviceBuffer::from_host(stream, &operands.a_k_major)?;
        let a_mn = DeviceBuffer::from_host(stream, &operands.a_mn_major)?;
        let b_k = DeviceBuffer::from_host(stream, &operands.b_k_major)?;
        let b_mn = DeviceBuffer::from_host(stream, &operands.b_mn_major)?;
        let maps = vec![
            encode_tma_map(
                stream,
                a_k.cu_deviceptr(),
                PROBE_K,
                PROBE_M,
                PROBE_K,
                PROBE_M,
                CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            )?,
            encode_tma_map(
                stream,
                a_mn.cu_deviceptr(),
                PROBE_M,
                PROBE_K,
                PROBE_M,
                PROBE_K,
                CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
            )?,
            encode_tma_map(
                stream,
                a_mn.cu_deviceptr(),
                PROBE_M,
                PROBE_K,
                SUBTILE_MN,
                PROBE_K,
                CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            )?,
            encode_tma_map(
                stream,
                b_k.cu_deviceptr(),
                PROBE_K,
                PROBE_N,
                PROBE_K,
                PROBE_N,
                CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            )?,
            encode_tma_map(
                stream,
                b_mn.cu_deviceptr(),
                PROBE_N,
                PROBE_K,
                PROBE_N,
                PROBE_K,
                CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            )?,
        ];
        Ok(Self {
            maps,
            _matrices: vec![a_k, a_mn, b_k, b_mn],
            expected: operands.expected,
        })
    }

    fn map(&self, source: Source) -> *const cuda_device::tma::TmaDescriptor {
        let index = match source {
            Source::AKMajor => 0,
            Source::AMnWide => 1,
            Source::AMnSubtile => 2,
            Source::BKMajor => 3,
            Source::BMnMajor => 4,
        };
        self.maps[index].cu_deviceptr() as *const cuda_device::tma::TmaDescriptor
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn instruction(a: &Layout, b: &Layout, m: usize) -> u32 {
    let shape = if m == 64 {
        Tcgen05MmaShape::M64_N64
    } else {
        Tcgen05MmaShape::M128_N64
    };
    Tcgen05InstructionDescriptor::builder()
        .shape(shape)
        .element_type(Tcgen05ElementType::BF16)
        .accumulator_type(Tcgen05AccumulatorType::F32)
        .transpose_a(a.transposed)
        .transpose_b(b.transposed)
        .build()
        .raw()
}

/// Compare the drained rows and, on failure, print the readout the encoding
/// was designed to expose.
fn check(encoding: Encoding, rows: usize, actual: &[f32], expected: &[f32]) -> Result<f32, String> {
    let (atol, rtol) = encoding.tolerance();
    let mut max_abs = 0.0f32;
    let mut failures = Vec::new();
    for m in 0..rows {
        for n in 0..PROBE_N {
            let index = m * PROBE_N + n;
            let (got, want) = (actual[index], expected[index]);
            let error = (got - want).abs();
            max_abs = max_abs.max(error);
            if error > atol + rtol * want.abs() && failures.len() < 4 {
                failures.push(format!("[{m},{n}] gpu {got} cpu {want}"));
            }
        }
    }
    if failures.is_empty() {
        return Ok(max_abs);
    }
    let readout = match encoding {
        // C[m,0] should be (m mod 64)+1: which B row (i.e. which K index) the
        // A operand's row m actually paired with.
        Encoding::AWalk => {
            let observed: Vec<i32> = (0..rows.min(16))
                .map(|m| actual[m * PROBE_N] as i32)
                .collect();
            format!("; a_walk C[0..16,0] = {observed:?} (want 1..16)")
        }
        // C[0,n] should be n+1.
        Encoding::BWalk => {
            let observed: Vec<i32> = (0..16).map(|n| actual[n] as i32).collect();
            format!("; b_walk C[0,0..16] = {observed:?} (want 1..16)")
        }
        _ => String::new(),
    };
    Err(format!(
        "max abs {max_abs:.3e}; {}{readout}",
        failures.join(", ")
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::from_module(context.load_module_from_file("transpose_probe.ptx")?)?;
    let math_input: Vec<f32> = (1..=256).map(|i| i as f32 / 64.0).collect();
    let math_expected: Vec<f32> = math_input
        .iter()
        .map(|&value| value.sqrt() + value.exp() + value.ln() + value.max(0.5))
        .collect();
    let math_input = DeviceBuffer::from_host(&stream, &math_input)?;
    let mut math_output = DeviceBuffer::<f32>::zeroed(&stream, math_expected.len())?;
    unsafe {
        module.libdevice_math_probe(
            &stream,
            LaunchConfig::for_num_elems(math_expected.len() as u32),
            &math_input,
            &mut math_output,
        )
    }?;
    let math_actual = math_output.to_host_vec(&stream)?;
    let max_math_error = math_actual
        .iter()
        .zip(&math_expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if max_math_error > 1e-5 {
        return Err(format!(
            "libdevice math failed in the tcgen05 artifact: max abs error {max_math_error:.3e}"
        )
        .into());
    }
    println!(
        "✓ libdevice math shares the pure-PTX tcgen05 artifact (max abs error {max_math_error:.3e})"
    );

    let mut atomic_output = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    unsafe {
        module.device_atomic_probe(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut atomic_output,
        )
    }?;
    let atomic_sum = atomic_output.to_host_vec(&stream)?[0];
    if atomic_sum != 2080.0 {
        return Err(format!(
            "device atomics failed in the tcgen05 artifact: got {atomic_sum}, expected 2080"
        )
        .into());
    }
    println!("✓ device atomics share the pure-PTX tcgen05 artifact (sum 2080/2080)\n");

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    let cases: Vec<(usize, usize)> = (0..A_LAYOUTS.len())
        .flat_map(|a| (0..B_LAYOUTS.len()).map(move |b| (a, b)))
        .collect();
    let mut results = vec![vec![String::new(); ENCODINGS.len()]; cases.len()];

    println!(
        "tcgen05 descriptor-transpose probe: C[{PROBE_M},{PROBE_N}] = A·Bᵀ over K={PROBE_K}\n"
    );
    for (encoding_index, &encoding) in ENCODINGS.iter().enumerate() {
        let staged = Staged::build(&stream, encoding)?;
        for (case_index, &(a_index, b_index)) in cases.iter().enumerate() {
            let (a, rows) = A_LAYOUTS[a_index];
            let b = B_LAYOUTS[b_index];
            let mut output = DeviceBuffer::<f32>::zeroed(&stream, PROBE_M * PROBE_N)?;
            // Announce (and flush) before every launch: a geometry the
            // hardware rejects takes the whole context down, and this line is
            // what says which one.
            println!("  launching {} x {} [{}]", a.name, b.name, encoding.name());
            std::io::stdout().flush()?;
            unsafe {
                module.descriptor_transpose_probe(
                    &stream,
                    config,
                    staged.map(a.source),
                    staged.map(b.source),
                    a.word(),
                    b.word(),
                    a.tiles,
                    a.tile_bytes,
                    a.chunk_step,
                    b.tiles,
                    b.tile_bytes,
                    b.chunk_step,
                    instruction(&a, &b, rows),
                    &mut output,
                )
            }?;
            let actual = output.to_host_vec(&stream)?;
            results[case_index][encoding_index] =
                match check(encoding, rows, &actual, &staged.expected) {
                    Ok(max_abs) => format!("pass ({max_abs:.1e})"),
                    Err(detail) => {
                        println!("    FAIL {detail}");
                        "FAIL".to_string()
                    }
                };
        }
    }

    println!("\n{:<34}{}", "case", {
        let mut header = String::new();
        for encoding in ENCODINGS {
            header.push_str(&format!("{:<16}", encoding.name()));
        }
        header
    });
    for (case_index, &(a_index, b_index)) in cases.iter().enumerate() {
        let (a, rows) = A_LAYOUTS[a_index];
        let b = B_LAYOUTS[b_index];
        let label = format!("{} x {} (M{rows})", a.name, b.name);
        let mut row = String::new();
        for result in &results[case_index] {
            row.push_str(&format!("{result:<16}"));
        }
        println!("{label:<34}{row}");
    }

    let passed = |case_index: usize| results[case_index].iter().all(|r| r.starts_with("pass"));
    let index_of = |a_name: &str, b_name: &str| {
        cases
            .iter()
            .position(|&(a, b)| A_LAYOUTS[a].0.name == a_name && B_LAYOUTS[b].name == b_name)
            .expect("case in table")
    };

    let control = index_of("a_k_major", "b_k_major");
    if !passed(control) {
        return Err(
            "control case (both operands K-major) failed: the probe harness, \
                    not the transpose bits, is wrong"
                .into(),
        );
    }
    println!("\n✓ control: K-major x K-major matches the oracle on every encoding");

    let transpose_b_only = index_of("a_k_major", "b_mn_major");
    println!(
        "  transpose_b alone: {}",
        if passed(transpose_b_only) {
            "pass (matches flash-attn's O = P·V geometry)"
        } else {
            "FAIL"
        }
    );

    let mut winner = None;
    for (case_index, &(a_index, b_index)) in cases.iter().enumerate() {
        let (a, rows) = A_LAYOUTS[a_index];
        if a.transposed && B_LAYOUTS[b_index].transposed && passed(case_index) {
            winner = Some((a, rows));
            break;
        }
    }
    match winner {
        Some((a, rows)) => {
            println!(
                "\n✓ weight-gradient geometry: A layout `{}` (M{rows}) with an MN-major B\n  \
                 A: LBO {} SBO {} swizzle {} , {} TMA tile(s) of {} bytes, {} bytes per K=16 chunk",
                a.name,
                a.leading_bytes,
                a.stride_bytes,
                a.swizzle,
                a.tiles,
                a.tile_bytes,
                a.chunk_step
            );
            Ok(())
        }
        None => Err(
            "no transposed-A geometry reproduced the oracle; see the table above \
                     for which axis each candidate got wrong"
                .into(),
        ),
    }
}
