//! Where a replicating TMA multicast's transaction bytes are counted.
//!
//! Run on B200 with `./run.sh gemm mcast_probe`. The question, the three
//! variants and why the answer gates a 4-CTA cluster GEMM are in
//! `src/mcast_probe.rs`; this is the harness and the readout.
//!
//! The staged matrix is `[256, 64]` bf16 with `A[r][c] = r`, so the first and
//! last `u16` a destination reports name the row range it received — rows
//! `0..128` for the half rank 0 fetches, `128..256` for rank 1's. That is what
//! separates *delivery* from *accounting*: a rank holding the right rows with
//! `completed = 0` says the mask reached it and only the charge went somewhere
//! else, which is a different fix from a mask that did not reach it at all.

use std::error::Error;
use std::mem::MaybeUninit;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::tma::TmaDescriptor;
use half::bf16;

#[path = "../mcast_probe.rs"]
mod device;

use device::{FIELDS, RANKS, TILE_COLS, TILE_ROWS, VARIANTS, kernels};

/// Long enough that a barrier which is going to flip has, and short enough that
/// the whole table lands in one launch.
const DEADLINE_NS: u64 = 200_000_000;

/// What each variant is asking, for the readout.
const QUESTIONS: [&str; VARIANTS as usize] = [
    "rank 0 issues mask {0,2} on its OWN barrier  -> is the operand an offset applied per destination?",
    "rank 1 issues mask {1,3} on rank 0's mapa'd  -> does an even-rank address take a replicating copy?",
    "  ...the same, charged TWICE at rank 0       -> or is one barrier charged once per destination?",
    "  ...the same, rank 2 charged too            -> does each destination's copy reach ITS pair leader?",
    "both halves, both aimed at the even rank     -> the kernel's arrangement, end to end",
];

fn yes_no(ok: bool) -> &'static str {
    if ok { "yes" } else { "no" }
}

fn row_value(row: usize) -> u16 {
    bf16::from_f32(row as f32).to_bits()
}

/// A `SWIZZLE_128B` map over the `[height, TILE_COLS]` bf16 matrix, delivering
/// the GEMM's own K-major `A` box.
///
/// Encoded here rather than through `gemm::create_bf16_tma_map` for the reason
/// `transpose_probe` encodes its own: a probe binary carrying a
/// `#[cuda_module]` of its own stays off the library, so its device artifact
/// holds the kernel under test and nothing else.
fn encode_map(
    stream: &CudaStream,
    base: u64,
    height: usize,
) -> Result<DeviceBuffer<u64>, Box<dyn Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dimensions = [TILE_COLS as u64, height as u64];
    let global_strides = [(TILE_COLS * 2) as u64];
    let box_dimensions = [TILE_COLS as u32, TILE_ROWS as u32];
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
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
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

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::from_module(context.load_module_from_file("mcast_probe.ptx")?)?;

    let rows = TILE_ROWS * 2;
    let matrix: Vec<u16> = (0..rows)
        .flat_map(|r| std::iter::repeat_n(row_value(r), TILE_COLS))
        .collect();
    let matrix = DeviceBuffer::from_host(&stream, &matrix)?;
    let map = encode_map(&stream, matrix.cu_deviceptr(), rows)?;

    let ctas = VARIANTS * RANKS;
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, FIELDS * ctas as usize)?;
    unsafe {
        module.multicast_accounting_probe(
            &stream,
            LaunchConfig {
                grid_dim: (ctas, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            map.cu_deviceptr() as *const TmaDescriptor,
            DEADLINE_NS,
            &mut out,
        )
    }?;
    let report = out.to_host_vec(&stream)?;

    println!(
        "replicating TMA multicast: where are the transaction bytes counted?\n\
         staged tile [{TILE_ROWS}, {TILE_COLS}] bf16, A[r][c] = r, 4-CTA cluster, \
         {}-ms deadline\n",
        DEADLINE_NS / 1_000_000
    );
    let half = |first: u64, last: u64| -> String {
        for issuer in 0..2usize {
            let (lo, hi) = (issuer * TILE_ROWS, issuer * TILE_ROWS + TILE_ROWS - 1);
            if first == row_value(lo) as u64 && last == row_value(hi) as u64 {
                return format!("rows {lo}..{}", hi + 1);
            }
        }
        if first == 0 && last == 0 {
            "-".to_string()
        } else {
            format!("?? {first:#06x}/{last:#06x}")
        }
    };

    let mut verdicts = Vec::new();
    for variant in 0..VARIANTS {
        println!("variant {variant}: {}", QUESTIONS[variant as usize]);
        println!("  rank  entered  charged  completed  received");
        let mut completions = Vec::new();
        for rank in 0..RANKS {
            let base = FIELDS * (variant * RANKS + rank) as usize;
            let (entered, completed, first, last, charged) = (
                report[base],
                report[base + 1],
                report[base + 2],
                report[base + 3],
                report[base + 4],
            );
            let state = if charged == 0 {
                "  (none)"
            } else if completed == 1 {
                "      yes"
            } else {
                "   NO    "
            };
            println!(
                "  {rank:>4}  {entered:>7}  {charged:>7}  {state}  {}",
                half(first, last)
            );
            completions.push((charged > 0, completed == 1));
        }
        let all_charged_completed = completions
            .iter()
            .filter(|(charged, _)| *charged)
            .all(|(_, done)| *done);
        verdicts.push(all_charged_completed);
        println!();
    }

    println!("verdict");
    println!(
        "  0  per-destination delivery and charge ......... {}",
        yes_no(verdicts[0])
    );
    println!(
        "  1  even-rank address takes the copy ............ {}",
        yes_no(verdicts[1])
    );
    println!(
        "  2  ...charged once per destination ............. {}",
        yes_no(verdicts[2])
    );
    println!(
        "  3  each copy reaches ITS OWN pair leader ....... {}",
        yes_no(verdicts[3])
    );
    println!(
        "  4  both halves, both leaders charged twice ..... {}",
        yes_no(verdicts[4])
    );
    println!();
    if verdicts[3] && verdicts[4] {
        println!(
            "  The arrival lands at the given OFFSET, in the CTA of each destination's\n\
             \x20 own cta_group::2 pair picked by the supplied address's rank parity. An\n\
             \x20 even-rank address therefore charges every destination's pair LEADER, so\n\
             \x20 a 4-CTA cluster keeps main's one-barrier-per-pair structure exactly and\n\
             \x20 needs no peer-progress signal at all."
        );
    } else if verdicts[0] {
        println!(
            "  Delivery and charge are per destination, but a copy does not reach its\n\
             \x20 destination's pair leader: the two ranks whose A arrives without a\n\
             \x20 leader behind it have to forward it themselves (ClusterSemaphore::arrive\n\
             \x20 from the peer's otherwise idle issuer warp)."
        );
    } else {
        println!("  A replicating multicast cannot signal more than one CTA. No 4-CTA route.");
    }
    Ok(())
}
