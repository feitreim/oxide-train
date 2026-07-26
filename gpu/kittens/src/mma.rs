//! Chained tcgen05 MMA walks over shared-tile operands (issue #61 phase 3).
//!
//! tcgen05's native step is `D (+)= A·Bᵀ` over a K=16 bf16 chunk; a full
//! tile multiply is a chain of those steps with `enable_d` linking every
//! step after the first into the accumulator. The walks here are the ones
//! the flash and gemm kernels proved out (SPEC decision 15: abstractions
//! are extracted from working variants, not designed first): flash's
//! single-CTA [`mma_abt`]/[`mma_ab`] with the layout in the types, and
//! gemm's cluster-pair (`M256_N256` multicast) [`mma_walk_cg2`] with the
//! layout in [`OperandWalk`] values, because that kernel selects K-major
//! vs MN-major at runtime.
//!
//! Chunk geometry, fixed by bf16 and the 128-byte swizzle atom: one K=16
//! chunk is 32 bytes along a row, so a subtile row holds four chunks. A
//! K-major operand walks `(k / 4) * SUBTILE_BYTES + (k % 4) * 32` across
//! its stacked subtiles; an MN-major operand (transpose-bit forms) supplies
//! K along rows instead, 16 rows (`16 * ATOM_BYTES` bytes) per chunk. The
//! instruction descriptor caller-side must match the walk: no transpose
//! bits for [`mma_abt`], `transpose_b` for [`mma_ab`], both transpose bits
//! for an MN-major [`mma_walk_cg2`].

use cuda_device::tcgen05::{
    tcgen05_commit_multicast_cg2, tcgen05_commit_shared_cluster, tcgen05_mma_f16,
    tcgen05_mma_f16_cg2,
};

use crate::shared::{Bf16, OperandWalk, SharedTile, Swizzle};
use crate::sync::Semaphore;

/// K elements per chained-MMA chunk (one bf16 core-matrix step).
const K_CHUNK: usize = 16;
/// Bytes of one K chunk along a row.
const K_CHUNK_BYTES: usize = K_CHUNK * 2;
/// K chunks per swizzle-atom row (four for bf16 under a 128-byte atom).
const CHUNKS_PER_ROW: usize = 128 / K_CHUNK_BYTES;

/// Byte offset of K chunk `k` in a K-major operand whose stacked subtiles
/// are `subtile_bytes` apart.
const fn k_major_offset(k: usize, subtile_bytes: usize) -> usize {
    (k / CHUNKS_PER_ROW) * subtile_bytes + (k % CHUNKS_PER_ROW) * K_CHUNK_BYTES
}

/// Byte offset of K chunk `k` in a `transpose_b` operand (K along rows).
const fn k_rows_offset(k: usize, atom_bytes: usize) -> usize {
    k * K_CHUNK * atom_bytes
}

/// `D (+)= A·Bᵀ` with both operands K-major `[rows, K]` — flash's
/// `S = Q·Kᵀ` walk. One chained MMA per K=16 chunk from the current leader
/// thread; `instruction` must carry no transpose bits and an `N` matching
/// the caller's accumulator band. The operands may stack different row
/// counts (7e15's paired-`[128, K]` A against an unpaired `[64, K]` B) —
/// each walks its own subtile stride.
///
/// # Safety
///
/// Exactly one thread issues this; `tmem` names an accumulator the
/// instruction's shape fits; both tiles hold committed operand data until
/// the MMA's own commit is observed.
#[inline(always)]
pub unsafe fn mma_abt<const AR: usize, const BR: usize, const K: usize, S: Swizzle>(
    tmem: u32,
    a: SharedTile<Bf16, AR, K, S>,
    b: SharedTile<Bf16, BR, K, S>,
    instruction: u32,
    accumulate: bool,
) {
    unsafe {
        let mut chunk = 0;
        while chunk < K / K_CHUNK {
            tcgen05_mma_f16(
                tmem,
                a.operand_descriptor(k_major_offset(
                    chunk,
                    SharedTile::<Bf16, AR, K, S>::SUBTILE_BYTES,
                )),
                b.operand_descriptor(k_major_offset(
                    chunk,
                    SharedTile::<Bf16, BR, K, S>::SUBTILE_BYTES,
                )),
                instruction,
                accumulate || chunk > 0,
            );
            chunk += 1;
        }
    }
}

/// `D (+)= A·B` — flash's `O = P·V` / gradient walk. `A` is K-major
/// `[rows, K]`; `B` supplies K along its rows (`instruction` must set
/// `transpose_b`), one 64-wide output band per `B` subtile, accumulated
/// into `tmem + subtile * 64`. `accumulate` false starts every band's
/// accumulator fresh.
///
/// # Safety
///
/// As [`mma_abt`]; `tmem` must own `64 * B::SUBTILES` fp32 columns.
#[inline(always)]
pub unsafe fn mma_ab<const AR: usize, const K: usize, const N: usize, S: Swizzle>(
    tmem: u32,
    a: SharedTile<Bf16, AR, K, S>,
    b: SharedTile<Bf16, K, N, S>,
    instruction: u32,
    accumulate: bool,
) {
    unsafe {
        let mut band = 0;
        while band < SharedTile::<Bf16, K, N, S>::SUBTILES {
            let band_base = band * SharedTile::<Bf16, K, N, S>::SUBTILE_BYTES;
            let mut chunk = 0;
            while chunk < K / K_CHUNK {
                tcgen05_mma_f16(
                    tmem + (band as u32) * 64,
                    a.operand_descriptor(k_major_offset(
                        chunk,
                        SharedTile::<Bf16, AR, K, S>::SUBTILE_BYTES,
                    )),
                    b.operand_descriptor(band_base + k_rows_offset(chunk, S::ATOM_BYTES)),
                    instruction,
                    accumulate || chunk > 0,
                );
                chunk += 1;
            }
            band += 1;
        }
    }
}

/// A cta_group::2 chained MMA over [`OperandWalk`] operands — one
/// instruction per chunk drives the CTA pair's shared `M256`-class
/// accumulator, each CTA contributing its own operand halves. The layout
/// lives in the walk values ([`SharedTile::k_walk`] /
/// [`SharedTile::mn_walk`]), so a kernel selecting K-major vs MN-major at
/// runtime (gemm's `transposed`) issues one loop either way; the
/// instruction descriptor's transpose bits must match the walks.
///
/// # Safety
///
/// As [`mma_abt`], from the *leader* CTA's issuing thread only, with the
/// cluster's peer holding its operand halves at the same shared offsets;
/// both walks must cover `CHUNKS` K=16 chunks of committed data.
#[inline(always)]
pub unsafe fn mma_walk_cg2<const CHUNKS: usize>(
    tmem: u32,
    a: OperandWalk,
    b: OperandWalk,
    instruction: u32,
    accumulate: bool,
) {
    unsafe {
        let mut chunk = 0;
        while chunk < CHUNKS {
            tcgen05_mma_f16_cg2(
                tmem,
                a.chunk_descriptor(chunk),
                b.chunk_descriptor(chunk),
                instruction,
                accumulate || chunk > 0,
            );
            chunk += 1;
        }
    }
}

/// Publish the issued MMA chain to `sem`: every consumer that `wait`s the
/// semaphore afterward observes the accumulator complete.
///
/// # Safety
///
/// Same issuing thread as the MMAs it commits; `sem` initialized.
#[inline(always)]
pub unsafe fn commit(sem: Semaphore) {
    unsafe { tcgen05_commit_shared_cluster(sem.raw() as *mut u64) }
}

/// Publish a cta_group::2 MMA chain to every CTA in `cta_mask`'s copy of
/// `sem` — the pair-UMMA commit (`0b11` for both halves of the pair).
///
/// # Safety
///
/// As [`commit`], from the leader CTA's issuing thread, with each masked
/// CTA holding an initialized barrier at `sem`'s address.
#[inline(always)]
pub unsafe fn commit_multicast_cg2(sem: Semaphore, cta_mask: u16) {
    unsafe { tcgen05_commit_multicast_cg2(sem.raw() as *mut u64, cta_mask) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_major_walk_matches_flash_score_mma() {
        // flash: (chunk / 4) * SUBTILE_BYTES + (chunk % 4) * 32 over 8 chunks.
        let subtile = 64 * 128;
        for chunk in 0..8 {
            assert_eq!(
                k_major_offset(chunk, subtile),
                (chunk / 4) * subtile + (chunk % 4) * 32
            );
        }
    }

    #[test]
    fn row_walk_matches_flash_grad_mma() {
        // flash: chunk * 2048 (16 rows of 128-byte atoms) over 4 chunks.
        for chunk in 0..4 {
            assert_eq!(k_rows_offset(chunk, 128), chunk * 2048);
        }
    }
}
