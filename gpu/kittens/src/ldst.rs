//! Warp-scope register↔shared tile movers.
//!
//! The store side of the swizzled-fragment path: a thread packs fp32
//! fragment values to bf16 (`cvt_f32x2_bf16x2`) and stores them through
//! `stmatrix` at addresses from [`crate::shared::SharedTile::swizzled_chunk`],
//! so an accumulating MMA reads the operand exactly like a TMA-loaded tile.

use cuda_device::ptx_asm;
use cuda_device::tcgen05::cvt_f32x2_bf16x2;

use crate::reg::Fragment;
use crate::shared::SwizzledChunks;

/// Pack one thread's [`Fragment`] to bf16 and store it into a single-subtile
/// swizzled tile — the store twin of
/// [`crate::tmem::TmemTile::fragment_tile`], addressed by the same
/// `(row, column)` the drain used.
///
/// Two `stmatrix.m8n8.x2` writes cover the fragment's two rows. Each takes its
/// addresses from lanes 0..15 only (lanes 0..7 the first matrix, 8..15 the
/// second), which is why the 16-byte chunk is the column's chunk index plus
/// one for the upper half-warp, and the row is `lane % 8` into the block.
///
/// # Safety
///
/// All 32 lanes of the warp owning TMEM rows `row..row+16` must call this
/// together, `chunks` must belong to a tile at least `row + 16` rows tall, and
/// the caller owes a `fence.proxy.async.shared::cta` before any MMA reads the
/// tile.
#[inline(always)]
pub unsafe fn store_fragment_bf16(
    chunks: SwizzledChunks,
    row: u32,
    column: u32,
    lane: u32,
    fragment: Fragment,
) {
    unsafe {
        let chunk = (column / 8) as usize + usize::from((8..16).contains(&lane));
        let low = row as usize + (lane % 8) as usize;
        let mut slot = 0usize;
        while slot < 2 {
            stmatrix_m8n8_x2(
                chunks.at(low + 8 * slot, chunk),
                cvt_f32x2_bf16x2(fragment.0[slot][0], fragment.0[slot][1]),
                cvt_f32x2_bf16x2(fragment.0[slot][2], fragment.0[slot][3]),
            );
            slot += 1;
        }
    }
}

/// Store two packed b16 matrix fragments (`stmatrix.sync.aligned.m8n8.x2`)
/// without routing through the unresolved LLVM stmatrix declaration emitted
/// by cuda-oxide b099f64.
///
/// # Safety
///
/// `smem_ptr` must be a 16-byte-aligned shared-memory address with 32 bytes
/// writable, and all 32 lanes of the warp must call this together.
#[inline(always)]
pub unsafe fn stmatrix_m8n8_x2(smem_ptr: *mut u8, r0: u32, r1: u32) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 smem; cvta.to.shared.u64 smem, %0; stmatrix.sync.aligned.m8n8.x2.shared.b16 [smem], {%1, %2}; }",
            in("l") smem_ptr as u64,
            in("r") r0,
            in("r") r1,
            clobber("memory"),
        );
    }
}
