//! Warp-scope register↔shared tile movers.
//!
//! The store side of the swizzled-fragment path: a thread packs fp32
//! fragment values to bf16 (`cvt_f32x2_bf16x2`) and stores them through
//! `stmatrix` at addresses from [`crate::shared::SharedTile::swizzled_chunk`],
//! so an accumulating MMA reads the operand exactly like a TMA-loaded tile.

use cuda_device::ptx_asm;

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
