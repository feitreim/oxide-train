//! TMEM accumulator views (issue #61 phase 3).
//!
//! A tcgen05 accumulator lives in tensor memory, addressed as
//! `base + (row << 16) + column`: the high half-word selects one of the
//! 128 TMEM lanes (accumulator rows), the low half-word a 4-byte column.
//! `TmemTile` carries that address plus the logical `[R, C]` fp32 shape,
//! so kernel code names its `S`/`O`/`dP` segments as tiles instead of
//! threading bare `u32` addresses and `(row << 16) + column` arithmetic
//! through every drain loop.
//!
//! MMA shapes with phantom rows (flash's `M128` over 64-row tiles) still
//! type the *drained* shape: `R`/`C` describe what the kernel reads back,
//! not what the instruction touches.

use cuda_device::cusimd::TmemRegs4;
use cuda_device::tcgen05::{tcgen05_ld_16x256b_pure, tcgen05_load_wait};

/// An fp32 accumulator segment in tensor memory.
#[derive(Clone, Copy)]
pub struct TmemTile<const R: usize, const C: usize> {
    address: u32,
}

impl<const R: usize, const C: usize> TmemTile<R, C> {
    /// Wrap a TMEM address (as returned through `tcgen05_alloc`'s shared
    /// staging word, plus any column offset already applied).
    pub const fn from_raw(address: u32) -> Self {
        Self { address }
    }

    /// The raw address, for the MMA issue path.
    pub const fn raw(self) -> u32 {
        self.address
    }

    /// Address of `(row, column)`: rows ride the high half-word.
    pub const fn at(self, row: u32, column: u32) -> u32 {
        self.address + (row << 16) + column
    }

    /// The segment `columns` fp32 columns to the right — accumulator
    /// ping-pong stages (gemm's `accum_stage * 256`) or a second output
    /// band sharing one allocation.
    pub const fn columns_right(self, columns: u32) -> Self {
        Self {
            address: self.address + columns,
        }
    }

    /// One thread's eight-value fragment of the 16-row block at `row`:
    /// two `16x256b` collective loads at `column` and `column + 8`, each
    /// drained through `tcgen05_load_wait`. Under the 16x256b map a thread
    /// holds rows `lane/4` and `lane/4 + 8` of the block at column offsets
    /// `2*(lane%4) + {0, 1, 8, 9}` — the low simd carries the `lane/4` row,
    /// the high simd the `+8` row.
    ///
    /// # Safety
    ///
    /// All 32 lanes of a warp that owns the TMEM rows `row..row+16` must
    /// call this together, after the MMA writing them has committed.
    #[inline(always)]
    pub unsafe fn fragment(self, row: u32, column: u32) -> (TmemRegs4, TmemRegs4) {
        unsafe {
            let low = tcgen05_ld_16x256b_pure(self.at(row, column));
            tcgen05_load_wait();
            let high = tcgen05_ld_16x256b_pure(self.at(row, column + 8));
            tcgen05_load_wait();
            (low, high)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addressing_rides_the_high_half_word() {
        let tile = TmemTile::<128, 64>::from_raw(0x0001_0000);
        assert_eq!(tile.at(0, 0), 0x0001_0000);
        assert_eq!(tile.at(32, 24), 0x0021_0018);
        assert_eq!(tile.columns_right(64).at(0, 0), 0x0001_0040);
    }
}
