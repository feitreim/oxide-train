//! Portable register-tiled fp32 GEMM kernels.

use cuda_core::LaunchConfig;
use cuda_device::shared::SharedArray;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// fp32 CTA output rows. Rewritten by the repository's `SWEEP` harness.
pub const BM: usize = 64;
/// fp32 CTA output columns. Rewritten by the repository's `SWEEP` harness.
pub const BN: usize = 64;
/// fp32 reduction tile. Rewritten by the repository's `SWEEP` harness.
pub const BK: usize = 64;
/// fp32 output rows held in each thread's registers.
pub const TM: usize = 4;
/// fp32 output columns held in each thread's registers.
pub const TN: usize = 4;

const THREADS_M: usize = BM / TM;
const THREADS_N: usize = BN / TN;
const THREADS: usize = THREADS_M * THREADS_N;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[inline(always)]
    unsafe fn gemm_impl<
        const TRANSPOSE_A: bool,
        const TRANSPOSE_B: bool,
        const ACCUMULATE: bool,
    >(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
    ) {
        unsafe {
            static mut TILE_A: SharedArray<f32, { BM * BK }> = SharedArray::UNINIT;
            static mut TILE_B: SharedArray<f32, { BK * BN }> = SharedArray::UNINIT;

            let tid = thread::threadIdx_x() as usize;
            let thread_row = tid / THREADS_N;
            let thread_col = tid % THREADS_N;
            let block_row = thread::blockIdx_y() as usize * BM;
            let block_col = thread::blockIdx_x() as usize * BN;
            let mut accumulators = [[0.0f32; TN]; TM];

            let mut k_base = 0usize;
            while k_base < k {
                let mut local = tid;
                while local < BM * BK {
                    // Assign transpose loads in the operand's physical
                    // row-major order, then scatter into the logical tile.
                    let (tile_row, tile_col) = if TRANSPOSE_A {
                        (local % BM, local / BM)
                    } else {
                        (local / BK, local % BK)
                    };
                    let global_row = block_row + tile_row;
                    let global_col = k_base + tile_col;
                    TILE_A[tile_row * BK + tile_col] = if global_row < m && global_col < k {
                        if TRANSPOSE_A {
                            a[global_col * m + global_row]
                        } else {
                            a[global_row * k + global_col]
                        }
                    } else {
                        0.0
                    };
                    local += THREADS;
                }

                local = tid;
                while local < BK * BN {
                    // B^T is stored as [N,K], so lanes must advance through K
                    // rather than issue strided reads across N.
                    let (tile_row, tile_col) = if TRANSPOSE_B {
                        (local % BK, local / BK)
                    } else {
                        (local / BN, local % BN)
                    };
                    let global_row = k_base + tile_row;
                    let global_col = block_col + tile_col;
                    TILE_B[tile_row * BN + tile_col] = if global_row < k && global_col < n {
                        if TRANSPOSE_B {
                            b[global_col * k + global_row]
                        } else {
                            b[global_row * n + global_col]
                        }
                    } else {
                        0.0
                    };
                    local += THREADS;
                }
                thread::sync_threads();

                let mut inner = 0usize;
                while inner < BK {
                    let mut row = 0usize;
                    while row < TM {
                        let av = TILE_A[(thread_row * TM + row) * BK + inner];
                        let mut col = 0usize;
                        while col < TN {
                            accumulators[row][col] +=
                                av * TILE_B[inner * BN + thread_col * TN + col];
                            col += 1;
                        }
                        row += 1;
                    }
                    inner += 1;
                }
                thread::sync_threads();
                k_base += BK;
            }

            let mut row = 0usize;
            while row < TM {
                let global_row = block_row + thread_row * TM + row;
                let mut col = 0usize;
                while col < TN {
                    let global_col = block_col + thread_col * TN + col;
                    if global_row < m && global_col < n {
                        let index = global_row * n + global_col;
                        if ACCUMULATE {
                            *c.get_unchecked_mut(index) += accumulators[row][col];
                        } else {
                            *c.get_unchecked_mut(index) = accumulators[row][col];
                        }
                    }
                    col += 1;
                }
                row += 1;
            }
        }
    }

    /// Register-tiled fp32 `C = A B`.
    #[kernel]
    pub unsafe fn register_gemm_store(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: DisjointSlice<f32>,
    ) {
        unsafe { gemm_impl::<false, false, false>(m, n, k, a, b, c) }
    }

    /// Register-tiled fp32 `C += A B`.
    #[kernel]
    pub unsafe fn register_gemm_accumulate(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: DisjointSlice<f32>,
    ) {
        unsafe { gemm_impl::<false, false, true>(m, n, k, a, b, c) }
    }

    /// Register-tiled fp32 `C = A B^T`, where `B` is stored as `[N,K]`.
    #[kernel]
    pub unsafe fn register_gemm_nt_store(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: DisjointSlice<f32>,
    ) {
        unsafe { gemm_impl::<false, true, false>(m, n, k, a, b, c) }
    }

    /// Register-tiled fp32 `C += A^T B`, where `A` is stored as `[K,M]`.
    #[kernel]
    pub unsafe fn register_gemm_tn_accumulate(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: DisjointSlice<f32>,
    ) {
        unsafe { gemm_impl::<true, false, true>(m, n, k, a, b, c) }
    }
}

/// Validate tuning constants and build the register-tiled fp32 launch.
pub fn launch_config(m: usize, n: usize) -> LaunchConfig {
    assert!(BM > 0 && BN > 0 && BK > 0 && TM > 0 && TN > 0);
    assert!(BM.is_multiple_of(TM) && BN.is_multiple_of(TN));
    assert!(THREADS <= 1024);
    assert!(m <= u32::MAX as usize && n <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            (n as u32).div_ceil(BN as u32),
            (m as u32).div_ceil(BM as u32),
            1,
        ),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}
