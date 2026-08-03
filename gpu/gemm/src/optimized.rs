// Adapted from cuda-oxide's Apache-2.0 `gemm_sol_final` kernel.
//
// One uniform epilogue mode keeps all four training variants on the same
// pair-UMMA compute pipeline:
//   0 = packed-bf16 store, 1 = packed-bf16 accumulate,
//   2 = fp32 store,        3 = fp32 accumulate.
#[cuda_module]
pub mod optimized_kernels {
    use super::*;

    #[inline(always)]
    unsafe fn store_output_pair(
        output: *mut u32,
        packed_index: usize,
        update: u32,
        mode: u32,
    ) {
        unsafe {
            if mode < 2 {
                let slot = output.add(packed_index);
                if mode == 0 {
                    *slot = update;
                } else {
                    *slot = super::kernels::accumulate_bf16_pair(*slot, update);
                }
            } else {
                let lo = super::kernels::bf16_to_f32(update as u16);
                let hi = super::kernels::bf16_to_f32((update >> 16) as u16);
                let slot = output.add(packed_index * 2);
                if mode == 2 {
                    let bits = (lo.to_bits() as u64) | ((hi.to_bits() as u64) << 32);
                    *(slot as *mut u64) = bits;
                } else {
                    *slot = f32::to_bits(f32::from_bits(*slot) + lo);
                    *slot.add(1) = f32::to_bits(f32::from_bits(*slot.add(1)) + hi);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn produce_stage(
        smem_a: *mut u8,
        smem_b: *mut u8,
        tma_sem: Semaphore,
        mma_sem: Semaphore,
        parity: u32,
        k_offset: i32,
        m_offset: i32,
        n_offset: i32,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        leader_cta: bool,
        lane_zero: bool,
        self_mask: u16,
        transposed: bool,
    ) {
        unsafe {
            mma_sem.wait(parity);
            if lane_zero {
                // Both ranks aim their halves at rank 0's barrier, and rank 0
                // alone charges for the whole cluster stage.
                let leader = tma_sem.at_rank(0);
                let charge = if transposed {
                    // MN-major operands: the map's fast axis is MN, so the
                    // coordinates swap and each 128-MN stage is one box per
                    // stacked subtile. Same transaction bytes, twice the boxes.
                    let a = MnStage::from_raw(smem_a);
                    let b = MnStage::from_raw(smem_b);
                    a.tma_load_2d_multicast_cg2(a_tma, m_offset, k_offset, leader, self_mask)
                        + b.tma_load_2d_multicast_cg2(b_tma, n_offset, k_offset, leader, self_mask)
                } else {
                    let a = KStage::from_raw(smem_a);
                    let b = KStage::from_raw(smem_b);
                    a.tma_load_2d_multicast_cg2(a_tma, k_offset, m_offset, leader, self_mask)
                        + b.tma_load_2d_multicast_cg2(b_tma, k_offset, n_offset, leader, self_mask)
                };
                if leader_cta {
                    tma_sem.expect_tx(charge.across_ranks(CTA_PAIR_RANKS));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn consume_stage(
        smem_a: *mut u8,
        smem_b: *mut u8,
        tma_sem: Semaphore,
        mma_sem: Semaphore,
        parity: u32,
        tmem: u32,
        shape: mma::MmaShape,
        accumulate_stage: bool,
        leader_cta: bool,
        lane_zero: bool,
        transposed: bool,
    ) {
        unsafe {
            if leader_cta {
                tma_sem.wait(parity);
                if lane_zero {
                    // The walk is value-level so both layouts share one issue
                    // loop — a select, not a duplicated MMA chain (the
                    // hand-written kernel's schedule, kept deliberately).
                    let (a, b) = if transposed {
                        (
                            MnStage::from_raw(smem_a).mn_walk(),
                            MnStage::from_raw(smem_b).mn_walk(),
                        )
                    } else {
                        (
                            KStage::from_raw(smem_a).k_walk(),
                            KStage::from_raw(smem_b).k_walk(),
                        )
                    };
                    mma::mma_walk_cg2::<Bf16, 4>(tmem, a, b, shape, accumulate_stage);
                    mma::commit_multicast_cg2(mma_sem, CTA_MASK_PAIR);
                }
            }
        }
    }

    const CTA_MASK_PAIR: u16 = 0b11;
    const CTA_PAIR_RANKS: u32 = 2;

    /// B200 GEMM: cta_group::2 pair-UMMA + four-stage TMA pipeline.
    ///
    /// Each CTA pair (cluster) computes one M256xN256 tile: a producer warp
    /// overlaps TMA with an MMA warp, while four epilogue warps drain the TMEM
    /// accumulator. The host launches exactly one cluster per output tile.
    ///
    /// This kernel does NOT use CLC work-stealing. The tcgen05 reference it was
    /// adapted from schedules tiles persistently via `clc_try_cancel`, but that
    /// cross-cluster cancel/steal handshake deadlocks a fraction of launches at
    /// small grids (a fast cluster cancels a not-yet-launched peer and the
    /// cancel accounting stalls; the multi-tile TMEM ACCUM ping-pong it exposes
    /// only compounds it). With an exact-cover grid, stealing buys nothing —
    /// every tile already has an owning cluster — so it is removed for a
    /// deterministic, deadlock-free schedule. The two-stage ACCUM buffer and its
    /// cross-cluster empty/full barriers are retained but inert (one tile each).
    ///
    /// `transposed` selects the operand layout. `0` is the default `C = A·Bᵀ`
    /// over K-major `[M,K]` and `[N,K]` operands. `1` sets the instruction
    /// descriptor's `transpose_a`/`transpose_b` bits so both operands are read
    /// MN-major — `A` as `[K,M]` and `B` as `[K,N]`, i.e. the *native*
    /// row-major activation and output-gradient panels of a weight gradient
    /// `dW += Aᵀ·B`, with nothing transposed in global memory.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn gemm_tcgen05_bf16_optimized_impl(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        output: *mut u32,
        n: i32,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
        mode: u32,
        transposed: u32,
    ) {
        unsafe {
            static mut SMEM_A0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A2: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A3: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B2: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B3: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 16384, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
            static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;

            static mut TMA_BARS: SharedArray<u64, 4, 8> = SharedArray::UNINIT;
            static mut MMA_BARS: SharedArray<u64, 4, 8> = SharedArray::UNINIT;
            static mut ACCUM_FULL: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut ACCUM_EMPTY: SharedArray<u64, 2, 8> = SharedArray::UNINIT;
            static mut TILE_READY: Barrier = Barrier::UNINIT;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;
            let rank = cluster::cluster_ctaidX();
            let leader_cta = rank == 0;
            let self_mask = 1u16 << rank;

            let tma_ring = SemaphoreRing::<4>::attach(&raw mut TMA_BARS as *mut Barrier);
            let mma_ring = SemaphoreRing::<4>::attach(&raw mut MMA_BARS as *mut Barrier);
            let accum_full = SemaphoreRing::<2>::attach(&raw mut ACCUM_FULL as *mut Barrier);
            let accum_empty = SemaphoreRing::<2>::attach(&raw mut ACCUM_EMPTY as *mut Barrier);
            let tile_ready = Semaphore::attach(&raw mut TILE_READY);

            if tid == 0 {
                tma_ring.init_all(1);
                mma_ring.init_all(1);
                accum_full.init_all(1);
                accum_empty.init_all(256);
                tile_ready.init(1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            // Pre-arrive every MMA barrier once so the producer's first ring
            // cycle finds its virgin slots "released" at parity 0.
            if tid == 0 {
                mma_ring.sem(0).arrive();
                mma_ring.sem(1).arrive();
                mma_ring.sem(2).arrive();
                mma_ring.sem(3).arrive();
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem = *(&raw const TMEM_ADDR as *const u32);
            let transposed = transposed != 0;
            // Element, accumulator and the transpose flags are the operand
            // walks' own; only the shape is still the caller's to state.
            let shape = mma::MmaShape::M256_N256;
            let k_iters = k as u32 / 64;
            let wide_tiles_n = tiles_n / 2;
            let wide_total = tiles_m * wide_tiles_n;
            let swizzle_g = if tiles_m <= 16 { 2 } else { 8 };
            cluster::cluster_sync();

            if warp_id == TMA_WARP {
                let lane_zero = lane_id == 0;
                let cluster_base = thread::blockIdx_x() - rank;
                let raw_tile = cluster_base / 2;
                let tile_seq = 0u32;

                // Exact-cover launch (one cluster per output tile): this cluster
                // owns exactly the tile at its block index, so it produces that
                // single tile and never steals. CLC work-stealing is deliberately
                // not used here — its cross-cluster cancel/steal handshake (and
                // the multi-tile TMEM ACCUM ping-pong it exercises) deadlocks a
                // fraction of launches at small grids. `tile_seq` stays 0 and the
                // consumer's `tile_iter` stays 0 to match.
                {
                    if raw_tile < wide_total {
                        let group_tiles = swizzle_g * tiles_m;
                        let group = raw_tile / group_tiles;
                        let within = raw_tile % group_tiles;
                        let n_start = group * swizzle_g;
                        let remaining = wide_tiles_n - n_start;
                        let band_width = if swizzle_g < remaining {
                            swizzle_g
                        } else {
                            remaining
                        };
                        let tile_m = within / band_width;
                        let tile_n = n_start + within % band_width;

                        if lane_zero {
                            *(&raw mut TILE_INFO as *mut u32).add(0) = tile_m;
                            *(&raw mut TILE_INFO as *mut u32).add(1) = tile_n;
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                            tile_ready.arrive();
                        }

                        let m_offset = (tile_m * 256 + rank * 128) as i32;
                        let n_offset = (tile_n * 256 + rank * 128) as i32;
                        let mut k_idx = 0u32;
                        while k_idx < k_iters {
                            let parity = ((tile_seq * k_iters + k_idx) >> 2) & 1;
                            produce_stage(
                                &raw mut SMEM_A0 as *mut u8,
                                &raw mut SMEM_B0 as *mut u8,
                                tma_ring.sem(0),
                                mma_ring.sem(0),
                                parity,
                                (k_idx * 64) as i32,
                                m_offset,
                                n_offset,
                                a_tma,
                                b_tma,
                                leader_cta,
                                lane_zero,
                                self_mask,
                                transposed,
                            );
                            produce_stage(
                                &raw mut SMEM_A1 as *mut u8,
                                &raw mut SMEM_B1 as *mut u8,
                                tma_ring.sem(1),
                                mma_ring.sem(1),
                                parity,
                                ((k_idx + 1) * 64) as i32,
                                m_offset,
                                n_offset,
                                a_tma,
                                b_tma,
                                leader_cta,
                                lane_zero,
                                self_mask,
                                transposed,
                            );
                            produce_stage(
                                &raw mut SMEM_A2 as *mut u8,
                                &raw mut SMEM_B2 as *mut u8,
                                tma_ring.sem(2),
                                mma_ring.sem(2),
                                parity,
                                ((k_idx + 2) * 64) as i32,
                                m_offset,
                                n_offset,
                                a_tma,
                                b_tma,
                                leader_cta,
                                lane_zero,
                                self_mask,
                                transposed,
                            );
                            produce_stage(
                                &raw mut SMEM_A3 as *mut u8,
                                &raw mut SMEM_B3 as *mut u8,
                                tma_ring.sem(3),
                                mma_ring.sem(3),
                                parity,
                                ((k_idx + 3) * 64) as i32,
                                m_offset,
                                n_offset,
                                a_tma,
                                b_tma,
                                leader_cta,
                                lane_zero,
                                self_mask,
                                transposed,
                            );
                            k_idx += 4;
                        }
                    }

                    // One tile produced (or none, if this cluster fell outside the
                    // work range); publish the terminating "no work" record so the
                    // MMA and epilogue warps drain and exit.
                    if lane_zero {
                        *(&raw mut TILE_INFO as *mut u32).add(2) = 0;
                        tile_ready.arrive();
                    }
                }
            }

            if warp_id == MMA_WARP {
                let lane_zero = lane_id == 0;
                let mut tile_iter = 0u32;
                let mut tile_parity = 0u32;
                loop {
                    tile_ready.wait(tile_parity);
                    tile_parity ^= 1;
                    if *(&raw const TILE_INFO as *const u32).add(2) == 0 {
                        break;
                    }

                    let accum_stage = tile_iter & 1;
                    let tmem_offset = accum_stage * 256;
                    if leader_cta && tile_iter >= 2 {
                        // Branch-selected literal stages keep the barrier
                        // addresses compile-time immediates, exactly like the
                        // hand-written kernel — a ring-computed address here
                        // perturbs the ptxas schedule.
                        let parity = ((tile_iter - 2) / 2) & 1;
                        let empty = if accum_stage == 0 {
                            accum_empty.sem(0)
                        } else {
                            accum_empty.sem(1)
                        };
                        empty.wait(parity);
                    }

                    let mut k_idx = 0u32;
                    while k_idx < k_iters {
                        let parity = ((tile_iter * k_iters + k_idx) >> 2) & 1;
                        consume_stage(
                            &raw mut SMEM_A0 as *mut u8,
                            &raw mut SMEM_B0 as *mut u8,
                            tma_ring.sem(0),
                            mma_ring.sem(0),
                            parity,
                            tmem + tmem_offset,
                            shape,
                            k_idx > 0,
                            leader_cta,
                            lane_zero,
                            transposed,
                        );
                        consume_stage(
                            &raw mut SMEM_A1 as *mut u8,
                            &raw mut SMEM_B1 as *mut u8,
                            tma_ring.sem(1),
                            mma_ring.sem(1),
                            parity,
                            tmem + tmem_offset,
                            shape,
                            true,
                            leader_cta,
                            lane_zero,
                            transposed,
                        );
                        consume_stage(
                            &raw mut SMEM_A2 as *mut u8,
                            &raw mut SMEM_B2 as *mut u8,
                            tma_ring.sem(2),
                            mma_ring.sem(2),
                            parity,
                            tmem + tmem_offset,
                            shape,
                            true,
                            leader_cta,
                            lane_zero,
                            transposed,
                        );
                        consume_stage(
                            &raw mut SMEM_A3 as *mut u8,
                            &raw mut SMEM_B3 as *mut u8,
                            tma_ring.sem(3),
                            mma_ring.sem(3),
                            parity,
                            tmem + tmem_offset,
                            shape,
                            true,
                            leader_cta,
                            lane_zero,
                            transposed,
                        );
                        k_idx += 4;
                    }
                    if leader_cta && lane_zero {
                        let full = if accum_stage == 0 {
                            accum_full.sem(0)
                        } else {
                            accum_full.sem(1)
                        };
                        mma::commit_multicast_cg2(full, CTA_MASK_PAIR);
                    }
                    tile_iter += 1;
                }
                tcgen05_relinquish_alloc_permit_cg2();
            }

            if warp_id < 4 {
                let accumulator = TmemTile::<128, 256>::from_raw(tmem);
                let mut tile_iter = 0u32;
                let mut tile_parity = 0u32;
                let leader_empty0 =
                    cluster::map_shared_rank(accum_empty.sem(0).raw() as *const Barrier, 0) as u64;
                let leader_empty1 =
                    cluster::map_shared_rank(accum_empty.sem(1).raw() as *const Barrier, 0) as u64;
                let warp_row = (warp_id * 32) as usize;
                let row_in_8 = (lane_id % 8) as usize;
                let matrix_offset = if (8..16).contains(&lane_id) {
                    16usize
                } else {
                    0usize
                };

                loop {
                    tile_ready.wait(tile_parity);
                    tile_parity ^= 1;
                    if *(&raw const TILE_INFO as *const u32).add(2) == 0 {
                        break;
                    }
                    let tile_m = *(&raw const TILE_INFO as *const u32);
                    let tile_n = *(&raw const TILE_INFO as *const u32).add(1);
                    let accum_stage = tile_iter & 1;
                    let stage_acc = accumulator.columns_right(accum_stage * 256);
                    let full_parity = (tile_iter / 2) & 1;
                    let full = if accum_stage == 0 {
                        accum_full.sem(0)
                    } else {
                        accum_full.sem(1)
                    };
                    full.wait(full_parity);

                    let mut row_block = 0u32;
                    while row_block < 2 {
                        let tmem_row = warp_id * 32 + row_block * 16;
                        let mut column_block = 0u32;
                        while column_block < 16 {
                            let column = (column_block * 16) as usize;
                            let (low, high) = stage_acc.fragment(tmem_row, column as u32);

                            let out_row =
                                warp_row + row_block as usize * 16 + row_in_8;
                            stmatrix_m8n8_x2(
                                (&raw mut SMEM_OUT as *mut u8).add(
                                    out_row * 512 + column * 2 + matrix_offset,
                                ),
                                cvt_f32x2_bf16x2(low[0], low[1]),
                                cvt_f32x2_bf16x2(high[0], high[1]),
                            );
                            stmatrix_m8n8_x2(
                                (&raw mut SMEM_OUT as *mut u8).add(
                                    (out_row + 8) * 512 + column * 2 + matrix_offset,
                                ),
                                cvt_f32x2_bf16x2(low[2], low[3]),
                                cvt_f32x2_bf16x2(high[2], high[3]),
                            );
                            column_block += 1;
                        }
                        row_block += 1;
                    }

                    let packed_n = n as usize / 2;
                    let global_row_base = (tile_m * 256 + rank * 128) as usize + warp_row;
                    let global_col_base = tile_n as usize * 128;
                    let mut element = lane_id as usize * 2;
                    while element < 4096 {
                        let row = element / 128;
                        let column = element % 128;
                        let smem = warp_row * 128 + row * 128 + column;
                        let global =
                            (global_row_base + row) * packed_n + global_col_base + column;
                        let packed = (SMEM_OUT[smem] as u64)
                            | ((SMEM_OUT[smem + 1] as u64) << 32);
                        if mode == 0 {
                            *(output.add(global) as *mut u64) = packed;
                        } else {
                            store_output_pair(output, global, packed as u32, mode);
                            store_output_pair(
                                output,
                                global + 1,
                                (packed >> 32) as u32,
                                mode,
                            );
                        }
                        element += 64;
                    }

                    if leader_cta {
                        if accum_stage == 0 {
                            accum_empty.sem(0).arrive();
                        } else {
                            accum_empty.sem(1).arrive();
                        }
                    } else if accum_stage == 0 {
                        mbarrier_arrive_cluster(leader_empty0);
                    } else {
                        mbarrier_arrive_cluster(leader_empty1);
                    }
                    tile_iter += 1;
                }
            }

            cluster::cluster_sync();
            if warp_id == 0 {
                tcgen05_dealloc_cg2(tmem, 512);
            }
            if tid == 0 {
                tma_ring.inval_all();
                mma_ring.inval_all();
                accum_full.inval_all();
                accum_empty.inval_all();
                tile_ready.inval();
            }
        }
    }

    /// Typed packed-bf16 entry point for overwrite/accumulate modes.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn gemm_tcgen05_bf16_optimized(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut output: DisjointSlice<u32>,
        n: i32,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
        mode: u32,
        transposed: u32,
    ) {
        unsafe {
            gemm_tcgen05_bf16_optimized_impl(
                a_tma,
                b_tma,
                output.as_mut_ptr(),
                n,
                k,
                tiles_m,
                tiles_n,
                mode,
                transposed,
            )
        }
    }

    /// Typed fp32 entry point. `output_offset` selects one matrix in a stacked
    /// allocation without host-side pointer marshalling.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn gemm_tcgen05_f32_optimized(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut output: DisjointSlice<f32>,
        output_offset: usize,
        n: i32,
        k: i32,
        tiles_m: u32,
        tiles_n: u32,
        mode: u32,
        transposed: u32,
    ) {
        unsafe {
            gemm_tcgen05_bf16_optimized_impl(
                a_tma,
                b_tma,
                output.as_mut_ptr().add(output_offset) as *mut u32,
                n,
                k,
                tiles_m,
                tiles_n,
                mode,
                transposed,
            )
        }
    }
}
