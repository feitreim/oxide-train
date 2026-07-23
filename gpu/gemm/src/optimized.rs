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
    fn build_smem_descriptor(
        address: u64,
        leading_bytes: u32,
        stride_bytes: u32,
        swizzle: u8,
    ) -> u64 {
        ((address >> 4) & 0x3fff)
            | ((((leading_bytes >> 4) & 0x3fff) as u64) << 16)
            | ((((stride_bytes >> 4) & 0x3fff) as u64) << 32)
            | (1u64 << 46)
            | ((swizzle as u64) << 61)
    }

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

    #[inline(always)]
    unsafe fn produce_stage(
        smem_a: *mut u8,
        smem_b: *mut u8,
        tma_bar: *const Barrier,
        tma_bar_mut: *mut Barrier,
        mma_bar: *const Barrier,
        parity: u32,
        k_offset: i32,
        m_offset: i32,
        n_offset: i32,
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        leader_cta: bool,
        lane_zero: bool,
        self_mask: u16,
    ) {
        unsafe {
            while !mbarrier_try_wait_parity(mma_bar, parity) {}
            if lane_zero {
                if leader_cta {
                    mbarrier_arrive_expect_tx(tma_bar, 1, (128 * 64 * 2) * 4);
                }
                let aliased = ((tma_bar_mut as u32) & 0xFEFFFFF8) as *mut Barrier;
                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                    smem_a, a_tma, k_offset, m_offset, aliased, self_mask,
                );
                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                    smem_b, b_tma, k_offset, n_offset, aliased, self_mask,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn consume_stage(
        smem_a: u64,
        smem_b: u64,
        tma_bar: *const Barrier,
        mma_bar: *mut Barrier,
        parity: u32,
        tmem: u32,
        instruction: u32,
        accumulate_stage: bool,
        leader_cta: bool,
        lane_zero: bool,
    ) {
        unsafe {
            if leader_cta {
                while !mbarrier_try_wait_parity(tma_bar, parity) {}
                if lane_zero {
                    let mut inner = 0u32;
                    while inner < 4 {
                        let offset = (inner * 32) as u64;
                        tcgen05_mma_f16_cg2(
                            tmem,
                            build_smem_descriptor(smem_a + offset, 16, 1024, 2),
                            build_smem_descriptor(smem_b + offset, 16, 1024, 2),
                            instruction,
                            accumulate_stage || inner > 0,
                        );
                        inner += 1;
                    }
                    tcgen05_commit_multicast_cg2(mma_bar as *mut u64, 0b11);
                }
            }
        }
    }

    /// B200 GEMM: CLC + cta_group::2 pair-UMMA + four-stage TMA pipeline.
    ///
    /// Each CTA pair computes M256xN256. A producer warp overlaps TMA with an
    /// MMA warp, while four epilogue warps drain one of two TMEM accumulator
    /// stages. The host launches one CTA pair per logical output tile; the CLC
    /// protocol remains available without overprovisioning redundant clusters.
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

            static mut TMA_BAR0: Barrier = Barrier::UNINIT;
            static mut TMA_BAR1: Barrier = Barrier::UNINIT;
            static mut TMA_BAR2: Barrier = Barrier::UNINIT;
            static mut TMA_BAR3: Barrier = Barrier::UNINIT;
            static mut MMA_BAR0: Barrier = Barrier::UNINIT;
            static mut MMA_BAR1: Barrier = Barrier::UNINIT;
            static mut MMA_BAR2: Barrier = Barrier::UNINIT;
            static mut MMA_BAR3: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;
            static mut TILE_READY: Barrier = Barrier::UNINIT;
            static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
            static mut CLC_BAR: Barrier = Barrier::UNINIT;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;
            const CTA_MASK_PAIR: u16 = 0b11;

            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;
            let rank = cluster::cluster_ctaidX();
            let leader_cta = rank == 0;
            let self_mask = 1u16 << rank;

            if tid == 0 {
                mbarrier_init(&raw mut TMA_BAR0, 1);
                mbarrier_init(&raw mut TMA_BAR1, 1);
                mbarrier_init(&raw mut TMA_BAR2, 1);
                mbarrier_init(&raw mut TMA_BAR3, 1);
                mbarrier_init(&raw mut MMA_BAR0, 1);
                mbarrier_init(&raw mut MMA_BAR1, 1);
                mbarrier_init(&raw mut MMA_BAR2, 1);
                mbarrier_init(&raw mut MMA_BAR3, 1);
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 256);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 256);
                mbarrier_init(&raw mut TILE_READY, 1);
                mbarrier_init(&raw mut CLC_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            if tid == 0 {
                mbarrier_arrive(&raw const MMA_BAR0);
                mbarrier_arrive(&raw const MMA_BAR1);
                mbarrier_arrive(&raw const MMA_BAR2);
                mbarrier_arrive(&raw const MMA_BAR3);
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem = *(&raw const TMEM_ADDR as *const u32);
            let instruction = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M256_N256)
                .element_type(Tcgen05ElementType::BF16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let k_iters = k as u32 / 64;
            let wide_tiles_n = tiles_n / 2;
            let wide_total = tiles_m * wide_tiles_n;
            let swizzle_g = if tiles_m <= 16 { 2 } else { 8 };
            cluster::cluster_sync();

            if warp_id == TMA_WARP {
                let lane_zero = lane_id == 0;
                let cluster_base = thread::blockIdx_x() - rank;
                let mut raw_tile = cluster_base / 2;
                let mut tile_seq = 0u32;
                let mut clc_iter = 0u32;
                let response = &raw mut CLC_RESPONSE as *mut u64;

                loop {
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
                            mbarrier_arrive(&raw const TILE_READY);
                        }

                        let m_offset = (tile_m * 256 + rank * 128) as i32;
                        let n_offset = (tile_n * 256 + rank * 128) as i32;
                        let mut k_idx = 0u32;
                        while k_idx < k_iters {
                            let parity = ((tile_seq * k_iters + k_idx) >> 2) & 1;
                            produce_stage(
                                &raw mut SMEM_A0 as *mut u8,
                                &raw mut SMEM_B0 as *mut u8,
                                &raw const TMA_BAR0,
                                &raw mut TMA_BAR0,
                                &raw const MMA_BAR0,
                                parity,
                                (k_idx * 64) as i32,
                                m_offset,
                                n_offset,
                                a_tma,
                                b_tma,
                                leader_cta,
                                lane_zero,
                                self_mask,
                            );
                            produce_stage(
                                &raw mut SMEM_A1 as *mut u8,
                                &raw mut SMEM_B1 as *mut u8,
                                &raw const TMA_BAR1,
                                &raw mut TMA_BAR1,
                                &raw const MMA_BAR1,
                                parity,
                                ((k_idx + 1) * 64) as i32,
                                m_offset,
                                n_offset,
                                a_tma,
                                b_tma,
                                leader_cta,
                                lane_zero,
                                self_mask,
                            );
                            produce_stage(
                                &raw mut SMEM_A2 as *mut u8,
                                &raw mut SMEM_B2 as *mut u8,
                                &raw const TMA_BAR2,
                                &raw mut TMA_BAR2,
                                &raw const MMA_BAR2,
                                parity,
                                ((k_idx + 2) * 64) as i32,
                                m_offset,
                                n_offset,
                                a_tma,
                                b_tma,
                                leader_cta,
                                lane_zero,
                                self_mask,
                            );
                            produce_stage(
                                &raw mut SMEM_A3 as *mut u8,
                                &raw mut SMEM_B3 as *mut u8,
                                &raw const TMA_BAR3,
                                &raw mut TMA_BAR3,
                                &raw const MMA_BAR3,
                                parity,
                                ((k_idx + 3) * 64) as i32,
                                m_offset,
                                n_offset,
                                a_tma,
                                b_tma,
                                leader_cta,
                                lane_zero,
                                self_mask,
                            );
                            k_idx += 4;
                        }
                        tile_seq += 1;
                    }

                    let clc_parity = clc_iter & 1;
                    if lane_zero {
                        mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                        if leader_cta {
                            clc_try_cancel_multicast(response as *mut u8, &raw mut CLC_BAR);
                        }
                    }
                    while !mbarrier_try_wait_parity(&raw const CLC_BAR, clc_parity) {}
                    clc_iter += 1;
                    let lo = *response;
                    let hi = *response.add(1);
                    if clc_query_is_canceled(lo, hi) == 0 {
                        if lane_zero {
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 0;
                            mbarrier_arrive(&raw const TILE_READY);
                        }
                        break;
                    }
                    raw_tile = clc_query_get_first_ctaid_x(lo, hi) / 2;
                }
            }

            if warp_id == MMA_WARP {
                let lane_zero = lane_id == 0;
                let mut tile_iter = 0u32;
                let mut tile_parity = 0u32;
                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;
                    if *(&raw const TILE_INFO as *const u32).add(2) == 0 {
                        break;
                    }

                    let accum_stage = tile_iter & 1;
                    let tmem_offset = accum_stage * 256;
                    if leader_cta && tile_iter >= 2 {
                        let parity = ((tile_iter - 2) / 2) & 1;
                        let empty = if accum_stage == 0 {
                            &raw const ACCUM_EMPTY0
                        } else {
                            &raw const ACCUM_EMPTY1
                        };
                        while !mbarrier_try_wait_parity(empty, parity) {}
                    }

                    let mut k_idx = 0u32;
                    while k_idx < k_iters {
                        let parity = ((tile_iter * k_iters + k_idx) >> 2) & 1;
                        consume_stage(
                            &raw const SMEM_A0 as u64,
                            &raw const SMEM_B0 as u64,
                            &raw const TMA_BAR0,
                            &raw mut MMA_BAR0,
                            parity,
                            tmem + tmem_offset,
                            instruction,
                            k_idx > 0,
                            leader_cta,
                            lane_zero,
                        );
                        consume_stage(
                            &raw const SMEM_A1 as u64,
                            &raw const SMEM_B1 as u64,
                            &raw const TMA_BAR1,
                            &raw mut MMA_BAR1,
                            parity,
                            tmem + tmem_offset,
                            instruction,
                            true,
                            leader_cta,
                            lane_zero,
                        );
                        consume_stage(
                            &raw const SMEM_A2 as u64,
                            &raw const SMEM_B2 as u64,
                            &raw const TMA_BAR2,
                            &raw mut MMA_BAR2,
                            parity,
                            tmem + tmem_offset,
                            instruction,
                            true,
                            leader_cta,
                            lane_zero,
                        );
                        consume_stage(
                            &raw const SMEM_A3 as u64,
                            &raw const SMEM_B3 as u64,
                            &raw const TMA_BAR3,
                            &raw mut MMA_BAR3,
                            parity,
                            tmem + tmem_offset,
                            instruction,
                            true,
                            leader_cta,
                            lane_zero,
                        );
                        k_idx += 4;
                    }
                    if leader_cta && lane_zero {
                        let full = if accum_stage == 0 {
                            &raw mut ACCUM_FULL0
                        } else {
                            &raw mut ACCUM_FULL1
                        };
                        tcgen05_commit_multicast_cg2(full as *mut u64, CTA_MASK_PAIR);
                    }
                    tile_iter += 1;
                }
                tcgen05_relinquish_alloc_permit_cg2();
            }

            if warp_id < 4 {
                let mut tile_iter = 0u32;
                let mut tile_parity = 0u32;
                let leader_empty0 = cluster::map_shared_rank(&raw const ACCUM_EMPTY0, 0) as u64;
                let leader_empty1 = cluster::map_shared_rank(&raw const ACCUM_EMPTY1, 0) as u64;
                let warp_row = (warp_id * 32) as usize;
                let row_in_8 = (lane_id % 8) as usize;
                let matrix_offset = if (8..16).contains(&lane_id) {
                    16usize
                } else {
                    0usize
                };

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;
                    if *(&raw const TILE_INFO as *const u32).add(2) == 0 {
                        break;
                    }
                    let tile_m = *(&raw const TILE_INFO as *const u32);
                    let tile_n = *(&raw const TILE_INFO as *const u32).add(1);
                    let accum_stage = tile_iter & 1;
                    let tmem_offset = accum_stage * 256;
                    let full_parity = (tile_iter / 2) & 1;
                    let full = if accum_stage == 0 {
                        &raw const ACCUM_FULL0
                    } else {
                        &raw const ACCUM_FULL1
                    };
                    while !mbarrier_try_wait_parity(full, full_parity) {}

                    let mut row_block = 0u32;
                    while row_block < 2 {
                        let tmem_row = warp_id * 32 + row_block * 16;
                        let mut column_block = 0u32;
                        while column_block < 16 {
                            let column = (column_block * 16) as usize;
                            let low = tcgen05_ld_16x256b_pure(
                                tmem
                                    + tmem_offset
                                    + (tmem_row << 16)
                                    + column as u32,
                            );
                            tcgen05_load_wait();
                            let high = tcgen05_ld_16x256b_pure(
                                tmem
                                    + tmem_offset
                                    + (tmem_row << 16)
                                    + column as u32
                                    + 8,
                            );
                            tcgen05_load_wait();

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
                            *(output.as_mut_ptr().add(global) as *mut u64) = packed;
                        } else {
                            store_output_pair(output.as_mut_ptr(), global, packed as u32, mode);
                            store_output_pair(
                                output.as_mut_ptr(),
                                global + 1,
                                (packed >> 32) as u32,
                                mode,
                            );
                        }
                        element += 64;
                    }

                    if leader_cta {
                        if accum_stage == 0 {
                            mbarrier_arrive(&raw const ACCUM_EMPTY0);
                        } else {
                            mbarrier_arrive(&raw const ACCUM_EMPTY1);
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
                mbarrier_inval(&raw mut TMA_BAR0);
                mbarrier_inval(&raw mut TMA_BAR1);
                mbarrier_inval(&raw mut TMA_BAR2);
                mbarrier_inval(&raw mut TMA_BAR3);
                mbarrier_inval(&raw mut MMA_BAR0);
                mbarrier_inval(&raw mut MMA_BAR1);
                mbarrier_inval(&raw mut MMA_BAR2);
                mbarrier_inval(&raw mut MMA_BAR3);
                mbarrier_inval(&raw mut ACCUM_FULL0);
                mbarrier_inval(&raw mut ACCUM_FULL1);
                mbarrier_inval(&raw mut ACCUM_EMPTY0);
                mbarrier_inval(&raw mut ACCUM_EMPTY1);
                mbarrier_inval(&raw mut TILE_READY);
                mbarrier_inval(&raw mut CLC_BAR);
            }
        }
    }
}
