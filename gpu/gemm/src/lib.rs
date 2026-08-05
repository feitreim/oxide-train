/*
 * The tcgen05 path is adapted from cuda-oxide's Apache-2.0-licensed
 * `examples/gemm_sol`, and rebuilt in #67 on ferro-kittens' shipped GEMM shape.
 */

//! GEMM performance ladder for the B200 training target.
//!
//! The portable first rung is an fp32 shared-memory/register-tiled kernel
//! ([`fp32`]). The Blackwell rung is [`optimized`]: row-major bf16 `A[M, K]`
//! and transposed row-major bf16 `B[N, K]`, a `cta_group::2` pair UMMA over a
//! **persistent** grid of 148 clusters, three TMA stages, and an fp32 TMEM
//! accumulator drained to either packed bf16 or fp32 `C`, overwritten or
//! accumulated into. All four training variants share that one compute
//! pipeline, so a backward pass's parameter gradients need no separate add.
//!
//! [`host`] is the launch side: the tensor maps, the launch geometry, and the
//! typed adapters over cuda-oxide's generated launchers.
//!
//! bf16 crosses this crate's *host* boundary as raw `u16` / packed `u32` until
//! milestone 7d's shared `Element` plumbing; inside the kernels it is
//! `kittens::shared::Bf16` throughout.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod fp32;
pub use fp32::{BK, BM, BN, TM, TN, launch_config as fp32_launch_config};
pub mod host;
pub use host::{
    Bf16PairsTmaMap, Bf16TmaMap, TC_BK, TC_K_PIPELINE, TC_M_TILE, TC_N_TILE, TC_TILE, Tcgen05Gemm,
    TmaLayout, create_bf16_pairs_tma_map, create_bf16_pairs_tma_map_prefix, create_bf16_tma_map,
    tcgen05_launch_config,
};
pub mod optimized;
pub use optimized::{GROUP, MAX_CLUSTERS, SHARED_BYTES};

/// The denominator `src/bin/bench.rs` divides by. Behind a feature because it
/// is a link-time dependency on `libcublasLt.so` that nothing else here wants.
#[cfg(feature = "cublas")]
pub mod cublaslt;
