//! kittens: a ThunderKittens-style tile library for cuda-oxide, tcgen05-only
//! (issue #61).
//!
//! New kernels are written against typed shared/register/TMEM tiles with
//! warp- and warpgroup-scoped ops instead of raw intrinsics and hand-threaded
//! index math. The library targets Blackwell (`sm_100a`) exclusively: the MMA
//! layer is tcgen05, with no wmma/wgmma backends and no arch dispatch.
//!
//! Everything here is a plain `#[inline(always)]` function or a Copy struct
//! of pointers and const generics — the crate ships no kernels and no
//! `#[cuda_module]`. Device code monomorphizes into the *calling* crate's
//! artifact the same way `cuda-device` does, so a kernel crate pays nothing
//! for the abstraction unless ptxas says otherwise (the bench-util budget
//! gate is the enforcement mechanism: same SASS, fewer lines).
//!
//! Pure-PTX discipline: nothing in this crate may call libdevice-lowered
//! math (`f32::exp/ln/sqrt/floor/max/min`) — kittens code must be linkable
//! into pure-PTX artifacts like flash-attn's `flash.ptx`, which libNVVM
//! would reject for its tcgen05 constructs.
//!
pub mod global;
pub mod ldst;
pub mod mma;
pub mod pipeline;
pub mod reg;
pub mod shared;
pub mod sync;
pub mod tmem;

pub use reg::{Fragment, RegTile, RegVec};
pub use shared::{Bf16, Element, OperandWalk, SharedTile, SharedTileRing, Swizzle, Swizzle128B};
pub use sync::{PhasedSemaphore, Semaphore, SemaphoreRing};
pub use tmem::TmemTile;
