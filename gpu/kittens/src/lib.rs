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
//! Libdevice math is legal beside tcgen05 in the same pure-PTX artifact at
//! cuda-oxide b099f64. Software approximations remain where their lowering is
//! a measured kernel optimization rather than an artifact-path workaround.
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
