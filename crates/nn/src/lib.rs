//! Typed, module-level reverse-mode differentiation.
//!
//! There is no runtime tape. A model is a static composition of [`Module`]s;
//! backward of a composition is the reverse composition, derived by
//! combinators like [`Chain`] and checked by the type system at compile time.
//! Each leaf module hand-implements its backward and is finite-difference
//! checked (see [`gradcheck`]) against the CPU reference ops.
//!
//! Ownership contract: `forward` takes its input *by value*. A module that
//! needs its input during backward moves it into its `Ctx` — no implicit
//! clones, which on the GPU side means no implicit device copies. Values that
//! must be used twice (e.g. residual streams) are duplicated by an explicit
//! combinator that owns that policy.

pub mod attention;
pub mod cross_entropy;
pub mod embedding;
pub mod gradcheck;
pub mod linear;
pub mod llama;
pub mod module;
pub mod moe;
pub mod moe_llama;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;

pub use attention::{AttentionInput, CausalAttention};
pub use cross_entropy::{SoftmaxCrossEntropy, SoftmaxCrossEntropyInput};
pub use embedding::{Embedding, TokenIds};
pub use linear::Linear;
pub use llama::{Llama, LlamaCtx};
pub use module::{Chain, Module};
pub use moe::{ExpertFfn, ExpertFfnCtx, MoeFfn, MoeFfnCtx, MoeRouting};
pub use moe_llama::{MoeLlama, MoeLlamaCtx};
pub use rms_norm::RmsNorm;
pub use rope::Rope;
pub use swiglu::SwiGlu;
