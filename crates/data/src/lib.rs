//! Data pipeline: Wikipedia -> tiktoken `r50k_base` -> flat `u16` token
//! shards -> statically-shaped `[B, T]` batches.
//!
//! Split into an *offline* half and a *training-loop* half:
//!
//! - Offline (`prepare-wiki` binary): download `wikimedia/wikipedia`
//!   `20231101.en` parquet from the HF hub, tokenize each article with
//!   `r50k_base` (+ `<|endoftext|>` separators), and append everything to
//!   fixed-size binary shards ([`shard`]). Run once, keep forever.
//! - Training loop ([`TokenFile`] + [`Batches`]): mmap a shard (zero-copy)
//!   and slice deterministic `(inputs, targets)` batches out of it as
//!   `CpuTensor<u16, Rank2<B, T>>` — shapes are const generics like
//!   everything else, so the loader can't hand the model a bad batch.
//!
//! r50k_base has 50,257 tokens, so ids fit `u16` and shards are 2
//! bytes/token (~9GB for all of English Wikipedia).

pub mod batch;
pub mod shard;
pub mod tokenizer;

pub use batch::Batches;
pub use shard::{ShardWriter, TokenFile};
pub use tokenizer::Tokenizer;
