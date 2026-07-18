//! tiktoken `r50k_base` (GPT-2 vocab, 50,257 tokens) wrapped for shard
//! production: document-level encode with an `<|endoftext|>` separator, ids
//! narrowed to `u16` (which r50k guarantees fits).

use anyhow::{Context, Result};
use tiktoken_rs::CoreBPE;

pub const EOT_TOKEN: u16 = 50256; // <|endoftext|> in r50k_base
pub const VOCAB_SIZE: usize = 50257;

pub struct Tokenizer {
    bpe: CoreBPE,
}

impl Tokenizer {
    /// Loads the embedded r50k_base vocab (no network).
    pub fn r50k() -> Result<Self> {
        let bpe = tiktoken_rs::r50k_base().context("load r50k_base")?;
        let this = Self { bpe };
        // The EOT constant is load-bearing for the shard format; verify it
        // against the actual vocab once at construction.
        let eot = this.bpe.encode_with_special_tokens("<|endoftext|>");
        anyhow::ensure!(
            eot.len() == 1 && eot[0] as usize == EOT_TOKEN as usize,
            "r50k_base <|endoftext|> id changed?"
        );
        Ok(this)
    }

    /// Encode one document as `tokens... EOT`. The trailing separator is what
    /// keeps unrelated articles from attending-across in a packed stream.
    pub fn encode_doc(&self, text: &str) -> Vec<u16> {
        let toks = self.bpe.encode_ordinary(text);
        let mut out = Vec::with_capacity(toks.len() + 1);
        out.extend(
            toks.into_iter()
                .map(|t| u16::try_from(t).expect("r50k token id exceeds u16")),
        );
        out.push(EOT_TOKEN);
        out
    }

    /// Decode for spot checks and sampling output.
    pub fn decode(&self, tokens: &[u16]) -> Result<String> {
        let wide: Vec<u32> = tokens.iter().map(|&t| u32::from(t)).collect();
        self.bpe.decode(&wide).context("decode")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_eot() {
        let tok = Tokenizer::r50k().unwrap();
        let text = "The quick brown fox jumps over the lazy dog. \u{1F980} caf\u{e9}";
        let ids = tok.encode_doc(text);
        assert_eq!(*ids.last().unwrap(), EOT_TOKEN);
        assert!(ids.iter().all(|&t| (t as usize) < VOCAB_SIZE));
        let back = tok.decode(&ids[..ids.len() - 1]).unwrap();
        assert_eq!(back, text);
    }
}
