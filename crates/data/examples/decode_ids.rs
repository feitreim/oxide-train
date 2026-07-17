//! Decode whitespace-separated r50k token ids from stdin.
//!
//! Companion to the GPU sampler binary, which prints ids because the GPU
//! crates build without the tokenizer's `offline` feature:
//!
//!     grep 'ids:' out.log | cut -d: -f2 | cargo run -p data --example decode_ids

use std::io::Read;

use data::Tokenizer;

fn main() -> anyhow::Result<()> {
    let tokenizer = Tokenizer::r50k()?;
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    for line in input.lines().filter(|l| !l.trim().is_empty()) {
        let tokens: Vec<u16> = line
            .split_whitespace()
            .map(|t| t.parse())
            .collect::<Result<_, _>>()?;
        println!("{}", tokenizer.decode(&tokens)?);
        println!("---");
    }
    Ok(())
}
