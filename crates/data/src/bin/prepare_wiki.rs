//! Offline preprocessing: `wikimedia/wikipedia` (`20231101.en`) -> `u16`
//! token shards.
//!
//! Downloads the parquet files from the HF hub (cached under
//! `~/.cache/huggingface`), tokenizes each article with `r50k_base` in
//! parallel (rayon), and appends `article tokens + <|endoftext|>` to rolling
//! shards. The first `--val-tokens` tokens go to a `wiki-val` shard; the rest
//! to `wiki-train-*` shards. Deterministic: file order and article order are
//! whatever the dataset ships, no shuffling here (the loader owns sampling).
//!
//!     cargo run --release -p data --bin prepare_wiki -- --out data/wiki
//!     cargo run --release -p data --bin prepare_wiki -- --limit-files 1  # smoke
//!     cargo run --release -p data --bin prepare_wiki -- --limit-files 1 --limit-articles 1000

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use data::shard::ShardSetWriter;
use data::tokenizer::Tokenizer;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use rayon::prelude::*;

const DATASET: &str = "wikimedia/wikipedia";
const DUMP: &str = "20231101.en";
/// Articles tokenized per rayon batch: bounds peak memory, keeps cores busy.
const DOC_BATCH: usize = 2048;

struct Args {
    out: PathBuf,
    tokens_per_shard: u64,
    val_tokens: u64,
    limit_files: Option<usize>,
    limit_articles: Option<u64>,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        out: PathBuf::from("data/wiki"),
        tokens_per_shard: 250_000_000, // 500MB per shard at 2 bytes/token
        val_tokens: 10_000_000,
        limit_files: None,
        limit_articles: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().context("missing value for flag");
        match flag.as_str() {
            "--out" => args.out = PathBuf::from(val()?),
            "--tokens-per-shard" => args.tokens_per_shard = val()?.parse()?,
            "--val-tokens" => args.val_tokens = val()?.parse()?,
            "--limit-files" => args.limit_files = Some(val()?.parse()?),
            "--limit-articles" => args.limit_articles = Some(val()?.parse()?),
            other => anyhow::bail!("unknown flag {other}"),
        }
    }
    Ok(args)
}

/// All `text` column values from one parquet file.
fn read_texts(path: &std::path::Path) -> Result<Vec<String>> {
    let reader = SerializedFileReader::new(File::open(path)?)
        .with_context(|| format!("open parquet {}", path.display()))?;
    let mut texts = Vec::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        for (name, field) in row.get_column_iter() {
            if name == "text"
                && let Field::Str(s) = field
            {
                texts.push(s.clone());
            }
        }
    }
    Ok(texts)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let started = Instant::now();

    let tokenizer = Tokenizer::r50k()?;

    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.repo(Repo::new(DATASET.to_owned(), RepoType::Dataset));

    let mut files: Vec<String> = repo
        .info()
        .context("fetch dataset file listing")?
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|f| f.starts_with(&format!("{DUMP}/")) && f.ends_with(".parquet"))
        .collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no parquet files found for {DUMP}");
    if let Some(limit) = args.limit_files {
        files.truncate(limit);
    }
    println!("{} parquet files to process", files.len());

    let mut val = ShardSetWriter::new(&args.out, "wiki-val", args.val_tokens)?;
    let mut train = ShardSetWriter::new(&args.out, "wiki-train", args.tokens_per_shard)?;
    let mut articles = 0u64;

    for (i, name) in files.iter().enumerate() {
        let local = repo.get(name).with_context(|| format!("download {name}"))?;
        let mut texts = read_texts(&local)?;
        if let Some(limit) = args.limit_articles {
            let remaining = limit.saturating_sub(articles);
            texts.truncate(usize::try_from(remaining).unwrap_or(usize::MAX));
        }
        articles += texts.len() as u64;

        for chunk in texts.chunks(DOC_BATCH) {
            let docs: Vec<Vec<u16>> =
                chunk.par_iter().map(|t| tokenizer.encode_doc(t)).collect();
            for doc in &docs {
                // Fill the val shard first, then stream everything to train.
                if val.total_tokens() < args.val_tokens {
                    val.write_tokens(doc)?;
                } else {
                    train.write_tokens(doc)?;
                }
            }
        }

        let tokens = val.total_tokens() + train.total_tokens();
        let mins = started.elapsed().as_secs_f64() / 60.0;
        println!(
            "[{}/{}] {name}: {articles} articles, {tokens} tokens, {mins:.1} min",
            i + 1,
            files.len(),
        );
        if args.limit_articles.is_some_and(|limit| articles >= limit) {
            break;
        }
    }

    let val_shards = val.finish()?;
    let train_shards = train.finish()?;
    println!(
        "done: {articles} articles -> {} val shard(s) + {} train shard(s) in {}",
        val_shards.len(),
        train_shards.len(),
        args.out.display(),
    );
    Ok(())
}
