//! Binary token shards: a tiny header + flat little-endian `u16` token ids.
//!
//! Layout (all little-endian):
//! ```text
//! offset  size  field
//! 0       4     magic   = "TOK1" (0x314B4F54 when read as little-endian u32)
//! 4       4     version = 1
//! 8       4     dtype   = 1 (u16)
//! 12      4     reserved (0)
//! 16      8     num_tokens
//! 24      2*n   tokens
//! ```
//! The 24-byte header keeps the token array 2-byte aligned, so [`TokenFile`]
//! can hand out a zero-copy `&[u16]` view of the mmap. Little-endian is
//! assumed on read (checked at compile time) — every target we care about is.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

pub const MAGIC: u32 = u32::from_le_bytes(*b"TOK1");
pub const VERSION: u32 = 1;
pub const DTYPE_U16: u32 = 1;
pub const HEADER_BYTES: usize = 24;

const _LITTLE_ENDIAN_ONLY: () = assert!(cfg!(target_endian = "little"));

fn encode_header(num_tokens: u64) -> [u8; HEADER_BYTES] {
    let mut h = [0u8; HEADER_BYTES];
    h[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    h[4..8].copy_from_slice(&VERSION.to_le_bytes());
    h[8..12].copy_from_slice(&DTYPE_U16.to_le_bytes());
    h[16..24].copy_from_slice(&num_tokens.to_le_bytes());
    h
}

/// Streaming writer for one shard file. Created by [`ShardSetWriter`] or
/// directly for tests; finalizes the header on [`finish`](Self::finish).
pub struct ShardWriter {
    path: PathBuf,
    out: BufWriter<File>,
    num_tokens: u64,
}

impl ShardWriter {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_owned();
        let file = File::create(&path)
            .with_context(|| format!("create shard {}", path.display()))?;
        let mut out = BufWriter::new(file);
        // Placeholder header; rewritten with the real count in finish().
        out.write_all(&encode_header(0))?;
        Ok(Self { path, out, num_tokens: 0 })
    }

    pub fn write_tokens(&mut self, tokens: &[u16]) -> Result<()> {
        // Safe cast: &[u16] -> &[u8] view for a bulk little-endian write.
        let bytes = unsafe {
            std::slice::from_raw_parts(tokens.as_ptr().cast::<u8>(), tokens.len() * 2)
        };
        self.out.write_all(bytes)?;
        self.num_tokens += tokens.len() as u64;
        Ok(())
    }

    pub fn num_tokens(&self) -> u64 {
        self.num_tokens
    }

    /// Flush, rewrite the header with the final token count, and close.
    pub fn finish(mut self) -> Result<u64> {
        self.out.flush()?;
        let mut file = self.out.into_inner()?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&encode_header(self.num_tokens))?;
        file.sync_all()
            .with_context(|| format!("finalize shard {}", self.path.display()))?;
        Ok(self.num_tokens)
    }
}

/// A finished shard, mmap'd read-only. `tokens()` is a zero-copy `&[u16]`.
pub struct TokenFile {
    mmap: Mmap,
    num_tokens: usize,
}

impl TokenFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file =
            File::open(path).with_context(|| format!("open shard {}", path.display()))?;
        let mut header = [0u8; HEADER_BYTES];
        file.read_exact(&mut header)
            .with_context(|| format!("read shard header {}", path.display()))?;

        let field = |i: usize| u32::from_le_bytes(header[i..i + 4].try_into().unwrap());
        ensure!(field(0) == MAGIC, "bad magic in {}", path.display());
        ensure!(field(4) == VERSION, "unsupported shard version in {}", path.display());
        ensure!(field(8) == DTYPE_U16, "unsupported dtype in {}", path.display());
        let num_tokens = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;

        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap shard {}", path.display()))?;
        if mmap.len() < HEADER_BYTES + num_tokens * 2 {
            bail!("shard {} truncated: header claims {num_tokens} tokens", path.display());
        }
        Ok(Self { mmap, num_tokens })
    }

    pub fn tokens(&self) -> &[u16] {
        let bytes = &self.mmap[HEADER_BYTES..HEADER_BYTES + self.num_tokens * 2];
        // Alignment holds: mmaps are page-aligned and HEADER_BYTES is even.
        debug_assert_eq!(bytes.as_ptr() as usize % align_of::<u16>(), 0);
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), self.num_tokens) }
    }

    pub fn len(&self) -> usize {
        self.num_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }
}

/// Rolls output across `wiki-00000.tok, wiki-00001.tok, ...` capped at
/// `tokens_per_shard`, so preprocessing never buffers a shard in memory and
/// downstream code can pick shards up individually.
pub struct ShardSetWriter {
    dir: PathBuf,
    prefix: String,
    tokens_per_shard: u64,
    current: Option<ShardWriter>,
    shard_index: usize,
    total_tokens: u64,
    finished: Vec<PathBuf>,
}

impl ShardSetWriter {
    pub fn new(dir: impl AsRef<Path>, prefix: &str, tokens_per_shard: u64) -> Result<Self> {
        ensure!(tokens_per_shard > 0);
        std::fs::create_dir_all(dir.as_ref())?;
        Ok(Self {
            dir: dir.as_ref().to_owned(),
            prefix: prefix.to_owned(),
            tokens_per_shard,
            current: None,
            shard_index: 0,
            total_tokens: 0,
            finished: Vec::new(),
        })
    }

    fn shard_path(&self, index: usize) -> PathBuf {
        self.dir.join(format!("{}-{index:05}.tok", self.prefix))
    }

    /// Append one document's tokens, splitting across shard boundaries as
    /// needed (documents may straddle shards; the stream is what matters).
    pub fn write_tokens(&mut self, mut tokens: &[u16]) -> Result<()> {
        while !tokens.is_empty() {
            if self.current.is_none() {
                let path = self.shard_path(self.shard_index);
                self.current = Some(ShardWriter::create(&path)?);
            }
            let writer = self.current.as_mut().unwrap();
            let room = (self.tokens_per_shard - writer.num_tokens()) as usize;
            let take = tokens.len().min(room);
            writer.write_tokens(&tokens[..take])?;
            self.total_tokens += take as u64;
            tokens = &tokens[take..];
            if writer.num_tokens() == self.tokens_per_shard {
                self.roll()?;
            }
        }
        Ok(())
    }

    fn roll(&mut self) -> Result<()> {
        if let Some(writer) = self.current.take() {
            let path = self.shard_path(self.shard_index);
            writer.finish()?;
            self.finished.push(path);
            self.shard_index += 1;
        }
        Ok(())
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Finalize the trailing partial shard and return all shard paths.
    pub fn finish(mut self) -> Result<Vec<PathBuf>> {
        self.roll()?;
        Ok(self.finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_roundtrip() {
        let dir = std::env::temp_dir().join("rust-trainer-shard-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.tok");

        let tokens: Vec<u16> = (0..10_000u32).map(|i| (i % 50_257) as u16).collect();
        let mut w = ShardWriter::create(&path).unwrap();
        w.write_tokens(&tokens[..7000]).unwrap();
        w.write_tokens(&tokens[7000..]).unwrap();
        assert_eq!(w.finish().unwrap(), 10_000);
        assert_eq!(&std::fs::read(&path).unwrap()[..4], b"TOK1");

        let f = TokenFile::open(&path).unwrap();
        assert_eq!(f.tokens(), &tokens[..]);
    }

    #[test]
    fn legacy_reversed_magic_is_rejected() {
        let path = std::env::temp_dir().join("rust-trainer-legacy-magic-test.tok");
        let mut w = ShardWriter::create(&path).unwrap();
        w.write_tokens(&[1, 2, 3]).unwrap();
        w.finish().unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[..4].copy_from_slice(&0x544F_4B31u32.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        assert!(TokenFile::open(path).is_err());
    }

    #[test]
    fn shard_set_rolls_at_cap() {
        let dir = std::env::temp_dir().join("rust-trainer-shardset-test");
        let _ = std::fs::remove_dir_all(&dir);

        let mut w = ShardSetWriter::new(&dir, "t", 1000).unwrap();
        // 3 docs of 700 tokens: 2100 tokens -> shards of 1000/1000/100.
        for d in 0..3u16 {
            let doc = vec![d; 700];
            w.write_tokens(&doc).unwrap();
        }
        assert_eq!(w.total_tokens(), 2100);
        let shards = w.finish().unwrap();
        assert_eq!(shards.len(), 3);

        let lens: Vec<usize> = shards.iter().map(|p| TokenFile::open(p).unwrap().len()).collect();
        assert_eq!(lens, [1000, 1000, 100]);

        // Stream order is preserved across the roll boundary.
        let all: Vec<u16> = shards
            .iter()
            .flat_map(|p| TokenFile::open(p).unwrap().tokens().to_vec())
            .collect();
        let expected: Vec<u16> = (0..3u16).flat_map(|d| vec![d; 700]).collect();
        assert_eq!(all, expected);
    }
}
