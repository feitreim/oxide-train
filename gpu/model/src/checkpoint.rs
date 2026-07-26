//! Versioned checkpoints for the reference trainer.
//!
//! Parameters are stored in their master dtype and moments in fp32. Since v5
//! that means every matrix-shaped parameter is two bytes per element (#57);
//! norms and the router stay fp32, as they do in memory.
//!
//! The lm-head is stored with the padded vocabulary columns stripped: those
//! columns (and their moments) are zero by construction, so the payload does
//! not depend on the build's choice of `VP`.

use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use cuda_core::CudaStream;
use optim::{AdamWConfig, AuxLossSchedule};
use tensor_core::{Rank2, Shape, bf16};

use super::tensor_device::{GpuBf16Tensor, GpuTensor};
use super::{GpuBf16Head, GpuDense, GpuDenseAdamW};

const MAGIC: &[u8; 8] = b"RTCKPT01";
/// v5 stores bf16 masters. v4 (fp32 masters) is rejected by the version check
/// rather than converted: the fp32 bits it carries are precision this build
/// cannot represent, and silently rounding a resume is worse than refusing it.
const VERSION: u32 = 5;
const CONFIG_FLOATS: usize = 7;

pub struct LoadedCheckpoint<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
    const L: usize,
> {
    pub model: GpuDense<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>,
    pub optimizer: GpuDenseAdamW<VOCAB, VP, D, FF, E>,
    pub next_batch: u64,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_f32(writer: &mut impl Write, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn write_tensor<S: Shape>(
    writer: &mut impl Write,
    tensor: &GpuTensor<f32, S>,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    let host = tensor.to_host(stream)?;
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), host.len() * 4) };
    writer.write_all(bytes)?;
    Ok(())
}

fn read_tensor<S: Shape>(
    reader: &mut impl Read,
    stream: &CudaStream,
) -> Result<GpuTensor<f32, S>, Box<dyn Error>> {
    let mut host = vec![0.0f32; S::NUM_ELEMENTS];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), host.len() * 4) };
    reader.read_exact(bytes)?;
    Ok(GpuTensor::from_host(stream, &host)?)
}

fn write_master<S: Shape>(
    writer: &mut impl Write,
    tensor: &GpuBf16Tensor<S>,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    for value in tensor.to_f32_host(stream)? {
        writer.write_all(&bf16::from_f32(value).to_bits().to_le_bytes())?;
    }
    Ok(())
}

/// Refill a master in place. The tensor is never replaced: since #58 the
/// forward and backward TMA descriptors are encoded against the master's own
/// device address.
fn read_master_into<S: Shape>(
    reader: &mut impl Read,
    tensor: &mut GpuBf16Tensor<S>,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    let mut bits = vec![0u16; S::NUM_ELEMENTS];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(bits.as_mut_ptr().cast::<u8>(), bits.len() * 2) };
    reader.read_exact(bytes)?;
    let values: Vec<f32> = bits.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
    Ok(tensor.load_f32_host(stream, &values)?)
}

/// Write a padded `[D, VP]` fp32 head tensor (a moment) as its first `vocab`
/// columns.
fn write_head_tensor<const D: usize, const VP: usize>(
    writer: &mut impl Write,
    tensor: &GpuTensor<f32, Rank2<D, VP>>,
    vocab: usize,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    let host = tensor.to_host(stream)?;
    for row in 0..D {
        let columns = &host[row * VP..row * VP + vocab];
        let bytes =
            unsafe { std::slice::from_raw_parts(columns.as_ptr().cast::<u8>(), columns.len() * 4) };
        writer.write_all(bytes)?;
    }
    Ok(())
}

/// Read `[D, vocab]` fp32 head moments back into padded `[D, VP]` form; the
/// padded columns are zero.
fn read_head_values<const D: usize, const VP: usize>(
    reader: &mut impl Read,
    vocab: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut padded = vec![0.0f32; D * VP];
    for row in 0..D {
        let columns = &mut padded[row * VP..row * VP + vocab];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(columns.as_mut_ptr().cast::<u8>(), columns.len() * 4)
        };
        reader.read_exact(bytes)?;
    }
    Ok(padded)
}

/// [`write_head_tensor`] for the bf16 head master.
fn write_head_master<const D: usize, const VP: usize>(
    writer: &mut impl Write,
    tensor: &GpuBf16Tensor<Rank2<D, VP>>,
    vocab: usize,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    let host = tensor.to_f32_host(stream)?;
    for row in 0..D {
        for &value in &host[row * VP..row * VP + vocab] {
            writer.write_all(&bf16::from_f32(value).to_bits().to_le_bytes())?;
        }
    }
    Ok(())
}

/// Read `[D, vocab]` bf16 head master values back into padded `[D, VP]` f32
/// form; the padded columns are zero.
fn read_head_master_values<const D: usize, const VP: usize>(
    reader: &mut impl Read,
    vocab: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut padded = vec![0.0f32; D * VP];
    let mut bits = vec![0u16; vocab];
    for row in 0..D {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(bits.as_mut_ptr().cast::<u8>(), bits.len() * 2)
        };
        reader.read_exact(bytes)?;
        for (slot, &b) in padded[row * VP..row * VP + vocab].iter_mut().zip(&bits) {
            *slot = bf16::from_bits(b).to_f32();
        }
    }
    Ok(padded)
}

fn write_config(
    writer: &mut impl Write,
    config: AdamWConfig,
    aux_schedule: AuxLossSchedule,
) -> io::Result<()> {
    for value in [
        config.learning_rate,
        config.beta1,
        config.beta2,
        config.epsilon,
        config.weight_decay,
        aux_schedule.base_coefficient,
        aux_schedule.decay_horizon,
    ] {
        write_f32(writer, value)?;
    }
    Ok(())
}

fn read_config(reader: &mut impl Read) -> io::Result<(AdamWConfig, AuxLossSchedule)> {
    let mut values = [0.0; CONFIG_FLOATS];
    for value in &mut values {
        *value = read_f32(reader)?;
    }
    let config = AdamWConfig {
        learning_rate: values[0],
        beta1: values[1],
        beta2: values[2],
        epsilon: values[3],
        weight_decay: values[4],
    };
    if !config.is_valid() {
        return Err(invalid("invalid AdamW checkpoint config"));
    }
    let aux_schedule = AuxLossSchedule {
        base_coefficient: values[5],
        decay_horizon: values[6],
    };
    if !aux_schedule.is_valid() {
        return Err(invalid("invalid auxiliary-loss checkpoint schedule"));
    }
    Ok((config, aux_schedule))
}

#[allow(clippy::too_many_arguments)]
pub fn save<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
    const L: usize,
>(
    path: impl AsRef<Path>,
    model: &GpuDense<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>,
    optimizer: &GpuDenseAdamW<VOCAB, VP, D, FF, E>,
    next_batch: u64,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    const { assert!(cfg!(target_endian = "little")) };
    assert_eq!(model.blocks.len(), L);
    assert_eq!(optimizer.blocks.len(), L);
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(MAGIC)?;
    write_u32(&mut writer, VERSION)?;
    for dimension in [N, T, VOCAB, D, H, HD, FF, E, K, C, L] {
        write_u64(&mut writer, dimension as u64)?;
    }
    write_u64(&mut writer, optimizer.step())?;
    write_u64(&mut writer, next_batch)?;
    write_config(&mut writer, optimizer.config(), optimizer.aux_schedule())?;

    macro_rules! write_master_parameter {
        ($parameter:expr, $moments:expr) => {
            write_master(&mut writer, $parameter, stream)?;
            write_tensor(&mut writer, &$moments.first, stream)?;
            write_tensor(&mut writer, &$moments.second, stream)?;
        };
    }
    macro_rules! write_fp32_parameter {
        ($parameter:expr, $moments:expr) => {
            write_tensor(&mut writer, $parameter, stream)?;
            write_tensor(&mut writer, &$moments.first, stream)?;
            write_tensor(&mut writer, &$moments.second, stream)?;
        };
    }
    write_master_parameter!(&model.embedding.w, optimizer.embedding);
    for (block, moments) in model.blocks.iter().zip(optimizer.blocks.iter()) {
        write_fp32_parameter!(&block.attention_norm.w, moments.attention_norm);
        write_master_parameter!(&block.qkv_proj.w, moments.qkv_proj);
        write_master_parameter!(&block.o_proj.w, moments.o_proj);
        write_fp32_parameter!(&block.ffn_norm.w, moments.ffn_norm);
        write_fp32_parameter!(&block.router, moments.router);
        write_master_parameter!(&block.experts.gate_up, moments.expert_gate_up);
        write_master_parameter!(&block.experts.down, moments.expert_down);
    }
    write_fp32_parameter!(&model.final_norm.w, optimizer.final_norm);
    write_head_master::<D, VP>(&mut writer, &model.lm_head.master, VOCAB, stream)?;
    write_head_tensor::<D, VP>(&mut writer, &optimizer.lm_head.first, VOCAB, stream)?;
    write_head_tensor::<D, VP>(&mut writer, &optimizer.lm_head.second, VOCAB, stream)?;

    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(&temporary, path)?;
    Ok(())
}

pub fn load<
    const N: usize,
    const NP: usize,
    const T: usize,
    const VOCAB: usize,
    const VP: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
    const E: usize,
    const K: usize,
    const C: usize,
    const L: usize,
>(
    path: impl AsRef<Path>,
    stream: &CudaStream,
    tensor: &super::tensor_kernels::LoadedModule,
) -> Result<LoadedCheckpoint<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>, Box<dyn Error>> {
    const { assert!(cfg!(target_endian = "little")) };
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0; MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("bad checkpoint magic").into());
    }
    let version = read_u32(&mut reader)?;
    if version != VERSION {
        return Err(invalid(format!("unsupported checkpoint version {version}")).into());
    }
    let expected = [N, T, VOCAB, D, H, HD, FF, E, K, C, L];
    for (name, expected) in ["N", "T", "VOCAB", "D", "H", "HD", "FF", "E", "K", "C", "L"]
        .into_iter()
        .zip(expected)
    {
        let actual = read_u64(&mut reader)?;
        if actual != expected as u64 {
            return Err(invalid(format!(
                "checkpoint {name} mismatch: file={actual}, binary={expected}"
            ))
            .into());
        }
    }
    let step = read_u64(&mut reader)?;
    let next_batch = read_u64(&mut reader)?;
    let (config, aux_schedule) = read_config(&mut reader)?;

    let mut model = GpuDense::<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C, L>::initialized(
        stream,
        0,
        aux_schedule.base_coefficient,
    )?;
    let mut optimizer = GpuDenseAdamW::new(stream, config, aux_schedule, L)?;

    macro_rules! read_master_parameter {
        ($parameter:expr, $moments:expr) => {
            read_master_into(&mut reader, $parameter, stream)?;
            $moments.first = read_tensor(&mut reader, stream)?;
            $moments.second = read_tensor(&mut reader, stream)?;
        };
    }
    macro_rules! read_fp32_parameter {
        ($parameter:expr, $moments:expr) => {
            *$parameter = read_tensor(&mut reader, stream)?;
            $moments.first = read_tensor(&mut reader, stream)?;
            $moments.second = read_tensor(&mut reader, stream)?;
        };
    }
    read_master_parameter!(&mut model.embedding.w, optimizer.embedding);
    for (block, moments) in model.blocks.iter_mut().zip(optimizer.blocks.iter_mut()) {
        read_fp32_parameter!(&mut block.attention_norm.w, moments.attention_norm);
        read_master_parameter!(&mut block.qkv_proj.w, moments.qkv_proj);
        read_master_parameter!(&mut block.o_proj.w, moments.o_proj);
        read_fp32_parameter!(&mut block.ffn_norm.w, moments.ffn_norm);
        read_fp32_parameter!(&mut block.router, moments.router);
        read_master_parameter!(&mut block.experts.gate_up, moments.expert_gate_up);
        read_master_parameter!(&mut block.experts.down, moments.expert_down);
    }
    read_fp32_parameter!(&mut model.final_norm.w, optimizer.final_norm);
    model.sync_compute(stream, tensor)?;
    model.lm_head = GpuBf16Head::from_master_values(
        stream,
        &read_head_master_values::<D, VP>(&mut reader, VOCAB)?,
    )?;
    optimizer.lm_head.first =
        GpuTensor::from_host(stream, &read_head_values::<D, VP>(&mut reader, VOCAB)?)?;
    optimizer.lm_head.second =
        GpuTensor::from_host(stream, &read_head_values::<D, VP>(&mut reader, VOCAB)?)?;
    optimizer.restore_step(step);

    let mut trailing = [0];
    if reader.read(&mut trailing)? != 0 {
        return Err(invalid("checkpoint has trailing data").into());
    }
    Ok(LoadedCheckpoint {
        model,
        optimizer,
        next_batch,
    })
}
