//! Versioned checkpoints for the reference trainer.
//!
//! The lm-head is stored as its fp32 master weights with the padded
//! vocabulary columns stripped: those columns (and their moments) are zero by
//! construction, so the payload is identical to the pre-bf16 format and does
//! not depend on the build's choice of `VP`.

use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use cuda_core::CudaStream;
use nn::MoeLlama;
use optim::{AdamWConfig, AuxLossSchedule};
use tensor_core::{Rank2, Shape};

use super::tensor_device::GpuTensor;
use super::{GpuBf16Head, GpuLlama, GpuLlamaAdamW};

const MAGIC: &[u8; 8] = b"RTCKPT01";
const VERSION: u32 = 3;
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
> {
    pub model: GpuLlama<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C>,
    pub optimizer: GpuLlamaAdamW<VOCAB, VP, D, FF, E>,
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

/// Write a padded `[D, VP]` head tensor as its first `vocab` columns.
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

/// Read `[D, vocab]` head values back into padded `[D, VP]` form; the padded
/// columns are zero.
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
>(
    path: impl AsRef<Path>,
    model: &GpuLlama<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C>,
    optimizer: &GpuLlamaAdamW<VOCAB, VP, D, FF, E>,
    next_batch: u64,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    const { assert!(cfg!(target_endian = "little")) };
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(MAGIC)?;
    write_u32(&mut writer, VERSION)?;
    for dimension in [N, T, VOCAB, D, H, HD, FF, E, K, C] {
        write_u64(&mut writer, dimension as u64)?;
    }
    write_u64(&mut writer, optimizer.step())?;
    write_u64(&mut writer, next_batch)?;
    write_config(&mut writer, optimizer.config(), optimizer.aux_schedule())?;

    macro_rules! write_parameter {
        ($field:ident) => {
            write_tensor(&mut writer, &model.$field.w, stream)?;
            write_tensor(&mut writer, &optimizer.$field.first, stream)?;
            write_tensor(&mut writer, &optimizer.$field.second, stream)?;
        };
    }
    write_parameter!(embedding);
    write_parameter!(attention_norm);
    write_parameter!(qkv_proj);
    write_parameter!(o_proj);
    write_parameter!(ffn_norm);
    write_tensor(&mut writer, &model.router, stream)?;
    write_tensor(&mut writer, &optimizer.router.first, stream)?;
    write_tensor(&mut writer, &optimizer.router.second, stream)?;
    write_tensor(&mut writer, &model.experts.gate_up, stream)?;
    write_tensor(&mut writer, &optimizer.expert_gate_up.first, stream)?;
    write_tensor(&mut writer, &optimizer.expert_gate_up.second, stream)?;
    write_tensor(&mut writer, &model.experts.down, stream)?;
    write_tensor(&mut writer, &optimizer.expert_down.first, stream)?;
    write_tensor(&mut writer, &optimizer.expert_down.second, stream)?;
    write_parameter!(final_norm);
    write_head_tensor::<D, VP>(&mut writer, &model.lm_head.master, VOCAB, stream)?;
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
>(
    path: impl AsRef<Path>,
    stream: &CudaStream,
    tensor: &super::tensor_kernels::LoadedModule,
) -> Result<LoadedCheckpoint<N, NP, T, VOCAB, VP, D, H, HD, FF, E, K, C>, Box<dyn Error>> {
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
    let expected = [N, T, VOCAB, D, H, HD, FF, E, K, C];
    for (name, expected) in ["N", "T", "VOCAB", "D", "H", "HD", "FF", "E", "K", "C"]
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

    let cpu = MoeLlama::<N, T, VOCAB, D, H, HD, FF, E, K, C>::new(0, aux_schedule.base_coefficient);
    let mut model = GpuLlama::from_cpu(stream, &cpu)?;
    let mut optimizer = GpuLlamaAdamW::new(stream, config, aux_schedule)?;

    macro_rules! read_parameter {
        ($field:ident) => {
            model.$field.w = read_tensor(&mut reader, stream)?;
            optimizer.$field.first = read_tensor(&mut reader, stream)?;
            optimizer.$field.second = read_tensor(&mut reader, stream)?;
        };
    }
    read_parameter!(embedding);
    read_parameter!(attention_norm);
    read_parameter!(qkv_proj);
    read_parameter!(o_proj);
    read_parameter!(ffn_norm);
    model.router = read_tensor(&mut reader, stream)?;
    optimizer.router.first = read_tensor(&mut reader, stream)?;
    optimizer.router.second = read_tensor(&mut reader, stream)?;
    model.experts.gate_up = read_tensor(&mut reader, stream)?;
    optimizer.expert_gate_up.first = read_tensor(&mut reader, stream)?;
    optimizer.expert_gate_up.second = read_tensor(&mut reader, stream)?;
    model.experts.down = read_tensor(&mut reader, stream)?;
    optimizer.expert_down.first = read_tensor(&mut reader, stream)?;
    optimizer.expert_down.second = read_tensor(&mut reader, stream)?;
    read_parameter!(final_norm);
    model.sync_compute(stream, tensor)?;
    model.lm_head =
        GpuBf16Head::from_master_values(stream, &read_head_values::<D, VP>(&mut reader, VOCAB)?)?;
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
