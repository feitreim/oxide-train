//! Versioned checkpoints for the fp32 reference trainer.

use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use cuda_core::CudaStream;
use nn::Llama;
use optim::AdamWConfig;
use tensor_core::Shape;

use super::tensor_device::GpuTensor;
use super::{GpuLlama, GpuLlamaAdamW};

const MAGIC: &[u8; 8] = b"RTCKPT01";
const VERSION: u32 = 2;
const CONFIG_FLOATS: usize = 5;

pub struct LoadedCheckpoint<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
> {
    pub model: GpuLlama<N, T, VOCAB, D, H, HD, FF>,
    pub optimizer: GpuLlamaAdamW<VOCAB, D, FF>,
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

fn write_config(writer: &mut impl Write, config: AdamWConfig) -> io::Result<()> {
    for value in [
        config.learning_rate,
        config.beta1,
        config.beta2,
        config.epsilon,
        config.weight_decay,
    ] {
        write_f32(writer, value)?;
    }
    Ok(())
}

fn read_config(reader: &mut impl Read) -> io::Result<AdamWConfig> {
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
    config.validate();
    Ok(config)
}

#[allow(clippy::too_many_arguments)]
pub fn save<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
>(
    path: impl AsRef<Path>,
    model: &GpuLlama<N, T, VOCAB, D, H, HD, FF>,
    optimizer: &GpuLlamaAdamW<VOCAB, D, FF>,
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
    for dimension in [N, T, VOCAB, D, H, HD, FF] {
        write_u64(&mut writer, dimension as u64)?;
    }
    write_u64(&mut writer, optimizer.step())?;
    write_u64(&mut writer, next_batch)?;
    write_config(&mut writer, optimizer.config())?;

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
    write_parameter!(gate_up_proj);
    write_parameter!(down_proj);
    write_parameter!(final_norm);
    write_parameter!(lm_head);

    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(&temporary, path)?;
    Ok(())
}

pub fn load<
    const N: usize,
    const T: usize,
    const VOCAB: usize,
    const D: usize,
    const H: usize,
    const HD: usize,
    const FF: usize,
>(
    path: impl AsRef<Path>,
    stream: &CudaStream,
) -> Result<LoadedCheckpoint<N, T, VOCAB, D, H, HD, FF>, Box<dyn Error>> {
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
    let expected = [N, T, VOCAB, D, H, HD, FF];
    for (name, expected) in ["N", "T", "VOCAB", "D", "H", "HD", "FF"]
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
    let config = read_config(&mut reader)?;

    let cpu = Llama::<N, T, VOCAB, D, H, HD, FF>::new(0);
    let mut model = GpuLlama::from_cpu(stream, &cpu)?;
    let mut optimizer = GpuLlamaAdamW::new(stream, config)?;

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
    read_parameter!(gate_up_proj);
    read_parameter!(down_proj);
    read_parameter!(final_norm);
    read_parameter!(lm_head);
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
