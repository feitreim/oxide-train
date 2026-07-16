//! Shared helpers for the GPU kernel host binaries: CUDA-event timing and
//! reproducible random inputs. (Adapted from cuda-learning's bench-util.)

use std::sync::Arc;

use cuda_core::CudaStream;

/// Re-export: `n` uniform-random `f32`s in `[-1, 1)` from a deterministic
/// PRNG — the *same* generator `CpuTensor::uniform` uses, so CPU/GPU parity
/// tests agree on inputs bit-for-bit.
pub use tensor_core::rng::uniform_vec;

/// Average per-iteration GPU time in milliseconds, measured with CUDA events.
///
/// Runs `warmup` untimed launches to settle clocks/caches, then times `iters`
/// launches between two recorded events (device-side timing, not wall clock).
pub fn time_gpu_iters<F>(
    stream: &Arc<CudaStream>,
    warmup: usize,
    iters: usize,
    mut launch: F,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..warmup {
        launch()?;
    }
    stream.synchronize()?;

    let flags = cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
    let start = stream.record_event(Some(flags))?;
    for _ in 0..iters {
        launch()?;
    }
    let end = stream.record_event(Some(flags))?;
    Ok(start.elapsed_ms(&end)? as f64 / iters as f64)
}
