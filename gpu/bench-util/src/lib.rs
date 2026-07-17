//! Shared helpers for the GPU kernel host binaries: CUDA-event timing and
//! reproducible random inputs. (Adapted from cuda-learning's bench-util.)

use std::sync::Arc;

use std::fmt;

use cuda_core::{CudaEvent, CudaStream, DriverError};

/// Re-export: `n` uniform-random `f32`s in `[-1, 1)` from a deterministic
/// PRNG — the *same* generator `CpuTensor::uniform` uses, so CPU/GPU parity
/// tests agree on inputs bit-for-bit.
pub use tensor_core::rng::uniform_vec;

fn timing_event(stream: &CudaStream) -> Result<CudaEvent, DriverError> {
    let flags = cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
    stream.record_event(Some(flags))
}

/// A sink for named CUDA kernel timings.
///
/// Model code is generic over this trait, so normal execution uses
/// [`NoopProfiler`] without recording events while profiling runs use
/// [`StepProfiler`].
pub trait KernelProfiler {
    /// Launch `kernel` between two timing-enabled CUDA events.
    fn measure<T, F>(
        &mut self,
        stream: &CudaStream,
        name: &'static str,
        kernel: F,
    ) -> Result<T, DriverError>
    where
        F: FnOnce() -> Result<T, DriverError>;
}

/// Zero-overhead profiler used by correctness and training runs that are not
/// collecting a breakdown.
#[derive(Default)]
pub struct NoopProfiler;

impl KernelProfiler for NoopProfiler {
    #[inline]
    fn measure<T, F>(
        &mut self,
        _stream: &CudaStream,
        _name: &'static str,
        kernel: F,
    ) -> Result<T, DriverError>
    where
        F: FnOnce() -> Result<T, DriverError>,
    {
        kernel()
    }
}

struct PendingKernel {
    name: &'static str,
    start: CudaEvent,
    end: CudaEvent,
}

/// CUDA events collected around one kernel launch.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelTiming {
    pub name: &'static str,
    pub milliseconds: f64,
}

/// Device-side timing breakdown for one full training step.
#[derive(Clone, Debug, PartialEq)]
pub struct StepProfile {
    pub step_milliseconds: f64,
    pub kernels: Vec<KernelTiming>,
}

impl StepProfile {
    pub fn kernel_milliseconds(&self) -> f64 {
        self.kernels.iter().map(|kernel| kernel.milliseconds).sum()
    }

    /// Device work inside the step events that was not inside a measured
    /// kernel span, such as H2D copies, allocations, and zero fills.
    pub fn unattributed_milliseconds(&self) -> f64 {
        (self.step_milliseconds - self.kernel_milliseconds()).max(0.0)
    }
}

impl fmt::Display for StepProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let percent = |milliseconds: f64| {
            if self.step_milliseconds == 0.0 {
                0.0
            } else {
                100.0 * milliseconds / self.step_milliseconds
            }
        };

        writeln!(f, "GPU training-step profile (CUDA events)")?;
        writeln!(f, "{:<52} {:>10} {:>8}", "kernel", "ms", "% step")?;
        writeln!(f, "{:-<52} {:-<10} {:-<8}", "", "", "")?;
        for kernel in &self.kernels {
            writeln!(
                f,
                "{:<52} {:>10.4} {:>7.2}%",
                kernel.name,
                kernel.milliseconds,
                percent(kernel.milliseconds)
            )?;
        }
        let kernel_ms = self.kernel_milliseconds();
        let unattributed_ms = self.unattributed_milliseconds();
        writeln!(f, "{:-<52} {:-<10} {:-<8}", "", "", "")?;
        writeln!(
            f,
            "{:<52} {:>10.4} {:>7.2}%",
            "all kernels",
            kernel_ms,
            percent(kernel_ms)
        )?;
        writeln!(
            f,
            "{:<52} {:>10.4} {:>7.2}%",
            "unattributed (copies/allocations/gaps)",
            unattributed_ms,
            percent(unattributed_ms)
        )?;
        write!(
            f,
            "{:<52} {:>10.4} {:>7.2}%",
            "full step",
            self.step_milliseconds,
            percent(self.step_milliseconds)
        )
    }
}

/// Records a device-side timeline for one training step.
///
/// Call [`StepProfiler::start`] immediately before the step, route every
/// kernel launch through [`KernelProfiler::measure`], then call
/// [`StepProfiler::finish`] immediately after the step. `finish` synchronizes
/// the recorded events before returning the report.
pub struct StepProfiler {
    step_start: CudaEvent,
    kernels: Vec<PendingKernel>,
}

impl StepProfiler {
    pub fn start(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            step_start: timing_event(stream)?,
            kernels: Vec::new(),
        })
    }

    pub fn finish(self, stream: &CudaStream) -> Result<StepProfile, DriverError> {
        let step_end = timing_event(stream)?;
        let step_milliseconds = self.step_start.elapsed_ms(&step_end)? as f64;
        let kernels = self
            .kernels
            .into_iter()
            .map(|kernel| {
                Ok(KernelTiming {
                    name: kernel.name,
                    milliseconds: kernel.start.elapsed_ms(&kernel.end)? as f64,
                })
            })
            .collect::<Result<Vec<_>, DriverError>>()?;
        Ok(StepProfile {
            step_milliseconds,
            kernels,
        })
    }
}

impl KernelProfiler for StepProfiler {
    fn measure<T, F>(
        &mut self,
        stream: &CudaStream,
        name: &'static str,
        kernel: F,
    ) -> Result<T, DriverError>
    where
        F: FnOnce() -> Result<T, DriverError>,
    {
        let start = timing_event(stream)?;
        let output = kernel()?;
        let end = timing_event(stream)?;
        self.kernels.push(PendingKernel { name, start, end });
        Ok(output)
    }
}

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

#[cfg(test)]
mod tests {
    use super::{KernelTiming, StepProfile};

    #[test]
    fn profile_accounts_for_unattributed_device_time() {
        let profile = StepProfile {
            step_milliseconds: 10.0,
            kernels: vec![
                KernelTiming {
                    name: "forward.gemm",
                    milliseconds: 3.0,
                },
                KernelTiming {
                    name: "backward.gemm",
                    milliseconds: 5.5,
                },
            ],
        };

        assert_eq!(profile.kernel_milliseconds(), 8.5);
        assert_eq!(profile.unattributed_milliseconds(), 1.5);
        let report = profile.to_string();
        assert!(report.contains("forward.gemm"));
        assert!(report.contains("unattributed (copies/allocations/gaps)"));
        assert!(report.contains("full step"));
    }

    #[test]
    fn unattributed_time_does_not_go_negative_from_event_rounding() {
        let profile = StepProfile {
            step_milliseconds: 1.0,
            kernels: vec![KernelTiming {
                name: "kernel",
                milliseconds: 1.000_001,
            }],
        };
        assert_eq!(profile.unattributed_milliseconds(), 0.0);
    }
}
