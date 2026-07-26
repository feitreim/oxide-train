//! Shared helpers for the GPU kernel host binaries: CUDA-event timing and
//! reproducible random inputs. (Adapted from cuda-learning's bench-util.)

use std::sync::Arc;

use std::error::Error;
use std::fmt;

use cuda_core::{CudaEvent, CudaFunction, CudaStream, DriverError};

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

    /// Kernel timings summed per name in first-seen order, so a span launched
    /// once per block reads as one whole-step row.
    pub fn aggregated_kernels(&self) -> Vec<KernelTiming> {
        let mut order: Vec<KernelTiming> = Vec::new();
        for kernel in &self.kernels {
            match order.iter_mut().find(|entry| entry.name == kernel.name) {
                Some(entry) => entry.milliseconds += kernel.milliseconds,
                None => order.push(kernel.clone()),
            }
        }
        order
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
        for kernel in self.aggregated_kernels() {
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

/// What ptxas actually gave a loaded kernel. `registers` is per thread and
/// `spill_bytes` is its local-memory frame — the direct read on whether a
/// `.maxntid` (i.e. `#[launch_bounds]`) value is squeezing the allocator,
/// which is invisible in the PTX because ptxas runs after it.
pub struct FunctionProfile {
    pub registers: i32,
    pub spill_bytes: i32,
    pub max_threads: i32,
}

fn function_attribute(function: &CudaFunction, attribute: u32) -> Result<i32, Box<dyn Error>> {
    use cuda_core::sys::{cuFuncGetAttribute, cudaError_enum_CUDA_SUCCESS};
    let mut value = 0i32;
    let status = unsafe { cuFuncGetAttribute(&mut value, attribute, function.cu_function()) };
    if status != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuFuncGetAttribute({attribute}) failed: {status:?}").into());
    }
    Ok(value)
}

/// Read a loaded kernel's register / spill / `.maxntid` facts.
pub fn function_profile(function: &CudaFunction) -> Result<FunctionProfile, Box<dyn Error>> {
    use cuda_core::sys::{
        CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_NUM_REGS,
    };
    Ok(FunctionProfile {
        registers: function_attribute(
            function,
            CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_NUM_REGS,
        )?,
        spill_bytes: function_attribute(
            function,
            CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        )?,
        max_threads: function_attribute(
            function,
            CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        )?,
    })
}

/// A kernel's pinned ptxas ceiling — the regression gate for abstraction
/// work (issue #61 phase 0): a refactor that costs registers or grows the
/// local frame is a library bug, not an acceptable tax.
pub struct KernelBudget {
    pub name: &'static str,
    pub max_registers: i32,
    pub max_spill_bytes: i32,
}

/// Print every kernel's ptxas budget and fail if any pinned ceiling is
/// exceeded. Kernels without a budget entry are print-only (pin them once a
/// gated run has recorded their numbers); a kernel now *under* its ceiling
/// prints a ratchet hint instead of failing.
pub fn enforce_kernel_budgets(
    kernels: &[(&'static str, &CudaFunction)],
    budgets: &[KernelBudget],
) -> Result<(), Box<dyn Error>> {
    println!("ptxas budgets (registers/thread, spill bytes, .maxntid)");
    let mut violations = Vec::new();
    for (name, function) in kernels {
        let profile = function_profile(function)?;
        let budget = budgets.iter().find(|budget| budget.name == *name);
        let note = match budget {
            None => "  (unpinned)".to_string(),
            Some(budget) => {
                if profile.registers > budget.max_registers
                    || profile.spill_bytes > budget.max_spill_bytes
                {
                    violations.push(format!(
                        "{name}: {} regs / {} spill bytes exceeds the pinned \
                         {} / {}",
                        profile.registers,
                        profile.spill_bytes,
                        budget.max_registers,
                        budget.max_spill_bytes
                    ));
                    "  ✗ over budget".to_string()
                } else if profile.registers < budget.max_registers
                    || profile.spill_bytes < budget.max_spill_bytes
                {
                    format!(
                        "  (under the pinned {} / {} — ratchet down)",
                        budget.max_registers, budget.max_spill_bytes
                    )
                } else {
                    String::new()
                }
            }
        };
        println!(
            "  {name:<19} {:>3} regs, {:>4} spill bytes, maxntid {}{note}",
            profile.registers, profile.spill_bytes, profile.max_threads
        );
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(format!("ptxas budget regression:\n  {}", violations.join("\n  ")).into())
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
