//! # cuBLASLt beside the GEMM — the denominator the benchmark did not have
//!
//! `bench` reported the tcgen05 GEMM in TFLOP/s and had nothing to divide by
//! except a spec sheet. "48% of dense peak" is a real number and the wrong
//! comparison: what a kernel is worth is how far it sits from a tuned library
//! **on the same device, at the same shape, on the same day**. This module is
//! that library, called in-process, so every future change to the kernel
//! carries a ratio instead of an absolute somebody has to re-contextualize.
//!
//! Adapted from `ferro-kittens`' `experiments/src/cublaslt.rs`, whose header
//! carries the full argument. Two parts of it are worth repeating here.
//!
//! ## The transpose trap
//!
//! cuBLASLt is column-major and we are row-major. Writing `Â` for "the bytes of
//! buffer `a`, read as column-major":
//!
//! - `a` holds `[m, k]` row-major with `k` contiguous, so `Â` is `k × m`,
//!   `ld = k`, and `Â = Aᵀ`.
//! - `b` holds `[n, k]` row-major, so `B̂` is `k × n`, `ld = k`, `B̂ = Bᵀ`.
//! - `c` holds `[m, n]` row-major, so `Ĉ` is `n × m`, `ld = n`, `Ĉ = Cᵀ`.
//!
//! The product to ask for is therefore not `C` but its transpose,
//! `Ĉ = Cᵀ = (A·Bᵀ)ᵀ = B·Aᵀ = (B̂)ᵀ · Â`. Reading right to left: **cuBLASLt's
//! first operand is our `b`** under `CUBLAS_OP_T`, its second is our `a` under
//! `CUBLAS_OP_N`, and the output layout is `(n, m, n)`. The operands are
//! *swapped* relative to the obvious spelling — and the obvious spelling is not
//! slower and does not fail. It computes a different matrix at full speed. That
//! is why the benchmark compares [`Baseline`]'s `C` against the kernel's before
//! any time is reported.
//!
//! ## What is fair here, and what is not
//!
//! - **Layout.** Both sides read byte-identical operands: a plain packed bf16
//!   `[m, k]` and `[n, k]`, K contiguous. Ours needs a TMA tensor map over that
//!   buffer, which is a descriptor and not a copy, built once outside the clock
//!   exactly as cuBLASLt's descriptors and heuristic are.
//! - **Generality.** Ours computes one form: both operands K-major, `α = 1`,
//!   `β = 0`, no epilogue, `m % 256 == n % 256 == k % 64 == 0`. cuBLASLt takes
//!   any of that. A like-for-like rate against a library that is also general
//!   is a comparison in our favour, and the ratios should be read knowing it.
//! - **Workspace.** cuBLASLt gets [`WORKSPACE_BYTES`]; ours takes none. It is
//!   allocated outside the timed region, which is the fair choice — a real
//!   pipeline keeps a workspace around — and it is a genuine allowance ours
//!   does not use.
//! - **Algorithm.** Whatever `cublasLtMatmulAlgoGetHeuristic` ranks first, no
//!   search and no tuning pass. [`describe`] prints its configuration so the
//!   baseline is reproducible rather than merely quoted.

use std::error::Error;
use std::ffi::{CStr, c_char, c_int, c_void};

use cuda_core::{CudaStream, DeviceBuffer};

/// Scratch cuBLASLt may use, and the cap the heuristic is chosen under.
///
/// 32 MiB is upstream `gemm_sol`'s figure, kept so this baseline is comparable
/// with theirs. It bounds which algorithms the heuristic may return — a larger
/// workspace admits split-K schemes a smaller one does not — so it is part of
/// the configuration and not an implementation detail.
const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

type Handle = *mut c_void;
type MatmulDesc = *mut c_void;
type MatrixLayout = *mut c_void;
type Preference = *mut c_void;
type Status = c_int;

const SUCCESS: Status = 0;

const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const DESC_TRANSA: c_int = 3;
const DESC_TRANSB: c_int = 4;
const PREF_MAX_WORKSPACE_BYTES: c_int = 1;

/// `cublasLtMatmulAlgo_t` — opaque to callers, copied by value into
/// `cublasLtMatmul`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Algo {
    /// Read only by cuBLASLt, through the pointer [`describe`] and
    /// `cublasLtMatmul` are handed.
    #[allow(dead_code)]
    data: [u64; 8],
}

/// `cublasLtMatmulHeuristicResult_t`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Heuristic {
    algo: Algo,
    workspace_bytes: usize,
    /// Whether *this entry* is usable. A returned count above zero is not the
    /// same claim, and the header is explicit that the per-result status has to
    /// be read separately.
    state: Status,
    waves: f32,
    /// Padding the header declares, so the struct is the right size.
    #[allow(dead_code)]
    reserved: [c_int; 4],
}

#[link(name = "cublasLt")]
unsafe extern "C" {
    fn cublasLtCreate(handle: *mut Handle) -> Status;
    fn cublasLtDestroy(handle: Handle) -> Status;
    fn cublasLtGetVersion() -> usize;
    fn cublasLtGetStatusString(status: Status) -> *const c_char;

    fn cublasLtMatmulDescCreate(
        desc: *mut MatmulDesc,
        compute_type: c_int,
        scale_type: c_int,
    ) -> Status;
    fn cublasLtMatmulDescDestroy(desc: MatmulDesc) -> Status;
    fn cublasLtMatmulDescSetAttribute(
        desc: MatmulDesc,
        attribute: c_int,
        value: *const c_void,
        bytes: usize,
    ) -> Status;

    fn cublasLtMatrixLayoutCreate(
        layout: *mut MatrixLayout,
        element_type: c_int,
        rows: u64,
        columns: u64,
        leading_dimension: i64,
    ) -> Status;
    fn cublasLtMatrixLayoutDestroy(layout: MatrixLayout) -> Status;

    fn cublasLtMatmulPreferenceCreate(preference: *mut Preference) -> Status;
    fn cublasLtMatmulPreferenceDestroy(preference: Preference) -> Status;
    fn cublasLtMatmulPreferenceSetAttribute(
        preference: Preference,
        attribute: c_int,
        value: *const c_void,
        bytes: usize,
    ) -> Status;

    fn cublasLtMatmulAlgoGetHeuristic(
        handle: Handle,
        desc: MatmulDesc,
        a: MatrixLayout,
        b: MatrixLayout,
        c: MatrixLayout,
        d: MatrixLayout,
        preference: Preference,
        requested: c_int,
        results: *mut Heuristic,
        returned: *mut c_int,
    ) -> Status;
    fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const Algo,
        attribute: c_int,
        buffer: *mut c_void,
        bytes: usize,
        written: *mut usize,
    ) -> Status;

    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmul(
        handle: Handle,
        desc: MatmulDesc,
        alpha: *const c_void,
        a: *const c_void,
        a_layout: MatrixLayout,
        b: *const c_void,
        b_layout: MatrixLayout,
        beta: *const c_void,
        c: *const c_void,
        c_layout: MatrixLayout,
        d: *mut c_void,
        d_layout: MatrixLayout,
        algo: *const Algo,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> Status;
}

/// Turn a non-zero status into an error naming both the call and what cuBLASLt
/// calls the code, since the numbers alone are not memorable.
fn checked(status: Status, call: &str) -> Result<(), Box<dyn Error>> {
    if status == SUCCESS {
        return Ok(());
    }
    // SAFETY: cuBLASLt returns a static string for every status, including
    // codes it does not recognize.
    let text = unsafe { CStr::from_ptr(cublasLtGetStatusString(status)) };
    Err(format!(
        "{call}: cuBLASLt status {status} ({})",
        text.to_string_lossy()
    )
    .into())
}

/// Every cuBLASLt object one measurement creates, destroyed in one place.
///
/// The fields start null and are filled in order, so a failure part way through
/// setup drops exactly what had been created — where straight-line C leaks
/// every handle it already made on any early return.
struct Session {
    handle: Handle,
    desc: MatmulDesc,
    a: MatrixLayout,
    b: MatrixLayout,
    d: MatrixLayout,
    preference: Preference,
}

impl Session {
    fn new() -> Self {
        Session {
            handle: std::ptr::null_mut(),
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            d: std::ptr::null_mut(),
            preference: std::ptr::null_mut(),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: each field is either null or an object this session made, and
        // nothing outlives it.
        unsafe {
            if !self.preference.is_null() {
                let _ = cublasLtMatmulPreferenceDestroy(self.preference);
            }
            for layout in [self.d, self.b, self.a] {
                if !layout.is_null() {
                    let _ = cublasLtMatrixLayoutDestroy(layout);
                }
            }
            if !self.desc.is_null() {
                let _ = cublasLtMatmulDescDestroy(self.desc);
            }
            if !self.handle.is_null() {
                let _ = cublasLtDestroy(self.handle);
            }
        }
    }
}

/// A `CUBLASLT_MATMUL_DESC_*` attribute that is one `int`, which both of the
/// two this benchmark sets are.
fn set_transpose(desc: MatmulDesc, attribute: c_int, value: i32) -> Result<(), Box<dyn Error>> {
    // SAFETY: `desc` is live and the buffer is exactly the width the attribute
    // is documented to take.
    checked(
        unsafe {
            cublasLtMatmulDescSetAttribute(
                desc,
                attribute,
                (&raw const value).cast(),
                size_of::<i32>(),
            )
        },
        "cublasLtMatmulDescSetAttribute",
    )
}

/// One column-major matrix layout: `rows × columns` at `leading_dimension`.
fn layout(
    slot: &mut MatrixLayout,
    element_type: c_int,
    rows: usize,
    columns: usize,
    leading_dimension: usize,
) -> Result<(), Box<dyn Error>> {
    // SAFETY: `slot` is a live null pointer this call fills in.
    checked(
        unsafe {
            cublasLtMatrixLayoutCreate(
                slot,
                element_type,
                rows as u64,
                columns as u64,
                leading_dimension as i64,
            )
        },
        "cublasLtMatrixLayoutCreate",
    )
}

/// The `uint32_t` configuration attributes that identify a chosen algorithm.
const CONFIG_ATTRIBUTES: [(&str, c_int); 6] = [
    ("id", 0),
    ("tile", 1),
    ("splitk", 2),
    ("reduction", 3),
    ("swizzle", 4),
    ("stages", 6),
];

/// The `uint16_t` ones, read through a `u16` buffer because reading them
/// through a `u32` would be a transcription bug. `cluster` is the attribute
/// oxide-train#80 wants most: it says whether the library's own schedule is a
/// CTA pair like ours or something else at these shapes.
const NARROW_ATTRIBUTES: [(&str, c_int); 2] = [("inner", 7), ("cluster", 8)];

/// The chosen algorithm, in one line, so the baseline can be reproduced rather
/// than merely believed.
fn describe(heuristic: &Heuristic) -> String {
    let read = |attribute: c_int, bytes: usize, value: *mut c_void| {
        let mut written = 0usize;
        // SAFETY: `value` is `bytes` wide and the attribute is declared at that
        // width; a width cuBLASLt disagrees with is an error status, not a
        // write past the end.
        let status = unsafe {
            cublasLtMatmulAlgoConfigGetAttribute(
                &heuristic.algo,
                attribute,
                value,
                bytes,
                &mut written,
            )
        };
        status == SUCCESS
    };
    let fields: Vec<String> = CONFIG_ATTRIBUTES
        .iter()
        .map(|&(name, attribute)| {
            let mut value = 0u32;
            match read(attribute, size_of::<u32>(), (&raw mut value).cast()) {
                true => format!("{name}={value}"),
                false => format!("{name}=?"),
            }
        })
        .chain(NARROW_ATTRIBUTES.iter().map(|&(name, attribute)| {
            let mut value = 0u16;
            match read(attribute, size_of::<u16>(), (&raw mut value).cast()) {
                true => format!("{name}={value}"),
                false => format!("{name}=?"),
            }
        }))
        .collect();
    format!(
        "{} waves={:.2} workspace={} B",
        fields.join(" "),
        heuristic.waves,
        heuristic.workspace_bytes
    )
}

/// `cuBLASLt <version>` — printed once above the comparison table.
pub fn about() -> String {
    // SAFETY: a pure version query, safe to call before any handle exists.
    format!("cuBLASLt version {}", unsafe { cublasLtGetVersion() })
}

/// Which product the baseline computes — the same two operand walks the kernel
/// has, spelled in cuBLASLt's column-major terms by [`Baseline::with_form`].
#[derive(Clone, Copy, PartialEq)]
pub enum Form {
    /// `C = A·Bᵀ` over K-major `[m, k]` and `[n, k]` operands, `β = 0`: the
    /// store modes.
    Store,
    /// `C += Aᵀ·B` over MN-major `[k, m]` and `[k, n]` panels, `β = 1`: the
    /// weight-gradient modes. `C` is read as well as written, exactly as the
    /// kernel's fold reads it.
    AccumulateTransposed,
}

/// `C`'s element — the one thing the kernel's two entry points disagree on, so
/// the denominator has to be priced at both widths too.
#[derive(Clone, Copy, PartialEq)]
pub enum OutElement {
    Bf16,
    F32,
}

/// A configured product at one shape: bf16 in, fp32 across, and a [`Form`] and
/// [`OutElement`] matching whichever kernel mode it is the denominator for.
///
/// Holds its session and workspace so a caller can launch it repeatedly under
/// whatever clock it likes, which is what makes "the same harness on both
/// sides" possible at all.
pub struct Baseline {
    session: Session,
    heuristic: Heuristic,
    workspace: DeviceBuffer<u8>,
    alpha: f32,
    beta: f32,
}

impl Baseline {
    /// Configure the baseline for `[m, k] × [n, k]ᵀ → [m, n]`, all packed
    /// row-major bf16 — [`Form::Store`] at [`OutElement::Bf16`], the signature
    /// `gemm_tcgen05_bf16_optimized`'s store mode has.
    pub fn new(stream: &CudaStream, m: usize, n: usize, k: usize) -> Result<Self, Box<dyn Error>> {
        Self::with_form(stream, m, n, k, Form::Store, OutElement::Bf16)
    }

    /// Configure the baseline for one of the kernel's four mode/element
    /// combinations, `[m, n]` out.
    ///
    /// The transpose algebra follows the module header. For
    /// [`Form::AccumulateTransposed`] the operands are their native row-major
    /// panels — `a` is `[k, m]`, `b` is `[k, n]` — and the target is
    /// `Ĉ = Cᵀ = (AᵀB)ᵀ = Bᵀ·A`. Read as column-major, `b`'s bytes *are* `Bᵀ`
    /// (`n × k`, `ld = n`, `CUBLAS_OP_N`) and `a`'s are `Aᵀ` (`m × k`,
    /// `ld = m`), so `A` itself is reached under `CUBLAS_OP_T`. The first
    /// operand cuBLASLt sees is still our `b`, as in the store form.
    pub fn with_form(
        stream: &CudaStream,
        m: usize,
        n: usize,
        k: usize,
        form: Form,
        out: OutElement,
    ) -> Result<Self, Box<dyn Error>> {
        // Outside anybody's timed region, as every allocation on both sides is.
        let workspace = DeviceBuffer::<u8>::zeroed(stream, WORKSPACE_BYTES)?;
        let mut session = Session::new();
        // SAFETY (this block): each call fills or consumes a field of
        // `session`, which owns every handle and destroys it exactly once.
        checked(
            unsafe { cublasLtCreate(&mut session.handle) },
            "cublasLtCreate",
        )?;
        checked(
            unsafe { cublasLtMatmulDescCreate(&mut session.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) },
            "cublasLtMatmulDescCreate",
        )?;

        match form {
            // `Ĉ = (B̂)ᵀ · Â`, per the module header: the first operand is our
            // `b`, transposed; the second is our `a`, not.
            Form::Store => {
                set_transpose(session.desc, DESC_TRANSA, CUBLAS_OP_T)?;
                set_transpose(session.desc, DESC_TRANSB, CUBLAS_OP_N)?;
                layout(&mut session.a, CUDA_R_16BF, k, n, k)?;
                layout(&mut session.b, CUDA_R_16BF, k, m, k)?;
            }
            // `Ĉ = Bᵀ·A`, per [`Baseline::with_form`]: our `b` read as-is, our
            // `a` transposed.
            Form::AccumulateTransposed => {
                set_transpose(session.desc, DESC_TRANSA, CUBLAS_OP_N)?;
                set_transpose(session.desc, DESC_TRANSB, CUBLAS_OP_T)?;
                layout(&mut session.a, CUDA_R_16BF, n, k, n)?;
                layout(&mut session.b, CUDA_R_16BF, m, k, m)?;
            }
        }
        // `C` at the kernel's width, because a denominator writing (and, under
        // the fold, reading) a different number of bytes than the kernel it
        // divides would report the difference as ours. The compute type is
        // untouched: both sides still accumulate in fp32.
        let out_element = match out {
            OutElement::Bf16 => CUDA_R_16BF,
            OutElement::F32 => CUDA_R_32F,
        };
        layout(&mut session.d, out_element, n, m, n)?;

        checked(
            unsafe { cublasLtMatmulPreferenceCreate(&mut session.preference) },
            "cublasLtMatmulPreferenceCreate",
        )?;
        // A local, not the constant: cuBLASLt reads this through a pointer, and
        // the address of a `const` is the address of a temporary.
        let workspace_bytes = WORKSPACE_BYTES;
        checked(
            unsafe {
                cublasLtMatmulPreferenceSetAttribute(
                    session.preference,
                    PREF_MAX_WORKSPACE_BYTES,
                    (&raw const workspace_bytes).cast(),
                    size_of::<usize>(),
                )
            },
            "cublasLtMatmulPreferenceSetAttribute",
        )?;

        let mut heuristic = Heuristic {
            algo: Algo { data: [0; 8] },
            workspace_bytes: 0,
            state: SUCCESS,
            waves: 0.0,
            reserved: [0; 4],
        };
        let mut returned: c_int = 0;
        checked(
            unsafe {
                cublasLtMatmulAlgoGetHeuristic(
                    session.handle,
                    session.desc,
                    session.a,
                    session.b,
                    session.d,
                    session.d,
                    session.preference,
                    1,
                    &mut heuristic,
                    &mut returned,
                )
            },
            "cublasLtMatmulAlgoGetHeuristic",
        )?;
        if returned == 0 {
            return Err(format!("cuBLASLt has no algorithm for {m}x{n}x{k}").into());
        }
        // A returned count above zero does not by itself mean the entry is
        // usable.
        checked(heuristic.state, "the algorithm the heuristic returned")?;

        Ok(Self {
            session,
            heuristic,
            workspace,
            alpha: 1.0,
            beta: match form {
                Form::Store => 0.0,
                Form::AccumulateTransposed => 1.0,
            },
        })
    }

    /// The chosen algorithm's identity.
    pub fn algorithm(&self) -> String {
        describe(&self.heuristic)
    }

    /// One launch on `stream`, reading the same buffers the kernel reads.
    ///
    /// # Safety
    ///
    /// `a`, `b` and `c` must be the packed bf16 buffers this baseline was
    /// configured for: `m * k`, `n * k` and `m * n` elements.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &DeviceBuffer<u16>,
    ) -> Result<(), Box<dyn Error>> {
        unsafe { self.launch_devptrs(stream, a.cu_deviceptr(), b.cu_deviceptr(), c.cu_deviceptr()) }
    }

    /// [`Baseline::launch`] by device pointer, for a `C` whose element the
    /// configuration chose — a typed signature per [`OutElement`] would say no
    /// more than the layouts already do.
    ///
    /// # Safety
    ///
    /// The pointers must name live device buffers of the shapes and elements
    /// this baseline was configured for: `a` and `b` the kernel's own bf16
    /// operands, `c` an `m * n` output in the configured [`OutElement`].
    pub unsafe fn launch_devptrs(
        &self,
        stream: &CudaStream,
        a: u64,
        b: u64,
        c: u64,
    ) -> Result<(), Box<dyn Error>> {
        // SAFETY: every handle is live for the call, the device pointers name
        // buffers sized by the layouts, and `c` is both `C` and `D` — legal,
        // and either unread (`β = 0`) or the in-place fold the form asks for.
        checked(
            unsafe {
                cublasLtMatmul(
                    self.session.handle,
                    self.session.desc,
                    (&raw const self.alpha).cast(),
                    b as *const c_void,
                    self.session.a,
                    a as *const c_void,
                    self.session.b,
                    (&raw const self.beta).cast(),
                    c as *const c_void,
                    self.session.d,
                    c as *mut c_void,
                    self.session.d,
                    &self.heuristic.algo,
                    self.workspace.cu_deviceptr() as *mut c_void,
                    WORKSPACE_BYTES,
                    stream.cu_stream().cast(),
                )
            },
            "cublasLtMatmul",
        )
    }
}
