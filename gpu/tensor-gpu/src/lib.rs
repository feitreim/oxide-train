//! Static-shape GPU tensor storage and foundational fp32 kernels.
//!
//! `GpuTensor<E, S>` deliberately wraps only storage. Operations remain
//! inherent methods and take an explicit stream and loaded kernel module; no
//! device dispatch or implicit synchronization is hidden behind `Tensor`.

use std::marker::PhantomData;

use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use tensor_core::{Element, Rank1, Rank2, Shape, Tensor};
use tensor_cpu::CpuTensor;

/// GEMM tile edge and launch block dimensions. This is intentionally public so
/// the repository sweep harness can rewrite it.
pub const TILE: usize = 16;
/// Threads in the single-block reduction kernels. Must remain a power of two.
pub const REDUCE_THREADS: usize = 256;
const TILE_ELEMENTS: usize = TILE * TILE;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = out.get_mut(index) {
            *slot = a[i] + b[i];
        }
    }

    #[kernel]
    pub fn mul(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = out.get_mut(index) {
            *slot = a[i] * b[i];
        }
    }

    #[kernel]
    pub fn scale(a: &[f32], factor: f32, mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = out.get_mut(index) {
            *slot = a[i] * factor;
        }
    }

    /// `dst += factor * src`, used by gradient accumulation and optimizers.
    #[kernel]
    pub fn add_scaled(src: &[f32], factor: f32, mut dst: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = dst.get_mut(index) {
            *slot += factor * src[i];
        }
    }

    /// One-block reduction. Threads accumulate grid-stride partial sums before
    /// a standard shared-memory tree reduction.
    #[kernel]
    pub fn sum(a: &[f32], len: u32, mut out: DisjointSlice<f32>) {
        static mut PARTIALS: SharedArray<f32, REDUCE_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let mut i = tid;
        let mut partial = 0.0f32;
        while i < len as usize {
            partial += a[i];
            i += REDUCE_THREADS;
        }
        unsafe {
            PARTIALS[tid] = partial;
        }
        thread::sync_threads();

        let mut stride = REDUCE_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIALS[tid] += PARTIALS[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        let index = thread::index_1d();
        if tid == 0
            && let Some(slot) = out.get_mut(index)
        {
            unsafe {
                *slot = PARTIALS[0];
            }
        }
    }

    #[kernel]
    pub fn dot(a: &[f32], b: &[f32], len: u32, mut out: DisjointSlice<f32>) {
        static mut PARTIALS: SharedArray<f32, REDUCE_THREADS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let mut i = tid;
        let mut partial = 0.0f32;
        while i < len as usize {
            partial += a[i] * b[i];
            i += REDUCE_THREADS;
        }
        unsafe {
            PARTIALS[tid] = partial;
        }
        thread::sync_threads();

        let mut stride = REDUCE_THREADS / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIALS[tid] += PARTIALS[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        let index = thread::index_1d();
        if tid == 0
            && let Some(slot) = out.get_mut(index)
        {
            unsafe {
                *slot = PARTIALS[0];
            }
        }
    }

    /// Auditable baseline: one output element per thread, reading both
    /// operands directly from global memory.
    #[kernel]
    pub fn gemm_naive(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::blockIdx_y() as usize * thread::blockDim_y() as usize
            + thread::threadIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        if row >= m as usize || col >= n as usize {
            return;
        }

        let mut acc = 0.0f32;
        for inner in 0..k as usize {
            acc += a[row * k as usize + inner] * b[inner * n as usize + col];
        }
        if let Some(index) = unsafe { thread::index_2d_runtime(n as usize) }
            && let Some(slot) = c.get_mut(index)
        {
            *slot = acc;
        }
    }

    /// Shared-memory tiled GEMM. Bounds checks make it valid for dimensions
    /// that are not multiples of `TILE`.
    #[kernel]
    pub fn gemm_tiled(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f32, TILE_ELEMENTS> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, TILE_ELEMENTS> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * TILE + ty;
        let col = thread::blockIdx_x() as usize * TILE + tx;
        let shared_index = ty * TILE + tx;
        let mut acc = 0.0f32;
        let tiles = (k as usize).div_ceil(TILE);

        for tile in 0..tiles {
            let a_col = tile * TILE + tx;
            let b_row = tile * TILE + ty;
            unsafe {
                TILE_A[shared_index] = if row < m as usize && a_col < k as usize {
                    a[row * k as usize + a_col]
                } else {
                    0.0
                };
                TILE_B[shared_index] = if b_row < k as usize && col < n as usize {
                    b[b_row * n as usize + col]
                } else {
                    0.0
                };
            }
            thread::sync_threads();

            for inner in 0..TILE {
                unsafe {
                    acc += TILE_A[ty * TILE + inner] * TILE_B[inner * TILE + tx];
                }
            }
            thread::sync_threads();
        }

        if row < m as usize
            && col < n as usize
            && let Some(index) = unsafe { thread::index_2d_runtime(n as usize) }
            && let Some(slot) = c.get_mut(index)
        {
            *slot = acc;
        }
    }

    /// `C = A^T . B`: `[M,K]^T x [M,N] -> [K,N]`.
    #[kernel]
    pub fn gemm_tn(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::blockIdx_y() as usize * thread::blockDim_y() as usize
            + thread::threadIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        if row >= k as usize || col >= n as usize {
            return;
        }
        let mut acc = 0.0f32;
        for inner in 0..m as usize {
            acc += a[inner * k as usize + row] * b[inner * n as usize + col];
        }
        if let Some(index) = unsafe { thread::index_2d_runtime(n as usize) }
            && let Some(slot) = c.get_mut(index)
        {
            *slot = acc;
        }
    }

    /// `C = A . B^T`: `[M,K] x [N,K]^T -> [M,N]`.
    #[kernel]
    pub fn gemm_nt(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::blockIdx_y() as usize * thread::blockDim_y() as usize
            + thread::threadIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        if row >= m as usize || col >= n as usize {
            return;
        }
        let mut acc = 0.0f32;
        for inner in 0..k as usize {
            acc += a[row * k as usize + inner] * b[col * k as usize + inner];
        }
        if let Some(index) = unsafe { thread::index_2d_runtime(n as usize) }
            && let Some(slot) = c.get_mut(index)
        {
            *slot = acc;
        }
    }
}

/// Owning, contiguous device tensor. Shape information is zero-sized and
/// exists only in `S`; the allocation contains exactly `S::NUM_ELEMENTS`.
pub struct GpuTensor<E: Element, S: Shape> {
    data: DeviceBuffer<E>,
    _shape: PhantomData<S>,
}

impl<E: Element, S: Shape> Tensor for GpuTensor<E, S> {
    type Elem = E;
    type Shape = S;
}

impl<E: Element + DeviceCopy, S: Shape> GpuTensor<E, S> {
    pub const LEN: usize = S::NUM_ELEMENTS;

    pub fn zeros(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            data: DeviceBuffer::zeroed(stream, S::NUM_ELEMENTS)?,
            _shape: PhantomData,
        })
    }

    pub fn from_host(stream: &CudaStream, src: &[E]) -> Result<Self, DriverError> {
        assert_eq!(src.len(), S::NUM_ELEMENTS, "slice length != shape volume");
        Ok(Self {
            data: DeviceBuffer::from_host(stream, src)?,
            _shape: PhantomData,
        })
    }

    pub fn from_cpu(stream: &CudaStream, src: &CpuTensor<E, S>) -> Result<Self, DriverError> {
        Self::from_host(stream, src.as_slice())
    }

    pub fn to_host(&self, stream: &CudaStream) -> Result<Vec<E>, DriverError> {
        self.data.to_host_vec(stream)
    }

    pub fn to_cpu(&self, stream: &CudaStream) -> Result<CpuTensor<E, S>, DriverError> {
        Ok(CpuTensor::from_slice(&self.to_host(stream)?))
    }

    pub fn as_device_buffer(&self) -> &DeviceBuffer<E> {
        &self.data
    }

    pub fn as_device_buffer_mut(&mut self) -> &mut DeviceBuffer<E> {
        &mut self.data
    }
}

fn elementwise_config<S: Shape>() -> LaunchConfig {
    assert!(S::NUM_ELEMENTS <= u32::MAX as usize);
    LaunchConfig::for_num_elems(S::NUM_ELEMENTS as u32)
}

fn reduction_config() -> LaunchConfig {
    assert!(REDUCE_THREADS.is_power_of_two());
    assert!(REDUCE_THREADS <= 1024);
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (REDUCE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn gemm_config<const M: usize, const N: usize>() -> LaunchConfig {
    assert!(TILE * TILE <= 1024);
    assert!(M <= u32::MAX as usize && N <= u32::MAX as usize);
    LaunchConfig {
        grid_dim: (
            (N as u32).div_ceil(TILE as u32),
            (M as u32).div_ceil(TILE as u32),
            1,
        ),
        block_dim: (TILE as u32, TILE as u32, 1),
        shared_mem_bytes: 0,
    }
}

impl<S: Shape> GpuTensor<f32, S> {
    pub fn add(
        &self,
        rhs: &Self,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<Self, DriverError> {
        let mut out = Self::zeros(stream)?;
        module.add(
            stream,
            elementwise_config::<S>(),
            &self.data,
            &rhs.data,
            &mut out.data,
        )?;
        Ok(out)
    }

    pub fn mul(
        &self,
        rhs: &Self,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<Self, DriverError> {
        let mut out = Self::zeros(stream)?;
        module.mul(
            stream,
            elementwise_config::<S>(),
            &self.data,
            &rhs.data,
            &mut out.data,
        )?;
        Ok(out)
    }

    pub fn scale(
        &self,
        factor: f32,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<Self, DriverError> {
        let mut out = Self::zeros(stream)?;
        module.scale(
            stream,
            elementwise_config::<S>(),
            &self.data,
            factor,
            &mut out.data,
        )?;
        Ok(out)
    }

    pub fn add_scaled_assign(
        &mut self,
        factor: f32,
        rhs: &Self,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<(), DriverError> {
        module.add_scaled(
            stream,
            elementwise_config::<S>(),
            &rhs.data,
            factor,
            &mut self.data,
        )
    }

    pub fn sum(
        &self,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank1<1>>, DriverError> {
        assert!(S::NUM_ELEMENTS <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        module.sum(
            stream,
            reduction_config(),
            &self.data,
            S::NUM_ELEMENTS as u32,
            &mut out.data,
        )?;
        Ok(out)
    }

    pub fn dot(
        &self,
        rhs: &Self,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank1<1>>, DriverError> {
        assert!(S::NUM_ELEMENTS <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        module.dot(
            stream,
            reduction_config(),
            &self.data,
            &rhs.data,
            S::NUM_ELEMENTS as u32,
            &mut out.data,
        )?;
        Ok(out)
    }
}

impl<const M: usize, const K: usize> GpuTensor<f32, Rank2<M, K>> {
    pub fn matmul_naive<const N: usize>(
        &self,
        rhs: &GpuTensor<f32, Rank2<K, N>>,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<M, N>>, DriverError> {
        assert!(K <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        module.gemm_naive(
            stream,
            gemm_config::<M, N>(),
            M as u32,
            N as u32,
            K as u32,
            &self.data,
            &rhs.data,
            &mut out.data,
        )?;
        Ok(out)
    }

    /// Default fp32 GEMM: shared-memory tiled `[M,K] x [K,N] -> [M,N]`.
    pub fn matmul<const N: usize>(
        &self,
        rhs: &GpuTensor<f32, Rank2<K, N>>,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<M, N>>, DriverError> {
        assert!(K <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        module.gemm_tiled(
            stream,
            gemm_config::<M, N>(),
            M as u32,
            N as u32,
            K as u32,
            &self.data,
            &rhs.data,
            &mut out.data,
        )?;
        Ok(out)
    }

    pub fn matmul_tn<const N: usize>(
        &self,
        rhs: &GpuTensor<f32, Rank2<M, N>>,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<K, N>>, DriverError> {
        assert!(M <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        module.gemm_tn(
            stream,
            gemm_config::<K, N>(),
            M as u32,
            N as u32,
            K as u32,
            &self.data,
            &rhs.data,
            &mut out.data,
        )?;
        Ok(out)
    }

    pub fn matmul_nt<const N: usize>(
        &self,
        rhs: &GpuTensor<f32, Rank2<N, K>>,
        stream: &CudaStream,
        module: &kernels::LoadedModule,
    ) -> Result<GpuTensor<f32, Rank2<M, N>>, DriverError> {
        assert!(K <= u32::MAX as usize);
        let mut out = GpuTensor::zeros(stream)?;
        module.gemm_nt(
            stream,
            gemm_config::<M, N>(),
            M as u32,
            N as u32,
            K as u32,
            &self.data,
            &rhs.data,
            &mut out.data,
        )?;
        Ok(out)
    }
}
