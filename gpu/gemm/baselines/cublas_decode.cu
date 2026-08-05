// nvcc-flags: -arch=sm_100a -lcuda -lcublasLt -lcupti -I/usr/local/cuda/extras/CUPTI/include
//
// What cuBLASLt actually launches at #80's shallow-K shapes.
//
// `cublasLtMatmulAlgoConfigGetAttribute` reports `tile=23 cluster=3 stages=35`
// at every shallow depth, and those are opaque library indices: #80 read
// `cluster=3` as a four-CTA cluster with a multicast `A`, built that kernel,
// and it lost by two. The reading has to come from outside the library.
//
// Nsight Compute cannot run in this container — its counter library reports
// `LibraryNotLoaded`, which is the driver's profiling support and not something
// an image can install. But the launch *configuration* needs no performance
// counters at all: CUPTI's callback API sees every `cuLaunchKernel` /
// `cuLaunchKernelEx` the library issues, and the driver will answer
// `cuFuncGetAttribute` and `cuOccupancyMaxActiveBlocksPerMultiprocessor` about
// the handle it is launching. Grid, block, cluster shape, dynamic and static
// shared memory, registers per thread, CTAs per SM and the mangled name —
// which for these kernels encodes the tile — all fall out of that.
//
// Both sides run: ours is launched from the Rust harness, so what is here is
// cuBLASLt at the same three shapes and the same operand layouts
// `src/cublaslt.rs` configures, with `describe`'s own fields printed so the
// algorithm can be confirmed identical to the one `model_shapes` divides by.

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    CUresult status = (call);                                                   \
    if (status != CUDA_SUCCESS) {                                               \
      const char *text = nullptr;                                               \
      cuGetErrorString(status, &text);                                          \
      printf("%s failed: %s\n", #call, text ? text : "?");                      \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

#define LT_CHECK(call)                                                         \
  do {                                                                         \
    cublasStatus_t status = (call);                                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
      printf("%s failed: %d\n", #call, (int)status);                            \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

static int sm_count = 0;

// One line per launch: everything the driver will say about a function it is
// being handed, which is everything about the configuration except what the
// kernel does with it.
static void report(const char *label, CUfunction f, unsigned gx, unsigned gy,
                   unsigned gz, unsigned bx, unsigned by, unsigned bz,
                   unsigned smem, unsigned cx, unsigned cy, unsigned cz,
                   const char *symbol) {
  int regs = 0, static_smem = 0, max_threads = 0, max_dynamic = 0;
  CUresult attr = cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
  cuFuncGetAttribute(&static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);
  cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);
  cuFuncGetAttribute(&max_dynamic,
                     CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f);
  int required_x = 0, required_y = 0, required_z = 0;
  cuFuncGetAttribute(&required_x, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, f);
  cuFuncGetAttribute(&required_y, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, f);
  cuFuncGetAttribute(&required_z, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, f);

  unsigned threads = bx * by * bz;
  int blocks_per_sm = 0;
  cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, f, (int)threads,
                                              smem);
  unsigned ctas = gx * gy * gz;

  const char *name = nullptr;
  CUresult named = cuFuncGetName(&name, f);

  printf("  [%s] grid %ux%ux%u = %u CTAs, block %u, cluster %ux%ux%u "
         "(required %d,%d,%d)\n",
         label, gx, gy, gz, ctas, threads, cx, cy, cz, required_x, required_y,
         required_z);
  printf("        smem dyn %u + static %d, regs %d, max threads %d, "
         "opted-in dyn %d (cuFuncGetAttribute %d)\n",
         smem, static_smem, regs, max_threads, max_dynamic, (int)attr);
  printf("        blocks/SM %d -> %d resident CTAs of %u launched (%.2f "
         "waves over %d SMs)\n",
         blocks_per_sm, blocks_per_sm * sm_count, ctas,
         (double)ctas / (blocks_per_sm * sm_count > 0
                             ? blocks_per_sm * sm_count
                             : 1),
         sm_count);
  printf("        name %s (cuFuncGetName %d, symbol %s)\n", name ? name : "?",
         (int)named, symbol ? symbol : "?");
  fflush(stdout);
}

static void CUPTIAPI on_driver_call(void *, CUpti_CallbackDomain,
                                    CUpti_CallbackId id,
                                    const CUpti_CallbackData *data) {
  if (data->callbackSite != CUPTI_API_ENTER) {
    return;
  }
  if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
      id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
    auto *p = (cuLaunchKernel_params *)data->functionParams;
    report("cuLaunchKernel", p->f, p->gridDimX, p->gridDimY, p->gridDimZ,
           p->blockDimX, p->blockDimY, p->blockDimZ, p->sharedMemBytes, 1, 1,
           1, data->symbolName);
    return;
  }
  if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
      id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz) {
    auto *p = (cuLaunchKernelEx_params *)data->functionParams;
    const CUlaunchConfig *c = p->config;
    unsigned cx = 1, cy = 1, cz = 1;
    for (unsigned i = 0; i < c->numAttrs; ++i) {
      if (c->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
        cx = c->attrs[i].value.clusterDim.x;
        cy = c->attrs[i].value.clusterDim.y;
        cz = c->attrs[i].value.clusterDim.z;
      }
    }
    report("cuLaunchKernelEx", p->f, c->gridDimX, c->gridDimY, c->gridDimZ,
           c->blockDimX, c->blockDimY, c->blockDimZ, c->sharedMemBytes, cx, cy,
           cz, data->symbolName);
  }
}

/// `src/cublaslt.rs`'s `describe`, so the algorithm decoded here can be
/// confirmed to be the one `model_shapes` divides by.
static void describe(const cublasLtMatmulAlgo_t &algo, float waves) {
  const char *wide_names[] = {"id",      "tile",    "splitk",
                              "reduction", "swizzle", "stages"};
  const int wide_ids[] = {0, 1, 2, 3, 4, 6};
  printf("        algo");
  for (int i = 0; i < 6; ++i) {
    uint32_t value = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(
            &algo, (cublasLtMatmulAlgoConfigAttributes_t)wide_ids[i], &value,
            sizeof(value), &written) == CUBLAS_STATUS_SUCCESS) {
      printf(" %s=%u", wide_names[i], value);
    }
  }
  const char *narrow_names[] = {"inner", "cluster"};
  const int narrow_ids[] = {7, 8};
  for (int i = 0; i < 2; ++i) {
    uint16_t value = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(
            &algo, (cublasLtMatmulAlgoConfigAttributes_t)narrow_ids[i], &value,
            sizeof(value), &written) == CUBLAS_STATUS_SUCCESS) {
      printf(" %s=%u", narrow_names[i], value);
    }
  }
  printf(" waves=%.2f\n", waves);
}

struct Shape {
  const char *name;
  int m, k, n;
};

int main() {
  CU_CHECK(cuInit(0));
  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, 0));
  cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                       device);
  // The runtime's primary context rather than `cuCtxCreate`, whose signature
  // moved in CUDA 13 and which nothing here needs a private context for.
  cudaSetDevice(0);
  cudaFree(nullptr);

  CUpti_SubscriberHandle subscriber;
  CUptiResult cupti = cuptiSubscribe(
      &subscriber, (CUpti_CallbackFunc)on_driver_call, nullptr);
  if (cupti != CUPTI_SUCCESS) {
    const char *text = nullptr;
    cuptiGetResultString(cupti, &text);
    printf("cuptiSubscribe failed: %s\n", text ? text : "?");
    return 1;
  }
  cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  printf("CUPTI callbacks live, %d SMs\n", sm_count);

  const Shape shapes[] = {
      {"qkv fwd     24576x3072x9216", 24576, 3072, 9216},
      {"gate_up fwd  6144x3072x8192", 6144, 3072, 8192},
      {"lm_head fwd 24576x3072x50432", 24576, 3072, 50432},
  };

  cublasLtHandle_t handle;
  LT_CHECK(cublasLtCreate(&handle));
  const size_t workspace_bytes = 32u << 20;
  void *workspace = nullptr;
  cudaMalloc(&workspace, workspace_bytes);

  for (const Shape &shape : shapes) {
    const int m = shape.m, k = shape.k, n = shape.n;
    void *a = nullptr, *b = nullptr, *c = nullptr;
    cudaMalloc(&a, (size_t)m * k * 2);
    cudaMalloc(&b, (size_t)n * k * 2);
    cudaMalloc(&c, (size_t)m * n * 2);
    cudaMemset(a, 0x3c, (size_t)m * k * 2);
    cudaMemset(b, 0x3c, (size_t)n * k * 2);

    // `Form::Store`, `OutElement::Bf16` — the layouts src/cublaslt.rs sets for
    // a row-major bf16 `C = A·Bᵀ`, read as the column-major `Ĉ = B̂ᵀ·Â`.
    cublasLtMatmulDesc_t desc;
    LT_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transa, sizeof(transa)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                            &transb, sizeof(transb)));
    cublasLtMatrixLayout_t la, lb, ld;
    LT_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, n, k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, m, k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, n, m, n));

    cublasLtMatmulPreference_t preference;
    LT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
        sizeof(workspace_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, ld, ld,
                                            preference, 1, &heuristic,
                                            &returned));
    if (returned == 0) {
      printf("  %s: no algorithm\n", shape.name);
      continue;
    }

    printf("%s\n", shape.name);
    describe(heuristic.algo, heuristic.wavesCount);
    const float alpha = 1.0f, beta = 0.0f;
    LT_CHECK(cublasLtMatmul(handle, desc, &alpha, b, la, a, lb, &beta, c, ld, c,
                            ld, &heuristic.algo, workspace, workspace_bytes,
                            nullptr));
    cudaDeviceSynchronize();

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(desc);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }
  cuptiUnsubscribe(subscriber);
  return 0;
}
