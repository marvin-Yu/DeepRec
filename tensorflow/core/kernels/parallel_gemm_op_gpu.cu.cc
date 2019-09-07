//
// Created by qiaoxj on 2019-09-06.
//

#include "parallel_gemm_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif
#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace {
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* gpu_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(gpu_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}
}

#define CUDA_KERNEL_LOOP(i, n)                                   \
  for (int64 i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_GET_BLOCKS(N, Threads) (N + Threads - 1) / Threads

inline int GetThreadsNum(int data_size, bool upper = false) {
  if (upper) {
    // Return threadnum >= data_size if data_size <= 512
    if (data_size < 16)
      return 16;
    else if (data_size < 32)
      return 32;
    else if (data_size < 64)
      return 64;
    else if (data_size < 128)
      return 128;
    else if (data_size < 256)
      return 256;
    else
      return 512;
  } else {
    // Rerurn theadnum <= data_size
    if (data_size >= 512)
      return 512;
    else if (data_size >= 256)
      return 256;
    else if (data_size >= 128)
      return 128;
    else if (data_size >= 64)
      return 64;
    else if (data_size >= 32)
      return 32;
    else if (data_size >= 16)
      return 16;
    else
      return data_size;
  }
}

// Get the number of blocks, elem_per_thread represents the number of
// elements processed by each thread
inline int GetBlockNum(int block_size, int elem_per_thread = 16) {
  int z = block_size / elem_per_thread;
  if (z <= 0)
    return 1;
  else
    return z;
}

template <typename DType>
__global__ void BatchedUBroadcastKernel(DType* y, int64 batch_count,
                                        int64 y_size, const DType* x,
                                        int64 x_size) {
  int64 total_y_size = batch_count * y_size;
  CUDA_KERNEL_LOOP(index, total_y_size) {
    int batch_index = index / y_size;
    int batch_offset = index % y_size;
    y[index] = x[batch_index * x_size + batch_offset % x_size];
  }
}

template <typename DType>
static void BatchedUBroadcast(DType* y, int64 batch_size,
                              const std::vector<int64>& y_shape, const DType* x,
                              const std::vector<int64>& x_shape,
                              const GPUDevice& gpu_device) {
  int64 y_size = std::accumulate(y_shape.begin(), y_shape.end(), 1,
                                 std::multiplies<int64>());
  int64 x_size = std::accumulate(x_shape.begin(), x_shape.end(), 1,
                                 std::multiplies<int64>());
  int thread_num = GetThreadsNum(y_size * batch_size);
  int block_num = GetBlockNum(CUDA_GET_BLOCKS(y_size * batch_size, thread_num));
  BatchedUBroadcastKernel<DType>
  <<<block_num, thread_num, 0, gpu_device.stream()>>>(y, batch_size, y_size,
      x, x_size);
}

template <typename DType>
static void RunGemmStridedBatched(OpKernelContext* context,
                                  perftools::gputools::blas::Transpose trans_a,
                                  perftools::gputools::blas::Transpose trans_b,
                                  uint64 m, uint64 n, uint64 k, DType alpha,
                                  DType* a, int64 stride_a, DType* b,
                                  int64 stride_b, DType beta, DType* c,
                                  int64 stride_c, int batch_count) {
  int lda =
      (trans_a == perftools::gputools::blas::Transpose::kNoTranspose) ? k : m;
  int ldb =
      (trans_b == perftools::gputools::blas::Transpose::kNoTranspose) ? n : k;
  int ldc = n;
  auto* stream = context->op_device_context()->stream();
  bool blas_launch_status = stream->ThenBlasGemmStridedBatched(
      trans_b, trans_a, n, m, k, alpha, *AsDeviceMemory(b), ldb, stride_b,
      *AsDeviceMemory(a), lda, stride_a, beta, *AsDeviceMemory(c), ldc,
      stride_c, batch_count);
  if (!blas_launch_status) {
    context->SetStatus(errors::Internal(
        "Blas GemmStridedBatched launch failed : m=", m, ", n=", n, ", k=", k));
  }
}

template <typename Scalar>
struct LaunchParallelGemm<GPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, Scalar alpha, const Tensor& in_x,
                     const Tensor& in_y, Scalar beta, const Tensor& in_c,
                     Tensor* out, int64 batch_size) {
    printf("parallel_gemm cu debug: %s", in_c.DebugString());
    VLOG(2) << "parallel_gemm cu debug: " << in_c.DebugString();
    std::cout << "parallel_gemm cu debug: " << in_c.DebugString() << std::endl;
    std::cerr << "parallel_gemm cu debug: " << in_c.DebugString() << std::endl;
    const uint64 m = in_x.dims() == 3 ? in_x.dim_size(1) : in_x.dim_size(0);
    const uint64 k = in_x.dims() == 3 ? in_x.dim_size(2) : in_x.dim_size(1);
    const uint64 n = in_y.dim_size(1);
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    auto trans_a = perftools::gputools::blas::Transpose::kNoTranspose;
    auto trans_b = perftools::gputools::blas::Transpose::kNoTranspose;
    auto* a_base_ptr = in_x.template flat<Scalar>().data();
    auto* b_base_ptr = in_y.template flat<Scalar>().data();
    auto* out_base_ptr = out->template flat<Scalar>().data();
    const GPUDevice& gpu_device = context->eigen_gpu_device();
    if (beta != 0) {
      auto* c_base_ptr = in_c.template flat<Scalar>().data();
      auto out_shape = out->shape().dim_sizes();
      auto c_shape = in_c.shape().dim_sizes();
      out_shape[0] /= batch_size;
      c_shape[0] /= batch_size;
      // broadcast in_c to out;
      BatchedUBroadcast(out_base_ptr, batch_size, out_shape, c_base_ptr,
                        c_shape, gpu_device);
      std::cout << "in_c: " << in_c.DebugString() << std::endl;
      std::cout << "out: " << out->DebugString() << std::endl;
    }
    if (in_x.dims() == 3) {
      // Now Cublas is not supported, We sequentially run.
      for (int i = 0; i < batch_size; i++) {
        RunGemmStridedBatched(context, trans_a, trans_b, m, n, k, alpha,
                              a_base_ptr, m * k, b_base_ptr + i * k * n, 0,
                              beta, out_base_ptr + i * m * n * in_x.dim_size(0),
                              m * n, in_x.dim_size(0));
        std::cout << "Batched gemm" << out->DebugString() << std::endl;
      }
    } else {
      RunGemmStridedBatched(context, trans_a, trans_b, m, n, k, alpha,
                            a_base_ptr, 0, b_base_ptr, k * n, beta,
                            out_base_ptr, m * n, batch_size);
      std::cout << "out" << out->DebugString() << std::endl;
    }
  }
};

#endif  // GOOGLE_CUDA
}