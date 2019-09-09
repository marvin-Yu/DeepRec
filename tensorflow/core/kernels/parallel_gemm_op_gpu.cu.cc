//
// Created by qiaoxj on 2019-09-06.
//

#include "tensorflow/core/kernels/parallel_gemm_op.h"

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
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

// namespace

#define CUDA_KERNEL_LOOP(i, n)                                   \
  for (int64 i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_GET_BLOCKS(N, Threads) (N + Threads - 1) / Threads

namespace {
template <typename T>
inline se::DeviceMemory<T> AsDeviceMemory(const T* gpu_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(gpu_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

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

template <typename Scalar>
__global__ void BatchedUBroadcastKernel(Scalar* y, int64 batch_count,
                                        int64 y_size, const Scalar* x,
                                        int64 x_size) {
  int64 total_y_size = batch_count * y_size;
  CUDA_KERNEL_LOOP(index, total_y_size) {
    int batch_index = index / y_size;
    int batch_offset = index % y_size;
    y[index] = x[batch_index * x_size + batch_offset % x_size];
  }
}

template <typename Scalar>
void RunGemmStridedBatched(OpKernelContext* context, bool trans_a, bool trans_b,
                           int64 m, int64 n, int64 k, Scalar alpha,
                           const se::DeviceMemory<Scalar>& a, int64 stride_a,
                           const se::DeviceMemory<Scalar>& b, int64 stride_b,
                           Scalar beta, se::DeviceMemory<Scalar>* c,
                           int64 stride_c, int batch_count) {
  int lda = trans_a ? m : k;
  int ldb = trans_b ? k : n;
  int ldc = n;
  auto trans_a_tf = trans_a ? se::blas::Transpose::kTranspose
                            : se::blas::Transpose::kNoTranspose;
  auto trans_b_tf = trans_b ? se::blas::Transpose::kTranspose
                            : se::blas::Transpose::kNoTranspose;
  auto* stream = context->op_device_context()->stream();
  bool blas_launch_status =
      stream
          ->ThenBlasGemmStridedBatched(trans_b_tf, trans_a_tf, n, m, k, alpha,
                                       b, ldb, stride_b, a, lda, stride_a, beta,
                                       c, ldc, stride_c, batch_count)
          .ok();
  if (!blas_launch_status) {
    context->SetStatus(errors::Internal(
        "Blas GemmStridedBatched launch failed : m=", m, ", n=", n, ", k=", k));
  }
}

template <typename Scalar>
void LaunchParallelGemm<GPUDevice, Scalar>::operator()(
    OpKernelContext* context, Scalar alpha, const Tensor& in_x,
    const Tensor& in_y, Scalar beta, const Tensor& in_c, Tensor* out,
    int64 batch_size) {
  const int64 m = in_x.dims() == 3 ? in_x.dim_size(1) : in_x.dim_size(0);
  const int64 k = in_x.dims() == 3 ? in_x.dim_size(2) : in_x.dim_size(1);
  const int64 n = in_y.dim_size(1);
  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
  auto a_base_ptr = in_x.template flat<Scalar>().data();
  auto b_base_ptr = in_y.template flat<Scalar>().data();
  auto out_base_ptr = out->template flat<Scalar>().data();
  const GPUDevice& gpu_device = context->eigen_gpu_device();
  if (beta != 0) {
    auto c_base_ptr = in_c.template flat<Scalar>().data();
    auto out_shape = out->shape().dim_sizes();
    auto c_shape = in_c.shape().dim_sizes();
    out_shape[0] /= batch_size;
    c_shape[0] /= batch_size;
    // broadcast in_c to out;
    int64 out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                     std::multiplies<int64>());
    int64 c_size = std::accumulate(c_shape.begin(), c_shape.end(), 1,
                                   std::multiplies<int64>());
    int thread_num = GetThreadsNum(out_size * batch_size);
    int block_num =
        GetBlockNum(CUDA_GET_BLOCKS(out_size * batch_size, thread_num));
    BatchedUBroadcastKernel<Scalar>
        <<<block_num, thread_num, 0, gpu_device.stream()>>>(
            out_base_ptr, batch_size, out_size, c_base_ptr, c_size);
  }
  if (in_x.dims() == 3) {
    // Now Cublas is not supported, We sequentially run.
    for (int i = 0; i < batch_size; i++) {
      auto a_ptr = AsDeviceMemory(in_x.template flat<Scalar>().data());
      auto b_ptr =
          AsDeviceMemory(in_y.template flat<Scalar>().data() + i * k * n);
      auto out_ptr = AsDeviceMemory(out->template flat<Scalar>().data() +
                                    i * m * n * in_x.dim_size(0));
      RunGemmStridedBatched<Scalar>(context, false, false, m, n, k, alpha,
                                    a_ptr, m * k, b_ptr, 0, beta, &out_ptr,
                                    m * n, in_x.dim_size(0));
    }
  } else {
    auto a_ptr = AsDeviceMemory(in_x.template flat<Scalar>().data());
    auto b_ptr = AsDeviceMemory(in_y.template flat<Scalar>().data());
    auto out_ptr = AsDeviceMemory(out->template flat<Scalar>().data());
    RunGemmStridedBatched<Scalar>(context, false, false, m, n, k, alpha, a_ptr,
                                  0, b_ptr, k * n, beta, &out_ptr, m * n,
                                  batch_size);
  }
}

template struct LaunchParallelGemm<GPUDevice, float>;
template struct LaunchParallelGemm<GPUDevice, double>;
#endif  // GOOGLE_CUDA
}