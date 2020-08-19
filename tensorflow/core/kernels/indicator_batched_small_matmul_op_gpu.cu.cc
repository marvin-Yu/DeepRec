//
// Created by lxc263790 on 2020-08-17.
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/indicator_batched_small_matmul_op.h"
#include "tensorflow/core/kernels/indicator_matmul_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
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
namespace indicator_batched_small_matmul {
template <typename Scalar, typename TIndex>
struct IMatmulParam {
  Scalar* A;
  Scalar* B;
  Scalar* C;
  TIndex* indicators;
  int m, n, k;
  int batch_a, batch_b;
};

template <bool use_tanh, typename Scalar, typename TIndex, int KSIZE, int NSIZE>
__global__ void ComputeIndicatorBatchedSmallMatmulKernel(
    IMatmulParam<Scalar, TIndex> param) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int toff = bz * KSIZE * NSIZE + tx * NSIZE + ty;

  // Bound check
  if (toff >= param.m) return;
  __shared__ Scalar Bs[KSIZE][NSIZE];

  int ind = (int)param.indicators[by];
  if (ind < 0 || ind >= param.batch_a) {
    ind = 0;
  }
  Scalar* A = param.A + bx * (param.batch_a * param.m * KSIZE) +
              ind * (param.m * KSIZE) + toff * KSIZE;
  Scalar* B =
      param.B + bx * (param.batch_b * KSIZE * NSIZE) + by * (KSIZE * NSIZE);

  // load B matrix
  Bs[tx][ty] = B[tx * NSIZE + ty];

  __syncthreads();

  float Csub[NSIZE] = {0};
#pragma unroll
  for (int k = 0; k < KSIZE; k++) {
    Scalar a_val = A[k];
#pragma unroll
    for (int j = 0; j < NSIZE; j++) {
      Csub[j] += float(a_val) * float(Bs[k][j]);
    }
  }

  Scalar* C = param.C + bx * (param.batch_b * param.m * NSIZE) +
              by * (param.m * NSIZE) + toff * NSIZE;

#pragma unroll
  for (int j = 0; j < NSIZE; j++) {
    if (use_tanh) {
      C[j] = Scalar(tanh(Csub[j]));
    } else {
      C[j] = Scalar(Csub[j]);
    }
  }
}

template <typename Scalar>
__global__ void ComputeInplaceTanhKernel(Scalar* src, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Bound check
  if (idx >= size) return;
  src[idx] = Scalar(tanh(float(src[idx])));
}

template <bool use_tanh, typename Scalar, typename TIndex>
Status LaunchIndicatorBatchedSmallMatmul<GPUDevice, use_tanh, Scalar, TIndex>::
operator()(OpKernelContext* context, bool trans_a, bool trans_b, int64 m,
           int64 n, int64 k, const Tensor& in_a, const Tensor& in_b,
           const Tensor& indicator, Tensor* out, int64 batch_a, int64 batch_b,
           int64 parallel_num) {
  IMatmulParam<Scalar, TIndex> param;
  param.A = const_cast<Scalar*>(in_a.template flat<Scalar>().data());
  param.B = const_cast<Scalar*>(in_b.template flat<Scalar>().data());
  param.C = out->template flat<Scalar>().data();
  param.indicators =
      const_cast<TIndex*>(indicator.template flat<TIndex>().data());
  param.m = m, param.n = n, param.k = k;
  param.batch_a = batch_a, param.batch_b = batch_b;
  dim3 grid_dim(parallel_num, batch_b, (m + k * n - 1) / (k * n));
  dim3 block_dim(k, n);

  if (trans_a || trans_b) {
    LaunchIndicatorMatmul<GPUDevice, Scalar, TIndex>()(
        context, trans_a, trans_b, m, n, k, in_a, in_b, indicator, out, batch_a,
        batch_b, parallel_num);
    return Status::OK();
  }

  const auto& d = context->eigen_device<GPUDevice>();
  if (k == 5 && n == 4) {
    // hold small matmul cases
    TF_CHECK_OK(GpuLaunchKernel(
        ComputeIndicatorBatchedSmallMatmulKernel<use_tanh, Scalar, TIndex, 5,
                                                 4>,
        grid_dim, block_dim, 5 * 4 * sizeof(Scalar), d.stream(), param));
    return Status::OK();
  } else {
    // hold general cases
    LaunchIndicatorMatmul<GPUDevice, Scalar, TIndex>()(
        context, trans_a, trans_b, m, n, k, in_a, in_b, indicator, out, batch_a,
        batch_b, parallel_num);
    if (use_tanh) {
      const int work_elem_count = parallel_num * batch_b * m * n;
      const int thread_per_block = std::min(1024, d.maxGpuThreadsPerBlock());
      const int blocks =
          (work_elem_count + thread_per_block - 1) / thread_per_block;
      TF_CHECK_OK(GpuLaunchKernel(ComputeInplaceTanhKernel<Scalar>, dim3(blocks),
                                  dim3(thread_per_block), 0, d.stream(),
                                  param.C, work_elem_count));
    }
    return Status::OK();
  }
  return Status::OK();
}  // namespace tensorflow

template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, false, float,
                                                  int32>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, false, double,
                                                  int32>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, false, Eigen::half,
                                                  int32>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, true, float,
                                                  int32>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, true, double,
                                                  int32>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, true, Eigen::half,
                                                  int32>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, false, float,
                                                  int64>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, false, double,
                                                  int64>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, false, Eigen::half,
                                                  int64>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, true, float,
                                                  int64>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, true, double,
                                                  int64>;
template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, true, Eigen::half,
                                                  int64>;
}  // namespace indicator_batched_small_matmul
}  // namespace tensorflow
#endif  // GOOGLE_CUDA