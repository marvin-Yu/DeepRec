//
// Created by qiaoxj on 2019-09-06.
//

#ifndef TENSORFLOW_PARALLEL_GEMM_OP_H
#define TENSORFLOW_PARALLEL_GEMM_OP_H
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Scalar>
struct LaunchParallelGemm {
  void operator()(OpKernelContext* context, Scalar alpha, const Tensor& in_x,
                  const Tensor& in_y, Scalar beta, const Tensor& in_c,
                  Tensor* out, int64 batch_size);
};

#if GOOGLE_CUDA
template <typename Scalar>
struct LaunchParallelGemm<GPUDevice, Scalar> {
  void operator()(OpKernelContext* context, Scalar alpha, const Tensor& in_x,
                  const Tensor& in_y, Scalar beta, const Tensor& in_c,
                  Tensor* out, int64 batch_size);
};

#endif
}  // namespace tensorflow

#endif  // TENSORFLOW_PARALLEL_GEMM_OP_H
