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
  static void Launch(OpKernelContext* context, Scalar alpha, const Tensor& in_x,
                     const Tensor& in_y, Scalar beta, const Tensor& in_c,
                     Tensor* out, int64 batch_size) {}
};

#define REGISTER_PARALLEL_GEMM_GPU(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ParallelGemm").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),   \
      ParallelGemmlOp<GPUDevice, TYPE>);

#define REGISTER_PARALLEL_GEMM_CPU(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ParallelGemm").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      ParallelGemmlOp<GPUDevice, TYPE>);

}  // namespace tensorflow

#endif  // TENSORFLOW_PARALLEL_GEMM_OP_H
