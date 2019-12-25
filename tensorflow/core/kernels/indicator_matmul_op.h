//
// Created by qiaoxj on 2019-12-10.
//

#ifndef TENSORFLOW_INDICATOR_MATMUL_OP_H
#define TENSORFLOW_INDICATOR_MATMUL_OP_H
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#define GOOGLE_CUDA 1
template <typename Device, typename Scalar>
struct LaunchIndicatorMatmul {
  void operator()(OpKernelContext* context, bool trans_a, bool trans_b, int64 m,
                  int64 n, int64 k, const Tensor& in_a, const Tensor& in_b,
                  const Tensor& indicator, Tensor* out, int64 batch_a,
                  int64 batch_b, int64 paralle_num);
};

#if GOOGLE_CUDA
template <typename Scalar>
struct LaunchIndicatorMatmul<GPUDevice, Scalar> {
  void operator()(OpKernelContext* context, bool trans_a, bool trans_b, int64 m,
                  int64 n, int64 k, const Tensor& in_a, const Tensor& in_b,
                  const Tensor& indicator, Tensor* out, int64 batch_a,
                  int64 batch_b, int64 paralle_num);
};

#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_INDICATOR_MATMUL_OP_H
