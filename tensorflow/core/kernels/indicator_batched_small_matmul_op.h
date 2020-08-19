//
// Created by lxc263790 on 2020-08-17.
//

#ifndef TENSORFLOW_INDICATOR_BATCHED_SMALL_MATMUL_OP_H
#define TENSORFLOW_INDICATOR_BATCHED_SMALL_MATMUL_OP_H
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace indicator_batched_small_matmul {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, bool use_tanh, typename Scalar, typename TIndex>
struct LaunchIndicatorBatchedSmallMatmul {
  void operator()(OpKernelContext* context, bool trans_a_, bool trans_b_,
                  int64 m, int64 n, int64 k, const Tensor& in_a,
                  const Tensor& in_b, const Tensor& indicator, Tensor* out,
                  int64 batch_a, int64 batch_b, int64 parallel_num);
};

#if GOOGLE_CUDA
template <bool use_tanh, typename Scalar, typename TIndex>
struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, use_tanh, Scalar, TIndex> {
  void operator()(OpKernelContext* context, bool trans_a, bool trans_b, int64 m,
                  int64 n, int64 k, const Tensor& in_a, const Tensor& in_b,
                  const Tensor& indicator, Tensor* out, int64 batch_a,
                  int64 batch_b, int64 parallel_num);
};
#endif

}  // namespace IndicatorBatchedSmallMatmul
}  // namespace tensorflow

#endif  // TENSORFLOW_INDICATOR_BATCHED_SMALL_MATMUL_OP_H