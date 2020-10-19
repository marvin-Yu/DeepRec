//
// Created by luoxinchen on 2020-10-01
//

#ifndef TENSORFLOW_BLAZE_ATTENTION_OP_H
#define TENSORFLOW_BLAZE_ATTENTION_OP_H

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Scalar>
struct LaunchBlazeAttention {
  Status operator()(OpKernelContext* context, const Tensor& in_fact,
                    const Tensor& in_query, Tensor* out, int pnum,
                    int batch_fact, int batch_query, int seq_len, int units);
};
template <typename Device, typename Scalar, typename TIndex>
struct LaunchBlazeAttentionIndicator {
  Status operator()(OpKernelContext* context, const Tensor& in_fact,
                    const Tensor& in_query, const Tensor& in_indicators,
                    Tensor* out, int pnum, int batch_fact, int batch_query,
                    int seq_len, int units);
};

#if GOOGLE_CUDA
template <typename Scalar>
struct LaunchBlazeAttention<GPUDevice, Scalar> {
  Status operator()(OpKernelContext* context, const Tensor& in_fact,
                    const Tensor& in_query, Tensor* out, int pnum,
                    int batch_fact, int batch_query, int seq_len, int units);
};
template <typename Scalar, typename TIndex>
struct LaunchBlazeAttentionIndicator<GPUDevice, Scalar, TIndex> {
  Status operator()(OpKernelContext* context, const Tensor& in_fact,
                    const Tensor& in_query, const Tensor& in_indicators,
                    Tensor* out, int pnum, int batch_fact, int batch_query,
                    int seq_len, int units);
};
#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_BLAZE_ATTENTION_OP_H