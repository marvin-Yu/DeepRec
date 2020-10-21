//
// Created by luoxinchen on 2020-10-10
//

#include "tensorflow/core/framework/op_kernel.h"

#ifndef TENSORFLOW_BLAZE_BIAS_DICE_OP_H
#define TENSORFLOW_BLAZE_BIAS_DICE_OP_H

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Scalar>
struct LaunchBlazeBiasDice {
  Status operator()(OpKernelContext* context, const Tensor& input,
                    const Tensor& bias, const Tensor& alpha,
                    const Tensor& moving_mean, const Tensor& gamma,
                    Tensor* output, int batch, int units);
};

#if GOOGLE_CUDA
template <typename Scalar>
struct LaunchBlazeBiasDice<GPUDevice, Scalar> {
  Status operator()(OpKernelContext* context, const Tensor& input,
                    const Tensor& bias, const Tensor& alpha,
                    const Tensor& moving_mean, const Tensor& gamma,
                    Tensor* output, int batch, int units);
};
#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_BLAZE_ATTENTION_OP_H