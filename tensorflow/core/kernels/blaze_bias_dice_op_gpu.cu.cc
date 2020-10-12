//
// Created by luoxinchen on 2020-10-01
//

#include "tensorflow/core/platform/stream_executor.h"

#define EIGEN_USE_GPU
#if GOOGLE_CUDA
#include "tensorflow/core/kernels/blaze_bias_dice_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace {
template <typename Scalar>
__global__ void ComputeBlazeBiasDice(const Scalar* input, const Scalar* bias,
                                     const Scalar* alpha,
                                     const Scalar* moving_mean,
                                     const Scalar* gamma, Scalar* output,
                                     int batch, int units) {
  const int b = blockIdx.x;
  int units_idx = blockIdx.y * blockDim.x + threadIdx.x;
  input = input + b * units;
  if (units_idx < units) {
    float fc_out = (float)input[units_idx] + (float)bias[units_idx];
    float bn_out =
        (float)alpha[units_idx] * (fc_out - (float)moving_mean[units_idx]);
    float logits = (__tanhf(bn_out / 2.0f) + 1.0f) / 2.0f;
    float out =
        (float)gamma[units_idx] * (1.0f - logits) * fc_out + logits * fc_out;
    output[b * units + units_idx] = (Scalar)out;
  }
}

}  // namespace

template <typename Scalar>
Status LaunchBlazeBiasDice<GPUDevice, Scalar>::operator()(
    OpKernelContext* context, const Tensor& input, const Tensor& bias,
    const Tensor& alpha, const Tensor& moving_mean, const Tensor& gamma,
    Tensor* output, int batch, int units) {
  dim3 grid_dim(batch, (units + 127) / 128);
  dim3 block_dim(128);
  const auto& d = context->eigen_device<GPUDevice>();
  TF_CHECK_OK(GpuLaunchKernel(
      ComputeBlazeBiasDice<Scalar>, grid_dim, block_dim, 0, d.stream(),
      input.template flat<Scalar>().data(), bias.template flat<Scalar>().data(),
      alpha.template flat<Scalar>().data(),
      moving_mean.template flat<Scalar>().data(),
      gamma.template flat<Scalar>().data(),
      output->template flat<Scalar>().data(), batch, units));
  return Status::OK();
}

template struct LaunchBlazeBiasDice<GPUDevice, Eigen::half>;
template struct LaunchBlazeBiasDice<GPUDevice, float>;
}  // namespace tensorflow

#endif // GOOGLE_CUDA