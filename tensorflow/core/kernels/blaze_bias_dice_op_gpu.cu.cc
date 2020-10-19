//
// Created by luoxinchen on 2020-10-01
//
#include "tensorflow/core/kernels/blaze_bias_dice_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace {
template <typename Scalar, int TILE>
__global__ void ComputeBlazeBiasDice(const Scalar* input, const Scalar* bias,
                                     const Scalar* alpha,
                                     const Scalar* moving_mean,
                                     const Scalar* gamma, Scalar* output,
                                     int batch, int units) {
  int idx = blockIdx.x * blockDim.x * TILE + threadIdx.x;
#pragma unroll
  for (int _ = 0; _ < TILE; _++) {
    int b = idx / units;
    int u = idx % units;
    float rinput, rbias, ralpha, rmoving_mean, rgamma;
    if (b < batch) {
      rinput = (float)input[idx];
      rbias = (float)bias[u];
      ralpha = (float)alpha[u];
      rmoving_mean = (float)moving_mean[u];
      rgamma = (float)gamma[u];
      float fc_out = rinput + rbias;
      float bn_out = ralpha * (fc_out - rmoving_mean);
      float logits = (tanh(bn_out * 0.5f) + 1.0f) * 0.5f;
      float out = rgamma * (1.0f - logits) * fc_out + logits * fc_out;
      output[idx] = (Scalar)out;
    }
    idx += blockDim.x;
  }
}
}  // namespace

template <typename Scalar>
Status LaunchBlazeBiasDice<GPUDevice, Scalar>::operator()(
    OpKernelContext* context, const Tensor& input, const Tensor& bias,
    const Tensor& alpha, const Tensor& moving_mean, const Tensor& gamma,
    Tensor* output, int batch, int units) {
  const auto& d = context->eigen_device<GPUDevice>();
  const int thread_per_block = 256;
  const int tile_block = 4;
  const int elems_per_thread = thread_per_block * tile_block;
  dim3 grid_dim((batch * units + elems_per_thread - 1) / elems_per_thread);
  dim3 block_dim(thread_per_block);
  TF_CHECK_OK(GpuLaunchKernel(
      ComputeBlazeBiasDice<Scalar, tile_block>, grid_dim, block_dim, 0,
      d.stream(), input.template flat<Scalar>().data(),
      bias.template flat<Scalar>().data(), alpha.template flat<Scalar>().data(),
      moving_mean.template flat<Scalar>().data(),
      gamma.template flat<Scalar>().data(),
      output->template flat<Scalar>().data(), batch, units));
  return Status::OK();
}

template struct LaunchBlazeBiasDice<GPUDevice, Eigen::half>;
template struct LaunchBlazeBiasDice<GPUDevice, float>;
}  // namespace tensorflow

#endif  // GOOGLE_CUDA