//
// Created by luoxinchen on 2020/10/10.
//

#include "tensorflow/core/kernels/blaze_bias_dice_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// template <typename T>
// struct BlazeBiasDiceCPUFunctor {
//   void operator()(const CPUDevice& d, const T* input, const T* bias,
//                   const T* alpha, const T* moving_mean, const T* gamma,
//                   T* output, int batch, int units) {
//     using InMatrix = typename tensorflow::TTypes<const T>::Matrix;
//     using OutMatrix = typename tensorflow::TTypes<T>::Matrix;
//     Eigen::DSizes<int, 2> batch_by_one(batch, 1);
//     InMatrix minput(input, batch, units);
//     InMatrix mbias(bias, 1, units);
//     InMatrix malpha(alpha, 1, units);
//     InMatrix mmoving_mean(moving_mean, 1, units);
//     InMatrix mgamma(gamma, 1, units);
//     OutMatrix mout(output, batch, units);
//     auto cinput = minput.template cast<float>();
//     auto cbias = mbias.template cast<float>().broadcast(batch_by_one);
//     auto calpha = malpha.template cast<float>().broadcast(batch_by_one);
//     auto cmoving_mean =
//         mmoving_mean.template cast<float>().broadcast(batch_by_one);
//     auto cgamma = mgamma.template cast<float>().broadcast(batch_by_one);
//     auto fc_out = cinput + cbias;
//     auto bn_out = calpha * (fc_out - cmoving_mean);
//     auto logits = (1.0f + (-1.0f * bn_out).exp()).inverse();
//     auto out = cgamma * (1.0f - logits) * fc_out + logits * fc_out;
//     mout.device(d) = out.template cast<T>().eval();
//   }
// };

template <typename T>
struct BlazeBiasDiceCPUFunctor {
  void operator()(const CPUDevice& d, const T* input, const T* bias,
                  const T* alpha, const T* moving_mean, const T* gamma,
                  T* output, int batch, int units) {
    for (int b = 0; b < batch; b++) {
      for (int i = 0; i < units; i++) {
        float fc_out = (float)*input + (float)bias[i];
        float bn_out = (float)alpha[i] * (fc_out - (float)moving_mean[i]);
        float logits = 1.0f / (1.0f + std::exp(-bn_out));
        float out = (float)gamma[i] * (1.0f - logits) * fc_out + logits * fc_out;
        *output = (T)out;
        output++;
        input++;
      }
    }
  }
};
}  // namespace functor




template <typename Scalar>
struct LaunchBlazeBiasDice<CPUDevice, Scalar> {
  Status operator()(OpKernelContext* context, const Tensor& input,
                    const Tensor& bias, const Tensor& alpha,
                    const Tensor& moving_mean, const Tensor& gamma,
                    Tensor* output, int batch, int units) {
    functor::BlazeBiasDiceCPUFunctor<Scalar> functor;
    functor(context->eigen_device<CPUDevice>(),
            input.template flat<Scalar>().data(),
            bias.template flat<Scalar>().data(),
            alpha.template flat<Scalar>().data(),
            moving_mean.template flat<Scalar>().data(),
            gamma.template flat<Scalar>().data(),
            output->template flat<Scalar>().data(), batch, units);
    return Status::OK();
  }
};

template <typename Device, typename Scalar>
class BlazeBiasDiceOp : public OpKernel {
 public:
  explicit BlazeBiasDiceOp(OpKernelConstruction* context) : OpKernel(context) {}

  ~BlazeBiasDiceOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& input = ctx->input(0);
    auto& bias = ctx->input(1);
    auto& alpha = ctx->input(2);
    auto& moving_mean = ctx->input(3);
    auto& gamma = ctx->input(4);
    OP_REQUIRES(
        ctx, input.dims() == 2,
        errors::InvalidArgument("In[0] ndims must be 2: ", input.dims()));
    int batch = input.dim_size(0);
    int units = input.dim_size(1);
    Tensor* out = nullptr;
    TensorShape out_shape;
    out_shape.AddDim(batch);
    out_shape.AddDim(units);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    LaunchBlazeBiasDice<Device, Scalar> launch;
    launch(ctx, input, bias, alpha, moving_mean, gamma,
           out, batch, units);
  }
};

#define REGISTER_CPU(T)                                                \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("BlazeBiasDice").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BlazeBiasDiceOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                \
  extern template struct LaunchBlazeBiasDice<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("BlazeBiasDice").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BlazeBiasDiceOp<GPUDevice, T>);
TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
#undef REGISTER_GPU
#endif

}  // namespace tensorflow