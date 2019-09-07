//
// Created by qiaoxj on 2019-09-06.
//

#include "parallel_gemm_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace tensorflow {

template <typename Scalar>
struct LaunchParallelGemm<CPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, Scalar alpha, const Tensor& in_x,
                     const Tensor& in_y, Scalar beta, const Tensor& in_c,
                     Tensor* out, int64 batch_size) {}
};

template <typename Device, typename Scalar>
class ParallelGemmlOp : public OpKernel {
 public:
  explicit ParallelGemmlOp(OpKernelConstruction* context) : OpKernel(context) {
    float alpha, beta;
    int parallel_num;
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta));
    OP_REQUIRES_OK(context, context->GetAttr("parallel_num", &parallel_num));
    alpha_ = Scalar(alpha);
    beta_ = Scalar(beta);
    parallel_num_ = int64(parallel_num);
  }

  ~ParallelGemmlOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& a = ctx->input(0);
    auto& b = ctx->input(1);

    OP_REQUIRES(ctx, a.dims() >= 2 && a.dims() <= 3,
                errors::InvalidArgument("In[0] ndims must 2 or 3: ", a.dims()));
    OP_REQUIRES(ctx, b.dims() == 2,
                errors::InvalidArgument("In[1] ndims must be 2: ", b.dims()));
    OP_REQUIRES(
        ctx, parallel_num_ >= 2,
        errors::InvalidArgument("parallel_num must >= 2: ", parallel_num_));
    int64 d0, d1, d2, d3;
    TensorShape out_shape;
    if (a.dims() == 3) {
      int64 batch_dim = parallel_num_ * a.dim_size(0);
      d0 = a.dim_size(1);
      d1 = a.dim_size(2);
      d2 = b.dim_size(0) / parallel_num_;
      d3 = b.dim_size(1);
      out_shape.AddDim(batch_dim);
      out_shape.AddDim(d0);
      out_shape.AddDim(d3);
    } else {
      d0 = a.dim_size(0);
      d1 = a.dim_size(1);
      d2 = b.dim_size(0) / parallel_num_;
      d3 = b.dim_size(1);
      out_shape.AddDim(d0 * parallel_num_);
      out_shape.AddDim(d3);
    }
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("a mismatch b shape: ", d1, " vs. ", d2,
                                        ": ", a.shape().DebugString(), " ",
                                        b.shape().DebugString()));
    VLOG(2) << "parallel_gemm debug: " << d0 << d1 << d2 << d3;
    std::cout << "parallel_gemm debug: " << d0 << d1 << d2 << d3 << std::endl;
    std::cerr << "parallel_gemm debug: " << d0 << d1 << d2 << d3 << std::endl;
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (a.NumElements() == 0 || b.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }

    if (beta_ == 0.0) {
      Tensor c;
      LaunchParallelGemm<Device, Scalar>::Launch(ctx, alpha_, a, b, beta_, c,
                                                 out, parallel_num_);
    } else {
      const Tensor& c = ctx->input(2);
      LaunchParallelGemm<Device, Scalar>::Launch(ctx, alpha_, a, b, beta_, c,
                                                 out, parallel_num_);
    }
  };

 private:
  int64 parallel_num_;
  Scalar alpha_, beta_;
};

TF_CALL_float(REGISTER_PARALLEL_GEMM_CPU);
TF_CALL_double(REGISTER_PARALLEL_GEMM_CPU);
#if GOOGLE_CUDA
TF_CALL_float(REGISTER_PARALLEL_GEMM_GPU);
TF_CALL_double(REGISTER_PARALLEL_GEMM_GPU);
#endif
}