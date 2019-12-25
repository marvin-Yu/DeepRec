//
// Created by qiaoxj on 2019-12-10.
//

#include "tensorflow/core/kernels/indicator_matmul_op.h"

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

namespace tensorflow {
template <typename Scalar>
struct LaunchIndicatorMatmul<CPUDevice, Scalar> {
  void operator()(OpKernelContext* context, bool trans_a, bool trans_b, int64 m,
                  int64 n, int64 k, const Tensor& in_a, const Tensor& in_b,
                  const Tensor& indicator, Tensor* out, int64 batch_a,
                  int64 batch_b, int64 paralle_num) {}
};

template <typename Device, typename Scalar>
class IndicatorMatmulOp : public OpKernel {
 public:
  explicit IndicatorMatmulOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &trans_a_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &trans_b_));
  }

  ~IndicatorMatmulOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& a = ctx->input(0);
    auto& b = ctx->input(1);
    auto& ind = ctx->input(2);

    OP_REQUIRES(ctx, a.dims() == 3,
                errors::InvalidArgument("In[0] ndims must be 3: ", a.dims()));
    OP_REQUIRES(ctx, b.dims() == 3,
                errors::InvalidArgument("In[1] ndims must be 3: ", b.dims()));
    OP_REQUIRES(ctx, ind.dims() == 1,
                errors::InvalidArgument("In[2] ndims must be 1: ", ind.dims()));

    int64 d0 = a.dim_size(1);
    int64 d1 = a.dim_size(2);
    if (trans_a_) {
      std::swap(d0, d1);
    }
    int64 d2 = b.dim_size(1);
    int64 d3 = b.dim_size(2);
    if (trans_b_) {
      std::swap(d2, d3);
    }
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("a mismatch b shape: ", d1, " vs. ", d2,
                                        ": ", a.shape().DebugString(), " ",
                                        b.shape().DebugString()));
    int64 batch_a = a.dim_size(0);
    int64 batch_b = b.dim_size(0);
    int64 ind_length = ind.dim_size(0);
    OP_REQUIRES(
        ctx, batch_b == ind_length,
        errors::InvalidArgument(
            "b_batch mismatch indicator length: ", batch_b, " vs. ", ind_length,
            ": ", b.shape().DebugString(), " ", ind.shape().DebugString()));

    TensorShape out_shape;
    out_shape.AddDim(batch_b);
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);
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
    LaunchIndicatorMatmul<Device, Scalar>()(ctx, trans_a_, trans_b_, d0, d3, d1,
                                            a, b, ind, out, batch_a, batch_b,
                                            1);
  }

 private:
  bool trans_a_;
  bool trans_b_;
};

template <typename Device, typename Scalar>
class ParallelIndicatorMatmulOp : public OpKernel {
 public:
  explicit ParallelIndicatorMatmulOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &trans_a_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &trans_b_));
    OP_REQUIRES_OK(context, context->GetAttr("parallel_num", &parallel_num));
  }

  ~ParallelIndicatorMatmulOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& a = ctx->input(0);
    auto& b = ctx->input(1);
    auto& ind = ctx->input(2);

    OP_REQUIRES(ctx, a.dims() == 4,
                errors::InvalidArgument("In[0] ndims must be 4: ", a.dims()));
    OP_REQUIRES(ctx, b.dims() == 4,
                errors::InvalidArgument("In[1] ndims must be 4: ", b.dims()));
    OP_REQUIRES(ctx, ind.dims() == 1,
                errors::InvalidArgument("In[2] ndims must be 1: ", ind.dims()));
    OP_REQUIRES(
        ctx, parallel_num >= 1,
        errors::InvalidArgument("parallel_num must >= 1: ", ind.dims()));

    int64 d0 = a.dim_size(2);
    int64 d1 = a.dim_size(3);
    if (trans_a_) {
      std::swap(d0, d1);
    }
    int64 d2 = b.dim_size(2);
    int64 d3 = b.dim_size(3);
    if (trans_b_) {
      std::swap(d2, d3);
    }
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("a mismatch b shape: ", d1, " vs. ", d2,
                                        ": ", a.shape().DebugString(), " ",
                                        b.shape().DebugString()));
    int64 parallel_a = a.dim_size(0);
    int64 parallel_b = b.dim_size(0);
    OP_REQUIRES(ctx, parallel_a == parallel_b,
                errors::InvalidArgument(
                    "parallel_a mismatch parallel_b : ", parallel_a, " vs. ",
                    parallel_b, ": ", a.shape().DebugString(), " ",
                    b.shape().DebugString()));
    int64 batch_a = a.dim_size(1);
    int64 batch_b = b.dim_size(1);
    int64 ind_length = ind.dim_size(0);
    OP_REQUIRES(
        ctx, batch_b == ind_length,
        errors::InvalidArgument(
            "b_batch mismatch indicator length: ", batch_b, " vs. ", ind_length,
            ": ", b.shape().DebugString(), " ", ind.shape().DebugString()));

    TensorShape out_shape;
    out_shape.AddDim(parallel_a);
    out_shape.AddDim(batch_b);
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);
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
    LaunchIndicatorMatmul<Device, Scalar>()(ctx, trans_a_, trans_b_, d0, d3, d1,
                                            a, b, ind, out, batch_a, batch_b,
                                            parallel_num);
  }

 private:
  bool trans_a_;
  bool trans_b_;
  int64 parallel_num;
};

#define REGISTER_INDICATOR_MATMUL_CPU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("IndicatorMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      IndicatorMatmulOp<CPUDevice, TYPE>);
#define REGISTER_PARALLEL_INDICATOR_MATMUL_CPU(TYPE)      \
  REGISTER_KERNEL_BUILDER(Name("ParallelIndicatorMatMul") \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          ParallelIndicatorMatmulOp<CPUDevice, TYPE>);
REGISTER_INDICATOR_MATMUL_CPU(float);
REGISTER_INDICATOR_MATMUL_CPU(double);
REGISTER_PARALLEL_INDICATOR_MATMUL_CPU(float);
REGISTER_PARALLEL_INDICATOR_MATMUL_CPU(double);

#if GOOGLE_CUDA
#define REGISTER_INDICATOR_MATMUL_GPU(TYPE)                      \
  extern template struct LaunchIndicatorMatmul<GPUDevice, TYPE>; \
  REGISTER_KERNEL_BUILDER(Name("IndicatorMatMul")                \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("indicator")           \
                              .TypeConstraint<TYPE>("T"),        \
                          IndicatorMatmulOp<GPUDevice, TYPE>);
#define REGISTER_PARALLEL_INDICATOR_MATMUL_GPU(TYPE)      \
  REGISTER_KERNEL_BUILDER(Name("ParallelIndicatorMatMul") \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("indicator")    \
                              .TypeConstraint<TYPE>("T"), \
                          ParallelIndicatorMatmulOp<GPUDevice, TYPE>);
REGISTER_INDICATOR_MATMUL_GPU(float);
REGISTER_INDICATOR_MATMUL_GPU(double);
REGISTER_PARALLEL_INDICATOR_MATMUL_GPU(float);
REGISTER_PARALLEL_INDICATOR_MATMUL_GPU(double);
#endif

}  // namespace tensorflow
