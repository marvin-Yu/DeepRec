//
// Created by lxc263790 on 2020-08-17.
//

#include "tensorflow/core/kernels/indicator_batched_small_matmul_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace {
typedef Eigen::ThreadPoolDevice CPUDevice;
template <bool use_tanh, typename T>
void Gemm(const CPUDevice& d, size_t m, size_t n, size_t k, const T* a,
          const T* b, T* c, bool trans_a, bool trans_b) {
  auto a0 = trans_a ? k : m;
  auto a1 = trans_a ? m : k;
  auto b0 = trans_b ? n : k;
  auto b1 = trans_b ? k : n;
  typename tensorflow::TTypes<const T>::Matrix a_matrix(a, a0, a1);
  typename tensorflow::TTypes<const T>::Matrix b_matrix(b, b0, b1);
  typename tensorflow::TTypes<T>::Matrix c_matrix(c, m, n);

  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
  dim_pair[0].first = trans_a ? 0 : 1;
  dim_pair[0].second = trans_b ? 1 : 0;
  c_matrix.device(d) = a_matrix.contract(b_matrix, dim_pair);
  if (use_tanh) {
    for (int i = 0; i < m * n; i++) {
      c[i] = T(tanh(float(c[i])));
    }
  }
}
}  // namespace

namespace tensorflow {
namespace indicator_batched_small_matmul {
template <bool use_tanh, typename Scalar, typename TIndex>
struct LaunchIndicatorBatchedSmallMatmul<CPUDevice, use_tanh, Scalar, TIndex> {
  Status operator()(OpKernelContext* context, bool trans_a, bool trans_b,
                    int64 m, int64 n, int64 k, const Tensor& in_a,
                    const Tensor& in_b, const Tensor& indicator, Tensor* out,
                    int64 batch_a, int64 batch_b, int64 paralle_num) {
    auto a_ptr = in_a.template flat<Scalar>().data();
    auto b_ptr = in_b.template flat<Scalar>().data();
    auto c_ptr = out->template flat<Scalar>().data();
    auto ind_ptr = indicator.template flat<TIndex>().data();
    for (int64 p = 0; p < paralle_num; p++) {
      for (int64 batch = 0; batch < batch_b; batch++) {
        Gemm<use_tanh, Scalar>(context->eigen_device<CPUDevice>(), m, n, k,
                               a_ptr + (p * batch_a + ind_ptr[batch]) * m * k,
                               b_ptr + (p * batch_b + batch) * k * n,
                               c_ptr + (p * batch_b + batch) * m * n, trans_a,
                               trans_b);
      }
    }
    return Status::OK();
  }
};

template <typename Device, typename Scalar, typename TIndex>
class ParallelIndicatorBatchedSmallMatmulOp : public OpKernel {
 public:
  explicit ParallelIndicatorBatchedSmallMatmulOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &trans_a_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &trans_b_));
    OP_REQUIRES_OK(context, context->GetAttr("use_tanh", &use_tanh_));
    OP_REQUIRES_OK(context, context->GetAttr("parallel_num", &parallel_num));
  }

  ~ParallelIndicatorBatchedSmallMatmulOp() = default;

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
    OP_REQUIRES(ctx, parallel_num >= 1,
                errors::InvalidArgument("parallel_num must >= 1:", ind.dims()));
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
    if (use_tanh_) {
      OP_REQUIRES_OK(
          ctx,
          LaunchIndicatorBatchedSmallMatmul<Device, true, Scalar, TIndex>()(
              ctx, trans_a_, trans_b_, d0, d3, d1, a, b, ind, out, batch_a,
              batch_b, parallel_num));
    } else {
      OP_REQUIRES_OK(
          ctx,
          LaunchIndicatorBatchedSmallMatmul<Device, false, Scalar, TIndex>()(
              ctx, trans_a_, trans_b_, d0, d3, d1, a, b, ind, out, batch_a,
              batch_b, parallel_num));
    }
  }

 private:
  bool trans_a_;
  bool trans_b_;
  bool use_tanh_;
  int64 parallel_num;
};

#define REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_CPU(TYPE, TIndex) \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("ParallelIndicatorBatchedSmallMatMul")                 \
          .Device(DEVICE_CPU)                                     \
          .TypeConstraint<TYPE>("T")                              \
          .TypeConstraint<TIndex>("Tindices"),                    \
      ParallelIndicatorBatchedSmallMatmulOp<CPUDevice, TYPE, TIndex>);

#define REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_CPU_ALL_INDICES(type) \
  REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_CPU(type, int32);           \
  REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_CPU(type, int64);

REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_CPU_ALL_INDICES(float);
REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_CPU_ALL_INDICES(double);
REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_CPU_ALL_INDICES(Eigen::half);

#undef REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_CPU_ALL_INDICES
#undef REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_CPU

#if GOOGLE_CUDA
#define REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_GPU(TYPE, TIndex)            \
  extern template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, false, \
                                                           TYPE, TIndex>;    \
  extern template struct LaunchIndicatorBatchedSmallMatmul<GPUDevice, true,  \
                                                           TYPE, TIndex>;    \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ParallelIndicatorBatchedSmallMatMul")                            \
          .Device(DEVICE_GPU)                                                \
          .TypeConstraint<TYPE>("T")                                         \
          .TypeConstraint<TIndex>("Tindices"),                               \
      ParallelIndicatorBatchedSmallMatmulOp<GPUDevice, TYPE, TIndex>);

#define REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_GPU_ALL_INDICES(type) \
  REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_GPU(type, int32);           \
  REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_GPU(type, int64);

REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_GPU_ALL_INDICES(float);
REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_GPU_ALL_INDICES(double);
REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_GPU_ALL_INDICES(Eigen::half);

#undef REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_GPU_ALL_INDICES
#undef REGISTER_INDICATOR_BATCHED_SMALL_MATMUL_GPU
#endif  // GOOGLE_CUDA
}  // namespace indicator_batched_small_matmul
}  // namespace tensorflow
