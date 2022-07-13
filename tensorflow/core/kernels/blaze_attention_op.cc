//
// Created by luoxinchen on 2020/10/01.
//

#include "tensorflow/core/kernels/blaze_attention_op.h"

#include "tensorflow/core/framework/op.h"
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

template <typename T>
void BlazeAttentionFunctor(const CPUDevice& d, const T* fact, const T* query,
                           T* out, size_t seq_len, size_t units) {
  using InMatrix = typename tensorflow::TTypes<const T>::Matrix;
  using OutMatrix = typename tensorflow::TTypes<T>::Matrix;
  InMatrix mfact(fact, seq_len, units);
  InMatrix mquery(query, 1, units);
  OutMatrix mout(out, units);
  Eigen::DSizes<int, 1> along_units(1);
  Eigen::DSizes<int, 2> one_by_seq(seq_len, 1);
  auto logits = (mfact * mquery.broadcast(one_by_seq)).sum(along_units);
  Eigen::DSizes<int, 1> s_along_units(0);
  Eigen::DSizes<int, 2> s_softmax(seq_len, 1);
  Eigen::DSizes<int, 2> s_units_by_one(1, units);
  Eigen::DSizes<int, 1> s_along_seq(0);
  Eigen::DSizes<int, 1> s_logits_shape(1);
  Eigen::DSizes<int, 1> s_logits_broadcast(seq_len);
  auto shifted_logits = (logits - logits.maximum(s_along_units)
                                      .eval()
                                      .reshape(s_logits_shape)
                                      .broadcast(s_logits_broadcast));
  auto tmp = shifted_logits.exp();
  auto softmax = tmp * tmp.sum(s_along_units)
                           .inverse()
                           .eval()
                           .reshape(s_logits_shape)
                           .broadcast(s_logits_broadcast);
  mout.device(d) =
      (mfact * softmax.reshape(s_softmax).broadcast(s_units_by_one))
          .sum(s_along_seq)
          .eval();
}

template <typename Scalar>
struct LaunchBlazeAttention<CPUDevice, Scalar> {
  Status operator()(OpKernelContext* context, const Tensor& in_fact,
                    const Tensor& in_query, Tensor* out, int pnum,
                    int batch_fact, int batch_query, int seq_len, int units) {
    if (batch_fact != 1) {
      return errors::InvalidArgument("batch_fact must be 1: ", batch_fact);
    }
    auto fact_ptr = in_fact.template flat<Scalar>().data();
    auto query_ptr = in_query.template flat<Scalar>().data();
    auto out_ptr = out->template flat<Scalar>().data();
    for (int p = 0; p < pnum; p++) {
      for (int b = 0; b < batch_query; b++) {
        BlazeAttentionFunctor(context->eigen_device<CPUDevice>(),
                              fact_ptr + p * seq_len * units,
                              query_ptr + (p * batch_query + b) * units,
                              out_ptr + (b * pnum + p) * units, seq_len, units);
      }
    }
    return Status::OK();
  }
};

template <typename Scalar, typename TIndex>
struct LaunchBlazeAttentionIndicator<CPUDevice, Scalar, TIndex> {
  Status operator()(OpKernelContext* context, const Tensor& in_fact,
                    const Tensor& in_query, const Tensor& indicators,
                    Tensor* out, int pnum, int batch_fact, int batch_query,
                    int seq_len, int units) {
    auto fact_ptr = in_fact.template flat<Scalar>().data();
    auto query_ptr = in_query.template flat<Scalar>().data();
    auto ind_ptr = indicators.template flat<TIndex>().data();
    auto out_ptr = out->template flat<Scalar>().data();
    for (int p = 0; p < pnum; p++) {
      for (int b = 0; b < batch_query; b++) {
        int ind = (int)ind_ptr[b];
        BlazeAttentionFunctor(
            context->eigen_device<CPUDevice>(),
            fact_ptr + (p * batch_fact + ind) * seq_len * units,
            query_ptr + (p * batch_query + b) * units,
            out_ptr + (b * pnum + p) * units, seq_len, units);
      }
    }
    return Status::OK();
  }
};

template <typename Device, typename Scalar>
class BlazeAttentionOp : public OpKernel {
 public:
  explicit BlazeAttentionOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  ~BlazeAttentionOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& fact = ctx->input(0);
    auto& query = ctx->input(1);

    OP_REQUIRES(
        ctx, fact.dims() == 4,
        errors::InvalidArgument("In[0] ndims must be 4: ", fact.dims()));
    OP_REQUIRES(
        ctx, query.dims() == 3,
        errors::InvalidArgument("In[1] ndims must be 3: ", query.dims()));

    OP_REQUIRES(ctx, fact.dim_size(2) <= 256,
                errors::InvalidArgument("seq length of fact must <= 256: got ",
                                        fact.dim_size(2)));
    OP_REQUIRES(
        ctx, fact.dim_size(3) == 32,
        errors::InvalidArgument("units must equal to 32: ", fact.dim_size(3)));
    int fact_units = fact.dim_size(3);
    int query_units = query.dim_size(2);
    OP_REQUIRES(ctx, fact_units == query_units,
                errors::InvalidArgument("fact units mismatch query units: ",
                                        fact.shape().DebugString(), " vs. ",
                                        query.shape().DebugString()));
    int fact_pnum = fact.dim_size(0);
    int query_pnum = query.dim_size(0);
    OP_REQUIRES(ctx, fact_pnum == query_pnum,
                errors::InvalidArgument("fact pnum mismatch query pnum: ",
                                        fact.shape().DebugString(), " vs. ",
                                        query.shape().DebugString()));
    int batch_fact = fact.dim_size(1);
    OP_REQUIRES(
        ctx, batch_fact == 1,
        errors::InvalidArgument("batch_fact must be 1: got ", batch_fact));
    int batch_query = query.dim_size(1);
    TensorShape out_shape({batch_query, query_pnum, query_units});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (fact.NumElements() == 0 || query.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }
    int seq_len = fact.dim_size(2);

    //[PROF-STATS]
    int64 delta = 2 * seq_len * out_shape.num_elements();
    if (ctx->traced_infos()) {
      ctx->traced_infos()->RecordFlops(delta, requested_device());
    }
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "FLOPs = " << delta
                << ", " << type_string()
                << ", " << name()
                << ", " << fact.shape().DebugString()
                << ", " << query.shape().DebugString();
    }

    OP_REQUIRES_OK(ctx, LaunchBlazeAttention<Device, Scalar>()(
                            ctx, fact, query, out, query_pnum, batch_fact,
                            batch_query, seq_len, query_units));
  }
};

template <typename Device, typename Scalar, typename TIndex>
class BlazeAttentionIndicatorOp : public OpKernel {
 public:
  explicit BlazeAttentionIndicatorOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  ~BlazeAttentionIndicatorOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& fact = ctx->input(0);
    auto& query = ctx->input(1);
    auto& ind = ctx->input(2);

    OP_REQUIRES(
        ctx, fact.dims() == 4,
        errors::InvalidArgument("In[0] ndims must be 4: ", fact.dims()));
    OP_REQUIRES(
        ctx, query.dims() == 3,
        errors::InvalidArgument("In[1] ndims must be 3: ", query.dims()));
    OP_REQUIRES(ctx, ind.dims() == 1,
                errors::InvalidArgument("In[2] ndims must be 1: ", ind.dims()));
    OP_REQUIRES(
        ctx, fact.dim_size(2) <= 256,
        errors::InvalidArgument("seq length of fact must be <= 256: got ",
                                fact.dim_size(2)));
    OP_REQUIRES(
        ctx, fact.dim_size(3) == 32,
        errors::InvalidArgument("units must equal to 32: ", fact.dim_size(3)));
    int fact_units = fact.dim_size(3);
    int query_units = query.dim_size(2);
    OP_REQUIRES(ctx, fact_units == query_units,
                errors::InvalidArgument("fact units mismatch query units: ",
                                        fact.shape().DebugString(), " vs. ",
                                        query.shape().DebugString()));
    int fact_pnum = fact.dim_size(0);
    int query_pnum = query.dim_size(0);
    OP_REQUIRES(ctx, fact_pnum == query_pnum,
                errors::InvalidArgument("fact pnum mismatch query pnum: ",
                                        fact.shape().DebugString(), " vs. ",
                                        query.shape().DebugString()));
    int batch_fact = fact.dim_size(1);
    int batch_query = query.dim_size(1);
    int ind_length = ind.dim_size(0);
    OP_REQUIRES(ctx, ind_length == batch_query,
                errors::InvalidArgument("ind_length mismatch batch_query: ",
                                        ind_length, " vs. ", batch_query));
    TensorShape out_shape({batch_query, query_pnum, query_units});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (fact.NumElements() == 0 || query.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }
    int seq_len = fact.dim_size(2);

    //[PROF-STATS]
    int64 delta = 2 * seq_len * out_shape.num_elements();
    if (ctx->traced_infos()) {
      ctx->traced_infos()->RecordFlops(delta, requested_device());
    }
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "FLOPs = " << delta
                << ", " << type_string()
                << ", " << name()
                << ", " << fact.shape().DebugString()
                << ", " << query.shape().DebugString();
    }

    OP_REQUIRES_OK(ctx, LaunchBlazeAttentionIndicator<Device, Scalar, TIndex>()(
                            ctx, fact, query, ind, out, query_pnum, batch_fact,
                            batch_query, seq_len, query_units));
  }
};

#define REGISTER_BLAZE_ATTENTION_CPU(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("BlazeAttention").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),    \
      BlazeAttentionOp<CPUDevice, TYPE>);                                     \
  REGISTER_KERNEL_BUILDER(Name("BlazeAttentionIndicator")                     \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<int32>("Tindices"),             \
                          BlazeAttentionIndicatorOp<CPUDevice, TYPE, int32>); \
  REGISTER_KERNEL_BUILDER(Name("BlazeAttentionIndicator")                     \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<int64>("Tindices"),             \
                          BlazeAttentionIndicatorOp<CPUDevice, TYPE, int64>);

REGISTER_BLAZE_ATTENTION_CPU(float);
REGISTER_BLAZE_ATTENTION_CPU(Eigen::half);

#undef REGISTER_BLAZE_ATTENTION_CPU

#if GOOGLE_CUDA
#define REGISTER_BLAZE_ATTENTION_GPU(TYPE)                                    \
  extern template struct LaunchBlazeAttention<GPUDevice, TYPE>;               \
  extern template struct LaunchBlazeAttentionIndicator<GPUDevice, TYPE,       \
                                                       int32>;                \
  extern template struct LaunchBlazeAttentionIndicator<GPUDevice, TYPE,       \
                                                       int64>;                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("BlazeAttention").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),    \
      BlazeAttentionOp<GPUDevice, TYPE>);                                     \
  REGISTER_KERNEL_BUILDER(Name("BlazeAttentionIndicator")                     \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<int32>("Tindices"),             \
                          BlazeAttentionIndicatorOp<GPUDevice, TYPE, int32>); \
  REGISTER_KERNEL_BUILDER(Name("BlazeAttentionIndicator")                     \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .TypeConstraint<int64>("Tindices"),             \
                          BlazeAttentionIndicatorOp<GPUDevice, TYPE, int64>);

REGISTER_BLAZE_ATTENTION_GPU(float);
REGISTER_BLAZE_ATTENTION_GPU(Eigen::half);
#undef REGISTER_BLAZE_ATTENTION_GPU
#endif
}  // namespace tensorflow
