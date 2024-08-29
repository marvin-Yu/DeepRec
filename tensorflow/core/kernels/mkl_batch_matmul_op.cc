/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc.

// This file uses oneDNN library for acceleration of Batch Matrix-Matrix
// Multiplication (MatMul) operations. We currently register this kernel only
// for oneDNN supported data types (float, bfloat16). The maximum number of
// dimensions (rank) for output tensor is 12 in oneDNN. If output tensor rank
// exceeds 12, we exit with reporting an error message.

#define EIGEN_USE_THREADS

#ifdef INTEL_MKL

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/batch_matmul_op_impl.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl_batch_matmul_helper.h"
#include "tensorflow/core/kernels/mkl_matmul_ops_common.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/mkl_util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

//  The third parameter v2_bcast is set to true if we are using V2 otherwise
//  we set it to false.
template <typename Device, typename Tlhs, typename Trhs, typename Toutput,
          bool v2_bcast>
class BatchMatMulMkl : public OpKernel {
 public:
  explicit BatchMatMulMkl(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
  }

  virtual ~BatchMatMulMkl() {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& lhs = ctx->input(0);
    const Tensor& rhs = ctx->input(1);

    if (std::is_same<Tlhs, float>::value) {
      (void)SetFPMathMode();
    }

    if (!v2_bcast) {
      // Using V1, so check to make sure lhs and rhs dimensions are correct and
      // no broadcasting is needed.
      OP_REQUIRES(ctx, lhs.dims() == rhs.dims(),
                  errors::InvalidArgument("lhs and rhs has different ndims: ",
                                          lhs.shape().DebugString(), " vs. ",
                                          rhs.shape().DebugString()));
      const int ndims = lhs.dims();
      OP_REQUIRES(
          ctx, ndims >= 2,
          errors::InvalidArgument("lhs and rhs ndims must be >= 2: ", ndims));
      for (int i = 0; i < ndims - 2; ++i) {
        OP_REQUIRES(ctx, lhs.dim_size(i) == rhs.dim_size(i),
                    errors::InvalidArgument(
                        "lhs.dim(", i, ") and rhs.dim(", i,
                        ") must be the same: ", lhs.shape().DebugString(),
                        " vs ", rhs.shape().DebugString()));
      }
    } else {
      OP_REQUIRES(
          ctx, lhs.dims() >= 2,
          errors::InvalidArgument("In[0] ndims must be >= 2: ", lhs.dims()));
      OP_REQUIRES(
          ctx, rhs.dims() >= 2,
          errors::InvalidArgument("In[1] ndims must be >= 2: ", rhs.dims()));
    }

    // lhs and rhs can have different dimensions
    const auto ndims_lhs = lhs.dims();
    const auto ndims_rhs = rhs.dims();

    // Get broadcast info
    MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();

    auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
    auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
    auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
    auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

    if (adj_x_) std::swap(lhs_rows, lhs_cols);
    if (adj_y_) std::swap(rhs_rows, rhs_cols);
    OP_REQUIRES(ctx, lhs_cols == rhs_rows,
                errors::InvalidArgument(
                    "lhs mismatch rhs shape: ", lhs_cols, " vs. ", rhs_rows,
                    ": ", lhs.shape().DebugString(), " ",
                    rhs.shape().DebugString(), " ", adj_x_, " ", adj_y_));

    out_shape.AddDim(lhs_rows);
    out_shape.AddDim(rhs_cols);
    // The maximum number of dimensions for a tensor in DNNL is 12.
    OP_REQUIRES(
        ctx, out_shape.dims() <= 12,
        errors::InvalidArgument(
            "Rank of output tensor must be <= 12, but is ", out_shape.dims(),
            ". Current implementation supports upto rank 12 tensors."));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Toutput> f;
      f(ctx->eigen_device<Device>(), out->flat<Toutput>());
      return;
    }

    // Compute parameters for DNNL matmul primitive.
    MklBatchMatMulHelper bmm;
    string prefix = "batchmatmul";
    memory::dims bias_dims = NONE_DIMS;
    if (fuse_bias_) {
      const Tensor& bias_tensor = ctx->input(2);
      if (out_shape.dims() == 4) {
        bias_dims = {1, 1, 1, bias_tensor.dim_size(0)};
      } else if (out_shape.dims() == 3) {
        bias_dims = {1, 1, bias_tensor.dim_size(0)};
      }
    }
    auto params = bmm.CreateMatMulParams(prefix, lhs.shape(), rhs.shape(),
                                         out_shape, adj_x_, adj_y_, bias_dims);
    this->ExtendMklMatMulParams(ctx, *params);

    // Create the oneDNN wrapper over Eigen threadpool and set max threads
    // in oneDNN.
    Eigen::ThreadPoolInterface* eigen_interface =
        EigenThreadPoolFromTfContext(ctx);
    OneDnnThreadPool eigen_tp(eigen_interface, ThreadPoolUseCallerThread());

    // Create or retrieve matmul primitive from cache.
    MklMatMulPrimitive<Tlhs, Trhs, Tlhs, Toutput>* matmul_prim =
        MklMatMulPrimitiveFactory<float, Tlhs, Trhs, Tlhs, Toutput>::Get(
            *params, false /* value for do_not_cache */);

    Trhs* weight_data = const_cast<Trhs*>(rhs.flat<Trhs>().data());

    UserScratchPad<unsigned char> scratch_pad;
    scratch_pad.AllocateSPTensor(matmul_prim, ctx);

    // Execute matmul primitive.
    std::shared_ptr<stream> cpu_stream;
    cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));
    if (this->fusion_data_.size() > 0) {
      if (fuse_bias_) {
        const Tensor& bias = ctx->input(2);
        matmul_prim->Execute(cpu_stream, lhs.flat<Tlhs>().data(), weight_data,
                             bias.flat<Tlhs>().data(),
                             out->flat<Toutput>().data(), *params,
                             scratch_pad.Get(), this->fusion_data_);
      } else {
        matmul_prim->Execute(cpu_stream, lhs.flat<Tlhs>().data(), weight_data,
                             nullptr, out->flat<Toutput>().data(), *params,
                             scratch_pad.Get(), this->fusion_data_);
      }
    } else {
      matmul_prim->Execute(cpu_stream, lhs.flat<Tlhs>().data(), weight_data,
                           nullptr, out->flat<Toutput>().data(), *params,
                           scratch_pad.Get());
    }
  }

 protected:
  virtual void ExtendMklMatMulParams(OpKernelContext* ctx,
                                     MklMatMulParams& params) {}
  std::vector<void*> fusion_data_;
  bool fuse_bias_ = false;

 private:
  bool adj_x_;
  bool adj_y_;
};

// OneDNN uses post-ops to implement different kind of fusions. The category of
// each individual post-op can be inferred from the fused_ops attribute. The
// following enum is used to identify list of required post-ops.
namespace {

enum class FusedComputationType {
  kUndefined,
  kMul,
  kAdd,
  kMulAdd,
  kDequantize,
  kMul_Dequantize,
  kAdd_Dequantize,
  kMulAdd_Dequantize,
  kRequantize,
  kMul_Requantize,
  kAdd_Requantize,
  kMulAdd_Requantize,
  kBiasGelu,
  kBias,
  kBiasAdd,
};

struct FusedComputationPattern {
  FusedComputationType fused_computation;
  std::vector<string> fused_ops;
};

}  // namespace
enum class PostOpKind {
  kNone,
  kOutputScale,
  kMul,
  kAdd,
  kLinear,
  kBias,
  kGelu
};

// FusedBatchMatMul has additional inputs, currently forcing all the operands
// of fusion to have same type `U`.
template <typename Device, typename Tlhs, typename Trhs, typename Toutput,
          /*type of additional tensors*/ typename U, bool v2_bcast>
class FusedBatchMatMulMkl
    : public BatchMatMulMkl<Device, Tlhs, Trhs, Toutput, v2_bcast> {
 public:
  explicit FusedBatchMatMulMkl(OpKernelConstruction* context)
      : BatchMatMulMkl<Device, Tlhs, Trhs, Toutput, v2_bcast>(context) {
    InitializeFusion(context);
  }

  virtual ~FusedBatchMatMulMkl() {}

 protected:
  struct PostOpInfo {
    PostOpKind post_op_kind;
    int input_idx = -1;  // Operand tensor index if needed by a post-op.
  };

  std::vector<PostOpInfo> post_op_info_list_;

  // This function is called from constructor.
  void InitializeFusion(OpKernelConstruction* context) {
    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
    OP_REQUIRES(context, !fused_ops.empty(),
                errors::InvalidArgument(
                    "Fused BatchMatMul must have at least one fused op."));

    if (fused_ops.size() >= 1 && fused_ops[0] == "BiasAdd")
      this->fuse_bias_ = true;

    using FCT = FusedComputationType;
    // TODO(intel-tf): Add more patterns when implemented. Refactor for
    // arbitrary fusion sequence when oneDNN is performant.
    std::vector<FusedComputationPattern> patterns{
        {FCT::kMul, {"Mul"}},
        {FCT::kAdd, {"Add"}},
        {FCT::kBias, {"BiasAdd"}},
        {FCT::kMulAdd, {"Mul", "Add"}},
        {FCT::kBiasGelu, {"BiasAdd", "GeluExact"}},
        {FCT::kBiasAdd, {"BiasAdd", "Add"}},
    };
    FusedComputationType fused_computation = FusedComputationType::kUndefined;
    for (const auto& pattern : patterns) {
      if (fused_ops == pattern.fused_ops) {
        fused_computation = pattern.fused_computation;
        break;
      }
    }

    // Configure oneDNN post-ops. Refactor for arbitrary fusion sequence when
    // oneDNN is performant.
    switch (fused_computation) {
      case FCT::kMul:
        post_op_info_list_ = {{PostOpKind::kMul, 2}};
        break;
      case FCT::kAdd:
        post_op_info_list_ = {{PostOpKind::kAdd, 2}};
        break;
      case FCT::kBias:
        post_op_info_list_ = {{PostOpKind::kBias, 2}};
        break;
      case FCT::kMulAdd:
        post_op_info_list_ = {{PostOpKind::kMul, 2}, {PostOpKind::kAdd, 3}};
        break;
      case FCT::kBiasGelu:
        post_op_info_list_ = {{PostOpKind::kBias, 2}, {PostOpKind::kGelu, 3}};
        break;
      case FCT::kBiasAdd:
        post_op_info_list_ = {{PostOpKind::kBias, 2}, {PostOpKind::kAdd, 3}};
        break;
      default:
        OP_REQUIRES_OK(
            context, errors::Unimplemented("Fusion is not implemented: [",
                                           absl::StrJoin(fused_ops, ","), "]"));
    }

    int num_args = 0;
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));
    this->fusion_data_.resize(num_args);
  }

  virtual void ExtendMklMatMulParams(OpKernelContext* ctx,
                                     MklMatMulParams& params) {
    int idx = 0;
    memory::format_tag format_tag;
    switch (params.c_dims.size()) {
      case 3:
        format_tag = memory::format_tag::abc;
        break;
      case 4:
        format_tag = memory::format_tag::abcd;
        break;
      default:
        OP_REQUIRES(ctx, false, errors::Unimplemented("Unimplemented"));
    }
    for (const auto& post_op_info : this->post_op_info_list_) {
      switch (post_op_info.post_op_kind) {
        case PostOpKind::kMul: {
          const Tensor& multiplicand_tensor =
              ctx->input(post_op_info.input_idx);
          // TODO(intel-tf): Relax restriction when oneDNN is performant for
          // arbitrary shapes.
          bool is_supported = multiplicand_tensor.NumElements() == 1 &&
                              params.c_dims.size() == 4;
          OP_REQUIRES(ctx, is_supported,
                      errors::Unimplemented(absl::StrCat(
                          "Unimplemented multiplicand shape for Mul fusion: ",
                          multiplicand_tensor.shape().DebugString())));
          memory::data_type data_type = MklDnnType<U>();
          memory::dims mul_dims(params.c_dims.size(), 1);
          params.post_op_params.push_back(
              {"mul", {}, mul_dims, data_type, format_tag});
          void* multiplicand_data = static_cast<void*>(
              const_cast<U*>(multiplicand_tensor.flat<U>().data()));
          this->fusion_data_[idx++] = multiplicand_data;
        } break;
        case PostOpKind::kAdd: {
          const Tensor& addend_tensor = ctx->input(post_op_info.input_idx);
          // TODO(intel-tf): Relax restriction when oneDNN is performant for
          // arbitrary shapes.
          bool is_supported =
              ((params.c_dims.size() == 4 &&
                addend_tensor.dims() == params.c_dims.size()) ||
               (params.c_dims.size() == 4 && addend_tensor.NumElements() == 1));
          memory::format_tag format_tag;
          switch (params.c_dims.size()) {
            case 3:
              format_tag = memory::format_tag::abc;
              break;
            case 4:
              format_tag = memory::format_tag::abcd;
              break;
            default:
              OP_REQUIRES(ctx, false, errors::Unimplemented("Unimplemented"));
          }
          memory::data_type data_type = MklDnnType<U>();
          int rank_addend_tensor = addend_tensor.shape().dims();

          memory::dims addend_dims(params.c_dims.size(), 1);
          if (addend_tensor.dims() == params.c_dims.size()) {
            addend_dims = TFShapeToMklDnnDims(addend_tensor.shape());
          }
          params.post_op_params.push_back(
              {"add", {}, addend_dims, data_type, format_tag});
          void* addend_data = static_cast<void*>(
              const_cast<U*>(addend_tensor.flat<U>().data()));
          this->fusion_data_[idx++] = addend_data;
        } break;
        case PostOpKind::kBias: {
          // Do nothing
        } break;
        case PostOpKind::kGelu: {
          float scale = 1.0f;
          float alpha = 0.0f;
          float beta = 0.0f;
          params.post_op_params.push_back({"GeluExact", {scale, alpha, beta}});
        } break;
        default:
          OP_REQUIRES(ctx, false, errors::Unimplemented("Unimplemented"));
      }
    }
  }
};
#define REGISTER_BATCH_MATMUL_MKL(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(Name("_MklBatchMatMul")                             \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          BatchMatMulMkl<CPUDevice, TYPE, TYPE, TYPE, false>)

#define REGISTER_BATCH_MATMUL_MKL_V2(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(Name("_MklBatchMatMulV2")                           \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          BatchMatMulMkl<CPUDevice, TYPE, TYPE, TYPE, true>)

#define REGISTER_FUSED_BATCH_MATMUL_MKL(TYPE) \
  REGISTER_KERNEL_BUILDER(                    \
      Name("_MklFusedBatchMatMulV2")          \
          .Device(DEVICE_CPU)                 \
          .TypeConstraint<TYPE>("T"),         \
      FusedBatchMatMulMkl<CPUDevice, TYPE, TYPE, TYPE, TYPE, true>)

TF_CALL_float(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_float(REGISTER_BATCH_MATMUL_MKL_V2);
TF_CALL_float(REGISTER_FUSED_BATCH_MATMUL_MKL);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_MKL_V2);
TF_CALL_bfloat16(REGISTER_FUSED_BATCH_MATMUL_MKL);
TF_CALL_half(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_half(REGISTER_BATCH_MATMUL_MKL_V2);
TF_CALL_half(REGISTER_FUSED_BATCH_MATMUL_MKL);

}  // end namespace tensorflow
#endif
