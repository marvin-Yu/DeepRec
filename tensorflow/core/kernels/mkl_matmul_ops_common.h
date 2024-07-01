/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_MATMUL_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_MATMUL_OPS_COMMON_H_

#ifdef INTEL_MKL
#include <memory>
#include <string>
#include <vector>

#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/mkl_kernel_util.h"
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/onednn_env_vars.h"
#ifdef DNNL_AARCH64_USE_ACL
#include "tensorflow/core/platform/mutex.h"
#endif

using dnnl::inner_product_forward;
using dnnl::primitive_attr;
using dnnl::prop_kind;
using dnnl::stream;

namespace tensorflow {
#ifndef ENABLE_ONEDNN_V3
#define APPEND_ELTWISE(scale, alg, alpha, beta) \
  append_eltwise(scale, alg, alpha, beta)
#define APPEND_ELTWISE_RELU6(scale, alpha, beta) \
  append_eltwise(scale, dnnl::algorithm::eltwise_bounded_relu, alpha, beta)
#define SET_MKL_LAYOUT(md) SetMklLayout(&md)
#else
#define APPEND_ELTWISE(scale, alg, alpha, beta) append_eltwise(alg, alpha, beta)
#define APPEND_ELTWISE_RELU6(scale, alpha, beta) \
  append_eltwise(dnnl::algorithm::eltwise_clip, 0.0, alpha)
#define SET_MKL_LAYOUT(md) SetMklLayout(md)
#endif  // !ENABLE_ONEDNN_V3

#if defined(ENABLE_DNNL_THREADPOOL) && !defined(ENABLE_ONEDNN_V3)
#define FWD_STREAM , *fwd_stream
#else
#define FWD_STREAM
#endif  // ENABLE_DNNL_THREADPOOL && !ENABLE_ONEDNN_V3

#define L1_SIZE 32 * 1024
typedef Eigen::ThreadPoolDevice CPUDevice;

inline bool ExecuteSingleThreadedGemm(int m, int n, int k) {
  // Ideally we would like to determine blocking and then come up with
  // a heuristic but what we are targeting are very small models whose
  // total size is < few L1's. So we will do this simple calculation
  // to determine if the matrix multiplication should be run on a single thread.
  constexpr int kHeuristicMultiplier = 8;
  return ((sizeof(float) * (m * n + k * (m + n))) <
          L1_SIZE * kHeuristicMultiplier);
}
// This structure aggregates multiple inputs to MklDnnMatMul* methods.
struct MklDnnMatMulFwdParams {
  memory::dims src_dims;
  memory::dims weight_dims;
  memory::dims bias_dims;
  memory::dims dst_dims;
  MEMORY_FORMAT src_format;
  MEMORY_FORMAT weight_format;
  MEMORY_FORMAT dst_format;
  string dtypes = string("");
  bool const_weight;
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  std::vector<PostOpParam> post_op_params;

  MklDnnMatMulFwdParams(memory::dims src_dims, memory::dims weight_dims,
                        memory::dims bias_dims, memory::dims dst_dims,
                        MEMORY_FORMAT src_format = MEMORY_FORMAT::any,
                        MEMORY_FORMAT weight_format = MEMORY_FORMAT::any,
                        MEMORY_FORMAT dst_format = MEMORY_FORMAT::any,
                        bool const_weight = false)
      : src_dims(src_dims),
        weight_dims(weight_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        src_format(src_format),
        weight_format(weight_format),
        dst_format(dst_format),
        const_weight(const_weight) {}
};

// With quantization, input, weight, bias, and output can have different types.
// So we use different template parameters for each type.
// TODO(intel-tf): The template type "T" is currently used to match the
// templatized class MklPrimitiveFactory (tensorflow/core/util/mkl_util.h).
// In the future, with the removal of "T" from MklPrimitiveFactory, this class
// needs to drop "T".
template <typename T, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class MklDnnMatMulFwdPrimitive : public MklPrimitive {
 public:
  explicit MklDnnMatMulFwdPrimitive(
      const MklDnnMatMulFwdParams& matmulFwdParams)
      : MklPrimitive(engine(ENGINE_CPU, 0)) {
    // Create matmul primitive
    if (context_.matmul_fwd == nullptr) {
      Setup(matmulFwdParams);
    }
  }

  ~MklDnnMatMulFwdPrimitive() {}

  // Inner-product forward execute with bias:
  //  - src_data: input data buffer of src
  //  - weight_data: input data buffer of weight
  //  - bias_data: input data buffer of bias
  //  - dst_data: output data buffer of dst
  void Execute(const Tinput* src_data, const Tweight* weight_data,
               const void* bias_data, Toutput* dst_data,
               const MklDnnMatMulFwdParams& matmul_fwd_params,
               std::shared_ptr<stream> fwd_stream) {
#ifdef DNNL_AARCH64_USE_ACL
    mutex_lock lock(primitive_execution_mu_);
#endif
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)) FWD_STREAM);
    context_.weight_mem->set_data_handle(
        static_cast<void*>(const_cast<Tweight*>(weight_data)) FWD_STREAM);
    context_.bias_mem->set_data_handle(const_cast<void*>(bias_data) FWD_STREAM);
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data) FWD_STREAM);

    execute_primitives(context_.fwd_primitives, fwd_stream, context_.net_args);

    // After execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.weight_mem->set_data_handle(DummyData);
    context_.bias_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<dnnl::inner_product_forward::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for inner-product Fwd op
  struct MklDnnMatMulFwdContext {
    // OneDNN memory.
    std::shared_ptr<dnnl::memory> src_mem;
    std::shared_ptr<dnnl::memory> weight_mem;
    std::shared_ptr<dnnl::memory> bias_mem;
    std::shared_ptr<dnnl::memory> dst_mem;

    // Descriptor and primitive-descriptor for forward inner-product.
#ifndef ENABLE_ONEDNN_V3
    std::shared_ptr<dnnl::inner_product_forward::desc> fwd_desc;
#endif  // !ENABLE_ONEDNN_V3
    std::shared_ptr<dnnl::inner_product_forward::primitive_desc> fwd_pd;

    // Memory descriptors.
    std::shared_ptr<dnnl::memory::desc> src_md;
    std::shared_ptr<dnnl::memory::desc> weight_md;
    std::shared_ptr<dnnl::memory::desc> bias_md;
    std::shared_ptr<dnnl::memory::desc> dst_md;

    // Inner-product primitive.
    std::shared_ptr<dnnl::primitive> matmul_fwd;
    std::vector<dnnl::primitive> fwd_primitives;
    std::vector<std::unordered_map<int, memory>> net_args;

    MklDnnMatMulFwdContext()
        : src_mem(nullptr),
          weight_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
#ifndef ENABLE_ONEDNN_V3
          fwd_desc(nullptr),
#endif  // !ENABLE_ONEDNN_V3
          fwd_pd(nullptr),
          src_md(nullptr),
          weight_md(nullptr),
          bias_md(nullptr),
          dst_md(nullptr),
          matmul_fwd(nullptr) {
    }
  };

  void Setup(const MklDnnMatMulFwdParams& matmul_fwd_params) {
    // Create memory descriptors for inner-product data without specified
    // format.
    context_.src_md.reset(new memory::desc({matmul_fwd_params.src_dims},
                                           MklDnnType<Tinput>(),
                                           matmul_fwd_params.src_format));

    context_.weight_md.reset(new memory::desc({matmul_fwd_params.weight_dims},
                                              MklDnnType<Tweight>(),
                                              matmul_fwd_params.weight_format));

    context_.dst_md.reset(new memory::desc({matmul_fwd_params.dst_dims},
                                           MklDnnType<Toutput>(),
                                           matmul_fwd_params.dst_format));

    context_.bias_md.reset(new memory::desc({matmul_fwd_params.bias_dims},
                                            MklDnnType<Tbias>(),
                                            MEMORY_FORMAT::any));
    // Create an inner-product.
#ifndef ENABLE_ONEDNN_V3
    context_.fwd_desc.reset(new inner_product_forward::desc(
        matmul_fwd_params.const_weight ? prop_kind::forward_inference : prop_kind::forward_training,
        *context_.src_md, *context_.weight_md,
        *context_.bias_md, *context_.dst_md));
    context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));
#endif  // !ENABLE_ONEDNN_V3

    // Check if there is any fusion as post-ops
    auto const& post_op_params = matmul_fwd_params.post_op_params;
    dnnl::primitive_attr post_ops_attr;
    dnnl::post_ops post_ops;
    if (!post_op_params.empty()) {
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "relu" || post_op_param.name == "leakyrelu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, ALGORITHM::eltwise_relu, op_alpha,
                                  op_beta);
        } else if (post_op_param.name == "relu6") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE_RELU6(op_scale, op_alpha, op_beta);
        } else if (post_op_param.name == "elu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, ALGORITHM::eltwise_elu, op_alpha,
                                  op_beta);
        } else if (post_op_param.name == "gelu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, ALGORITHM::eltwise_gelu_tanh,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "gelu_erf") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, ALGORITHM::eltwise_gelu_erf,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "tanh") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, ALGORITHM::eltwise_tanh, op_alpha,
                                  op_beta);
        } else if (post_op_param.name == "logistic") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, ALGORITHM::eltwise_logistic,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "output_scale") {
#ifndef ENABLE_ONEDNN_V3
          DCHECK_EQ(post_op_param.param.size(), 1);
          std::vector<float> scales;
          scales.push_back(post_op_param.param[0]);
          post_ops_attr.set_output_scales(0, scales);
#endif  // !ENABLE_ONEDNN_V3
        } else if (post_op_param.name == "sum") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          float op_scale = post_op_param.param[0];
          post_ops.append_sum(op_scale);
        } else {
          DCHECK(
              (post_op_param.name == "relu") ||
              (post_op_param.name == "relu6") ||
              (post_op_param.name == "elu") || (post_op_param.name == "gelu") ||
              (post_op_param.name == "gelu_erf") ||
              (post_op_param.name == "sum") || (post_op_param.name == "tanh") ||
              (post_op_param.name == "logistic") ||
              (post_op_param.name == "output_scale"));
        }
      }
      post_ops_attr.set_post_ops(post_ops);
#ifndef ENABLE_ONEDNN_V3
      context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
          *context_.fwd_desc, post_ops_attr, cpu_engine_));
#else
      context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
          cpu_engine_, prop_kind::forward_inference, *context_.src_md,
          *context_.weight_md, *context_.bias_md, *context_.dst_md,
          post_ops_attr));
#endif  // !ENABLE_ONEDNN_V3
    } else {
#ifndef ENABLE_ONEDNN_V3
      context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
          *context_.fwd_desc, cpu_engine_));
#else
      context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
          cpu_engine_, prop_kind::forward_inference, *context_.src_md,
          *context_.weight_md, *context_.bias_md, *context_.dst_md));
#endif  // !ENABLE_ONEDNN_V3
    }

    // Create memory primitive based on dummy data
    context_.src_mem.reset(new MEMORY_CONSTRUCTOR(
        context_.fwd_pd.get()->PRIMITIVE_DESC_SRC, cpu_engine_, DummyData));
    context_.weight_mem.reset(new MEMORY_CONSTRUCTOR(
        context_.fwd_pd.get()->PRIMITIVE_DESC_WEIGHTS, cpu_engine_, DummyData));
    context_.dst_mem.reset(new MEMORY_CONSTRUCTOR(
        context_.fwd_pd.get()->PRIMITIVE_DESC_DST, cpu_engine_, DummyData));
    context_.bias_mem.reset(
        new memory(context_.fwd_pd.get()->bias_desc(), cpu_engine_, DummyData));

    // Create inner-product primitive.
    context_.matmul_fwd.reset(new inner_product_forward(*context_.fwd_pd));
    std::unordered_map<int, memory> net_args = {
        {DNNL_ARG_SRC, *context_.src_mem},
        {DNNL_ARG_WEIGHTS, *context_.weight_mem},
        {DNNL_ARG_BIAS, *context_.bias_mem},
        {DNNL_ARG_DST, *context_.dst_mem}};

    context_.net_args.push_back(net_args);
    context_.fwd_primitives.push_back(*context_.matmul_fwd);
    return;
  }

  struct MklDnnMatMulFwdContext context_;

#ifdef DNNL_AARCH64_USE_ACL
  // Guards Execution()
  mutex primitive_execution_mu_;
#endif
};

template <typename T, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class MklDnnMatMulFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>* Get(
      const MklDnnMatMulFwdParams& dnnl_matmul_fwd_dims, bool do_not_cache) {
    MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>* matmul_fwd =
        nullptr;

    if (do_not_cache) {
      // Always create new primitive
      matmul_fwd =
          new MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>(
              dnnl_matmul_fwd_dims);
    } else {
      // Try to find a suitable one in pool
      matmul_fwd = dynamic_cast<
          MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>*>(
          MklDnnMatMulFwdPrimitiveFactory<T, Tinput, Tweight, Tbias,
                                          Toutput>::GetInstance()
              .GetMklDnnMatMulFwd(dnnl_matmul_fwd_dims));
      if (matmul_fwd == nullptr) {
        matmul_fwd =
            new MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>(
                dnnl_matmul_fwd_dims);
        MklDnnMatMulFwdPrimitiveFactory<T, Tinput, Tweight, Tbias,
                                        Toutput>::GetInstance()
            .SetMklDnnMatMulFwd(dnnl_matmul_fwd_dims, matmul_fwd);
      }
    }
    return matmul_fwd;
  }

 private:
  MklDnnMatMulFwdPrimitiveFactory() {}
  ~MklDnnMatMulFwdPrimitiveFactory() {}

  static MklDnnMatMulFwdPrimitiveFactory& GetInstance() {
    static MklDnnMatMulFwdPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklDnnMatMulFwdParams& dnnl_matmul_fwd_dims) {
    string prefix = "matmul_fwd_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(dnnl_matmul_fwd_dims.src_dims);
    key_creator.AddAsKey(dnnl_matmul_fwd_dims.weight_dims);
    key_creator.AddAsKey(dnnl_matmul_fwd_dims.bias_dims);
    key_creator.AddAsKey(dnnl_matmul_fwd_dims.dst_dims);
    key_creator.AddAsKey(dnnl_matmul_fwd_dims.dtypes);
    key_creator.AddAsKey(dnnl_matmul_fwd_dims.weight_format);

    // Generate keys for post-ops
    for (auto const& post_op_param : dnnl_matmul_fwd_dims.post_op_params) {
      if (post_op_param.name == "relu" || post_op_param.name == "relu6" ||
          post_op_param.name == "elu" || post_op_param.name == "gelu" ||
          post_op_param.name == "gelu_erf" || post_op_param.name == "tanh" ||
          post_op_param.name == "logistic") {
        DCHECK_EQ(post_op_param.param.size(), 3);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
        key_creator.AddAsKey(post_op_param.param[1]);
        key_creator.AddAsKey(post_op_param.param[2]);
      } else if (post_op_param.name == "output_scale") {
#ifndef ENABLE_ONEDNN_V3
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
#endif  // !ENABLE_ONEDNN_V3
      } else if (post_op_param.name == "sum") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
      } else {
        return string("not_a_key");
      }
    }
    return key_creator.GetKey();
  }

  MklPrimitive* GetMklDnnMatMulFwd(
      const MklDnnMatMulFwdParams& dnnl_matmul_fwd_dims) {
    string key = CreateKey(dnnl_matmul_fwd_dims);
    return this->GetOp(key);
  }

  void SetMklDnnMatMulFwd(const MklDnnMatMulFwdParams& dnnl_matmul_fwd_dims,
                          MklPrimitive* op) {
    string key = CreateKey(dnnl_matmul_fwd_dims);
    this->SetOp(key, op);
  }
};

template <class Tweight, class Toutput>
class MklDnnMatMulOpBase : public OpKernel {
 public:
  explicit MklDnnMatMulOpBase(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override = 0;

  // Allocate output tensor.
  virtual void AllocateOutputTensor(
      OpKernelContext* context,
      const inner_product_forward::primitive_desc& dnnl_matmul_prim_desc,
      const memory::dims& output_dims_mkl_order,
      MKL_TENSOR_FORMAT output_tf_format, Tensor** output_tensor) {
    DCHECK(output_tensor);
    auto dst_pd = dnnl_matmul_prim_desc.PRIMITIVE_DESC_DST;

    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SET_MKL_LAYOUT(dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<Toutput>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    TensorShape output_tf_shape;
    output_tf_shape.AddDim((dst_pd.get_size() / sizeof(Toutput)));

    // Allocate Output Tensor
    AllocateOutputSetMklShape(context, kOutputIndexDst, output_tensor,
                              output_tf_shape, output_mkl_shape);
  }

  // LOCKS_EXCLUDED annotation ensures that the lock (mu_) cannot
  // be acquired before entering the function, since it is acquired
  // inside the function.
  inline bool IsWeightCacheEmpty(OpKernelContext* context) LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    return (weight_oi_.NumElements() == 0);
  }

  // Cache the converted weight in a persistent tensor.
  // Only one thread can execute this method at any given time.
  void CacheWeight(
      OpKernelContext* context,
      const std::shared_ptr<dnnl::inner_product_forward::primitive_desc>&
          matmul_fwd_pd,
      Tweight* weight_data, const Tensor& weight_tensor,
      MklDnnData<Tweight>& weight, const memory::desc& weight_md)
      LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    const Tensor& weight_t = *weight_oi_.AccessTensor(context);

    // If the weights are already cached, there's nothing to do
    if (weight_t.NumElements() > 0) {
      return;
    }

#ifdef ENABLE_ONEDNN_V3
    // For now, cache weights only for blocked format
    if (weight_md.get_format_kind() != memory::format_kind::blocked) {
      return;
    }
#endif  // ENABLE_ONEDNN_V3

    // reorder and cache the weight
    weight.SetUsrMem(weight_md, &weight_tensor);
    weight.CheckReorderToOpMem(
        MEMORY_PD_WITHOUT_DATA(matmul_fwd_pd.get()->PRIMITIVE_DESC_WEIGHTS,
                               cpu_engine_),
        context);
    weight_data = static_cast<Tweight*>(weight.GetOpMem().get_data_handle());

    Tensor* weight_tensor_ptr = nullptr;

    size_t weight_size = matmul_fwd_pd.get()->PRIMITIVE_DESC_WEIGHTS.get_size();
    TensorShape weight_tf_shape;
    weight_tf_shape.AddDim(weight_size / sizeof(Tweight));

    OP_REQUIRES_OK(context, context->allocate_persistent(
                                DataTypeToEnum<Tweight>::value, weight_tf_shape,
                                &weight_oi_, &weight_tensor_ptr));

    void* weight_oi_t_data = weight.GetTensorBuffer(weight_tensor_ptr);
    memcpy(weight_oi_t_data, weight_data, weight_size);

    // cache the memory descriptor
    auto expected_md = GET_WEIGHTS_DESC_FROM_OP_PD(matmul_fwd_pd);
#ifndef ENABLE_ONEDNN_V3
    Tensor* weight_md_tensor_ptr = nullptr;
    TensorShape weight_mkl_format;
    weight_mkl_format.AddDim(sizeof(expected_md) / sizeof(Tweight));

    OP_REQUIRES_OK(
        context, context->allocate_persistent(DataTypeToEnum<Tweight>::value,
                                              weight_mkl_format, &weight_oi_md_,
                                              &weight_md_tensor_ptr));
    *reinterpret_cast<memory::desc*>(
        weight_md_tensor_ptr->flat<Tweight>().data()) = expected_md;
#else
    weight_oi_md_ = FilterMemoryDesc(
        expected_md.get_ndims(), expected_md.get_inner_nblks(),
        expected_md.get_data_type(), expected_md.get_dims(),
        expected_md.get_inner_blks(), expected_md.get_inner_idxs(),
        expected_md.get_strides());
#endif  // !ENABLE_ONEDNN_V3
  }

  Tweight* GetCachedWeight(OpKernelContext* context,
                           const memory::desc& expected_md)
      LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    const Tensor& weight_t = *weight_oi_.AccessTensor(context);
#ifndef ENABLE_ONEDNN_V3
    const Tensor& weight_md_t = *weight_oi_md_.AccessTensor(context);

    // Check if the memory descriptor of the cached weight is same as
    // expected_md. if so use the cached memory, else return NULL
    if (weight_md_t.flat<Tweight>().size()) {
      const memory::desc& stored_md =
          *(static_cast<memory::desc*>(weight_md_t.data()));
      if (stored_md == expected_md) {
        return static_cast<Tweight*>(
            const_cast<Tweight*>(weight_t.flat<Tweight>().data()));
      }
    }
    return nullptr;
#else
    // Return the cached weights only if the dimensions of the cached weights
    // and the current weights match. Otherwise, return nullptr.
    //
    // TODO(intel-tf): The following check assumes that all dimensions are
    // known before checking for equality. We may have to modify it in the
    // future once we support runtime dimensions (especially if the dimensions
    // are still unknown at this point).
    if (weight_oi_md_ ==
        FilterMemoryDesc(expected_md.get_ndims(), expected_md.get_inner_nblks(),
                         expected_md.get_data_type(), expected_md.get_dims(),
                         expected_md.get_inner_blks(),
                         expected_md.get_inner_idxs(),
                         expected_md.get_strides())) {
      return static_cast<Tweight*>(
          const_cast<Tweight*>(weight_t.flat<Tweight>().data()));
    }
    return nullptr;
#endif  // !ENABLE_ONEDNN_V3
  }

  engine cpu_engine_ = engine(ENGINE_CPU, 0);

 protected:
  // Tensor to save reordered weight
  mutex mu_;
  PersistentTensor weight_oi_ GUARDED_BY(mu_);
#ifndef ENABLE_ONEDNN_V3
  PersistentTensor weight_oi_md_ GUARDED_BY(mu_);
#else
  FilterMemoryDesc weight_oi_md_ GUARDED_BY(mu_);
#endif  // !ENABLE_ONEDNN_V3

  bool is_weight_const_ = false;

  const int kInputIndexSrc = 0;
  const int kInputIndexWeight = 1;
  const int kInputIndexBias = 2;
  const int kOutputIndexDst = 0;
};

using dnnl::matmul;

namespace {

struct MklMatMulParams {
  string prefix;
  memory::dims a_dims;
  memory::dims b_dims;
  memory::dims c_dims;
  memory::dims a_strides;
  memory::dims b_strides;
  memory::dims c_strides;
  struct PostOpParam {
    string name;
    std::vector<float> param;
    memory::dims dims;
    memory::data_type data_type;
    memory::format_tag format_tag;
  };
  std::vector<PostOpParam> post_op_params;

  MklMatMulParams(string prefix, memory::dims a_dims, memory::dims b_dims,
                  memory::dims c_dims, memory::dims a_strides,
                  memory::dims b_strides, memory::dims c_strides)
      : prefix(prefix),
        a_dims(a_dims),
        b_dims(b_dims),
        c_dims(c_dims),
        a_strides(a_strides),
        b_strides(b_strides),
        c_strides(c_strides) {}
};

template <typename Tlhs, typename Trhs, typename Toutput>
class MklMatMulPrimitive : public MklPrimitive {
 public:
  explicit MklMatMulPrimitive(const MklMatMulParams& params)
      : MklPrimitive(engine(ENGINE_CPU, 0)) {
    // Create matmul primitive
    Setup(params);
  }

  ~MklMatMulPrimitive() {}

  void Execute(const std::shared_ptr<stream>& stream, const Tlhs* a_data,
               const Trhs* b_data, const Toutput* c_data,
               void* mul_data = nullptr, void* add_data = nullptr) {
#ifdef DNNL_AARCH64_USE_ACL
    mutex_lock lock(primitive_execution_mu_);
#endif
#if defined(ENABLE_DNNL_THREADPOOL) && !defined(ENABLE_ONEDNN_V3)
    context_.a_mem->set_data_handle(
        static_cast<void*>(const_cast<Tlhs*>(a_data)), *stream);
    context_.b_mem->set_data_handle(
        static_cast<void*>(const_cast<Trhs*>(b_data)), *stream);
    context_.c_mem->set_data_handle(
        static_cast<void*>(const_cast<Toutput*>(c_data)), *stream);
    if (mul_data != nullptr)
      context_.mul_mem->set_data_handle(mul_data, *stream);
    if (add_data != nullptr)
      context_.add_mem->set_data_handle(add_data, *stream);
#else
    context_.a_mem->set_data_handle(
        static_cast<void*>(const_cast<Tlhs*>(a_data)));
    context_.b_mem->set_data_handle(
        static_cast<void*>(const_cast<Trhs*>(b_data)));
    context_.c_mem->set_data_handle(
        static_cast<void*>(const_cast<Toutput*>(c_data)));
    if (mul_data != nullptr) context_.mul_mem->set_data_handle(mul_data);
    if (add_data != nullptr) context_.add_mem->set_data_handle(add_data);
#endif  // ENABLE_DNNL_THREADPOOL && !ENABLE_ONEDNN_V3
    execute_primitives(context_.matmul_primitives, stream, context_.net_args);

    // After execution, set data handle back
    context_.a_mem->set_data_handle(DummyData);
    context_.b_mem->set_data_handle(DummyData);
    context_.c_mem->set_data_handle(DummyData);
    if (mul_data != nullptr) context_.mul_mem->set_data_handle(DummyData);
    if (add_data != nullptr) context_.add_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<dnnl::matmul::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.prim_desc;
  }

 private:
  // Primitive reuse context for MatMul op
  struct MklMatMulContext {
    // OneDNN memory.
    std::shared_ptr<dnnl::memory> a_mem;
    std::shared_ptr<dnnl::memory> b_mem;
    std::shared_ptr<dnnl::memory> c_mem;
    std::shared_ptr<dnnl::memory> mul_mem;
    std::shared_ptr<dnnl::memory> add_mem;

    // Descriptor and primitive-descriptor for MatMul.
#ifndef ENABLE_ONEDNN_V3
    std::shared_ptr<matmul::desc> desc;
#endif  // !ENABLE_ONEDNN_V3
    std::shared_ptr<matmul::primitive_desc> prim_desc;

    // Memory descriptors.
    std::shared_ptr<dnnl::memory::desc> a_md;
    std::shared_ptr<dnnl::memory::desc> b_md;
    std::shared_ptr<dnnl::memory::desc> c_md;
    std::shared_ptr<dnnl::memory::desc> mul_md;
    std::shared_ptr<dnnl::memory::desc> add_md;

    // MatMul primitive.
    std::vector<dnnl::primitive> matmul_primitives;
    std::vector<std::unordered_map<int, memory>> net_args;

    MklMatMulContext()
        : a_mem(nullptr),
          b_mem(nullptr),
          c_mem(nullptr),
          mul_mem(nullptr),
          add_mem(nullptr),
#ifndef ENABLE_ONEDNN_V3
          desc(nullptr),
#endif  // !ENABLE_ONEDNN_V3
          prim_desc(nullptr),
          a_md(nullptr),
          b_md(nullptr),
          c_md(nullptr),
          mul_md(nullptr),
          add_md(nullptr) {
    }
  };

  void Setup(const MklMatMulParams& params) {
    std::shared_ptr<dnnl::primitive> matmul_primitive = nullptr;

    // Create MatMul descriptor and primitive descriptor.
    context_.a_md.reset(new memory::desc({params.a_dims}, MklDnnType<Tlhs>(),
                                         params.a_strides));

    context_.b_md.reset(new memory::desc({params.b_dims}, MklDnnType<Trhs>(),
                                         params.b_strides));

    context_.c_md.reset(new memory::desc({params.c_dims}, MklDnnType<Toutput>(),
                                         params.c_strides));
    // Create matmul.
#ifndef ENABLE_ONEDNN_V3
    context_.desc.reset(
        new matmul::desc(*context_.a_md, *context_.b_md, *context_.c_md));
#endif  // !ENABLE_ONEDNN_V3

    // Check if there is any fusion as post-ops
    auto const& post_op_params = params.post_op_params;
    dnnl::primitive_attr post_ops_attr;
    dnnl::post_ops post_ops;
    if (!post_op_params.empty()) {
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "output_scale") {
#ifndef ENABLE_ONEDNN_V3
          // TODO(intel-tf): Verify if this code is needed. If not, it needs to
          // be removed.
          DCHECK_EQ(post_op_param.param.size(), 1);
          std::vector<float> scales;
          scales.push_back(post_op_param.param[0]);
          post_ops_attr.set_output_scales(0, scales);
#endif  // !ENABLE_ONEDNN_V3
        } else if (post_op_param.name == "mul") {
          context_.mul_md.reset(new memory::desc({post_op_param.dims},
                                                 post_op_param.data_type,
                                                 post_op_param.format_tag));
          post_ops.append_binary(dnnl::algorithm::binary_mul, *context_.mul_md);
        } else if (post_op_param.name == "add") {
          context_.add_md.reset(new memory::desc({post_op_param.dims},
                                                 post_op_param.data_type,
                                                 post_op_param.format_tag));
          post_ops.append_binary(dnnl::algorithm::binary_add, *context_.add_md);
        } else {
          DCHECK((post_op_param.name == "output_scale"));
        }
      }
      post_ops_attr.set_post_ops(post_ops);
    }
#ifndef ENABLE_ONEDNN_V3
    context_.prim_desc.reset(
        new matmul::primitive_desc(*context_.desc, post_ops_attr, cpu_engine_));
#else
    context_.prim_desc.reset(
        new matmul::primitive_desc(cpu_engine_, *context_.a_md, *context_.b_md,
                                   *context_.c_md, post_ops_attr));
#endif  // !ENABLE_ONEDNN_V3

    // Create memory primitive based on dummy data.
    context_.a_mem.reset(
        new dnnl::memory(*context_.a_md, cpu_engine_, DummyData));
    context_.b_mem.reset(
        new dnnl::memory(*context_.b_md, cpu_engine_, DummyData));
    context_.c_mem.reset(
        new dnnl::memory(*context_.c_md, cpu_engine_, DummyData));

    // Create matmul primitive.
    matmul_primitive.reset(new dnnl::matmul(*context_.prim_desc));
    context_.net_args.push_back({{DNNL_ARG_SRC, *context_.a_mem},
                                 {DNNL_ARG_WEIGHTS, *context_.b_mem},
                                 {DNNL_ARG_DST, *context_.c_mem}});

    if (!post_op_params.empty()) {
      int count = 0;
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "mul") {
          context_.mul_mem.reset(
              new dnnl::memory(*context_.mul_md, cpu_engine_, DummyData));
          context_.net_args[0].insert(
              {DNNL_ARG_ATTR_MULTIPLE_POST_OP(count) | DNNL_ARG_SRC_1,
               *context_.mul_mem});
          count++;
        } else if (post_op_param.name == "add") {
          context_.add_mem.reset(
              new dnnl::memory(*context_.add_md, cpu_engine_, DummyData));
          context_.net_args[0].insert(
              {DNNL_ARG_ATTR_MULTIPLE_POST_OP(count) | DNNL_ARG_SRC_1,
               *context_.add_mem});
          count++;
        }
      }
    }

    context_.matmul_primitives.push_back(*matmul_primitive);
    return;
  }

  struct MklMatMulContext context_;

#ifdef DNNL_AARCH64_USE_ACL
  mutex primitive_execution_mu_;
#endif
};

template <typename T, typename Tlhs, typename Trhs, typename Toutput>
class MklMatMulPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklMatMulPrimitive<Tlhs, Trhs, Toutput>* Get(
      const MklMatMulParams& params, bool do_not_cache) {
    MklMatMulPrimitive<Tlhs, Trhs, Toutput>* matmul_prim = nullptr;

    if (do_not_cache) {
      // Always create new primitive
      matmul_prim = new MklMatMulPrimitive<Tlhs, Trhs, Toutput>(params);
    } else {
      // Try to find a suitable one in pool
      matmul_prim = dynamic_cast<MklMatMulPrimitive<Tlhs, Trhs, Toutput>*>(
          MklMatMulPrimitiveFactory<T, Tlhs, Trhs, Toutput>::GetInstance()
              .GetMklMatMul(params));
      if (matmul_prim == nullptr) {
        matmul_prim = new MklMatMulPrimitive<Tlhs, Trhs, Toutput>(params);
        MklMatMulPrimitiveFactory<T, Tlhs, Trhs, Toutput>::GetInstance()
            .SetMklMatMul(params, matmul_prim);
      }
    }

    return matmul_prim;
  }

 private:
  MklMatMulPrimitiveFactory() {}
  ~MklMatMulPrimitiveFactory() {}

  static MklMatMulPrimitiveFactory& GetInstance() {
    static MklMatMulPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklMatMulParams& params) {
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(params.prefix);
    key_creator.AddAsKey(params.a_dims);
    key_creator.AddAsKey(params.b_dims);
    key_creator.AddAsKey(params.c_dims);
    key_creator.AddAsKey(params.a_strides);
    key_creator.AddAsKey(params.b_strides);
    key_creator.AddAsKey(params.c_strides);
    key_creator.AddAsKey(typeid(T).name());
    key_creator.AddAsKey(typeid(Tlhs).name());
    key_creator.AddAsKey(typeid(Trhs).name());
    key_creator.AddAsKey(typeid(Toutput).name());

    // Generate keys for post-ops
    for (auto const& post_op_param : params.post_op_params) {
      if (post_op_param.name == "output_scale") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
      } else if (post_op_param.name == "mul" || post_op_param.name == "add") {
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.dims);
      } else {
        return string("not_a_key");
      }
    }

    return key_creator.GetKey();
  }

  MklPrimitive* GetMklMatMul(const MklMatMulParams& params) {
    string key = CreateKey(params);
    return this->GetOp(key);
  }

  void SetMklMatMul(const MklMatMulParams& params, MklPrimitive* op) {
    string key = CreateKey(params);
    this->SetOp(key, op);
  }
};

template <typename T>
void dnnl_gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
               float alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
               float beta, T* c, int64_t ldc, OpKernelContext* ctx = nullptr) {
  using dims = dnnl::memory::dims;

  // Prepare strides based on the transa and transb flags: transposed
  // matrices have strides swapped
  dims a_dims = dims{m, k};
  dims b_dims = dims{k, n};
  dims c_dims = dims{m, n};
  dims a_strides = tolower(transa) == 'n' ? dims{lda, 1} : dims{1, lda};
  dims b_strides = tolower(transb) == 'n' ? dims{ldb, 1} : dims{1, ldb};
  dims c_strides = dims{ldc, 1};

  // MklMatMul uses const alpha and beta, make guarantee here to ensure
  // they are never changed.
  DCHECK_EQ(alpha, 1.0f);
  DCHECK_EQ(beta, 0.f);

  MklMatMulParams params("dnnl_gemm", a_dims, b_dims, c_dims, a_strides,
                         b_strides, c_strides);
  MklMatMulPrimitive<T, T, T>* matmul_prim =
      MklMatMulPrimitiveFactory<T, T, T, T>::Get(params, 0);

  // Execute matmul primitive.
  std::shared_ptr<stream> cpu_stream;
  MklDnnThreadPool eigen_tp(ctx);
  cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));
  matmul_prim->Execute(cpu_stream, a, b, c);
}

  if (do_not_cache)
    delete matmul_prim;
}

}  // anonymous namespace

#undef APPEND_ELTWISE
#undef APPEND_ELTWISE_RELU6
#undef SET_MKL_LAYOUT
#undef FWD_STREAM

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_MATMUL_OPS_COMMON_H_
