/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <memory>
#include <numeric>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

std::string DtypeToString(DataType dtype) {
  switch(dtype) {
    case DT_FLOAT:
      return "FLOAT";
    case DT_HALF:
      return "HALF";
    default:
      return "UNKNOW";
  }

  return "UNKNOW";
}

template <typename T>
class CheckInputFiniteOp : public OpKernel {
 public:
  explicit CheckInputFiniteOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dump_input", &dump_input_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    OP_REQUIRES(
        context, input.dtype() == DT_HALF || input.dtype() == DT_FLOAT,
        errors::Unimplemented("CheckInputFiniteOp only available for float16/half type"));
    const int data_len = input.shape().num_elements();
    auto input_data = input.flat<T>().data();
    //LOG(INFO) << name() << "dump input: " << dump_input_ << " value[0]: " << input_data[0];
    int count = 0;
    std::string data_str;
    for (int i = 0 ; i < data_len; i++) {
      if (!std::isfinite(static_cast<float>(input_data[i]))) {
        if (count < 3) {
          char binary_buf[100];
          sprintf(binary_buf, "%f, binary=%x", input_data[i], input_data[i]);
          std::string bi_s(binary_buf);
          data_str = data_str + ", data[" + std::to_string(i) + "]=" + bi_s;
        }
        count++;
      }
    }
    if (count) {
      std::string log_str = "find nan or inf value in " + name() +
                            ", op:" + type_string() + ", dtype:" + DtypeToString(input.dtype()) +
                            ", shape:" + input.shape().DebugString() + ", total " + std::to_string(count) + data_str;
      LOG(INFO) << log_str;
    }
    context->set_output(0, input);
  }
 private:
  bool dump_input_ = false;
  //static int nan_node_count;
};

//int CheckInputFiniteOp<float>::nan_node_count = 0;
//int CheckInputFiniteOp<Eigen::half>::nan_node_count = 0;

#define REGISTER_CHECK_INPUT_FINITE_OP(T) \
  REGISTER_KERNEL_BUILDER(Name("CheckInputFinite")   \
                               .Device(DEVICE_CPU)  \
                               .TypeConstraint<T>("T")   \
                               , CheckInputFiniteOp<T>); \
  REGISTER_KERNEL_BUILDER(Name("CheckInputFinite")      \
                               .Device(DEVICE_GPU)          \
                               .TypeConstraint<T>("T")  \
                               .HostMemory("x")         \
                               , CheckInputFiniteOp<T>);

REGISTER_CHECK_INPUT_FINITE_OP(float)
REGISTER_CHECK_INPUT_FINITE_OP(Eigen::half)

#undef REGISTER_CHECK_INPUT_FINITE_OP

}  // namespace tensorflow
