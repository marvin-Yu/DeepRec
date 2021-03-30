#include "tensorflow/core/framework/op_kernel.h"
#include <algorithm>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class GroupedTopK : public OpKernel {
 public:
  explicit GroupedTopK(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //output[dst_idx[i]:dst_idx[i+1]) = topk of input[src_idx[i]:src_idx[i+1])
    const Tensor & input_tensor = context->input(0);
    const auto & input = input_tensor.flat_inner_dims<T>();
    int k = context->input(1).scalar<int>();
    const auto & splits = context->input(2).vec<int>();

    //TODO: remove this check if k >=3 implemented
    OP_REQUIRES(context, 0 < k && k <= 2, 
                errors::InvalidArgument("only 0 < k <= 2 support, but k=", k);

    int batch_size = input.dimension(0);
    int input_len = input.dimension(1);
    int num_group = splits.dimension(0);

    //TODO: [opt]calculate only once if splits is constant
    // maybe in a seperate kernel
    int sum = 0, output_len = 0
    for (int i = 0; i < num_group; ++i) {
      src_idx[i] = sum;
      dst_idx[i] = output_len;
      int len = splits[i];
      sum += len;
      output_len += std::min(k, len);
    }

    OP_REQUIRES(context, sum == input_len, 
                errors::InvalidArgument("sum of splits do NOT match size of input: ", sum ,"!=", input_len);
    
    //Allocate Output
    TensorShape output_shape = input_tensor.shape();
    output_shape.set_dim(output_shape.dims()-1, ouput_len);
    Tensor *value_output, *index_output;
    TF_RETURN_IF_ERROR(context->allocate_output(0, output_shape, &value_output));
    TF_RETURN_IF_ERROR(context->allocate_output(1, output_shape, &index_output));
    auto value = value_output->flat_inner_dims();
    auto index = index_output->flat_inner_dims();

    //TODO: parallelize this calculation
    for (int b=0; b < batch_size; ++b) {
      T*   v = input.data() + b*input_len;
      T*   val = value.data() + b*output_len;
      int* idx = index.data() + b*output_len;
      for (int i = 0; i < num_group; ++i) {
        safe_topk_with_offset(v, k, /*offset=*/src_idx[i], /*len=*/src_idx[i+1] - src_idx[i],
                              /*val_v=*/val_v + dst_idx[i], /*idx_v=*/idx_v + dst_idx[i]);
      }
    }
  };
};

template <typename T>
void safe_topk_with_offset(T* v, int k, int offset, int len, 
                           T* value_v, int* idx_v){
  if (len <= k) {
    for (int i = 0; i < len; ++i) {
      idx_v[i] = offset+i;
      val_v[i] = v[idx_v[i]];
    }
    return;
  }
  // len > k
  if (k == 1) {
    int idx = offset;
    for (int i = offset+1; i < offset+len; ++i) {
      if (v[i] > v[idx]) idx = i;
    }
    idx_v[0] = idx; val_v[0] = v[idx];
  } else if (k == 2) {
    int idx0 = offset;
    int idx1 = offset+1;
    if (v[idx0] < v[idx1]) std::swap(idx0, idx1);
    for (int i = offset+2; i < offset+len; ++i) {
      if (v[i] > v[idx1]) {
        idx1 = i;
        if (v[idx0] < v[idx1]) std::swap(idx0, idx1);
      }
    }
    idx_v[0] = idx0; val_v[0] = v[idx0];
    idx_v[1] = idx1; val_v[1] = v[idx1];
  } else {
    //TODO: fast-topk algorithm
  } 
}

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("GroupedTopK").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      GroupedTopK<CPUDevice, T>);
REGISTER_CPU(double);
REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);
