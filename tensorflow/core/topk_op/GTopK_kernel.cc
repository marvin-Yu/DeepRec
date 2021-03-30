#include "tensorflow/core/framework/op_kernel.h"
#include <algorithm>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


template <typename T>
void safe_topk_with_offset(const T* v, int k, int offset, int len, 
                           T* val_v, int* idx_v){
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

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class GroupedTopK : public OpKernel {
 public:
  explicit GroupedTopK(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //output[dst_idx[i]:dst_idx[i+1]) = topk of input[src_idx[i]:src_idx[i+1])
    const Tensor & input_tensor = context->input(0);
    const T* input = input_tensor.flat_inner_dims<T>().data();
    int k = context->input(1).scalar<int>()();

    const Tensor & splits_tensor = context->input(2);
    const int* splits = splits_tensor.vec<int>().data();

    //TODO: remove this check if k >=3 implemented
    OP_REQUIRES(context, 0 < k && k <= 2, 
                errors::InvalidArgument("only 0 < k <= 2 support, but k=", k));

    int batch_size = input_tensor.dim_size(0);
    int input_len = input_tensor.dim_size(1);
    int num_group = splits_tensor.dim_size(0);

    Tensor src_idx_tensor, dst_idx_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, splits_tensor.shape(), &src_idx_tensor));
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, splits_tensor.shape(), &dst_idx_tensor));
    auto src_idx = src_idx_tensor.vec<int>().data();
    auto dst_idx = dst_idx_tensor.vec<int>().data();

    //TODO: [opt]calculate only once if splits is constant
    // maybe in a seperate kernel
    int sum = 0, output_len = 0;
    for (int i = 0; i < num_group; ++i) {
      src_idx[i] = sum;
      dst_idx[i] = output_len;
      int len = splits[i];
      sum += len;
      output_len += std::min(k, len);
    }

    OP_REQUIRES(context, sum == input_len, 
                errors::InvalidArgument("sum of splits do NOT match size of input: ", sum ,"!=", input_len));
    
    //Allocate Output
    TensorShape output_shape = input_tensor.shape();
    output_shape.set_dim(output_shape.dims()-1, output_len);
    Tensor *value_output, *index_output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &value_output));
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &index_output));
    auto value = value_output->flat_inner_dims<T>().data();
    auto index = index_output->flat_inner_dims<int>().data();

    //TODO: parallelize this calculation
    for (int b=0; b < batch_size; ++b) {
      const T* v = input + b*input_len;
      T* val = value + b*output_len;
      int* idx = index + b*output_len;
      for (int i = 0; i < num_group; ++i) {
        safe_topk_with_offset(v, k, /*offset=*/src_idx[i], /*len=*/splits[i],
                              /*val=*/val + dst_idx[i], /*idx=*/idx + dst_idx[i]);
      }
    }
  };
};


// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("GroupedTopK").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      GroupedTopK<CPUDevice, T>);
REGISTER_CPU(double);
REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);
