#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"
#include <algorithm>
#include <vector>
#include <iostream>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.

//Impl of grouped topk algorithm
//The third input splits is a vector containing length of each group.
template <typename Device, typename T>
class GTopKV2 : public OpKernel {
 private:
  bool descending_ = true;
 public:
  explicit GTopKV2(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("desc", &descending_));
  }

  void Compute(OpKernelContext* context) override {
    //output[dst_idx[i]:dst_idx[i+1]) = topk of input[src_idx[i]:src_idx[i+1])
    const Tensor & indicator_tensor = context->input(0);
    const auto& indicator = indicator_tensor.flat_inner_dims<int>();

    const Tensor & value_tensor = context->input(1);
    const auto& value = value_tensor.flat_inner_dims<T>();

    int batch_size = indicator.dimension(0);
    int input_len = indicator.dimension(1);

    OP_REQUIRES(context, batch_size  == value.dimension(0) , 
                errors::InvalidArgument("batch size not match", batch_size ,"!=", value.dimension(0)));

    OP_REQUIRES(context, input_len == value.dimension(1) , 
                errors::InvalidArgument("input length not match", input_len ,"!=", value.dimension(1)));

    if (VLOG_IS_ON(1)) {
      LOG(INFO) << ", batch_size=" << batch_size
                << ", input_len=" << input_len;
    }
 
    //Allocate Output
    TensorShape output_shape = indicator_tensor.shape();
    output_shape.set_dim(output_shape.dims()-1, input_len);
    Tensor *index_output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &index_output));
    auto index = index_output->flat_inner_dims<int>();

    std::function<void(int64, int64)> shard = [&](int64 begin, int64 end) {

      //outside for loop for reuse
      std::vector<int> sorted_idx(input_len);
      for (int i = begin; i < end; ++i) {
        std::iota(sorted_idx.begin(),  sorted_idx.end(), 0);
        const int* input_i = indicator.data() + i * input_len;
        const T* input_v = value.data() + i * input_len;
        if(descending_) {
          std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
              return (input_i[a] < input_i[b]) || (input_i[a] == input_i[b] && input_v[a] > input_v[b]);
              });
        } else {
          std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
              return (input_i[a] < input_i[b]) || (input_i[a] == input_i[b] && input_v[a] < input_v[b]);
              });
        } 
        //copy candidate to ouput buffer
        int* index_v = index.data() + i * input_len;
        std::memcpy(index_v, sorted_idx.data(), sizeof(int) * input_len);
      }
    };

    const DeviceBase::CpuWorkerThreads* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int64 kCostPerBatch = 3*input_len*std::log(input_len);
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size, kCostPerBatch, shard);
  };
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("GroupedTopkV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      GTopKV2<CPUDevice, T>);
REGISTER_CPU(double);
REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);


