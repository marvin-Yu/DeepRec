#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/topk_op/util.h"
#include <algorithm>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.

//Impl of grouped topk algorithm
//The third input splits is a vector containing length of each group.
template <typename Device, typename T>
class GTopK : public OpKernel {
 public:
  explicit GTopK(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //output[dst_idx[i]:dst_idx[i+1]) = topk of input[src_idx[i]:src_idx[i+1])
    const Tensor & input_tensor = context->input(0);
    const auto& input = input_tensor.flat_inner_dims<T>();

    int k = context->input(1).scalar<int>()();

    const Tensor & splits_tensor = context->input(2);
    const auto& splits = splits_tensor.vec<int>();

    //TODO: remove this check if k >=3 implemented
    OP_REQUIRES(context, 0 < k && k <= 2, 
                errors::InvalidArgument("only 0 < k <= 2 supported, but k=", k));

    int batch_size = input.dimension(0);
    int input_len = input.dimension(1);
    int num_group = splits.dimension(0);

    Tensor src_idx_tensor, dst_idx_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, splits_tensor.shape(), &src_idx_tensor));
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, splits_tensor.shape(), &dst_idx_tensor));
    auto src_idx = src_idx_tensor.vec<int>();
    auto dst_idx = dst_idx_tensor.vec<int>();

    //TODO: [opt]calculate only once if splits is constant
    // maybe in a seperate kernel
    int sum = 0, output_len = 0;
    for (int i = 0; i < num_group; ++i) {
      src_idx(i) = sum;
      dst_idx(i) = output_len;
      int len = splits(i);
      sum += len;
      output_len += std::min(k, len);
    }

    OP_REQUIRES(context, sum == input_len, 
                errors::InvalidArgument("sum of splits do NOT match size of input: ", sum ,"!=", input_len));

    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "GTopK: k=" << k
                << ", batch_size=" << batch_size
                << ", input_len=" << input_len
                << ", num_group=" << num_group
                << ", output_len=" << output_len;
    }
 
    //Allocate Output
    TensorShape output_shape = input_tensor.shape();
    output_shape.set_dim(output_shape.dims()-1, output_len);
    Tensor *value_output, *index_output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &value_output));
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &index_output));
    auto value = value_output->flat_inner_dims<T>();
    auto index = index_output->flat_inner_dims<int>();

    std::function<void(int64, int64)> shard = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        int head = src_idx(i);
        int tail = head + splits(i);
        for (int b=0; b < batch_size; ++b) {
          const T* v = input.data() + b*input_len;
          T* val = value.data() + b*output_len;
          int* idx = index.data() + b*output_len;
          safe_topk_in_range(v, k, head, tail,
                             /*val=*/val + dst_idx(i), /*idx=*/idx + dst_idx(i));
        }
      }
    };

    const DeviceBase::CpuWorkerThreads* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;
    int avg_group_len = input_len / num_group;
    int64 kCostPerGroup = 4 * avg_group_len * batch_size;
    Shard(num_threads, worker_threads->workers, num_group, kCostPerGroup, shard);
  };
};


// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("GroupedTopK").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      GTopK<CPUDevice, T>);
REGISTER_CPU(double);
REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);
