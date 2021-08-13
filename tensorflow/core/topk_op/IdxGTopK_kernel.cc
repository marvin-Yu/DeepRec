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
//The 3rd input shows the input idx range for each group, in form [a0, a1, ..., an)
//The 4th input shows the output idx range for each group, in form [0, b1, ..., bn)

template <typename Device, typename T>
class IdxGTopK : public OpKernel {
 public:
  explicit IdxGTopK(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //output[dst_idx[i]:dst_idx[i+1]) = topk of input[src_idx[i]:src_idx[i+1])
    const Tensor & input_tensor = context->input(0);
    const auto& input = input_tensor.flat_inner_dims<T>();

    int k = context->input(1).scalar<int>()();

    const Tensor & src_idx_tensor = context->input(2);
    const auto& src_idx = src_idx_tensor.vec<int>();
    const Tensor & dst_idx_tensor = context->input(3);
    const auto& dst_idx = dst_idx_tensor.vec<int>();

    //TODO: remove this check if k >=3 implemented
    OP_REQUIRES(context, 0 < k && k <= 2, 
                errors::InvalidArgument("only 0 < k <= 2 supported, but k=", k));
    OP_REQUIRES(context, src_idx.dimension(0) == dst_idx.dimension(0), 
                errors::InvalidArgument("src_idx and dst_idx must equal, but ", 
                                        src_idx.dimension(0), "!=", dst_idx.dimension(0)));

    int batch_size = input.dimension(0);
    int input_len = input.dimension(1);
    int num_group = dst_idx.dimension(0)-1;
    int output_len = dst_idx(num_group);

    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "IdxGTopK: k=" << k
                << ", batch_size=" << batch_size
                << ", input_len=" << input_len
                << ", num_group=" << num_group
                << ", output_len=" << output_len;
    }


    OP_REQUIRES(context, dst_idx(0) == 0, 
                errors::InvalidArgument("dst_idx(0) shall be 0, but is ", dst_idx(0)));
    OP_REQUIRES(context, output_len <= k * num_group, 
                errors::InvalidArgument("output_len(",output_len,") > k(",k,") * num_group(",num_group,"), ",
                                        "invalid inputs"));

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
        int tail = src_idx(i+1);
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
      Name("IdxGTopK").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      IdxGTopK<CPUDevice, T>);
REGISTER_CPU(double);
REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);
