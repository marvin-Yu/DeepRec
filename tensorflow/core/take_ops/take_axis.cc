#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "take_axis_lib.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA

template <typename Device, typename T, typename Index>
class TakeAxisOp : public OpKernel {
 public:
  explicit TakeAxisOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(c, c->GetAttr("size", &size_));
    OP_REQUIRES_OK(c, c->GetAttr("reverse", &reverse_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor &input = c->input(0);
    const TensorShape &input_shape = input.shape();
    OP_REQUIRES(c, input_shape.dims() > 1,
                errors::InvalidArgument("input shape ", input_shape.DebugString(),
                                        " should be more than 1-D"));
    OP_REQUIRES(c, axis_ > 0 && axis_ < input_shape.dims(),
                errors::InvalidArgument("axis ", axis_, " should be in (0,",
                                        input_shape.dims(), ")"));

    int64 axis_size = input_shape.dim_size(axis_);
    OP_REQUIRES(c, size_ > 0,
                errors::InvalidArgument("size ", size_, " should > 0"));

    const Tensor &begin = c->input(1);
    OP_REQUIRES(c, begin.shape().dims() == axis_,
                errors::InvalidArgument("begin shape dim ", begin.shape().DebugString(),
                                        " should be ", axis_));

    int64 before_dim = 1;
    for (int i = 0; i < axis_; ++i) {
      OP_REQUIRES(c, begin.shape().dim_size(i) == input_shape.dim_size(i),
                  errors::InvalidArgument("begin shape ", begin.shape().DebugString(),
                                          " should be equal with input shape[0:", axis_+1, "] ",
                                          input_shape.DebugString()));
      before_dim *= input_shape.dim_size(i);
    }

    int64 after_dim = 1;
    for (int i = axis_ + 1; i < input_shape.dims(); ++i) {
      after_dim *= input_shape.dim_size(i);
    }

    TensorShape output_shape(input_shape);
    output_shape.set_dim(axis_, size_);

    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() > 0) {
      auto input_flat = input.shaped<T, 3>({before_dim, axis_size, after_dim});
      auto begin_flat = begin.flat<Index>();

      auto output_flat = output->shaped<T, 3>({before_dim, size_, after_dim});
#if GOOGLE_CUDA
      if (std::is_same<Device, GPUDevice>::value) {
        //printf("TakeAxis GPU %s -> %s\n", input_shape.DebugString().c_str(),
        //       output_shape.DebugString().c_str());
        //std::cout << "TakeAxis GPU " << input_shape.DebugString()
        //    << " -> " << output_shape.DebugString() << std::endl;
        TakeAxisGPU<T, Index>(c, input_flat, begin_flat, reverse_, &output_flat);
        return;
      }
#endif  // GOOGLE_CUDA
      //printf("TakeAxis CPU %s -> %s\n", input_shape.DebugString().c_str(),
      //       output_shape.DebugString().c_str());
      //std::cout << "TakeAxis CPU " << input_shape.DebugString()
      //    << " -> " << output_shape.DebugString() << std::endl;
      TakeAxisCPU<T, Index>(c->device(), input_flat, begin_flat, reverse_, &output_flat);
    }
  }

 private:
  int axis_;
  int size_;
  bool reverse_;
};

#define REGISTER_CPU(type, index_type)                                 \
  REGISTER_KERNEL_BUILDER(Name("TakeAxis")                            \
                              .Device(tensorflow::DEVICE_CPU)          \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Index"),    \
                              TakeAxisOp<CPUDevice, type, index_type>);

#define REGISTER_GPU(type, index_type)                                 \
  REGISTER_KERNEL_BUILDER(Name("TakeAxis")                            \
                              .Device(tensorflow::DEVICE_GPU)          \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Index"),    \
                              TakeAxisOp<GPUDevice, type, index_type>);

#define REGISTER_CPU_ALL(type)     \
  REGISTER_CPU(type, int32);       \
  REGISTER_CPU(type, int64);

#define REGISTER_GPU_ALL(type)     \
  REGISTER_GPU(type, int32);       \
  REGISTER_GPU(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_ALL);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_ALL);
#undef REGISTER_CPU_ALL
#undef REGISTER_CPU
#undef REGISTER_GPU_ALL
#undef REGISTER_GPU
/*
REGISTER_CPU(float, int32)
*/

