#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "take_grad_lib.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA

template <typename Device, typename T, typename Index>
class TakeGradOp : public OpKernel {
 public:
  explicit TakeGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    const Tensor *out_grad;
    OP_REQUIRES_OK(c, c->input("grad", &out_grad));
    OP_REQUIRES(c, out_grad->shape().dims() == 3,
                errors::InvalidArgument("grad has incorrect shape: ",
                                        out_grad->shape().DebugString()));

    OpInputList coords;
    OP_REQUIRES_OK(c, c->input_list("coords", &coords));
    const int N = coords.size();

    std::vector<std::unique_ptr<typename TTypes<T, 2>::Matrix>> value_grads_flat;
    std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>> coords_flat;

    value_grads_flat.reserve(N);
    coords_flat.reserve(N);

    int64 unit_size = out_grad->shape().dim_size(2);
    for (int n = 0; n < N; ++n) {
      const auto &coord = coords[n];
      const TensorShape &coord_shape = coord.shape();
      OP_REQUIRES(c, coord_shape.dims() == 2,
                  errors::InvalidArgument("coords ", n,
                                          " should be a vector, but got shape ",
                                          coord_shape.DebugString()));

      if (coord.NumElements() > 0) {
        coords_flat.emplace_back(new typename TTypes<Index, 2>::ConstMatrix(
                coord.shaped<Index, 2>(coord_shape.dim_sizes())));

        TensorShape value_shape({coord_shape.dim_size(0), unit_size});

        Tensor* value_grad = nullptr;
        OP_REQUIRES_OK(c, c->allocate_output(n, value_shape, &value_grad));

        value_grads_flat.emplace_back(new typename TTypes<T, 2>::Matrix(
                value_grad->shaped<T, 2>(value_shape.dim_sizes())));
      }
    }

    auto out_grad_flat = out_grad->shaped<T, 3>(out_grad->shape().dim_sizes());

#if GOOGLE_CUDA
    if (std::is_same<Device, GPUDevice>::value) {
      std::cout << "TakeGrad GPU" << std::endl;
      TakeGradGPU<T, Index>(c, out_grad_flat, coords_flat, &value_grads_flat);
      return;
    }
#endif  // GOOGLE_CUDA
    std::cout << "TakeGrad CPU" << std::endl;
    TakeGradCPU<T, Index>(c->device(), out_grad_flat, coords_flat, &value_grads_flat);
  }
};

#define REGISTER_CPU(type, index_type)                                 \
  REGISTER_KERNEL_BUILDER(Name("TakeGrad")                             \
                              .Device(tensorflow::DEVICE_CPU)          \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                              TakeGradOp<CPUDevice, type, index_type>);

#define REGISTER_CPU_ALL(type)     \
  REGISTER_CPU(type, int32);       \
  REGISTER_CPU(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_ALL);
#undef REGISTER_CPU_ALL
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(type, index_type)                                 \
  REGISTER_KERNEL_BUILDER(Name("TakeGrad")                             \
                              .Device(tensorflow::DEVICE_GPU)          \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                              TakeGradOp<GPUDevice, type, index_type>);


#define REGISTER_GPU_ALL(type)     \
  REGISTER_GPU(type, int32);       \
  REGISTER_GPU(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_ALL);
#undef REGISTER_GPU_ALL
#undef REGISTER_GPU
#endif  // GOOGLE_CUDA
/*
REGISTER_CPU(float, int32)
*/

