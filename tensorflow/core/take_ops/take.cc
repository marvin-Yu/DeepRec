#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "take_lib.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA

template <typename Device, typename T, typename Index>
class TakeOp : public OpKernel {
 public:
  explicit TakeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    OpInputList values;
    OP_REQUIRES_OK(c, c->input_list("values", &values));
    const int N = values.size();

    OpInputList coords;
    OP_REQUIRES_OK(c, c->input_list("coords", &coords));
    OP_REQUIRES(c, coords.size() == N,
                errors::InvalidArgument("count of coords ", coords.size(),
                                        " should be as many as values ", values.size()));

    const Tensor *output_shape;
    OP_REQUIRES_OK(c, c->input("output_shape", &output_shape));
    OP_REQUIRES(c, IsLegacyVector(output_shape->shape()),
                errors::InvalidArgument("output_shape's shape should be a vector, got shape ",
                                        output_shape->shape().DebugString()));
    OP_REQUIRES(c, output_shape->NumElements() == 2,
                errors::InvalidArgument(
                    "output_shape has incorrect number of elements: ",
                    output_shape->NumElements(), " should be: ", 2));

    std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>> values_flat;
    std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>> coords_flat;

    values_flat.reserve(N);
    coords_flat.reserve(N);

    int64 unit_size = 0;
    for (int n = 0; n < N; ++n) {
      const auto &value = values[n];
      const TensorShape &value_shape = value.shape();
      OP_REQUIRES(c, value_shape.dims() == 2,
                  errors::InvalidArgument("values ", n,
                                          " should be a vector, but got shape ",
                                          value_shape.DebugString()));

      const auto &coord = coords[n];
      const TensorShape &coord_shape = coord.shape();
      OP_REQUIRES(c, coord_shape.dims() == 2,
                  errors::InvalidArgument("coords ", n,
                                          " should be a vector, but got shape ",
                                          coord_shape.DebugString()));

      if (n == 0) {
        unit_size = value_shape.dim_size(1);
      } else {
        OP_REQUIRES(c, unit_size == value_shape.dim_size(1),
                    errors::InvalidArgument("values ", n,
                                            " should be a vector(*,", unit_size,
                                            "), but got shape ",
                                            value_shape.DebugString()));
      }

      OP_REQUIRES(c, value_shape.dim_size(0) == coord_shape.dim_size(0),
                  errors::InvalidArgument("shape of values & coords ", n,
                                          " at dim0 should be same, ",
                                          value_shape.DebugString(), " VS ",
                                          coord_shape.DebugString()));
       
      if (value.NumElements() > 0) {
        values_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
                value.shaped<T, 2>(value_shape.dim_sizes())));

        coords_flat.emplace_back(new typename TTypes<Index, 2>::ConstMatrix(
                coord.shaped<Index, 2>(coord_shape.dim_sizes())));
      }
    }

    auto output_shape_vec = output_shape->flat<Index>();
    TensorShape output_tensor_shape({output_shape_vec(0), output_shape_vec(1), unit_size});

    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_tensor_shape, &output));
    if (output->NumElements() > 0) {
      auto output_flat = output->shaped<T, 3>(output_tensor_shape.dim_sizes());
#if GOOGLE_CUDA
      if (std::is_same<Device, GPUDevice>::value) {
        std::cout << "Take GPU" << std::endl;
        TakeGPU<T, Index>(c, values_flat, coords_flat, &output_flat);
        return;
      }
#endif  // GOOGLE_CUDA
      std::cout << "Take CPU" << std::endl;
      TakeCPU<T, Index>(c->device(), values_flat, coords_flat, &output_flat);
    }
  }
};

#define REGISTER_CPU(type, index_type)                                 \
  REGISTER_KERNEL_BUILDER(Name("Take")                                 \
                              .Device(tensorflow::DEVICE_CPU)          \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices")  \
                              .HostMemory("output_shape"),             \
                              TakeOp<CPUDevice, type, index_type>);

#define REGISTER_GPU(type, index_type)                                 \
  REGISTER_KERNEL_BUILDER(Name("Take")                                 \
                              .Device(tensorflow::DEVICE_GPU)          \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices")  \
                              .HostMemory("output_shape"),             \
                              TakeOp<GPUDevice, type, index_type>);

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

