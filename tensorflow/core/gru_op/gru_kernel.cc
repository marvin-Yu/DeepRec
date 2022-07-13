#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/gru_op/gru_func.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class GRUOp : public OpKernel {
 public:
  explicit GRUOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& x = context->input(0);
    const Tensor& h2h = context->input(1);
    const Tensor& i2h = context->input(2);
    const Tensor& h2hBias = context->input(3);
    const Tensor& i2hBias = context->input(4);

    OP_REQUIRES(context, x.dims() == 3,
                errors::InvalidArgument("x ndims must be 3: ", x.dims()));

    int batch_size = x.dim_size(0);
    int rounds = x.dim_size(1);
    int elts = x.dim_size(2);

    if (batch_size == 1) {
      OP_REQUIRES(context, h2h.dims() == 2,
                  errors::InvalidArgument("h2h ndims must be 2: ", h2h.dims()));
      OP_REQUIRES(context, i2h.dims() == 2,
                  errors::InvalidArgument("i2h ndims must be 2: ", i2h.dims()));
      OP_REQUIRES(
          context, h2hBias.dims() == 1,
          errors::InvalidArgument("h2hBias ndims must be 1: ", h2hBias.dims()));
      OP_REQUIRES(
          context, i2hBias.dims() == 1,
          errors::InvalidArgument("i2hBias ndims must be 1: ", i2hBias.dims()));
      OP_REQUIRES(context, elts == h2h.dim_size(0),
                  errors::InvalidArgument("h2h dim[0] must equel to: ", elts));
      OP_REQUIRES(context, elts == i2h.dim_size(0),
                  errors::InvalidArgument("i2h dim[0] must equel to: ", elts));

    } else {
      OP_REQUIRES(context, h2h.dims() == 3,
                  errors::InvalidArgument("h2h ndims must be 3: ", h2h.dims()));
      OP_REQUIRES(context, i2h.dims() == 3,
                  errors::InvalidArgument("i2h ndims must be 3: ", i2h.dims()));
      OP_REQUIRES(
          context, h2hBias.dims() == 2,
          errors::InvalidArgument("h2hBias ndims must be 2: ", h2hBias.dims()));
      OP_REQUIRES(
          context, i2hBias.dims() == 2,
          errors::InvalidArgument("i2hBias ndims must be 2: ", i2hBias.dims()));
      OP_REQUIRES(context, elts == h2h.dim_size(1),
                  errors::InvalidArgument("h2h dim[1] must equel to: ", elts));
      OP_REQUIRES(context, elts == i2h.dim_size(1),
                  errors::InvalidArgument("i2h dim[1] must equel to: ", elts));
      OP_REQUIRES(
          context, batch_size == h2h.dim_size(0),
          errors::InvalidArgument("h2h dim[0] must equel to: ", batch_size));
      OP_REQUIRES(
          context, batch_size == i2h.dim_size(0),
          errors::InvalidArgument("i2h dim[0] must equel to: ", batch_size));
      OP_REQUIRES(context, batch_size == h2hBias.dim_size(0),
                  errors::InvalidArgument("h2hBias dim[0] must equel to: ",
                                          batch_size));
      OP_REQUIRES(context, batch_size == i2hBias.dim_size(0),
                  errors::InvalidArgument("i2hBias dim[0] must equel to: ",
                                          batch_size));
    }

    //[PROF-STATS]
    int64 delta = batch_size * rounds * (12 * elts * elts + 25 * elts);
    if (context->traced_infos()) {
      context->traced_infos()->RecordFlops(delta, requested_device());
    }
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "FLOPs = " << delta << ", " << type_string() << ", "
                << name() << ", " << x.shape().DebugString() << ", ";
    }

    // Create an output tensor
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &y));

    const T* x_p = x.flat<T>().data();
    T* y_p = y->flat<T>().data();

    const T* h2h_p = h2h.flat<T>().data();
    const T* i2h_p = i2h.flat<T>().data();
    const T* h2hBias_p = h2hBias.flat<T>().data();
    const T* i2hBias_p = i2hBias.flat<T>().data();

    // Do the computation.
    OP_REQUIRES_OK(context, GRUFunctor<Device, T>()(
                                context->eigen_device<Device>(), context,
                                batch_size, rounds, elts, y_p, x_p, h2h_p,
                                i2h_p, h2hBias_p, i2hBias_p));
  };
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BlazeGRU").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      GRUOp<CPUDevice, T>);
REGISTER_CPU(float);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                           \
  /* Declare explicit instantiations in kernel_GRU.cu.cc. */      \
  extern template struct GRUFunctor<GPUDevice, T>;                \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BlazeGRU").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      GRUOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
