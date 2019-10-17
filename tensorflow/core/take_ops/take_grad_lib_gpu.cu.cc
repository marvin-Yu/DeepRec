#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

#if GOOGLE_CUDA

#include "tensorflow/core/kernels/gpu_device_array.h"

namespace tensorflow {

template <typename T, typename Index>
void TakeGradGPUImpl(const Eigen::GpuDevice& d,
                     typename TTypes<T, 3>::ConstTensor& out_grad,
                     const GpuDeviceArrayStruct<const Index*>& coord_ptrs,
                     const GpuDeviceArrayStruct<Index>& input_scan,
                     int unit_size, Index total_size,
                     const GpuDeviceArrayStruct<T*>& value_ptrs);

namespace {

template <typename T, typename Index>
void TakeGradGPUCall(
    OpKernelContext* c,
    typename TTypes<T, 3>::ConstTensor& out_grad,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    std::vector<std::unique_ptr<typename TTypes<T, 2>::Matrix>>* value_grads) {
  size_t N = value_grads->size();
  CHECK(coords.size() == N);
  GpuDeviceArrayOnHost<T*> value_ptrs(c, N);
  GpuDeviceArrayOnHost<const Index*> coord_ptrs(c, N);
  OP_REQUIRES_OK(c, value_ptrs.Init());
  OP_REQUIRES_OK(c, coord_ptrs.Init());
  for (size_t n = 0; n < N; ++n) {
    value_ptrs.Set(n, (*value_grads)[n]->data());
    coord_ptrs.Set(n, coords[n]->data());
  }
  OP_REQUIRES_OK(c, value_ptrs.Finalize());
  OP_REQUIRES_OK(c, coord_ptrs.Finalize());

  GpuDeviceArrayOnHost<Index> input_scan(c, N + 1);
  OP_REQUIRES_OK(c, input_scan.Init());
  int64 unit_size = out_grad.dimension(2);
  Index scan = 0;
  input_scan.Set(0, scan);
  for (size_t n = 0; n < N; ++n) {
    const auto& coord = coords[n];
    scan += coord->dimension(0);
    input_scan.Set(n + 1, scan);
  }
  OP_REQUIRES_OK(c, input_scan.Finalize());

  TakeGradGPUImpl<T, Index>(c->eigen_gpu_device(), out_grad,
                            coord_ptrs.data(), input_scan.data(),
                            unit_size, scan, value_ptrs.data());
}

}  // end namespace

template <typename T, typename Index>
void TakeGradGPU(
    OpKernelContext* c,
    typename TTypes<T, 3>::ConstTensor& out_grad,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    std::vector<std::unique_ptr<typename TTypes<T, 2>::Matrix>>* value_grads) {
    TakeGradGPUCall<T, Index>(c, out_grad, coords, value_grads);
}

#define REGISTER(T, Index)                                                         \
  template void TakeGradGPU<T, Index>(                                             \
      OpKernelContext*,                                                            \
      typename TTypes<T, 3>::ConstTensor&,                                         \
      const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&, \
      std::vector<std::unique_ptr<typename TTypes<T, 2>::Matrix>>*);

#define REGISTER_ALL(type) \
  REGISTER(type, int32);   \
  REGISTER(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ALL)

#undef REGISTER_ALL
#undef REGISTER

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
