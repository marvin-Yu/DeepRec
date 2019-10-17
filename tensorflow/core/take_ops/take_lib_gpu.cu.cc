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
void TakeGPUImpl(const Eigen::GpuDevice& d,
                 const GpuDeviceArrayStruct<const T*>& value_ptrs,
                 const GpuDeviceArrayStruct<const Index*>& coord_ptrs,
                 const GpuDeviceArrayStruct<Index>& input_scan,
                 int unit_size, Index total_size,
                 typename TTypes<T, 3>::Tensor* output);

namespace {

template <typename T, typename Index>
void TakeGPUCall(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        values,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    typename TTypes<T, 3>::Tensor* output) {
  size_t N = values.size();
  CHECK(coords.size() == N);
  GpuDeviceArrayOnHost<const T*> value_ptrs(c, N);
  GpuDeviceArrayOnHost<const Index*> coord_ptrs(c, N);
  OP_REQUIRES_OK(c, value_ptrs.Init());
  OP_REQUIRES_OK(c, coord_ptrs.Init());
  for (size_t n = 0; n < N; ++n) {
    value_ptrs.Set(n, values[n]->data());
    coord_ptrs.Set(n, coords[n]->data());
  }
  OP_REQUIRES_OK(c, value_ptrs.Finalize());
  OP_REQUIRES_OK(c, coord_ptrs.Finalize());

  GpuDeviceArrayOnHost<Index> input_scan(c, N + 1);
  OP_REQUIRES_OK(c, input_scan.Init());
  int64 unit_size = output->dimension(2);
  Index scan = 0;
  input_scan.Set(0, scan);
  for (size_t n = 0; n < N; ++n) {
    const auto& value = values[n];
    CHECK(value->dimension(1) == unit_size);
    scan += value->dimension(0);
    input_scan.Set(n + 1, scan);
  }
  OP_REQUIRES_OK(c, input_scan.Finalize());

  TakeGPUImpl<T, Index>(c->eigen_gpu_device(), value_ptrs.data(),
                        coord_ptrs.data(), input_scan.data(),
                        unit_size, scan, output);
}

}  // end namespace

template <typename T, typename Index>
void TakeGPU(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        values,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    typename TTypes<T, 3>::Tensor* output) {
    TakeGPUCall<T, Index>(c, values, coords, output);
}

#define REGISTER(T, Index)                                                         \
  template void TakeGPU<T, Index>(                                                 \
      OpKernelContext*,                                                            \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&,     \
      const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&, \
      typename TTypes<T, 3>::Tensor* output);

#define REGISTER_ALL(type) \
  REGISTER(type, int32);   \
  REGISTER(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ALL)

#undef REGISTER_ALL
#undef REGISTER

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
