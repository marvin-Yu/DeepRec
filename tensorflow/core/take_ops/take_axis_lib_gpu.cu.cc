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
void TakeAxisGPUImpl(const Eigen::GpuDevice& d,
                      const typename TTypes<T, 3>::ConstTensor& input,
                      const typename TTypes<Index, 1>::ConstTensor& begin,
                      bool reserve,
                      typename TTypes<T, 3>::Tensor* output);

template <typename T, typename Index>
void TakeAxisGPU(
    OpKernelContext* c,
    const typename TTypes<T, 3>::ConstTensor& input,
    const typename TTypes<Index, 1>::ConstTensor& begin,
    bool reserve,
    typename TTypes<T, 3>::Tensor* output) {
    TakeAxisGPUImpl<T, Index>(c->eigen_gpu_device(), input, begin, reserve, output);
}

#define REGISTER(T, Index)                           \
  template void TakeAxisGPU<T, Index>(               \
      OpKernelContext*,                              \
      const typename TTypes<T, 3>::ConstTensor&,     \
      const typename TTypes<Index, 1>::ConstTensor&, \
      bool reserve,                                  \
      typename TTypes<T, 3>::Tensor* output);

#define REGISTER_ALL(type) \
  REGISTER(type, int32);   \
  REGISTER(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ALL)

#undef REGISTER_ALL
#undef REGISTER

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
