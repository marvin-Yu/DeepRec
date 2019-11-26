#ifndef TENSORFLOW_CORE_USER_OPS_TAKE_LIB_H_
#define TENSORFLOW_CORE_USER_OPS_TAKE_LIB_H_

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

// Assumes all inputs are nonempty
template <typename T, typename Index>
void TakeCPU(
    DeviceBase* d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        values,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    typename TTypes<T, 3>::Tensor* output);

#if GOOGLE_CUDA
template <typename T, typename Index>
void TakeGPU(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        values,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    typename TTypes<T, 3>::Tensor* output);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_USER_OPS_TAKE_LIB_H_
