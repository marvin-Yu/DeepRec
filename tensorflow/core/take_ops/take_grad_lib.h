#ifndef TENSORFLOW_CORE_USER_OPS_TAKE_GRAD_LIB_H_
#define TENSORFLOW_CORE_USER_OPS_TAKE_GRAD_LIB_H_

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

// Assumes all inputs are nonempty
template <typename T, typename Index>
void TakeGradCPU(
    DeviceBase* d,
    typename TTypes<T, 3>::ConstTensor &out_grad,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    std::vector<std::unique_ptr<typename TTypes<T, 2>::Matrix>>* value_grads);

#if GOOGLE_CUDA
template <typename T, typename Index>
void TakeGradGPU(
    OpKernelContext* c,
    typename TTypes<T, 3>::ConstTensor &out_grad,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    std::vector<std::unique_ptr<typename TTypes<T, 2>::Matrix>>* value_grads);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_USER_OPS_TAKE_GRAD_LIB_H_
