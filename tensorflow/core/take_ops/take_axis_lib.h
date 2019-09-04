#ifndef TENSORFLOW_CORE_USER_OPS_TAKE_AXIS_LIB_H_
#define TENSORFLOW_CORE_USER_OPS_TAKE_AXIS_LIB_H_

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

// Assumes all inputs are nonempty
template <typename T, typename Index>
void TakeAxisCPU(
    DeviceBase* d,
    const typename TTypes<T, 3>::ConstTensor& input,
    const typename TTypes<Index, 1>::ConstTensor& begin,
    bool reserve,
    typename TTypes<T, 3>::Tensor* output);

#if GOOGLE_CUDA
template <typename T, typename Index>
void TakeAxisGPU(
    OpKernelContext* c,
    const typename TTypes<T, 3>::ConstTensor& input,
    const typename TTypes<Index, 1>::ConstTensor& begin,
    bool reserve,
    typename TTypes<T, 3>::Tensor* output);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_USER_OPS_TAKE_AXIS_LIB_H_
