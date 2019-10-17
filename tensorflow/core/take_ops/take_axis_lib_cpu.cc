#define EIGEN_USE_THREADS

#include <vector>
#include "take_axis_lib_cpu.h"
#include "tensorflow/core/framework/register_types.h"
#include "take_axis_lib.h"

namespace tensorflow {

template <typename T, typename Index>
void TakeAxisCPU(
    DeviceBase* d,
    const typename TTypes<T, 3>::ConstTensor& input,
    const typename TTypes<Index, 1>::ConstTensor& begin,
    bool reserve,
    typename TTypes<T, 3>::Tensor* output) {
    TakeAxisCPUImpl<T, Index>(d, input, begin, sizeof(T), reserve, output);
}

#define REGISTER(T, Index)                                 \
  template void TakeAxisCPU<T, Index>(                     \
      DeviceBase*,                                         \
      const typename TTypes<T, 3>::ConstTensor& input,     \
      const typename TTypes<Index, 1>::ConstTensor& begin, \
      bool reserve,                                        \
      typename TTypes<T, 3>::Tensor* output);

#define REGISTER_ALL(type) \
  REGISTER(type, int32);   \
  REGISTER(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_ALL)

#undef REGISTER_ALL
#undef REGISTER

}  // namespace tensorflow
