#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/register_types.h"
#include "take_grad_lib_cpu.h"
#include "take_lib.h"

namespace tensorflow {

template <typename T, typename Index>
void TakeGradCPU(
    DeviceBase* d,
    typename TTypes<T, 3>::ConstTensor &out_grad,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    std::vector<std::unique_ptr<typename TTypes<T, 2>::Matrix>>* value_grads) {
    TakeGradCPUImpl<T, Index>(d, out_grad, coords, sizeof(T), value_grads);
}

#define REGISTER(T, Index)                                                         \
  template void TakeGradCPU<T, Index>(                                             \
      DeviceBase*,                                                                 \
      typename TTypes<T, 3>::ConstTensor&,                                         \
      const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&, \
      std::vector<std::unique_ptr<typename TTypes<T, 2>::Matrix>>* value_grads);

#define REGISTER_ALL(type) \
  REGISTER(type, int32);   \
  REGISTER(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_ALL)

#undef REGISTER_ALL
#undef REGISTER

}  // namespace tensorflow
