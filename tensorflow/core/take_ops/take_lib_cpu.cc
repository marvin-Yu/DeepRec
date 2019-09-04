#define EIGEN_USE_THREADS

#include <vector>
#include "take_lib_cpu.h"
#include "tensorflow/core/framework/register_types.h"
#include "take_lib.h"

namespace tensorflow {

template <typename T, typename Index>
void TakeCPU(
    DeviceBase* d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        values,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    typename TTypes<T, 3>::Tensor* output) {
    TakeCPUImpl<T, Index>(d, values, coords, sizeof(T), output);
}

#define REGISTER(T, Index)                                                         \
  template void TakeCPU<T, Index>(                                                 \
      DeviceBase*,                                                                 \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&,     \
      const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&, \
      typename TTypes<T, 3>::Tensor* output);

#define REGISTER_ALL(type) \
  REGISTER(type, int32);   \
  REGISTER(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_ALL)

#undef REGISTER_ALL
#undef REGISTER

}  // namespace tensorflow
