#include "gru_func.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;



template <typename T>
struct GRUFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, OpKernelContext* context,
                  int batch_size, int rounds, int elts,
                  T* y, const T* x,
                  const T* h2h, const T* i2h, const T* h2hBias, const T* i2hBias);
};

template <typename T>
void GRUFunctor<CPUDevice, T>::operator()(const CPUDevice& d, OpKernelContext* context,
                            int batch_size, int rounds, int elts,
                            T* y, const T* x,
                            const T* h2h, const T* i2h, const T* h2hBias, const T* i2hBias) {
  // empty impl
  *y = *x;
};

template struct GRUFunctor<CPUDevice, float>;

}//namespace

