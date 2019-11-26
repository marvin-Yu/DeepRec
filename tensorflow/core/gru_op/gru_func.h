#ifndef GRU_OP_H_
#define GRU_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {


const int gru_weights_per_thread = 8;
const int gru_threads_per_block = 96;
#define alignN(n, a) ((((n) + (a) - 1) / (a)) * (a))

template <typename Device, typename T>
struct GRUFunctor {
  void operator()(const Device& d, OpKernelContext* context,
                  int batch_size, int rounds, int elts,
                  T* y, const T* x,
                  const T* h2h, const T* i2h, const T* h2hBias, const T* i2hBias);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct GRUFunctor<Eigen::GpuDevice, T>{
  void operator()(const Eigen::GpuDevice& d, OpKernelContext* context,
                  int batch_size, int rounds, int elts,
                  T* y, const T* x,
                  const T* h2h, const T* i2h, const T* h2hBias, const T* i2hBias);
};
#endif

} //namespace

#endif //GRU_OP_H_
