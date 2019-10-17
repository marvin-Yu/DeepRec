#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <memory>
#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// cannot be in anonymous namespace due to extern shared memory
template <typename T, typename Index, bool reverse>
__global__ void take_axis_kernel(
    const T* val,
    const Index* beg,
    int64 unit_size, int64 col_size, int64 row_size, int64 axis_size, 
    T* out) {
  int64 gidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; gidx < col_size; gidx += blockDim.x * gridDim.x) {
    int64 y = gidx;
    Index b = beg[y];
    int64 gidy = blockIdx.y * blockDim.y + threadIdx.y;
    for (; gidy < row_size*unit_size; gidy += blockDim.y * gridDim.y) {
      int64 x = gidy / unit_size;
      int64 z = gidy % unit_size;
      if (reverse) {
        if (x < b || x-b >= axis_size) {
          out[y*row_size*unit_size + x*unit_size + z] = (T)0;
        } else {
          out[y*row_size*unit_size + x*unit_size + z] = 
              val[y*axis_size*unit_size + (x-b)*unit_size + z];
        }
      } else {
        if (x+b >= axis_size) {
          out[y*row_size*unit_size + x*unit_size + z] = (T)0;
        } else {
          out[y*row_size*unit_size + x*unit_size + z] = 
              val[y*axis_size*unit_size + (x+b)*unit_size + z];
        }
      }
    }
  }
}


template <typename T, typename Index>
void TakeAxisGPUImpl(const Eigen::GpuDevice& gpu_device,
                      const typename TTypes<T, 3>::ConstTensor& input,
                      const typename TTypes<Index, 1>::ConstTensor& begin,
                      bool reverse,
                      typename TTypes<T, 3>::Tensor* output) {
  int64 unit_size = output->dimension(2);
  CHECK(input.dimension(2) == unit_size);

  int64 col_size = output->dimension(0);
  CHECK(input.dimension(0) == col_size);

  int64 row_size = output->dimension(1);
  int64 axis_size = input.dimension(1);

  if (reverse) {
    auto config = GetCuda2DLaunchConfig(col_size, row_size*unit_size, gpu_device);
    take_axis_kernel<T, Index, true><<<config.block_count, config.thread_per_block, 0,
        gpu_device.stream()>>>(input.data(), begin.data(),
                               unit_size, col_size, row_size, axis_size, 
                               output->data());
  } else {
    auto config = GetCuda2DLaunchConfig(col_size, row_size*unit_size, gpu_device);
    take_axis_kernel<T, Index, false><<<config.block_count, config.thread_per_block, 0,
        gpu_device.stream()>>>(input.data(), begin.data(),
                               unit_size, col_size, row_size, axis_size, 
                               output->data());
  }

}


#define REGISTER_GPU(T, Index)                       \
  template void TakeAxisGPUImpl<T, Index>(          \
      const Eigen::GpuDevice&,                       \
      const typename TTypes<T, 3>::ConstTensor&,     \
      const typename TTypes<Index, 1>::ConstTensor&, \
      bool reserve,                                  \
      typename TTypes<T, 3>::Tensor*);

#define REGISTER_GPU_ALL(T)  \
    REGISTER_GPU(T, int32);  \
    REGISTER_GPU(T, int64);


TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_ALL);

#undef REGISTER_GPU
#undef REGISTER_GPU_ALL

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
