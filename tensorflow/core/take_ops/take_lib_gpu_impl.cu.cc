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
template <typename T, typename Index, bool useSmem>
__global__ void take_kernel(
    GpuDeviceArrayStruct<const T*> value_ptr_data,
    GpuDeviceArrayStruct<const Index*> coord_ptr_data,
    GpuDeviceArrayStruct<Index> input_scan,
    int unit_size, Index total_size, Index col_size, Index row_size, T* output) {
  const T** value_ptrs = GetGpuDeviceArrayOnDevice(&value_ptr_data);
  const Index** coord_ptrs = GetGpuDeviceArrayOnDevice(&coord_ptr_data);
  Index* col_scan = GetGpuDeviceArrayOnDevice(&input_scan);

  // do upper_bound on col to find which pointer we should be using
  Index gid = blockIdx.x * blockDim.x + threadIdx.x;
  Index gidx = gid / unit_size;
  Index gidz = gid % unit_size;
  Index N = value_ptr_data.size;

  // verbose declaration needed due to template
  extern __shared__ __align__(8) unsigned char smem[];
  Index* smem_col_scan = reinterpret_cast<Index*>(smem);

  if (useSmem) {
    Index lidx = threadIdx.x;
    Index blockSize = blockDim.x;

    for (Index i = lidx; i < input_scan.size; i += blockSize) {
      smem_col_scan[i] = col_scan[i];
    }

    __syncthreads();

    col_scan = smem_col_scan;
  }

  if (gidx >= total_size) {
    return;
  }

  //printf("blockDim:(%d, %d) blockIdx:(%d, %d) threadIdx:(%d, %d)\n",
  //       blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

  // do an initial binary search and then scan linearly from there
  // works well when there are many small segments and when the
  // segments are much longer
  Index segment =
      cuda_helper::upper_bound<Index>(col_scan, N, gidx) - 1;

  Index curr_offset = col_scan[segment];
  Index curr_segment = segment;

  for (; gidx < total_size; gidx += blockDim.x * gridDim.x) {
    Index curr_col_offset;
    while ((curr_col_offset = col_scan[curr_segment + 1]) <= gidx) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    Index local_col = gidx - curr_offset;
    const T* val = value_ptrs[curr_segment];
    const Index* coo = coord_ptrs[curr_segment];
    Index y = coo[local_col*2];
    Index x = coo[local_col*2+1];
    output[y*row_size*unit_size + x*unit_size + gidz] = val[local_col*unit_size+gidz];
  }
}


template <typename T, typename Index>
void TakeGPUImpl(const Eigen::GpuDevice& gpu_device,
                 const GpuDeviceArrayStruct<const T*>& value_ptrs,
                 const GpuDeviceArrayStruct<const Index*>& coord_ptrs,
                 const GpuDeviceArrayStruct<Index>& input_scan,
                 int unit_size, Index total_size,
                 typename TTypes<T, 3>::Tensor* output) {
  Index N = value_ptrs.size;
  CHECK(input_scan.size == N + 1); 
  auto config = GetCudaLaunchConfig(total_size * unit_size, gpu_device);

  Index smem_max = gpu_device.sharedMemPerBlock();
  Index smem_usage = input_scan.size * sizeof(Index);
  // performance crossover is less than using maximum available shared memory
  // on most processors
  // possibly due to decreasing occupancy
  // 4096 inputs is a lot, most code will take the smem path
  const int32 kMaxSmemBytesPerformance = 16384;
  if (smem_usage < smem_max && smem_usage < kMaxSmemBytesPerformance) {
    take_kernel<T, Index, true><<<config.block_count, config.thread_per_block, smem_usage,
        gpu_device.stream()>>>(value_ptrs, coord_ptrs, input_scan,
                               unit_size, total_size, output->dimension(0),
                               output->dimension(1), output->data());
  } else {
    take_kernel<T, Index, false><<<config.block_count, config.thread_per_block, 0,
        gpu_device.stream()>>>(value_ptrs, coord_ptrs, input_scan,
                               unit_size, total_size, output->dimension(0),
                               output->dimension(1), output->data());
  }
}


#define REGISTER_GPU(T, Index)                     \
  template void TakeGPUImpl<T, Index>(             \
      const Eigen::GpuDevice&,                     \
      const GpuDeviceArrayStruct<const T*>&,      \
      const GpuDeviceArrayStruct<const Index*>&,  \
      const GpuDeviceArrayStruct<Index>&,         \
      int, Index,                                  \
      typename TTypes<T, 3>::Tensor*);

#define REGISTER_GPU_ALL(T)  \
    REGISTER_GPU(T, int32);  \
    REGISTER_GPU(T, int64);


TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_ALL);

#undef REGISTER_GPU
#undef REGISTER_GPU_ALL

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
