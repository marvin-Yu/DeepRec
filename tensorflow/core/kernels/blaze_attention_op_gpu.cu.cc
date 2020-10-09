//
// Created by luoxinchen on 2020-10-01
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#define EIGEN_USE_GPU
#include "tensorflow/core/kernels/blaze_attention_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace {

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;  // in-warp idx
  int wid = threadIdx.x >> 5;     // warp idx

  val = warpReduceMax(val);  // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

template <bool use_indicator, typename Scalar, typename TIndex, int UNITS = 32>
__global__ void ComputeBlazeAttentionV2(
    const Scalar* in_fact, const Scalar* in_query, const TIndex* in_indicators,
    Scalar* output, int batch_fact, int batch_query, int seq_len, int pnum) {
  int ind = 0;
  if (use_indicator) {
    ind = (int)in_indicators[blockIdx.x];
    if (ind < 0 || ind >= batch_fact) {
      ind = 0;
    }
  }

  __shared__ float s_buf[1024];
  __shared__ float s_fact[32][33];
  const Scalar* fact = in_fact + blockIdx.y * batch_fact * seq_len * UNITS +
                       ind * seq_len * UNITS;
  const Scalar* query =
      in_query + blockIdx.y * batch_query * UNITS + blockIdx.x * UNITS;
  int block = (seq_len + 31) / 32;
  for (int b = 0; b < block; b++) {
    int i = threadIdx.x;
    for (int iq = 0; iq < 32; iq++) {
      int q = b * 32 + iq;
      s_fact[iq][i] = q < seq_len ? (float)fact[q * UNITS + i] : 0.0f;
    }
    __syncthreads();
    float sum = 0.0f;
    int q = b * 32 + threadIdx.x;
#pragma unroll
    for (int i = 0; i < UNITS; i++) {
      sum += s_fact[threadIdx.x][i] * (float)query[i];
    }
    s_buf[q] = sum;
  }
  __syncthreads();

  __shared__ float s_max, s_sum;
  float t_max = -1e20f;
  for (int q = threadIdx.x; q < seq_len; q += blockDim.x) {
    t_max = max(t_max, s_buf[q]);
  }
  float tt_max = warpReduceMax<float>(t_max);
  if (threadIdx.x == 0) {
    s_max = tt_max;
  }
  __syncthreads();
  for (int q = threadIdx.x; q < seq_len; q += blockDim.x) {
    s_buf[q] = __expf(s_buf[q] - s_max);
  }
  float t_sum = 0.0f;
  for (int q = threadIdx.x; q < seq_len; q += blockDim.x) {
    t_sum += s_buf[q];
  }
  float sum_val = warpReduceSum<float>(t_sum);
  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();
  for (int q = threadIdx.x; q < seq_len; q += blockDim.x) {
    s_buf[q] = s_buf[q] / s_sum;
  }
  __syncthreads();
  Scalar* out = output + (blockIdx.x * pnum + blockIdx.y) * UNITS;
  for (int i = threadIdx.x; i < UNITS; i += blockDim.x) {
    float sum = 0.0f;
    for (int q = 0; q < seq_len; q++) {
      sum += (float)fact[q * UNITS + i] * s_buf[q];
    }
    out[i] = (Scalar)sum;
  }
}
}  // namespace

template <typename Scalar>
Status LaunchBlazeAttention<GPUDevice, Scalar>::operator()(
    OpKernelContext* context, const Tensor& in_fact, const Tensor& in_query,
    Tensor* out, int pnum, int batch_fact, int batch_query, int seq_len,
    int units) {
  if (seq_len > 1024) {
    return errors::InvalidArgument(
        "this implementation requires seq_len <= 1024: ", seq_len);
  }
  if (units != 32) {
    return errors::InvalidArgument("this implementation requires units == 32: ",
                                   units);
  }
  dim3 grid_dim(batch_query, pnum);
  dim3 block_dim(32);
  size_t shared_memory_size = (1024 + 32 * 33 + 2) * sizeof(float);
  const auto& d = context->eigen_device<GPUDevice>();
  TF_CHECK_OK(GpuLaunchKernel(
      ComputeBlazeAttentionV2<false, Scalar, int32, 32>, grid_dim, block_dim,
      shared_memory_size, d.stream(),
      reinterpret_cast<const Scalar*>(in_fact.template flat<Scalar>().data()),
      reinterpret_cast<const Scalar*>(in_query.template flat<Scalar>().data()),
      nullptr, const_cast<Scalar*>(out->template flat<Scalar>().data()),
      batch_fact, batch_query, seq_len, pnum));
  return Status::OK();
}

template <typename Scalar, typename TIndex>
Status LaunchBlazeAttentionIndicator<GPUDevice, Scalar, TIndex>::operator()(
    OpKernelContext* context, const Tensor& in_fact, const Tensor& in_query,
    const Tensor& in_indicators, Tensor* out, int pnum, int batch_fact,
    int batch_query, int seq_len, int units) {
  if (seq_len > 1024) {
    return errors::InvalidArgument(
        "this implementation requires seq_len <= 1024: ", seq_len);
  }
  if (units != 32) {
    return errors::InvalidArgument("this implementation requires units == 32: ",
                                   units);
  }
  dim3 grid_dim(batch_query, pnum);
  dim3 block_dim(32);
  int shared_memory_size = (1024 + 32 * 33 + 2) * sizeof(float);
  const auto& d = context->eigen_device<GPUDevice>();
  TF_CHECK_OK(GpuLaunchKernel(
      ComputeBlazeAttentionV2<true, Scalar, TIndex, 32>, grid_dim, block_dim,
      shared_memory_size, d.stream(),
      reinterpret_cast<const Scalar*>(in_fact.template flat<Scalar>().data()),
      reinterpret_cast<const Scalar*>(in_query.template flat<Scalar>().data()),
      reinterpret_cast<const TIndex*>(
          in_indicators.template flat<TIndex>().data()),
      const_cast<Scalar*>(out->template flat<Scalar>().data()), batch_fact,
      batch_query, seq_len, pnum));
  return Status::OK();
}

template struct LaunchBlazeAttention<GPUDevice, Eigen::half>;
template struct LaunchBlazeAttention<GPUDevice, float>;
template struct LaunchBlazeAttentionIndicator<GPUDevice, float, int32>;
template struct LaunchBlazeAttentionIndicator<GPUDevice, float, int64>;
template struct LaunchBlazeAttentionIndicator<GPUDevice, Eigen::half, int32>;
template struct LaunchBlazeAttentionIndicator<GPUDevice, Eigen::half, int64>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA