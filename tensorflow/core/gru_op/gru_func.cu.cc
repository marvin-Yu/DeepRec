#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "gru_func.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

__global__ void GRUPadZeros(float* x, float* y, int* padded_iterations,
                            unsigned int* finished, const int round,
                            const int elts, const int hidden_num) {
  const int batch_idx = blockIdx.x;
  x += batch_idx * round * elts;
  y += batch_idx * round * hidden_num;
  padded_iterations += batch_idx;
  __shared__ int all_zero[1];
  int i;
  for (i = 0; i < round; i++) {
    if (threadIdx.x == 0) {
      all_zero[0] = 1;
    }
    __syncthreads();
    const float* p = x + i * elts;
    int k = threadIdx.x;
    while (k < elts) {
      if (p[k] != 0) {
        all_zero[0] = 0;
        break;
      }
      k += blockDim.x;
    }
    __syncthreads();
    if (!all_zero[0]) {
      break;
    }
    float* q = y + i * hidden_num;
    k = threadIdx.x;
    while (k < hidden_num) {
      q[k] = 0;
      k += blockDim.x;
    }
  }
  if (threadIdx.x == 0) {
    *padded_iterations = i;
  }
  for (i = threadIdx.x; i < round; i += blockDim.x) {
    finished[i] = 0;
  }
}

__global__ void GRUPadZeros(Eigen::half* x, Eigen::half* y,
                            int* padded_iterations, unsigned int* finished,
                            const int round, const int elts,
                            const int hidden_num) {
  const int batch_idx = blockIdx.x;
  const int elts1 = alignN(elts, 2) / 2;
  auto no_padding = elts == elts1 * 2;
  const int hidden_num1 = alignN(hidden_num, 2) / 2;
  auto x_ = reinterpret_cast<__half2*>(x) + batch_idx * round * elts1;
  auto y_ = reinterpret_cast<__half2*>(y) + batch_idx * round * hidden_num1;
  padded_iterations += batch_idx;
  const auto zero = __floats2half2_rn(0, 0);
  __shared__ int all_zero[1];
  int i;
  for (i = 0; i < round; i++) {
    if (threadIdx.x == 0) {
      all_zero[0] = 1;
    }
    __syncthreads();
    auto p = &x_[i * elts1];
    int k = threadIdx.x;
    while (k < elts1) {
      __half2 v = p[k];
      bool is_zero = (k < elts1 - 1 || no_padding ? __hbeq2(v, zero)
                                                  : __low2float(v) == 0.0);
      if (!is_zero) {
        all_zero[0] = 0;
        break;
      }
      k += blockDim.x;
    }
    __syncthreads();
    if (!all_zero[0]) {
      break;
    }
    p = &y_[i * hidden_num1];
    k = threadIdx.x;
    while (k < hidden_num1) {
      p[k] = zero;
      k += blockDim.x;
    }
  }
  if (threadIdx.x == 0) {
    *padded_iterations = i;
  }
  for (i = threadIdx.x; i < round; i += blockDim.x) {
    finished[i] = 0;
  }
}

__global__ void GRUPrepare(unsigned int* finished, const int round) {
  for (int i = 0; i < round; i++) {
    finished[i] = 0;
  }
}

__device__ void check_readiness(bool* ready, int iter, unsigned int* counter,
                                unsigned int count) {
  if (threadIdx.x == 0) {
    *ready = iter == 0 || atomicAdd(counter, 0) == count;
  }
  __syncthreads();
}

__device__ void finish(unsigned int* counter) {
  __syncthreads();
  __threadfence();
  if (threadIdx.x == 0) {
    atomicAdd(counter, 1);
  }
}

enum GRUType { H2H_R, H2H_Z, H2H_H, I2H_R, I2H_Z, I2H_H };
// 108, 1, 108, 84
__device__ int calc_offset(int hidden_num, int slot_per_block,
                           int slot_per_batch, int threads_per_slot) {
  int rem = hidden_num % slot_per_block;
  int k = blockIdx.x % slot_per_batch;
  int slots = (rem != 0 && k + 1 == slot_per_batch ? rem : slot_per_block);
  return (threadIdx.x >= slots * threads_per_slot
              ? -1
              : (k * slot_per_block + threadIdx.x / threads_per_slot));
}

__forceinline__ __device__ float sigmoidf(float x) {
  return 0.5 + 0.5 * tanhf(0.5 * x);
}

template <const int weights_per_thread>
__device__ void load_weights(float* weights, const float* all_weights,
                             const GRUType type, const int offset,
                             const int offset_idx, const int elts,
                             const int hidden_num, const int batch_idx) {
  auto k = (type % 3 * hidden_num + offset +
            offset_idx * weights_per_thread * 3 * hidden_num +
            batch_idx * elts * 3 * hidden_num);
  const auto max = (batch_idx + 1) * elts * hidden_num * 3;
#pragma unroll
  for (int i = 0; i < weights_per_thread; i++) {
    if (k < max) {
      weights[i] = all_weights[k];
      k += hidden_num * 3;
    }
  }
}

template <const int weights_per_thread>
__device__ void multiply(const float* weights, const float* inp, float* out,
                         const GRUType type, const int iter,
                         const int offset_idx, const int elts,
                         const float* init_h) {
  float res = 0;
  auto k = weights_per_thread * offset_idx;
  const float* p = type >= 3 || iter > 0 ? inp : init_h;
#pragma unroll
  for (int i = 0; i < weights_per_thread; i++) {
    if (k < elts) {
      float a = p == nullptr ? 0 : p[k];
      res += weights[i] * a;
      k++;
    }
  }
  out[threadIdx.x] = res;
}

__device__ void sum(const float* vals, float* out, const int offset_idx,
                    const int threads_per_slot) {
  if (offset_idx != 0) {
    return;
  }
  float res = 0;
  auto k = threadIdx.x;
  for (int i = 0; i < threads_per_slot; i++) {
    res += vals[k++];
  }
  out[threadIdx.x] = res;
}

__device__ void calc_final(const float* vals, float* out, GRUType type,
                           const int offset, const int offset_idx,
                           const int threads_per_slot, const float prev_h,
                           const float hbr, const float hbz, const float hbh,
                           const float ibr, const float ibz, const float ibh) {
  if (type != 0 || offset_idx != 0) {
    return;
  }
  auto k = threadIdx.x;
  auto r1 = vals[k];
  k += threads_per_slot;
  auto z1 = vals[k];
  k += threads_per_slot;
  auto h1 = vals[k];
  k += threads_per_slot;
  auto r0 = vals[k];
  k += threads_per_slot;
  auto z0 = vals[k];
  k += threads_per_slot;
  auto h0 = vals[k];
  auto r2 = sigmoidf(r0 + ibr + r1 + hbr);
  auto z2 = sigmoidf(z0 + ibz + z1 + hbz);
  auto h2 = tanh(h0 + ibh + r2 * (h1 + hbh));
  out[offset] = (1 - z2) * h2 + z2 * prev_h;
}

template <const int weights_per_thread>
__global__ void GRUKernel(const float* x, const float* h2h,
                          const float* h2h_bias, const float* i2h,
                          const float* i2h_bias, float* y,
                          unsigned int* finished, const int batch_size,
                          const int round, const int elts, const int hidden_num,
                          const int* padded_iterations, const float* init_h) {
  const int total_per_slot_0 = alignN(elts, gru_weights_per_thread);     // 112
  const int threads_per_slot_0 = total_per_slot_0 / weights_per_thread;  // 14
  const int total_per_slot = total_per_slot_0 * 6;                       // 672
  const int threads_per_slot = total_per_slot / weights_per_thread;      // 84
  const int slot_per_block = (alignN(threads_per_slot, gru_threads_per_block) /
                              threads_per_slot);  // 1
  const int slot_per_batch =
      alignN(hidden_num, slot_per_block) / slot_per_block;  // 108
  const int offset =
      calc_offset(hidden_num, slot_per_block, slot_per_batch, threads_per_slot);
  if (offset < 0) {
    return;
  }
  const int offset_idx = threadIdx.x % threads_per_slot_0;
  const GRUType type =
      static_cast<GRUType>(threadIdx.x % threads_per_slot / threads_per_slot_0);
  const int batch_idx = blockIdx.x / slot_per_batch;
  x += batch_idx * round * elts;
  y += batch_idx * round * hidden_num;
  if (init_h) {
    init_h += batch_idx * hidden_num;
  }
  float weights[weights_per_thread];
  auto all_weights = type < 3 ? h2h : i2h;
  load_weights<weights_per_thread>(weights, all_weights, type, offset,
                                   offset_idx, elts, hidden_num, batch_idx);
  const float hbr = h2h_bias[offset + batch_idx * 3 * hidden_num];
  const float hbz = h2h_bias[offset + hidden_num + batch_idx * 3 * hidden_num];
  const float hbh =
      h2h_bias[offset + hidden_num * 2 + batch_idx * 3 * hidden_num];
  const float ibr = i2h_bias[offset + batch_idx * 3 * hidden_num];
  const float ibz = i2h_bias[offset + hidden_num + batch_idx * 3 * hidden_num];
  const float ibh =
      i2h_bias[offset + hidden_num * 2 + batch_idx * 3 * hidden_num];
  const int padded_iteration = padded_iterations[batch_idx];
  extern __shared__ float vals[];
  __shared__ bool ready[1];
  for (int iter = 0; iter < round;) {
    check_readiness(ready, iter, &finished[iter - 1], gridDim.x);
    if (!ready[0]) {
      continue;
    }
    if (iter >= padded_iteration) {
      auto inp = (type < 3 ? y + (iter - 1) * hidden_num : x + iter * elts);
      multiply<weights_per_thread>(weights, inp, vals, type, iter, offset_idx,
                                   elts, init_h);
      __syncthreads();
      sum(vals, vals, offset_idx, threads_per_slot_0);
      __syncthreads();
      float prev_h;
      if (offset_idx == 0 && type == 0) {
        prev_h = (iter == 0 ? (init_h == nullptr ? 0 : init_h[offset])
                            : y[(iter - 1) * hidden_num + offset]);
      }
      calc_final(vals, &y[iter * hidden_num], type, offset, offset_idx,
                 threads_per_slot_0, prev_h, hbr, hbz, hbh, ibr, ibz, ibh);
    }
    finish(&finished[iter]);
    iter++;
  }
}

__forceinline__ __device__ __half __sigmoidh(__half x) {
  return __float2half(sigmoidf(__half2float(x)));
}

__forceinline__ __device__ __half __tanhh(__half x) {
  return __float2half(tanh(__half2float(x)));
}

template <const int weights_per_thread>
__device__ void load_weights(__half2* weights, const __half* all_weights,
                             const GRUType type, const int offset,
                             const int offset_idx, const int elts,
                             const int hidden_num, const int batch_idx) {
  const auto base = alignN(hidden_num, 2);
  auto k =
      (type % 3 * base + offset + offset_idx * weights_per_thread * 3 * base +
       batch_idx * elts * 3 * base);
  const auto max = (batch_idx + 1) * elts * hidden_num * 3;
  bool partial;
  __half weight[2];
  int i;
#pragma unroll
  for (i = 0; i < weights_per_thread; i++) {
    if (k >= max) {
      break;
    }
    weight[i % 2] = all_weights[k];
    k += hidden_num * 3;
    if (i % 2) {
      weights[i / 2] = *reinterpret_cast<__half2*>(weight);
    }
    partial = !(i % 2);
  }
  if (partial) {
    weights[i / 2] = *reinterpret_cast<__half2*>(weight);
  }
}

template <const int weights_per_thread>
__device__ void multiply(const __half2* weights, const __half* inp, __half* out,
                         const GRUType type, const int iter,
                         const int offset_idx, const int elts,
                         const __half* init_h) {
  __half2 res = __floats2half2_rn(0, 0);
  auto k = weights_per_thread * offset_idx;
  auto p = reinterpret_cast<const __half2*>(type < 3 && iter == 0 ? &init_h[k]
                                                                  : &inp[k]);
  if (type >= 3 || iter > 0 || init_h != nullptr) {
#pragma unroll
    for (int i = 0; i < weights_per_thread / 2; i++) {
      if (k < elts - 1) {
        __half2 v = *p++;
        res = __hfma2(weights[i], v, res);
        k += 2;
      } else if (k < elts) {
        __half v0 = *reinterpret_cast<const __half*>(p);
        __half2 v = __halves2half2(v0, __float2half(0));
        res = __hfma2(weights[i], v, res);
        break;
      }
    }
  }
  out[threadIdx.x] = __hadd(__low2half(res), __high2half(res));
}

__device__ void sum(const __half* vals, __half* out, const int offset_idx,
                    const int threads_per_slot) {
  if (offset_idx != 0) {
    return;
  }
  __half2 res = __floats2half2_rn(0, 0);
  auto p = reinterpret_cast<const __half2*>(&vals[threadIdx.x]);
  for (int i = 0; i < threads_per_slot; i += 2) {
    if (i < threads_per_slot - 1) {
      __half2 v = *(p++);
      res = __hadd2(res, v);
    } else {
      __half v0 = *reinterpret_cast<const __half*>(p);
      __half2 v = __halves2half2(v0, __float2half(0));
      res = __hadd2(res, v);
    }
  }
  out[threadIdx.x] = __hadd(__low2half(res), __high2half(res));
}

__device__ void calc_final(const __half* vals, __half* out, GRUType type,
                           const int offset, const int offset_idx,
                           const int threads_per_slot, const __half prev_h,
                           const __half hbr, const __half hbz, const __half hbh,
                           const __half ibr, const __half ibz,
                           const __half ibh) {
  if (type != 0 || offset_idx != 0) {
    return;
  }
  auto k = threadIdx.x;
  auto r1 = vals[k];
  k += threads_per_slot;
  auto z1 = vals[k];
  k += threads_per_slot;
  auto h1 = vals[k];
  k += threads_per_slot;
  auto r0 = vals[k];
  k += threads_per_slot;
  auto z0 = vals[k];
  k += threads_per_slot;
  auto h0 = vals[k];
  auto r2 = __sigmoidh(__hadd(r0, __hadd(ibr, __hadd(r1, hbr))));
  auto z2 = __sigmoidh(__hadd(z0, __hadd(ibz, __hadd(z1, hbz))));
  auto h2 = __tanhh(__hadd(h0, __hadd(ibh, __hmul(r2, __hadd(h1, hbh)))));
  out[offset] =
      __hadd(__hmul(__hsub(__float2half(1), z2), h2), __hmul(z2, prev_h));
}

template <const int weights_per_thread>
__global__ void GRUKernel(const Eigen::half* x, const Eigen::half* h2h,
                          const Eigen::half* h2h_bias, const Eigen::half* i2h,
                          const Eigen::half* i2h_bias, Eigen::half* y,
                          unsigned int* finished, const int batch_size,
                          const int round, const int elts, const int hidden_num,
                          const int* padded_iterations,
                          const Eigen::half* init_h) {
  const int total_per_slot_0 = alignN(elts, gru_weights_per_thread);
  const int threads_per_slot_0 = total_per_slot_0 / weights_per_thread;
  const int total_per_slot = total_per_slot_0 * 6;
  const int threads_per_slot = total_per_slot / weights_per_thread;
  const int slot_per_block =
      (alignN(threads_per_slot, gru_threads_per_block) / threads_per_slot);
  const int slot_per_batch =
      alignN(hidden_num, slot_per_block) / slot_per_block;
  const int offset =
      calc_offset(hidden_num, slot_per_block, slot_per_batch, threads_per_slot);
  if (offset < 0) {
    return;
  }
  const int offset_idx = threadIdx.x % threads_per_slot_0;
  const GRUType type =
      static_cast<GRUType>(threadIdx.x % threads_per_slot / threads_per_slot_0);
  const int batch_idx = blockIdx.x / slot_per_batch;
  const __half* x_ =
      reinterpret_cast<const __half*>(x + batch_idx * round * alignN(elts, 2));
  __half* y_ =
      reinterpret_cast<__half*>(y + batch_idx * round * alignN(hidden_num, 2));
  const __half* init_h_ =
      (init_h == nullptr ? nullptr
                         : reinterpret_cast<const __half*>(
                               init_h + batch_idx * alignN(hidden_num, 2)));
  const __half* h2h_ = reinterpret_cast<const __half*>(h2h);
  const __half* i2h_ = reinterpret_cast<const __half*>(i2h);
  const __half* h2h_bias_ = reinterpret_cast<const __half*>(h2h_bias);
  const __half* i2h_bias_ = reinterpret_cast<const __half*>(i2h_bias);
  __half2 weights[weights_per_thread / 2];
  auto all_weights = type < 3 ? h2h_ : i2h_;
  load_weights<weights_per_thread>(weights, all_weights, type, offset,
                                   offset_idx, elts, hidden_num, batch_idx);
  const int hidden_num1 = alignN(hidden_num, 2);
  const __half hbr = h2h_bias_[offset + batch_idx * 3 * hidden_num];
  const __half hbz =
      h2h_bias_[offset + hidden_num1 + batch_idx * 3 * hidden_num];
  const __half hbh =
      h2h_bias_[offset + hidden_num1 * 2 + batch_idx * 3 * hidden_num];
  const __half ibr = i2h_bias_[offset + batch_idx * 3 * hidden_num];
  const __half ibz =
      i2h_bias_[offset + hidden_num1 + batch_idx * 3 * hidden_num];
  const __half ibh =
      i2h_bias_[offset + hidden_num1 * 2 + batch_idx * 3 * hidden_num];
  const int padded_iteration = padded_iterations[batch_idx];
  extern __shared__ __half half_vals[];
  __shared__ bool ready[1];
  for (int iter = 0; iter < round;) {
    check_readiness(ready, iter, &finished[iter - 1], gridDim.x);
    if (!ready[0]) {
      continue;
    }
    if (iter >= padded_iteration) {
      auto inp = (type < 3 ? y_ + (iter - 1) * alignN(hidden_num, 2)
                           : x_ + iter * alignN(elts, 2));
      multiply<weights_per_thread>(weights, inp, half_vals, type, iter,
                                   offset_idx, elts, init_h_);
      __syncthreads();
      sum(half_vals, half_vals, offset_idx, threads_per_slot_0);
      __syncthreads();
      __half prev_h;
      if (offset_idx == 0 && type == 0) {
        prev_h = (iter == 0
                      ? (init_h_ == nullptr ? __float2half(0) : init_h_[offset])
                      : y_[(iter - 1) * alignN(hidden_num, 2) + offset]);
      }
      calc_final(half_vals, &y_[iter * alignN(hidden_num, 2)], type, offset,
                 offset_idx, threads_per_slot_0, prev_h, hbr, hbz, hbh, ibr,
                 ibz, ibh);
    }
    finish(&finished[iter]);
    iter++;
  }
}

inline int GetThreadsNum(int data_size, bool upper = false) {
  if (upper) {
    // Return threadnum >= data_size if data_size <= 512
    if (data_size < 16)
      return 16;
    else if (data_size < 32)
      return 32;
    else if (data_size < 64)
      return 64;
    else if (data_size < 128)
      return 128;
    else if (data_size < 256)
      return 256;
    else
      return 512;
  } else {
    // Rerurn theadnum <= data_size
    if (data_size >= 512)
      return 512;
    else if (data_size >= 256)
      return 256;
    else if (data_size >= 128)
      return 128;
    else if (data_size >= 64)
      return 64;
    else if (data_size >= 32)
      return 32;
    else if (data_size >= 16)
      return 16;
    else
      return data_size;
  }
}

template <typename T>
void GRUFunctor<Eigen::GpuDevice, T>::operator()(
    const Eigen::GpuDevice& d, OpKernelContext* context, int batch_size,
    int rounds, int elts, T* y, const T* x, const T* h2h, const T* i2h,
    const T* h2hBias, const T* i2hBias) {
  VLOG(2) << "== GPU GRUFunctor ===";

  Tensor finished;  //[batch_size]
  OP_REQUIRES_OK(context, context->allocate_temp(
                              DT_UINT32, TensorShape({rounds}), &finished));
  unsigned int* finished_p = finished.flat<unsigned int>().data();
  Tensor padded_iterations;  // [batch_size]
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DT_INT32, TensorShape({batch_size}),
                                        &padded_iterations));
  int* padded_iterations_p = padded_iterations.flat<int>().data();
  T* init_h = nullptr;

  const int hidden_num = elts;                                        // 108
  const int total_per_slot_0 = alignN(elts, gru_weights_per_thread);  // 112
  // 3 for input and 3 for hidden, so 6 in total
  const int total_per_slot = total_per_slot_0 * 6;  // 112 * 6 = 672
  const int threads_per_slot = total_per_slot / gru_weights_per_thread;  // 84
  const int slot_per_block = (alignN(threads_per_slot, gru_threads_per_block) /
                              threads_per_slot);  // 1
  const int block_count =
      (alignN(elts, slot_per_block) / slot_per_block * batch_size);  // 108
  const size_t cache_size =
      sizeof(T) * slot_per_block * threads_per_slot;  // 84 * 4 = 336

   GRUPadZeros<<<batch_size, GetThreadsNum(elts*sizeof(T)/sizeof(float),
   true), 0, d.stream()>>>(
      const_cast<T*>(x), y, padded_iterations_p, finished_p, rounds, elts, hidden_num);
//  TF_CHECK_OK(GpuLaunchKernel(
//      GRUPadZeros, batch_size,
//      GetThreadsNum(elts * sizeof(T) / sizeof(float), true), 0, d.stream(), const_cast<T*>(x),
//      y, padded_iterations_p, finished_p, rounds, elts, hidden_num));


   GRUKernel<gru_weights_per_thread><<<block_count, gru_threads_per_block,
   cache_size, d.stream()>>>(
      x, h2h, h2hBias, i2h, i2hBias,
      y, finished_p, batch_size, rounds, elts,hidden_num, padded_iterations_p, init_h);
//  TF_CHECK_OK(GpuLaunchKernel(
//      GRUKernel<T,gru_weights_per_thread>, block_count, gru_threads_per_block,
//      cache_size, d.stream(), x, h2h, h2hBias, i2h, i2hBias, y, finished_p,
//      batch_size, rounds, elts, hidden_num, padded_iterations_p, init_h));
}

template struct GRUFunctor<Eigen::GpuDevice, float>;
template struct GRUFunctor<Eigen::GpuDevice, Eigen::half>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
