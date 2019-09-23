#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif  // GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_persistent_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

GPUPersistentAllocator::GPUPersistentAllocator(const CUDAGraphOptions& options,
                                               PlatformGpuId platform_gpu_id)
  : large_chunk_size_((options.large_chunk_size_mb() > 0
                       ? options.large_chunk_size_mb() : 8)
                      * 1024 * 1024),
    chunk_size_(large_chunk_size_) {
  stream_exec_ =
    GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
}

GPUPersistentAllocator::~GPUPersistentAllocator() {
#ifdef GOOGLE_CUDA
  for (auto& e: free_large_chunks_) {
    for (auto& c: e.second) {
      CUresult res = cuMemFree(reinterpret_cast<CUdeviceptr>(c.ptr));
      if (res != CUDA_SUCCESS) {
        LOG(ERROR) << "cuMemFree failed to free " << c.ptr;
      }
    }
  }
  for (auto& e: used_large_chunks_) {
    for (auto& c: e.second) {
      CUresult res = cuMemFree(reinterpret_cast<CUdeviceptr>(c.ptr));
      if (res != CUDA_SUCCESS) {
        LOG(ERROR) << "cuMemFree failed to free " << c.ptr;
      }
    }
  }
  for (auto& c: chunks_) {
    CUresult res = cuMemFree(reinterpret_cast<CUdeviceptr>(c.ptr));
    if (res != CUDA_SUCCESS) {
      LOG(ERROR) << "cuMemFree failed to free " << c.ptr;
    }
  }
#endif
}

void GPUPersistentAllocator::Reset() {
#ifdef GOOGLE_CUDA
  index_ = 0;
  offset_ = 0;
  for (auto it = used_large_chunks_.begin();
       it != used_large_chunks_.end();
       ++it) {
    auto size = it->first;
    auto& s = free_large_chunks_[size];
    for (auto& c: it->second) { s.push_back(std::move(c)); }
    used_large_chunks_.erase(it);
  }
#endif
}

void GPUPersistentAllocator::LogAllocations() {
  VLOG(2) << "Allocated chunk size=" << allocated_size_
          << ", large chunk size=" << allocated_large_size_;
}

void* GPUPersistentAllocator::AllocateLarge(size_t size) {
#ifdef GOOGLE_CUDA
  auto& s = free_large_chunks_[size];
  if (s.empty()) {
    se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
    CUdeviceptr rv = 0;
    CUresult res = cuMemAlloc(&rv, size);
    if (res != CUDA_SUCCESS) {
      LOG(ERROR) << "cuMemAlloc failed to allocate " << size;
      return nullptr;
    } else {
      allocated_large_size_ += size;
      LogAllocations();
      Chunk c;
      c.ptr = reinterpret_cast<char*>(rv);
      c.size = size;
      used_large_chunks_[size].push_back(std::move(c));
      return reinterpret_cast<void*>(rv);
    }
  } else {
    auto c = s.front();
    s.pop_front();
    auto ptr = c.ptr;
    used_large_chunks_[size].push_back(std::move(c));
    return reinterpret_cast<void*>(ptr);
  }
#else
  return nullptr;
#endif
}

void* GPUPersistentAllocator::AllocateNormal(size_t size) {
#ifdef GOOGLE_CUDA
  if (chunks_.empty() || chunk_size_ - offset_ < size) {
    if (index_ + 1 < chunks_.size()) {
      index_++;
      offset_ = 0;
    } else {
      se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
      CUdeviceptr rv = 0;
      CUresult res = cuMemAlloc(&rv, chunk_size_);
      if (res != CUDA_SUCCESS) {
        LOG(ERROR) << "cuMemAlloc failed to allocate " << size;
        return nullptr;
      } else {
        allocated_size_ += chunk_size_;
        LogAllocations();
        Chunk c;
        c.ptr = reinterpret_cast<char*>(rv);
        c.size = chunk_size_;
        if (chunks_.empty()) {
          chunks_.push_back(std::move(c));
        } else {
          chunks_.push_back(std::move(c));
          index_++;
          offset_ = 0;
        }
      }
    }
  }
  auto& c = chunks_[index_];
  auto res = c.ptr + offset_;
  offset_ += size;
  return reinterpret_cast<void*>(res);
#else
  return nullptr;
#endif
}

void* GPUPersistentAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
#ifdef GOOGLE_CUDA
  mutex_lock l(mu_);
  num_bytes = (num_bytes + alignment_ - 1) / alignment_ * alignment_;
  return (num_bytes >= large_chunk_size_
          ? AllocateLarge(num_bytes)
          : AllocateNormal(num_bytes));
#else
  return nullptr;
#endif
}

} // namespace tensorflow
