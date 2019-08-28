#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_PERSISTENT_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_PERSISTENT_ALLOCATOR_H_

#include <deque>
#include <map>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// An allocator that never releases its allocated memory, for use in
// conjunction with CUDA Graph.
class GPUPersistentAllocator : public Allocator {
 public:
  explicit GPUPersistentAllocator(const GPUOptions& options,
                                  PlatformGpuId platform_gpu_id);
  ~GPUPersistentAllocator() override;
  string Name() override { return "gpu_persistent"; }
  void Reset() override;
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void*) override { }
  bool TracksAllocationSizes() const override { return false; }

 private:
  const size_t large_chunk_size_;
  const size_t chunk_size_;
  const size_t alignment_ = 64;
  struct Chunk {
    char* ptr;
    size_t size;
  };
  std::map<size_t, std::deque<Chunk>> free_large_chunks_;
  std::map<size_t, std::deque<Chunk>> used_large_chunks_;
  std::vector<Chunk> chunks_;
  int index_ = 0;
  size_t offset_ = 0;
  size_t allocated_size_ = 0, allocated_large_size_ = 0;

  void LogAllocations();
  void* AllocateLarge(size_t size);
  void* AllocateNormal(size_t size);

  se::StreamExecutor* stream_exec_; // Not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(GPUPersistentAllocator);
};

} // namespace tensorflow

#endif // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_PERSISTENT_ALLOCATOR_H_
