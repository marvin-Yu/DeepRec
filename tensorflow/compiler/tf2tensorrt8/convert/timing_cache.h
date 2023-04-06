/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TIMING_CACHE_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TIMING_CACHE_H_
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include <unordered_map>

#include "absl/base/call_once.h"
#include "tensorflow/compiler/tf2tensorrt8/common/utils.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// A registry for holding serialized TensorRT autotuner timing caches.
class TimingCacheRegistry {
 public:
  TimingCacheRegistry() = default;
  ~TimingCacheRegistry() = default;

  using TimingCache = nvinfer1::ITimingCache;
  using TimingCachePtr = std::unique_ptr<TimingCache>;

  // Load timing cache from file.
  bool MaybeLoadFromFile(absl::string_view name);

  // Find a timing cache using the given name. The provided BuilderConfig is
  // used to deserialize the cache. If no timing cache is found, a new timing
  // cache is returned.
  TimingCachePtr GetCache(absl::string_view name,
                          nvinfer1::IBuilderConfig* builder_config);

  // Dump timing cache to file.
  void DumpToFile(absl::string_view name);

  // Serialize the cache to string and dump the string to file.
  void Update(absl::string_view name, TimingCache* cache);

 private:
  mutex mu_;
  std::vector<uint8_t> data_;
};

TimingCacheRegistry* GetTimingCacheRegistry();

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TIMING_CACHE_H_
