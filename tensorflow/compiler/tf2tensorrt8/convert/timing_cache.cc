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
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt8/convert/timing_cache.h"

#include <unordered_map>

#include "absl/base/call_once.h"
#include "tensorflow/compiler/tf2tensorrt8/common/utils.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/env_var.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// Initialize the file path of timing cache via TF_TRT_TIMING_CACHE_DIR.
static string timing_cache_dir;
static void InitTimingCacheDir() {
  static absl::once_flag init_once;
  absl::call_once(init_once, [] {
    tensorflow::ReadStringFromEnvVar("TF_TRT_TIMING_CACHE_DIR", "",
                                     &timing_cache_dir);
    if (timing_cache_dir.empty()) {
      VLOG(0) << "TRT will not dump timing cache to file";
    } else {
      VLOG(0) << "TRT will dump timing cache to " << timing_cache_dir;
    }
  });
}

bool TimingCacheRegistry::MaybeLoadFromFile(absl::string_view name) {
  if (!data_.empty()) {
    return true;
  }
  auto env = tensorflow::Env::Default();
  string file_path = tensorflow::io::JoinPath(timing_cache_dir, name);
  if (env->FileExists(file_path).ok()) {
    VLOG(0) << "Will load timing cache from file: " << file_path;
    string timing_cache;
    Status status = (tensorflow::ReadFileToString(tensorflow::Env::Default(),
          file_path, &timing_cache));
    if (status.ok()) {
      data_.reserve(timing_cache.size());
      for (const char c: timing_cache) {
        data_.push_back(static_cast<uint8_t>(c));
      }
      if (data_.size() <= 10) {
        VLOG(2) << "Timing cache too short, size= " << data_.size();
        data_.clear();
        return false;
      }
      VLOG(0) << "Load timing cache success";
      return true;
    }
    VLOG(2) << "Read timing cache file error";
    return false;
  }
  VLOG(0) << "Timing cache file not exit: " << file_path;
  return false;
}

TimingCacheRegistry::TimingCachePtr TimingCacheRegistry::GetCache(
    absl::string_view name, nvinfer1::IBuilderConfig* builder_config) {
  if (builder_config == nullptr) {
    return nullptr;
  }

  if (!timing_cache_dir.empty()) {
    mutex_lock scoped_lock(mu_);
    // Load timing cache from file.
    if (MaybeLoadFromFile(name)) {
      return std::unique_ptr<nvinfer1::ITimingCache>(
          builder_config->createTimingCache(data_.data(), data_.size()));
    }
  }

  // If no such timing cache exists, create a new timing cache.
  VLOG(0) << "TRT will create a new timing cache";
  return std::unique_ptr<nvinfer1::ITimingCache>(
      builder_config->createTimingCache(nullptr, 0));
}

void TimingCacheRegistry::DumpToFile(absl::string_view name) {
  VLOG(0) << "Will dump timing cache " << name << " to " << timing_cache_dir;
  auto env = tensorflow::Env::Default();
  if (!env->IsDirectory(timing_cache_dir).ok()) {
    auto status = env->RecursivelyCreateDir(timing_cache_dir);
    if (!status.ok() && !env->IsDirectory(timing_cache_dir).ok()) {
      VLOG(2) << "Could not create directory " << timing_cache_dir
              << " for dump timing cache: " << status;
      return ;
    }
  }
  string file_path = tensorflow::io::JoinPath(timing_cache_dir, name);
  string timing_cache(data_.begin(), data_.end());
  auto status = tensorflow::WriteStringToFile(env, file_path, timing_cache);
  if (!status.ok()) {
    VLOG(2) << "Could not write timing cache to " << file_path
            << ": " << status;
  }
  VLOG(0) << "Dump timing cache " << name << " success";
  return ;
}

void TimingCacheRegistry::Update(absl::string_view name,
                                 TimingCache* cache) {
  if (timing_cache_dir.empty()) {
    return ;
  }
  nvinfer1::IHostMemory* memory = cache->serialize();
  if (memory == nullptr) {
    return;
  }

  {
    mutex_lock scoped_lock(mu_);
    // Update memory cache.
    data_.resize(memory->size());
    std::copy_n(static_cast<uint8_t*>(memory->data()), memory->size(),
                                      data_.begin());
    // Update file cache.
    DumpToFile(name);
  }
  memory->destroy();
}

TimingCacheRegistry* GetTimingCacheRegistry() {
  InitTimingCacheDir();
  static TimingCacheRegistry* registry = new TimingCacheRegistry();
  return registry;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
