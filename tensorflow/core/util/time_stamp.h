//
//// Copyright (c) 2023 Alibaba Inc. All rights reserved.
////
//// Author: anker.wang
////

#ifndef TENSORFLOW_CORE_UTIL_TIME_STAMP_H_
#define TENSORFLOW_CORE_UTIL_TIME_STAMP_H_

#include <queue>
#include <thread>
#include <mutex>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

inline bool IsTraceQueryDistribution(const std::string& scene) {
  static std::set<std::string> trace_scene_set = [] {
    std::string param = "";
    std::set<std::string> scene_set;
    if (!tensorflow::ReadStringFromEnvVar("TF_TRACE_QUERY_DISTRIBUTION_PARAM", "", &param).ok()) {
      return std::set<std::string>();
    }
    for (auto x : str_util::Split(param, ";")) {
      VLOG(0) << "scene set insert " << x;
      scene_set.insert(x);
    }
    return scene_set;
  }();
  const auto it = trace_scene_set.find(scene);
  if (it != trace_scene_set.end()) {
    return true;
  }
  return false;
}

class QueryTimestampRecorder {
 public:
  explicit QueryTimestampRecorder(const std::string scene);
  ~QueryTimestampRecorder();

  void Record();

  void HandleQueue();

  void DumpToFile();

 private:
  struct TimeQuery {
    long int tv_sec;
    long int tv_usec;
  };

  const std::string scene_;
  struct TimeQuery* data_[2];
  uint64_t buffer_id_;
  uint64_t buffer_offset_;
  uint64_t max_buffer_size_;
  bool stop_;
  std::mutex mu_;
  std::unique_ptr<std::thread> thread_;
  std::queue<struct timeval> timestamp_queue_;
};

class TimeStampRecorderFactory {
 public:
  static TimeStampRecorderFactory* Singleton();

  QueryTimestampRecorder* Register(const std::string& name);
  QueryTimestampRecorder* get(const std::string& name);

  TimeStampRecorderFactory() {}

  ~TimeStampRecorderFactory();

  TimeStampRecorderFactory(const TimeStampRecorderFactory &) = delete;

  TimeStampRecorderFactory &operator=(const TimeStampRecorderFactory &) = delete;

  std::unordered_map<std::string, QueryTimestampRecorder*> recorder_map_;
 private:
  std::mutex mu_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TIME_STAMP_H_
