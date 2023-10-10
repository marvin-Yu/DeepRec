//
//// Copyright (c) 2023 Alibaba Inc. All rights reserved.
////
//// Author: anker.wang
////

#include "tensorflow/core/util/time_stamp.h"

#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/env.h"
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>

namespace tensorflow {

// 1h = 1000 * 60 * 60, size = 16 * 3600000 = 55MB, dump to file cost ~50ms on skylake CPU

int64 GetRecorderPeriod() {
  const int64 kDefaultDenseThreadsNum = 60;
  int64 period;
  if (!tensorflow::ReadInt64FromEnvVar("TF_TRACE_QUERY_DISTRIBUTION_PERIOD",
                   kDefaultDenseThreadsNum, &period).ok()) {
    return kDefaultDenseThreadsNum;
  }
  VLOG(0) << "set query distribution period " << period;
  return (uint64_t)period;
}

QueryTimestampRecorder::QueryTimestampRecorder(const std::string scene)
    : scene_(scene) {
  buffer_id_ = 0;
  buffer_offset_ = 0;
  stop_ = false;
  max_buffer_size_ = GetRecorderPeriod()*1000*60;
  data_[0] = new struct TimeQuery[max_buffer_size_];
  data_[1] = new struct TimeQuery[max_buffer_size_];

  std::function<void(void)> HandleQueue = [this]() {
    pthread_setname_np(pthread_self(), "ts_recorder");
    LOG(INFO) << scene_ << " started recording query timestamps";
    while(!stop_) {
      if (!timestamp_queue_.empty()) {
        auto tm = timestamp_queue_.front();
        {
          std::lock_guard<std::mutex> guard(mu_);
          timestamp_queue_.pop();
        }
        struct TimeQuery& tq = data_[buffer_id_][buffer_offset_];
        tq.tv_sec = tm.tv_sec;
        tq.tv_usec = tm.tv_usec;
        buffer_offset_++;
        VLOG(1) << "pop queue, query length " << timestamp_queue_.size() << ", buffer_offset " << buffer_offset_;
        if (buffer_offset_ >= max_buffer_size_) {
          buffer_offset_ = 0;
          LOG(INFO) << "start to dump file";
          DumpToFile();
          LOG(INFO) << "end to dump file, queue length " << timestamp_queue_.size();
          buffer_id_ = buffer_id_ ^ 0x1;
        }
      } else {
        Env::Default()->SleepForMicroseconds(30);
      }
    }
  };
  thread_ = absl::make_unique<std::thread>(HandleQueue);
}

void QueryTimestampRecorder::Record() {
  struct timeval curTime;
  gettimeofday(&curTime, NULL);
  std::lock_guard<std::mutex> guard(mu_);
  timestamp_queue_.emplace(curTime);
}

std::string GetLocalTime(const long int sec) {
  char buffer[32] = {0};
  struct tm nowTime;
  localtime_r(&sec, &nowTime);
  strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", &nowTime);

  return buffer;
}

std::string GetLocalTime(const long int sec, const long int usec) {
  char currentTime[64] = {0};
  std::string date = GetLocalTime(sec);
  snprintf(currentTime, sizeof(currentTime), "%s-%ld", date.c_str(), usec);

  return currentTime;
}

void QueryTimestampRecorder::DumpToFile() {
  std::string start = GetLocalTime(data_[buffer_id_][0].tv_sec);
  std::string end = GetLocalTime(data_[buffer_id_][max_buffer_size_ - 1].tv_sec);
  std::string file_name = strings::StrCat("/tmp/timestamp_", scene_, "_", start, "_", end);
  std::fstream outFile(file_name.c_str(), std::fstream::in);
  if (outFile.good()) {
    LOG(INFO) << "file exist: " << file_name;
    outFile.close();
    return;
  }
  outFile.close();
  outFile.open(file_name.c_str(), std::fstream::out | std::fstream::binary);
  LOG(INFO) << "dump query timestamp into file " << file_name;
  outFile.write((char*)data_[buffer_id_], max_buffer_size_ * sizeof(struct TimeQuery));
  outFile.close();
}

QueryTimestampRecorder::~QueryTimestampRecorder() {
  delete [] data_[0];
  delete [] data_[1];
  stop_ = true;
}

QueryTimestampRecorder* TimeStampRecorderFactory::Register(const std::string &name) {
  std::lock_guard<std::mutex> guard(mu_);
  const auto &iter = recorder_map_.find(name);
  if (iter != recorder_map_.end()) {
    LOG(ERROR) << "QueryTimestampRecorder name=" << name << " has already been registered";
    return nullptr;
  }
  QueryTimestampRecorder* recorder = new QueryTimestampRecorder(name);
  recorder_map_[name] = recorder;
  return recorder;
}

TimeStampRecorderFactory* TimeStampRecorderFactory::Singleton() {
  static TimeStampRecorderFactory* instance = new TimeStampRecorderFactory;
  return instance;
}

QueryTimestampRecorder* TimeStampRecorderFactory::get(const std::string & name) {
  const auto &iter = recorder_map_.find(name);
  if (iter != recorder_map_.end()) {
    return iter->second;
  }
  return Register(name);
}

TimeStampRecorderFactory::~TimeStampRecorderFactory() {
  std::lock_guard<std::mutex> guard(mu_);
  for (auto iter : recorder_map_) {
    delete iter.second;
  }
}

}  // namespace tensorflow
