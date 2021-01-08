#include "tensorflow/core/kernels/benchmark_helper.h"

#include <chrono>
#include <fstream>

namespace tensorflow {
void BenchmarkHelper::Start() {
  if (!is_running_) {
    std::lock_guard<std::mutex> l(mu_);
    reporter_thread_ = std::thread(ReportFunc, this);
    is_running_ = true;
  }
}

void BenchmarkHelper::Stop() {
  if (!stop_) {
    std::lock_guard<std::mutex> l(stop_mu_);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    stop_ = true;
  }
}

void BenchmarkHelper::RecordTM(float ts) {
  int idx = ((int)(ts)) % 10;
  idx = idx >= 10 ? 9 : idx;
  ++time_recorder_[idx].second;
}

void BenchmarkHelper::ReportFunc(BenchmarkHelper* helper) {
  static std::ofstream stream("/tmp/blaze_report.log");
  stream.clear();
  while(!helper->stop_) {
    stream << "blaze kernel qps: " << helper->counter_ << "\n";
    stream << "ts : ";
    for (auto& pair : helper->time_recorder_) {
      stream << "time:  " << pair.first << ", value: " << pair.second << "; ";
    }
    stream << "\n";
    helper->Clear();
    stream.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

}
