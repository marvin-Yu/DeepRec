#ifndef TENSORFLOW_CORE_KERNELS_BENCHMARK_HELPER_H_
#define TENSORFLOW_CORE_KERNELS_BENCHMARK_HELPER_H_

#include <atomic>
#include <mutex>
#include <thread>

namespace tensorflow {
class BenchmarkHelper {
 public:
  static BenchmarkHelper& GetInstance() {
    static BenchmarkHelper instance;
    return instance;
  }

  void Add() { ++counter_; }

  void Start();

  void Stop();

 private:
  BenchmarkHelper() {
    counter_ = 0;
    is_running_ = false;
    stop_ = false;
  }

  void Clear() { counter_ = 0; }

  static void ReportFunc(BenchmarkHelper* helper);

 private:
  std::atomic<uint64_t> counter_;
  bool is_running_;
  std::mutex mu_;
  std::mutex stop_mu_;

  bool stop_;
  std::thread reporter_thread_;
};
}
#endif //end TENSORFLOW_CORE_KERNELS_BENCHMARK_HELPER_H_
