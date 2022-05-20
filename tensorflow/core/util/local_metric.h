/**
 *  Copyright 2022 Alibaba Inc. All rights reserved.
 * All rights reserved.
 *
 * 文件名称: local_metric.h
 * 摘要:
 *
 * 作者: zhenyuan.lzy<zhenyuan.lzy@taobao.com>
 * 修改日期: 2022-05-13 16:38
 *
**/

#ifndef TENSORFLOW_CORE_UTIL_LOCAL_METRIC_H_
#define TENSORFLOW_CORE_UTIL_LOCAL_METRIC_H_

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>
#include <unistd.h>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/test_log.pb.h"

namespace tensorflow {
template<class MetricType>
class LocalMetric {
 public:
  typedef std::map<std::string, std::vector<MetricType*> *> MetricGroupMap;
  typedef std::map<std::string, MetricType*> MetricGroup;
  virtual ~LocalMetric() {
    running_ = false;
  }
  virtual void Update(MetricGroupMap *metric_group_map)= 0;

  static void Init(LocalMetric *local_metric) {
    if (!running_) {
      metric_group_map_ = new MetricGroupMap();
      running_ = true;
      pthread_t thread;
      pthread_create(&thread, NULL, Run, local_metric);
    }
  }

  static void *Run(void *data) {
    LocalMetric *local_metric = (LocalMetric*)data;
    while (running_) {
      {
        mutex_lock lock(mu_);
        local_metric->Update(metric_group_map_);
      }
      sleep(update_time_);
    }
    return NULL;
  }

  static MetricType *GetMetric(const char *metric_name, LocalMetric *local_metric) {
    MetricGroup *metric_group = metric_group_;
    if (metric_group == NULL) {
      metric_group = new MetricGroup();
      metric_group_ = metric_group;
    }
    auto it = metric_group->find(metric_name);
    if (it == metric_group->end()) {
      MetricType *metric = new MetricType();
      metric_group->insert(std::make_pair(metric_name, metric));
      {
        mutex_lock lock(mu_);
        if (metric_group_map_ == NULL) {
          Init(local_metric);
        }
        auto group_it = metric_group_map_->find(metric_name);
        if (group_it == metric_group_map_->end()) {
          std::vector<MetricType *> *metric_list = new std::vector<MetricType *>();
          metric_list->push_back(metric);
          metric_group_map_->insert(std::make_pair(metric_name, metric_list));
        } else {
          group_it->second->push_back(metric);
        }
      }
      return metric;
    }
    return it->second;
  }
 protected:
  static MetricGroupMap *metric_group_map_;
  static volatile bool running_;
  static mutex mu_;
  static int update_time_;
  thread_local static MetricGroup *metric_group_;
};

class RTMetric {
 public:
  void Update(uint32_t rt) {
    if (reset_) {
        data_ = 0;
        reset_ = false;
    }
    uint64_t old_data = data_;
    uint64_t rt_sum = (old_data >> 24) + rt;
    uint64_t count_cum = (old_data & 0xFFFFFF) + 1;
    uint64_t new_data = (rt_sum << 24) + count_cum;
    data_ = new_data;
  }
  void Merge(uint64_t *rt_sum, uint64_t *count_sum) {
    if (reset_) {
      return;
    }
    uint64_t old_data = data_;
    *rt_sum += (old_data >> 24);
    *count_sum += (old_data & 0xFFFFFF);
    reset_ = true;
  }
 private:
  uint64_t data_ = 0;
  bool reset_ = false;
};

class LocalRTMetric: LocalMetric<RTMetric> {
 public:
  void Update(MetricGroupMap *metric_group_map);
  static inline void UpdateRT(uint32_t rt, const char *metric_name) {
    RTMetric *metric = GetMetric(metric_name, &metric_impl_);
    metric->Update(rt);
  }
 private:
  static LocalRTMetric metric_impl_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_LOCAL_METRIC_H_
