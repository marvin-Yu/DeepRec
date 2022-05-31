/**
 *  Copyright 2022 Alibaba Inc. All rights reserved.
 * All rights reserved.
 *
 * 文件名称: local_metric.cc
 * 摘要:
 *
 * 作者: zhenyuan.lzy<zhenyuan.lzy@taobao.com>
 * 修改日期: 2022-05-17 02:58
 *
**/

#include "tensorflow/core/util/local_metric.h"

namespace tensorflow {
template<class MetricType>
typename LocalMetric<MetricType>::MetricGroupMap *LocalMetric<MetricType>::metric_group_map_ = NULL;
template<class MetricType>
volatile bool LocalMetric<MetricType>::running_ = false;
template<class MetricType>
mutex LocalMetric<MetricType>::mu_(LINKER_INITIALIZED);
template<class MetricType>
thread_local typename LocalMetric<MetricType>::MetricGroup *LocalMetric<MetricType>::metric_group_ = NULL;
template<class MetricType>
int LocalMetric<MetricType>::update_time_ = 60;
LocalRTMetric LocalRTMetric::metric_impl_;
template<>
thread_local LocalMetric<RTMetric>::MetricGroup *LocalMetric<RTMetric>::metric_group_ = NULL;
template<>
LocalMetric<RTMetric>::MetricGroupMap *LocalMetric<RTMetric>::metric_group_map_ = NULL;

void LocalRTMetric::Update(MetricGroupMap *metric_group_map) {
  auto it = metric_group_map->begin();
  while (it != metric_group_map->end()) {
    std::vector<RTMetric *> *metric_list = it->second;
    uint64_t rt_sum = 0;
    uint64_t count_sum = 0;
    for (int i = 0; i < metric_list->size(); ++i) {
        (*metric_list)[i]->Merge(&rt_sum, &count_sum);
    }
    std::cout << it->first << ", avg: " << ((count_sum == 0) ? 0 : rt_sum / count_sum) << ", qps: " << (count_sum / update_time_) <<std::endl;
    ++it;
  }
}

}  // namespace tensorflow
