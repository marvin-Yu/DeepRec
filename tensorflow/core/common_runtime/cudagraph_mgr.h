// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/25
// Description:
// CudaGrpah Manager, singleton
// capture cudagraph and manage cudagraph instance, streams for launching cudagraph

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_CUDAGRAPH_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_CUDAGRAPH_MGR_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class CudagraphMgr {
public:
  static CudagraphMgr& Singleton();

  Status CaptureCudagraph(const GraphDef& graph_def, const int batch_size, const int num_instance);
  Status GetCudagraphExecInstance(const std::string& cudagrpah_name);

private:
  CudagraphMgr(/* args */);
  ~CudagraphMgr();

  TF_DISALLOW_COPY_AND_ASSIGN(CudagraphMgr);

private:
  // holds the tensors allocated during graph capturing
  // model_name --> tensor_holders
  // for each model, multiple graphs can be captured,
  // so we can run multiple graph instances in parallel
  // (to separate their memory, mutiple graphs are needed).
  std::map<std::string, std::vector<TensorHolder>> cuda_graph_gpu_tensors_;
  std::map<std::string, std::vector<cudaGraph_t>> cuda_graphs_;
  std::map<std::string, std::vector<cudaGraphExec_t>> cuda_graph_instances_;
  std::map<std::pair<string, int>, std::vector<std::pair<void*, void*>>> src_dst_mapping_;
};

/* static */ CudagraphMgr& CudagraphMgr::Singleton() {
  static auto a = new CudagraphMgr;
  return *a;
}

CudagraphMgr::CudagraphMgr() = default;
CudagraphMgr::~CudagraphMgr() = default;


}

#endif // TENSORFLOW_CORE_COMMON_RUNTIME_CUDAGRAPH_MGR_H_