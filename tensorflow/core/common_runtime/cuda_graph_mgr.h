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
#include <unordered_set>
#include <vector>

#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef struct CudaGraphMeta {
  size_t graph_hash_;
  int batch_bucket_;
  std::string subgraph_name;
} CudaGraphMeta;

class CudaGraphMgr {
public:
  static CudaGraphMgr& Singleton();

  Status CaptureCudagraph(const GraphDef& graph_def, const std::string& graph_name, 
                          const int batch_size, const int num_instance);
  Status GetCudagraphExecInstance(const std::string& cudagrpah_name);

private:
  CudaGraphMgr(/* args */);
  ~CudaGraphMgr();

  // assistant functions for capturing
  void GenerateInputs(GraphDef& graph_def, const std::vector<string>& input_names,
                    std::vector<Tensor>& input_tensors, int batch_size);
  void FillInputsMap(InputsMap& inputs_map, std::vector<std::string>& input_names,
                   std::vector<Tensor>& input_tensors);
  void LogCudaGraphStatus(Session* sess);
  bool CheckGraphCaptured(const size_t graph_id) {
    return captured_graph_ids_.find(graph_id) == captured_graph_ids_.end();
  };

  TF_DISALLOW_COPY_AND_ASSIGN(CudaGraphMgr);

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

  std::unordered_set<size_t> captured_graph_ids_;
};

/* static */ CudaGraphMgr& CudaGraphMgr::Singleton() {
  static auto a = new CudaGraphMgr;
  return *a;
}

CudaGraphMgr::CudaGraphMgr() = default;
CudaGraphMgr::~CudaGraphMgr() = default;


}

#endif // TENSORFLOW_CORE_COMMON_RUNTIME_CUDAGRAPH_MGR_H_