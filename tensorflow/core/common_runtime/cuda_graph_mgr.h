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

#include "tensorflow/core/common_runtime/cuda_graph_meta.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

typedef std::vector<std::pair<std::string, Tensor>> InputsMap;
typedef std::unordered_map<int, std::vector<CudaGraphMeta*>> BatchGraphMetaMap;
typedef std::unordered_map<int, std::pair<std::mutex*, std::condition_variable*>> BatchMetaLockMap;

class CudaGraphMgr {
public:
  static CudaGraphMgr& Singleton();

  Status CaptureCudagraph(const GraphDef& graph_def, 
                          const std::string& graph_name, 
                          const std::vector<std::string>& input_node_names,
                          const std::vector<std::string>& output_node_names,
                          const std::vector<int>& batch_size);
  Status GetCudagraphMeta(const std::string& cudagrpah_name, 
                          const int bucket,
                          CudaGraphMeta*& meta);
  Status ReturnCudaGraphMeta(CudaGraphMeta* meta);
  void DestoryCudagraphMeta();
  Status GetCudaStream(int req_id, cudaStream_t& stream);

private:
  CudaGraphMgr(/* args */) { Init(); };
  ~CudaGraphMgr();

  void Init();
  void InitTraffic();

  // assistant functions for capturing
  void GenerateInputs(const GraphDef& graph_def, const std::vector<string>& input_names,
                    std::vector<Tensor>& input_tensors, int batch_size);
  void FillInputsMap(InputsMap& inputs_map, const std::vector<std::string>& input_names,
                   std::vector<Tensor>& input_tensors);
  void CheckCudaGraphScore(const GraphDef& graph_def,
                          CudaGraphMeta* meta,
                          const std::vector<std::string>& input_node_names,
                          const std::vector<std::string>& output_node_names);
  static void LaunchGraphInMeta(CudaGraphMeta* meta, cudaStream_t* stream);
public:
  void PrintTensorData(const Tensor &t);

public:
  // Check given graph names are captured already
  // If all captured return true, else reture false and record all uncaptured name index in graph_names 
  // to uncaptured_index.
  bool CheckGraphAllCaptured(const std::vector<std::string>& graph_names, std::vector<int>& uncaptured_index);

  TF_DISALLOW_COPY_AND_ASSIGN(CudaGraphMgr);

private:
  // holds the tensors allocated during graph capturing
  // model_name --> tensor_holders
  // for each model, multiple graphs can be captured,
  // so we can run multiple graph instances in parallel
  // (to separate their memory, mutiple graphs are needed).
  std::unordered_map<std::string, BatchGraphMetaMap> graphname_batch_metas_map_;
  std::unordered_map<std::string, BatchMetaLockMap> meta_pool_lock_;
  std::vector<cudaStream_t> streams_;
  // stream and cuda graph instance count, each instance corresponds to one stream
  int num_instance_; 
  Allocator* host_allocator_;
};

/* static */ CudaGraphMgr& CudaGraphMgr::Singleton() {
  static auto a = new CudaGraphMgr;
  return *a;
}

CudaGraphMgr::~CudaGraphMgr() = default;
}

#endif // TENSORFLOW_CORE_COMMON_RUNTIME_CUDAGRAPH_MGR_H_
