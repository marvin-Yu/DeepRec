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
#ifdef GOOGLE_CUDA

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

class CudaGraphMgr {
public:
  static CudaGraphMgr& Singleton();

  Status CaptureCudagraph(const GraphDef& graph_def, 
                          const std::string& group_name,
                          const std::string& origin_graph_name, 
                          const std::vector<std::string>& input_node_names,
                          const std::vector<std::string>& output_node_names,
                          const std::vector<int>& batch_size);
  Status GetCudagraphMeta(const int req_id,
                          const std::string& cudagrpah_name, 
                          const int bucket,
                          CudaGraphMeta*& meta);
  Status ReturnCudaGraphMeta(CudaGraphMeta* meta);

  void RegisterCudaGraphGroup(const std::string& group_name,
                              const std::string& cudagraph_name);
  // destory all cudagraph resouce belongs to certain group
  // return num of destoried cudagraphs 
  void DestoryAllCudaGraphResource();

  int DestoryCudaGraphGroupResource(const std::string& group_name);

  Status GetCudaStream(int req_id, cudaStream_t& stream);

  int GetStreamNum() { return num_stream_; };

private:
  CudaGraphMgr(/* args */) { 
    Init();  
  //  InitTraffic();
  };
  ~CudaGraphMgr();

  void Init();
  void InitTraffic();

  // assistant functions for capturing
  void GetInputDim0(const GraphDef& graph_def, const std::vector<string>& input_names,
                    std::vector<int>& input_dim0);
  void GenerateInputs(const GraphDef& graph_def, const std::vector<string>& input_names,
                      std::vector<Tensor>& input_tensors, int batch_size);
  void FillInputsMap(InputsMap& inputs_map, const std::vector<std::string>& input_names,
                     std::vector<Tensor>& input_tensors);
  void CheckCudaGraphScore(const GraphDef& graph_def,
                           CudaGraphMeta* meta,
                           const std::vector<std::string>& input_node_names,
                           const std::vector<std::string>& output_node_names);
  static void LaunchGraphInMeta(CudaGraphMeta* meta, cudaStream_t* stream);

  bool DestoryCudaGraphResource(const std::string& subgraph_name);
public:
  TF_DISALLOW_COPY_AND_ASSIGN(CudaGraphMgr);

private:
  // holds the tensors allocated during graph capturing
  // model_name --> tensor_holders
  // for each model, multiple graphs can be captured,
  // so we can run multiple graph instances in parallel
  // (to separate their memory, mutiple graphs are needed).
  std::unordered_map<std::string, BatchGraphMetaMap> graphname_batch_metas_map_;
  // devide all graph name into different group, support 
  // manage all cudagraph instances for certain graph
  std::unordered_map<std::string, std::unordered_set<std::string>> graphname_group_map_;
  std::vector<cudaStream_t> streams_;
  // stream and cuda graph instance count, each instance corresponds to one stream
  int num_stream_;
  int num_meta_instance_;
  // reserve gpu mem flag and its mutex
  bool gpu_mem_reserved_;

  Allocator* host_allocator_;
};

/* static */ CudaGraphMgr& CudaGraphMgr::Singleton() {
  static auto a = new CudaGraphMgr;
  return *a;
}

CudaGraphMgr::~CudaGraphMgr() = default;
}

#endif   // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_COMMON_RUNTIME_CUDAGRAPH_MGR_H_
