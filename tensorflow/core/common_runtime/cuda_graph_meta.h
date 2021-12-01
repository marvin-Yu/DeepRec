// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/30
// Description:
// CudaGrpahMeta, record all infos and data structure for cuda graph instance 

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_CUDAGRAPH_META_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_CUDAGRAPH_META_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#ifdef GOOGLE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensorflow {

typedef struct CudaGraphOutputInfo {
  const void* device_buffer_;
  DataType dtype_;
  TensorShape shape_;
  size_t ele_num_per_dim0_;
} CudaGraphOutputInfo;

typedef struct CudaGraphMeta {
#ifdef GOOGLE_CUDA
  cudaGraph_t cuda_graph_;
  cudaGraphExec_t cuda_graph_instance_;
  TensorHolder tensor_holder_;
  std::vector<std::pair<void*, void*>> src_dst_mapping_;
  // todo: check input tensor dim0
  std::vector<std::pair<void*, void*>> output_dst_src_mappping_;
  std::vector<CudaGraphOutputInfo> output_infos_;
  std::string graph_name_;
  int batch_size_;
  std::vector<int> input_dim0_;
  // mutex for ensure launch cudagraph atomic
  std::mutex mutex_;

CudaGraphMeta(const std::string& graph_name, int batch_size, const std::vector<int> dim0) :
    graph_name_(graph_name),
    batch_size_(batch_size),
    input_dim0_(dim0) {};

~CudaGraphMeta() {
  cudaGraphExecDestroy(cuda_graph_instance_);
  cudaGraphDestroy(cuda_graph_);
  src_dst_mapping_.clear();
  input_dim0_.clear();
  output_dst_src_mappping_.clear();
  output_infos_.clear();
}
#endif

} CudaGraphMeta;

} // tensorflow

#endif // TENSORFLOW_CORE_COMMON_RUNTIME_CUDAGRAPH_META_H_
