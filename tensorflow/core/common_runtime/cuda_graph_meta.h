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
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#endif

namespace tensorflow {

typedef struct CudaGraphMeta {
#ifdef GOOGLE_CUDA
  cudaGraph_t cuda_graph_;
  cudaGraphExec_t cuda_graph_instance_;
  TensorHolder tensor_holder_;
  std::vector<std::pair<void*, void*>> src_dst_mapping_;
  std::vector<Tensor> output_tensors_;
#endif

CudaGraphMeta::~CudaGraphMeta() {
  cudaGraphExecDestroy(cuda_graph_instance_);
  cudaGraphDestroy(cuda_graph_);
  output_tensors_.clear();
  src_dst_mapping_.clear();
}

} CudaGraphMeta;

} // tensorflow

#endif // TENSORFLOW_CORE_COMMON_RUNTIME_CUDAGRAPH_META_H_
