// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/25
// Description:
// CudaGrpah Manager, singleton
// capture cudagraph and manage cudagraph instance, streams for launching cudagraph

#include "tensorflow/core/common_runtime/cuda_graph_mgr.h"

#include <string>
#include <vector>
#include <cuda_fp16.h>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"

static const int NUM_INSTANCE = 3;

namespace tensorflow {

void CheckCudaError(cudaError_t ERR) {
  if ((ERR) != cudaSuccess) {
    std::cout << "cuda error: " << ERR << " " << cudaGetErrorString(ERR)
              << std::endl;
  }
}

TensorShape getNodeShape(const GraphDef& graph_def, const std::string name,
                         int batch_size) {
  for (int i = 0; i < graph_def.node_size(); i++) {
    auto n = graph_def.node(i);
    if (n.name() == name) {
      auto shape = n.attr().at("shape").shape();
      int dims = shape.dim_size();
      TensorShape tensorShape;

      for (int d = 0; d < dims; d++) {
        int dim_size = shape.dim(d).size();

        if (d == 0 && dim_size == -1) {
          int new_size = batch_size;

          // assume the first dimension is batch size, note that it may not be
          // true for some models.
          LOG(INFO) << "change batch size from: " << dim_size << " to "
                    << new_size << std::endl;
          dim_size = new_size;
        }
        tensorShape.AddDim(dim_size);
      }

      return tensorShape;
    }
  }
  LOG(ERROR) << "Cannot find the node" << name << std::endl;
  exit(1);
}

DataType getNodeType(const GraphDef& graph_def, const std::string name) {
  for (int i = 0; i < graph_def.node_size(); i++) {
    auto n = graph_def.node(i);
    if (n.name() == name) {
      auto dtype = n.attr().at("dtype").type();
      return dtype;
    }
  }
  LOG(ERROR) << "Cannot find the node" << name << std::endl;
  exit(1);
}

void RandomInitialize(Tensor& t) {
  int num_elements = t.NumElements();
  if (t.dtype() == DT_HALF) {
    __half* data = reinterpret_cast<__half*>(t.flat<Eigen::half>().data());
    for (int i = 0; i < num_elements; i++) {
      float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
      data[i] = __float2half(value);
    }
  } else if (t.dtype() == DT_FLOAT) {
    float* data = t.flat<float>().data();
    for (int i = 0; i < num_elements; i++) {
      float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
      data[i] = value;
    }
  } else if (t.dtype() == DT_INT32) {
    int* data = t.flat<int>().data();
    for (int i = 0; i < num_elements; i++) {
      int value = static_cast<int>(rand() % 10);
      data[i] = value;
    }
  } else if (t.dtype() == DT_BOOL) {
    bool* data = t.flat<bool>().data();
    for (int i = 0; i < num_elements; i++) {
      bool value = static_cast<bool>(rand() % 2);
      data[i] = value;
    }
  } else if (t.dtype() == DT_INT64) {
    int64* data = t.flat<int64>().data();
    for (int i = 0; i < num_elements; i++) {
      int64 value = static_cast<int64>(rand() % 10);
      data[i] = value;
    }
  } else {
    std::cout << t.dtype() << std::endl;
    std::cout << "Random init: unsupported data type." << std::endl;
  }
}

void CudaGraphMgr::Init() {
  host_allocator_ = nullptr;
  num_instance_ = NUM_INSTANCE;

  // create launching streams
  streams_.resize(num_instance_);
  for (int i = 0; i < num_instance; i++) {
    CheckCudaError(cudaStreamCreate(&streams_[i]));
  }
}

void CudaGraphMgr::GenerateInputs(const GraphDef& graph_def, const std::vector<string>& input_names,
                    std::vector<Tensor>& input_tensors, int batch_size) {
  input_tensors.clear();
  for (int i = 0; i < input_names.size(); i++) {
    auto tensorshape = getNodeShape(graph_def, input_names[i], batch_size);
    auto tensortype = getNodeType(graph_def, input_names[i]);

    Tensor t;
    t = Tensor(host_allocator_, tensortype, tensorshape);
    RandomInitialize(t);

    input_tensors.push_back(t);
  }
}

void CudaGraphMgr::FillInputsMap(InputsMap& inputs_map, std::vector<std::string>& input_names,
                   std::vector<Tensor>& input_tensors) {
  assert(input_names.size() == input_tensors.size());

  for (size_t i = 0; i < input_tensors.size(); i++) {
    inputs_map.push_back(
        std::pair<std::string, Tensor>(input_names[i], input_tensors[i]));
  }
}

void CudaGraphMgr::LogCudaGraphStatus(Session* sess) {
  int num_models = sess->NumCapturedModels();
  LOG(INFO) << "Captured " << num_models << " models";
  for (int i = 0; i < num_models; i++) {
    std::string model_name = sess->CapturedModelName(i);
    int num_graphs = sess->NumCapturedGraphs(model_name);
    int total_bytes = sess->AllocatedBytesCudaGraph(model_name);
    LOG(INFO) << "model: " << model_name;
    LOG(INFO) << "num graphs: " << num_graphs;
    LOG(INFO) << "allocated bytes: " << total_bytes;
  }
}

bool CudaGraphMgr::CheckGraphAllCaptured(const std::vector<std::string>& graph_names, std::vector<int>& uncaptured_index) {
  uncaptured_index.clear();
  for (int i = 0; i < graph_names.size(); ++i) {
    if (cuda_graphs_.find(graph_names[i]) == cuda_graphs_.end()) {
      uncaptured_index.push_back(i);
    }
  }
  return 0 == uncaptured_index.size();
}

// todo: move batch size, num instance into options.
Status CudaGraphMgr::CaptureCudagraph(const GraphDef& graph_def, 
                                      const std::string& graph_name, 
                                      const std::vector<std::string>& input_node_names,
                                      const std::vector<std::string>& output_node_names,
                                      const std::vector<int>& batch_size) {
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(false);
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->CreateForCapture(graph_def, i));

  // init host_allocator
  if (host_allocator_ == nullptr) {
    const DeviceMgr* device_manager;
    TF_CHECK_OK(session->LocalDeviceManager(&device_manager));
    std::vector<Device*> devices = device_manager->ListDevices();
    for (auto* d : devices) {
      if (d->name().find("CPU") != std::string::npos) {
        // todo: reuse this allocator
        host_allocator_ = dynamic_cast<ThreadPoolDevice*>(d)->GetAllocator(
            AllocatorAttributes());
      }
    }
  }

  // capture the cuda graph
  assert(session->SupportsCudaGraph());
  cudaStream_t stream = session->EnableGraphCapture(graph_name);
  LOG(INFO) << "capturing on stream -- " << stream;
  if (stream == NULL) {
    return Status(error::Code::INTERNAL,
                  "Get stream for graph capturing failed.");
  }
  // For multiple-stream runs,
  // We need to capture multiple independent cuda graphs
  // with seperated inputs/output buffers, and buffers for intermedidate
  // layers
  std::vector<InputsMap> inputs_cuda_graph(num_instance);
  std::vector<std::vector<Tensor>> input_tensors_cuda_graph(num_instance);
  std::vector<std::vector<Tensor>> output_tensors_cuda_graph(num_instance);

  // prepare inputs
  for (int i = 0; i < num_instance; i++) {
    GenerateInputs(graph_def, input_node_names, input_tensors_cuda_graph[i],
                   batch_size);
    FillInputsMap(inputs_cuda_graph[i], input_node_names,
                  input_tensors_cuda_graph[i]);
  }
  // capture multiple graphs
  std::vector<std::string> output_names;  // todo: gen output_name
  for (int i = 0; i < num_instance; i++) {
    TF_CHECK_OK(session->Run(inputs_cuda_graph[i], output_names, {},
                             &output_tensors_cuda_graph[i]));
  }
  // turn off graph capture mode
  session->DisableGraphCapture();
  LogCudaGraphStatus(session.get());
  return Status::OK();
}

} // tensorflow
