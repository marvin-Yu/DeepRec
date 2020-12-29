// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/25
// Description:
// CudaGrpah Manager, singleton
// capture cudagraph and manage cudagraph instance, streams for launching cudagraph

#include "tensorflow/core/common_runtime/cudagraph_mgr.h"

#include <string>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"



namespace tensorflow {

void CudaGraphMgr::GenerateInputs(GraphDef& graph_def, const std::vector<string>& input_names,
                    std::vector<Tensor>& input_tensors, int batch_size) {
  input_tensors.clear();
  for (int i = 0; i < input_names.size(); i++) {
    auto tensorshape = getNodeShape(graph_def, input_names[i], batch_size);
    auto tensortype = getNodeType(graph_def, input_names[i]);

    Tensor t;
    t = Tensor(host_allocator, tensortype, tensorshape);
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

// todo: move batch size, num instance into options.
Status CudaGraphMgr::CaptureCudagraph(const GraphDef& graph_def, const SessionOptions& options,
                                      const int batch_size, const int num_instance) {
  // todo: check params > 0
  // Creates a session.
  Allocator* host_allocator = nullptr;
  int num_graph = options.config.graph_options()
                      .optimization_options()
                      .subgraph_descriptions_size();
  for (int i = 0; i < num_graph; ++i) {
    std::unique_ptr<Session> session(NewSession(options));
    TF_CHECK_OK(session->CreateForCapture(graph_def, i));

    // init host_allocator
    if (host_allocator == nullptr) {
      const DeviceMgr* device_manager;
      Allocator* host_allocator;
      TF_CHECK_OK(session->LocalDeviceManager(&device_manager));
      std::vector<Device*> devices = device_manager->ListDevices();
      for (auto* d : devices) {
        if (d->name().find("CPU") != std::string::npos) {
          // todo: reuse this allocator
          host_allocator = dynamic_cast<ThreadPoolDevice*>(d)->GetAllocator(
              AllocatorAttributes());
        }
      }
    }

    // capture the cuda graph
    assert(session->SupportsCudaGraph());
    cudaStream_t streams[MAX_NUM_STREAMS];

    // create launching streams
    for (int i = 0; i < num_streams; i++) {
      CheckCudaError(cudaStreamCreate(&streams[i]));
    }

    std::string graph_name = options.config.graph_options().
                                 optimization_options().
                                 subgraph_descriptions(i).
                                 subgraph_name();
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
      GenerateInputs(graph_def, input_names, input_tensors_cuda_graph[i],
                     batch_size);
      FillInputsMap(inputs_cuda_graph[i], input_names,
                    input_tensors_cuda_graph[i]);
    }
    // capture multiple graphs
    for (int i = 0; i < num_streams; i++) {
      TF_CHECK_OK(session->Run(inputs_cuda_graph[i], output_names, {},
                               &output_tensors_cuda_graph[i]));
    }
    // turn off graph capture mode
    session->DisableGraphCapture();
    LogCudaGraphStatus(session.get());
  }
  return Status::OK();
}

} // tensorflow
