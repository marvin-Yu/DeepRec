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
#include <sstream>
#include <thread>
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
#include "tensorflow/tools/traffic/traffic.h"

static const int NUM_INSTANCE = 1;

namespace tensorflow {

void CudaGraphMgr::PrintTensorData(const Tensor& t) {
  const void* data;
  if (t.dtype() == DT_HALF) {
    data = static_cast<const void*>(t.flat<Eigen::half>().data());
  } else if (t.dtype() == DT_FLOAT) {
    data = static_cast<const void*>(t.flat<float>().data());
  } else if (t.dtype() == DT_BOOL) {
    data = static_cast<const void*>(t.flat<bool>().data());
  } else if (t.dtype() == DT_INT32) {
    data = static_cast<const void*>(t.flat<int>().data());
  } else {
    LOG(INFO) << "Print Tensor: Unsupported data type!" << std::endl;
    return;
  }

  int dims = t.dims();
  std::ostringstream tensor_string;
  tensor_string << "shape: " << std::endl;
  for (int i = 0; i < dims; i++) {
    tensor_string << t.dim_size(i) << ", ";
  }
  tensor_string << std::endl;

  int size = t.NumElements();
  size = size > 32 ? 32 : size;

  for (int i = 0; i < size; i++) {
    float value;
    if (t.dtype() == DT_HALF) {
      value = __half2float(static_cast<const __half*>(data)[i]);
    } else if (t.dtype() == DT_INT32) {
      value = static_cast<const int*>(data)[i];
    } else if (t.dtype() == DT_BOOL) {
      value = static_cast<const bool*>(data)[i];
    } else {
      value = static_cast<const float*>(data)[i];
    }
    tensor_string << value << ",";
  }
  LOG(INFO) << tensor_string.str();
}

void CheckCudaError(cudaError_t ERR) {
  if ((ERR) != cudaSuccess) {
    LOG(INFO) << "cuda error: " << ERR << " " << cudaGetErrorString(ERR)
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
    //  float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
      float value = 0.1;
      data[i] = __float2half(value);
    }
  } else if (t.dtype() == DT_FLOAT) {
    float* data = t.flat<float>().data();
    for (int i = 0; i < num_elements; i++) {
   //   float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
      float value = 0.1;
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
    LOG(INFO) << t.dtype() << std::endl;
    LOG(INFO) << "Random init: unsupported data type." << std::endl;
  }
}

void CudaGraphMgr::Init() {
  host_allocator_ = nullptr;
  num_instance_ = NUM_INSTANCE;

  // create launching streams
  streams_.resize(num_instance_);
  for (int i = 0; i < num_instance_; i++) {
    CheckCudaError(cudaStreamCreate(&streams_[i]));
  }
}

void CudaGraphMgr::InitTraffic() {
  ::Traffic::Instance()->RegistRecord("CgBatchSize", ::Traffic::STAT, "CudaGraph");
  ::Traffic::Instance()->RegistRecord("CgBatchSize-32", ::Traffic::COUNT, "CudaGraph");
  ::Traffic::Instance()->RegistRecord("CgBatchSize33-64", ::Traffic::COUNT, "CudaGraph");
  ::Traffic::Instance()->RegistRecord("CgBatchSize64-96", ::Traffic::COUNT, "CudaGraph");
  ::Traffic::Instance()->RegistRecord("CgBatchSize97-128", ::Traffic::COUNT, "CudaGraph");
  ::Traffic::Instance()->RegistRecord("CgBatchSize129-160", ::Traffic::COUNT, "CudaGraph");
  ::Traffic::Instance()->RegistRecord("CgBatchSize161-192", ::Traffic::COUNT, "CudaGraph");
  ::Traffic::Instance()->RegistRecord("CgBatchSize192-", ::Traffic::COUNT, "CudaGraph");
}

Status CudaGraphMgr::GetCudaStream(int req_id, cudaStream_t& stream) {
  stream = streams_[req_id % num_instance_];
  return Status::OK();
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

void CudaGraphMgr::FillInputsMap(InputsMap& inputs_map, const std::vector<std::string>& input_names,
                   std::vector<Tensor>& input_tensors) {
  assert(input_names.size() == input_tensors.size());

  for (size_t i = 0; i < input_tensors.size(); i++) {
    inputs_map.push_back(
        std::pair<std::string, Tensor>(input_names[i], input_tensors[i]));
  }
}

void CudaGraphMgr::CheckCudaGraphScore(const GraphDef& graph_def,
                          CudaGraphMeta* meta,
                          const std::vector<std::string>& input_node_names,
                          const std::vector<std::string>& output_node_names) {
  if (host_allocator_ == nullptr) {
    return;
  }

  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(false);
  std::unique_ptr<Session> session(NewSession(options));
  
  TF_CHECK_OK(session->CreateForCapture(graph_def));

  int warm_batch = 1;
  std::vector<Tensor> input_tensors_tf;
  GenerateInputs(graph_def, input_node_names, input_tensors_tf, warm_batch);
  InputsMap inputs_tf; // input map for Normal TF run
  FillInputsMap(inputs_tf, input_node_names, input_tensors_tf);
  std::vector<Tensor> output_tensors_tf;
  TF_CHECK_OK(session->Run(inputs_tf, output_node_names, {}, &output_tensors_tf));

  LOG(INFO) << "TF results: ";
  PrintTensorData(output_tensors_tf[0]); 

  for (int i = 0; i < input_tensors_tf.size(); ++i) {
    const Tensor& input = input_tensors_tf[i];
    const void* host_buffer;
    size_t ele_size = 1;
    if (input.dtype() == DT_HALF) {
      ele_size = 2;
      host_buffer = reinterpret_cast<const void*>(input.flat<Eigen::half>().data());
    } else if (input.dtype() == DT_FLOAT) {
      ele_size = 4;
      host_buffer = reinterpret_cast<const void*>(input.flat<float>().data());
    } else if (input.dtype() == DT_INT32) {
      ele_size = 4;
      host_buffer = reinterpret_cast<const void*>(input.flat<int>().data());
    } else if (input.dtype() == DT_BOOL) {
      ele_size = 1;
      host_buffer = reinterpret_cast<const void*>(input.flat<bool>().data());
    } else if (input.dtype() == DT_INT64) {
      ele_size = 8;
      host_buffer = reinterpret_cast<const void*>(input.flat<int64>().data());
    } else {
      std::cout << "Unsupported data type!" << std::endl;
      exit(1);
    }

    size_t num_elements = input.NumElements();
    size_t num_bytes = num_elements * ele_size;
    void* device_buffer = meta->src_dst_mapping_[i].second;
    if (cudaMemcpyAsync(device_buffer, host_buffer, num_bytes,
        cudaMemcpyHostToDevice, streams_[0]) != cudaSuccess) {
      LOG(ERROR) << "CudaMemCpy to " << i <<  " st tensor failed, device addr: " << device_buffer;
    } 
  }

  // run cuda graph instance
  LaunchGraphInMeta(meta, &(streams_[0]));
  PrintTensorData(meta->output_tensors_[0]);

  return;
}

void CudaGraphMgr::LaunchGraphInMeta(CudaGraphMeta* meta, cudaStream_t* stream) {
  cudaError_t ret = cudaGraphLaunch(meta->cuda_graph_instance_, *stream);
  if (ret != cudaSuccess) {
    LOG(ERROR) << "cudagraph launch faild: " << ret;
  }
  cudaEvent_t event;
  CheckCudaError(cudaEventCreateWithFlags(&event, cudaEventBlockingSync));
  CheckCudaError(cudaEventRecord(event, *stream));
  CheckCudaError(cudaEventSynchronize(event));
  CheckCudaError(cudaEventDestroy(event));
  return;
}

void CudaGraphMgr::DestoryCudagraphMeta() {
  graphname_batch_metas_map_.clear();
}

bool CudaGraphMgr::CheckGraphAllCaptured(const std::vector<std::string>& graph_names, std::vector<int>& uncaptured_index) {
  uncaptured_index.clear();
  for (int i = 0; i < graph_names.size(); ++i) {
    if (graphname_batch_metas_map_.find(graph_names[i]) == graphname_batch_metas_map_.end()) {
       uncaptured_index.push_back(i);
    }
  }
  return 0 == uncaptured_index.size();
}

// todo: move batch size, num instance into options.
/**
 * @brief Capture cuda graph by given graph def
 * 
 * @param graph_def graph def which is cutted, all nodes should be placed on GPU
 * @param graph_name name for indenty
 * @param input_node_names input nodde name for feeding input
 * @param output_node_names 
 * @param batch_size 
 * @return Status 
 */
Status CudaGraphMgr::CaptureCudagraph(const GraphDef& graph_def, 
                                      const std::string& graph_name, 
                                      const std::vector<std::string>& input_node_names,
                                      const std::vector<std::string>& output_node_names,
                                      const std::vector<int>& batch_size) {
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(true);
  LOG(INFO) << "[Jieluo] begin create new session for capturing";
  std::unique_ptr<Session> session(NewSession(options));
  LOG(INFO) << "[Jieluo] create new session for capture finished";
  TF_CHECK_OK(session->CreateForCapture(graph_def));

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
  LOG(INFO) << "[Jieluo] begin run session for capturing warmup";
  // First session run, init needed resources
  for (int i = 0; i < batch_size.size(); ++i) {
    int warm_batch = batch_size[i];
    std::vector<Tensor> input_tensors_tf;
    GenerateInputs(graph_def, input_node_names, input_tensors_tf, warm_batch);

    InputsMap inputs_tf; // input map for Normal TF run
    FillInputsMap(inputs_tf, input_node_names, input_tensors_tf);
    
    std::vector<Tensor> output_tensors_tf;
    TF_CHECK_OK(session->Run(inputs_tf, output_node_names, {}, &output_tensors_tf));
  }
  LOG(INFO) << "[Jieluo] run session for capturing warmup finished";

  bool destoried = DestoryCudaGraphResource(graph_name);
  if (destoried) {
    LOG(INFO) << "Old cuda graph resources for " << graph_name << " destoried";
  }

  // capture the cuda graph
  assert(session->SupportsCudaGraph());
  cudaStream_t stream = session->EnableGraphCapture();
  LOG(INFO) << "capturing on stream -- " << stream;
  if (stream == NULL) {
    return Status(error::Code::INTERNAL,
                  "Get stream for graph capturing failed.");
  }

  CudaGraphMeta* meta_check = nullptr;
  for (int i = 0; i < batch_size.size(); ++i) {
    if (graphname_batch_metas_map_.find(graph_name) == graphname_batch_metas_map_.end()) {
      graphname_batch_metas_map_.emplace(graph_name, BatchGraphMetaMap());
    }
    auto& batch_meta_map = graphname_batch_metas_map_[graph_name];
    if (batch_meta_map.find(batch_size[i]) == batch_meta_map.end()) {
      batch_meta_map.emplace(batch_size[i], std::vector<CudaGraphMeta*>());
    }
    std::vector<Tensor> input_tensors_cuda_graph;
    InputsMap inputs_cuda_graph;
    GenerateInputs(graph_def, input_node_names, input_tensors_cuda_graph, batch_size[i]);
    FillInputsMap(inputs_cuda_graph, input_node_names, input_tensors_cuda_graph);
    
    for (int j = 0; j < num_instance_; j++) {
      CudaGraphMeta* meta = new CudaGraphMeta(graph_name, batch_size[i]);
      batch_meta_map[batch_size[i]].push_back(meta);
      TF_CHECK_OK(session->RunForCapture(inputs_cuda_graph, output_node_names, {}, meta));
      if (meta_check == nullptr) {
        meta_check = meta;
      }
    }
    LOG(INFO) << "[Jieluo] run session for capturing finish, batch size " << batch_size[i];
    // add pool lock
    if (meta_pool_lock_.find(graph_name) == meta_pool_lock_.end()) {
      meta_pool_lock_.emplace(graph_name, BatchMetaLockMap());
    }
    if (meta_pool_lock_[graph_name].find(batch_size[i]) == meta_pool_lock_[graph_name].end()) {
      std::mutex* mutex_ptr = new std::mutex();
      std::condition_variable* cv_ptr = new std::condition_variable();
      meta_pool_lock_[graph_name].emplace(batch_size[i], std::make_pair(mutex_ptr, cv_ptr));
    }  
  }
  // turn off graph capture mode
  session->DisableGraphCapture();
 // CheckCudaGraphScore(graph_def, meta_check, input_node_names, output_node_names);
  return Status::OK();
}

Status CudaGraphMgr::GetCudagraphMeta(const std::string& cudagraph_name, 
                                      const int bucket,
                                      CudaGraphMeta*& meta) {
  if (meta_pool_lock_.find(cudagraph_name) == meta_pool_lock_.end()) {
    return errors::Internal("Cuda graph instance with name ", cudagraph_name, " not fount.");
  } else if (meta_pool_lock_[cudagraph_name].find(bucket) == meta_pool_lock_[cudagraph_name].end()) {
    return errors::Internal("Cuda graph intannce with name ", 
                            cudagraph_name, " and batch size ", 
                            bucket, " not found");
  }

  std::mutex* mutex = meta_pool_lock_[cudagraph_name][bucket].first;
  std::condition_variable* cv = meta_pool_lock_[cudagraph_name][bucket].second;
  std::vector<CudaGraphMeta*>& metas = graphname_batch_metas_map_[cudagraph_name][bucket];

  std::unique_lock<std::mutex> lock(*mutex);
  if (metas.size() > 0) {
    meta = metas[metas.size() - 1];
    metas.pop_back();
    return Status::OK();
  } else {
    while(metas.size() == 0) {
      cv->wait(lock);
    }
    meta = metas[metas.size() - 1];
    metas.pop_back();
    return Status::OK();
  }
}

Status CudaGraphMgr::ReturnCudaGraphMeta(CudaGraphMeta* meta) {
  const std::string& graph_name = meta->graph_name_;
  const int bucket = meta->batch_size_;
  if (meta_pool_lock_.find(graph_name) == meta_pool_lock_.end()) {
    return errors::Internal("Cuda graph instance with name ", graph_name, " not fount.");
  } else if (meta_pool_lock_[graph_name].find(bucket) == meta_pool_lock_[graph_name].end()) {
    return errors::Internal("Cuda graph intannce with name ", 
                            graph_name, " and batch size ", 
                            bucket, " not found");
  }

  std::mutex* mutex = meta_pool_lock_[graph_name][bucket].first;
  std::condition_variable* cv = meta_pool_lock_[graph_name][bucket].second;
  std::vector<CudaGraphMeta*>& metas = graphname_batch_metas_map_[graph_name][bucket];

  std::unique_lock<std::mutex> lock(*mutex);
  metas.push_back(meta);
  if (metas.size() == 1) {
    cv->notify_one();
  }
  return Status::OK();
}

// todo: make sure thread safety!
bool CudaGraphMgr::DestoryCudaGraphResource(const std::string& subgraph_name) {
  bool delete_meta = false;
  auto iter = graphname_batch_metas_map_.find(subgraph_name);
  if (iter != graphname_batch_metas_map_.end()) {
    BatchGraphMetaMap& meta_map = iter->second;
    for (auto meta_iter = meta_map.begin(); meta_iter != meta_map.end(); ++meta_iter) {
      for (int i = 0; i < meta_iter->second.size(); ++i) {
        delete (meta_iter->second)[i];
      }
      meta_iter->second.clear();
    }
    graphname_batch_metas_map_.erase(iter);
    delete_meta = true;
  }
  
  auto lock_iter = meta_pool_lock_.find(subgraph_name);
  if (lock_iter != meta_pool_lock_.end()) {
    BatchMetaLockMap& lock_map = iter->second;
    for (auto mutex_iter = lock_map.begin(); mutex_iter != lock_map.end(); ++mutex_iter) {
      delete mutex_iter->second.first;
      delete mutex_iter->second.second;
    }
    meta_pool_lock_.erase(lock_iter);
  }
  return delete_meta;
}

} // tensorflow
