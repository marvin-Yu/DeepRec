// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/25
// Description:
// CudaGrpah Manager, singleton
// capture cudagraph and manage cudagraph instance, streams for launching cudagraph

#ifdef GOOGLE_CUDA
#include "tensorflow/core/common_runtime/cuda_graph_mgr.h"

#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <cuda_fp16.h>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/traffic/traffic.h"

static const int NUM_INSTANCE_DEFAULT = 3;
static const int NUM_STREAM_DEFAULT = 3;
static const float RESERVE_GPUMEM_RATIO_DEFAULT = 0.3;
static const size_t CHUNK_SIZE = 256 * 1024 * 1024;
static const size_t MAX_RESERVE_CHUNK = 16;
static const std::string DEFAULT_GROUP = "_DEFAULT_";

namespace tensorflow {

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
      float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
    //  float value = 0.1;
      data[i] = __float2half(value);
    }
  } else if (t.dtype() == DT_FLOAT) {
    float* data = t.flat<float>().data();
    for (int i = 0; i < num_elements; i++) {
      float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
    //  float value = 0.1;
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
  gpu_mem_reserved_ = false;
  host_allocator_ = nullptr;
  
  char* stream_num_var = getenv("TF_CUDA_GRAPH_STREAM_NUM");
  if (stream_num_var == NULL) {
    LOG(INFO) << "Environment Variable TF_CUDA_GRAPH_STREAM_NUM not set, use default " << NUM_STREAM_DEFAULT;
    num_stream_ = NUM_STREAM_DEFAULT;
  } else {
    num_stream_ = atoi(stream_num_var);
    if (num_stream_ <= 0) {
      num_stream_ = NUM_STREAM_DEFAULT;
      LOG(INFO) << "Environment Variable TF_CUDA_GRAPH_STREAM_NUM invalid " << stream_num_var 
                << " use default " << NUM_STREAM_DEFAULT;
    }
    LOG(INFO) << "cuda stream num " << num_stream_;
  }

  char* instance_num_var = getenv("TF_CUDA_GRAPH_INSTANCE_NUM");
  if (instance_num_var == NULL) {
    LOG(INFO) << "Environment Variable TF_CUDA_GRAPH_INSTANCE_NUM not set, use default " << NUM_INSTANCE_DEFAULT;
    num_meta_instance_ = NUM_INSTANCE_DEFAULT;
  } else {
    num_meta_instance_ = atoi(instance_num_var);
    if (num_meta_instance_ <= 0) {
      num_meta_instance_ = NUM_INSTANCE_DEFAULT;
      LOG(INFO) << "Environment Variable TF_CUDA_GRAPH_INSTANCE_NUM invalid " << instance_num_var 
                << " use default " << NUM_INSTANCE_DEFAULT;
    }
    LOG(INFO) << "cuda meta instance num " << num_meta_instance_;
  }

  // create launching streams
  streams_.resize(num_stream_);
  for (int i = 0; i < num_stream_; i++) {
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

  static auto traffic_cb = [](const char* param) {
    LOG(WARNING) << "Traffic:[" << param << "]";
  };
  ::Traffic::Instance()->Start(traffic_cb);
}

Status CudaGraphMgr::GetCudaStream(int req_id, cudaStream_t& stream) {
  stream = streams_[req_id % num_stream_];
  return Status::OK();
}

void CudaGraphMgr::GenerateInputs(const GraphDef& graph_def, const std::vector<string>& input_names,
                    std::vector<Tensor>& input_tensors, int batch_size) {
  input_tensors.clear();
  for (int i = 0; i < input_names.size(); ++i) {
    auto tensorshape = getNodeShape(graph_def, input_names[i], batch_size);
    auto tensortype = getNodeType(graph_def, input_names[i]);

    Tensor t;
    t = Tensor(host_allocator_, tensortype, tensorshape);
    RandomInitialize(t);

    input_tensors.push_back(t);
  }
}

void CudaGraphMgr::GetInputDim0(const GraphDef& graph_def, const std::vector<string>& input_names,
                    std::vector<int>& input_dim0) {
  input_dim0.clear();
  for (int i = 0; i < input_names.size(); ++i) {
    bool found = false;
    for (int j = 0; j < graph_def.node_size(); ++j) {
      auto n = graph_def.node(j);
      if (n.name() == input_names[i]) { 
        found = true;   
        auto shape = n.attr().at("shape").shape();
        int dims = shape.dim_size();
        if (dims > 0) {
          LOG(INFO) << "add dim0 " << shape.dim(0).size() << " for input " << i;
          input_dim0.emplace_back(shape.dim(0).size());
        } else {
          LOG(INFO) << "add dim0 0 for const input " << i;
          input_dim0.emplace_back(0);
        }
        break; 
      }
    }
    if (!found) {
      LOG(INFO) << "add dim0 0 for not found input " << i;
      input_dim0.emplace_back(0);
    }
  }
  return;
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
  options.config.mutable_gpu_options()->set_allow_growth(true);
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
 // PrintTensorData(output_tensors_tf[0]); 

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
      LOG(ERROR) << "Unsupported data type: " << input.dtype();
      return;
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
  LOG(INFO) << "CudaGraph results: ";
  // PrintTensorData(meta->output_tensors_[0]);

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

void CudaGraphMgr::RegisterCudaGraphGroup(const std::string& group_name,
                                          const std::string& cudagraph_name) {
  auto iter = graphname_group_map_.find(group_name);
  if (iter == graphname_group_map_.end()) {
    graphname_group_map_.emplace(group_name, std::unordered_set<std::string>());
  }
  graphname_group_map_[group_name].emplace(cudagraph_name);
  LOG(INFO) << "register cuda graph name " << cudagraph_name 
            << "in group " << group_name;
  return;
}

// not thread safe!
void CudaGraphMgr::DestoryAllCudaGraphResource() {
  LOG(INFO) << "begin destory all cuda graph resource";
  // step2. clear group info
  graphname_group_map_.clear();

  // step3. clear metas
  for (auto graph_iter = graphname_batch_metas_map_.begin(); graph_iter != graphname_batch_metas_map_.end(); ++graph_iter) {
    BatchGraphMetaMap& meta_map = graph_iter->second;
    for (auto meta_iter = meta_map.begin(); meta_iter != meta_map.end(); ++meta_iter) {
      if (meta_iter->second.size() != num_meta_instance_) {
        LOG(WARNING) << "not all meta intance returned, expect " << num_meta_instance_
                     << " actual " << meta_iter->second.size();
      }
      for (int i = 0; i < meta_iter->second.size(); ++i) {
        delete (meta_iter->second)[i];
      }
      meta_iter->second.clear();
    }
  }
  return;
}

int CudaGraphMgr::DestoryCudaGraphGroupResource(const std::string& group_name) {
  int destoried_num = 0;
  LOG(INFO) << "begin release all cuda graph resources in group " << group_name;
  auto iter = graphname_group_map_.find(group_name);
  if (iter != graphname_group_map_.end()) {
    for (auto& name_iter : iter->second) {
      LOG(INFO) << "release all cuda graph resources for " << name_iter;
      if(DestoryCudaGraphResource(name_iter)) {
        ++destoried_num;
      }
    }
    graphname_group_map_.erase(iter);
  }
  return destoried_num;
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
                                      const std::string& group_name,
                                      const std::string& origin_graph_name, 
                                      const std::vector<std::string>& input_node_names,
                                      const std::vector<std::string>& output_node_names,
                                      const std::vector<int>& batch_size) {
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(true);
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->CreateForCapture(graph_def));

  std::string graph_name = origin_graph_name; 

  const DeviceMgr* device_manager;
  TF_CHECK_OK(session->LocalDeviceManager(&device_manager));
  std::vector<Device*> devices = device_manager->ListDevices();
  // init host_allocator
  if (host_allocator_ == nullptr) {
    for (auto* d : devices) {
      LOG(INFO) << "device name is " << d->name() << " type is  "  << d->attributes().device_type();
      if (d->attributes().device_type() == "CPU") {
        host_allocator_ = dynamic_cast<ThreadPoolDevice*>(d)->GetAllocator(
            AllocatorAttributes());
      }
    }
  }

  // reserve gpu mem
  if (!gpu_mem_reserved_) {
      // reserve
    for (auto * d : devices){
      if(d->attributes().device_type() == "GPU"){
        auto gpu = dynamic_cast<BaseGPUDevice*>(d);
        gpu_mem_reserved_ = true;
        float reserve_gpu_mem_ratio = RESERVE_GPUMEM_RATIO_DEFAULT;
        char* reserve_gpumem_ratio_var = getenv("TF_CUDA_GRAPH_GPUMEM_RATIO");
        if (reserve_gpumem_ratio_var == NULL) {
          LOG(INFO) << "Environment Variable TF_CUDA_GRAPH_INSTANCE_NUM not "
                       "set, use default "
                    << RESERVE_GPUMEM_RATIO_DEFAULT;
        } else {
          reserve_gpu_mem_ratio = atof(reserve_gpumem_ratio_var);
          if (reserve_gpu_mem_ratio < 0.0001 || reserve_gpu_mem_ratio > 0.999) {
            LOG(INFO)
                << "Environment Variable TF_CUDA_GRAPH_INSTANCE_NUM invalid "
                << reserve_gpumem_ratio_var << " use default "
                << RESERVE_GPUMEM_RATIO_DEFAULT;
            reserve_gpu_mem_ratio = RESERVE_GPUMEM_RATIO_DEFAULT;
          }
        }

        int64_t mem_limit = gpu->attributes().memory_limit();
        int reserve_block = mem_limit * reserve_gpu_mem_ratio / CHUNK_SIZE;
        LOG(INFO) << "GPU memory limit is " << mem_limit
                  << ", reserve block num " << reserve_block;
        if (!gpu->ReserveGPUMemChunks(CHUNK_SIZE, reserve_block)) {
          LOG(ERROR) << "Reserve chunk failed, request chunk num "
                     << reserve_block;
        }
      }
    }
  }

  // First session run, init needed resources by normal tf run
  for (int i = 0; i < batch_size.size(); ++i) {
    int warm_batch = batch_size[i];
    std::vector<Tensor> input_tensors_tf;
    GenerateInputs(graph_def, input_node_names, input_tensors_tf, warm_batch);

    InputsMap inputs_tf; // input map for Normal TF run
    FillInputsMap(inputs_tf, input_node_names, input_tensors_tf);
    for (int j = 0; j < num_meta_instance_; ++j) { 
      std::vector<Tensor> output_tensors_tf;
      TF_CHECK_OK(session->Run(inputs_tf, output_node_names, {}, &output_tensors_tf));
    }
  }

  // destory existed meta with same key (update)
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

  std::vector<int> dim0;
  GetInputDim0(graph_def, input_node_names, dim0);

  CudaGraphMeta* meta_check = nullptr;
  for (int i = 0; i < batch_size.size(); ++i) {
    if (graphname_batch_metas_map_.find(graph_name) == graphname_batch_metas_map_.end()) {
      LOG(INFO) << "find " << graph_name << " in meta map failed, insert";
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
    
    for (int j = 0; j < num_meta_instance_; j++) {
      CudaGraphMeta* meta = new CudaGraphMeta(graph_name, batch_size[i], dim0);
      batch_meta_map[batch_size[i]].push_back(meta);
      TF_CHECK_OK(session->RunForCapture(inputs_cuda_graph, output_node_names, {}, meta));
      if (meta_check == nullptr) {
        meta_check = meta;
      }
    }
  }
  // turn off graph capture mode
  session->DisableGraphCapture();
  RegisterCudaGraphGroup(group_name, graph_name);
 // CheckCudaGraphScore(graph_def, meta_check, input_node_names, output_node_names);
  return Status::OK();
}

Status CudaGraphMgr::GetCudagraphMeta(const int req_id,
                                      const std::string& cudagraph_name, 
                                      const int bucket,
                                      CudaGraphMeta*& meta) {
  if (graphname_batch_metas_map_.find(cudagraph_name) == graphname_batch_metas_map_.end()) {
    return errors::Internal("Cuda graph instance with name ", cudagraph_name, " not fount.");
  } else if (graphname_batch_metas_map_[cudagraph_name].find(bucket) == graphname_batch_metas_map_[cudagraph_name].end()) {
    return errors::Internal("Cuda graph intannce with name ", 
                            cudagraph_name, " and batch size ", 
                            bucket, " not found");
  }
  std::vector<CudaGraphMeta*>& metas = graphname_batch_metas_map_[cudagraph_name][bucket];
  meta = metas[req_id % num_meta_instance_];
  return Status::OK();
}

Status CudaGraphMgr::ReturnCudaGraphMeta(CudaGraphMeta* meta) {
  // do nothing
  return Status::OK();
}

// todo: make sure thread safety!
bool CudaGraphMgr::DestoryCudaGraphResource(const std::string& subgraph_name) {
  bool delete_meta = false;

  auto iter = graphname_batch_metas_map_.find(subgraph_name);
  if (iter != graphname_batch_metas_map_.end()) {
    BatchGraphMetaMap& meta_map = iter->second;
    for (auto meta_iter = meta_map.begin(); meta_iter != meta_map.end(); ++meta_iter) {
      if (meta_iter->second.size() != num_meta_instance_) {
        LOG(WARNING) << "Not all meta intance returned, expect " << num_meta_instance_
                     << " actual " << meta_iter->second.size();
      }
      for (int i = 0; i < meta_iter->second.size(); ++i) {
        delete (meta_iter->second)[i];
      }
      meta_iter->second.clear();
    }
    graphname_batch_metas_map_.erase(iter);
    delete_meta = true;
  }
  return delete_meta;
}

} // tensorflow

#endif  // GOOGLE_CUDA
