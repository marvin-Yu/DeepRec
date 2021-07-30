#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/cuda_graph_session.h"

#include <cassert>

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

namespace tensorflow {

class CudaGraphSessionFactory : public SessionFactory {
 public:
  CudaGraphSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return options.target == CUDA_GRAPH_TARGET_NAME;
  }

  Status NewSession(const SessionOptions& options,
                    Session** out_session) override {
    std::vector<Device*> devices;
    TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));

    CudaGraphSession* session =
        new CudaGraphSession(options, new DeviceMgr(devices), this);
    {
      mutex_lock l(sessions_lock_);
      sessions_.push_back(session);
    }
    *out_session = session;
    return Status::OK();
  }

  Status Reset(const SessionOptions& options,
               const std::vector<string>& containers) override {
    return errors::Unimplemented("Reset()");
  }

  void Deregister(const CudaGraphSession* session) {
    mutex_lock l(sessions_lock_);
    sessions_.erase(std::remove(sessions_.begin(), sessions_.end(), session),
                    sessions_.end());
  }
 private:
  mutex sessions_lock_;
  std::vector<CudaGraphSession*> sessions_ GUARDED_BY(sessions_lock_);
};

class CudaGraphSessionRegistrar {
 public:
  CudaGraphSessionRegistrar() {
    SessionFactory::Register("CUDA_GRAPH_SESSION",
                             new CudaGraphSessionFactory());
  }
};
static CudaGraphSessionRegistrar registrar;

const std::set<DataType> CudaGraphSession::unsupported_types = {
    DT_INT32, DT_STRING, DT_RESOURCE, DT_VARIANT};

CudaGraphSession::CudaGraphSession(const SessionOptions& options,
                                   const DeviceMgr* device_mgr,
                                   CudaGraphSessionFactory* const factory)
    : device_mgr_(device_mgr),
      factory_(factory),
      tensor_holder_(new TensorHolder()) {
  SessionOptions new_options = options;
  new_options.target = "";
  options_ = new_options.config.cuda_graph_options();
  new_options.config.mutable_gpu_options()->set_allow_growth(false);
  session_.reset(dynamic_cast<DirectSession*>(NewSession(new_options)));
  assert(session_);
}

CudaGraphSession::~CudaGraphSession() {
  if (!closed_) Close().IgnoreError();
  if (device_mgr_) {
    for (auto d : device_mgr_->ListDevices()) {
      d->ClearResourceMgr();
    }
  }
  ctx_dispatcher_map_.clear();
  tensor_holder_.reset(nullptr);
  AllocatorStats stats;
  if (gpu_allocator_) {
    gpu_allocator_->GetStats(&stats);
    LOG(INFO) << "cuda graph destory mem total:" << stats.bytes_limit << " used:" << stats.bytes_in_use;
  }
}

Status CudaGraphSession::Close() {
  session_->Close();
  {
    mutex_lock l(closed_lock_);
    if (closed_) return ::tensorflow::Status::OK();
    closed_ = true;
  }
  if (factory_ != nullptr) factory_->Deregister(this);
  if (feed_gpu_fetch_gpu_) {
      TF_RETURN_IF_ERROR(session_->ReleaseCallable(feed_gpu_fetch_gpu_));
  }
  return ::tensorflow::Status::OK();
}

bool CudaGraphSession::IsCUDATensor(const Tensor* t) {
  cudaPointerAttributes attributes;
  cudaError_t err =
      cudaPointerGetAttributes(&attributes, t->tensor_data().data());
  if (err == cudaErrorInvalidValue) return false;
  CHECK_EQ(cudaSuccess, err) << cudaGetErrorString(err);
  return (attributes.type == cudaMemoryTypeDevice);
}

Status CudaGraphSession::InitInOutInfos(
    const GraphDef& graph, std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs) {
  inputs_info_.clear();
  input_idxs_.clear();
  inputs.clear();
  output_idxs_.clear();
  for (auto& node : graph.node()) {
    if (node.device().find("GPU") == std::string::npos) {
      return errors::Internal("node device is not gpu of: ", node.name());
    }
    if (node.op() == "Placeholder") {
      PartialTensorShape shape = node.attr().at("shape").shape();
      DataType dtype = node.attr().at("dtype").type();
      if (shape.unknown_rank()) {
        return errors::Internal("input shape is unknown of: ", node.name());
      }
      if (unsupported_types.find(dtype) != unsupported_types.end()) {
        return errors::Internal(
            "input type not supported by cuda graph: ", node.name(), dtype);
      }
      inputs_info_[node.name()] = std::make_pair(shape, dtype);
      inputs.push_back(node.name());
      input_idxs_[node.name()] = inputs.size() - 1;
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    output_idxs_[outputs[i]] = i;
  }
  return Status::OK();
}

Status CudaGraphSession::InitCudaGraphInputs(int batch_size,
        std::vector<Tensor>& inputs,
        const cudaStream_t& stream) {
  inputs.clear();
  inputs.resize(inputs_info_.size());
  for (auto& it : inputs_info_) {
    auto dtype = it.second.second;
    PartialTensorShape input_shape = it.second.first;
    TensorShape new_shape;
    if (input_shape.dim_size(0) < 0) {
      input_shape.set_dim(0, batch_size);
    }
    if (!input_shape.AsTensorShape(&new_shape)) {
      return errors::Internal("part shape convert to tensor shape failed.");
    }
    auto input = Tensor(gpu_allocator_, dtype, new_shape);
    if (dtype == DT_INT64) {
        cudaError_t err =
            cudaMemsetAsync(input.base<void>(), 0, input.TotalBytes(), stream);
        if (err != cudaSuccess) {
            return errors::Internal("init set int32 failed");
        }
    }
    inputs[input_idxs_[it.first]] = input;
  }
  return Status::OK();
}

Status CudaGraphSession::InitCallableInputs(
    const std::vector<std::pair<string, Tensor> >& inputs,
    std::vector<Tensor>& callable_inputs) {
  if (inputs.size() != input_idxs_.size()) {
    return errors::Internal("input size is not equal to require.");
  }
  callable_inputs.clear();
  callable_inputs.resize(inputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    auto it = input_idxs_.find(inputs[i].first);
    if (it == input_idxs_.end()) {
      return errors::Internal("can not find input idx: ", inputs[i].first);
    }
    if (IsCUDATensor(&(inputs[i].second))) {
      callable_inputs[it->second] = inputs[i].second;
    } else {
      Tensor input(gpu_allocator_, inputs[i].second.dtype(),
                   inputs[i].second.shape());
      CopyTensorData(&(inputs[i].second), &input, input.TotalBytes(),
                     capturing_stream_);
      callable_inputs[it->second] = input;
    }
  }
  cudaStreamSynchronize(capturing_stream_);
  return Status::OK();
}

Status CudaGraphSession::InitGPUInfo(const DeviceMgr* device_manager) {
  std::vector<Device*> devices = device_manager->ListDevices();
  for (auto* d : devices) {
    if (d->name().find("GPU") != std::string::npos) {
      auto gpu = dynamic_cast<BaseGPUDevice*>(d);
      if (!gpu) {
        return errors::Internal("cast gpu device failed");
      }
      capturing_stream_ = gpu->GetSingleStream();
      gpu_allocator_ = gpu->GetGPUAllocator();
      gpu_device_name_ = gpu->name();
      break;
    }
  }
  if (!capturing_stream_) {
    return errors::Internal("get stream failed");
  }
  auto allocator = dynamic_cast<GPUBFCAllocator*>(gpu_allocator_);
  if (!gpu_allocator_ || !allocator) {
    return errors::Internal("get gpu allocator failed");
  }
  {
      tf_shared_exclusive_lock lock(DirectSession::capture_run_mu_, false);
      allocator->ExtendAll(32, 1 << 26);
  }
  host_allocator_ = GPUProcessState::singleton()->GetCUDAHostAllocator(0);
  return Status::OK();
}

Status CudaGraphSession::CaptureCudaGraph(
    int batch_size, const Session::CallableHandle& handle) {
  assert(session_);
  CudaGraphContextPtr cuda_ctx(new CudaGraphContext());
  Status s;
  cudaError_t ret;
  TF_RETURN_IF_ERROR(InitCudaGraphInputs(batch_size, cuda_ctx->inputs, cuda_ctx->stream));
  {
    tf_shared_exclusive_lock lock(DirectSession::capture_run_mu_, true);
    ret =
        cudaStreamBeginCapture(capturing_stream_, cudaStreamCaptureModeGlobal);
    if (ret != cudaSuccess) {
      return errors::Internal("cuda graph begin capture failed.");
    }
    // capture run
    s = session_->RunCallable(handle, cuda_ctx->inputs,
                                             &(cuda_ctx->outputs), nullptr);
    ret = cudaStreamEndCapture(capturing_stream_, &(cuda_ctx->cuda_graph));
    if (!s.ok()) {
        return errors::Internal("cuda graph capture run failed:", s.ToString());
    }
    if (ret != cudaSuccess) {
      return errors::Internal("cuda graph end capture failed.");
    }
    // tmp not sure create and instantiate can parallel
    ret = cudaGraphInstantiate(&(cuda_ctx->cuda_graph_exec),
                               cuda_ctx->cuda_graph, NULL, NULL, 0);
    if (ret != cudaSuccess) {
      return errors::Internal("cuda graph create execute instance failed.");
    }
    ret = cudaStreamCreate(&(cuda_ctx->stream));

    if (ret != cudaSuccess) {
      return errors::Internal("cuda stream create error");
    }
  }
  if (!ctx_dispatcher_map_[batch_size]) {
    ctx_dispatcher_map_[batch_size].reset(new CudaGraphDispatcher());
  }
  ctx_dispatcher_map_[batch_size]->PutContext(cuda_ctx);
  return Status::OK();
}

Status CudaGraphSession::CopyTensorData(const Tensor* from, Tensor* to,
                                        size_t dataSize,
                                        const cudaStream_t& stream) {
  if (from->dtype() != to->dtype()) {
    return errors::Internal("cuda input type not consist with input");
  }
  auto fromShape = from->shape();
  auto toShape = to->shape();
  TF_RETURN_IF_ERROR(CheckShape(fromShape, toShape));
  if (dataSize > from->TotalBytes() || dataSize > to->TotalBytes()) {
    return errors::Internal("copy size large than to tensor size");
  }
  cudaMemcpyKind copyKind;
  if (!IsCUDATensor(from)) {
    copyKind = cudaMemcpyHostToDevice;
  } else if (!IsCUDATensor(to)) {
    copyKind = cudaMemcpyDeviceToHost;
  } else {
    copyKind = cudaMemcpyDeviceToDevice;
  }
  auto err = cudaMemcpyAsync(to->base<void>(), from->base<void>(), dataSize,
                             copyKind, stream);
  if (err != cudaSuccess) {
    return errors::Internal("copy output failed");
  }
  return Status::OK();
}

Status CudaGraphSession::CopyCudaGraphInput(
    const std::vector<Tensor>& cuda_graph_inputs,
    const std::vector<std::pair<std::string, Tensor> >& inputs,
    const cudaStream_t& stream) {
  for (auto& input : inputs) {
    string name = input.first;
    Tensor t = input.second;
    auto it = input_idxs_.find(name);
    if (it == input_idxs_.end()) {
      return errors::Internal("input name not consisit ", name);
    } else {
      assert(it->second < cuda_graph_inputs.size());
      Tensor dst = cuda_graph_inputs[it->second];
      auto num_bytes = dst.TotalBytes();
      if (t.dtype() == DT_INT64) {
        cudaError_t err =
            cudaMemsetAsync(dst.base<void>(), 0, num_bytes, stream);
        if (err != cudaSuccess) {
          return errors::Internal("set int32 failed");
        }
      }
      TF_RETURN_IF_ERROR(CopyTensorData(&t, &dst, t.TotalBytes(), stream));
    }
  }
  return Status::OK();
}

Status CudaGraphSession::CopyCudaOutput(
    const std::vector<Tensor>& cuda_outputs,
    const std::vector<string>& output_tensor_names,
    std::vector<Tensor>* outputs, int batch_size, const cudaStream_t& stream) {
  outputs->clear();
  for (auto& name : output_tensor_names) {
    auto it = output_idxs_.find(name);
    if (it == output_idxs_.end()) {
      return errors::Internal("not found output: ", name);
    }
    assert(it->second < cuda_outputs.size());
    Tensor cuda_output = cuda_outputs[it->second];
    TensorShape shape = cuda_output.shape();
    if (batch_size > shape.dim_size(0)) {
      return errors::Internal(
          "copy output failed batch size larget than cuda out");
    }
    shape.set_dim(0, batch_size);
    Tensor output;
    if (options_.output_on_cpu()) {
        output = Tensor(host_allocator_, cuda_output.dtype(), shape);
    } else {
        output = Tensor(gpu_allocator_, cuda_output.dtype(), shape);
        cudaStreamSynchronize(capturing_stream_);
    }
    TF_RETURN_IF_ERROR(
        CopyTensorData(&cuda_output, &output, output.TotalBytes(), stream));
    outputs->emplace_back(output);
  }
  cudaStreamSynchronize(stream);
  return Status::OK();
}

Status CudaGraphSession::InitCallableOptions(
    CallableOptions& opts, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs) {
  for (auto& name : inputs) {
    opts.add_feed(name);
    opts.mutable_feed_devices()->insert({name, gpu_device_name_});
  }
  for (auto& name : outputs) {
    opts.add_fetch(name);
    opts.mutable_fetch_devices()->insert({name, gpu_device_name_});
  }
  // on cuda graph capture mode, sync with cuda call is not supported.
  opts.set_fetch_skip_sync(true);
  return Status::OK();
}

template <class Shape>
Status CudaGraphSession::CheckShape(const Shape& fromShape,
                                    const TensorShape& toShape) {
  if (fromShape.dims() != toShape.dims()) {
    return errors::Internal("cuda input dim size not consist with input:",
                            fromShape.dims(), ":", toShape.dims());
  }
  for (int d = 1; d < toShape.dims(); d++) {
    if (fromShape.dim_size(d) != toShape.dim_size(d)) {
      return errors::Internal("cuda input dim not consist with input at ", d);
    }
  }
  return Status::OK();
}

Status CudaGraphSession::CheckInputsInfo(
    const std::vector<std::pair<string, Tensor> >& inputs, int& batch_size) {
  batch_size = -1;
  std::vector<bool> visited(input_idxs_.size(), false);
  for (auto& input : inputs) {
    const Tensor& input_tensor = input.second;
    auto info_it = inputs_info_.find(input.first);
    if (info_it == inputs_info_.end()) {
        return errors::Internal("do not need input name: ", input.first);
    }
    auto idx_it = input_idxs_.find(input.first);
    if (idx_it == input_idxs_.end() || idx_it->second >= visited.size()) {
      return errors::Internal("find input idx is not right: ", input.first);
    }
    visited[idx_it->second] = true;
    if (input_tensor.dtype() != info_it->second.second) {
      return errors::Internal("data type mismatch for ", input.first);
    }
    const PartialTensorShape& capture_shape = info_it->second.first;
    TF_RETURN_IF_ERROR(CheckShape(capture_shape, input_tensor.shape()));
    int capture_dim_0 = capture_shape.dim_size(0);
    int input_dim_0 = input_tensor.shape().dim_size(0);
    if (input_dim_0 < 0) {
      return errors::Internal("input dim 0 size < 0");
    }
    if (capture_dim_0 > 0 && input_dim_0 != capture_dim_0) {
      return errors::Internal("capture dim 0 not consist with input");
    } else if (capture_dim_0 < 0) {
      if (batch_size < 0) {
        batch_size = input_dim_0;
      }
      if (batch_size != input_dim_0) {
        return errors::Internal("input item batch not consist, node:",
                                input.first);
      }
    }
  }
  for (size_t i = 0; i < visited.size(); ++i) {
    if (!visited[i]) {
      return errors::Internal("lack inputs num:", i);
    }
  }
  return Status::OK();
}

CudaGraphSession::CudaGraphContextPtr CudaGraphSession::GetCudaGraphContext(
    int batch_size) {
  auto it = ctx_dispatcher_map_.lower_bound(batch_size);
  if (it == ctx_dispatcher_map_.end() || closed_) {
    return nullptr;
  }
  return it->second->GetContext();
}

void CudaGraphSession::PutBackCudaGraphContext(
    int batch_size, CudaGraphSession::CudaGraphContextPtr& ctx) {
  auto it = ctx_dispatcher_map_.lower_bound(batch_size);
  assert(it != ctx_dispatcher_map_.end());
  it->second->PutContext(ctx);
}

Status CudaGraphSession::Create(const GraphDef& graph) {
  assert(session_);
  if (has_inited_.test_and_set()) {
    return errors::Internal("can not create second time");
  }
  std::vector<std::string> input_names;
  TF_RETURN_IF_ERROR(InitGPUInfo(device_mgr_.get()));
  auto output_names = std::vector<std::string>(options_.outputs().begin(),
                                               options_.outputs().end());

  TF_RETURN_IF_ERROR(InitInOutInfos(graph, input_names, output_names));
  TF_RETURN_IF_ERROR(session_->Create(graph));
  std::vector<Tensor> tmp_input;
  std::vector<Tensor> tmp_output;
  TF_RETURN_IF_ERROR(InitCudaGraphInputs(1, tmp_input, capturing_stream_));
  cudaStreamSynchronize(capturing_stream_);
  // init CallableHandle
  CallableOptions opts;

  TF_RETURN_IF_ERROR(InitCallableOptions(opts, input_names, output_names));
  // first normal init run
  {
    tf_shared_exclusive_lock lock(DirectSession::capture_run_mu_, false);
    TF_RETURN_IF_ERROR(session_->MakeCallable(opts, &feed_gpu_fetch_gpu_));
    TF_RETURN_IF_ERROR(session_->RunCallable(feed_gpu_fetch_gpu_, tmp_input,
                                             &tmp_output, nullptr));
  }
  if (options_.batchs_size() <= 0 ||
      options_.batchs_size() != options_.copies_size()) {
    return errors::Internal(
        "capture batch size less than 0 or not euqal to copies");
  }
  session_->SetTensorHolder(tensor_holder_.get());
  for (int i = 0; i < options_.batchs_size(); ++i) {
    for (int j = 0; j < options_.copies(i); ++j) {
      TF_RETURN_IF_ERROR(
          CaptureCudaGraph(options_.batchs(i), feed_gpu_fetch_gpu_));
    }
  }
  session_->SetTensorHolder(nullptr);
  inited_succ_ = true;
  AllocatorStats stats;
  gpu_allocator_->GetStats(&stats);
  LOG(INFO) << "cuda graph create mem total:" << stats.bytes_limit << " used:" << stats.bytes_in_use;
  return Status::OK();
}

Status CudaGraphSession::RunCudaGraph(
    const std::vector<std::pair<string, Tensor> >& inputs,
    const std::vector<string>& output_names, std::vector<Tensor>* outputs,
    const CudaGraphSession::CudaGraphContext* ctx, int batch_size) {
  TF_RETURN_IF_ERROR(CopyCudaGraphInput(ctx->inputs, inputs, ctx->stream));
  cudaError_t ret = cudaGraphLaunch(ctx->cuda_graph_exec, ctx->stream);
  if (ret != cudaSuccess) {
    return errors::Internal("cuda graph launch failed with cudaError:", ret);
  }
  TF_RETURN_IF_ERROR(CopyCudaOutput(ctx->outputs, output_names, outputs,
                                    batch_size, ctx->stream));
  return Status::OK();
}

Status CudaGraphSession::Run(
    const std::vector<std::pair<string, Tensor> >& inputs,
    const std::vector<string>& output_names,
    const std::vector<string>& target_nodes, std::vector<Tensor>* outputs) {
  if (target_nodes.size()) {
    return errors::Internal("cuda graph session run not support target nodes");
  }
  int batch_size = -1;
  if (!inited_succ_) {
    return errors::Internal("please make sure create succeed before run");
  }
  TF_RETURN_IF_ERROR(CheckInputsInfo(inputs, batch_size));
  auto ctx = GetCudaGraphContext(batch_size);
  if (!ctx) {
    // available batch size not found, we report error here.
    // user could retry with normal session run.
    return errors::Internal("can not found match batch:", batch_size);
  }
  auto s = RunCudaGraph(inputs, output_names, outputs, ctx.get(), batch_size);
  PutBackCudaGraphContext(batch_size, ctx);
  return s;
}

Status CudaGraphSession::Run(
    const RunOptions& run_options,
    const std::vector<std::pair<string, Tensor> >& inputs,
    const std::vector<string>& output_names,
    const std::vector<string>& target_nodes, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata) {
  if (!inited_succ_) {
    return errors::Internal("please make sure create succeed before run");
  }
  if (target_nodes.size()) {
    return errors::Internal("cuda graph session run not support target nodes");
  }
  cudaStreamSynchronize(capturing_stream_);
  auto s = Run(inputs, output_names, target_nodes, outputs);
  if (!s.ok()) {
    if (run_options.cuda_graph_use_back_up()) {
      LOG(WARNING) << "cuda graph run on back up because: " << s.ToString();
      std::vector<Tensor> callable_inputs;
      TF_RETURN_IF_ERROR(InitCallableInputs(inputs, callable_inputs));
      std::vector<Tensor> tmp_outputs;
      tf_shared_exclusive_lock lock(DirectSession::capture_run_mu_, false);
      TF_RETURN_IF_ERROR(session_->RunCallable(
          feed_gpu_fetch_gpu_, callable_inputs, &tmp_outputs, nullptr));
      for (auto &tensor: tmp_outputs) {
          if (options_.output_on_cpu()) {
              Tensor output(host_allocator_, tensor.dtype(), tensor.shape());
              TF_RETURN_IF_ERROR(
                      CopyTensorData(&tensor, &output, output.TotalBytes(), capturing_stream_));
              outputs->emplace_back(output);
          } else {
              outputs->emplace_back(tensor);
          }
      }
      if (options_.output_on_cpu()) {
          cudaStreamSynchronize(capturing_stream_);
      }
      run_metadata->set_cuda_graph_fallback_used(true);
      return Status::OK();
    }
  }
  return s;
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
