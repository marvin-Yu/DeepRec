#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#endif
#include "tensorflow/core/kernels/blaze_predictor.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/hydra_base64_util.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/gpu_utils.h"
using tensorflow::se::Event;
#endif

#include <mutex>
#include <condition_variable>

namespace tensorflow {
const int kBlazeStartStepId = 1024;
const std::string kCpuDeviceName = "/job:localhost/replica:0/task:0/device:CPU:0";

mutex BlazePredictor::session_mu_;
BlazePredictor::SessionMap BlazePredictor::session_map_;
mutex BlazePredictor::log_mu_;

BlazePredictor::BlazePredictor(OpKernelConstruction* ctx) : device_type_(ctx->device_type().type()) {
  ReadInt64FromEnvVar("BLAZE_LOG_LEVEL", 0, &log_level_);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("input_names", &input_names_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_names", &output_names_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("graph_def", &graph_def_str_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("blaze_option_path", &blaze_option_path_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("InT", &input_types_));
  OP_REQUIRES_OK(ctx, ParseAttr(ctx->def().device()));
  OP_REQUIRES_OK(ctx, InitSplitConf(ctx));
  ctx_ = ctx;
}

BlazePredictor::BlazePredictor(const std::vector<std::string>& input_names,
                          const std::vector<std::string>& output_names,
                          const GraphDef& graph_def, const std::string& device,
                          const BlazeKernelOptions& options, const string& device_string,
                          const std::vector<DataType>& input_types,
                          OpKernelConstruction* ctx) :
    input_names_(input_names), output_names_(output_names),
    graph_def_(graph_def), request_device_(device),
    blaze_run_options_(options), device_type_(device_string),
    input_types_(input_types), ctx_(ctx) {
  need_trace_ = false;
  ReadInt64FromEnvVar("BLAZE_LOG_LEVEL", 0, &log_level_);
  // rewrite HugeConst
  std::string root_path;
  auto status = ctx_->GetAttr("_extra_conf_root_path", &root_path);
 	if (status.ok()) {
    LOG(INFO) << "get _extra_conf_root_path" << root_path;
 	  for (int i = 0; i < graph_def_.node_size() ;i++) {
      auto node = graph_def_.mutable_node(i);
      if (node->op() == "HugeConst") {
        auto* attr_map = node->mutable_attr();
        if (attr_map != nullptr) {
          auto it = attr_map->find("path");
          if (it != attr_map->end()) {
            std::string ad_vec_path = root_path + '/' + it->second.s();
            it->second.set_s(ad_vec_path);
            LOG(INFO) << "Update path attr of HugeConst op " << node->name() << " to " << ad_vec_path;
          }
        }
      }
    }
  } else { LOG(INFO) << "not get _extra_conf_root_path" << root_path; }
  
  OP_REQUIRES_OK(ctx, InitSplitConf(ctx));
}

BlazePredictor::~BlazePredictor() {
  if (!session_key_.empty()) {
    mutex_lock l(session_mu_);
    auto it = session_map_.find(session_key_); 
    if (it == session_map_.end()) {
      LOG(ERROR) << "release session not found in static map, should not happen";
      return;
    }
    if (--(it->second.count) == 0) {
      session_map_.erase(it);
    }
 	}
}

Status BlazePredictor::ParseAttr(const std::string& device) {
  if (!ReadTextProto(Env::Default(), blaze_option_path_,
                     &blaze_run_options_).ok()) {
    VLOG(0) << "Parse blaze options from file failed, try as readable string";
  } else {
    if (!::tensorflow::protobuf::TextFormat::ParseFromString(
            blaze_option_path_, &blaze_run_options_)) {
      return errors::Internal("parse proto from ", blaze_option_path_,  " failed");
    }
  }

  if (!ReadTextProto(Env::Default(), graph_def_str_, &graph_def_).ok()) {
    if (!ReadBinaryProto(Env::Default(), graph_def_str_, &graph_def_).ok()) {
      LOG(ERROR) << "Parse graph from " << graph_def_str_ << " failed";
      return errors::Internal("Parse graph from ", graph_def_str_, " failed");
    }
  }
  
  if (device.size() == 0) {
    return errors::Internal("ctx device not set");
  }
  request_device_ = device;

  return Status::OK();
}

Status BlazePredictor::GenSessionOptions(SessionOptions& options) {
  options.config.MergeFrom(blaze_run_options_.config_proto());
  //disable caller thread
  options.config.set_force_run_in_caller_thread(false);
  options.config.set_is_blaze(true);
  return Status::OK();
}

Status BlazePredictor::PrepareGraph(GraphDef& graph_def) {
  const char* const kDevicePrefix = "/job:localhost/replica:0/task:0";
  const char* const kBlazeDevicePrefix = "/device:";
  auto st = ctx_->GetAttr(kBlazeRealDevice, &blaze_real_deive_);
  if (!st.ok()) {
    device_ = kDevicePrefix + request_device_;
  } else {
    if (blaze_real_deive_.rfind(kBlazeDevicePrefix, 0) != 0) {
      return errors::Internal("node attr ", kBlazeRealDevice,
          " : ", blaze_real_deive_, " not startswith ", kBlazeDevicePrefix);
    }
    device_ = kDevicePrefix + blaze_real_deive_;
  }

  LOG(INFO) << "BlazePredictor will use device " << device_;
  graph_def = graph_def_;
  SetDeviceInGraphDef(device_, &graph_def);
  SetCPUDeviceInGraphDef(kCpuDeviceName, &graph_def);

  return Status::OK();
}

Status BlazePredictor::MakeCallable() {
  CallableOptions callable_options;
  TF_RETURN_IF_ERROR(PrepareCallableOptions(callable_options));
  LOG(INFO) << "create session with callable options " <<
        callable_options.DebugString();
  return session_->MakeCallable(callable_options, &handle_);
}

Status BlazePredictor::PrepareCallableOptions(CallableOptions &callable_options) {
  std::set<std::string> cpu_inputs;
  for (const auto& node : graph_def_.node()) {
    if (node.op() == "Placeholder") {
      auto it = node.attr().find("dtype");
      if (it != node.attr().end()) {
        if (it->second.type() == DT_INT32 || it->second.type() == DT_UINT32) {
          cpu_inputs.insert(node.name());
        }
      }
      if (node.device().find("/device:CPU:0") != std::string::npos) {
        cpu_inputs.insert(node.name());
      }
    }
  }
  for (const auto& input : input_names_) {
    if (cpu_inputs.find(input) == cpu_inputs.end()) {
      callable_options.add_feed(input);
      callable_options.mutable_feed_devices()->insert({input, device_});
      copyable_.push_back(true);
    } else {
      callable_options.add_feed(input);
      callable_options.mutable_feed_devices()->insert({input, kCpuDeviceName});
      copyable_.push_back(false);
    }
  }

  for (const auto& output : output_names_) {
    callable_options.add_fetch(output);
    callable_options.mutable_fetch_devices()->insert({output, device_});
  }
  callable_options.set_fetch_skip_sync(true);
  if (blaze_run_options_.run_mode() == BlazeKernelOptions::TRACE) {
    need_trace_ = true;
    callable_options.mutable_run_options()->set_trace_tensor_infos(true);
  } else if (blaze_run_options_.run_mode() == BlazeKernelOptions::TIMELINE) {
    need_trace_ = true;
    callable_options.mutable_run_options()->set_trace_level(RunOptions::SOFTWARE_TRACE);
  }
  return Status::OK();
}

Status BlazePredictor::Warmup() {
  return Status::OK();
}

Status BlazePredictor::InitSession() {
  TF_RETURN_IF_ERROR(PrepareData());

  SessionOptions options;
  *(options.config.mutable_gpu_options()) = BlazeConfSingleton::GetInstance()
      ->GetConfig().gpu_options();

  options.config.MergeFrom(blaze_run_options_.config_proto());
  TF_RETURN_IF_ERROR(GenSessionOptions(options));
  auto status = ctx_->GetAttr("_session_key", &session_key_);
 	if (status.ok()) {
 	  LOG(INFO) << "_session_key detected, static Session enabled";
 	} else {
 	  LOG(INFO) << "_session_key not detected, won't use static predictor map";
 	  session_key_.clear();
  }
  {
    GraphDef graph_def;
    TF_RETURN_IF_ERROR(PrepareGraph(graph_def));
 	  mutex_lock l(session_mu_);
    auto it = session_map_.find(session_key_);
    if (it == session_map_.end()) {
      VLOG(0) << "create session with config " << options.config.DebugString();
      session_ = std::shared_ptr<Session>(NewSession(options));
      if (session_ == nullptr) {
        LOG(ERROR) << "create session failed";
        return errors::Internal("Create session failed");
      }
      auto dir_session = reinterpret_cast<DirectSession*>(session_.get());
      dir_session->SetStepInitId(kBlazeStartStepId);
      LOG(INFO) << "Blaze start with step id " << kBlazeStartStepId;
      LOG(INFO) << "Creat session succ " << this;

      auto status = session_->Create(graph_def);
      if (!status.ok()) {
        LOG(ERROR) << "create session with GraphDef failed " << status.ToString();
        return status;
      }
      TF_RETURN_IF_ERROR(MakeCallable());
      LOG(INFO) << "MakeCallable succ " << this;
      if (!session_key_.empty()) {
        session_map_.emplace(session_key_, SessionTuple(session_, 1, handle_));
      }
    } else {
      CallableOptions callable_options;
      TF_RETURN_IF_ERROR(PrepareCallableOptions(callable_options));
      session_ = it->second.session;
      it->second.count++;
      handle_ = it->second.handle;
    }
    TF_RETURN_IF_ERROR(SetDeviceInfo(ctx_));

    auto warm_status = Warmup();
    if (warm_status != Status::OK()) {
      return warm_status;
    }
    return  Status::OK();
  }
}

Status BlazePredictor::Compute(OpKernelContext* ctx) {
  if (need_split_) { return ComputeSplited(ctx); }
  if (log_level_ > 0) RawInputsDebugLogging(ctx);

  int num_inputs = ctx->num_inputs();
  if (num_inputs != input_names_.size()) {
    return errors::Internal("ctx input size ", num_inputs,
        " != ", input_names_.size());
  }
  if (ctx->num_outputs() != output_names_.size()) {
    return errors::Internal("ctx output size ", ctx->num_outputs(),
        " != ", output_names_.size());
  }

  std::vector<Tensor> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(ctx->input(i));
  }

  std::vector<Tensor> outputs;

  std::vector<Tensor> real_inputs(inputs.size());
  TF_RETURN_IF_ERROR(PrepareInputs(inputs, &real_inputs, ctx));

  if (need_trace_ || (ctx->traced_infos() && ctx->traced_infos()->enable_sampling_prof_stats)) {
    RunMetadata metadata;
    TF_RETURN_IF_ERROR(session_->RunCallable(handle_, real_inputs, &outputs, &metadata, ctx->stream_id()));
    if (ctx->traced_infos() && ctx->traced_infos()->enable_sampling_prof_stats) {
      ctx->traced_infos()->UpdateProfStats(&metadata);
    }
    if (need_trace_) {
      DumpFile(metadata);
    }
  } else {
    TF_RETURN_IF_ERROR(session_->RunCallable(handle_, real_inputs, &outputs, nullptr, ctx->stream_id()));
  }

  std::vector<Tensor> real_outputs(outputs.size());
  TF_RETURN_IF_ERROR(PrepareOutputs(outputs, &real_outputs, ctx));
  for (int i = 0; i < real_outputs.size(); ++i) {
    ctx->set_output(i, real_outputs[i]);
  }
  return Status::OK();
}

Status BlazePredictor::ComputeSplited(OpKernelContext* ctx) {
  if (log_level_ > 0) RawInputsDebugLogging(ctx);

  int num_inputs = ctx->num_inputs();
  if (num_inputs != input_names_.size()) {
    return errors::Internal("ctx input size ", num_inputs,
        " != ", input_names_.size());
  }
  if (ctx->num_outputs() != output_names_.size()) {
    return errors::Internal("ctx output size ", ctx->num_outputs(),
        " != ", output_names_.size());
  }

  std::vector<Tensor> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(ctx->input(i));
  }

  std::vector<std::vector<Tensor>> splited_inputs;
  TF_RETURN_IF_ERROR(SplitInputs(inputs, splited_inputs));
  std::vector<std::vector<Tensor>> sp_outputs;
  sp_outputs.resize(splited_inputs.size());

  std::mutex m;
  std::condition_variable cv;
  std::shared_ptr<std::atomic<int>> barrier_shared = std::make_shared<std::atomic<int>>(0);
  bool run_ok = true;
  int total = splited_inputs.size();
  for (int i = 0; i < splited_inputs.size(); ++i) {
    auto& inputs = splited_inputs[i];

    auto func = [this, ctx, &inputs, &cv, barrier_shared, &run_ok, i, &sp_outputs, total, &m]() {
      std::vector<Tensor> outputs;
      std::vector<Tensor> real_inputs(inputs.size());
      Status st;
      st = PrepareInputs(inputs, &real_inputs, ctx);
#define RETURN_AND_SUB() \
      if (!st.ok()) { \
        std::unique_lock<std::mutex> lock(m); \
        VLOG(0) << st.ToString(); \
        run_ok = false; \
        if(barrier_shared->fetch_add(1) == total -1) { \
          cv.notify_all(); \
          return; \
        } \
      }
      RETURN_AND_SUB();
      if (need_trace_ || (ctx->traced_infos() && ctx->traced_infos()->enable_sampling_prof_stats)) {
        RunMetadata metadata;
        st = session_->RunCallable(this->handle_, real_inputs, &outputs, &metadata, ctx->stream_id());
        if (ctx->traced_infos() && ctx->traced_infos()->enable_sampling_prof_stats) {
          ctx->traced_infos()->UpdateProfStats(&metadata);
        }
        if (need_trace_) {
          DumpFile(metadata, i);
        }
      } else {
        st = session_->RunCallable(this->handle_, real_inputs, &outputs, nullptr, ctx->stream_id());
      }
      RETURN_AND_SUB();
      std::vector<Tensor> real_outputs(outputs.size());
      st = this->PrepareOutputs(outputs, &real_outputs, ctx);
      RETURN_AND_SUB();
      sp_outputs[i] = std::move(outputs);
      std::unique_lock<std::mutex> lock(m);
      if(barrier_shared->fetch_add(1) == total -1) {
        cv.notify_all();
        return;
      }
    };
    split_thread_pool_->Schedule(std::move(func));
  }
  auto lock = std::unique_lock<std::mutex>(m);
  cv.wait(lock, [&]() { return barrier_shared->load() == total; });
  if (!run_ok) {
    return errors::Internal("split run fail");
  }
  std::vector<Tensor> mg_tensors;
  TF_RETURN_IF_ERROR(MergeOutputs(ctx, sp_outputs, mg_tensors));
  for (int i = 0; i < mg_tensors.size(); ++i) {
    ctx->set_output(i, mg_tensors[i]);
  }
  return Status::OK();
}

void BlazePredictor::SetDeviceInGraphDef(const std::string device_name,
                                         GraphDef* graph_def) {
  VLOG(2) << "Before setting device: \n" << graph_def->DebugString();
  int node_size = graph_def->node_size();
  for (int i = 0; i < node_size; i++) {
    NodeDef* node = graph_def->mutable_node(i);
    if (node->device() == "/device:CPU:0") {
      VLOG(1) << "node " << node->name() << " device /device:CPU:0, do not overwrite to GPU";
      continue;
    }
    node->set_device(device_name);
  }
  VLOG(2) << "After setting device: \n" << graph_def->DebugString();
}

void BlazePredictor::SetCPUDeviceInGraphDef(const std::string device_name,
                                            GraphDef* graph_def) {
  VLOG(2) << "Before setting device: \n" << graph_def->DebugString();
  int node_size = graph_def->node_size();
  for (int i = 0; i < node_size; i++) {
    NodeDef* node = graph_def->mutable_node(i);
    if (node->device() == "/device:CPU:0") {
      VLOG(1) << "node " << node->name() << " device /device:CPU:0, overwrite to " << device_name;
      node->set_device(device_name);
    }
  }
  VLOG(2) << "After setting device: \n" << graph_def->DebugString();
}

Status BlazePredictor::SetDeviceInfo(OpKernelConstruction* ctx) {
  auto st = ctx->GetAttr(kBlazeRealDevice, &blaze_real_deive_);
  if (!st.ok()) {
    VLOG(0) << "Blaze not set device, using " << request_device_;
    same_device_ = true;
    blaze_device_ = nullptr;
    return Status::OK();
  } else {
    DeviceNameUtils::ParsedName req_name;
    DeviceNameUtils::ParsedName blaze_name;
    if (!DeviceNameUtils::ParseFullName(request_device_, &req_name)) {
      return errors::Internal(request_device_, " parse failed");
    }

    if (!DeviceNameUtils::ParseFullName(blaze_real_deive_, &blaze_name)) {
      return errors::Internal(blaze_real_deive_, " parse failed");
    }
    
    auto req_dev = DeviceNameUtils::LocalName(request_device_);
    auto blaze_dev = DeviceNameUtils::LocalName(blaze_real_deive_);

    vgpu_id_ = blaze_name.id;
    if (req_dev != blaze_dev) {
      VLOG(0) << "req_dev: " << req_dev << "; blaze_dev: " << blaze_dev;
      if (req_name.type == "cpu" || blaze_name.type == "CPU") {
        return errors::Internal("req_dev.type: ", req_name.type,
            ", blaze_dev.type: ", blaze_name.type, " not supported");
      }
      same_device_ = false;
      const DeviceMgr* mgr = nullptr;
      TF_RETURN_IF_ERROR(session_->LocalDeviceManager(&mgr));
      if (mgr == nullptr) {
        return errors::Internal("DeviceMgr not found");
      }
      TF_RETURN_IF_ERROR(mgr->LookupDevice(blaze_dev, &blaze_device_));
      auto* dev_info = blaze_device_->tensorflow_gpu_device_info();
      if (!dev_info) {
        return errors::Internal("get gpu device info failed");
      }
      blaze_allocator_ = GetAllocator();
      if (!blaze_allocator_) {
        return errors::Internal("get gpu allocator failed");
      }
      auto stream_ = GetStream();
      if (!stream_) {
        return errors::Internal("get stream_ for ", blaze_device_, " failed" );
      }
    }
    return Status::OK();
  }
}

stream_executor::Stream* BlazePredictor::GetStream(int stream_id) const {
  #if GOOGLE_CUDA
  TfGpuId tf_gpu_id(vgpu_id_);
  // turn off multi-stream, the original stream id is 0.
  if (stream_id == -1) stream_id = 0;
  auto* se = GpuIdUtil::ExecutorForTfGpuId(tf_gpu_id, stream_id).ValueOrDie();

  if (!se) { return nullptr; }
  static tensorflow::GPUOptions gpu_options;
  auto sg = tensorflow::StreamGroupFactory::Global().GetOrCreate(
      tf_gpu_id, stream_id, se, gpu_options);
  if (!sg) {
    VLOG(0) << "get stream group failed";
    return nullptr;
  }
  return sg->compute;

  #else
    return nullptr;
  #endif
}

int BlazePredictor::GetStreamNum() const {
  if (blaze_device_ == nullptr) {
    return 1;
  }
  return blaze_device_->GetStreamNum() > 1 ? blaze_device_->GetStreamNum() : 1;
}

Allocator* BlazePredictor::GetAllocator(int stream_id) const {
  AllocatorAttributes alloc_attrs;
  alloc_attrs.set_on_host(false);
  if (stream_id == -1) {
    return blaze_device_->GetAllocator(alloc_attrs);
  }
  return blaze_device_->GetStreamDevice(stream_id)->GetAllocator(alloc_attrs);
}

void BlazePredictor::RawInputsDebugLogging(OpKernelContext* ctx) const {
  mutex_lock l(log_mu_);

  for (int i = 0; i < ctx->num_inputs(); ++i) {

    const Tensor& input = ctx->input(i);
    VLOG(1) << "input ["<<i<<"]:\n"
            << input.DebugString();

    const string& name_string = input_names_[i];

    string shape_string;
    std::stringstream stream;
    for (int d = 0; d < input.dims(); d++) {
      stream << input.dim_size(d) << " ";
    }
    stream << "(" << input.NumElements() << ")";
    shape_string = stream.str();

    string dtype_string = DataTypeString(input.dtype());

    string data_string = hydra::base64_encode((const char*)input.data(), input.TotalBytes());

    LOG(INFO) << "blaze input blob [" << i << "]:"
              << " name:" << name_string
              << " shape: " << shape_string
              << " type: " << dtype_string
              << " data: " << data_string;

  }
}

Status BlazePredictor::PrepareInputs(const std::vector<Tensor>& inputs,
  std::vector<Tensor>* real_inputs, OpKernelContext* ctx) {
  if (!same_device_) {
    return CopyTensorCPUToGPU(inputs, real_inputs, ctx);
  }
  *real_inputs = inputs;
  return Status::OK();
}

Status BlazePredictor::CopyTensorCPUToGPU(const std::vector<Tensor>& inputs,
    std::vector<Tensor>* real_inputs,
    OpKernelContext* ctx) {
  for (int i = 0; i < inputs.size(); ++i) {
    if (!copyable_[i]) {
      (*real_inputs)[i] = inputs[i];
      continue;
    }
    Tensor copyed_tensor(GetAllocator(ctx->stream_id()), inputs[i].dtype(), inputs[i].shape());
    (*real_inputs)[i] = copyed_tensor;

    const uint8* input_ptr = (uint8*)GetTensorAddress(&inputs[i]);
    uint8* real_ptr = (uint8*)GetTensorAddress(&(*real_inputs)[i]);
    uint64 input_size = GetTensorSize(&inputs[i]);
    uint64 real_size = GetTensorSize(&(*real_inputs)[i]);
    if (input_ptr == nullptr || real_ptr == nullptr) {
      return errors::Internal(
          "Error when getting input address or size");
    }
#if GOOGLE_CUDA
      auto real_dev_ptr = AsDeviceMemory(real_ptr, real_size);
      bool copy_status =
          GetStream(ctx->stream_id())->ThenMemcpy(&real_dev_ptr, input_ptr, input_size).ok();
      if (!copy_status) {
        return errors::Internal("MemcpyH2D for padding inputs failed.");
      }
      if (ctx->traced_infos()) {
        ++ctx->traced_infos()->prof_stats->pcie_h2d_times;
        ctx->traced_infos()->prof_stats->pcie_h2d_size += input_size;
      }
#else
      return errors::Internal("CUDA not suaported");
#endif
  }
  return Status::OK();
}

Status BlazePredictor::PrepareOutputs(const std::vector<Tensor>& outputs,
  std::vector<Tensor>* real_outputs, OpKernelContext* ctx) {
  if (!same_device_) {
    return CopyTensorGPUToCPU(outputs, real_outputs, ctx);
  }
  *real_outputs = outputs;
  return Status::OK();
}

Status BlazePredictor::CopyTensorGPUToCPU(const std::vector<Tensor>& gpu_tensors,
    std::vector<Tensor>* cpu_tensors,
    OpKernelContext* ctx) {
  for (int i = 0; i < gpu_tensors.size(); ++i) {
#if GOOGLE_CUDA
    TensorShape slice_to_shape = gpu_tensors[i].shape();
    const auto& tmp_tensor = gpu_tensors[i];
    uint8* tmp_ptr = (uint8*)GetTensorAddress(&tmp_tensor);
    uint64 tmp_size = GetTensorSize(&tmp_tensor);
    auto tmp_dev_ptr = AsDeviceMemory(tmp_ptr, tmp_size);
    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_on_host(true);
    alloc_attrs.set_gpu_compatible(true);
    TF_RETURN_IF_ERROR(ctx->allocate_temp(tmp_tensor.dtype(),
          tmp_tensor.shape(), &((*cpu_tensors)[i]), alloc_attrs));
    uint8* host_add = (uint8*)GetTensorAddress(&((*cpu_tensors)[i]));
    auto stream = GetStream(ctx->stream_id());
    stream->ThenMemcpy(host_add, tmp_dev_ptr, tmp_size);
    auto event = std::make_shared<Event>(stream->parent());
    if (!event->Init()) {
      LOG(ERROR) << "event init failed!";
      return errors::Internal("SliceToDynamic GPU2CPU failed event init");
    }
    stream->ThenRecordEvent(event.get());
    stream->ThenSynchronizeEvent(event.get());
    if (ctx->traced_infos()) {
      ++ctx->traced_infos()->prof_stats->pcie_d2h_times;
      ctx->traced_infos()->prof_stats->pcie_d2h_size += tmp_size;
    }
#else
    return errors::Internal("cuda not supported");
#endif
  }
  return Status::OK();
}

Status BlazePredictor::InitSplitConf(OpKernelConstruction* ctx) {
  const char* kNComm = "comm";
  need_split_ = false;
  if (blaze_run_options_.need_split()) {
    split_size_ = blaze_run_options_.split_size();
    if (split_size_ <= 0) {
      return errors::Internal("splist size <=0 ", split_size_);
    }

    if (ctx->device_type().type_string() != DEVICE_CPU) {
      return errors::Internal("blaze split does not support device ", ctx->device_type().type_string());
    }
    for (const auto& input : input_names_) {
      need_split_column_.push_back(kNComm == input ? false : true);
    }

    const int kDefaultDenseThreadsNum = 2;
    int64 dense_threads_num;
    ReadInt64FromEnvVar("BLAZE_SPLIT_THREADS_NUM", 
        kDefaultDenseThreadsNum, &dense_threads_num);
    VLOG(0) << "blaze split set thread pool size " << dense_threads_num;

    split_thread_pool_ = absl::make_unique<thread::ThreadPool>(
        Env::Default(), "blaze_split_kernel", dense_threads_num);
    need_split_ = true;
  }
  return Status::OK();
}

Status BlazePredictor::SplitInputs(std::vector<Tensor>& inputs,
   std::vector<std::vector<Tensor>>& splited_inputs) const {
  int batch_size = -1;
  // infer batchsize
  for (int i = 0; i < inputs.size(); ++i) {
    if (need_split_column_[i]) {
      int dim_0 = inputs[i].shape().dims() == 0 ? 0 : inputs[i].dim_size(0);
      if (batch_size == -1) {
        batch_size = dim_0;
      } else {
        if (batch_size != dim_0) {
          return errors::Internal("invalid input size ", batch_size, dim_0);
        }
      }
    }
  }
  // split by split_dim
  int split_count = batch_size / split_size_;
  int index = 0;
#define SPLIT_TENSOR(START, END) { \
    std::vector<Tensor> tensors; \
    tensors.reserve(inputs.size()); \
    for (int j = 0; j < inputs.size(); ++j) { \
      if (need_split_column_[j]) { \
		TensorShape shape(inputs[j].shape()); \
		shape.set_dim(0, END-START); \
		Tensor tensor(inputs[j].dtype(), shape); \
		int data_len = tensor.TotalBytes() / tensor.NumElements(); \
		int delta = inputs[j].NumElements() / inputs[j].dim_size(0); \
		auto* addr = tensor.data(); \
		auto* src = inputs[j].data() + START*delta * data_len; \
		std::memcpy(addr, src, tensor.TotalBytes()); \
        tensors.push_back(std::move(tensor)); \
      } else { \
        tensors.push_back(inputs[j]); \
      } \
    } \
    splited_inputs.push_back(std::move(tensors)); \
}

  for (int i = 0; i < split_count; ++i) {
    auto end = index+split_size_;
    SPLIT_TENSOR(index, end);
    index += split_size_;
  }

  if (index < batch_size) {
    SPLIT_TENSOR(index, batch_size);
  }
//can delete?
for (int i = 0; i < splited_inputs.size(); ++i) {
	for (auto& tensor : splited_inputs[i]) {
}
}
  return Status::OK();
}

Status BlazePredictor::MergeOutputs(OpKernelContext* ctx, 
    std::vector<std::vector<Tensor>>& sp_outputs, std::vector<Tensor>& outputs) const {
  outputs.reserve(output_names_.size());
  if (sp_outputs.size() == 0) {
    return errors::Internal("nothing calculated");
  }
  std::vector<TensorShape> all_shapes;
  all_shapes.reserve(output_names_.size());

  //generate merged shapes
  for (int i = 0; i < output_names_.size(); ++i) {
    const auto& base_shape = sp_outputs[0][i].shape();
    if (base_shape.dims() == 0) {
      return errors::Internal("0 batchsize generated");
    }
    int merge_dims = base_shape.dim_size(0);
    for (int j = 1; j < sp_outputs.size(); ++j) {
      merge_dims += (sp_outputs[j][i].dims() > 0 ? sp_outputs[j][i].dim_size(0) : 0);
    }
    TensorShape concat_shape(base_shape);
    concat_shape.set_dim(0, merge_dims);
    all_shapes.push_back(std::move(concat_shape));
  }

  //merge tensor
  for (int i = 0; i < output_names_.size(); ++i) {
    Tensor tensor;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(sp_outputs[0][i].dtype(), all_shapes[i], &tensor));
    auto* base_addr = tensor.data();
    for (int j = 0; j < sp_outputs.size(); ++j) {
      auto size = sp_outputs[j][i].TotalBytes();
      if (size > 0) {
        std::memcpy(base_addr, sp_outputs[j][i].data(), size);
      }
      base_addr += size;
    }
    outputs.push_back(tensor);
  }
  return Status::OK();
}
}
