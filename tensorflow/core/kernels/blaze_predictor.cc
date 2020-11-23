#include "tensorflow/core/kernels/blaze_predictor.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
BlazePredictor::BlazePredictor(OpKernelConstruction* ctx) : device_type_(ctx->device_type().type()) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("input_names", &input_names_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_names", &output_names_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("graph_def", &graph_def_str_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("blaze_option_path", &blaze_option_path_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("InT", &input_types_));
  OP_REQUIRES_OK(ctx, ParseAttr(ctx->def().device()));
  ctx_ = ctx;
}

Status BlazePredictor::ParseAttr(const std::string& device) {
  if (!ReadTextProto(Env::Default(), blaze_option_path_,
                     &blaze_run_options_).ok()) {
    return errors::Internal("parse proto from ", blaze_option_path_,  " failed");
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
  return Status::OK();
}

Status BlazePredictor::PrepareGraph(GraphDef& graph_def) {

  const char* const kDevicePrefix = "/job:localhost/replica:0/task:0";
  device_ = kDevicePrefix + request_device_;

  LOG(INFO) << "BlazePredictor will use device " << device_;
  graph_def = graph_def_;
  SetDeviceInGraphDef(device_, &graph_def);

  return Status::OK();
}

Status BlazePredictor::MakeCallable() {
  CallableOptions callable_options;
  for (const auto& input : input_names_) {
    callable_options.add_feed(input);
    callable_options.mutable_feed_devices()->insert({input, device_});
  }

  for (const auto& output : output_names_) {
    callable_options.add_fetch(output);
    callable_options.mutable_fetch_devices()->insert({output, device_});
  }
  callable_options.set_fetch_skip_sync(true);
  LOG(INFO) << "create session with callable options " <<
      callable_options.DebugString();
  return session_->MakeCallable(callable_options, &handle_);
}

Status BlazePredictor::Warmup() {
  return Status::OK();
}

Status BlazePredictor::InitSession() {
  TF_RETURN_IF_ERROR(PrepareData());

  SessionOptions options;
  TF_RETURN_IF_ERROR(GenSessionOptions(options));
  session_ = std::move(std::unique_ptr<Session>(NewSession(options)));
  if (session_ == nullptr) {
    LOG(ERROR) << "create session failed";
    return errors::Internal("create session failed");
  }

  GraphDef graph_def;
  TF_RETURN_IF_ERROR(PrepareGraph(graph_def));

  auto status = session_->Create(graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "create session with GraphDef failed " << status.ToString();
    return status;
  }

  TF_RETURN_IF_ERROR(MakeCallable());
  return Warmup();
}

void BlazePredictor::Compute(OpKernelContext* ctx) {
  int num_inputs = ctx->num_inputs();
  OP_REQUIRES(ctx, num_inputs == input_names_.size(),
              errors::Internal("ctx input size ", num_inputs,
                               " != ", input_names_.size()));
  OP_REQUIRES(ctx, ctx->num_outputs() == output_names_.size(),
              errors::Internal("ctx output size ", ctx->num_outputs(),
                               " != ", output_names_.size()));
  std::vector<Tensor> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(ctx->input(i));
  }

  std::vector<Tensor> outputs;
  if (ctx->prof_stats()) {
    RunMetadata metadata;
    OP_REQUIRES_OK(ctx, session_->RunCallable(handle_, inputs, &outputs, &metadata));
    ctx->prof_stats()->flops += metadata.prof_stats().flops();
  } else {
    OP_REQUIRES_OK(ctx, session_->RunCallable(handle_, inputs, &outputs, nullptr));
  }
  for (int i = 0; i < outputs.size(); ++i) {
    ctx->set_output(i, outputs[i]);
  }
  return;
}

void BlazePredictor::SetDeviceInGraphDef(const std::string device_name,
                                         GraphDef* graph_def) {
  VLOG(2) << "Before setting device: \n" << graph_def->DebugString();
  int node_size = graph_def->node_size();
  for (int i = 0; i < node_size; i++) {
    NodeDef* node = graph_def->mutable_node(i);
    node->set_device(device_name);
  }
  VLOG(2) << "After setting device: \n" << graph_def->DebugString();
}
}
