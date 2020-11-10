#include "tensorflow/core/kernels/blaze_predictor.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
BlazePredictor::BlazePredictor(OpKernelConstruction* ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("input_names", &input_names_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_names", &output_names_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("graph_def", &graph_def_str_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("blaze_option_path", &blaze_option_path_));
  OP_REQUIRES_OK(ctx, InitSession(ctx));
}

Status BlazePredictor::InitSession(OpKernelConstruction* ctx) {
  TF_RETURN_IF_ERROR(ReadTextProto(Env::Default(), blaze_option_path_, &blaze_run_options_));

  SessionOptions options;
  options.config.MergeFrom(blaze_run_options_.config_proto());

  if (!protobuf::TextFormat::ParseFromString(graph_def_str_, &graph_def_)) {
    return errors::InvalidArgument("parse ", graph_def_str_, " to protobuf failed");
  }
  auto status = NewSession(options, &session_);
  if (!status.ok()) {
    LOG(ERROR) << "create session failed";
    return status;
  }

  std::string request_device = ctx->def().device();
  std::string device_name = "/job:localhost/replica:0/task:0" + request_device;

  SetDeviceInGraphDef(device_name, &graph_def_);
  status = session_->Create(graph_def_);
  if (!status.ok()) {
    LOG(ERROR) << "create session with GraphDef failed " << status.ToString();
    return status;
  }

  CallableOptions callable_options;
  for (const auto& input : input_names_) {
    callable_options.add_feed(input);
    callable_options.mutable_feed_devices()->insert({input, device_name});
  }

  for (const auto& output : output_names_) {
    callable_options.add_fetch(output);
    callable_options.mutable_fetch_devices()->insert({output, device_name});
  }
  callable_options.set_fetch_skip_sync(true);
  LOG(INFO) << "create session with callable options " <<
      callable_options.DebugString();
  status = session_->MakeCallable(callable_options, &handle_);
  return status;
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
  RunMetadata metadata;
  OP_REQUIRES_OK(ctx, session_->RunCallable(handle_, inputs, &outputs, &metadata));
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
