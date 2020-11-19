/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/kernels/blaze_predictor.h"
#include "tensorflow/core/kernels/blaze_xla_predictor.h"

namespace tensorflow {
class BlazeXlaOp : public OpKernel {
 public:
  explicit BlazeXlaOp(OpKernelConstruction* context);
  ~BlazeXlaOp() {
    if (predictor_) {
      delete predictor_;
    }
  }

  void Compute(OpKernelContext* context) override;

  Status ParseRunOptions(BlazeKernelOptions& run_options);
 private:
  Status ParseAttr();
  void InitPredictor(OpKernelConstruction* context);

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string blaze_option_path_;
  std::string graph_def_path_;
  std::string graph_def_str_;
  string device_string_;
  std::vector<DataType> input_types_;
  
  std::string device_;
  GraphDef graph_def_;
  BlazeKernelOptions blaze_run_options_;
  BlazePredictor* predictor_;
};

void BlazeXlaOp::InitPredictor(OpKernelConstruction* context) {
  if (blaze_run_options_.xla_compilation()) {
    predictor_ = new BlazeXlaPredictor(input_names_, output_names_,
                                       graph_def_, device_, blaze_run_options_,
                                       device_string_, input_types_, context);
  } else {
    predictor_ = new BlazePredictor(input_names_, output_names_,
                                    graph_def_, device_, blaze_run_options_,
                                    device_string_, input_types_, context);
  }
}

BlazeXlaOp::BlazeXlaOp(OpKernelConstruction* context)
    : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("input_names", &input_names_));
  OP_REQUIRES_OK(context, context->GetAttr("output_names", &output_names_));
  OP_REQUIRES_OK(context, context->GetAttr("graph_def", &graph_def_path_));
  OP_REQUIRES_OK(context, context->GetAttr("blaze_option_path", &blaze_option_path_));
  OP_REQUIRES_OK(context, context->GetAttr("InT", &input_types_));
  OP_REQUIRES_OK(context, ParseAttr());
  device_string_ = context->device_type().type_string();
  device_ = context->def().device();
  InitPredictor(context);
  OP_REQUIRES_OK(context, predictor_->InitSession());
}

Status BlazeXlaOp::ParseAttr() {
  if (!ReadTextProto(Env::Default(), blaze_option_path_,
                     &blaze_run_options_).ok()) {
    return errors::Internal("parse proto from ", blaze_option_path_,  " failed");
  }

  if (!ReadTextProto(Env::Default(), graph_def_path_,
                     &graph_def_).ok()) {
    if (!ReadBinaryProto(Env::Default(), graph_def_path_,
                       &graph_def_).ok()) {
      return errors::Internal("parse proto from ", graph_def_path_,  " failed");
    }
  }

  graph_def_str_ = graph_def_.DebugString();
// if (!protobuf::TextFormat::ParseFromString(graph_def_str_, &graph_def_)) {
//    return errors::InvalidArgument("parse ", graph_def_str_, " to protobuf failed");
//  }
  
  return Status::OK();
}

void BlazeXlaOp::Compute(OpKernelContext* ctx) {
  predictor_->Compute(ctx);
}

REGISTER_KERNEL_BUILDER(Name("BlazeXlaOp").Device(DEVICE_CPU), BlazeXlaOp);
REGISTER_KERNEL_BUILDER(Name("BlazeXlaOp").Device(DEVICE_GPU), BlazeXlaOp);
}  // namespace tensorflow
