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
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/kernels/blaze_predictor.h"
#include "tensorflow/core/kernels/blaze_xla_predictor.h"

namespace tensorflow {
class BlazeXlaOp : public OpKernel {
 public:
  explicit BlazeXlaOp(OpKernelConstruction* context);
  ~BlazeXlaOp() {}

  void Compute(OpKernelContext* context) override;

  Status ParseRunOptions(BlazeKernelOptions& run_options);
 private:
  Status ParseAttr();
  void InitPredictor(OpKernelConstruction* context);
  void TraceTensors(OpKernelContext* ctx);
  void CopyTensor(MemoryType, OpKernelContext* ctx,
                  const string& name, const Tensor& tensor);

  DeviceType device_type_;
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
  std::unique_ptr<BlazePredictor> predictor_;
  Env* env_;
  std::mutex tracing_mu_;
};

void BlazeXlaOp::InitPredictor(OpKernelConstruction* context) {
  if (blaze_run_options_.xla_compilation()) {
    predictor_ = absl::make_unique<BlazeXlaPredictor>(input_names_, output_names_,
                                       graph_def_, device_, blaze_run_options_,
                                       device_string_, input_types_, context);
  } else {
    predictor_ = absl::make_unique<BlazePredictor>(input_names_, output_names_,
                                    graph_def_, device_, blaze_run_options_,
                                    device_string_, input_types_, context);
  }
}

BlazeXlaOp::BlazeXlaOp(OpKernelConstruction* context)
    : OpKernel(context), device_type_(context->device_type().type()) {
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
  env_ = Env::Default();
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
  
  return Status::OK();
}

void BlazeXlaOp::Compute(OpKernelContext* ctx) {
  if (!ctx->traced_infos()) {
    predictor_->Compute(ctx);
  } else {
    auto start_ms = env_->NowNanos();
    predictor_->Compute(ctx);
    auto end_ms = env_->NowNanos();
    if (ctx->traced_infos()->enable_prof_stats) {
      ctx->traced_infos()->prof_stats->blaze_latency_ms = ((end_ms - start_ms) / 1000.0f);
    }

    if (ctx->traced_infos()->enable_trace_tensors) {
      TraceTensors(ctx);
    }
  }
}

void BlazeXlaOp::TraceTensors(OpKernelContext* ctx) {
  if (ctx->status().ok()) {
    int num_inputs = ctx->num_inputs();
    for (int i = 0; i < num_inputs; ++i) {
      const auto& tensor = ctx->input(i);
      const auto& name = input_names_[i];
      CopyTensor(ctx->input_memory_type(i), ctx, name, tensor);
    }

    for (int i = 0; i < ctx->num_outputs(); ++i) {
      const auto tensor = ctx->mutable_output(i);
      const auto& name = output_names_[i];
      CopyTensor(ctx->output_memory_type(i), ctx, name, *tensor);
    }
  }
}

void BlazeXlaOp::CopyTensor(MemoryType mtype, OpKernelContext* ctx,
                            const string& name, const Tensor& tensor) {
  if (device_type_ == DEVICE_GPU && mtype == DEVICE_MEMORY) {
    DeviceContext* device_ctxt = ctx->op_device_context();
    Device* device = static_cast<Device*>(ctx->device());

    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_gpu_compatible(true);
    host_alloc_attrs.set_on_host(true);
    Allocator* cpu_allocator = device->GetAllocator(host_alloc_attrs);
    Tensor* cpu_tensor =
        new Tensor(cpu_allocator, tensor.dtype(), tensor.shape());
    device_ctxt->CopyDeviceTensorToCPU(
        &tensor, "TensorTrace", device, cpu_tensor,
        [this, cpu_tensor, ctx, &name](const Status& s) {
          ctx->SetStatus(s);
          if (s.ok()) {
            std::lock_guard<std::mutex> l(tracing_mu_);
            auto name_tensor = ctx->traced_infos()->traced_tensors->
              mutable_name_tensors()->Add();
            name_tensor->set_name(name);
            cpu_tensor->AsProtoField(name_tensor->mutable_tensor()); 
          }
          if (ctx->status().ok()) {
            ctx->set_output(0, *cpu_tensor);
          }
          delete cpu_tensor;
        });
  } else {
    auto name_tensor = ctx->traced_infos()->traced_tensors->
      mutable_name_tensors()->Add();
    name_tensor->set_name(name);
    tensor.AsProtoField(name_tensor->mutable_tensor());
  }
}

REGISTER_KERNEL_BUILDER(Name("BlazeXlaOp").Device(DEVICE_CPU), BlazeXlaOp);
REGISTER_KERNEL_BUILDER(Name("BlazeXlaOp").Device(DEVICE_GPU), BlazeXlaOp);
}  // namespace tensorflow
