#include "tensorflow/core/kernels/blaze_xla_predictor.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/gpu_utils.h"
#endif

namespace tensorflow {
InputNodeMap BlazeXlaPredictor::ToInputNodeMap() {
  InputNodeMap node_map;
  for (int i = 0; i < graph_def_.node_size(); ++i) {
    auto& node = graph_def_.node(i);
    for (int j = 0; j < node.input_size(); ++j) {
      auto& input = node.input(j);
      auto iter = node_map.find(input);
      if (iter == node_map.end()) {
        std::vector<NodeDef> node_list;
        node_list.push_back(node);
        node_map[input] = std::move(node_list);
      } else {
        iter->second.push_back(node);
      }
    }
  }
  return node_map;
}

Status BlazeXlaPredictor::FindBlackPaddingInputs() {
  skip_padding_.resize(input_names_.size());
  auto node_map = ToInputNodeMap();

  for (int i = 0; i < input_names_.size(); ++i) {
    skip_padding_[i] = false;
    auto& name = input_names_[i];

    auto iter = node_map.find(name);
    if (iter == node_map.end()) {
      return errors::Internal("input ", name, " not in graph ",
                              graph_def_.DebugString());
    }
    for (auto& node : iter->second) {
      bool xla_enable = true;
      if (TryGetNodeAttr(node, "_XlaCompile", &xla_enable) && xla_enable == false) {
        LOG(INFO) << name << " will not padding in xla";
        skip_padding_[i] = true;
        break;
      }
    }
  }
  return Status::OK();
}

Status BlazeXlaPredictor::InitXlaWarmup() {
  if (blaze_run_options_.warmup_batchsize_size() == 0) {
    return errors::Internal("xla not setting warmup batchsize");
  }

  std::vector<int> warm(blaze_run_options_.warmup_batchsize().begin(),
                        blaze_run_options_.warmup_batchsize().end());

  std::sort(warm.begin(), warm.end());
  for (auto val : warm) {
    if (val <= 0) {
      return errors::Internal("warmuup batchsize ", val, " invalid");
    }
  }

  batch_sizes_ = std::move(warm);
}

Status BlazeXlaPredictor::PrepareData() {
  TF_RETURN_IF_ERROR(FindBlackPaddingInputs());
  TF_RETURN_IF_ERROR(InitXlaWarmup());

  return Status::OK();
}

#define TYPECASE_0(dt, X, Y)                                    \
  case dt: {                                                  \
    return (void*)X->flat<EnumToDataType<dt>::Type>().data(); \
  }

void* GetTensorAddress(const Tensor* tensor_ptr) {
  auto tensor_type = tensor_ptr->dtype();
  switch (tensor_type) {
    TYPECASE_0(DT_FLOAT, tensor_ptr, dest_ptr);
    TYPECASE_0(DT_HALF, tensor_ptr, dest_ptr);
    TYPECASE_0(DT_INT8, tensor_ptr, dest_ptr);
    TYPECASE_0(DT_INT32, tensor_ptr, dest_ptr);
    TYPECASE_0(DT_INT64, tensor_ptr, dest_ptr);
    default: {
      LOG(ERROR) << "Unsupported Data type " << DataTypeString(tensor_type);
      return nullptr;
    }
  }
}

#define TYPECASE_1(dt, X, Y)                                    \
  case dt: {                                                  \
    return X->flat<EnumToDataType<dt>::Type>().size() * sizeof(EnumToDataType<dt>::Type); \
  }
uint64 GetTensorSize(const Tensor* tensor_ptr) {
  auto tensor_type = tensor_ptr->dtype();
  switch (tensor_type) {
    TYPECASE_1(DT_FLOAT, tensor_ptr, dest_ptr);
    TYPECASE_1(DT_HALF, tensor_ptr, dest_ptr);
    TYPECASE_1(DT_INT8, tensor_ptr, dest_ptr);
    TYPECASE_1(DT_INT32, tensor_ptr, dest_ptr);
    TYPECASE_1(DT_INT64, tensor_ptr, dest_ptr);
    default: {
      LOG(ERROR) << "Unsupported Data type " << DataTypeString(tensor_type);
      return 0;
    }
  }
}

int BlazeXlaPredictor::InferBatchSize(const std::vector<Tensor>& tensors) {
  int batchsize = -1;
  for (int i = 0; i < tensors.size(); ++i) {
    VLOG(1) << "Shape of input " << i << ": "
            << tensors[i].shape().DebugString();
    if (skip_padding_[i]) continue;
    int dims = tensors[i].shape().dims();
    if (dims == 0) continue;
    int64 first_dim = tensors[i].shape().dim_size(0);
    if (batchsize == -1 || (batchsize == 1 && first_dim != 1)) {
      batchsize = first_dim;
    }
    if (batchsize != 1 && first_dim != 1 && first_dim != batchsize) {
      batchsize = -1;
      VLOG(1) << "Cannot infer batchsize: tensors have different dim_size(0).";
      break;
    }
  }
  return batchsize;
}
Status BlazeXlaPredictor::PadToStatic(const std::vector<Tensor>& inputs,
                                      std::vector<Tensor>* padded_inputs,
                                      int batchsize, int pad_to_batchsize,
                                      OpKernelContext* ctx) {
  for (int i = 0; i < inputs.size(); ++i) {
    VLOG(1) << "Shape of input " << i << ": "
            << inputs[i].shape().DebugString();
    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_on_host(ctx->input_memory_type(i) == HOST_MEMORY);
    TensorShape pad_to_shape;
    TensorShape shape = inputs[i].shape();
    pad_to_shape = shape;
    int64 first_dim = shape.dim_size(0);
    first_dim = (first_dim == 1)? 1 : pad_to_batchsize;
    if (first_dim == 1 || skip_padding_[i]) {
      (*padded_inputs)[i] = inputs[i];
      VLOG(1) << "Shape of padded_input " << i << ": "
              << (*padded_inputs)[i].shape().DebugString();
      continue;
    }
    pad_to_shape.set_dim(0, first_dim);
    Status allocate_status =
        ctx->allocate_temp(inputs[i].dtype(),
                           pad_to_shape,
                           &(*padded_inputs)[i], alloc_attrs);
    if (!allocate_status.ok()) {
      return allocate_status;
    }
    const uint8* input_ptr = (uint8*)GetTensorAddress(&inputs[i]);
    uint8* padded_ptr = (uint8*)GetTensorAddress(&(*padded_inputs)[i]);
    uint64 input_size = GetTensorSize(&inputs[i]);
    uint64 padded_size = GetTensorSize(&(*padded_inputs)[i]);
    if (input_ptr == nullptr || padded_ptr == nullptr ||
        input_size == 0 || padded_size == 0) {
      return errors::Internal(
          "Error when getting input address or size");
    }
    if (device_type_ == DEVICE_GPU && ctx->input_memory_type(i) == DEVICE_MEMORY) {
#if GOOGLE_CUDA
      auto* stream = ctx->op_device_context()->stream();
      auto input_dev_ptr = AsDeviceMemory(input_ptr, input_size);
      auto padded_dev_ptr = AsDeviceMemory(padded_ptr, padded_size);
      bool copy_status =
          stream->ThenMemcpyD2D(&padded_dev_ptr, input_dev_ptr, input_size).ok();
      if (!copy_status) {
        return errors::Internal("MemcpyD2D for padding inputs failed.");
      }
#endif
    } else {
      std::memset(padded_ptr, 0, padded_size);
      std::memcpy(padded_ptr, input_ptr, input_size);
    }

    VLOG(1) << "Shape of padded_input " << i << ": "
            << (*padded_inputs)[i].shape().DebugString();
  }
  return Status::OK();
}

Status BlazeXlaPredictor::SliceToDynamic(const std::vector<Tensor>& padded_outputs,
                                         std::vector<Tensor*>* outputs,
                                         int batchsize, int pad_to_batchsize,
                                         OpKernelContext* ctx) {
  for (int i = 0; i < padded_outputs.size(); ++i) {
    VLOG(1) << "Shape of padded_output " << i << ": "
            << padded_outputs[i].shape().DebugString();
    TensorShape slice_to_shape = padded_outputs[i].shape();
    if (slice_to_shape.dim_size(0) != pad_to_batchsize) {
      return errors::Internal(
          "Shape error, cannot slice output: padded_output shape = " +
          slice_to_shape.DebugString() +
          ", pad_to_batchsize = " +
          std::to_string(pad_to_batchsize));
    }
    slice_to_shape.set_dim(0, batchsize);
    Status allocate_status =
        ctx->allocate_output(i, slice_to_shape, &(*outputs)[i]);
    if (!allocate_status.ok()) {
      return allocate_status;
    }
    uint8* output_ptr = (uint8*)GetTensorAddress((*outputs)[i]);
    const uint8* padded_ptr = (uint8*)GetTensorAddress(&padded_outputs[i]);
    uint64 output_size = GetTensorSize((*outputs)[i]);
    uint64 padded_size = GetTensorSize(&padded_outputs[i]);
    if (output_ptr == nullptr || padded_ptr == nullptr ||
        output_size == 0 || padded_size == 0) {
      return errors::Internal(
          "Error when getting output address or size");
    }
    if (device_type_ == DEVICE_GPU && ctx->input_memory_type(i) == DEVICE_MEMORY) {
#if GOOGLE_CUDA
      auto* stream = ctx->op_device_context()->stream();
      auto output_dev_ptr = AsDeviceMemory(output_ptr, output_size);
      auto padded_dev_ptr = AsDeviceMemory(padded_ptr, padded_size);
      bool copy_status =
          stream->ThenMemcpyD2D(&output_dev_ptr, padded_dev_ptr, output_size).ok();
      if (!copy_status) {
        return errors::Internal("MemcpyD2D for unpadding outputs failed.");
      }
#endif
    } else {
      memcpy(output_ptr, padded_ptr, output_size);
    }
    VLOG(1) << "Shape of output " << i << ": "
            << (*outputs)[i]->shape().DebugString();
  }
  return Status::OK();
}

void BlazeXlaPredictor::Compute(OpKernelContext* ctx) {
  // Infer inputs' batchsize
  int num_inputs = ctx->num_inputs();
  std::vector<Tensor> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(ctx->input(i));
  }
  int batchsize = InferBatchSize(inputs);
  if (batchsize == -1) {
    ctx->SetStatus(
        errors::Internal("Cannot infer inputs' batchsize"));
    return;
  }

  int pad_to_batchsize = batchsize;
  for (int n : batch_sizes_) {
    if (n >= batchsize) {
      pad_to_batchsize = n;
      break;
    }
  }

  VLOG(1) << "batchsize = " << batchsize
          << ", pad_to_batchsize = " << pad_to_batchsize;

  if (pad_to_batchsize != batchsize) {
    // Pad inputs
    std::vector<Tensor> padded_inputs(num_inputs);
    Status status = PadToStatic(inputs, &padded_inputs,
        batchsize, pad_to_batchsize, ctx);
    if (!status.ok()) {
      ctx->SetStatus(status);
      return;
    }

    // Call SessionRun
    std::vector<Tensor> padded_outputs;
    if (ctx->prof_stats()) {
      RunMetadata metadata;
      OP_REQUIRES_OK(ctx, session_->RunCallable(
              handle_, padded_inputs, &padded_outputs, &metadata));
      ctx->prof_stats()->flops += metadata.prof_stats().flops();
    } else {
      OP_REQUIRES_OK(ctx, session_->RunCallable(
              handle_, padded_inputs, &padded_outputs, nullptr));
    }

    // Unpad outputs
    std::vector<Tensor*> outputs(ctx->num_outputs());
    status = SliceToDynamic(padded_outputs, &outputs,
                            batchsize, pad_to_batchsize, ctx);
    if (!status.ok()) {
      ctx->SetStatus(status);
      return;
    }
  } else {
    // Call SessionRun
    VLOG(1) << "Skip padding: input bathsize = " << batchsize
            << ", input pad_to_batchsize = " << pad_to_batchsize;
    std::vector<Tensor> outputs;
    if (ctx->prof_stats()) {
      RunMetadata metadata;
      OP_REQUIRES_OK(ctx, session_->RunCallable(
              handle_, inputs, &outputs, &metadata));
      ctx->prof_stats()->flops += metadata.prof_stats().flops();
    } else {
      OP_REQUIRES_OK(ctx, session_->RunCallable(
              handle_, inputs, &outputs, nullptr));
    }
    for (int i = 0; i < outputs.size(); ++i) {
      ctx->set_output(i, outputs[i]);
    }
  }
  return;
}
}
