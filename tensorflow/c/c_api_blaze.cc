/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/c_api.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>

#include "absl/strings/match.h"
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"  // NOLINT

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eval_const_tensor.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"

// The implementation below is at the top level instead of the
// brain namespace because we are defining 'extern "C"' functions.
using tensorflow::AllocationDescription;
using tensorflow::DataType;
using tensorflow::ExtendSessionGraphHelper;
using tensorflow::Env;
using tensorflow::Graph;
using tensorflow::GraphDef;
using tensorflow::mutex_lock;
using tensorflow::NameRangeMap;
using tensorflow::NameRangesForNode;
using tensorflow::NewSession;
using tensorflow::Node;
using tensorflow::NodeBuilder;
using tensorflow::NodeDef;
using tensorflow::OpDef;
using tensorflow::OpRegistry;
using tensorflow::OutputTensor;
using tensorflow::PartialTensorShape;
using tensorflow::RunMetadata;
using tensorflow::RunOptions;
using tensorflow::Session;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorBuffer;
using tensorflow::TensorId;
using tensorflow::TensorShape;
using tensorflow::TensorShapeProto;
using tensorflow::VersionDef;
using tensorflow::errors::FailedPrecondition;
using tensorflow::errors::InvalidArgument;
using tensorflow::gtl::ArraySlice;
using tensorflow::strings::StrCat;
using tensorflow::MetaGraphDef;

extern "C" {

namespace {

Status ReadGraphDefFromFile(const string& graph_def_path, GraphDef* result) {
  Status status;
  if (!ReadBinaryProto(Env::Default(), graph_def_path, result).ok()) {
    return ReadTextProto(Env::Default(), graph_def_path, result);
  }
  return status;
}

Status ReadMetaGraphDefFromFile(const string& graph_def_path,
                                MetaGraphDef* result) {
  Status status;
  if (!ReadBinaryProto(Env::Default(), graph_def_path, result).ok()) {
    return ReadTextProto(Env::Default(), graph_def_path, result);
  }
  return status;
}

TF_Operation* ToOperation(Node* node) {
  return static_cast<TF_Operation*>(static_cast<void*>(node));
}

}

TF_Buffer* TF_ReadGraphDefFromFile(
    const char* graph_def_path,
    TF_Status* status) {
  GraphDef graph_def;
  status->status = ReadGraphDefFromFile(
      graph_def_path, &graph_def);
  if (!status->status.ok()) {
    return nullptr;
  }
  TF_Buffer* ret = TF_NewBuffer();
  status->status = MessageToBuffer(graph_def, ret);
  if (!status->status.ok()) {
    return nullptr;
  } else {
    return ret;
  }
}

TF_Buffer* TF_ReadMetaGraphDefFromFile(
    const char* graph_def_path,
    TF_Status* status) {
  MetaGraphDef graph_def;
  status->status = ReadMetaGraphDefFromFile(
      graph_def_path, &graph_def);
  if (!status->status.ok()) {
    return nullptr;
  }
  TF_Buffer* ret = TF_NewBuffer();
  status->status = MessageToBuffer(graph_def, ret);
  if (!status->status.ok()) {
    return nullptr;
  } else {
    return ret;
  }
}

void TF_GraphSetDevice(TF_Graph* graph,
                       const char* device) {
  mutex_lock l(graph->mu);
  Graph* g = &(graph->graph);
  for (Node* node : g->nodes()) {
    std::string requested_device = node->requested_device();
    // To improve performance, users may manually place
    // some memory-intensive nodes on CPU
    // (e.g., Concat after Placeholder(s)).
    // In this case, we respect such placement.
    // Also, for these manually specified nodes, we turn off
    // XLA compilation since XLA may ignore such placement.
    if (requested_device.find("CPU") == std::string::npos &&
        requested_device.find("cpu") == std::string::npos) {
      node->set_requested_device(device);
    } else {
      node->AddAttr("_XlaCompile", false);
    }
  }
}

static void GraphImportGraphDefLocked(TF_Graph* graph, const GraphDef& def,
                                      const TF_ImportGraphDefOptions* opts,
                                      TF_ImportGraphDefResults* tf_results,
                                      TF_Status* status)
    EXCLUSIVE_LOCKS_REQUIRED(graph->mu) {
  const int last_node_id = graph->graph.num_node_ids();
  tensorflow::ImportGraphDefResults results;
  status->status = tensorflow::ImportGraphDef(opts->opts, def, &graph->graph,
                                              &graph->refiner, &results);
  if (TF_GetCode(status) != TF_OK) return;

  // Add new nodes to name_map
  for (int i = last_node_id; i < graph->graph.num_node_ids(); ++i) {
    auto* node = graph->graph.FindNodeId(i);
    if (node != nullptr) graph->name_map[node->name()] = node;
  }

  // Populate return_tensors
  DCHECK(tf_results->return_tensors.empty());
  tf_results->return_tensors.resize(results.return_tensors.size());
  for (int i = 0; i < results.return_tensors.size(); ++i) {
    tf_results->return_tensors[i].oper =
        ToOperation(results.return_tensors[i].first);
    tf_results->return_tensors[i].index = results.return_tensors[i].second;
  }

  // Populate return_nodes
  DCHECK(tf_results->return_nodes.empty());
  tf_results->return_nodes.resize(results.return_nodes.size());
  for (int i = 0; i < results.return_nodes.size(); ++i) {
    tf_results->return_nodes[i] = ToOperation(results.return_nodes[i]);
  }

  // Populate missing unused map keys
  DCHECK(tf_results->missing_unused_key_names.empty());
  DCHECK(tf_results->missing_unused_key_indexes.empty());
  DCHECK(tf_results->missing_unused_key_names_data.empty());

  size_t size = results.missing_unused_input_map_keys.size();
  tf_results->missing_unused_key_names.resize(size);
  tf_results->missing_unused_key_indexes.resize(size);

  for (int i = 0; i < size; ++i) {
    TensorId id = results.missing_unused_input_map_keys[i];
    tf_results->missing_unused_key_names_data.emplace_back(id.first);
    tf_results->missing_unused_key_names[i] =
        tf_results->missing_unused_key_names_data.back().c_str();
    tf_results->missing_unused_key_indexes[i] = id.second;
  }
}

TF_Session* TF_LoadSessionFromCheckpoint(
    const TF_SessionOptions* session_options, const TF_Buffer* run_options,
    const char* export_dir, TF_Graph* graph, TF_Buffer* meta_graph_def,
    TF_Status* status) {
// TODO(sjr): Remove the IS_MOBILE_PLATFORM guard. This will require ensuring
// that the tensorflow/cc/saved_model:loader build target is mobile friendly.
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Loading a SavedModel is not supported on mobile. File a bug at "
      "https://github.com/tensorflow/tensorflow/issues if this feature is "
      "important to you");
  return nullptr;
#else
  mutex_lock l(graph->mu);
  if (!graph->name_map.empty()) {
    status->status = InvalidArgument("Graph is non-empty.");
    return nullptr;
  }

  RunOptions run_options_proto;
  if (run_options != nullptr && !run_options_proto.ParseFromArray(
                                    run_options->data, run_options->length)) {
    status->status = InvalidArgument("Unparseable RunOptions proto");
    return nullptr;
  }

  tensorflow::SavedModelBundle bundle;
  status->status =
      tensorflow::LoadCheckpoint(session_options->options, run_options_proto,
                                 export_dir, &bundle);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  // Create a TF_Graph from the MetaGraphDef. This is safe as long as Session
  // extends using GraphDefs. The Graph instance is different, but equivalent
  // to the one used to create the session.
  //
  // TODO(jhseu): When Session is modified to take Graphs instead of
  // GraphDefs, return the Graph generated in LoadSavedModel().
  TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
  TF_ImportGraphDefResults results;
  GraphImportGraphDefLocked(graph, bundle.meta_graph_def.graph_def(),
                            import_opts, &results, status);
  TF_DeleteImportGraphDefOptions(import_opts);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  if (meta_graph_def != nullptr) {
    status->status = MessageToBuffer(bundle.meta_graph_def, meta_graph_def);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }

  if (VLOG_IS_ON(1)) {
    std::fstream f;
    f.open("ckpt.metagraph.pbtxt", std::fstream::out);
	f << bundle.meta_graph_def.DebugString();
    f.close();
    f.open("ckpt.graph.pb", std::fstream::out | std::fstream::binary);
    f << bundle.meta_graph_def.graph_def().SerializeAsString();
    f.close();
  }

  TF_Session* session = new TF_Session(bundle.session.release(), graph);

  graph->sessions[session] = "";
  session->last_num_graph_nodes = graph->graph.num_node_ids();
  return session;
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

void TF_GetIONamesFromMetaGraphDef(
    const TF_Buffer* meta_graph_def,
	bool use_method_name,
	const char* method_name,
    int* ninput, char*** input_names,
    int* noutput, char*** output_names, TF_Status* status) {
  MetaGraphDef meta_graph_def_obj;
  if (meta_graph_def == nullptr) {
    status->status = InvalidArgument("MetaGraphDef Ptr is Null"); 
    return;
  }
  if (!meta_graph_def_obj.ParseFromArray(
      meta_graph_def->data, meta_graph_def->length)) {
    status->status = InvalidArgument(
        "MetaGraphDef Object Parse From Array Failed!");
    return;
  }

  const auto& signature_def_map = meta_graph_def_obj.signature_def();
  if (signature_def_map.size() == 0) {
    status->status = InvalidArgument(
        "MetaGraphDef does not contain signature_def!");
    return;
  }

  auto sig_iter = signature_def_map.begin();
  if (use_method_name) {
    sig_iter = signature_def_map.find(method_name);
    if (sig_iter == signature_def_map.end()) {
      status->status = InvalidArgument(
          "Method name is not contained in signature map");
      return;
    }
  }
  const auto& signature_def = sig_iter->second;
  int input_num = signature_def.inputs().size();
  *ninput = input_num;
  *input_names = (char**)malloc(sizeof(char*) * input_num);
  int i = 0;
  for (auto iter = signature_def.inputs().begin();
       iter != signature_def.inputs().end(); ++iter) {
    const auto& input_tensor_info = iter->second;
    const std::string& name = input_tensor_info.name();
    (*input_names)[i] = (char*)malloc(sizeof(char) * (name.length() + 1));
    strcpy((*input_names)[i], name.c_str());
    ++i;
  }

  int output_num = signature_def.outputs().size();
  *noutput = output_num;
  *output_names = (char**)malloc(sizeof(char*) * output_num);
  i = 0;
  for (auto iter = signature_def.outputs().begin();
       iter != signature_def.outputs().end(); ++iter) {
    const auto& output_tensor_info = iter->second;
    const std::string& name = output_tensor_info.name();
    (*output_names)[i] = (char*)malloc(sizeof(char) * (name.length() + 1));
    strcpy((*output_names)[i], name.c_str());
    ++i;
  }
  status->status = Status::OK();
}

void TF_EnableSoftDevicePlacement(TF_SessionOptions* options,
                                  unsigned char enable) {
  options->options.config.set_allow_soft_placement(enable);
}

void TF_EnableGemmOptimization(TF_SessionOptions* options,
                               unsigned char enable) {
  tensorflow::ConfigProto& config = options->options.config;
  auto* rewrite_config =
      config.mutable_graph_options()->mutable_rewrite_options();
  if (enable) {
    rewrite_config->set_gemm_optimization(tensorflow::RewriterConfig::ON);
  } else {
    rewrite_config->set_gemm_optimization(tensorflow::RewriterConfig::OFF);
  }
}

void TF_EnableGPUMemoryAllowGrowth(TF_SessionOptions* options,
                                   unsigned char enable) {
  auto* gpu_options = options->options.config.mutable_gpu_options();
  gpu_options->set_allow_growth(enable);
}

void TF_EnableCudaGraph(TF_Buffer* run_options, unsigned char enable,
                        unsigned char init, int count, TF_Status* status) {
  tensorflow::RunOptions run_options_proto;
  if (run_options != nullptr &&
      !run_options_proto.ParseFromArray(run_options->data,
                                        run_options->length)) {
    status->status = InvalidArgument("Unparseable RunOptions proto");
    return;
  }
  if (run_options->data_deallocator != nullptr) {
    (*run_options->data_deallocator)(const_cast<void*>(run_options->data),
                                     run_options->length);
  }
  run_options->data = nullptr;
  run_options->length = 0;

  run_options_proto.mutable_cuda_graph_options()->set_enable(enable);
  run_options_proto.mutable_cuda_graph_options()->set_initializing(init);
  if (count != -1) {
    run_options_proto.mutable_cuda_graph_options()->set_count(count);
  }
  TF_CHECK_OK(MessageToBuffer(run_options_proto, run_options));
  status->status = Status::OK();
}

}  // end extern "C"
