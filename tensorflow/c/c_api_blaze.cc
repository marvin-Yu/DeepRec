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

#include "tensorflow/c/c_api_blaze.h"
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
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
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
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/env_var.h"

// The implementation below is at the top level instead of the
// brain namespace because we are defining 'extern "C"' functions.
using tensorflow::AllocationDescription;
using tensorflow::CallableOptions;
using tensorflow::DataType;
using tensorflow::ExtendSessionGraphHelper;
using tensorflow::Env;
using tensorflow::GpuIdManager;
using tensorflow::GpuIdUtil;
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
using tensorflow::PlatformGpuId;
using tensorflow::RunMetadata;
using tensorflow::RunOptions;
using tensorflow::Session;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::TfGpuId;
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

static std::unordered_set<string> GetNonComputeIntensiveNodes() {
  std::unordered_set<string> ops = {
      //"ExpandDims",
      //"Gather",
      //"GatherV2",
      //"GatherNd",
      //"Identity",
      //"Pack",
      //"Slice",
      //"Squeeze",
      //"StridedSlice",
      //"Split",
      //"SplitV",
      //"Tile",
      //"Transpose",
      //"Unpack",
      //"Where",
      "TakeAxis",
      "Sum",
      "NotEqual",
      "Reshape",
      "Concat",
      "ConcatV2"};
  return ops;
}

static int InsertToCPUSet(Node* node,
                   const std::unordered_set<string>& candidates,
                   std::unordered_set<Node*>* cpu_nodes) {
  // Put Placeholders on CPU
  if (node->type_string() == "Placeholder" ||
      node->type_string() == "PlaceholderV2") {
    VLOG(2) << "InsertToCPUSet: " << node->DebugString();
    return cpu_nodes->insert(node).second;
  }
  // Put INT32/INT64 node, which is also the input of CPU nodes, on CPU.
  DataType output_type = node->output_type(0);
  if ((output_type == tensorflow::DT_INT32 ||
       output_type == tensorflow::DT_INT64) &&
      node->type_string() != "Shape") {
    for (const tensorflow::Edge* e : node->out_edges()) {
      Node* n = e->dst();
      if (cpu_nodes->find(n) != cpu_nodes->end()) {
        VLOG(2) << "InsertToCPUSet: " << n->DebugString();
        return cpu_nodes->insert(node).second;
      }
    }
    return 0;
  }

  // Put Concats/Reshapes/... after Placeholders on CPU
  if (candidates.find(node->type_string()) == candidates.end()) {
    return 0;
  }
  std::vector<Node*> int_or_const_inputs;
  for (const tensorflow::Edge* e : node->in_edges()) {
    Node* n = e->src();
    DataType type = n->output_type(e->src_output());
    if (((type == tensorflow::DT_INT32 || type == tensorflow::DT_INT64) &&
         n->type_string() != "Shape") ||
        n->type_string() == "Const") {
      int_or_const_inputs.emplace_back(n);
      continue;
    }
    if (cpu_nodes->find(n) == cpu_nodes->end()) {
      return 0;
    }
  }
  int new_insertion = 0;
  for (Node* n : int_or_const_inputs) {
    VLOG(2) << "InsertToCPUSet: " << n->DebugString();
    if (cpu_nodes->insert(n).second) new_insertion++;
  }
  VLOG(2) << "InsertToCPUSet: " << node->DebugString();
  if (cpu_nodes->insert(node).second) new_insertion++;
  return new_insertion;
}

static void AutoPlaceNodesOnCPU(Graph* graph) {
  // To improve CPU-GPU memcpy and GPU compute efficiency,
  // we place some memory intensive nodes to run on CPU.
  // Traverse from placeholders, mark cheap nodes
  // after placeholders to run on CPU.
  std::unordered_set<Node*> cpu_nodes;
  std::unordered_set<string> candidates =
      GetNonComputeIntensiveNodes();
  while(1) {
    int new_insertion = 0;
    for (Node* node : graph->nodes()) {
      VLOG(2) << "Check node: " << node->DebugString();
      new_insertion += InsertToCPUSet(node, candidates, &cpu_nodes);
      VLOG(2) << "Check node: new_insertion = " << new_insertion;
    }
    if (new_insertion == 0) break;
  }

  std::string cpu_device = "/device:CPU:0";
  for (Node* node : cpu_nodes) {
    node->set_requested_device(cpu_device);
    VLOG(2) << "Place on CPU: " << node->DebugString();
  }
}

void TF_GraphSetDevice(TF_Graph* graph, int cpu_id, int gpu_id) {
  mutex_lock l(graph->mu);
  Graph* g = &(graph->graph);

  // TODO
//  AutoPlaceNodesOnCPU(g);

  LOG(INFO) << "TF_GraphSetDevice: cpu_id, gpu_id = "
            << cpu_id << ", " << gpu_id;
  std::string cpu_device = "/device:CPU:" + std::to_string(cpu_id);
  std::string gpu_device = "/device:GPU:" + std::to_string(gpu_id);
  std::string device;
  if (gpu_id >= 0) {
    device = gpu_device;
  } else {
    device = cpu_device;
  }
  for (Node* node : g->nodes()) {
    std::string requested_device = node->requested_device();
    if (requested_device.find("CPU") != std::string::npos ||
        requested_device.find("cpu") != std::string::npos) {
      node->set_requested_device(cpu_device);
      node->AddAttr("_XlaCompile", false);
      VLOG(1) << "Place node " << node->name() << " on " << cpu_device;
    } else {
      node->set_requested_device(device);
      VLOG(1) << "Place node " << node->name() << " on " << device;
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

void TF_EnableAutoMixedPrecision(TF_SessionOptions* options,
                                 unsigned char enable) {
  tensorflow::ConfigProto& config = options->options.config;
  auto* rewrite_config =
      config.mutable_graph_options()->mutable_rewrite_options();
  if (enable) {
    rewrite_config->set_auto_mixed_precision(tensorflow::RewriterConfig::ON);
  } else {
    rewrite_config->set_auto_mixed_precision(tensorflow::RewriterConfig::OFF);
  }
}

void TF_EnableVirtualGPUDevices(
    TF_SessionOptions* options,
    int num_virtual_gpus_per_device,
    int memory_limit_mb_per_virtual_gpu,
    int num_phisical_gpus) {
  if (num_virtual_gpus_per_device <= 0 || num_phisical_gpus <= 0) return;
  auto* gpu_options = options->options.config.mutable_gpu_options();
  for (int i = 0; i < num_phisical_gpus; i++) {
    auto virtual_devices =
        gpu_options->mutable_experimental()->add_virtual_devices();
    for (int j = 0; j < num_virtual_gpus_per_device; j++) {
      virtual_devices->add_memory_limit_mb(
          memory_limit_mb_per_virtual_gpu);
    }
  }
}

void TF_SetCPUDeviceCount(
    TF_SessionOptions* options, int num_cpus) {
  if (num_cpus <= 0) return;
  auto* device_count = options->options.config.mutable_device_count();
  device_count->insert({"CPU", num_cpus});
}

void TF_SetThreadPoolOptions(
    TF_SessionOptions* options,
    int num_inter_op_threads,
    int num_intra_op_threads) {
  if (num_inter_op_threads > 0) {
    auto* pool_config = options->options.config.
                        add_session_inter_op_thread_pool();
    pool_config->set_num_threads(num_inter_op_threads);
  }
  if (num_intra_op_threads > 0) {
    options->options.config.set_intra_op_parallelism_threads(
        num_intra_op_threads);
  }
}

void TF_SetGPUMemoryOptions(TF_SessionOptions* options,
                            unsigned char allow_growth,
                            unsigned char force_gpu_compatible) {
  auto* gpu_options = options->options.config.mutable_gpu_options();
  gpu_options->set_allow_growth(allow_growth);
  gpu_options->set_force_gpu_compatible(force_gpu_compatible);
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

void TF_SessionMakeCallable(TF_Session* tf_sess, TF_CallableHandle* callable_handle,
                            const char* const* feed_names, int feed_count,
                            const char* const* fetch_names, int fetch_count,
                            const char* device_name, TF_Status* status) {
  std::vector<tensorflow::DeviceAttributes> devices;
  tf_sess->session->ListDevices(&devices);
  for (const auto& device : devices) {
    LOG(INFO) << device.name();
  }
  // directly, instead of requiring us to serialize to a GraphDef and
  // call Session::Extend().
  if (tf_sess->extend_before_run &&
      !ExtendSessionGraphHelper(tf_sess, status)) {
    return;
  }

  CallableOptions opts;
  for (int i = 0; i < feed_count; ++i) {
    const char* feed_name = feed_names[i];
    opts.add_feed(feed_name);
    opts.mutable_feed_devices()->insert({feed_name, device_name});
  }
  for (int i = 0; i < fetch_count; ++i) {
    const char* fetch_name = fetch_names[i];
    opts.add_fetch(fetch_name);
    opts.mutable_fetch_devices()->insert({fetch_name, device_name});
  }
  opts.set_fetch_skip_sync(true);
  LOG(INFO) << opts.DebugString();
  Session::CallableHandle handle;
  status->status = tf_sess->session->MakeCallable(opts, &handle);
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "session make callable failed!";
    return;
  }
  *callable_handle = handle;
}

void TF_SessionRunCallable(TF_Session* tf_sess, TF_CallableHandle callable_handle,
                           TF_Tensor* const* input_values, int ninputs,
                           TF_Tensor** output_values, int noutputs,
                           TF_Buffer* run_metadata, TF_Status* status) {
  std::vector<Tensor> input_tensors(ninputs);
  for (int i = 0; i < ninputs; ++i) {
    status->status = tensorflow::TF_TensorToTensor(input_values[i], &input_tensors[i]);
    if (TF_GetCode(status) != TF_OK) return;
  }

  std::vector<Tensor> output_tensors;
  RunMetadata run_metadata_proto;
  status->status = tf_sess->session->RunCallable(callable_handle, input_tensors,
                                                 &output_tensors, &run_metadata_proto);
  // Serialize back to upstream client, who now owns the new buffer
  if (run_metadata != nullptr) {
    status->status = MessageToBuffer(run_metadata_proto, run_metadata);
    if (TF_GetCode(status) != TF_OK) return;
  }
  if (output_tensors.size() != noutputs) return;
  for (int i = 0; i < noutputs; ++i) {
    output_values[i] = tensorflow::TF_TensorFromTensor(output_tensors[i], status);
    if (TF_GetCode(status) != TF_OK) return;
  }
}

void TF_SessionReleaseCallable(TF_Session* tf_sess, TF_CallableHandle callable_handle,
                               TF_Status* status) {
  status->status = tf_sess->session->ReleaseCallable(callable_handle);
}

stream_executor::Stream* GetStreamOfVirtualDevice(int virtual_gpu_id) {
  TfGpuId tf_gpu_id(virtual_gpu_id);
  PlatformGpuId platform_gpu_id;
  Status s = GpuIdManager::TfToPlatformGpuId(tf_gpu_id, &platform_gpu_id);
  if (!s.ok()) {
    LOG(ERROR) << "invalid tf gpu id: " << virtual_gpu_id;
    return nullptr;
  }
  stream_executor::StreamExecutor* se =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
  static tensorflow::GPUOptions gpu_options;
  tensorflow::BaseGPUDevice::StreamGroup* stream_group = tensorflow::StreamGroupFactory::Global().GetOrCreate(
      tf_gpu_id, 0, se, gpu_options);
  return stream_group->compute;
}

bool TF_CudaMemAlloc(int virtual_gpu_id, void** gpu_ptr, size_t length) {
  stream_executor::Stream* stream = GetStreamOfVirtualDevice(virtual_gpu_id);
  if (stream == nullptr) {
    return false;
  }
  stream->parent()->UnifiedMemoryAllocate(length);
  return true;
}

bool TF_CudaMemDealloc(int virtual_gpu_id, void* gpu_ptr) {
  stream_executor::Stream* stream = GetStreamOfVirtualDevice(virtual_gpu_id);
  if (stream == nullptr) {
    return false;
  }
  stream->parent()->UnifiedMemoryDeallocate(gpu_ptr);
  return true;
}

bool TF_CudaMemCopyHostToDeviceAsync(int virtual_gpu_id, void* device_ptr, const void* host_ptr, size_t length) {
  stream_executor::Stream* stream = GetStreamOfVirtualDevice(virtual_gpu_id);
  if (stream == nullptr) {
    return false;
  }
  stream_executor::DeviceMemoryBase device_memory(device_ptr, length);
  stream->ThenMemcpy(&device_memory, host_ptr, length);
  return true;
}

bool TF_CudaMemCopyDeviceToHostAsync(int virtual_gpu_id, void* host_ptr, const void* device_ptr, size_t length) {
  stream_executor::Stream* stream = GetStreamOfVirtualDevice(virtual_gpu_id);
  if (stream == nullptr) {
    return false;
  }
  stream_executor::DeviceMemoryBase device_memory(const_cast<void*>(device_ptr), length);
  stream->ThenMemcpy(host_ptr, device_memory, length);
  return true;
}

bool TF_CudaBlockStreamUntilDone(int virtual_gpu_id, TF_Status* status) {
  stream_executor::Stream* stream = GetStreamOfVirtualDevice(virtual_gpu_id);
  if (stream == nullptr) {
    return false;
  }
  stream->BlockHostUntilDone();
  return true;
}

}  // end extern "C"

