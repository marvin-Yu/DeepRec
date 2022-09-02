/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/partial_mixed_precision.h"

#include <fstream>
#include <queue>
#include <map>
#include <algorithm>

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"


namespace tensorflow {
namespace grappler {

namespace {

bool GemmOpSet(string op) {
  static std::unordered_set<string> op_set = {
      "MatMul",
      "BatchMatMul",
      "BatchMatMulV2",
      "IndicatorMatMul",
      "CoAction",
      "CoActionIndicator"};
  if (op_set.find(op) != op_set.end()) {
    return true;
  }
  return false;
}

Node* NodeConstructor(Graph* graph, string name, const Node* base,
                     const std::function<Status(NodeDef&)>& node_builder) {
  NodeDef def;
  Status status = node_builder(def);
  if (!status.ok()) {
    LOG(ERROR) << name << " Adding nodedef build failed " << status;
    return nullptr;
  }
  def.set_device(base->def().device());
  Node *node = graph->AddNode(def, &status);
  if (!status.ok()) {
    LOG(ERROR) << name <<" Adding node failed " << status;
    return nullptr;
  }
  node->set_assigned_device_name(base->assigned_device_name());
  return node;
}

Node* ConstuctCastOp(Graph* graph, const Node* base, int port,
                     DataType src, DataType dst, string cast_name) {
  std::function<Status(NodeDef&)> cast_builder = [&](NodeDef& def) {
    return NodeDefBuilder(cast_name, "Cast")
                          .Input({base->name(), port, src})
                          .Attr("SrcT", src)
                          .Attr("DstT", dst)
                          .Finalize(&def);
  };
  Node* cast = NodeConstructor(graph, cast_name, base, cast_builder);
  VLOG(1) << "cast " << cast->DebugString();
  return cast;
}

Node* ConstuctMatMulOp(Graph* graph, const Edge* input,
                       const Edge* weight, string sufix) {
  const Node* matmul = input->dst();
  string name =  matmul->name() + sufix;
  string trans_key_a = "transpose_a";
  string trans_key_b = "transpose_b";
  bool transpose_a = false;
  bool transpose_b = false;
  if (matmul->def().attr().find("adj_x") != matmul->def().attr().end()) {
    trans_key_a = "adj_x";
    trans_key_b = "adj_y";
    transpose_a = matmul->def().attr().at("adj_x").b();
    transpose_b = matmul->def().attr().at("adj_y").b();
  } else {
    transpose_a = matmul->def().attr().at("transpose_a").b();
    transpose_b = matmul->def().attr().at("transpose_b").b();
  }
  std::vector<NodeDefBuilder::NodeOut> matmul_inputs;
  matmul_inputs.emplace_back(input->src()->name(), input->src_output(), DT_HALF);
  matmul_inputs.emplace_back(weight->src()->name(), weight->src_output(), DT_HALF);
  if (matmul->type_string() == "IndicatorMatMul") {
    const Edge* indice;
    matmul->input_edge(2, &indice);
    matmul_inputs.emplace_back(indice->src()->name(), indice->src_output(),
                               indice->src()->output_type(0));

  }
  std::function<Status(NodeDef&)> matmul_builder = [&](NodeDef& def) {
    NodeDefBuilder builder(name, matmul->type_string());
    builder.Input(matmul_inputs[0]);
    builder.Input(matmul_inputs[1]);
    if (matmul->type_string() == "IndicatorMatMul") {
      builder.Input(matmul_inputs[2]);
    }
    return builder.Attr(trans_key_a, transpose_a)
                  .Attr(trans_key_b, transpose_b)
                  .Attr("T", DT_HALF)
                  .Finalize(&def);
  };
  Node* new_matmul = NodeConstructor(graph, name, matmul, matmul_builder);
  VLOG(1) << "matmul " << new_matmul->DebugString();
  return new_matmul;
}

Node* ConstuctCoActionOp(Graph* graph, const Edge* input,
                       const Edge* weight, string sufix) {
  const Node* co_action = input->dst();
  string name =  co_action->name() + sufix;
  std::vector<NodeDefBuilder::NodeOut> inputs;
  inputs.emplace_back(input->src()->name(), input->src_output(), DT_HALF);
  inputs.emplace_back(weight->src()->name(), weight->src_output(), DT_HALF);
  if (co_action->type_string() == "CoActionIndicator") {
    const Edge* indice;
    co_action->input_edge(2, &indice);
    inputs.emplace_back(indice->src()->name(), indice->src_output(),
                        indice->src()->output_type(0));

  }
  std::function<Status(NodeDef&)> co_action_builder = [&](NodeDef& def) {
    NodeDefBuilder builder(name, co_action->type_string());
    builder.Input(inputs[0]);
    builder.Input(inputs[1]);
    if (co_action->type_string() == "CoActionIndicator") {
      DataType type = co_action->def().attr().at("Tindices").type();
      builder.Input(inputs[2]);
      builder.Attr("Tindices", type);
    }
    int pow_num = co_action->def().attr().at("pow_num").i();
    return builder.Attr("T", DT_HALF)
                  .Attr("pow_num", pow_num)
                  .Finalize(&def);
  };
  Node* new_co_action = NodeConstructor(graph, name, co_action, co_action_builder);
  VLOG(1) << "co_action " << new_co_action->DebugString();
  return new_co_action;
}

Node* SearchGemmInputConstNode(const Node* matmul) {
  Node* weight;
  matmul->input_node(1, &weight);
  if (weight->type_string() == "Identity") {
    Node* const_weight;
    weight->input_node(0, &const_weight);
    if (const_weight->type_string() == "Const") {
      return const_weight;
    }
  } else if (weight->type_string() == "Const") {
    return weight;
  }
  return nullptr;
}

TensorShape GetGemmInputConstNumElements(const Node* matmul) {
  Node* weight = SearchGemmInputConstNode(matmul);
  if (weight != nullptr) {
    DataType type = weight->def().attr().at("dtype").type();
    if (type != DT_FLOAT) return TensorShape({});
    const auto& tensor_proto = weight->def().attr().at("value").tensor();
    Tensor tensor;
    if (!tensor.FromProto(tensor_proto)) {
      LOG(WARNING) << "Cannot parse weight tensor proto: " << weight->name();
      return TensorShape({});
    }
    return tensor.shape();
  }
  return TensorShape({});
}

bool CastGemmFloatToHalf(Graph* graph, std::set<Node*>& candidate) {
  for (auto n:candidate) VLOG(1) << n->type_string() << ", " << n->name();

  std::map<string, Node*> node_cache;
  for (auto n:candidate) {
    const Edge* input;
    n->input_edge(0, &input);
    Node* input_cast;
    string input_cast_name = input->src()->name() + "_" +
                             std::to_string(input->src_output()) +
                             "/PartialMixedPrecision_cast_float2half";
    if (node_cache.find(input_cast_name) == node_cache.end()) {
      input_cast = ConstuctCastOp(graph, input->src(), input->src_output(),
                                      DT_FLOAT, DT_HALF, input_cast_name);
      graph->AddEdge(input->src(), input->src_output(), input_cast, 0);
      node_cache.insert(std::pair<string, Node*>(input_cast_name, input_cast));
    } else {
      VLOG(1) << "find input cast in cache " << input_cast_name;
      input_cast = node_cache.find(input_cast_name)->second;
    }
    const Edge* weight;
    n->input_edge(1, &weight);
    Node* weight_cast;
    string weight_cast_name = weight->src()->name() + "_" +
                             std::to_string(input->src_output()) +
                             "/PartialMixedPrecision_cast_float2half";
    if (node_cache.find(weight_cast_name) == node_cache.end()) {
      weight_cast = ConstuctCastOp(graph, weight->src(), weight->src_output(),
                                       DT_FLOAT, DT_HALF, weight_cast_name);
      graph->AddEdge(weight->src(), weight->src_output(), weight_cast, 0);
      node_cache.insert(std::pair<string, Node*>(weight_cast_name, weight_cast));
    } else {
      VLOG(1) << "find weight cast in cache " << weight_cast_name;
      weight_cast = node_cache.find(weight_cast_name)->second;
    }
    string compute_cast_name = n->name() + "/PartialMixedPrecision_cast_half2float";
    Node* compute_cast = ConstuctCastOp(graph, n, 0, DT_HALF, DT_FLOAT, compute_cast_name);
    string op_str = n->type_string();
    Node* new_compute;
    if (op_str == "CoAction" || op_str == "CoActionIndicator") {
      new_compute = ConstuctCoActionOp(graph, input, weight,
                                     "/PartialMixedPrecision_half_compute");
    } else {
      new_compute = ConstuctMatMulOp(graph, input, weight,
                                     "/PartialMixedPrecision_half_compute");
    }

    graph->AddEdge(input_cast, 0, new_compute, 0);
    graph->AddEdge(weight_cast, 0, new_compute, 1);
    if (op_str == "IndicatorMatMul" || op_str == "CoActionIndicator") {
      const Edge* indice;
      n->input_edge(2, &indice);
      graph->AddEdge(indice->src(), 0, new_compute, 2);
    }
    for (auto e:n->out_edges()) {
      graph->UpdateEdge(compute_cast, 0, e->dst(), e->dst_input());
    }
    graph->AddEdge(new_compute, 0, compute_cast, 0);
    graph->RemoveNode(n);
  }
  VLOG(0) << "convert " << candidate.size() << " gemm to half";
  return true;
}

Status ConvertGemm(Graph* graph) {
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  Status status;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  // skip conversion if num_elements less than 128*128
  TensorShape max_shape({128, 128});
  std::set<Node*> candidate;
  for (Node* node : nodes) {
    if (node->type_string() != "MatMul") continue;
    if (node->def().attr().at("T").type() == DT_HALF) continue;
    TensorShape shape = GetGemmInputConstNumElements(node);
    if (shape.num_elements() > max_shape.num_elements()) {
      max_shape = shape;
      candidate.clear();
      candidate.insert(node);
    } else if (shape.num_elements() == max_shape.num_elements()) {
      candidate.insert(node);
    }
  }
  if (candidate.empty()) {
    VLOG(0) << "no candidate gemm, skip conversion";
    return Status::OK();
  }
  VLOG(0) << "largest weight: " << max_shape.DebugString();
  CastGemmFloatToHalf(graph, candidate);
  return Status::OK();
}

bool PreorderHasGemm(Graph* graph, Node* node) {
  std::queue<Node*> unvisited_queue;
  std::unordered_set<string> visited;
  for (auto e : node->in_edges()) {
    unvisited_queue.push(e->src());
  }
  while(!unvisited_queue.empty()) {
    Node* top = unvisited_queue.front();
    unvisited_queue.pop();
    if (visited.count(top->name()) != 0) continue;
    visited.insert(top->name());
    if (GemmOpSet(top->type_string())) {
      return true;
    }
    for (auto e:top->in_edges()) {
      if (visited.count(e->src()->name()) != 0) continue;
      unvisited_queue.push(e->src());
    }
  }
  return false;
}

Status ConvertFirstLayerGemm(Graph* graph) {
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  Status status;
  std::set<Node*> candidate;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  for (Node* node : nodes) {
    if (!GemmOpSet(node->type_string())) continue;
    if (node->def().attr().at("T").type() == DT_HALF) continue;
    // 如果前序节点中没有gemm类op，则认为可以转化为FP16
    if (!PreorderHasGemm(graph, node)) {
      candidate.insert(node);
    }
  }
  CastGemmFloatToHalf(graph, candidate);
  return Status::OK();
}

bool IsCastFloatToHalf(NodeDef* node) {
  if (node->op() != "Cast") {
    return false;
  }
  DataType src_type = node->attr().at("SrcT").type();
  DataType dst_type = node->attr().at("DstT").type();
  if (src_type == DT_FLOAT && dst_type == DT_HALF) {
    return true;
  }

  return false;
}

bool IsCastHalfToFloat(NodeDef* node) {
  if (node->op() != "Cast") {
    return false;
  }
  DataType src_type = node->attr().at("SrcT").type();
  DataType dst_type = node->attr().at("DstT").type();
  if (src_type == DT_HALF && dst_type == DT_FLOAT) {
    return true;
  }

  return false;
}

Status Collapse(GraphDef* graph) {
  std::unordered_set<string> nodes_removable;
  NodeMap node_map(graph);
  for (int i = 0; i < graph->node_size(); i++) {
    auto node = graph->mutable_node(i);
    if (node->op() != "Cast") continue;
    auto cast_input = node_map.GetNode(node->input(0));
    if ((IsCastFloatToHalf(node) && IsCastHalfToFloat(cast_input)) ||
       (IsCastHalfToFloat(node) && IsCastFloatToHalf(cast_input))) {
      const string& cast_first = node->input(0);
      auto first_outputs = node_map.GetOutputs(cast_first);
      if (first_outputs.size() != 1) {
        continue;
      }

      const string& cast_second = node->name();
      auto outputs = node_map.GetOutputs(cast_second);
      if (outputs.size() != 1) {
        continue;
      }
      NodeDef* output = *outputs.begin();
      string input = node_map.GetNode(cast_first)->input(0);
      for (int i = 0; i < output->input_size(); i++) {
        if (output->input(i).compare(cast_second) == 0) {
          *output->mutable_input(i) = input;
          break;
        }
      }
      VLOG(1) << "remove " << cast_first;
      VLOG(1) << "remove " << cast_second;
      nodes_removable.insert(cast_first);
      nodes_removable.insert(cast_second);
    }
  }
  graph->mutable_node()->erase(
      std::remove_if(
          graph->mutable_node()->begin(), graph->mutable_node()->end(),
          [nodes_removable](const NodeDef& node) {
            return nodes_removable.find(node.name()) != nodes_removable.end();
          }),
      graph->mutable_node()->end());
  return Status::OK();
}

}  // end namespace

Status PartialMixedPrecision::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool opt = true;
  ReadBoolFromEnvVar("TF_ENABLE_PARTIAL_MIXED_PRECISION", true, &opt);
  if (!opt) {
    *optimized_graph = item.graph;
    return Status::OK();
  }
  static int pass = 0;
  VLOG(0) << "PartialMixedPrecision is on." << pass;
  if (VLOG_IS_ON(1)) {
    std::fstream f;
    f.open("before_partial_mixed_precision_" + std::to_string(pass) + ".pb",
           std::fstream::out);
    f << item.graph.SerializeAsString();
    f.close();
  }

  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());
  Graph graph(flib);
  Status status = ConvertGraphDefToGraph(GraphConstructorOptions(),
                                  item.graph, &graph);
  if (!status.ok()) {
    LOG(WARNING) << "ConvertGraphDefToGraph failed: " << status.ToString();
    *optimized_graph = item.graph;
    return Status::OK();
  }
  VLOG(0) << "PartialMixedPrecision round 1";
  status = ConvertGemm(&graph);
  if (!status.ok()) {
    LOG(WARNING) << " convert gemm to half round 1 failed: " << status.ToString();
    *optimized_graph = item.graph;
    return Status::OK();
  }
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();
  ReadBoolFromEnvVar("TF_ENABLE_PARTIAL_MIXED_PRECISION_REDICAL", true, &opt);
  if (opt) {
    VLOG(0) << "PartialMixedPrecision round 2";
    status = ConvertGemm(&graph);
    if (!status.ok()) {
      LOG(WARNING) << " convert gemm to half round 2 failed: " << status.ToString();
      return Status::OK();
    }
    graph.ToGraphDef(optimized_graph);
    *optimized_graph->mutable_versions() = item.graph.versions();
  }

  if (VLOG_IS_ON(1)) {
    std::fstream f;
    f.open("after_partial_mixed_precision_" + std::to_string(pass) + ".pb",
           std::fstream::out);
    f << optimized_graph->SerializeAsString();
    f.close();
  }
  pass++;
  return Status::OK();
}

void PartialMixedPrecision::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

Status PartialMixedPrecisionSecondStage::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool opt = true;
  ReadBoolFromEnvVar("TF_ENABLE_PARTIAL_MIXED_PRECISION", true, &opt);
  if (!opt) {
    *optimized_graph = item.graph;
    return Status::OK();
  }
  static int pass = 0;
  VLOG(0) << "PartialMixedPrecisionSecondStage is on." << pass;

  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());
  Graph graph(flib);
  Status status = ConvertGraphDefToGraph(GraphConstructorOptions(),
                                  item.graph, &graph);
  if (!status.ok()) {
    LOG(WARNING) << "ConvertGraphDefToGraph failed: " << status.ToString();
    *optimized_graph = item.graph;
    return Status::OK();
  }
  VLOG(0) << "PartialMixedPrecision convert all first layer gemm";
  status = ConvertFirstLayerGemm(&graph);
  if (!status.ok()) {
    LOG(WARNING) << " convert all first layer gemm to half failed: " << status.ToString();
    *optimized_graph = item.graph;
    return Status::OK();
  }
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

  int node_before = optimized_graph->node_size();
  status = Collapse(optimized_graph);
  VLOG(0) << "Collapse Cast pairs " << node_before << "/" << optimized_graph->node_size();
  if (VLOG_IS_ON(0)) {
    std::fstream f;
    f.open("after_partial_mixed_precision_" + std::to_string(pass) + ".pb",
           std::fstream::out);
    f << optimized_graph->SerializeAsString();
    f.close();
  }
  pass++;
  return Status::OK();
}

void PartialMixedPrecisionSecondStage::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
