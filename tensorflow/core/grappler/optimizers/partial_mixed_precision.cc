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
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"


namespace tensorflow {
namespace grappler {

namespace {

Node* NodeConstructor(Graph* graph, string name, const Node* base,
                     const std::function<Status(NodeDef&)>& node_builder) {
  NodeDef def;
  Status status = node_builder(def);
  if (!status.ok()) {
    LOG(ERROR) << name << " Adding nodedef build failed " << status;
    return nullptr;
  }
  def.set_device(base->def().device());
  VLOG(1) << def.DebugString();
  Node *node = graph->AddNode(def, &status);
  if (!status.ok()) {
    LOG(ERROR) << name <<" Adding node failed " << status;
    return nullptr;
  }
  node->set_assigned_device_name(base->assigned_device_name());
  return node;
}

Node* ConstuctCastOp(Graph* graph, const Node* base, int port,
                     DataType src, DataType dst, string sufix) {
  string cast_name =  base->name() + sufix;
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
  bool transpose_a = matmul->def().attr().at("transpose_a").b();
  bool transpose_b = matmul->def().attr().at("transpose_b").b();

  std::function<Status(NodeDef&)> matmul_builder = [&](NodeDef& def) {
    return NodeDefBuilder(name, "MatMul")
                          .Input({input->src()->name(), input->src_output(), DT_HALF})
                          .Input({weight->src()->name(), weight->src_output(), DT_HALF})
                          .Attr("transpose_a", transpose_a)
                          .Attr("transpose_b", transpose_b)
                          .Attr("T", DT_HALF)
                          .Finalize(&def);
  };
  Node* new_matmul = NodeConstructor(graph, name, matmul, matmul_builder);
  VLOG(1) << "matmul " << new_matmul->DebugString();
  return new_matmul;
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
  int num_elements = 0;
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

Status ConvertGemm(Graph* graph) {
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  Status status;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  TensorShape max_shape;
  std::vector<Node*> candidate;
  for (Node* node : nodes) {
    if (node->type_string() != "MatMul") continue;
    TensorShape shape = GetGemmInputConstNumElements(node);
    int num_ele = shape.num_elements();
    if (shape.num_elements() > max_shape.num_elements()) {
      max_shape = shape;
      candidate.clear();
      candidate.push_back(node);
    } else if (shape.num_elements() == max_shape.num_elements()) {
      candidate.push_back(node);
    }
  }
  VLOG(0) << "largest weight: " << max_shape.DebugString();
  for (auto n:candidate) VLOG(1) << n->DebugString();
  for (auto matmul:candidate) {
    const Edge* input;
    matmul->input_edge(0, &input);
    Node* input_cast = ConstuctCastOp(graph, input->src(), input->src_output(),
                                      DT_FLOAT, DT_HALF, "/cast_float2half");
    const Edge* weight;
    matmul->input_edge(1, &weight);
    Node* weight_cast = ConstuctCastOp(graph, weight->src(), weight->src_output(),
                                       DT_FLOAT, DT_HALF, "/cast_float2half");
    Node* matmul_cast = ConstuctCastOp(graph, matmul, 0,
                                       DT_HALF, DT_FLOAT, "/cast_half2float");
    Node* new_matmul = ConstuctMatMulOp(graph, input, weight, "/half_compute");

    graph->AddEdge(input->src(), input->src_output(), input_cast, 0);
    graph->AddEdge(input_cast, 0, new_matmul, 0);
    graph->AddEdge(weight->src(), weight->src_output(), weight_cast, 0);
    graph->AddEdge(weight_cast, 0, new_matmul, 1);
    for (auto e:matmul->out_edges()) {
      graph->UpdateEdge(matmul_cast, 0, e->dst(), e->dst_input());
    }
    graph->AddEdge(new_matmul, 0, matmul_cast, 0);
    graph->RemoveNode(matmul);
  }
  VLOG(0) << "convert " << candidate.size() << " gemm to half";
  return Status::OK();
}
}  // end namespace

Status PartialMixedPrecision::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool opt = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE", true, &opt);
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
  bool opt2 = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE_ROUND_2", true, &opt2);
  if (opt2) {
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

}  // end namespace grappler
}  // end namespace tensorflow
