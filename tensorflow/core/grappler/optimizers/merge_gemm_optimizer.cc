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

#include "tensorflow/core/grappler/optimizers/merge_gemm_optimizer.h"

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

struct SharedInputGemmPattern {
  const Node* input;
  std::vector<Node*> reshape;
  std::vector<Node*> identity;
  std::vector<Node*> weight;
  std::vector<Node*> matmul;
  std::vector<const Edge*> output;
  void DebugPattern() {
    VLOG(0) << "input:" << input->DebugString();
    VLOG(0) << "reshape:";
    for (auto n:reshape) VLOG(0) << n->DebugString();
    VLOG(0) << "weight:";
    for (auto n:weight) VLOG(0) << n->DebugString();
    VLOG(0) << "identity:";
    for (auto n:identity) VLOG(0) << n->DebugString();
    VLOG(0) << "matmul:";
    for (auto n:matmul) VLOG(0) << n->DebugString();
    VLOG(0) << "output:";
    for (auto e:output) VLOG(0) << e->DebugString();

  }
};

void DebugSharedInputGemmPattern(std::map<std::string, SharedInputGemmPattern>& collection) {
  for (auto iter:collection) {
    VLOG(0) << iter.first << ", parallel path size: " << iter.second.matmul.size();
    VLOG(0) << "*************************************************";
    iter.second.DebugPattern();
  }
}

Node* NodeConstructor(Graph* graph, string name, Node* base,
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

Node* CreateConstNode(Graph* graph, string const_name, Tensor &t_const, Node* base) {
  std::function<Status(NodeDef&)> const_builder = [&](NodeDef& def) {
    return NodeDefBuilder(const_name, "Const")
                          .Attr("dtype", t_const.dtype())
                          .Attr("value", t_const)
                          .Finalize(&def);
  };
  return NodeConstructor(graph, const_name, base, const_builder);
}

Node* ConstuctConcatOp(Graph* graph, Node* matmul,
                     std::vector<Node*>& input_nodes, string sufix) {
  // indice const，将二维权重按列concat起来
  string idx_name = matmul->name() + sufix + "_indice";
  Tensor t(DT_INT64, TensorShape({}));
  t.scalar<int64>()() = 1;
  Node* idx_const = CreateConstNode(graph, idx_name, t, matmul);
  if (idx_const == nullptr) {
    LOG(ERROR) << "construct indice const failed";
    return false;
  }
  string concat_name =  matmul->name() + sufix;
  int input_size = input_nodes.size();
  VLOG(0) << concat_name << ", size:" << input_size;
  std::vector<NodeDefBuilder::NodeOut> inputs;
  for (auto n:input_nodes) {
    inputs.emplace_back(n->name(), 0, matmul->output_type(0));
  }
  std::function<Status(NodeDef&)> concat_builder = [&](NodeDef& def) {
    return NodeDefBuilder(concat_name, "ConcatV2")
                          .Input(inputs)
                          .Input({idx_name, 0, DT_INT64})
                          .Attr("N", input_size)
                          .Attr("T", matmul->output_type(0))
                          .Attr("Tidx", DT_INT64)
                          .Finalize(&def);
  };
  Node* concat = NodeConstructor(graph, concat_name, matmul, concat_builder);
  int port = 0;
  for (auto n:input_nodes) {
    graph->AddEdge(n, 0, concat, port);
    port++;
  }
  graph->AddEdge(idx_const, 0, concat, port);
  graph->UpdateEdge(concat, 0, matmul, 1);
  return concat;
}

Node* ConstuctSplitOp(Graph* graph, Node* matmul, int split_num,
                       std::vector<const Edge*>& out_edges, string sufix) {
  // dim const
  string dim_name = matmul->name() + sufix + "_split_dim";
  Tensor t(DT_INT32, TensorShape({}));
  t.scalar<int32>()() = 1;
  Node* dim_const = CreateConstNode(graph, dim_name, t, matmul);
  if (dim_const == nullptr) {
    LOG(ERROR) << "construct split dim const failed";
    return false;
  }
  string split_name =  matmul->name() + sufix;
  int out_size = out_edges.size();
  VLOG(1) << split_name << ", size:" << out_size;
  std::function<Status(NodeDef&)> split_builder = [&](NodeDef& def) {
    return NodeDefBuilder(split_name, "Split")
                          .Input({dim_name, 0, DT_INT32})
                          .Input({matmul->name(), 0, matmul->output_type(0)})
                          .Attr("num_split", split_num)
                          .Attr("T", matmul->output_type(0))
                          .Finalize(&def);
  };
  Node* split = NodeConstructor(graph, split_name, matmul, split_builder);
  int port = 0;
  graph->AddEdge(dim_const, 0, split, 0);
  graph->AddEdge(matmul, 0, split, 1);
  for (auto e:out_edges) {
    graph->UpdateEdge(split, port, e->dst(), e->dst_input());
    port++;
  }
  return split;
}

int GetConstNumElements(const Node* weight) {
  int num_elements = 0;
  if (weight != nullptr) {
    const auto& tensor_proto = weight->def().attr().at("value").tensor();
    Tensor tensor;
    if (!tensor.FromProto(tensor_proto)) {
      LOG(WARNING) << "Cannot parse weight tensor proto: " << weight->name();
      return 0;
    }
    return tensor.NumElements();
  }
  return 0;
}

bool GetSharedInputGemmPattern(const Node* input, SharedInputGemmPattern& pattern) {
  if (input->out_edges().size() < 2) {
    VLOG(1) << "num output less 2";
    return false;
  }
  if (input->type_string() == "Split") {
    return false;
  }
  for (auto e:input->out_edges()) {
    if (e->src_output() > 0) {
      return false;
    }
  }
  pattern.input = input;
  VLOG(1) << input->name() << " output count:" << input->out_edges().size();
  for (auto n:input->out_nodes()) {
    if (n->type_string() == "Reshape") {
      pattern.reshape.push_back(n);
    } else {
      VLOG(1) << "find invalid node:" << n->DebugString();
      continue;
    }
  }
  for (auto reshape:pattern.reshape) {
    for (auto n:reshape->out_nodes()) {
      if (n->type_string() == "MatMul") {
        pattern.matmul.push_back(n);
      } else {
        VLOG(1) << "find invalid node:" << n->DebugString();
        return false;
      }
    }
  }
  for (auto n:pattern.matmul) {
    Node* weight;
    n->input_node(1, &weight);
    if (weight->type_string() == "Identity") {
      pattern.identity.push_back(weight);
      Node* const_weight;
      weight->input_node(0, &const_weight);
      if (const_weight->type_string() == "Const") {
        pattern.weight.push_back(const_weight);
      } else {
        VLOG(0) << "find invalid node:" << const_weight->DebugString();
        return false;
      }
    } else if (weight->type_string() == "Const") {
      pattern.weight.push_back(weight);
    } else {
      VLOG(0) << "find invalid node:" << weight->DebugString();
      return false;
    }
    for (auto e:n->out_edges()) {
      pattern.output.push_back(e);
    }
  }
  if (pattern.matmul.size() < 2) {
    VLOG(1) << "matmul op less 2";
    return false;
  }
  if (pattern.matmul.size() != pattern.weight.size()) {
    VLOG(0) << "matmul op not equal to weight: "
            << pattern.matmul.size() << " VS " << pattern.weight.size();
    return false;
  }
  int num_element = 0;
  DataType dtype = DT_FLOAT;
  for (auto weight:pattern.weight) {
    if (num_element == 0) {
      dtype = weight->def().attr().at("dtype").type();
      num_element = GetConstNumElements(weight);
    } else {
      if (dtype != weight->def().attr().at("dtype").type()) {
        VLOG(0) << "weight data type not match!";
        pattern.DebugPattern();
        return false;
      }
      if (num_element != GetConstNumElements(weight)) {
        VLOG(0) << "weight shape not match";
        pattern.DebugPattern();
        return false;
      }
    }

  }

  return true;
}

// TODO merge reshape and biasadd after splited matmul
//         MatMul                     MatMul
//           |                          | concat const
//         Split                        | /
//         /   \  shape              BiasAdd
//        /     \ /                     | new shape
//  Reshape   Reshape        -->        | /  
//     | const1  | const2            Reshape
//     | /       |  /                   |
//  BiasAdd   BiasAdd                 Split
//     |         |                    /  \
//   out1       out2                out1 out2
Status MergeGemm(Graph* graph) {
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  Status status;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  std::map<std::string, SharedInputGemmPattern> collection;
  VLOG(1) << "start to merge gemm node, " << nodes.size();
  for (Node* node : nodes) {
    if (node->type_string() != "MatMul") continue;
    Node* gemm = node;
    const Node* reshape;
    gemm->input_node(0, &reshape);
    if (reshape->type_string() != "Reshape") continue;
    const Node* input_a;
    reshape->input_node(0, &input_a);
    std::string key = input_a->name();
    SharedInputGemmPattern pattern;
    if (collection.find(key) != collection.end()) {
      continue;
    }
    if (GetSharedInputGemmPattern(input_a, pattern)) {
      VLOG(1) << "find " << key;
      collection[key] = std::move(pattern);
    }
  }
  if (VLOG_IS_ON(1)) DebugSharedInputGemmPattern(collection);
  for (auto iter:collection) {
    // 1.concat weight input
    // 2.连接concat到其中一个matmul op，其余删除
    Node* matmul = (iter.second.matmul)[0];
    Node* merged_weight = ConstuctConcatOp(graph, matmul, iter.second.weight, "/merge_weight");
    // 3.split为多个输出
    Node* split = ConstuctSplitOp(graph, matmul, iter.second.weight.size(),
                                  iter.second.output, "/split_output");
    // 4.删除多余节点
    for (auto n:iter.second.identity) {
      graph->RemoveNode(n);
    }
    for (auto n:iter.second.matmul) {
      if (n != matmul) {
        graph->RemoveNode(n);
      }
    }
  }
  return Status::OK();
}
}  // end namespace

Status MergeGemmOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool opt = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE", true, &opt);
  if (!opt) {
    *optimized_graph = item.graph;
    return Status::OK();
  }
  static int pass = 0;
  VLOG(0) << "MergeGemmOptimizer is on." << pass;
  if (VLOG_IS_ON(1)) {
    std::fstream f;
    f.open("before_merge_gemm_" + std::to_string(pass) + ".pb",
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
  status = MergeGemm(&graph);
  if (!status.ok()) {
    LOG(WARNING) << " merge gemm failed: " << status.ToString();
    *optimized_graph = item.graph;
    return Status::OK();
  }
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

  if (VLOG_IS_ON(1)) {
    std::fstream f;
    f.open("after_merge_gemm_" + std::to_string(pass) + ".pb",
           std::fstream::out);
    f << optimized_graph->SerializeAsString();
    f.close();
  }
  pass++;
  return Status::OK();
}

void MergeGemmOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
