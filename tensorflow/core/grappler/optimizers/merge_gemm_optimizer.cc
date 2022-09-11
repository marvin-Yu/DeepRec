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
  std::vector<std::vector<const Edge*>> output;
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
    for (auto set:output) {
      for (auto e:set) VLOG(0) << e->DebugString();
    }
  }
};

void DebugSharedInputGemmPattern(std::map<std::string, SharedInputGemmPattern>& collection) {
  for (auto iter:collection) {
    VLOG(0) << iter.first << ", parallel path size: " << iter.second.matmul.size();
    VLOG(0) << "*************************************************";
    iter.second.DebugPattern();
  }
}

struct MergeBiasAddPattern {
  Node* split;
  std::vector<Node*> reshape;
  std::vector<Node*> shape;
  std::vector<Node*> add;
  std::vector<Node*> bias;
  std::vector<std::vector<const Edge*>> output;
  void DebugPattern() {
    VLOG(0) << "split:" << split->DebugString();
    VLOG(0) << "reshape:";
    for (auto n:reshape) VLOG(0) << n->DebugString();
    VLOG(0) << "shape:";
    for (auto n:shape) VLOG(0) << n->DebugString();
    VLOG(0) << "add:";
    for (auto n:add) VLOG(0) << n->DebugString();
    VLOG(0) << "bias:";
    for (auto n:bias) VLOG(0) << n->DebugString();
    VLOG(0) << "output:";
    for (auto set:output) {
      for (auto e:set) VLOG(0) << e->DebugString();
    }
  }
};

void DebugMergeBiasAddPattern(std::map<std::string, MergeBiasAddPattern>& collection) {
  for (auto iter:collection) {
    VLOG(0) << iter.first << ", parallel path size: " << iter.second.add.size();
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

Node* ConstructConcatOp(Graph* graph, Node* compute, int64 axis,
                     std::vector<Node*>& input_nodes, string sufix) {
  // indice const，将二维权重按列concat起来
  string idx_name = compute->name() + sufix + "_indice";
  Tensor t(DT_INT64, TensorShape({}));
  t.scalar<int64>()() = axis;
  Node* idx_const = CreateConstNode(graph, idx_name, t, compute);
  if (idx_const == nullptr) {
    LOG(ERROR) << "construct indice const failed";
    return nullptr;
  }
  string concat_name =  compute->name() + sufix;
  int input_size = input_nodes.size();
  VLOG(0) << concat_name << ", size:" << input_size;
  std::vector<NodeDefBuilder::NodeOut> inputs;
  for (auto n:input_nodes) {
    inputs.emplace_back(n->name(), 0, compute->output_type(0));
  }
  std::function<Status(NodeDef&)> concat_builder = [&](NodeDef& def) {
    return NodeDefBuilder(concat_name, "ConcatV2")
                          .Input(inputs)
                          .Input({idx_name, 0, DT_INT64})
                          .Attr("N", input_size)
                          .Attr("T", compute->output_type(0))
                          .Attr("Tidx", DT_INT64)
                          .Finalize(&def);
  };
  Node* concat = NodeConstructor(graph, concat_name, compute, concat_builder);
  int port = 0;
  for (auto n:input_nodes) {
    graph->AddEdge(n, 0, concat, port);
    port++;
  }
  graph->AddEdge(idx_const, 0, concat, port);
  graph->UpdateEdge(concat, 0, compute, 1);
  return concat;
}

Node* ConstructSplitOp(Graph* graph, Node* compute, int split_num, int axis,
                       std::vector<std::vector<const Edge*>>& out_edges, string sufix) {
  // dim const
  string dim_name = compute->name() + sufix + "_split_dim";
  Tensor t(DT_INT32, TensorShape({}));
  t.scalar<int32>()() = axis;
  Node* dim_const = CreateConstNode(graph, dim_name, t, compute);
  if (dim_const == nullptr) {
    LOG(ERROR) << "construct split dim const failed";
    return nullptr;
  }
  string split_name =  compute->name() + sufix;
  int out_size = out_edges.size();
  VLOG(1) << split_name << ", size:" << out_size;
  std::function<Status(NodeDef&)> split_builder = [&](NodeDef& def) {
    return NodeDefBuilder(split_name, "Split")
                          .Input({dim_name, 0, DT_INT32})
                          .Input({compute->name(), 0, compute->output_type(0)})
                          .Attr("num_split", split_num)
                          .Attr("T", compute->output_type(0))
                          .Finalize(&def);
  };
  Node* split = NodeConstructor(graph, split_name, compute, split_builder);
  int port = 0;
  graph->AddEdge(dim_const, 0, split, 0);
  graph->AddEdge(compute, 0, split, 1);
  for (auto set:out_edges) {
    for (auto e:set) {
      Status status = graph->UpdateEdge(split, port, e->dst(), e->dst_input());
      if (!status.ok()) {
        LOG(WARNING) << "update edge failed: " << status.ToString();
      }
    }
    port++;
  }
  return split;
}

bool GetConstTensor(const Node* node, Tensor& tensor) {
  if (node != nullptr) {
    const auto& tensor_proto = node->def().attr().at("value").tensor();
    if (!tensor.FromProto(tensor_proto)) {
      LOG(WARNING) << "Cannot parse Const tensor proto: " << node->name();
      return false;
    }
    return true;
  }
  return false;
}

bool IsSameConst(std::vector<Node*>& nodes, bool check_value) {
  if (nodes.empty()) {
    return false;
  }
  Tensor base;
  DataType dtype = DT_FLOAT;
  for (auto node:nodes) {
    if (base.NumElements() == 0) {
      dtype = node->def().attr().at("dtype").type();
      if (!GetConstTensor(node, base)) {
        return false;
      }
    } else {
      if (dtype != node->def().attr().at("dtype").type()) {
        VLOG(0) << "const data type not match!";
        return false;
      }
      Tensor cmp;
      if (!GetConstTensor(node, cmp)) {
        return false;
      }
      if (base.shape().DebugString() != cmp.shape().DebugString()) {
        VLOG(0) << "const shape not match";
        return false;
      }
      if (check_value) {
        return base.DebugString() == cmp.DebugString();
      }
    }
  }
  return true;
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
    std::vector<const Edge*> out_edges;
    for (auto e:n->out_edges()) {
      out_edges.push_back(e);
    }
    pattern.output.push_back(std::move(out_edges));
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
  if (!IsSameConst(pattern.weight, false)) {
    VLOG(0) << "pattern weight not same";
    return false;
  }
  return true;
}

bool GetMergeBiasAddPattern(Node* split, MergeBiasAddPattern& pattern) {
  std::vector<const Edge*> split_out;
  for (auto e:split->out_edges()) {
    split_out.push_back(e);
  }
  std::sort(split_out.begin(), split_out.end(),
      [](const Edge* a, const Edge* b) {
        return a->src_output() < b->src_output();
      });
  for (auto e:split_out) {
    Node* reshape = e->dst();
    if (reshape->type_string() == "Reshape") {
      Node* shape;
      reshape->input_node(1, &shape);
      pattern.reshape.push_back(reshape);
      pattern.shape.push_back(shape);
    } else {
      VLOG(0) << "not reshape";
      return false;
    }
    for (auto e:reshape->out_edges()) {
      Node* add = e->dst();
      if (add->type_string() != "BiasAdd" &&
          add->type_string() != "Add" &&
          add->type_string() != "AddV2") {
        VLOG(0) << "not add " << add->DebugString();
        return false;
      }
      Node* bias;
      add->input_node(1, &bias);
      if (bias->type_string() != "Const") {
        VLOG(0) << "not const";
        return false;
      }
      pattern.add.push_back(add);
      pattern.bias.push_back(bias);
      std::vector<const Edge*> out_edges;
      for (auto e:add->out_edges()) {
        out_edges.push_back(e);
      }
      pattern.output.push_back(std::move(out_edges));
    }
  }
  pattern.split = split;
  int num_split = split->def().attr().at("num_split").i();
  if (num_split != pattern.reshape.size() ||
      num_split != pattern.add.size() ||
      num_split != pattern.shape.size() ||
      num_split != pattern.bias.size()) {
    LOG(WARNING) << "split output size not equal to parallel";
    return false;
  }

  if (!IsSameConst(pattern.bias, false)) {
    return false;
  }
  if (!IsSameConst(pattern.shape, true)) {
    return false;
  }
  Node* bias = pattern.bias[0];
  Tensor bias_t;
  if (!GetConstTensor(bias, bias_t)) {
    return false;
  }
  if (bias_t.dims() != 1) {
    LOG(WARNING) << "bias dim not equals to 1: " << bias_t.shape().DebugString();
    return false;
  }
  Node* shape = pattern.shape[0];
  Tensor shape_t;
  if (!GetConstTensor(shape, shape_t)) {
    return false;
  }
  int change_dim = shape_t.NumElements() - 1;
  int dim_last;
  if (shape_t.dtype() == DT_INT32) {
    auto data = shape_t.flat<int32>();
    dim_last = data(change_dim);
  } else {
    auto data = shape_t.flat<int64>();
    dim_last = data(change_dim);
  }
  if (dim_last != bias_t.NumElements()) {
    LOG(WARNING) << "reshape not match with bias: " << shape_t.DebugString()
                 << " VS " << bias_t.shape().DebugString();
    return false;
  }
  return true;
}

bool MergeGemm(Graph* graph, std::vector<Node*>& new_splits) {
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
    Tensor weight;
    if (!GetConstTensor(iter.second.weight[0], weight)) {
      return false;
    }
    Node* merged_weight = ConstructConcatOp(graph, matmul, weight.dims() - 1,
                                           iter.second.weight, "/merge_weight");
    // 3.split为多个输出
    Node* split = ConstructSplitOp(graph, matmul, iter.second.weight.size(),
                                  weight.dims() - 1, iter.second.output,
                                  "/split_output");
    // 4.删除多余节点
    for (auto n:iter.second.identity) {
      graph->RemoveNode(n);
    }
    for (auto n:iter.second.matmul) {
      if (n != matmul) {
        graph->RemoveNode(n);
      }
    }
    new_splits.push_back(split);
  }
  VLOG(0) << "merge gemm done";
  return true;
}

// merge reshape and biasadd after splited matmul
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
bool MergeBiasAdd(Graph* graph, std::vector<Node*>& new_splits) {
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  Status status;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  VLOG(1) << "start to merge BiasAdd node, " << nodes.size();
  std::map<std::string, MergeBiasAddPattern> collection;
  for (auto split:new_splits) {
    std::string key = split->name();
    if (collection.find(key) != collection.end()) {
      continue;
    }
    MergeBiasAddPattern pattern;
    if (GetMergeBiasAddPattern(split, pattern)) {
      VLOG(1) << "find " << key;
      collection[key] = std::move(pattern);
    }
  }

  if (VLOG_IS_ON(1)) DebugMergeBiasAddPattern(collection);
  for (auto iter:collection) {
    // 1.concat多路bias
    Node* add = iter.second.add[0];
    Tensor bias;
    if (!GetConstTensor(iter.second.bias[0], bias)) {
      return false;
    }
    Node* merged_bias = ConstructConcatOp(graph, add, bias.dims() - 1,
                                           iter.second.bias, "/merge_bias");
    const Edge* input;
    iter.second.split->input_edge(1, &input);
    graph->UpdateEdge(input->src(), input->src_output(), add, 0);
    // 2.构建新的shape
    Node* old_shape = iter.second.shape[0];
    Tensor new_shape_t;
    if (!GetConstTensor(old_shape, new_shape_t)) {
      return false;
    }
    int change_dim = new_shape_t.NumElements() - 1;
    if (new_shape_t.dtype() == DT_INT32) {
      auto data = new_shape_t.flat<int32>();
      data(change_dim) = data(change_dim) * iter.second.add.size();
    } else {
      auto data = new_shape_t.flat<int64>();
      data(change_dim) = data(change_dim) * iter.second.add.size();
    }
    string new_shape_name = old_shape->name() + "/merge_shape";
    Node* new_shape = CreateConstNode(graph, new_shape_name, new_shape_t, add);
    VLOG(1) << new_shape->DebugString();
    Node* reshape = iter.second.reshape[0];
    graph->UpdateEdge(add, 0, reshape, 0);
    graph->UpdateEdge(new_shape, 0, reshape, 1);
    // 3.split为多个输出
    Node* split = ConstructSplitOp(graph, reshape, iter.second.bias.size(),
                                  new_shape_t.NumElements() - 1,
                                  iter.second.output, "/split_output");
    // 4.删除多余节点
    for (auto n:iter.second.add) {
      if (n != add) {
        graph->RemoveNode(n);
      }
    }
    for (auto n:iter.second.reshape) {
      if (n != reshape) {
        graph->RemoveNode(n);
      }
    }
    graph->RemoveNode(iter.second.split);
  }
  VLOG(0) << "merge BiasAdd done";
  return true;
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
  std::vector<Node*> new_splits;
  if (!MergeGemm(&graph, new_splits)) {
    LOG(WARNING) << " merge gemm failed";
    *optimized_graph = item.graph;
    return Status::OK();
  }
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

  if (!MergeBiasAdd(&graph, new_splits)) {
    LOG(WARNING) << " merge BiasAdd failed";
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
