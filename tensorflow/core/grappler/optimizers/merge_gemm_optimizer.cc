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
#include "tensorflow/core/grappler/optimizers/original_delivery_common.h"

#include <fstream>
#include <queue>
#include <map>
#include <algorithm>

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"


namespace tensorflow {
namespace grappler {

namespace {

bool IsBinaryOp(string op) {
  static std::unordered_set<string> op_set = {
      "Add", "AddV2"};
  if (op_set.find(op) != op_set.end()) {
    return true;
  }
  return false;
}

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

struct AttentionPattern {
  Node* concat;
  struct PathPattern {
    const Edge* gemm_pre_input_a;
    const Edge* gemm_pre_input_b;
    const Edge* gemm_pre;
    const Edge* softmax;
    const Edge* add;
    const Edge* add_y;
    const Edge* gemm_tail;
    const Edge* split;
    const Edge* split_dim;
    const Edge* shape;
    const Edge* reshape;
    const Edge* input;
    void DebugPattern() {
      VLOG(0) << "input:" << input->src()->DebugString();
      VLOG(0) << "shape:" << shape->src()->DebugString();
      VLOG(0) << "reshape:" << reshape->src()->DebugString();
      VLOG(0) << "split:" << split->src()->DebugString();
      VLOG(0) << "split_dim:" << split_dim->src()->DebugString();
      VLOG(0) << "input a:" << gemm_pre_input_a->src()->DebugString();
      VLOG(0) << "input b:" << gemm_pre_input_b->src()->DebugString();
      VLOG(0) << "gemm_pre:" << gemm_pre->src()->DebugString();
      VLOG(0) << "softmax:" << softmax->src()->DebugString();
      VLOG(0) << "add:" << add->src()->DebugString();
      VLOG(0) << "add y:" << add_y->src()->DebugString();
      VLOG(0) << "gemm_tail:" << gemm_tail->src()->DebugString();
    }
  };
  std::vector<PathPattern> path;
  std::vector<const Edge*> output;
  Node* split;
  void DebugPattern() {
    VLOG(0) << "concat:" << concat->DebugString();
    VLOG(0) << "all path:";
    for (int i = 0; i < path.size(); i++) {
      VLOG(0) << "path " << i;
      path[i].DebugPattern();
    }
    VLOG(0) << "output:";
    for (auto e:output) VLOG(0) << e->DebugString();
  }
};

void DebugAttentionPattern(std::map<std::string, AttentionPattern>& collection) {
  for (auto iter:collection) {
    VLOG(0) << iter.first << ", parallel path size: " << iter.second.path.size();
    VLOG(0) << "*************************************************";
    iter.second.DebugPattern();
  }
}

string OpTypePattern::DebugString() const {
  string result = "{" + op + ", {";
  for (const OpTypePattern& input : inputs) {
    result += input.DebugString() + ",";
  }
  result += "}}";
  return result;
}

string NodeMatch::DebugString() const {
  string result = "{";
  if (edge != nullptr && edge->src() != nullptr) result += edge->src()->DebugString();
  result += ", {";
  for (const NodeMatch& input : inputs) {
    result += input.DebugString() + ",";
  }
  result += "}}";
  return result;
}

// target node is the src of edge
bool DoesOpTypeMatch(const Edge* edge, const OpTypePattern& pattern,
                     NodeMatch* match) {
  Node* node = edge->src();
  VLOG(1) << "Looking at node " << node->DebugString();
  VLOG(1) << "pattern=" << pattern.DebugString();
  // VLOG(2) << "match=" << match->DebugString();
  bool pattern_matched = false;
  if (pattern.op == "*") {
    pattern_matched = true;
  } else {
    std::vector<string> pattern_ops = str_util::Split(pattern.op, '|');
    for (const string& pattern_op : pattern_ops) {
      if (node->type_string() == pattern_op) {
        pattern_matched = true;
      }
    }
  }
  if (!pattern_matched) {
    VLOG(1) << "node.op() != pattern.op()";
    return false;
  }
  match->edge = edge;
  // Ignore any control inputs for pattern-matching purposes
  std::vector<const Edge*> non_control_inputs;
  for (auto input : node->in_edges()) {
    if (!input->IsControlEdge()) {
      non_control_inputs.push_back(input);
    }
  }
  if (pattern.inputs.empty()) {
    // If there are no inputs, assume that's the end of the pattern.
    return true;
  }
  if (non_control_inputs.size() != pattern.inputs.size()) {
    VLOG(0) << "non_control_inputs.size() != pattern.inputs.size()";
    return false;
  }
  std::sort(non_control_inputs.begin(), non_control_inputs.end(),
      [](const Edge* a, const Edge* b) {
        return a->dst_input() < b->dst_input();
      });
  bool reverse_check = false;
  for (int i = 0; i < pattern.inputs.size(); ++i) {
    const Edge* input_edge = non_control_inputs[i];
    const OpTypePattern& input_pattern = pattern.inputs[i];
    match->inputs.push_back(NodeMatch());
    NodeMatch* input_match = &(match->inputs.back());
    if (!DoesOpTypeMatch(input_edge, input_pattern, input_match)) {
      if (IsBinaryOp(node->type_string())) {
        VLOG(1) << "uncertain binary op input order, reverse check";
        reverse_check = true;
        break;
      }
      return false;
    }
  }
  if (reverse_check && IsBinaryOp(node->type_string())) {
    VLOG(1) << "check reverse binary op";
    match->inputs.clear();
    for (int i = 0; i < pattern.inputs.size(); ++i) {
      const Edge* input_edge = non_control_inputs[pattern.inputs.size()-1-i];
      const OpTypePattern& input_pattern = pattern.inputs[i];
      match->inputs.push_back(NodeMatch());
      NodeMatch* input_match = &(match->inputs.back());
      if (!DoesOpTypeMatch(input_edge, input_pattern, input_match)) {
        return false;
      }
    }
  }
  return true;
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
  unsigned int num_split = split->def().attr().at("num_split").i();
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

bool GetAttentionPattern(Node* concat, AttentionPattern& pattern) {
  for (auto e:concat->in_edges()) {
    if (e->src()->type_string() == "Const") continue;
    if (e->src()->type_string() != "BatchMatMulV2") return false;
    pattern.concat = concat;
    NodeMatch match;
    if (DoesOpTypeMatch(e, attention_path_pattern, &match)) {
      VLOG(1) << "match attention path!!";
      AttentionPattern::PathPattern path_pattern;
      path_pattern.gemm_tail = match.edge;
      path_pattern.add = match.inputs[0].edge;
      path_pattern.softmax = match.inputs[0].inputs[0].edge;
      path_pattern.add_y = match.inputs[0].inputs[1].edge;
      path_pattern.gemm_pre = match.inputs[0].inputs[0].inputs[0].edge;
      // biasAdd
      path_pattern.gemm_pre_input_a = match.inputs[0].inputs[0].inputs[0].inputs[0].edge;
      // Split
      path_pattern.gemm_pre_input_b = match.inputs[0].inputs[0].inputs[0].inputs[1].edge;
      path_pattern.split = match.inputs[1].edge;
      path_pattern.split_dim = match.inputs[1].inputs[0].edge;
      path_pattern.reshape = match.inputs[1].inputs[1].edge;
      path_pattern.input = match.inputs[1].inputs[1].inputs[0].edge;
      path_pattern.shape = match.inputs[1].inputs[1].inputs[1].edge;
      pattern.path.push_back(std::move(path_pattern));
      if (path_pattern.split->src() != path_pattern.gemm_pre_input_b->src() ||
          path_pattern.split->src_output() != path_pattern.gemm_pre_input_b->src_output()) {
        return false;
      }
    } else {
      return false;
    }
  }

  for (auto e:concat->out_edges()) {
    pattern.output.push_back(e);
  }
  pattern.split = pattern.path[0].split->src();
  for (auto path:pattern.path) {
    if (pattern.split != path.split->src()) {
     VLOG(0) << "split not same: "
             << pattern.split->DebugString()
             << " VS " << path.split->src()->DebugString();
       return false;
    }
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
                                            iter.second.weight,
                                            matmul->name() + "/merge_weight");
    TF_RETURN_FALSE_IF_NULL(merged_weight, "merge weight")
    graph->UpdateEdge(merged_weight, 0, matmul, 1);
    // 3.split为多个输出
    Node* split = ConstructSplitOp(graph, matmul, iter.second.weight.size(),
                                  weight.dims() - 1, iter.second.output,
                                  matmul->name() + "/split_output");
    TF_RETURN_FALSE_IF_NULL(split, "split output")
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
    VLOG(0) << "merge gemm " << matmul->name() << ", size "
            << iter.second.weight.size();
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
                                          iter.second.bias,
                                          add->name() + "/merge_bias");
    TF_RETURN_FALSE_IF_NULL(merged_bias, "merge bias")
    graph->UpdateEdge(merged_bias, 0, add, 1);
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
    TF_RETURN_FALSE_IF_NULL(new_shape, "merge shape")
    VLOG(1) << new_shape->DebugString();
    Node* reshape = iter.second.reshape[0];
    graph->UpdateEdge(add, 0, reshape, 0);
    graph->UpdateEdge(new_shape, 0, reshape, 1);
    // 3.split为多个输出
    Node* split = ConstructSplitOp(graph, reshape, iter.second.bias.size(),
                                  new_shape_t.NumElements() - 1,
                                  iter.second.output,
                                  reshape->name() + "/split_output");
    TF_RETURN_FALSE_IF_NULL(split, "split output")
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
    VLOG(0) << "merge BiasAdd " << add->name() << ", size "
            << iter.second.bias.size();
  }
  VLOG(0) << "merge BiasAdd done";
  return true;
}

template<typename T>
bool SetShapeTensor(Tensor& old_shape_t, Tensor& new_shape_t, int reshape_dims,
                    int& last_dim_size, int& first_dim_size, int parallel) {
  auto old_data = old_shape_t.flat<T>();
  auto new_data = new_shape_t.flat<T>();
  for (auto i = 0; i < reshape_dims - 1; i++) {
    new_data(i) = old_data(i);
  }
  new_data(reshape_dims - 1) = parallel;
  last_dim_size = old_data(reshape_dims - 1);
  new_data(reshape_dims) = last_dim_size / parallel;
  first_dim_size = old_data(0);
  return true;
}

bool MergeAttention(Graph* graph) {
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  Status status;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  VLOG(1) << "start to merge attention node, " << nodes.size();

  std::map<std::string, AttentionPattern> collection;
  for (Node* node : nodes) {
    if (node->type_string() != "ConcatV2") continue;
    std::string key = node->name();
    AttentionPattern pattern;
    if (collection.find(key) != collection.end()) {
      continue;
    }
    if (GetAttentionPattern(node, pattern)) {
      VLOG(1) << "find " << key;
      std::sort(pattern.path.begin(), pattern.path.end(),
          [](AttentionPattern::PathPattern& a,
             AttentionPattern::PathPattern& b) {
            return a.split->src_output() < b.split->src_output();
          });
      collection[key] = std::move(pattern);
    }
  }
  if (VLOG_IS_ON(1)) DebugAttentionPattern(collection);
  for (auto iter:collection) {
    int parallel = iter.second.path.size();
    AttentionPattern::PathPattern& reserve_path = iter.second.path[0];
    // 1.constrcut new shape
    Node* old_shape = reserve_path.shape->src();
    Tensor old_shape_t;
    if (!GetConstTensor(old_shape, old_shape_t)) {
      return false;
    }
    int reshape_dims = old_shape_t.NumElements();
    int first_dim_size;
    int last_dim_size;
    Tensor new_shape_t(old_shape_t.dtype(), {reshape_dims + 1});
    if (new_shape_t.dtype() == DT_INT32) {
      SetShapeTensor<int32>(old_shape_t, new_shape_t, reshape_dims,
                            last_dim_size, first_dim_size, parallel);
    } else {
      SetShapeTensor<int64>(old_shape_t, new_shape_t, reshape_dims,
                            last_dim_size, first_dim_size, parallel);
    }
    string new_shape_name = reserve_path.reshape->src()->name() + "/extend_shape";
    Node* new_shape = CreateConstNode(graph, new_shape_name, new_shape_t, old_shape);
    graph->UpdateEdge(new_shape, 0, reserve_path.reshape->src(), 1);

    // 2.construct new transpose
    Tensor perm_t(DT_INT32, {reshape_dims + 1});
    auto perm_data = perm_t.flat<int32>();
    for (auto i = 1; i <= reshape_dims - 1; i++) {
      perm_data(i) = i - 1;
    }
    perm_data(0) = reshape_dims - 1;
    perm_data(reshape_dims) = reshape_dims;
    string transpose_name = reserve_path.reshape->src()->name() + "/transpose";
    Node* transpose = ConstructTransposeOp(graph, reserve_path.reshape,
                                           transpose_name, perm_t);

    status = graph->UpdateEdge(transpose, 0, reserve_path.gemm_pre->src(), 1);
    if (!status.ok()) {
      LOG(WARNING) << "update edge failed: " << status.ToString();
      return false;
    }
    status = graph->UpdateEdge(transpose, 0, reserve_path.gemm_tail->src(), 1);
    if (!status.ok()) {
      LOG(WARNING) << "update edge failed: " << status.ToString();
      return false;
    }

    // 3.Pack BiasAdd
    std::vector<const Edge*> in_edges;
    for (auto path:iter.second.path) {
      in_edges.push_back(path.gemm_pre_input_a);
    }
    string pack_name = reserve_path.gemm_pre->src()->name() + "_pack_input";
    Node* pack = ConstructPackOp(graph, reserve_path.gemm_pre->src(),
                                 pack_name, in_edges);
    Status status = graph->UpdateEdge(pack, 0, reserve_path.gemm_pre->src(), 0);
    TF_RETURN_FALSE_IF_ERROR(status, "update pack output edge")

    // 4.re-transpose
    for (auto i = 0; i <= reshape_dims - 2; i++) {
      perm_data(i) = i + 1;
    }
    perm_data(reshape_dims - 1) = 0;
    perm_data(reshape_dims) = reshape_dims;
    string re_transpose_name = reserve_path.gemm_tail->src()->name() + "/transpose";
    Node* re_transpose = ConstructTransposeOp(graph, reserve_path.gemm_tail,
                                              re_transpose_name, perm_t);

    // 5.reshape
    Tensor re_shape_t(DT_INT32, {2});
    auto re_shape_data = re_shape_t.flat<int32>();
    re_shape_data(0) = first_dim_size;
    re_shape_data(1) = last_dim_size;
    string re_shape_name = re_transpose->name() + "/reshape";
    Node* re_reshape = ConstructReshapeOp(graph, re_transpose,
                                          re_shape_name, re_shape_t);
    for (auto e:iter.second.output) {
      graph->UpdateEdge(re_reshape, 0, e->dst(), e->dst_input());
    }

    // 6.delete node
    for (int i = 1; i < iter.second.path.size(); i++) {
      graph->RemoveNode(iter.second.path[i].gemm_pre->src());
      graph->RemoveNode(iter.second.path[i].softmax->src());
      graph->RemoveNode(iter.second.path[i].add->src());
      graph->RemoveNode(iter.second.path[i].gemm_tail->src());
    }
    graph->RemoveNode(iter.second.split);
    graph->RemoveNode(iter.second.concat);
  }
  VLOG(0) << "merge " << collection.size() << " attention pattern";
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
    string name = "before_merge_gemm_" + std::to_string(pass) + ".pb";
    DumpModelFile(*optimized_graph, name);
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
    string name = "after_merge_gemm_" + std::to_string(pass) + ".pb";
    DumpModelFile(*optimized_graph, name);
  }
  pass++;
  return Status::OK();
}

void MergeGemmOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
}

Status MergeGemmOptimizerSecondStage::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool opt = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE", true, &opt);
  if (!opt) {
    *optimized_graph = item.graph;
    return Status::OK();
  }
  static int pass = 0;
  VLOG(0) << "MergeGemmOptimizerSecondStage is on." << pass;

  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());
  Graph graph(flib);
  Status status = ConvertGraphDefToGraph(GraphConstructorOptions(),
                                  item.graph, &graph);
  if (!status.ok()) {
    LOG(WARNING) << "ConvertGraphDefToGraph failed: " << status.ToString();
    *optimized_graph = item.graph;
    return Status::OK();
  }

  if (!MergeAttention(&graph)) {
    LOG(WARNING) << " merge attention failed";
    *optimized_graph = item.graph;
    return Status::OK();
  }
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

  return Status::OK();
}

void MergeGemmOptimizerSecondStage::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
