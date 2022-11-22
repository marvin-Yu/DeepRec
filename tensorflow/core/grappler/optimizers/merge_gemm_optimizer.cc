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

struct AttentionPattern {
  Node* concat;
  struct PathPattern {
    Node* query_split;
    const Edge* query_split_edge;
    Node* query_shape;
    Node* query_reshape;
    Node* query_input;
    Node* fact_split;
    const Edge* fact_split_edge;
    Node* fact_shape;
    Node* fact_reshape;
    Node* fact_input;
    Node* gemm_pre;
    Node* softmax;
    Node* add;
    Node* add_y;
    Node* gemm_tail;
    void DebugPattern() {
      VLOG(0) << "query split:" << query_split_edge->DebugString();
      VLOG(0) << "query shape:" << query_shape->DebugString();
      VLOG(0) << "fact split:" << fact_split_edge->DebugString();
      VLOG(0) << "fact shape:" << fact_shape->DebugString();
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

string EdgeMatch::DebugString() const {
  string result = "{";
  if (edge != nullptr && edge->src() != nullptr) result += edge->src()->DebugString();
  result += ", {";
  for (const EdgeMatch& input : inputs) {
    result += input.DebugString() + ",";
  }
  result += "}}";
  return result;
}

// target node is the src of edge
bool DoesEdgeMatchOpType(const Edge* edge, const OpTypePattern& pattern,
                         EdgeMatch* match) {
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
  match->node = node;
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
    match->inputs.push_back(EdgeMatch());
    EdgeMatch* input_match = &(match->inputs.back());
    if (!DoesEdgeMatchOpType(input_edge, input_pattern, input_match)) {
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
      match->inputs.push_back(EdgeMatch());
      EdgeMatch* input_match = &(match->inputs.back());
      if (!DoesEdgeMatchOpType(input_edge, input_pattern, input_match)) {
        return false;
      }
    }
  }
  return true;
}

bool DoesEdgeMatchOpType(const Node* node, const OpTypePattern& pattern,
                         EdgeMatch* match) {
  if (node->out_edges().size() < 1) return false;
  return DoesEdgeMatchOpType(*(node->out_edges().begin()), pattern, match);
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
        VLOG(0) << "const shape not match: " << base.shape().DebugString()
                << " VS " << cmp.shape().DebugString();
        return false;
      }
      if (check_value) {
        return base.DebugString() == cmp.DebugString();
      }
    }
  }
  return true;
}

struct MultiHeadPattern {
  std::vector<Node*> batch_matmul;

  struct QKVPattern {
    Node* input;
    std::vector<Node*> arr_shape1;
    std::vector<Node*> arr_reshape1;
    std::vector<Node*> arr_weight;
    std::vector<Node*> arr_matmul;
    std::vector<Node*> arr_shape2;
    std::vector<Node*> arr_reshape2;
    std::vector<Node*> arr_bias;
    std::vector<Node*> arr_add;
    std::vector<Node*> arr_gather;
    std::vector<Node*> arr_gather_ind;
    std::vector<std::vector<const Edge*>> outputs;
    QKVPattern() {
      input = nullptr;
    }

    void DebugPattern(string prefix) {
      VLOG(0) << prefix << " input:" << input->DebugString();
      VLOG(0) << prefix << " head count:" << arr_matmul.size();
    }

    bool PushQKVPattern(EdgeMatch* match) {
      if (input == nullptr) {
        input = match->inputs[0].inputs[0].inputs[0].inputs[0].node;
      } else {
        Node* new_input = match->inputs[0].inputs[0].inputs[0].inputs[0].node;
        if (new_input != input) {
          LOG(WARNING) << "qkv input not match: " << input->name()
                       << input->name();
          return false;
        }
      }
      arr_shape1.push_back(match->inputs[0].inputs[0].inputs[0].inputs[1].node);
      arr_reshape1.push_back(match->inputs[0].inputs[0].inputs[0].node);
      arr_weight.push_back(match->inputs[0].inputs[0].inputs[1].node);
      arr_matmul.push_back(match->inputs[0].inputs[0].node);
      arr_shape2.push_back(match->inputs[0].inputs[1].node);
      arr_reshape2.push_back(match->inputs[0].node);
      arr_bias.push_back(match->inputs[1].node);
      arr_add.push_back(match->node);
      std::vector<const Edge*> out_edges;
      for (auto e:match->node->out_edges()) {
        out_edges.push_back(e);
      }
      outputs.push_back(std::move(out_edges));
      return true;
    }

    bool MergeQKV(Graph* graph) {
      // 1.concat weight input
      // 2.连接concat到其中一个matmul op，其余删除
      if (!IsSameConst(arr_weight, false) || !IsSameConst(arr_bias, false)) {
        DebugPattern("weight/bias not same");
        return false;
      }
      if (!IsSameConst(arr_shape2, true)) {
        DebugPattern("shape before bias add not same");
        return false;
      }

      Node* matmul = arr_matmul[0];
      Tensor t_weight;
      if (!GetConstTensor(arr_weight[0], t_weight)) {
        return false;
      }
      Node* merged_weight = ConstructConcatOp(graph, matmul, t_weight.dims() - 1, arr_weight,
                                              matmul->name() + "/merge_weight");
      TF_RETURN_FALSE_IF_NULL(merged_weight, "merge weight")
      graph->UpdateEdge(merged_weight, 0, matmul, 1);
      // 3.concat bias
      Node* add = arr_add[0];
      Tensor t_bias;
      if (!GetConstTensor(arr_bias[0], t_bias)) {
        return false;
      }
      if (t_bias.dims() != 1) {
        LOG(WARNING) << "bias dim not equals to 1: " << t_bias.shape().DebugString();
        return false;
      }
      Node* merged_bias = ConstructConcatOp(graph, add, t_bias.dims() - 1,
                                            arr_bias, add->name() + "/merge_bias");
      TF_RETURN_FALSE_IF_NULL(merged_bias, "merge bias")
      graph->UpdateEdge(merged_bias, 0, add, 1);
      graph->UpdateEdge(matmul, 0, add, 0);
      // 2.构建新的shape
      Node* old_shape = arr_shape2[0];
      Tensor new_shape_t;
      if (!GetConstTensor(old_shape, new_shape_t)) {
        return false;
      }
      int change_dim = new_shape_t.NumElements() - 1;
      if (new_shape_t.dtype() == DT_INT32) {
        auto data = new_shape_t.flat<int32>();
        data(change_dim) = data(change_dim) * arr_add.size();

      } else {
        auto data = new_shape_t.flat<int64>();
        data(change_dim) = data(change_dim) * arr_add.size();
      }
      string new_shape_name = old_shape->name() + "/merge_shape";
      Node* new_shape = CreateConstNode(graph, new_shape_name, new_shape_t,
                        old_shape->def().device(), old_shape->assigned_device_name());
      TF_RETURN_FALSE_IF_NULL(new_shape, "merge shape")
      VLOG(1) << new_shape->DebugString();
      Node* reshape = arr_reshape2[0];
      graph->UpdateEdge(add, 0, reshape, 0);
      graph->UpdateEdge(new_shape, 0, reshape, 1);
      // 3.split为多个输出
      Node* split = ConstructSplitOp(graph, reshape, arr_bias.size(),
                                    new_shape_t.NumElements() - 1,
                                    outputs, reshape->name() + "/split_output");
      TF_RETURN_FALSE_IF_NULL(split, "split output")
      // 4.删除多余节点
      for (auto n:arr_matmul) {
        if (n != matmul) {
          graph->RemoveNode(n);
        }
      }
      for (auto n:arr_add) {
        if (n != add) {
          graph->RemoveNode(n);
        }
      }
      for (auto n:arr_reshape2) {
        if (n != reshape) {
          graph->RemoveNode(n);
        }
      }
      VLOG(0) << "merge qkv " << input->name() << ", size "
              << arr_bias.size();
      return true;
    }
  };

  QKVPattern query;
  QKVPattern fact;

  bool MatchedGatherPattern() {
    return fact.arr_gather.size() > 0;
  }

  bool PushMultiHeadPattern(EdgeMatch& match, bool matched_gather) {
    EdgeMatch* query_match = &(match.inputs[0]);
    EdgeMatch* fact_match;
    if (matched_gather) {
      fact_match = &(match.inputs[1].inputs[0]);
      fact.arr_gather.push_back(match.inputs[1].node);
      fact.arr_gather_ind.push_back(match.inputs[1].inputs[1].node);
    } else {
      fact_match = &(match.inputs[1]);
    }
    VLOG(1) << "push qkv pattern, match gather pattern:" << matched_gather;
    query.PushQKVPattern(query_match);
    fact.PushQKVPattern(fact_match);
    batch_matmul.push_back(match.node);
    return true;
  }

  bool DoMerge(Graph* graph) {
    bool result = query.MergeQKV(graph);
    fact.MergeQKV(graph);
    return true;
  }
  void DebugPattern() {
    VLOG(0) << "match gather pattenr: " << MatchedGatherPattern();
    query.DebugPattern("query");
    fact.DebugPattern("fact");
  }
};

void DebugMultiHeadPattern(std::map<std::string, MultiHeadPattern>& collection) {
  for (auto iter:collection) {
    VLOG(0) << iter.first << "   ******************************************";
    iter.second.DebugPattern();
  }
}

bool GetAttentionPattern(Node* concat, AttentionPattern& pattern) {
  for (auto e:concat->in_edges()) {
    if (e->src()->type_string() == "Const") continue;
    if (e->src()->type_string() != "BatchMatMulV2") return false;
    pattern.concat = concat;
    EdgeMatch match;
    if (DoesEdgeMatchOpType(e, attention_path_pattern, &match)) {
      VLOG(1) << "match attention path!!";
      AttentionPattern::PathPattern path_pattern;
      path_pattern.gemm_tail = match.node;
      path_pattern.add = match.inputs[0].node;
      path_pattern.softmax = match.inputs[0].inputs[0].node;
      path_pattern.add_y = match.inputs[0].inputs[1].node;
      path_pattern.gemm_pre = match.inputs[0].inputs[0].inputs[0].node;
      // query split
      path_pattern.query_split = match.inputs[0].inputs[0].inputs[0].inputs[0].node;
      path_pattern.query_split_edge = match.inputs[0].inputs[0].inputs[0].inputs[0].edge;
      path_pattern.query_reshape = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
      path_pattern.query_input = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].inputs[0].node;
      path_pattern.query_shape = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].inputs[1].node;
      // fact Split
      path_pattern.fact_split = match.inputs[0].inputs[0].inputs[0].inputs[1].node;
      path_pattern.fact_split_edge = match.inputs[0].inputs[0].inputs[0].inputs[0].edge;
      path_pattern.fact_reshape = match.inputs[0].inputs[0].inputs[0].inputs[1].inputs[1].node;
      path_pattern.fact_input = match.inputs[0].inputs[0].inputs[0].inputs[1].inputs[1].inputs[0].node;
      path_pattern.fact_shape = match.inputs[0].inputs[0].inputs[0].inputs[1].inputs[1].inputs[1].node;
      pattern.path.push_back(std::move(path_pattern));
      if (path_pattern.query_split_edge->src_output() != path_pattern.fact_split_edge->src_output()) {
        LOG(WARNING) << "query split port not equal to fact:"
                     << path_pattern.query_split_edge->DebugString()
                     << path_pattern.fact_split_edge->DebugString();
        return false;
      }
    } else {
      return false;
    }
  }

  for (auto e:concat->out_edges()) {
    pattern.output.push_back(e);
  }
  for (int i = 1; i < pattern.path.size(); ++i) {
    if (pattern.path[0].query_split != pattern.path[i].query_split) {
     VLOG(0) << "query split not same: "
             << pattern.path[0].query_split->DebugString()
             << " VS " << pattern.path[i].query_split->DebugString();
       return false;
    }
    if (pattern.path[0].fact_split != pattern.path[i].fact_split) {
     VLOG(0) << "fact split not same: "
             << pattern.path[0].fact_split->DebugString()
             << " VS " << pattern.path[i].fact_split->DebugString();
       return false;
    }
  }
  return true;
}

void GetMultiHeadPattern(const Node* batch_matmul,
                         std::map<std::string, MultiHeadPattern>& collection) {
  EdgeMatch match_multi_head;
  EdgeMatch match_multi_head_gather;
  EdgeMatch* match = &match_multi_head_gather;
  bool matched_gather = false;
  Node* fact_input;
  if (DoesEdgeMatchOpType(batch_matmul, multi_head_gather_pattern, match)) {
    matched_gather = true;
    fact_input = match->inputs[1].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
  } else if (DoesEdgeMatchOpType(batch_matmul, multi_head_pattern, &match_multi_head)) {
    match = &match_multi_head;
    fact_input = match->inputs[1].inputs[0].inputs[0].inputs[0].inputs[0].node;
  } else {
    return;
  }
 
  string key = fact_input->name();
  if (collection.find(key) != collection.end()) {
    if (!collection[key].PushMultiHeadPattern(*match, matched_gather)) {
      LOG(WARNING) << "push multi-head pattern faild: " << key;
    }
  } else {
    MultiHeadPattern pattern;
    if (!pattern.PushMultiHeadPattern(*match, matched_gather)) {
      LOG(WARNING) << "push multi-head pattern faild: " << key;
    }
    collection[key] = std::move(pattern);
  }
}

// merge gemm,reshape and biasadd
//   Reshape   Reshape
//     |         |                     MatMul
//     |         |                       | concat const
//   MatMul    MatMul                    | /
//     | shape   | shape              BiasAdd
//     | /       | /                     | new shape
//  Reshape   Reshape        -->         | /
//     | const1  | const2             Reshape
//     | /       |  /                    |
//  BiasAdd   BiasAdd                  Split
//     |         |                      /  \
//   out1       out2                  out1 out2
bool MergeMultiHead(Graph* graph) {
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  Status status;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  std::map<std::string, MultiHeadPattern> collection;
  VLOG(1) << "start to merge multi-head, " << nodes.size();
  for (Node* node : nodes) {
    if (node->type_string() != "BatchMatMulV2") continue;
    GetMultiHeadPattern(node, collection);
  }
  if (VLOG_IS_ON(1)) DebugMultiHeadPattern(collection);
  for (auto iter:collection) {
    iter.second.DoMerge(graph);
  }
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
            return a.query_split_edge->src_output() < b.query_split_edge->src_output();
          });
      collection[key] = std::move(pattern);
    }
  }
  if (VLOG_IS_ON(1)) DebugAttentionPattern(collection);
  for (auto iter:collection) {
    auto handle_pre_gemm_input = [](Graph* graph, Node* old_shape, Node* reshape,
                                    Node* gemm_pre, Node* gemm_tail,
                                    int& first_dim_size, int& last_dim_size,
                                    int& reshape_dims, int parallel, int port)->bool {
      // 1.constrcut new shape
      Tensor old_shape_t;
      if (!GetConstTensor(old_shape, old_shape_t)) {
        return false;
      }
      reshape_dims = old_shape_t.NumElements();
      Tensor new_shape_t(old_shape_t.dtype(), {reshape_dims + 1});
      // [-1, seq_len, dim*p] -> [-1, seq_len, p, dim]
      if (new_shape_t.dtype() == DT_INT32) {
        SetShapeTensor<int32>(old_shape_t, new_shape_t, reshape_dims,
                              last_dim_size, first_dim_size, parallel);
      } else {
        SetShapeTensor<int64>(old_shape_t, new_shape_t, reshape_dims,
                              last_dim_size, first_dim_size, parallel);
      }
      string new_shape_name = reshape->name() + "/extend_shape";
      Node* new_shape = CreateConstNode(graph, new_shape_name, new_shape_t,
                        old_shape->def().device(), old_shape->assigned_device_name());
      graph->UpdateEdge(new_shape, 0, reshape, 1);

      // 2.construct new transpose
      Tensor perm_t(DT_INT32, {reshape_dims + 1});
      auto perm_data = perm_t.flat<int32>();
      for (auto i = 1; i <= reshape_dims - 1; i++) {
        perm_data(i) = i - 1;
      }
      perm_data(0) = reshape_dims - 1;
      perm_data(reshape_dims) = reshape_dims;
      string transpose_name = reshape->name() + "/transpose";
      Node* transpose = ConstructTransposeOp(graph, reshape, 0,
                                             transpose_name, perm_t);
      Status status = graph->UpdateEdge(transpose, 0, gemm_pre, port);
      TF_RETURN_FALSE_IF_ERROR(status, "update edge failed")
      // query out not feed to last gemm
      if (port == 1) {
        status = graph->UpdateEdge(transpose, 0, gemm_tail, port);
        TF_RETURN_FALSE_IF_ERROR(status, "update edge failed")
      }
      return true;
    };
    int parallel = iter.second.path.size();
    AttentionPattern::PathPattern& reserve_path = iter.second.path[0];
    int query_first_dim_size = 0;
    int query_last_dim_size = 0;
    int query_reshape_dims = 0;
    if (!handle_pre_gemm_input(graph, reserve_path.query_shape,
                               reserve_path.query_reshape,
                               reserve_path.gemm_pre,
                               reserve_path.gemm_tail,
                               query_first_dim_size, query_last_dim_size,
                               query_reshape_dims, parallel, 0)) {
      return false;
    }
    int fact_first_dim_size = 0;
    int fact_last_dim_size = 0;
    int fact_reshape_dims = 0;
    if (!handle_pre_gemm_input(graph, reserve_path.fact_shape,
                               reserve_path.fact_reshape,
                               reserve_path.gemm_pre,
                               reserve_path.gemm_tail,
                               fact_first_dim_size, fact_last_dim_size,
                               fact_reshape_dims, parallel, 1)) {
      return false;
    }
    if (query_first_dim_size != fact_first_dim_size ||
        query_last_dim_size != query_last_dim_size ||
        query_reshape_dims != fact_reshape_dims) {
      LOG(WARNING) << "query first/last/reshape dim size not equal with fact"
                   << query_first_dim_size << " VS " << fact_first_dim_size
                   << query_last_dim_size << " VS " << fact_last_dim_size
                   << query_reshape_dims << " VS " << fact_reshape_dims;
      return false;
    }

    // 4.re-transpose
    Tensor perm_t(DT_INT32, {fact_reshape_dims + 1});
    auto perm_data = perm_t.flat<int32>();
    for (auto i = 0; i <= fact_reshape_dims - 2; i++) {
      perm_data(i) = i + 1;
    }
    perm_data(fact_reshape_dims - 1) = 0;
    perm_data(fact_reshape_dims) = fact_reshape_dims;
    string re_transpose_name = reserve_path.gemm_tail->name() + "/transpose";
    Node* re_transpose = ConstructTransposeOp(graph, reserve_path.gemm_tail, 0,
                                              re_transpose_name, perm_t);

    // 5.reshape
    Tensor re_shape_t(DT_INT32, {2});
    auto re_shape_data = re_shape_t.flat<int32>();
    re_shape_data(0) = fact_first_dim_size;
    re_shape_data(1) = fact_last_dim_size;
    string re_shape_name = re_transpose->name() + "/reshape";
    Node* re_reshape = ConstructReshapeOp(graph, re_transpose,
                                          re_shape_name, re_shape_t);
    for (auto e:iter.second.output) {
      graph->UpdateEdge(re_reshape, 0, e->dst(), e->dst_input());
    }

    // 6.delete node
    for (int i = 1; i < iter.second.path.size(); i++) {
      graph->RemoveNode(iter.second.path[i].gemm_pre);
      graph->RemoveNode(iter.second.path[i].softmax);
      graph->RemoveNode(iter.second.path[i].add);
      graph->RemoveNode(iter.second.path[i].gemm_tail);
    }
    graph->RemoveNode(iter.second.path[0].query_split);
    graph->RemoveNode(iter.second.path[0].fact_split);
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
  VLOG(0) << "MergeGemmOptimizer is on.";
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
  if (!MergeMultiHead(&graph)) {
    LOG(WARNING) << " merge multi head failed";
    *optimized_graph = item.graph;
    return Status::OK();
  }
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

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
  if (VLOG_IS_ON(1)) {
    string name = "after_merge_gemm_" + std::to_string(pass) + ".pb";
    DumpModelFile(*optimized_graph, name);
  }
  pass++;

  return Status::OK();
}

void MergeGemmOptimizerSecondStage::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
