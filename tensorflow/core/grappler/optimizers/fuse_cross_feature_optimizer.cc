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

#include "tensorflow/core/grappler/optimizers/fuse_cross_feature_optimizer.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"

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

#define CHECK_NULL(target)   \
  if (target == nullptr) {   \
    return errors::Internal("got nullptr!"); \
  }                          \

struct CoActionPattern {
  std::vector<const Edge*> input_a;
  std::vector<const Edge*> input_b;
  std::vector<Node*> co_action;
  std::vector<const Edge*> output;
};

void DebugCoActionPattern(std::map<std::string, CoActionPattern>& collection) {
  for (auto iter:collection) {
    VLOG(0) << iter.first << ", " << iter.second.co_action.size();
    VLOG(0) << "*************************************************";
    VLOG(0) << "input a:";
    for (auto e:iter.second.input_a) VLOG(0) << e->DebugString();
    VLOG(0) << "input b:";
    for (auto e:iter.second.input_b) VLOG(0) << e->DebugString();
    VLOG(0) << "co action:";
    for (auto n:iter.second.co_action) VLOG(0) << n->DebugString();
    VLOG(0) << "output:";
    for (auto e:iter.second.output) VLOG(0) << e->DebugString();
  }
}

void GetAllMatchNodes(std::vector<NodeDef>& nodes, std::set<string>& node_set, const NodeMatch& match) {
  if (!node_set.count(match.node.name())) {
    nodes.push_back(match.node);
    node_set.insert(match.node.name());
  }
  for (const NodeMatch& input : match.inputs) {
    GetAllMatchNodes(nodes, node_set, input);
  }
  return;
}

bool ReplaceCoAction(GraphDef &input_graph_def, GraphDef* output_graph_def, int& count) {
  VLOG(1) << "start to replace co_action model, " << cross_feature_pattern.DebugString();
  bool is_changed = false;
  Status status = ReplaceMatchingOpTypes(
      input_graph_def,
      cross_feature_pattern,
      [&is_changed, &count](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        VLOG(1) << "match co aciton-------------------------";
        // 1. 匹配到pattern
        // 2. 获取有用的节点
        //    输入节点是StridedSlice和Reshape
        //    输入节点的前序节点都是要保留的，匹配是为了做合法性检查和修改语义
        const NodeDef& concat_node = match.node;
        // user stridedslice
        const NodeDef& strideslice1_node = match.inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& slice1_const1_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& slice1_const2_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].node;
        const NodeDef& slice1_const3_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[3].node;
        const NodeDef& gather_user_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_ind_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& gather_const_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].node;
        const NodeDef& reshape_node = match.inputs[0].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& const_node1 = match.inputs[0].inputs[1].node;
        const NodeDef& const_node2 = match.inputs[1].inputs[1].node;
        const NodeDef& const_node3 = match.inputs[2].node;
        // ad stridedslice
        const NodeDef& strideslice2_node = match.inputs[0].inputs[0].inputs[0].inputs[1].inputs[0].node;
        // [-1, 5, 4] -> [-1 , 1, 5, 4]
        const NodeDef& reshape_const_node = match.inputs[0].inputs[0].inputs[0].inputs[1].inputs[1].node;

        std::vector<NodeDef> match_nodes;
        std::set<string> node_set;
        GetAllMatchNodes(match_nodes, node_set, match);
        VLOG(1) << "match nodes number:" << match_nodes.size();

        // 3. 检查placeholder和Reshape shape const 值
        string indicator_name = gather_ind_node.name();
        if (gather_ind_node.op() == "GatherV2") {
          indicator_name = gather_ind_node.input(0);
        }
        VLOG(1) << "indicator is " << indicator_name;
        if (indicator_name.find("user_creative_indicator") == string::npos &&
            indicator_name.find("user_ad_indicator") == string::npos &&
            indicator_name.find("cate_ad_indicator") == string::npos &&
            indicator_name.find("nick_cate_indicators") == string::npos) {
            VLOG(0) << "gather input indicator placeholder not match:" << indicator_name;
            new_nodes->insert(new_nodes->end(), match_nodes.begin(), match_nodes.end());
            return Status::OK();
        }
        Tensor reshape_const_tensor = GetNodeTensorAttr(reshape_const_node, "value");
        auto const_dtype = reshape_const_node.attr().at("dtype").type();
        if (const_dtype == DT_INT32) {
          auto reshape_const_value = reshape_const_tensor.flat<int32>();
          if (reshape_const_value(0) != -1 ||
              reshape_const_value(1) != 5 ||
              reshape_const_value(2) != 4) {
            VLOG(0) << "reshape shape const value not valid:"
                    << ", " << reshape_const_value(0)
                    << ", " << reshape_const_value(1)
                    << ", " << reshape_const_value(2);
          }
        } else if (const_dtype == DT_INT64) {
          auto reshape_const_value = reshape_const_tensor.flat<int64>();
          if (reshape_const_value(0) != -1 ||
              reshape_const_value(1) != 5 ||
              reshape_const_value(2) != 4) {
            VLOG(0) << "reshape shape const value not valid:"
                    << ", " << reshape_const_value(0)
                    << ", " << reshape_const_value(1)
                    << ", " << reshape_const_value(2);
          }
        } else {
          LOG(ERROR) << "reshape const dtype is not int:" << const_dtype;
          new_nodes->insert(new_nodes->end(), match_nodes.begin(), match_nodes.end());
          return Status::OK();
        }

        // 4. 创建Co_action算子, 改造算子语义，支持三维输入，模型第二维Parallel是1，省去reshape成四维的逻辑
        NodeDef co_action_node;
        if (indicator_name.find("nick_cate_indicators") != string::npos ||
            indicator_name.find("cate_ad_indicator") != string::npos) {
          co_action_node.set_op("CoActionIndicator");
        } else {
          co_action_node.set_op("CoAction");
        }
        co_action_node.set_name(concat_node.name() + "_co_action");
        co_action_node.set_device(concat_node.device());
        AddNodeInput(strideslice1_node.name(), &co_action_node);
        AddNodeInput(reshape_node.name(), &co_action_node);
        if (indicator_name.find("nick_cate_indicators") != string::npos ||
            indicator_name.find("cate_ad_indicator") != string::npos) {
          AddNodeInput(gather_ind_node.name(), &co_action_node);
          if (gather_ind_node.op() == "GatherV2") {
            CopyNodeAttr(gather_ind_node, "Tindices", "Tindices", &co_action_node);
          } else {
            CopyNodeAttr(gather_ind_node, "dtype", "Tindices", &co_action_node);
          }
        }
        SetNodeAttr("pow_num", 2, &co_action_node);
        CopyNodeAttr(concat_node, "T", "T", &co_action_node);
        new_nodes->push_back(co_action_node);
        VLOG(1) << "create co_action node:" << co_action_node.DebugString();
        // 5. CoAction算子输出四维，需要创建Reshape，[batch, parallel, pow_num, 4] -> [batch, parallel*pow_num*4]
        NodeDef reshape_co_action_const_node;
        reshape_co_action_const_node.set_op("Const");
        reshape_co_action_const_node.set_name(concat_node.name() + "_shape");
        reshape_co_action_const_node.set_device(concat_node.device());
        CopyNodeAttr(reshape_const_node, "dtype", "dtype", &reshape_co_action_const_node);
        if (const_dtype == DT_INT32) {
          auto reshape_const_value = reshape_const_tensor.flat<int32>();
          SetNodeTensorAttr<int32>("value", {2}, {-1, 2*reshape_const_value(2)},
                                                 &reshape_co_action_const_node);
        } else {
          auto reshape_const_value = reshape_const_tensor.flat<int64>();
          SetNodeTensorAttr<int64>("value", {2}, {-1, 2*reshape_const_value(2)},
                                                 &reshape_co_action_const_node);
        }
        VLOG(1) << "create reshape_co_action_const_node node:" << reshape_co_action_const_node.DebugString();
        new_nodes->push_back(reshape_co_action_const_node);

        NodeDef reshape_co_action_node;
        reshape_co_action_node.set_op("Reshape");
        reshape_co_action_node.set_name(concat_node.name());
        reshape_co_action_node.set_device(concat_node.device());
        AddNodeInput(co_action_node.name(), &reshape_co_action_node);
        AddNodeInput(reshape_co_action_const_node.name(), &reshape_co_action_node);
        CopyNodeAttr(reshape_node, "T", "T", &reshape_co_action_node);
        CopyNodeAttr(reshape_node, "Tshape", "Tshape", &reshape_co_action_node);
        VLOG(1) << "create reshape_co_action_node node:" << reshape_co_action_node.DebugString();
        new_nodes->push_back(reshape_co_action_node);
        
        // 6. 保留匹配的节点
        new_nodes->push_back(strideslice1_node);
        new_nodes->push_back(slice1_const1_node);
        new_nodes->push_back(slice1_const2_node);
        new_nodes->push_back(slice1_const3_node);
        new_nodes->push_back(gather_node);
        new_nodes->push_back(gather_user_node);
        new_nodes->push_back(gather_ind_node);
        new_nodes->push_back(gather_const_node);
        new_nodes->push_back(reshape_node);
        new_nodes->push_back(reshape_const_node);
        new_nodes->push_back(strideslice2_node);
        //new_nodes->push_back(const_node1);
        //new_nodes->push_back(const_node2);
        //new_nodes->push_back(const_node3);
        
        is_changed = true;
        count++;
        return Status::OK();
      },
      {}, output_graph_def, true);
  if (!status.ok()) {
    LOG(ERROR) << "replace cross feature failed " << status;
  }
  return is_changed;

}

bool MergeAdCreativeGather(GraphDef &input_graph_def, GraphDef* output_graph_def, int& count) {
  VLOG(1) << "start to replace co_action model, " << cross_feature_creative_pattern.DebugString();
  bool is_changed = false;
  Status status = ReplaceMatchingOpTypes(
      input_graph_def,
      cross_feature_creative_pattern,
      [&is_changed, &count](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        VLOG(1) << "match creative cross pattern-------------------------";
        const NodeDef& concat_node = match.node;
        const NodeDef& gather_slice = match.inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_slice_ind = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& gather_mul = match.inputs[1].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_mul_ind = match.inputs[1].inputs[0].inputs[0].inputs[0].inputs[1].node;
        // user stridedslice
        const NodeDef& slice_user = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_user = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_user_input = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_user_ind = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& gather_user_axis = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].node;
        const NodeDef& matmul_slice = match.inputs[0].inputs[0].inputs[0].node;
        const NodeDef& matmul_mul = match.inputs[1].inputs[0].inputs[0].node;

        std::vector<NodeDef> match_nodes;
        std::set<string> node_set;
        GetAllMatchNodes(match_nodes, node_set, match);
        VLOG(1) << "match nodes number:" << match_nodes.size();

        bool valid = true;
        // 3. 检查placeholder和Reshape shape const 值
        if (gather_user_ind.name().find("user_ad_indicator") == string::npos &&
            gather_user_ind.name().find("cate_ad_indicator") == string::npos) {
          VLOG(0) << "gather user input indicator placeholder not match:" << gather_user_ind.name();
          valid = false;
        }
        if (gather_slice_ind.name().find("ad_creative_indicator") == string::npos) {
          VLOG(0) << "gather ad to creative indicator placeholder not match:" << gather_slice_ind.name();
          valid = false;
        }
        if (gather_mul_ind.name().find("ad_creative_indicator") == string::npos) {
          VLOG(0) << "gather ad to creative indicator placeholder not match:" << gather_mul_ind.name();
          valid = false;
        }
        if (!valid) {
          new_nodes->insert(new_nodes->end(), match_nodes.begin(), match_nodes.end());
          return Status::OK();
        }

        NodeDef new_matmul_slice;
        new_matmul_slice.CopyFrom(matmul_slice);
        *(new_matmul_slice.mutable_input(0)) = gather_slice.input(0);
        new_nodes->push_back(new_matmul_slice);
        NodeDef new_matmul_mul;
        new_matmul_mul.CopyFrom(matmul_mul);
        *(new_matmul_mul.mutable_input(0)) = gather_mul.input(0);
        new_nodes->push_back(new_matmul_mul);
        NodeDef new_gather_indicator;
        new_gather_indicator.CopyFrom(gather_slice);
        CopyNodeAttr(gather_user, "Tindices", "Tparams", &new_gather_indicator);
        *(new_gather_indicator.mutable_input(0)) = gather_user_ind.name();
        new_nodes->push_back(new_gather_indicator);
        NodeDef new_gather_user;
        new_gather_user.CopyFrom(gather_user);
        *(new_gather_user.mutable_input(1)) = new_gather_indicator.name();
        new_nodes->push_back(new_gather_user);
        // 6. 保留匹配的节点
        for (auto& iter:match_nodes) {
          if (iter.name() != matmul_slice.name() &&
              iter.name() != matmul_mul.name() &&
              iter.name() != gather_mul.name() &&
              iter.name() != gather_slice.name() &&
              iter.name() != gather_user.name()) {
            new_nodes->push_back(iter);
          }
        }

        is_changed = true;
        count++;
        return Status::OK();
      },
      {}, output_graph_def, true);
  if (!status.ok()) {
    LOG(ERROR) << "replace cross feature failed " << status;
  }
  return is_changed;

}

Node* GetTargetOpInputNode(Graph *graph, Node* node, string target) {
  Node* target_node = nullptr;
  for(auto n:node->in_nodes()) {
    VLOG(1) << "GetTargetOpInputNode," << node->name() << " input node:" << n->name();
    if (n->type_string() == target) {
      if (graph->IsValidNode(n).ok()) {
        target_node = n;
      }
      break;
    }
  }
  return target_node;
}

int GetTargetOpInputPort(Node* node, string target) {
  int port = -1;
  for(auto e:node->in_edges()) {
    if (e->src()->type_string() == target) {
      port = e->dst_input();
      break;
    }
  }
  return port;
}

Status RemoveGather(Graph* graph, std::set<string>& skip_merge_nodes) {
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  Status status;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  VLOG(1) << "start to Remove gather node";
  for (Node* node : nodes) {
    if (node->type_string() != "CoAction" &&
        node->type_string() != "CoActionIndicator") continue;
    if (skip_merge_nodes.find(node->name()) != skip_merge_nodes.end()) continue;
    Node *co_action = node;
    Node *strided_slice = GetTargetOpInputNode(graph, co_action, "StridedSlice");
    if (strided_slice == nullptr) {
      LOG(WARNING) << "CoAction node cant find input StridedSlice";
      continue;
    }
    Node* gather = GetTargetOpInputNode(graph, strided_slice, "GatherV2");
    if (gather == nullptr) {
      LOG(WARNING) << "StridedSlice node cant find input Gather";
      continue;
    }
    Node* ind_placeholder = GetTargetOpInputNode(graph, gather, "Placeholder");
    Node* ind_tile = GetTargetOpInputNode(graph, gather, "Tile");
    Node* ind_gather = GetTargetOpInputNode(graph, gather, "GatherV2");
    if (ind_placeholder == nullptr && ind_tile == nullptr && ind_gather == nullptr) {
      LOG(WARNING) << "Gather node cant find input indicator";
      continue;
    }
    const Edge* in_edge = nullptr;
    gather->input_edge(0, &in_edge);
    if (in_edge == nullptr) {
      LOG(WARNING) << "gather in_edge is nullptr";
      continue;
    }
    Node* input_node = in_edge->src();
    if (input_node == nullptr) {
      LOG(WARNING) << "gather input node is nullptr";
      continue;
    }
    int src_port = in_edge->src_output();
    int dst_port = GetTargetOpInputPort(strided_slice, "GatherV2");
    status = graph->UpdateEdge(input_node, src_port, strided_slice, dst_port);
    if (!status.ok()) {
      LOG(WARNING) << "update strided slice input failed";
      return status;
    }
    if (gather->out_edges().empty()) {
      VLOG(1) << gather->name() << " gather all out edges had been update, remove";
      graph->RemoveNode(gather);
    }
  }
  return Status::OK();
}

bool GetCoActionPattern(Node* co_action, CoActionPattern& pattern) {
  pattern.co_action.push_back(co_action);
  const Edge* in;
  co_action->input_edge(0, &in);
  pattern.input_a.push_back(in);
  co_action->input_edge(1, &in);
  pattern.input_b.push_back(in);
  for (auto e: co_action->out_edges()) {
    if (e->dst()->type_string() == "Reshape") {
      pattern.output.push_back(e);
    } else {
      VLOG(0) << "find invalid node:" << e->dst()->DebugString();
      return false;
    }
  }
  return true;
}

Node* ConstructPackOp(Graph* graph, Node* co_action,
                     std::vector<const Edge*>& input_edges, string sufix) {
  string device_name = "/device:CPU:0";
  string pack_name =  co_action->name() + sufix;
  NodeDef pack_node;
  int input_size = input_edges.size();
  VLOG(1) << pack_name << ", size:" << input_size;
  std::vector<NodeDefBuilder::NodeOut> pack_inputs;
  for (auto e:input_edges) {
    pack_inputs.emplace_back(e->src()->name(), e->src_output(), co_action->output_type(0));
  }
  Status status = NodeDefBuilder(pack_name, "Pack")
                                .Input(pack_inputs)
                                .Attr("T", co_action->output_type(0))
                                .Attr("N", input_size)
                                .Attr("axis", 1)
                                .Finalize(&pack_node);
  if (!status.ok()) {
    LOG(ERROR) << "Adding pack nodedef build failed " << status;
    return nullptr;
  }
  pack_node.set_device(device_name);
  VLOG(1) << pack_node.DebugString();
  Node* pack = graph->AddNode(pack_node, &status);
  if (!status.ok()) {
    LOG(ERROR) << "Adding pack node failed " << status;
    return nullptr;
  }
  pack->set_assigned_device_name(device_name);
  int port = 0;
  for (auto e:input_edges) {
    graph->AddEdge(e->src(), e->src_output(), pack, port);
    port++;
    graph->RemoveEdge(e);
  }
  // 减少embedding H2D拷贝次数，将pack之前的输入设备全部设为CPU
  std::unordered_set<string> visited;
  std::queue<Node*> unvisited_queue;
  unvisited_queue.push(pack);
  while(!unvisited_queue.empty()) {
    Node* top = unvisited_queue.front();
    unvisited_queue.pop();
    if (visited.count(top->name()) != 0) continue;
    visited.insert(top->name());
    VLOG(1) << "set device cpu " << top->name();
    top->set_assigned_device_name(device_name);
    for (auto e : top->in_edges()) {
      if (visited.count(e->src()->name()) != 0) continue;
      unvisited_queue.push(e->src());
    }
  }
  return pack;
}

Node* ConstructUnpackOp(Graph* graph, Node* co_action,
                       std::vector<const Edge*>& out_edges, string sufix) {
  string unpack_name =  co_action->name() + sufix;
  NodeDef unpack_node;
  int out_size = out_edges.size();
  VLOG(1) << unpack_name << ", size:" << out_size;
  Status status = NodeDefBuilder(unpack_name, "Unpack")
                                .Input(NodeDefBuilder::NodeOut{co_action->name(), 0, co_action->output_type(0)})
                                .Attr("T", co_action->output_type(0))
                                .Attr("num", out_size)
                                .Attr("axis", 1)
                                .Finalize(&unpack_node);
  if (!status.ok()) {
    LOG(ERROR) << "Adding unpack nodedef build failed " << status;
    return nullptr;
  }
  unpack_node.set_device(co_action->def().device());
  VLOG(1) << unpack_node.DebugString();
  Node* unpack = graph->AddNode(unpack_node, &status);
  if (!status.ok()) {
    LOG(ERROR) << "Adding unpack node failed " << status;
    return nullptr;
  }
  unpack->set_assigned_device_name(co_action->assigned_device_name());
  int port = 0;
  graph->AddEdge(co_action, 0, unpack, 0);
  for (auto e:out_edges) {
    graph->UpdateEdge(unpack, port, e->dst(), e->dst_input());
    port++;
  }
  return unpack;
}

Status MergeCoAction(Graph* graph, std::set<string>& skip_merge_nodes) {
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  Status status;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  std::map<std::string, CoActionPattern> collection;
  VLOG(1) << "start to merge CoAtion node, " << nodes.size();
  for (Node* node : nodes) {
    if (node->type_string() != "CoAction" &&
        node->type_string() != "CoActionIndicator") continue;
    if (skip_merge_nodes.find(node->name()) != skip_merge_nodes.end()) continue;
    Node* co_action = node;
    const Node* input_a;
    co_action->input_node(0, &input_a);
    if (input_a->def().attr().find("_output_shapes") == input_a->def().attr().end()) {
      return errors::Internal("cant find attr: _output_shapes, merge CoAction failed! ",
                              input_a->def().DebugString());
    }
    int dim_size = input_a->def().attr().at("_output_shapes").list().shape(0).dim(1).size();
    std::string key = "co_action";
    if (co_action->type_string() == "CoActionIndicator") key = "co_action_indicator";
    if (dim_size == 150) {
      key = key + "_150";
    } else if (dim_size == 50) {
      key = key + "_50";
    } else {
      return errors::Internal("CoAction op support 150 and 50 only, but get ", dim_size);
    }
    if (!GetCoActionPattern(co_action, collection[key])) {
      continue;
    }
  }
  if (VLOG_IS_ON(1)) DebugCoActionPattern(collection);
  VLOG(0) << "merge CoAction, reserve " << collection.size() << " CoAction/CoActionIndicator";
  for (auto iter:collection) {
    // 1.pack CoAction input
    Node* co_action = (iter.second.co_action)[0];
    Node* pack_a = ConstructPackOp(graph, co_action, iter.second.input_a, "/pack_input_a");
    CHECK_NULL(pack_a)
    Node* pack_b = ConstructPackOp(graph, co_action, iter.second.input_b, "/pack_input_b");
    CHECK_NULL(pack_b)
    // 2.连接Pack到其中一个CoAction op，其他的可以不用了
    graph->AddEdge(pack_a, 0, co_action, 0);
    graph->AddEdge(pack_b, 0, co_action, 1);
    // 3.Unpack为多个输出
    Node* unpack = ConstructUnpackOp(graph, co_action, iter.second.output, "/unpack_output");
    CHECK_NULL(unpack)
    // 4.删除多余的CoAction
    for (auto n:iter.second.co_action) {
      if (n != co_action) {
        graph->RemoveNode(n);
      }
    }
    skip_merge_nodes.insert(co_action->name());
  }
  return Status::OK();
}

bool OptimizeCrossFeatureScope(const GrapplerItem& item, GraphDef& input_graph,
                               GraphDef* optimized_graph, std::set<string>& skip_merge_nodes) {
  bool changed = false;
  int count = 0;
  // 1.替换低效结构为CoAtion算子
  while(1) {
    // 共48个子结构，有公共输入，因此一次遍历无法完成所有的匹配和替换(已经遍历过的节点不再继续遍历)
    if(ReplaceCoAction(input_graph, optimized_graph, count)) {
      changed = true;
      std::swap(input_graph, *optimized_graph);
    } else {
      break;
    }
  }

  VLOG(0) << "Replace " << count << " cross feature structure to CoAction/CoActionIndicator";
  if (!changed) return false;
  // 2.删除Gather算子，CoAction只支持batch size为1，推理时输入经过Gather，都是重复数据
  // convert graphdef to graph
  FunctionLibraryDefinition flib(OpRegistry::Global(), input_graph.library());
  Graph graph(flib);
  Status status = ConvertGraphDefToGraph(GraphConstructorOptions(),
                                         *optimized_graph, &graph);
  if (!status.ok()) {
    LOG(WARNING) << "ConvertGraphDefToGraph failed: " << status.ToString();
    return false;
  }
  status = RemoveGather(&graph, skip_merge_nodes);
  if (!status.ok()) {
    LOG(WARNING) << " remove gather failed: " << status.ToString();
    return false;
  }
  GraphDef temp_graph_def;
  graph.ToGraphDef(&temp_graph_def);

  // 3.合并相同输入的CoAction
  GrapplerItem new_item = item;
  new_item = new_item.WithGraph(std::move(temp_graph_def));
  GraphProperties graph_properties(new_item);
  status = graph_properties.InferStatically(/*assume_valid_feeds=*/false,
                                            /*aggressive_shape_inference=*/false,
                                            /*include_tensor_values=*/false);
  if (!status.ok()) {
    LOG(WARNING) << "infer statically return status: " << status.ToString();
    return false;
  }
  status = graph_properties.AnnotateOutputShapes(optimized_graph);
  if (!status.ok()) {
    LOG(WARNING) << "Annotate shape return status: " << status.ToString();
    return false;
  }
  Graph new_graph(flib);
  status = ConvertGraphDefToGraph(GraphConstructorOptions(),
                                  *optimized_graph, &new_graph);
  if (!status.ok()) {
    LOG(WARNING) << "ConvertGraphDefToGraph failed: " << status.ToString();
    return false;
  }
  status = MergeCoAction(&new_graph, skip_merge_nodes);
  if (!status.ok()) {
    LOG(WARNING) << " merge CoAction failed: " << status.ToString();
    return false;
  }
  new_graph.ToGraphDef(optimized_graph);

  if (!changed) return false;
  return true;
}

}  // end namespace

Status FuseCrossFeatureOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool cross_feature = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE", true, &cross_feature);
  if (!cross_feature) {
    *optimized_graph = item.graph;
    return Status::OK();
  }
  static int pass = 0;
  VLOG(0) << "FuseCrossFeatureOptimizer is on." << pass;
  if (VLOG_IS_ON(1)) {
    std::fstream f;
    f.open("before_fuse_cross_feature_" + std::to_string(pass) + ".pb",
           std::fstream::out);
    f << item.graph.SerializeAsString();
    f.close();
  }

  GraphDef input_graph_def = item.graph;
  std::set<string> skip_merge_nodes;
  if (!OptimizeCrossFeatureScope(item, input_graph_def, optimized_graph, skip_merge_nodes)) {
    *optimized_graph = item.graph;
    return Status::OK();
  }
  input_graph_def = *optimized_graph;
  int count = 0;
  bool changed = false;
  while(1) {
    if (MergeAdCreativeGather(input_graph_def, optimized_graph, count)) {
      changed = true;
      std::swap(input_graph_def, *optimized_graph);
    } else {
      break;
    }
  }
  VLOG(0) << "Merge " << count << " ad-creative Gather";
  input_graph_def = *optimized_graph;
  GraphDef temp_graph_def = *optimized_graph;
  if (changed) {
    if(!OptimizeCrossFeatureScope(item, input_graph_def, optimized_graph, skip_merge_nodes)) {
      *optimized_graph = temp_graph_def;
    }
  }

  *optimized_graph->mutable_versions() = item.graph.versions();

  if (VLOG_IS_ON(1)) {
    std::fstream f;
    f.open("after_fuse_cross_feature_" + std::to_string(pass) + ".pb",
           std::fstream::out);
    f << optimized_graph->SerializeAsString();
    f.close();
  }
  pass++;
  return Status::OK();
}

void FuseCrossFeatureOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
