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

#include "tensorflow/core/grappler/optimizers/gemm_optimizer.h"

#include <fstream>

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace grappler {

namespace {

// TODO(ylxu): add all ops that do not change tensor shapes
std::unordered_set<string> GetOpsWithUnchangedShape() {
  static std::unordered_set<string> ops_with_unchanged_shape = {
      "BiasAdd",
      "Sigmoid",
      "Tanh",
      "Relu"};
  return ops_with_unchanged_shape;
}

std::unordered_set<string> GetUnaryOps() {
  static std::unordered_set<string> ops = {
      "Softmax",
      "Sigmoid",
      "Tanh",
      "Relu"};
  return ops;
}

std::unordered_set<string> GetBinaryOps() {
  std::unordered_set<string> ops = {
      "IndicatorMatMul",
      "MatMul",
      "BatchMatMul",
      "BatchMatMulV2",
      "Add",
      "AddV2",
      "Sub",
      "Mul"};
  return ops;
}

Status UpdateAllEdge(Graph* graph, Node* new_src_node, Node* old_dst_node) {
  std::vector<Node*> dst_nodes;
  std::vector<int> dst_inputs;
  for (const Edge* e : old_dst_node->out_edges()) {
    dst_nodes.push_back(e->dst());
    dst_inputs.push_back(e->dst_input());
  }
  for (unsigned int i = 0; i < dst_nodes.size(); i++) {
    TF_RETURN_IF_ERROR(
        graph->UpdateEdge(new_src_node, 0, dst_nodes[i], dst_inputs[i]));
  }
  return Status::OK();
}

// Change MatMul_0->[Reshape]*n->MatMul_1 to MatMul_0->MatMul_1
bool RemoveReshapesBeforeMatMul(Graph* graph) {
  bool changed = false; 
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  const std::unordered_set<string> ops_with_unchanged_shape =
      GetOpsWithUnchangedShape();
  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "MatMul") continue;
    Node* current = nullptr;
    node->input_node(0, &current);
    string type = current->type_string();
    if (type != "Reshape") continue;

    Node* non_reshape_before_matmul_1 = nullptr;
    while (type == "Reshape" || (ops_with_unchanged_shape.find(type) !=
                                 ops_with_unchanged_shape.end())) {
      if (type != "Reshape") non_reshape_before_matmul_1 = current;
      Node* temp = nullptr;
      current->input_node(0, &temp);
      current = temp;
      type = current->type_string();
    }
    if (type != "MatMul") continue;

    // Check shape compatibility of two MatMuls.
    // If compatible, Reshapes before the second MatMul can be skipped;
    Node* matmuls[2];
    matmuls[0] = current;
    matmuls[1] = node;
    bool can_remove = true;
    int w_shapes[2][2];
    for (int i = 0; i < 2; i++) {
      Node* w = nullptr;
      matmuls[i]->input_node(1, &w);
      if (w->type_string() != "Const") {
        can_remove = false;
        break;
      }
      bool transpose_b = matmuls[i]->def().attr().at("transpose_b").b();
      TensorShapeProto s = w->def().attr().at("value").
                           tensor().tensor_shape();
      if (transpose_b) {
        for (int j = s.dim_size() - 1; j >= 0; j--) {
          w_shapes[i][j] = s.dim(j).size();
        }
      } else {
        for (int j = 0; j < s.dim_size(); j++) {
          w_shapes[i][j] = s.dim(j).size();
        }
      }
    }
    if (!can_remove || w_shapes[0][1] != w_shapes[1][0]) continue;

    if (non_reshape_before_matmul_1 == nullptr) {
      non_reshape_before_matmul_1 = matmuls[0];
    }
    const Edge* e;
    matmuls[1]->input_edge(0, &e);
    graph->RemoveEdge(e);
    graph->AddEdge(non_reshape_before_matmul_1, 0, matmuls[1], 0);
    changed = true;
  }
  return changed;
}

// Change ->MatMul->Reshape->BiasAdd-> to ->MatMul->BiasAdd->Reshape->
bool ReorderReshapeAndBiasAdd(Graph* graph) {
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() == "Reshape") {
      if (node->out_edges().size() != 1) continue;
      Node* reshape_in = nullptr;
      Node* reshape_out = nullptr;
      node->input_node(0, &reshape_in);
      if (reshape_in->type_string() != "MatMul") continue; 
      for (Node* n : node->out_nodes()) {
        if (n->type_string() != "BiasAdd") continue;
        reshape_out = n;
        break;
      }
      if (reshape_out == nullptr) continue;

      Node* reshape = node;
      Node* matmul = reshape_in;
      Node* bias = reshape_out;
      const Edge* matmul_reshape;
      const Edge* reshape_bias;
      reshape->input_edge(0, &matmul_reshape); 
      bias->input_edge(0, &reshape_bias); 
      graph->RemoveEdge(matmul_reshape);
      graph->RemoveEdge(reshape_bias);
      std::vector<Node*> bias_dst_nodes;
      std::vector<int> bias_dst_inputs;
      for (const Edge* e : bias->out_edges()) {
        bias_dst_nodes.push_back(e->dst());
        bias_dst_inputs.push_back(e->dst_input());
      }
      for (unsigned int i = 0; i < bias_dst_nodes.size(); i++) {
        graph->UpdateEdge(reshape, 0, bias_dst_nodes[i], bias_dst_inputs[i]);
      }
      graph->AddEdge(matmul, 0, bias, 0);
      graph->AddEdge(bias, 0, reshape, 0);
      changed = true;
    }
  }
  return changed;
}

// Change (A*B+C)*D+E to A*(B*C)+(C*D)+E when B, C, D and E are consts,
// such that B*C, and (C*D)+E can be folded to consts.
bool ConstantFoldingForContinuousMatMulsOnePass(Graph* graph);
bool ConstantFoldingForContinuousMatMuls(Graph* graph) {
  VLOG(2) << "ConstantFoldingForContinuousMatMuls";
  bool changed = false;
  while (1) {
    if (ConstantFoldingForContinuousMatMulsOnePass(graph)) {
      changed = true;
    } else {
      break;
    }
  }
  return changed;
}

bool ConstantFoldingForContinuousMatMulsOnePass(Graph* graph) {
  static int count = 0;
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;

    Node* matmuls[2];
    Node* biasadds[2];
    Node* weights[2];
    Node* biases[2];
    for (int i = 0; i < 2; i++) {
      matmuls[i] = nullptr;
      biasadds[i] = nullptr;
      weights[i] = nullptr;
      biases[i] = nullptr;
    }

    // find pattern
    if (node->type_string() != "MatMul") continue;
    matmuls[1] = node;
    for (Node* n : matmuls[1]->out_nodes()) {
      if (n->type_string() == "BiasAdd") {
        biasadds[1] = n;
        break;
      }
    }
    Node* temp = nullptr;
    matmuls[1]->input_node(0, &temp);
    if (temp->type_string() == "BiasAdd") {
      biasadds[0] = temp;
      biasadds[0]->input_node(0, &matmuls[0]);
      if (matmuls[0]->type_string() != "MatMul") continue;
    } else if (temp->type_string() == "MatMul") {
      matmuls[0] = temp;
    } else {
      continue;
    }
    for (int i = 0; i < 2; i++) {
      if (biasadds[i]) {
        biasadds[i]->input_node(1, &biases[i]);
      }
      matmuls[i]->input_node(1, &weights[i]);
    }

    VLOG(2) << "ConstantFoldingForContinuousMatMuls: found pattern";
    Node* temps[5];
    for (int i = 0; i < 5; i++) {
      temps[i] = nullptr;
    }

    string prefix = "GemmOptimizer/ConstantFoldingForContinuousMatMuls/" +
                    matmuls[0]->name() + "/" + std::to_string(count++);

    // Do constant folding as follows:
    // T0 = (B*D)
    // T1 = A*T0
    // T2 = C*D
    // T3 = T2+E
    // T4 = T1+T3

    // T0 = (B*D)
    string temp0_name = prefix + "/temp0";
    std::vector<NodeDefBuilder::NodeOut> temp0_inputs;
    DataType dtype = weights[0]->output_type(0);
    temp0_inputs.emplace_back(weights[0]->name(), 0, dtype);
    temp0_inputs.emplace_back(weights[1]->name(), 0, dtype);
    NodeDefBuilder temp0_builder(temp0_name, "MatMul");
    temp0_builder.Input(temp0_inputs[0]);
    temp0_builder.Input(temp0_inputs[1]);
    NodeDef temp0_node;
    Status status =
        temp0_builder
            .Attr("transpose_a",
                  matmuls[0]->def().attr().at("transpose_b").b())
            .Attr("transpose_b",
                  matmuls[1]->def().attr().at("transpose_b").b())
            .Attr("T", dtype)
            .Finalize(&temp0_node);
    if (!status.ok()) {
      LOG(ERROR) << "MatMul node construction failed with" << status;
      return false;
    }
    temp0_node.set_device(weights[0]->def().device());
    temps[0] = graph->AddNode(temp0_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    temps[0]->set_assigned_device_name(weights[0]->assigned_device_name());
    graph->AddEdge(weights[0], 0, temps[0], 0);
    graph->AddEdge(weights[1], 0, temps[0], 1);

    // T1 = A*T0
    string temp1_name = prefix + "/temp1";
    std::vector<NodeDefBuilder::NodeOut> temp1_inputs;
    const Edge* e = nullptr;
    matmuls[0]->input_edge(0, &e);
    temp1_inputs.emplace_back(e->src()->name(), e->src_output(), dtype);
    temp1_inputs.emplace_back(temp0_name, 0, dtype);
    NodeDefBuilder temp1_builder(temp1_name, "MatMul");
    temp1_builder.Input(temp1_inputs[0]);
    temp1_builder.Input(temp1_inputs[1]);
    NodeDef temp1_node;
    status =
        temp1_builder
            .Attr("transpose_a",
                  matmuls[0]->def().attr().at("transpose_a").b())
            .Attr("transpose_b", false)
            .Attr("T", dtype)
            .Finalize(&temp1_node);
    if (!status.ok()) {
      LOG(ERROR) << "MatMul node construction failed with" << status;
      return false;
    }
    temp1_node.set_device(matmuls[0]->def().device());
    temps[1] = graph->AddNode(temp1_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    temps[1]->set_assigned_device_name(matmuls[0]->assigned_device_name());
    graph->AddEdge(e->src(), e->src_output(), temps[1], 0);
    graph->AddEdge(temps[0], 0, temps[1], 1);

    // T2 = C*D
    if (biasadds[0]) {
      string temp2_name = prefix + "/temp2";

      string dim_name = temp2_name + "/ExpandDims" + "/axis";
      NodeDefBuilder dim_builder(dim_name, "Const");
      NodeDef dim_node;
      Tensor t_dim((int)0);
      status =
          dim_builder
              .Attr("dtype", t_dim.dtype())
              .Attr("value", t_dim)
              .Finalize(&dim_node);
      if (!status.ok()) {
        LOG(ERROR) << "Const node construction failed with" << status;
        return false;
      }
      dim_node.set_device(biases[0]->def().device());
      Node* dim = graph->AddNode(dim_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      dim->set_assigned_device_name(biases[0]->assigned_device_name());

      string expand_name = temp2_name + "/ExpandDims";
      std::vector<NodeDefBuilder::NodeOut> expand_inputs;
      expand_inputs.emplace_back(biases[0]->name(), 0, dtype);
      expand_inputs.emplace_back(dim_name, 0, t_dim.dtype());
      NodeDefBuilder expand_builder(expand_name, "ExpandDims");
      expand_builder.Input(expand_inputs[0]);
      expand_builder.Input(expand_inputs[1]);
      NodeDef expand_node;
      status =
          expand_builder
              .Attr("T", dtype)
              .Attr("Tdim", t_dim.dtype())
              .Finalize(&expand_node);
      if (!status.ok()) {
        LOG(ERROR) << "ExpandDims node construction failed with" << status;
        return false;
      }
      expand_node.set_device(biases[0]->def().device());
      Node* expand = graph->AddNode(expand_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      expand->set_assigned_device_name(biases[0]->assigned_device_name());
      graph->AddEdge(biases[0], 0, expand, 0);
      graph->AddEdge(dim, 0, expand, 1);

      string matmul_name = temp2_name + "/MatMul";
      std::vector<NodeDefBuilder::NodeOut> matmul_inputs;
      matmul_inputs.emplace_back(expand_name, 0, dtype);
      matmul_inputs.emplace_back(weights[1]->name(), 0, dtype);
      NodeDefBuilder matmul_builder(matmul_name, "MatMul");
      matmul_builder.Input(matmul_inputs[0]);
      matmul_builder.Input(matmul_inputs[1]);
      NodeDef matmul_node;
      status =
          matmul_builder
              .Attr("transpose_a", false)
              .Attr("transpose_b",
                    matmuls[1]->def().attr().at("transpose_b").b())
              .Attr("T", dtype)
              .Finalize(&matmul_node);
      if (!status.ok()) {
        LOG(ERROR) << "MatMul node construction failed with" << status;
        return false;
      }
      matmul_node.set_device(weights[1]->def().device());
      Node* matmul = graph->AddNode(matmul_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      matmul->set_assigned_device_name(
          weights[1]->assigned_device_name());
      graph->AddEdge(expand, 0, matmul, 0);
      graph->AddEdge(weights[1], 0, matmul, 1);

      string squeeze_name = temp2_name + "/Squeeze";
      NodeDefBuilder::NodeOut squeeze_input(matmul->name(), 0, dtype);
      NodeDefBuilder squeeze_builder(squeeze_name, "Squeeze");
      squeeze_builder.Input(squeeze_input);
      NodeDef squeeze_node;
      status =
          squeeze_builder
              .Attr("T", dtype)
              .Finalize(&squeeze_node);
      if (!status.ok()) {
        LOG(ERROR) << "Squeeze node construction failed with" << status;
        return false;
      }
      squeeze_node.set_device(weights[1]->def().device());
      temps[2] = graph->AddNode(squeeze_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      temps[2]->set_assigned_device_name(
          weights[1]->assigned_device_name());
      graph->AddEdge(matmul, 0, temps[2], 0);
    }
    // T3 = T2+E
    if (biasadds[0] && biasadds[1]) {
      string temp3_name = prefix + "/temp3";
      std::vector<NodeDefBuilder::NodeOut> temp3_inputs;
      temp3_inputs.emplace_back(temps[2]->name(), 0, dtype);
      temp3_inputs.emplace_back(biases[1]->name(), 0, dtype);
      NodeDefBuilder temp3_builder(temp3_name, "Add");
      temp3_builder.Input(temp3_inputs[0]);
      temp3_builder.Input(temp3_inputs[1]);
      NodeDef temp3_node;
      status =
          temp3_builder
              .Attr("T", dtype)
              .Finalize(&temp3_node);
      if (!status.ok()) {
        LOG(ERROR) << "BiasAdd node construction failed with" << status;
        return false;
      }
      temp3_node.set_device(biases[1]->def().device());
      temps[3] = graph->AddNode(temp3_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      temps[3]->set_assigned_device_name(biases[1]->assigned_device_name());
      graph->AddEdge(temps[2], 0, temps[3], 0);
      graph->AddEdge(biases[1], 0, temps[3], 1);
    }
    if (biasadds[0] && !biasadds[1]) {
      temps[3] = temps[2];
    }
    if (!biasadds[0] && biasadds[1]) {
      temps[3] = biases[1]; 
    }
    // T4 = T1+T3
    if (temps[3]) {
      string temp4_name = prefix + "/temp4";
      std::vector<NodeDefBuilder::NodeOut> temp4_inputs;
      temp4_inputs.emplace_back(temps[1]->name(), 0, dtype);
      temp4_inputs.emplace_back(temps[3]->name(), 0, dtype);
      NodeDefBuilder temp4_builder(temp4_name, "BiasAdd");
      temp4_builder.Input(temp4_inputs[0]);
      temp4_builder.Input(temp4_inputs[1]);
      NodeDef temp4_node;
      status =
          temp4_builder
              .Attr("T", dtype)
              .Finalize(&temp4_node);
      if (!status.ok()) {
        LOG(ERROR) << "Add node construction failed with" << status;
        return false;
      }
      temp4_node.set_device(temps[3]->def().device());
      temps[4] = graph->AddNode(temp4_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      temps[4]->set_assigned_device_name(temps[3]->assigned_device_name());
      graph->AddEdge(temps[1], 0, temps[4], 0);
      graph->AddEdge(temps[3], 0, temps[4], 1);
    } else {
      temps[4] = temps[1];
    }

    Node* last_node = nullptr;
    if (biasadds[1]) {
      last_node = biasadds[1];
    } else {
      last_node = matmuls[1];
    } 
    std::vector<Node*> dst_nodes;
    std::vector<int> dst_inputs;
    for (const Edge* e : last_node->out_edges()) {
      dst_nodes.push_back(e->dst());
      dst_inputs.push_back(e->dst_input());
    }
    for (unsigned int i = 0; i < dst_nodes.size(); i++) {
      graph->UpdateEdge(temps[4], 0, dst_nodes[i], dst_inputs[i]);
    }
    graph->RemoveNode(last_node);
    changed = true;
  }
  return changed;
}

// op--->MatMul to op->BatchMatMul->Unpack--->
//    |->MatMul                            |->
//    |->MatMul                            |->
//       ...
bool FuseMatMuls(Graph* graph) {
  static int count = 0;
  VLOG(2) << "FuseMatMuls";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    // Find pattern op--->MatMul->
    //                 |->MatMul->
    //                 |->MatMul->
    //                    ...
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "MatMul") continue;
    Node* in = nullptr;
    node->input_node(0, &in);
    std::vector<Node*> matmuls;
    std::vector<int> src_outputs;
    for (const Edge* e : in->out_edges()) {
      Node* n = e->dst();
      if (n->type_string() == "MatMul") {
        matmuls.push_back(n);
        src_outputs.push_back(e->src_output());
      }
    }
    if (matmuls.size() < 2) continue;
    // Check the inputs of matmuls has the same src_output
    bool same_input = true;
    for (unsigned int i = 1; i < src_outputs.size(); i++) {
      if (src_outputs[i] != src_outputs[0]) same_input = false;
    }
    if (!same_input) continue;

    VLOG(2) << "FuseMatMuls: found pattern";
    std::sort(matmuls.begin(), matmuls.end(),
              [node](Node* a, Node* b){
      return a->name().compare(b->name()) < 0;
    });
    std::vector<Node*> weights;
    VLOG(2) << "the following ops are fused: ";
    string weight_shape;
    bool can_fuse = true;
    for (Node* m : matmuls) {
      VLOG(2) << m->name();
      Node* w = nullptr;
      m->input_node(1, &w);

      // Check all weights
      // (1) are Const ops, and
      // (2) have the same shape.
      if (w->type_string() != "Const") {
        can_fuse = false;
        break;
      }
      TensorShapeProto s = w->def().attr().at("value").
                           tensor().tensor_shape();
      string temp;
      for (int i = 0; i < s.dim_size(); i++) {
        temp += std::to_string(s.dim(i).size()) + ",";
      }
      if (temp.empty()) {
        can_fuse = false;
        break;
      }
      if (weight_shape.empty()) {
        weight_shape = temp;
      } else {
        if (temp != weight_shape) {
          can_fuse = false;
          break;
        }
      }

      weights.push_back(w);
    }
    if (!can_fuse) continue;
    // Add a Pack node to group weights
    string prefix = "GemmOptimizer/FuseMatMuls/" +
                    std::to_string(count++);
    string pack_name = prefix + "/Pack";
    std::vector<NodeDefBuilder::NodeOut> pack_inputs;
    DataType dtype = weights[0]->output_type(0);
    for (Node* w : weights) {
      pack_inputs.emplace_back(w->name(), 0, dtype);
    }
    NodeDefBuilder pack_builder(pack_name, "Pack");
    pack_builder.Input(pack_inputs);
    NodeDef pack_node;
    Status status =
        pack_builder
            .Attr("N", (int)weights.size())
            .Attr("T", dtype)
            .Attr("axis", 0)
            .Finalize(&pack_node);
    if (!status.ok()) {
      LOG(ERROR) << "Pack node construction failed with" << status;
      return false;
    }
    pack_node.set_device(weights[0]->def().device());
    Node* pack = graph->AddNode(pack_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    pack->set_assigned_device_name(weights[0]->assigned_device_name());
    for (unsigned int i = 0; i < weights.size(); i++) {
      graph->AddEdge(weights[i], 0, pack, i);
    }

    // Add a new node BatchMatMul
    string matmul_name = prefix + "/BatchMatMulV2";
    std::vector<NodeDefBuilder::NodeOut> matmul_inputs;
    const Edge* e;
    matmuls[0]->input_edge(0, &e);
    int src_output = e->src_output();
    matmul_inputs.emplace_back(in->name(), src_output, dtype);
    matmul_inputs.emplace_back(pack_name, 0, dtype);
    NodeDefBuilder matmul_builder(matmul_name, "BatchMatMulV2");
    matmul_builder.Input(matmul_inputs[0]);
    matmul_builder.Input(matmul_inputs[1]);
    NodeDef matmul_node;
    status =
        matmul_builder
            .Attr("adj_x", matmuls[0]->def().attr().at("transpose_a").b())
            .Attr("adj_y", matmuls[0]->def().attr().at("transpose_b").b())
            .Attr("T", dtype)
            .Finalize(&matmul_node);
    if (!status.ok()) {
      LOG(ERROR) << "BatchMatMul node construction failed with" << status;
      return false;
    }
    matmul_node.set_device(matmuls[0]->def().device());
    Node* matmul = graph->AddNode(matmul_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    matmul->set_assigned_device_name(matmuls[0]->assigned_device_name());
    graph->AddEdge(in, src_output, matmul, 0);
    graph->AddEdge(pack, 0, matmul, 1);

    // Add an Unpack node to split result
    string unpack_name = prefix + "/Unpack";
    NodeDefBuilder::NodeOut unpack_input(matmul_name, 0, dtype);
    NodeDefBuilder unpack_builder(unpack_name, "Unpack");
    unpack_builder.Input(unpack_input);
    NodeDef unpack_node;
    status =
        unpack_builder
            .Attr("num", (int)weights.size())
            .Attr("T", dtype)
            .Attr("axis", 0)
            .Finalize(&unpack_node);
    if (!status.ok()) {
      LOG(ERROR) << "Unpack node construction failed with" << status;
      return false;
    }
    unpack_node.set_device(matmuls[0]->def().device());
    Node* unpack = graph->AddNode(unpack_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    unpack->set_assigned_device_name(matmuls[0]->assigned_device_name());
    graph->AddEdge(matmul, 0, unpack, 0);
 
    // Add edges to forward split results to nodes after original matmuls,
    // and remove original matmuls
    int index = 0;
    for (Node* m : matmuls) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : m->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      for (unsigned int i = 0; i < dst_nodes.size(); i++) {
        graph->UpdateEdge(unpack, index, dst_nodes[i], dst_inputs[i]);
      }
      graph->RemoveNode(m);
      index++;
    }
    changed = true;
  }
  return changed;
}

// BatchMatMul->Unpack--->BiasAdd  to BatchMatMul->BiasAdd->Unpack
//                     |->BiasAdd
//                     |->BiasAdd
//                        ...
// TODO(ylxu): to support fusing more types of ops after unpack
bool FuseBiasAddsAfterBatchMatMulUnpack(Graph* graph) {
  static int count = 0;
  VLOG(2) << "FuseBiasAdds";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    Node* matmul = nullptr;
    node->input_node(0, &matmul);
    if ((matmul->type_string() != "BatchMatMulV2") &&
        (matmul->type_string() != "BatchMatMul")) continue;
    std::vector<Node*> biasadds;
    bool can_fuse = true;
    std::set<int> src_outputs;
    for (const Edge* e : node->out_edges()) {
      Node* out = e->dst();
      int src_output = e->src_output();
      if (out->type_string() != "BiasAdd" ||
          // check each output of Unpack is used only once
          src_outputs.find(src_output) != src_outputs.end()) {
        can_fuse = false;
        break;
      }
      biasadds.push_back(out);
      src_outputs.insert(src_output);
    }
    if (!can_fuse || biasadds.size() < 2) continue;

    VLOG(2) << "FuseBiasAdds: found pattern";
    std::sort(biasadds.begin(), biasadds.end(),
              [node](Node* a, Node* b){
      const Edge* e = nullptr;
      a->input_edge(0, &e);
      int a_src_output = e->src_output();
      b->input_edge(0, &e);
      int b_src_output = e->src_output();
      return a_src_output < b_src_output;
    });
    std::vector<Node*> biases;
    string bias_shape;
    for (Node* n : biasadds) {
      Node* bias = nullptr;
      n->input_node(1, &bias);

      // Check all biases
      // (1) are Const ops, and
      // (2) have the same shape.
      if (bias->type_string() != "Const") {
        can_fuse = false;
        break;
      }
      TensorShapeProto s = bias->def().attr().at("value").
                           tensor().tensor_shape();
      string temp;
      for (int i = 0; i < s.dim_size(); i++) {
        temp += std::to_string(s.dim(i).size()) + ",";
      }
      if (temp.empty()) {
        can_fuse = false;
        break;
      }
      if (bias_shape.empty()) {
        bias_shape = temp;
      } else {
        if (temp != bias_shape) {
          can_fuse = false; 
          break;
        }
      }

      biases.push_back(bias);
    }
    if (!can_fuse) continue;

    // Add a Pack node to group biases
    string prefix = "GemmOptimizer/FuseBiasAddsAfterBatchMatMulUnpack/" +
                    std::to_string(count++);
    string pack_name = prefix + "/Pack";
    std::vector<NodeDefBuilder::NodeOut> pack_inputs;
    DataType dtype = biases[0]->output_type(0);
    for (Node* b : biases) {
      pack_inputs.emplace_back(b->name(), 0, dtype);
    }
    NodeDefBuilder pack_builder(pack_name, "Pack");
    pack_builder.Input(pack_inputs);
    NodeDef pack_node;
    Status status =
        pack_builder
            .Attr("N", (int)biases.size())
            .Attr("T", dtype)
            .Attr("axis", 0)
            .Finalize(&pack_node);
    if (!status.ok()) {
      LOG(ERROR) << "Pack node construction failed with" << status;
      return false;
    }
    pack_node.set_device(biases[0]->def().device());
    Node* pack = graph->AddNode(pack_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    pack->set_assigned_device_name(biases[0]->assigned_device_name());
    for (unsigned int i = 0; i < biases.size(); i++) {
      graph->AddEdge(biases[i], 0, pack, i);
    }

    // Add an ExpandDims after Pack
    string dim_name = prefix + "/ExpandDims/" + "/axis";
    NodeDefBuilder dim_builder(dim_name, "Const");
    NodeDef dim_node;
    Tensor t_dim((int)1);
    status =
        dim_builder
            .Attr("dtype", t_dim.dtype())
            .Attr("value", t_dim)
            .Finalize(&dim_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    dim_node.set_device(biases[0]->def().device());
    Node* dim = graph->AddNode(dim_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    dim->set_assigned_device_name(biases[0]->assigned_device_name());

    string expand_name = prefix + "/ExpandDims" ;
    std::vector<NodeDefBuilder::NodeOut> expand_inputs;
    expand_inputs.emplace_back(pack_name, 0, dtype);
    expand_inputs.emplace_back(dim_name, 0, t_dim.dtype());
    NodeDefBuilder expand_builder(expand_name, "ExpandDims");
    expand_builder.Input(expand_inputs[0]);
    expand_builder.Input(expand_inputs[1]);
    NodeDef expand_node;
    status =
        expand_builder
            .Attr("T", dtype)
            .Attr("Tdim", t_dim.dtype())
            .Finalize(&expand_node);
    if (!status.ok()) {
      LOG(ERROR) << "ExpandDims node construction failed with" << status;
      return false;
    }
    expand_node.set_device(biases[0]->def().device());
    Node* expand = graph->AddNode(expand_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    expand->set_assigned_device_name(biases[0]->assigned_device_name());
    graph->AddEdge(pack, 0, expand, 0);
    graph->AddEdge(dim, 0, expand, 1);

    // Add a new node Add
    string biasadd_name = prefix + "/Add";
    std::vector<NodeDefBuilder::NodeOut> biasadd_inputs;
    biasadd_inputs.emplace_back(matmul->name(), 0, dtype);
    biasadd_inputs.emplace_back(expand_name, 0, dtype);
    NodeDefBuilder biasadd_builder(biasadd_name, "Add");
    biasadd_builder.Input(biasadd_inputs[0]);
    biasadd_builder.Input(biasadd_inputs[1]);
    NodeDef biasadd_node;
    status =
        biasadd_builder
            .Attr("T", dtype)
            .Finalize(&biasadd_node);
    if (!status.ok()) {
      LOG(ERROR) << "BiasAdd node construction failed with" << status;
      return false;
    }
    biasadd_node.set_device(biasadds[0]->def().device());
    Node* biasadd = graph->AddNode(biasadd_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    biasadd->set_assigned_device_name(biasadds[0]->assigned_device_name());
    graph->AddEdge(matmul, 0, biasadd, 0);
    graph->AddEdge(expand, 0, biasadd, 1);

    // Add an Unpack node to split result
    string unpack_name = prefix + "/Unpack";
    NodeDefBuilder::NodeOut unpack_input(biasadd_name, 0, dtype);
    NodeDefBuilder unpack_builder(unpack_name, "Unpack");
    unpack_builder.Input(unpack_input);
    NodeDef unpack_node;
    status =
        unpack_builder
            .Attr("num", (int)biases.size())
            .Attr("T", dtype)
            .Attr("axis", 0)
            .Finalize(&unpack_node);
    if (!status.ok()) {
      LOG(ERROR) << "Unpack node construction failed with" << status;
      return false;
    }
    unpack_node.set_device(biasadds[0]->def().device());
    Node* unpack = graph->AddNode(unpack_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    unpack->set_assigned_device_name(biasadds[0]->assigned_device_name());
    graph->AddEdge(biasadd, 0, unpack, 0);
 
    // Add edges to forward split results to nodes after original matmuls,
    // and remove original matmuls
    int index = 0;
    for (Node* b : biasadds) {
      std::vector<Node*> out_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : b->out_edges()) {
        out_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      for (unsigned int i = 0; i < out_nodes.size(); i++) {
        graph->UpdateEdge(unpack, index, out_nodes[i], dst_inputs[i]);
      }
      graph->RemoveNode(b);
      index++;
    }
    changed = true;
  }
  return changed;
}

// [-, shape_param]->Reshape->Shape->Op to [shape_param]->Op
bool RemoveShapeAfterReshape(Graph* graph) {
  VLOG(2) << "RemoveShapeAfterReshape";
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Shape") continue;
    Node* shape = node;
    Node* reshape = nullptr;
    shape->input_node(0, &reshape);
    if (reshape->type_string() != "Reshape") continue;
    VLOG(2) << "RemoveShapeAfterReshape: found pattern";
    Node* reshape_in_1 = nullptr;
    reshape->input_node(1, &reshape_in_1);

    std::vector<Node*> shape_dst_nodes;
    std::vector<int> shape_dst_inputs;
    for (const Edge* e : shape->out_edges()) {
      shape_dst_nodes.push_back(e->dst());
      shape_dst_inputs.push_back(e->dst_input());
    }

    const Edge* e;
    reshape->input_edge(1, &e);
    int src_output = e->src_output();
    graph->RemoveNode(shape);
    for (unsigned int i = 0; i < shape_dst_nodes.size(); i++) {
      graph->AddEdge(reshape_in_1, src_output,
                     shape_dst_nodes[i], shape_dst_inputs[i]);
    }
    changed = true;
  }
  return changed;
}

// Change ->Unpack->Reshape*n-> to ->Reshape->Unpack->
bool FuseReshapesAfterUnpack(Graph* graph) {
  static int count = 0;
  VLOG(2) << "FuseReshapesAfterUnpack";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    Node* unpack = node;
    std::vector<Node*> reshapes;
    std::vector<Node*> reshape_in_1;
    bool can_reorder = true;
    for (Node* out : unpack->out_nodes()) {
      if (out->type_string() != "Reshape") {
        can_reorder = false;
        break;
      }
      reshapes.push_back(out);
      Node* n = nullptr;
      out->input_node(1, &n);
      reshape_in_1.push_back(n);
    }
    if (!can_reorder || reshapes.size() < 2) continue;
    for (unsigned int i = 1; i < reshapes.size(); i++) {
      if (reshape_in_1[i] != reshape_in_1[0]){ 
        can_reorder = false;
        break;
	    }
    }
    if (!can_reorder) continue;
    VLOG(2) << "FuseReshapesAfterUnpack: found pattern";
    
    // Add a new Shape to get the shape of Unpack's input
    const Edge* to_unpack;
    unpack->input_edge(0, &to_unpack);
    int src_output = to_unpack->src_output();
    Node* unpack_in = to_unpack->src();
    string prefix = "GemmOptimizer/FuseReshapesAfterUnpack/" +
                    std::to_string(count++);
    string shape_name = prefix + "/Shape";
    NodeDefBuilder::NodeOut shape_input(unpack_in->name(), src_output,
                                        unpack->input_type(0));
    NodeDefBuilder shape_builder(shape_name, "Shape");
    shape_builder.Input(shape_input);
    NodeDef shape_node;
    DataType shape_dtype = reshape_in_1[0]->output_type(0);
    Status status =
        shape_builder
            .Attr("T", unpack->input_type(0))
            .Attr("out_type", shape_dtype)
            .Finalize(&shape_node);
    if (!status.ok()) {
      LOG(ERROR) << "Shape node construction failed with" << status;
      return false;
    }
    shape_node.set_device(reshape_in_1[0]->def().device());
    Node* shape = graph->AddNode(shape_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    shape->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());
    graph->AddEdge(unpack_in, src_output, shape, 0);
 
    // Add a Slice to obtain the first element of the new shape 
    string slice_name = prefix + "/Slice";
    string zero_name = prefix + "Slice/zero";
    string one_name = prefix + "Slice/one";

    NodeDefBuilder zero_builder(zero_name, "Const");
    NodeDef zero_node;
    Tensor t_zero(DT_INT32, TensorShape({1}));
    auto zero_data = t_zero.tensor<int, 1>();
    zero_data(0) = 0;
    status =
        zero_builder
            .Attr("dtype", t_zero.dtype())
            .Attr("value", t_zero)
            .Finalize(&zero_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    zero_node.set_device(reshape_in_1[0]->def().device());
    Node* zero = graph->AddNode(zero_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    zero->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());

    NodeDefBuilder one_builder(one_name, "Const");
    NodeDef one_node;
    Tensor t_one(DT_INT32, TensorShape({1}));
    auto one_data = t_one.tensor<int, 1>();
    one_data(0) = 1;
    status =
        one_builder
            .Attr("dtype", t_one.dtype())
            .Attr("value", t_one)
            .Finalize(&one_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    one_node.set_device(reshape_in_1[0]->def().device());
    Node* one = graph->AddNode(one_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    one->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());

    std::vector<NodeDefBuilder::NodeOut> slice_inputs;
    slice_inputs.emplace_back(shape_name, 0, shape_dtype);
    slice_inputs.emplace_back(zero_name, 0, t_zero.dtype());
    slice_inputs.emplace_back(one_name, 0, t_one.dtype());
    NodeDefBuilder slice_builder(slice_name, "Slice");
    slice_builder.Input(slice_inputs[0]);
    slice_builder.Input(slice_inputs[1]);
    slice_builder.Input(slice_inputs[2]);
    NodeDef slice_node;
    status =
        slice_builder
            .Attr("T", shape_dtype)
            .Finalize(&slice_node);
    if (!status.ok()) {
      LOG(ERROR) << "Slice node construction failed with" << status;
      return false;
    }
    slice_node.set_device(reshape_in_1[0]->def().device());
    Node* slice = graph->AddNode(slice_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    slice->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());
    graph->AddEdge(shape, 0, slice, 0);
    graph->AddEdge(zero, 0, slice, 1);
    graph->AddEdge(one, 0, slice, 2);

    // Add a Concat Op to generate new shape
    string concat_name = prefix + "/Concat";
    string zero_scalar_name = concat_name + "/concat_dim";
    NodeDefBuilder zero_scalar_builder(zero_scalar_name, "Const");
    NodeDef zero_scalar_node;
    Tensor t_zero_scalar(int(0));
    status =
        zero_scalar_builder
            .Attr("dtype", t_zero_scalar.dtype())
            .Attr("value", t_zero_scalar)
            .Finalize(&zero_scalar_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    zero_scalar_node.set_device(reshape_in_1[0]->def().device());
    Node* zero_scalar = graph->AddNode(zero_scalar_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    zero_scalar->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());

    NodeDefBuilder concat_builder(concat_name, "Concat");
    NodeDefBuilder::NodeOut concat_dim(zero_scalar->name(), 0, shape_dtype);
    std::vector<NodeDefBuilder::NodeOut> concat_inputs;
    concat_inputs.emplace_back(slice_name, 0, shape_dtype);
    const Edge* e1;
    reshapes[0]->input_edge(1, &e1);
    concat_inputs.emplace_back(reshape_in_1[0]->name(), e1->src_output(), shape_dtype);
    concat_builder.Input(concat_dim);
    concat_builder.Input(concat_inputs);
    NodeDef concat_node;
    status =
        concat_builder
            .Attr("N", 2)
            .Attr("T", shape_dtype)
            .Finalize(&concat_node);
    if (!status.ok()) {
      LOG(ERROR) << "Concat node construction failed with" << status;
      return false;
    }
    concat_node.set_device(reshape_in_1[0]->def().device());
    Node* concat = graph->AddNode(concat_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    concat->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());
    graph->AddEdge(zero_scalar, 0, concat, 0);
    graph->AddEdge(slice, 0, concat, 1);
    graph->AddEdge(reshape_in_1[0], e1->src_output(), concat, 2);

    // Add a new Reshape Op
    string reshape_name = prefix + "/Reshape";
    NodeDefBuilder reshape_builder(reshape_name, "Reshape");
    std::vector<NodeDefBuilder::NodeOut> reshape_inputs;
    reshape_inputs.emplace_back(unpack_in->name(), src_output,
                                reshapes[0]->input_type(0));
    reshape_inputs.emplace_back(concat->name(), 0, shape_dtype);
    reshape_builder.Input(reshape_inputs[0]);
    reshape_builder.Input(reshape_inputs[1]);
    NodeDef reshape_node;
    status =
        reshape_builder
            .Attr("T", reshapes[0]->input_type(0))
            .Attr("Tshape", shape_dtype)
            .Finalize(&reshape_node);
    if (!status.ok()) {
      LOG(ERROR) << "Reshape node construction failed with" << status;
      return false;
    }
    reshape_node.set_device(reshapes[0]->def().device());
    Node* reshape = graph->AddNode(reshape_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    reshape->set_assigned_device_name(
        reshapes[0]->assigned_device_name());
    graph->AddEdge(unpack_in, src_output, reshape, 0);
    graph->AddEdge(concat, 0, reshape, 1);
 
    graph->UpdateEdge(reshape, 0, unpack, 0);
    for (Node* r : reshapes) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : r->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      const Edge* e = nullptr;
      r->input_edge(0, &e);
      int src_output = e->src_output();
      for (unsigned int i = 0; i < dst_nodes.size(); i++) {
        graph->UpdateEdge(unpack, src_output, dst_nodes[i], dst_inputs[i]);
      }
      graph->RemoveNode(r);
    }
    changed= true;
  }
  return changed;
}

// Change op->Unpack->Shape to op->Shape->Slice,
// such that memory-intensive Unpack can be avoided.
bool RemoveUnpackBeforeShape(Graph* graph) {
  static int count = 0;
  VLOG(2) << "RemoveUnpackBeforeShape";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Shape") continue;
    Node* unpack = nullptr;
    node->input_node(0, &unpack);
    if (unpack->type_string() != "Unpack") continue;
    VLOG(2) << "RemoveUnpackBeforeShape: found pattern";

    // Add a new Shape to get the shape of Unpack's input
    const Edge* to_unpack = nullptr;
    unpack->input_edge(0, &to_unpack);
    int src_output = to_unpack->src_output();
    Node* unpack_in = to_unpack->src();
    string prefix = "GemmOptimizer/RemoveUnpackBeforeShape/" +
                    std::to_string(count++);
    string shape_name = prefix + "/Shape";
    NodeDefBuilder::NodeOut shape_input(unpack_in->name(), src_output,
                                        unpack->input_type(0));
    NodeDefBuilder shape_builder(shape_name, "Shape");
    shape_builder.Input(shape_input);
    NodeDef shape_node;
    DataType shape_dtype = node->output_type(0);
    Status status =
        shape_builder
            .Attr("T", unpack->input_type(0))
            .Attr("out_type", shape_dtype)
            .Finalize(&shape_node);
    if (!status.ok()) {
      LOG(ERROR) << "Shape node construction failed with" << status;
      return false;
    }
    shape_node.set_device(node->def().device());
    Node* shape = graph->AddNode(shape_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    shape->set_assigned_device_name(node->assigned_device_name());
    graph->AddEdge(unpack_in, src_output, shape, 0);
 
    string slice_name = prefix + "/Slice";
    string one_name = prefix + "Slice/one";
    string minus_one_name = prefix + "Slice/minus_one";

    NodeDefBuilder one_builder(one_name, "Const");
    NodeDef one_node;
    Tensor t_one(DT_INT32, TensorShape({1}));
    auto one_data = t_one.tensor<int, 1>();
    one_data(0) = 1;
    status =
        one_builder
            .Attr("dtype", t_one.dtype())
            .Attr("value", t_one)
            .Finalize(&one_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    one_node.set_device(node->def().device());
    Node* one = graph->AddNode(one_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    one->set_assigned_device_name(node->assigned_device_name());

    NodeDefBuilder minus_one_builder(minus_one_name, "Const");
    NodeDef minus_one_node;
    Tensor t_minus_one(DT_INT32, TensorShape({1}));
    auto minus_one_data = t_minus_one.tensor<int, 1>();
    minus_one_data(0) = -1;
    status =
        minus_one_builder
            .Attr("dtype", t_minus_one.dtype())
            .Attr("value", t_minus_one)
            .Finalize(&minus_one_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    minus_one_node.set_device(node->def().device());
    Node* minus_one = graph->AddNode(minus_one_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    minus_one->set_assigned_device_name(node->assigned_device_name());

    std::vector<NodeDefBuilder::NodeOut> slice_inputs;
    slice_inputs.emplace_back(shape_name, 0, shape_dtype);
    slice_inputs.emplace_back(one_name, 0, t_one.dtype());
    slice_inputs.emplace_back(minus_one_name, 0, t_minus_one.dtype());
    NodeDefBuilder slice_builder(slice_name, "Slice");
    slice_builder.Input(slice_inputs[0]);
    slice_builder.Input(slice_inputs[1]);
    slice_builder.Input(slice_inputs[2]);
    NodeDef slice_node;
    status =
        slice_builder
            .Attr("T", shape_dtype)
            .Finalize(&slice_node);
    if (!status.ok()) {
      LOG(ERROR) << "Slice node construction failed with" << status;
      return false;
    }
    slice_node.set_device(node->def().device());
    Node* slice = graph->AddNode(slice_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    slice->set_assigned_device_name(node->assigned_device_name());
    graph->AddEdge(shape, 0, slice, 0);
    graph->AddEdge(one, 0, slice, 1);
    graph->AddEdge(minus_one, 0, slice, 2);

    std::vector<Node*> dst_nodes;
    std::vector<int> dst_inputs;
    for (const Edge* e : node->out_edges()) {
      dst_nodes.push_back(e->dst());
      dst_inputs.push_back(e->dst_input());
    }
    for (unsigned int i = 0; i < dst_nodes.size(); i++) {
      graph->UpdateEdge(slice, 0, dst_nodes[i], dst_inputs[i]);
    }
    graph->RemoveNode(node);

    changed = true;
  }
  return changed;
}

// Change ->Unpack->Transpose*n-> to ->Transpose->Unpack->
bool FuseTransposesAfterUnpack(Graph* graph) {
  static int count = 0;
  VLOG(2) << "FuseTransposesAfterUnpack";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    Node* unpack = node;
    std::vector<Node*> transposes;
    std::vector<Node*> transpose_in_1;
    bool can_reorder = true;
    for (Node* out : unpack->out_nodes()) {
      if (out->type_string() != "Transpose") {
        can_reorder = false;
        break;
      }
      transposes.push_back(out);
      Node* n = nullptr;
      out->input_node(1, &n);
      transpose_in_1.push_back(n);
    }
    if (!can_reorder || transposes.size() < 2) continue;
    for (unsigned int i = 1; i < transposes.size(); i++) {
      if (transpose_in_1[i] != transpose_in_1[0]) continue;
    }
    VLOG(2) << "FuseTransposesAfterUnpack: found pattern";

    string prefix = "GemmOptimizer/FuseTransposesAfterUnpack/" +
                    std::to_string(count++);
    string one_name = prefix + "Transpose/Add/one";
    NodeDefBuilder one_builder(one_name, "Const");
    NodeDef one_node;
    Tensor t_one(DT_INT32, TensorShape({1}));
    auto one_data = t_one.tensor<int, 1>();
    one_data(0) = 1;
    Status status =
        one_builder
            .Attr("dtype", t_one.dtype())
            .Attr("value", t_one)
            .Finalize(&one_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    one_node.set_device(transpose_in_1[0]->def().device());
    Node* one = graph->AddNode(one_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    one->set_assigned_device_name(transpose_in_1[0]->assigned_device_name());
 
    DataType shape_dtype = transpose_in_1[0]->output_type(0);
    string add_name = prefix + "/Add";
    std::vector<NodeDefBuilder::NodeOut> add_inputs;
    const Edge* param_to_transpose = nullptr;
    transposes[0]->input_edge(1, &param_to_transpose);
    add_inputs.emplace_back(param_to_transpose->src()->name(),
                            param_to_transpose->src_output(), shape_dtype);
    add_inputs.emplace_back(one_name, 0, shape_dtype);
    NodeDefBuilder add_builder(add_name, "Add");
    add_builder.Input(add_inputs[0]);
    add_builder.Input(add_inputs[1]);
    NodeDef add_node;
    status =
        add_builder
            .Attr("T", shape_dtype)
            .Finalize(&add_node);
    if (!status.ok()) {
      LOG(ERROR) << "BiasAdd node construction failed with" << status;
      return false;
    }
    add_node.set_device(transpose_in_1[0]->def().device());
    Node* add = graph->AddNode(add_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    add->set_assigned_device_name(transpose_in_1[0]->assigned_device_name());
    graph->AddEdge(param_to_transpose->src(),
        param_to_transpose->src_output(), add, 0);
    graph->AddEdge(one, 0, add, 1);

    string zero_name = prefix + "Transpose/Concat/zero";
    NodeDefBuilder zero_builder(zero_name, "Const");
    NodeDef zero_node;
    Tensor t_zero(DT_INT32, TensorShape({1}));
    auto zero_data = t_zero.tensor<int, 1>();
    zero_data(0) = 0;
    status =
        zero_builder
            .Attr("dtype", t_zero.dtype())
            .Attr("value", t_zero)
            .Finalize(&zero_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    zero_node.set_device(transpose_in_1[0]->def().device());
    Node* zero = graph->AddNode(zero_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    zero->set_assigned_device_name(transpose_in_1[0]->assigned_device_name());

    // Add a Concat Op to generate new shape
    string zero_scalar_name = prefix + "/Transpose/Concat/concat_dim";
    NodeDefBuilder zero_scalar_builder(zero_scalar_name, "Const");
    NodeDef zero_scalar_node;
    Tensor t_zero_scalar(int(0));
    status =
        zero_scalar_builder
            .Attr("dtype", t_zero_scalar.dtype())
            .Attr("value", t_zero_scalar)
            .Finalize(&zero_scalar_node);
    if (!status.ok()) {
      LOG(ERROR) << "Const node construction failed with" << status;
      return false;
    }
    zero_scalar_node.set_device(transpose_in_1[0]->def().device());
    Node* zero_scalar = graph->AddNode(zero_scalar_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    zero_scalar->set_assigned_device_name(
        transpose_in_1[0]->assigned_device_name());

    string concat_name = prefix + "/Transpose/Concat";
    NodeDefBuilder concat_builder(concat_name, "Concat");
    NodeDefBuilder::NodeOut concat_dim(zero_scalar->name(), 0, shape_dtype);
    std::vector<NodeDefBuilder::NodeOut> concat_inputs;
    concat_inputs.emplace_back(zero_name, 0, shape_dtype);
    concat_inputs.emplace_back(add_name, 0, shape_dtype);
    concat_builder.Input(concat_dim);
    concat_builder.Input(concat_inputs);
    NodeDef concat_node;
    status =
        concat_builder
            .Attr("N", 2)
            .Attr("T", shape_dtype)
            .Finalize(&concat_node);
    if (!status.ok()) {
      LOG(ERROR) << "Concat node construction failed with" << status;
      return false;
    }
    concat_node.set_device(transpose_in_1[0]->def().device());
    Node* concat = graph->AddNode(concat_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    concat->set_assigned_device_name(
        transpose_in_1[0]->assigned_device_name());
    graph->AddEdge(zero_scalar, 0, concat, 0);
    graph->AddEdge(zero, 0, concat, 1);
    graph->AddEdge(add, 0, concat, 2);

    // Add a new Transpose Op
    string transpose_name = prefix + "/Transpose";
    NodeDefBuilder transpose_builder(transpose_name, "Transpose");
    std::vector<NodeDefBuilder::NodeOut> transpose_inputs;
    const Edge* to_unpack;
    unpack->input_edge(0, &to_unpack);
    int src_output = to_unpack->src_output();
    Node* unpack_in = to_unpack->src();
    transpose_inputs.emplace_back(unpack_in->name(), src_output,
                                  transposes[0]->input_type(0));
    transpose_inputs.emplace_back(concat->name(), 0, shape_dtype);
    transpose_builder.Input(transpose_inputs[0]);
    transpose_builder.Input(transpose_inputs[1]);
    NodeDef transpose_node;
    status =
        transpose_builder
            .Attr("T", transposes[0]->input_type(0))
            .Attr("Tperm", shape_dtype)
            .Finalize(&transpose_node);
    if (!status.ok()) {
      LOG(ERROR) << "Reshape node construction failed with" << status;
      return false;
    }
    transpose_node.set_device(transposes[0]->def().device());
    Node* transpose = graph->AddNode(transpose_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    transpose->set_assigned_device_name(
        transposes[0]->assigned_device_name());
    graph->AddEdge(unpack_in, src_output, transpose, 0);
    graph->AddEdge(concat, 0, transpose, 1);
 
    graph->UpdateEdge(transpose, 0, unpack, 0);
    for (Node* t : transposes) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : t->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      const Edge* e = nullptr;
      t->input_edge(0, &e);
      int src_output = e->src_output();
      for (unsigned int i = 0; i < dst_nodes.size(); i++) {
        graph->UpdateEdge(unpack, src_output, dst_nodes[i], dst_inputs[i]);
      }
      graph->RemoveNode(t);
    }
    changed= true;
  }
  return changed;
}

bool FuseBinaryOpsSharingACommonInputAfterUnpack(Graph* graph);

// ->Unpack->BinaryOp*n to ->BinaryOp->Unpack
bool FuseBinaryOpsAfterUnpack(Graph* graph) {
  static int count = 0;
  VLOG(2) << "FuseBinaryOpsAfterUnpack";
  bool changed = FuseBinaryOpsSharingACommonInputAfterUnpack(graph);
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  const std::unordered_set<string> binary_op_set = GetBinaryOps();
  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    // group ops based on its inputs
    std::vector<Node*> binary_ops;
    std::map<string, std::vector<Node*>> binary_ops_m;
    std::vector<std::pair<Node*, int>> common_extra_input;
    const Node* another_node = nullptr;
    bool can_fuse = true;
    string binary_type;
    for (Node* out : node->out_nodes()) {
      string out_type = out->type_string();
      if (binary_type.empty()) {
        binary_type = out->type_string();
        if (binary_op_set.find(binary_type) == binary_op_set.end()) {
          can_fuse = false;
          break;
        }
      }
      if (out->type_string() != binary_type) {
        can_fuse = false;
        break;
      }

      if (out->num_inputs() < 2) {
        can_fuse = false;
        break;
      }

      bool has_same_another_input = true;
      bool another_input_from_unpack_or_const = true;
      bool has_common_extra_input = true;
      for (int i = 0; i < out->num_inputs(); ++i) {
        const Edge* edge = nullptr;
        out->input_edge(i, &edge);
        Node* n = edge->src();
        if (i < 2) {
          if (n != node) {
            if (another_node == nullptr) {
              string type = n->type_string();
              if (type == "Unpack" || type == "Const") {
                another_node = n;
              } else {
                another_input_from_unpack_or_const = false;
                break;
              }
            } else {
              if (n != another_node) {
                has_same_another_input = false;
              }
            }
          }
        } else {
          size_t vec_idx = i - 2;
          if (vec_idx >= common_extra_input.size()) {
            common_extra_input.emplace_back(std::make_pair(n, edge->src_output()));
          } else {
            if (n != common_extra_input[vec_idx].first ||
                edge->src_output() != common_extra_input[vec_idx].second) {
              has_common_extra_input = false;
            }
          }
        }
      }

      if (has_same_another_input && another_input_from_unpack_or_const && has_common_extra_input)
        binary_ops.push_back(out);
    }

    if (!can_fuse || binary_ops.size() < 2) continue;

    for (Node* b : binary_ops) {
      for (int i = 0; i < std::min(b->num_inputs(), 2); ++i) {
        const Node* n = nullptr;
        b->input_node(i, &n);
        if (n != node) {
          string key;
          if (another_node->type_string() == "Unpack") {
            key = n->name();
          } else {
            TensorShapeProto s = n->def().attr().at("value").
                                 tensor().tensor_shape();
            for (int i = 0; i < s.dim_size(); i++) {
              key += std::to_string(s.dim(i).size()) + ",";
            }
          }
          if (binary_ops_m.find(key) != binary_ops_m.end()) {
            binary_ops_m[key].push_back(b);
          } else {
            std::vector<Node*> ins;
            ins.push_back(b);
            binary_ops_m[key] = ins;
          }
          break;
        }
      }
    }
    VLOG(2) << "FuseBinaryOpsAfterUnpack: found pattern";

    // Fuse binary_ops in each group.
    // For each group, do:
    // (1) add two Pack nodes to stack inputs on both sides respectively,
    // (2) add a new node to replace old binary_ops, and
    // (3) add a Unpack node to split result.
    DataType dtype = node->output_type(0);
    std::map<string, std::vector<Node*>>::iterator iter;
    iter = binary_ops_m.begin();
    while (iter != binary_ops_m.end()) {
      std::vector<Node *> *binary_ops_group = &(iter->second);
      if (binary_ops_group->size() < 2) iter++;
      std::sort(binary_ops_group->begin(), binary_ops_group->end(),
                [node](Node *a, Node *b) {
                  const Edge *e = nullptr;
                  a->input_edge(0, &e);
                  int a_src_output = e->src_output();
                  b->input_edge(0, &e);
                  int b_src_output = e->src_output();
                  return a_src_output < b_src_output;
                });
      std::vector<const Edge *> inputs[2];
      VLOG(2) << "the following ops are fused: ";
      for (Node *b : *binary_ops_group) {
        for (int i = 0; i < 2; ++i) {
          VLOG(2) << b->name();

          const Edge *e = nullptr;
          b->input_edge(i, &e);
          inputs[e->dst_input()].push_back(e);
        }
      }

      // Add two Pack nodes to group on two sides, respectively
      Node *packs[2];
      string pack_names[2];
      string prefix = "GemmOptimizer/FuseBinaryOpsAfterUnpack/" +
                      std::to_string(count++);
      pack_names[0] = prefix + "/Pack_0";
      pack_names[1] = prefix + "/Pack_1";
      Status status;
      for (int i = 0; i < 2; i++) {
        std::vector<NodeDefBuilder::NodeOut> pack_inputs;
        for (const Edge *e : inputs[i]) {
          string s = e->src()->name();
          pack_inputs.emplace_back(s, e->src_output(), dtype);
        }
        NodeDefBuilder pack_builder(pack_names[i], "Pack");
        pack_builder.Input(pack_inputs);
        NodeDef pack_node;
        status =
            pack_builder
                .Attr("N", (int) inputs[i].size())
                .Attr("T", dtype)
                .Attr("axis", 0)
                .Finalize(&pack_node);
        if (!status.ok()) {
          LOG(ERROR) << "Pack node construction failed with" << status;
          return false;
        }
        pack_node.set_device(inputs[i][0]->src()->def().device());
        packs[i] = graph->AddNode(pack_node, &status);
        if (!status.ok()) {
          LOG(ERROR) << "Adding node failed " << status;
          return false;
        }
        packs[i]->set_assigned_device_name(
            inputs[i][0]->src()->assigned_device_name());
        for (unsigned int j = 0; j < inputs[i].size(); j++) {
          graph->AddEdge(inputs[i][j]->src(), inputs[i][j]->src_output(),
                         packs[i], j);
        }
      }

      // Add a new BatchMatMulV2
      std::vector<NodeDefBuilder::NodeOut> binary_op_inputs;
      binary_op_inputs.emplace_back(pack_names[0], 0, dtype);
      binary_op_inputs.emplace_back(pack_names[1], 0, dtype);
      if (!common_extra_input.empty()) {
        for (size_t i = 0; i < common_extra_input.size(); ++i) {
          const Node *extra_node = common_extra_input[i].first;
          int extra_src_idx = common_extra_input[i].second;
          binary_op_inputs.emplace_back(extra_node->name(), extra_src_idx, extra_node->output_type(extra_src_idx));
        }
      }
      string binary_op_name = prefix;
      string type = (*binary_ops_group)[0]->type_string();
      string new_type;
      if (type == "MatMul" ||
          type == "BatchMatMul" ||
          type == "BatchMatMulV2") {
        binary_op_name += "/BatchMatMulV2";
        new_type = "BatchMatMulV2";
      } else if (type == "IndicatorMatMul") {
        binary_op_name += "/ParallelIndicatorMatMul";
        new_type = "ParallelIndicatorMatMul";
      } else {
        binary_op_name += "/" + type;
        new_type = type;
      }
      NodeDefBuilder binary_op_builder(binary_op_name, new_type);
      for (size_t i = 0; i < binary_op_inputs.size(); ++i) {
        binary_op_builder.Input(binary_op_inputs[i]);
      }
      NodeDef binary_op_node;
      bool transpose_a = false;
      bool transpose_b = false;
      if (type == "MatMul") {
        transpose_a = (*binary_ops_group)[0]->def().attr().at("transpose_a").b();
        transpose_b = (*binary_ops_group)[0]->def().attr().at("transpose_b").b();
      } else if (type == "BatchMatMul" || type == "BatchMatMulV2" || type == "IndicatorMatMul") {
        transpose_a = (*binary_ops_group)[0]->def().attr().at("adj_x").b();
        transpose_b = (*binary_ops_group)[0]->def().attr().at("adj_y").b();
      }

      if (type == "MatMul" ||
          type == "BatchMatMul" ||
          type == "BatchMatMulV2") {
        status =
            binary_op_builder
                .Attr("adj_x", transpose_a)
                .Attr("adj_y", transpose_b)
                .Attr("T", dtype)
                .Finalize(&binary_op_node);

      } else if (type == "IndicatorMatMul") {
        status =
            binary_op_builder
                .Attr("adj_x", transpose_a)
                .Attr("adj_y", transpose_b)
                .Attr("parallel_num", (int) (*binary_ops_group).size())
                .Attr("T", dtype)
                .Finalize(&binary_op_node);
      } else {
        status =
            binary_op_builder
                .Attr("T", dtype)
                .Finalize(&binary_op_node);
      }
      if (!status.ok()) {
        LOG(ERROR) << "BatchMatMulV2 node construction failed with" << status;
        return false;
      }
      binary_op_node.set_device((*binary_ops_group)[0]->def().device());
      Node* binary_op = graph->AddNode(binary_op_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      binary_op->set_assigned_device_name((*binary_ops_group)[0]->
                                          assigned_device_name());
      graph->AddEdge(packs[0], 0, binary_op, 0);
      graph->AddEdge(packs[1], 0, binary_op, 1);

      // Add an Unpack node to split result
      string unpack_name = prefix + "/Unpack" ;
      NodeDefBuilder::NodeOut unpack_input(binary_op_name, 0, dtype);
      NodeDefBuilder unpack_builder(unpack_name, "Unpack");
      unpack_builder.Input(unpack_input);
      NodeDef unpack_node;
      status =
          unpack_builder
              .Attr("num", (int)(*binary_ops_group).size())
              .Attr("T", dtype)
              .Attr("axis", 0)
              .Finalize(&unpack_node);
      if (!status.ok()) {
        LOG(ERROR) << "Unpack node construction failed with" << status;
        return false;
      }
      unpack_node.set_device((*binary_ops_group)[0]->def().device());
      Node* unpack = graph->AddNode(unpack_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      unpack->set_assigned_device_name((*binary_ops_group)[0]->
                                       assigned_device_name());
      graph->AddEdge(binary_op, 0, unpack, 0);
   
      // Add edges to forward split results to nodes after original binary_ops,
      // and remove original binary_ops
      int index = 0;
      for (Node* b : *binary_ops_group) {
        std::vector<Node*> dst_nodes;
        std::vector<int> dst_inputs;
        for (const Edge* e : b->out_edges()) {
          dst_nodes.push_back(e->dst());
          dst_inputs.push_back(e->dst_input());
        }
        for (unsigned int i = 0; i < dst_nodes.size(); i++) {
          graph->UpdateEdge(unpack, index, dst_nodes[i], dst_inputs[i]);
        }
        graph->RemoveNode(b);
        index++;
      }
 
      changed = true;
      iter++;
    }
  }
 
  return changed;
}

// ->Unpack->UnaryOp*n to ->UnaryOp->Unpack
bool FuseUnaryOpsAfterUnpack(Graph* graph) {
  VLOG(2) << "FuseUnaryOpsAfterUnpack";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  const std::unordered_set<string> unary_op_set = GetUnaryOps();
  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    Node* unpack = node;
    std::vector<Node*> unary_ops;
    bool can_reorder = true;
    string unary_type;
    for (Node* out : unpack->out_nodes()) {
      if (unary_type.empty()) {
        unary_type = out->type_string();
        if (unary_op_set.find(unary_type) == unary_op_set.end()) {
          can_reorder = false;
          break;
        }
      }
      if (out->type_string() != unary_type) {
        can_reorder = false;
        break;
      }
      unary_ops.push_back(out);
    }
    if (!can_reorder || unary_ops.size() < 2) continue;
    VLOG(2) << "FuseUnaryOpsAfterUnpack: found pattern";
    bool is_first = true;
    for (Node* u : unary_ops) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : u->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      const Edge* e = nullptr;
      u->input_edge(0, &e);
      int src_output = e->src_output();
      for (unsigned int i = 0; i < dst_nodes.size(); i++) {
        graph->UpdateEdge(unpack, src_output, dst_nodes[i], dst_inputs[i]);
      }
      if (is_first) {
        is_first = false;
      } else {
        graph->RemoveNode(u);
      }
    }
    const Edge* to_unpack = nullptr;
    unpack->input_edge(0, &to_unpack);
    graph->UpdateEdge(to_unpack->src(), to_unpack->src_output(),
                      unary_ops[0], 0);
    graph->UpdateEdge(unary_ops[0], 0, unpack, 0);
    changed = true;
  }

  return changed;
}

// ->Unpack->BinaryOp*n to ->BinaryOp->Unpack
bool FuseBinaryOpsSharingACommonInputAfterUnpack(Graph* graph) {
  VLOG(2) << "FuseBinaryOpsSharingACommonInputAfterUnpack";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  const std::unordered_set<string> binary_op_set = GetBinaryOps();
  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    Node* unpack = node;
    std::vector<Node*> binary_ops;

    string binary_type;
    Node* another_input = nullptr;
    int another_src_output = -1;
    int unpack_dst_input = -1;
    bool can_reorder = true;

    // To fuse, all candidates must
    // (1) be binary ops listed in binary_op_set,
    // (2) have the same op type, and
    // (3) have the same inputs (other than inputs from unpack).
    for (Node* out : unpack->out_nodes()) {
      if (binary_type.empty()) {
        binary_type = out->type_string();
        if (binary_op_set.find(binary_type) == binary_op_set.end()) {
          can_reorder = false;
          break;
        }
        if (out->in_edges().size() != 2) {
          can_reorder = false;
          break;
        }
        for (const Edge* e : out->in_edges()) {
          if (e->src() == unpack) {
            unpack_dst_input = e->dst_input();
            continue;
          }
          another_input = e->src();
          another_src_output = e->src_output();
        }
      }
      // check conditions (1) and (2)
      if (out->type_string() != binary_type ||
          another_input == nullptr) {
        can_reorder = false;
        break;
      }

      // check condition (3)
      for (const Edge* e : out->in_edges()) {
        if (e->src() == unpack) continue;
        if (e->src() != another_input ||
            e->src_output() != another_src_output) {
          can_reorder = false;
          break;
        }
      }

      if (!can_reorder) break;
      binary_ops.push_back(out);
    }

    if (!can_reorder || binary_ops.size() < 2) continue;
    VLOG(2) << "FuseBinaryOpsSharingACommonInputAfterUnpack: found pattern";
    bool is_first = true;
    for (Node* b : binary_ops) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : b->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      const Edge* e = nullptr;
      b->input_edge(unpack_dst_input, &e);
      int src_output = e->src_output();
      for (unsigned int i = 0; i < dst_nodes.size(); i++) {
        graph->UpdateEdge(unpack, src_output,
                          dst_nodes[i], dst_inputs[i]);
      }
      if (is_first) {
        is_first = false;
      } else {
        graph->RemoveNode(b);
      }
    }
    const Edge* to_unpack = nullptr;
    unpack->input_edge(0, &to_unpack);
    graph->UpdateEdge(to_unpack->src(), to_unpack->src_output(),
                      binary_ops[0], unpack_dst_input);
    graph->UpdateEdge(binary_ops[0], 0, unpack, 0);
    changed = true;
  }

  return changed;
}

void RemoveDeadUnpacksAndPacks(Graph* graph) {
  VLOG(2) << "RemoveDeadUnpacksAndPacks";
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack" &&
        node->type_string() != "Pack") continue;
    if (node->out_edges().size() == 0) {
      graph->RemoveNode(node);
    }
  }
}

bool RemoveUnpacksAndPacks(Graph* graph) {
  VLOG(2) << "RemoveUnpacksAndPacks";
  bool changed = false;
  static int count = 0;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    Node* unpack = node;

    bool can_remove = true;
    int num_split = -1;
    int size_per_split = -1;
    std::map<Node*, int> unpack_split_map;
    for (Node* out : unpack->out_nodes()) {
      if (unpack_split_map.find(out) != unpack_split_map.end()) continue;
      if (out->type_string() != "Pack") {
        can_remove = false;
        break;
      }
      // Check Unpack's outputs are grouped evenly
      if (num_split == -1) {
        int unpack_num = unpack->num_outputs();
        size_per_split = out->num_inputs();
        num_split = unpack_num / size_per_split;
        if ((size_per_split *  num_split) != unpack_num) {
          can_remove = false;
          break;
        }
      }

      // Make sure each Pack takes a full group of Unpack's outputs.
      // This is realized by using the following two checks:
      int src_output = -1;
      for (int i = 0; i < out->num_inputs(); i++) {
        const Edge* e = nullptr;
        out->input_edge(i, &e);
        int temp = e->src_output();
        if (src_output != -1) {
          // (1) check order
          if (temp != (src_output + 1)) {
            can_remove = false;
            break;
          }
        } else {
          // (2) check size and begin index
          if ((temp % size_per_split != 0) ||
              (out->num_inputs() != size_per_split)) {
            can_remove = false;
            break;
          }
          // This Unpack is replaced by Split:(temp/size_per_split)
          unpack_split_map[out] = temp / size_per_split;
        }
        src_output = temp;
      }
      if (!can_remove) break;
    }

    if (!can_remove || unpack_split_map.size() == 0) continue;
    VLOG(2) << "RemoveUnpacksAndPacks: found pattern";

    // for Unpack->Pack*n, insert a Split Op, and
    // repalace Packs' outputs with Split's outputs.
    const Edge* to_unpack = nullptr;
    unpack->input_edge(0, &to_unpack);
    Node* unpack_in = to_unpack->src();
    int unpack_src_output = to_unpack->src_output();
    Node* split = nullptr;
    if (num_split > 1) {
      string prefix = "GemmOptimizer/RemoveUnpacksAndPacks/" +
                      std::to_string(count++);
      string zero_name = prefix + "/Split/zero";
      NodeDefBuilder zero_builder(zero_name, "Const");
      NodeDef zero_node;
      Tensor t_zero(int(0));
      Status status =
          zero_builder
              .Attr("dtype", t_zero.dtype())
              .Attr("value", t_zero)
              .Finalize(&zero_node);
      if (!status.ok()) {
        LOG(ERROR) << "Const node construction failed with" << status;
        return false;
      }
      zero_node.set_device(unpack->def().device());
      Node* zero = graph->AddNode(zero_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      zero->set_assigned_device_name(unpack->assigned_device_name());

      std::vector<NodeDefBuilder::NodeOut> split_inputs;
      split_inputs.emplace_back(zero_name, 0, t_zero.dtype());
      split_inputs.emplace_back(unpack_in->name(), unpack_src_output,
                                unpack_in->output_type(unpack_src_output));
      string split_name = prefix + "/Split";
      NodeDefBuilder split_builder(split_name, "Split");
      split_builder.Input(split_inputs[0]);
      split_builder.Input(split_inputs[1]);
      NodeDef split_node;
      status =
          split_builder
              .Attr("T", unpack_in->output_type(unpack_src_output))
              .Attr("num_split", num_split)
              .Finalize(&split_node);
      if (!status.ok()) {
        LOG(ERROR) << "Split node construction failed with" << status;
        return false;
      }
      split_node.set_device(unpack->def().device());
      split = graph->AddNode(split_node, &status);
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      split->set_assigned_device_name(unpack->assigned_device_name());
      graph->AddEdge(zero, 0, split, 0);
      graph->AddEdge(unpack_in, unpack_src_output, split, 1);
    }

    // for Unpack->Pack pair, update graph directly
    std::map<Node*, int>::iterator it;
    for (it = unpack_split_map.begin();
         it != unpack_split_map.end(); it++) {
      Node* pack = it->first;

      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : pack->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      for (unsigned int i = 0; i < dst_nodes.size(); i++) {
        if (num_split == 1) {
         graph->UpdateEdge(unpack_in, unpack_src_output,
                           dst_nodes[i], dst_inputs[i]);
        } else {
         graph->UpdateEdge(split, it->second,
                           dst_nodes[i], dst_inputs[i]);

        }
      }
      graph->RemoveNode(pack);
    }
    changed = true;
  }
  RemoveDeadUnpacksAndPacks(graph);
  return changed;
}

bool FuseGatherBeforeMatMul(Graph* graph) {
  VLOG(2) << "FuseGatherBeforeMatMul";
  
  static int count = 0;
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  for (Node* node : nodes) {
    if (node->type_string() != "BatchMatMulV2") continue;
    Node *matmul = node;
    // Check Gather
    Node *gather = nullptr;
    matmul->input_node(0, &gather);
    if (!graph->IsValidNode(gather).ok()) continue;
    string type = gather->type_string();
    if (type != "GatherV2") continue;
    // Check axis
    Node* axis = nullptr;
    gather->input_node(2, &axis);
    if (!axis) continue;
    type = axis->type_string();
    if (type != "Const") continue;
    if (axis->def().attr().at("value").i() != 0) continue;

    // Prepare op name
    string prefix = "GemmOptimizer/FuseGatherBeforeMatMul/" + matmul->name();
    string op_name = prefix + "/IndicatorMatMul_" + std::to_string(count++);

    // Build NodeDef
    std::vector<NodeDefBuilder::NodeOut> ind_matmul_inputs;
    Node *input_nodes[3] = {nullptr, nullptr, nullptr};
    int input_idx[3] = {0, 0, 0};

    const Edge* gather_input_0;
    gather->input_edge(0, &gather_input_0);
    input_nodes[0] = gather_input_0->src();
    input_idx[0] = gather_input_0->src_output();

    const Edge* matmul_input_1;
    matmul->input_edge(1, &matmul_input_1);
    input_nodes[1] = matmul_input_1->src();
    input_idx[1] = matmul_input_1->src_output();

    const Edge* gather_input_1;
    gather->input_edge(1, &gather_input_1);
    input_nodes[2] = gather_input_1->src();
    input_idx[2] = gather_input_1->src_output();

    if (!input_nodes[0] || !input_nodes[1] || !input_nodes[2]) continue;
    DataType dtype = input_nodes[0]->output_type(0);
    DataType ind_dtype = input_nodes[2]->output_type(0);
    ind_matmul_inputs.emplace_back(input_nodes[0]->name(), input_idx[0], dtype);
    ind_matmul_inputs.emplace_back(input_nodes[1]->name(), input_idx[1], dtype);
    ind_matmul_inputs.emplace_back(input_nodes[2]->name(), input_idx[2], ind_dtype);
    NodeDefBuilder ind_matmul_builder(op_name, "IndicatorMatMul");
    ind_matmul_builder.Input(ind_matmul_inputs[0]);
    ind_matmul_builder.Input(ind_matmul_inputs[1]);
    ind_matmul_builder.Input(ind_matmul_inputs[2]);

    NodeDef ind_matmul_def;
    bool transpose_a = false, transpose_b = false;
    if (matmul->def().attr().find("adj_x") != matmul->def().attr().end()) {
      transpose_a = matmul->def().attr().at("adj_x").b();
    }
    if (matmul->def().attr().find("adj_y") != matmul->def().attr().end()) {
      transpose_b = matmul->def().attr().at("adj_y").b();
    }
    Status status = ind_matmul_builder
        .Attr("adj_x", transpose_a)
        .Attr("adj_y", transpose_b)
        .Attr("T", dtype)
        .Device(matmul->def().device())
        .Finalize(&ind_matmul_def);
    if (!status.ok()) {
      LOG(ERROR) << "IndicatorMatMul node construction failed with " << status;
      return changed;
    }

    // Insert IndicatorMatMul node
    Node *ind_matmul_node = graph->AddNode(ind_matmul_def, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return changed;
    }
    // Update input edge
    graph->AddEdge(input_nodes[0], input_idx[0], ind_matmul_node, 0);
    graph->AddEdge(input_nodes[1], input_idx[1], ind_matmul_node, 1);
    graph->AddEdge(input_nodes[2], input_idx[2], ind_matmul_node, 2);
    // Update output edge
    UpdateAllEdge(graph, ind_matmul_node, matmul);
    // Remove useless node
    auto RemoveNodeSafely = [&](Node* node) {
      if (node->out_edges().empty()) {
        graph->RemoveNode(node);
      }
    };
    RemoveNodeSafely(matmul);
    RemoveNodeSafely(gather);
    RemoveNodeSafely(axis);
    changed = true;
  }
  return changed;
}

void SplitConcatBasedOnInputDevices(Graph* graph) {
  // 1. Get all GPU concats
  std::vector<Node*> gpu_concats;
  for (Node* node : graph->nodes()) {
    if (node->type_string() == "Concat" ||
        node->type_string() == "ConcatV2") {
      std::string device = node->requested_device();
      if (device.find("GPU") != std::string::npos ||
          device.find("gpu") != std::string::npos) {
        VLOG(2) << "Found a SplitConcatBasedOnInputDevices candidate: "
                  << node->DebugString();
        gpu_concats.emplace_back(node);
      }
    }
  }

  // 2. Split the concats whose inputs are from different devices
  //    into multiple concats
  for (Node* concat : gpu_concats) {
    VLOG(2) << "Try to split concat: " << concat->DebugString();
    // 2.1. Group inputs of each concat into clusters
    int num_inputs = concat->num_inputs();
    std::vector<const Edge*> inputs(num_inputs);
    std::vector<std::string> devices(num_inputs);

    bool all_inputs_are_int = true;
    for (const Edge* e : concat->in_edges()) {
      Node* src = e->src();
      int dst_input = e->dst_input();
      inputs[dst_input] = e;
      std::string device = src->requested_device();
      devices[dst_input] = device;
      if (concat->input_type(dst_input) != DT_INT32 &&
          concat->input_type(dst_input) != DT_INT64) {
        all_inputs_are_int = false;
      }
    }
    if (all_inputs_are_int) continue;

    int axis_input = 0;
    if (concat->type_string() == "ConcatV2") {
      axis_input = inputs.size() - 1;
    }

    std::vector<std::vector<const Edge*>> cluster_inputs;
    std::vector<std::string> cluster_devices;
    std::vector<const Edge*> temp0;
    cluster_inputs.emplace_back(temp0);
    int cluster_idx = 0;

    int begin = 0, end = num_inputs;
    if (axis_input == 0) {
      begin = 1;
    } else {
      end = num_inputs - 1;
    }
 
    std::string device = devices[begin];
    cluster_inputs[0].emplace_back(inputs[begin]);
    cluster_devices.emplace_back(device);
    VLOG(2) << "Clustering, cluster " << cluster_idx
              << ", on " << cluster_devices[cluster_idx]
              << " includes: "
              << inputs[begin]->DebugString();
    for (int i = begin + 1; i < end; i++) {
      if (devices[i] != device ||
          (devices[i].find("GPU") != std::string::npos ||
           devices[i].find("gpu") != std::string::npos)) {
        std::vector<const Edge*> temp1;
        cluster_inputs.emplace_back(temp1);
        device = devices[i];
        cluster_devices.emplace_back(device);
        cluster_idx++;
      }
      VLOG(2) << "Clustering, cluster " << cluster_idx
                << ", on " << cluster_devices[cluster_idx]
                << " includes: "
                << inputs[i]->DebugString();
      cluster_inputs[cluster_idx].emplace_back(inputs[i]);
    }
    int num_clusters = cluster_inputs.size();
    CHECK(num_clusters <= num_inputs - 1);
    if (num_clusters == 1 || num_clusters == num_inputs - 1) {
      VLOG(2) << "Do not split concat " << concat->DebugString();
      continue;
    }

    VLOG(2) << "Split concat: " << concat->DebugString();
    std::vector<Node*> new_inputs(num_clusters);
    std::vector<int> new_input_src_outputs(num_clusters);
    std::string concat_name = concat->name();
    DataType type = concat->input_type(1);

    const Edge* axis_edge = inputs[axis_input];
    Node* axis = axis_edge->src();
    std::string axis_name = axis->name();
    int axis_src_output = axis_edge->src_output(); 
    DataType axis_type = axis->output_type(axis_src_output);
    NodeDefBuilder::NodeOut concat_axis =
        {axis_name, axis_src_output, axis_type};

    // 2.2. Add a new concat for each cluster
    for (int i = 0; i < num_clusters; i++) {
      VLOG(2) << "Splitting, cluster " << i
                << ", on " << cluster_devices[i]
                << " includes: ";
      std::vector<const Edge*> cluster = cluster_inputs[i];
      if (cluster.size() == 1) {
        new_inputs[i] = cluster[0]->src();
        new_input_src_outputs[i] = cluster[0]->src_output();
        VLOG(2) << "node: " << new_inputs[i]->DebugString();
      } else {
        // add a new concat
        NodeDefBuilder concat_builder(
            concat_name + "_cluster_" + std::to_string(i), "ConcatV2");
        std::vector<NodeDefBuilder::NodeOut> concat_inputs;
        for (unsigned int j = 0; j < cluster.size(); j++) {
          Node* src = cluster[j]->src();
          VLOG(2) << "node: " << src->DebugString();
          int src_output = cluster[j]->src_output();
          concat_inputs.emplace_back(src->name(), src_output, type);
        }
        concat_builder.Input(concat_inputs);
        concat_builder.Input(concat_axis);
        NodeDef concat_node;
        Status status =
            concat_builder
                .Attr("N", (int)cluster.size())
                .Attr("T", type)
                .Attr("Tidx", axis_type)
                .Finalize(&concat_node);
        if (!status.ok()) {
          LOG(ERROR) << "Concat node construction failed with" << status;
          return;
        }
        concat_node.set_device(cluster_devices[i]);
        Node* cluster_concat = graph->AddNode(concat_node, &status);
        if (!status.ok()) {
          LOG(ERROR) << "Adding node failed " << status;
          return;
        }
        cluster_concat->set_assigned_device_name(
            cluster[0]->src()->assigned_device_name());
        for (unsigned int j = 0; j < cluster.size(); j++) {
          graph->AddEdge(cluster[j]->src(), cluster[j]->src_output(),
                         cluster_concat, j);
        }
        graph->AddEdge(axis, axis_src_output,
			           cluster_concat, cluster.size());
        new_inputs[i] = cluster_concat;
        new_input_src_outputs[i] = 0;
      }
    }

    // 2.3. Add a new concat to concat all clusters
    NodeDefBuilder concat_builder(
        concat_name + "_new", "ConcatV2");
    std::vector<NodeDefBuilder::NodeOut> concat_inputs;
    for (unsigned int i = 0; i < new_inputs.size(); i++) {
      concat_inputs.emplace_back(new_inputs[i]->name(),
                                 new_input_src_outputs[i], type);
    }
    concat_builder.Input(concat_inputs);
    concat_builder.Input(concat_axis);
    NodeDef concat_node;
    Status status =
        concat_builder
            .Attr("N", (int)new_inputs.size())
            .Attr("T", type)
            .Attr("Tidx", axis_type)
            .Finalize(&concat_node);
    if (!status.ok()) {
      LOG(ERROR) << "Concat node construction failed with" << status;
      return;
    }
    concat_node.set_device(concat->def().device());
    Node* new_concat = graph->AddNode(concat_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return;
    }
    new_concat->set_assigned_device_name(concat->assigned_device_name());
    for (unsigned int i = 0; i < new_inputs.size(); i++) {
      graph->AddEdge(new_inputs[i], new_input_src_outputs[i],
                     new_concat, i);
    }
    graph->AddEdge(axis, axis_src_output, new_concat, new_inputs.size());
 
    // 2.4. Remove old concat
    std::vector<Node*> dst_nodes;
    std::vector<int> dst_inputs;
    for (const Edge* e : concat->out_edges()) {
      dst_nodes.push_back(e->dst());
      dst_inputs.push_back(e->dst_input());
    }
    for (unsigned int i = 0; i < dst_nodes.size(); i++) {
      graph->UpdateEdge(new_concat, 0, dst_nodes[i], dst_inputs[i]);
    }
    graph->RemoveNode(concat);
  }
}

void SetXlaCompileFlag(Graph* graph) {
  for (Node* node : graph->nodes()) {
    std::string requested_device = node->requested_device();
    VLOG(2) << "node: " << node->DebugString();
    VLOG(2) << "node requested_device: " << node->requested_device();
    VLOG(2) << "node device: " << node->def().device();
    VLOG(2) << "node assigned_device_name: " << node->assigned_device_name();
    if (requested_device.find("CPU") != std::string::npos ||
        requested_device.find("cpu") != std::string::npos) {
      VLOG(2) << "node: " << node->DebugString();
      node->AddAttr("_XlaCompile", false);
    }
  }
}

void FuseGemmKernels(Graph* graph) {  
  bool gemm_fusion = true;
  ReadBoolFromEnvVar("TF_ENABLE_GEMM_FUSION", true, &gemm_fusion);
  if (!gemm_fusion) return;

  SplitConcatBasedOnInputDevices(graph);
  while(1) {
    bool graph_changed =
        FuseGatherBeforeMatMul(graph) ||
        ReorderReshapeAndBiasAdd(graph) ||
        RemoveReshapesBeforeMatMul(graph) ||
        ConstantFoldingForContinuousMatMuls(graph) ||
        FuseMatMuls(graph) ||
        FuseBiasAddsAfterBatchMatMulUnpack(graph) ||
        RemoveShapeAfterReshape(graph) ||
        FuseReshapesAfterUnpack(graph) ||
        FuseBinaryOpsAfterUnpack(graph) ||
        FuseTransposesAfterUnpack(graph) ||
        RemoveUnpackBeforeShape(graph) ||
        FuseUnaryOpsAfterUnpack(graph) ||
        RemoveUnpacksAndPacks(graph);
    if (!graph_changed) break;
  }
  SetXlaCompileFlag(graph);
}
}  // end namespace

Status GemmOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  VLOG(1) << "GemmOptimizer is on.";
  static int pass = 0;
  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile("before_gemm", item.graph);
    std::fstream f;
    f.open("before_gemm_" + std::to_string(pass) + ".pb",
           std::fstream::out | std::fstream::binary);
    f << item.graph.SerializeAsString();
    f.close();
  }

  // convert graphdef to graph
  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());
  Graph graph(flib);
  Status status = ConvertGraphDefToGraph(GraphConstructorOptions(),
                                         item.graph, &graph);
  if (!status.ok()) {
    LOG(WARNING) << "ConvertGraphDefToGraph failed: " << status.ToString();
    *optimized_graph = item.graph;
    return Status::OK();
  }

  FuseGemmKernels(&graph);

  // convert graph to graphdef
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile("after_gemm", *optimized_graph);
    std::fstream f;
    f.open("after_gemm_" + std::to_string(pass) + ".pb",
           std::fstream::out | std::fstream::binary);
    f << optimized_graph->SerializeAsString();
    f.close();
  }
  pass++;
  return Status::OK();
}

void GemmOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
