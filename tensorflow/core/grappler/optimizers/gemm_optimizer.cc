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

namespace tensorflow {
namespace grappler {

namespace {

// TODO(ylxu): add all ops that do not change tensor shapes
std::set<string> GetOpsWithUnchangedShape() {
  std::set<string> ops_with_unchanged_shape = {
      "BiasAdd",
      "Sigmoid",
      "Tanh",
      "Relu"};
  return ops_with_unchanged_shape;
}

// Change MatMul_0->[Reshape]*n->MatMul_1 to MatMul_0->MatMul_1
bool RemoveReshapeBetweenMatMuls(Graph* graph) {
  bool changed = false; 
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() == "MatMul") {
      Node* current = nullptr;
      node->input_node(0, &current);
      string op = current->type_string();
      if (op != "Reshape") continue;

      Node* non_reshape_before_matmul_1 = nullptr;
      std::set<string> ops_with_unchanged_shape = GetOpsWithUnchangedShape();
      while (op == "Reshape" || (ops_with_unchanged_shape.find(op) !=
                                 ops_with_unchanged_shape.end())) {
        if (op != "Reshape") non_reshape_before_matmul_1 = current;
        Node* temp = nullptr;
        current->input_node(0, &temp);
        current = temp;
        op = current->type_string();
      }
      if (op == "MatMul") {
        Node* matmul_1 = node;
        Node* matmul_0 = current;
        // TODO(ylxu): check shape compatibility using matmul weights
        if (non_reshape_before_matmul_1 == nullptr) {
          non_reshape_before_matmul_1 = matmul_0;
        }
        const Edge* e;
        matmul_1->input_edge(0, &e);
        graph->RemoveEdge(e);
        graph->AddEdge(non_reshape_before_matmul_1, 0, matmul_1, 0);
        changed = true;
      }
    }
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
      int num = bias_dst_nodes.size();
      for (int i = 0; i < num; i++) {
        graph->UpdateEdge(reshape, 0, bias_dst_nodes[i], bias_dst_inputs[i]);
      }
      graph->AddEdge(matmul, 0, bias, 0);
      graph->AddEdge(bias, 0, reshape, 0);
      changed = true;
    }
  }
  return changed;
}

// op--->MatMul to op->BatchMatMul->Unstack--->
//    |->MatMul                             |->
//    |->MatMul                             |->
//       ...
bool FuseMatMuls(Graph* graph) {
  static int count = 0;
  LOG(INFO) << "FuseMatMuls";
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
    for (int i = 1; i < src_outputs.size(); i++) {
      if (src_outputs[i] != src_outputs[0]) same_input = false;
    }
    if (!same_input) {
      LOG(INFO) << "Op->MatMuls*n does not have the same input, skip fusion.";
      continue;
    }
    // TODO(ylxu): check the compatibility of shapes (weights)
    // and attrs (transpose_a and transpose_b).

    LOG(INFO) << "FuseMatMuls: found pattern";
    std::sort(matmuls.begin(), matmuls.end(),
              [node](Node* a, Node* b){
      return a->name().compare(b->name()) < 0;
    });
    std::vector<Node*> weights;
    for (Node* m : matmuls) {
      Node* w = nullptr;
      m->input_node(1, &w);
      weights.push_back(w);
    }
    // Add a Pack node to group weights
    string prefix = "GemmOptimizer/FuseMatMuls/" + std::to_string(count++);
    string pack_name = prefix + "/Pack";
    std::vector<NodeDefBuilder::NodeOut> pack_inputs;
    DataType dtype = weights[0]->output_type(0);
    for (Node* w : weights) {
      // TODO(ylxu): src_output may not be 0.
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
    pack->set_assigned_device_name(weights[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    for (int i = 0; i < weights.size(); i++) {
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
    matmul->set_assigned_device_name(matmuls[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
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
    unpack->set_assigned_device_name(matmuls[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
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
      int num = dst_nodes.size();
      for (int i = 0; i < num; i++) {
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
  LOG(INFO) << "FuseBiasAdds";
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
    std::vector<Node*> biases;
    bool can_fuse = true;
    for (Node* out : node->out_nodes()) {
      if (out->type_string() != "BiasAdd") {
        can_fuse = false;
        break;
      }
      biasadds.push_back(out);
      Node* bias = nullptr;
      out->input_node(1, &bias);
      biases.push_back(bias);
    }
    if (!can_fuse || biases.size() < 2) continue;

    LOG(INFO) << "FuseBiasAdds: found pattern";
 
    // Add a Pack node to group biases
    string prefix = "GemmOptimizer/FuseBiasAddsAfterBatchMatMulUnpack/" +
                    std::to_string(count++);
    string pack_name = prefix + "/Pack";
    std::vector<NodeDefBuilder::NodeOut> pack_inputs;
    DataType dtype = biases[0]->output_type(0);
    for (Node* b : biases) {
      // TODO(ylxu): src_output may not be 0.
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
    pack->set_assigned_device_name(biases[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    for (int i = 0; i < biases.size(); i++) {
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
    dim->set_assigned_device_name(biases[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }

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
    expand->set_assigned_device_name(biases[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    graph->AddEdge(pack, 0, expand, 0);
    graph->AddEdge(dim, 0, expand, 1);

    // Add a new node BiasAdd
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
    biasadd->set_assigned_device_name(biasadds[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
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
    unpack->set_assigned_device_name(biasadds[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
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
      int num = out_nodes.size();
      for (int i = 0; i < num; i++) {
        graph->UpdateEdge(unpack, index, out_nodes[i], dst_inputs[i]);
      }
      graph->RemoveNode(b);
      index++;
    }
    changed = true;
  }
  return changed;
}

// [-, shape]->ReshapeOp->ShapeOp->Op to shape->Op
bool RemoveShapeAfterReshape(Graph* graph) {
  LOG(INFO) << "RemoveShapeAfterReshape";
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
    LOG(INFO) << "RemoveShapeAfterReshape: found pattern";
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
    int num = shape_dst_nodes.size();
    graph->RemoveNode(shape);
    for (int i = 0; i < num; i++) {
      graph->AddEdge(reshape_in_1, src_output,
                     shape_dst_nodes[i], shape_dst_inputs[i]);
    }
    changed = true;
  }
  return changed;
}

// Change ->Unpack->Reshape-> to ->Reshape->Unpack->
bool ReorderReshapeAndUnpack(Graph* graph) {
  static int count = 0;
  LOG(INFO) << "ReorderReshapeAndUnpack";
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
    for (int i = 1; i < reshapes.size(); i++) {
      if (reshape_in_1[i] != reshape_in_1[0]) continue;
    }
    LOG(INFO) << "ReorderReshapeAndUnpack: found pattern";
    
    // Add a new Shape to get the shape of Unpack's input
    const Edge* to_unpack;
    unpack->input_edge(0, &to_unpack);
    int src_output = to_unpack->src_output();
    Node* unpack_in = to_unpack->src();
    string prefix = "GemmOptimizer/ReorderReshapeAndUnpack/" +
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
    shape->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
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
    zero->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }

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
    one->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }

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
    slice->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
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
    zero_scalar->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }

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
    concat->set_assigned_device_name(reshape_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
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
    reshape->set_assigned_device_name(
        reshapes[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    graph->AddEdge(unpack_in, src_output, reshape, 0);
    graph->AddEdge(concat, 0, reshape, 1);
 
    graph->UpdateEdge(reshape, 0, unpack, 0);
    int index = 0;
    for (Node* r : reshapes) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : r->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      int num = dst_nodes.size();
      for (int i = 0; i < num; i++) {
        graph->UpdateEdge(unpack, index, dst_nodes[i], dst_inputs[i]);
      }
      graph->RemoveNode(r);
      index++;
    }
    changed= true;
  }
  return changed;
}

bool RemoveUnpackBeforeShape(Graph* graph) {
  static int count = 0;
  LOG(INFO) << "RemoveUnpackBeforeShape";
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
    LOG(INFO) << "RemoveUnpackBeforeShape: found pattern";

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
    shape->set_assigned_device_name(node->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
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
    one->set_assigned_device_name(node->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }

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
    minus_one->set_assigned_device_name(node->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }

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
    slice->set_assigned_device_name(node->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    graph->AddEdge(shape, 0, slice, 0);
    graph->AddEdge(one, 0, slice, 1);
    graph->AddEdge(minus_one, 0, slice, 2);

    std::vector<Node*> dst_nodes;
    std::vector<int> dst_inputs;
    for (const Edge* e : node->out_edges()) {
      dst_nodes.push_back(e->dst());
      dst_inputs.push_back(e->dst_input());
    }
    int num = dst_nodes.size();
    for (int i = 0; i < num; i++) {
      graph->UpdateEdge(slice, 0, dst_nodes[i], dst_inputs[i]);
    }
    graph->RemoveNode(node);

    changed = true;
  }
  return changed;
}

// Change ->Unpack->Transpose-> to ->Transpose->Unpack->
bool ReorderTransposeAndUnpack(Graph* graph) {
  static int count = 0;
  LOG(INFO) << "ReorderTransposeAndUnpack";
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
    for (int i = 1; i < transposes.size(); i++) {
      if (transpose_in_1[i] != transpose_in_1[0]) continue;
    }
    LOG(INFO) << "ReorderTransposeAndUnpack: found pattern";

    string prefix = "GemmOptimizer/ReorderTransposeAndUnpack/" +
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
    one->set_assigned_device_name(transpose_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
 
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
    add->set_assigned_device_name(transpose_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
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
    zero->set_assigned_device_name(transpose_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }

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
    zero_scalar->set_assigned_device_name(
        transpose_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }

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
    concat->set_assigned_device_name(
        transpose_in_1[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
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
    transpose->set_assigned_device_name(
        transposes[0]->assigned_device_name());
    if (!status.ok()) {
      LOG(ERROR) << "Adding node failed " << status;
      return false;
    }
    graph->AddEdge(unpack_in, src_output, transpose, 0);
    graph->AddEdge(concat, 0, transpose, 1);
 
    graph->UpdateEdge(transpose, 0, unpack, 0);
    int index = 0;
    for (Node* t : transposes) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : t->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      int num = dst_nodes.size();
      for (int i = 0; i < num; i++) {
        graph->UpdateEdge(unpack, index, dst_nodes[i], dst_inputs[i]);
      }
      graph->RemoveNode(t);
      index++;
    }
    changed= true;
  }
  return changed;
}

// ->Unpack--->MatMul/BatchMatMul to ->BatchMatMul->Unpack
//          |->MatMul/BatchMatMul
//          |->MatMul/BatchMatMul
//             ...
bool FuseMatMulsAfterUnpack(Graph* graph) {
  static int count = 0;
  LOG(INFO) << "FuseMatMulsAfterUnpack";
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (node->type_string() != "Unpack") continue;
    // group matmuls based on its inputs
    std::vector<Node*> matmuls;
    std::map<string, std::vector<Node*>> matmuls_m;
    bool another_input_from_unpack = true;
    bool another_input_from_consts = true;
    for (Node* out : node->out_nodes()) {
      string out_type = out->type_string();
      if (out_type == "MatMul" ||
          out_type == "BatchMatMul" ||
          out_type == "BatchMatMulV2") {
        matmuls.push_back(out);
        for (Node* n : out->in_nodes()) {
          if (n != node) {
            string type = n->type_string();
            if (type != "Unpack") {
              another_input_from_unpack = false;
            }
            if (type != "Const") {
              another_input_from_consts = false;
            }
            break;
          }
        }
      }
    }
    if (matmuls.size() < 2) continue;
    if (!another_input_from_unpack &&
        !another_input_from_consts) continue;

    for (Node* m : matmuls) {
      for (Node* n : m->in_nodes()) {
        if (n != node) {
          string key;
          if (another_input_from_unpack) {
            key = n->name();
          } else {
            // key = n->def().attr().at("value").at("tensor_shape").DebugString();
            // TODO(ylxu): use tensor_shape as key
            key = "Const";
          }
          if (matmuls_m.find(key) != matmuls_m.end()) {
            matmuls_m[key].push_back(m);
          } else {
            std::vector<Node*> ins;
            ins.push_back(m);
            matmuls_m[key] = ins;
          }
          break;
        }
      }
    }
    LOG(INFO) << "FuseMatMulsAfterUnpack: found pattern";

    // Fuse matmuls in each group.
    // For each group, do:
    // (1) add two Pack nodes to stack inputs on both sides respectively,
    // (2) add a new BatchMatMulV2 node to replace old matmuls, and
    // (3) add a Unpack node to split result.
    DataType dtype = node->output_type(0);
    std::map<string, std::vector<Node*>>::iterator iter;
    iter = matmuls_m.begin();
    while (iter != matmuls_m.end()) {
      std::vector<Node*>* matmuls_group = &(iter->second);
      if (matmuls_group->size() < 2) continue;
      std::sort(matmuls_group->begin(), matmuls_group->end(),
                [node](Node* a, Node* b){
        const Edge* in_a = nullptr;
        const Edge* in_b = nullptr;
        for (const Edge* e : a->in_edges()) {
          if (e->src() == node) {
            in_a = e;
            break;
          }
        }
        for (const Edge* e : b->in_edges()) {
          if (e->src() == node) {
            in_b = e;
            break;
          }
        }
        return in_a->src_output() < in_b->src_output(); 
      });
      std::vector<const Edge*> inputs[2];
      for (Node* n : *matmuls_group) {
        for (const Edge* e : n->in_edges()) {
          inputs[e->dst_input()].push_back(e);
        }
      }
      // Add two Pack nodes to group on two sides, respectively
      Node* packs[2];
      string pack_names[2];
      string prefix = "GemmOptimizer/FuseMatMulsAfterUnpack/" +
                      std::to_string(count++);
      pack_names[0] = prefix + "/Pack_0";
      pack_names[1] = prefix + "/Pack_1";
      Status status; 
	  for (int i = 0; i < 2; i++) {
        std::vector<NodeDefBuilder::NodeOut> pack_inputs;
        for (const Edge* e : inputs[i]) {
          string s = e->src()->name() + std::to_string(e->src_output());
          pack_inputs.emplace_back(s, e->src_output(), dtype);
        }
        NodeDefBuilder pack_builder(pack_names[i], "Pack");
        pack_builder.Input(pack_inputs);
        NodeDef pack_node;
        status =
            pack_builder
                .Attr("N", (int)inputs[i].size())
                .Attr("T", dtype)
                .Attr("axis", 0)
                .Finalize(&pack_node);
        if (!status.ok()) {
          LOG(ERROR) << "Pack node construction failed with" << status;
          return false;
        }
        pack_node.set_device(inputs[i][0]->src()->def().device());
        packs[i] = graph->AddNode(pack_node, &status);
        packs[i]->set_assigned_device_name(
            inputs[i][0]->src()->assigned_device_name());
        if (!status.ok()) {
          LOG(ERROR) << "Adding node failed " << status;
          return false;
        }
        for (int j = 0; j < inputs[i].size(); j++) {
          graph->AddEdge(inputs[i][j]->src(), inputs[i][j]->src_output(),
                         packs[i], j);
        }
      }
      // Add a new BatchMatMulV2
      string matmul_name = prefix + "/BatchMatMulV2";
      std::vector<NodeDefBuilder::NodeOut> matmul_inputs;
      matmul_inputs.emplace_back(pack_names[0], 0, dtype);
      matmul_inputs.emplace_back(pack_names[1], 0, dtype);
      NodeDefBuilder matmul_builder(matmul_name, "BatchMatMulV2");
      matmul_builder.Input(matmul_inputs[0]);
      matmul_builder.Input(matmul_inputs[1]);
      NodeDef matmul_node;
      bool transpose_a = false;
      bool transpose_b = false;
      if ((*matmuls_group)[0]->type_string() == "MatMul") {
        transpose_a = (*matmuls_group)[0]->def().attr().at("transpose_a").b();
        transpose_b = (*matmuls_group)[0]->def().attr().at("transpose_b").b();
      } else {
        transpose_a = (*matmuls_group)[0]->def().attr().at("adj_x").b();
        transpose_b = (*matmuls_group)[0]->def().attr().at("adj_y").b();
      }
      status =
          matmul_builder
              .Attr("adj_x", transpose_a)
              .Attr("adj_y", transpose_b)
              .Attr("T", dtype)
              .Finalize(&matmul_node);
      if (!status.ok()) {
        LOG(ERROR) << "BatchMatMulV2 node construction failed with" << status;
        return false;
      }
      matmul_node.set_device((*matmuls_group)[0]->def().device());
      Node* matmul = graph->AddNode(matmul_node, &status);
      matmul->set_assigned_device_name((*matmuls_group)[0]->
                                       assigned_device_name());
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      graph->AddEdge(packs[0], 0, matmul, 0);
      graph->AddEdge(packs[1], 0, matmul, 1);

      // Add an Unpack node to split result
      string unpack_name = prefix + "/Unpack" ;
      NodeDefBuilder::NodeOut unpack_input(matmul_name, 0, dtype);
      NodeDefBuilder unpack_builder(unpack_name, "Unpack");
      unpack_builder.Input(unpack_input);
      NodeDef unpack_node;
      status =
          unpack_builder
              .Attr("num", (int)(*matmuls_group).size())
              .Attr("T", dtype)
              .Attr("axis", 0)
              .Finalize(&unpack_node);
      if (!status.ok()) {
        LOG(ERROR) << "Unpack node construction failed with" << status;
        return false;
      }
      unpack_node.set_device((*matmuls_group)[0]->def().device());
      Node* unpack = graph->AddNode(unpack_node, &status);
      unpack->set_assigned_device_name((*matmuls_group)[0]->
                                       assigned_device_name());
      if (!status.ok()) {
        LOG(ERROR) << "Adding node failed " << status;
        return false;
      }
      graph->AddEdge(matmul, 0, unpack, 0);
   
      // Add edges to forward split results to nodes after original matmuls,
      // and remove original matmuls
      int index = 0;
      for (Node* m : *matmuls_group) {
        std::vector<Node*> dst_nodes;
        std::vector<int> dst_inputs;
        for (const Edge* e : m->out_edges()) {
          dst_nodes.push_back(e->dst());
          dst_inputs.push_back(e->dst_input());
        }
        int num = dst_nodes.size();
        for (int i = 0; i < num; i++) {
          graph->UpdateEdge(unpack, index, dst_nodes[i], dst_inputs[i]);
        }
        graph->RemoveNode(m);
        index++;
      }
 
      changed = true;
      iter++;
    }
  }
  
  return changed;
}

std::set<string> GetUnaryOps() {
  std::set<string> ops = {
      "Softmax",
      "Sigmoid",
      "Tanh",
      "Relu"};
  return ops;
}

bool ReorderUnaryOpAndUnpack(Graph* graph) {
  LOG(INFO) << "ReorderUnaryOpAndUnpack";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  std::set<string> unary_op_set = GetUnaryOps();
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
    LOG(INFO) << "ReorderUnaryOpAndUnpack: found pattern";
    int index = 0;
    for (Node* u : unary_ops) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : u->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      int num = dst_nodes.size();
      for (int i = 0; i < num; i++) {
        graph->UpdateEdge(unpack, index, dst_nodes[i], dst_inputs[i]);
      }
      if (index != 0) graph->RemoveNode(u);
      index++;
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

std::set<string> GetBinaryOps() {
  std::set<string> ops = {
      "Add",
      "Sub",
      "Mul"};
  return ops;
}

bool ReorderBinaryOpAndUnpack(Graph* graph) {
  LOG(INFO) << "ReorderBinaryOpAndUnpack";
  bool changed = false;

  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  std::set<string> binary_op_set = GetBinaryOps();
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
    // (1) be a binary op listed in binary_op_set,
    // (2) have the same op type, and
    // (3) have the same inputs (other than inputs from unpack).
    for (Node* out : unpack->out_nodes()) {
      if (binary_type.empty()) {
        binary_type = out->type_string();
        if (binary_op_set.find(binary_type) == binary_op_set.end()) {
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
    LOG(INFO) << "ReorderBinaryOpAndUnpack: found pattern";
    int index = 0;
    for (Node* b : binary_ops) {
      std::vector<Node*> dst_nodes;
      std::vector<int> dst_inputs;
      for (const Edge* e : b->out_edges()) {
        dst_nodes.push_back(e->dst());
        dst_inputs.push_back(e->dst_input());
      }
      int num = dst_nodes.size();
      for (int i = 0; i < num; i++) {
        graph->UpdateEdge(unpack, index, dst_nodes[i], dst_inputs[i]);
      }
      if (index != 0) graph->RemoveNode(b);
      index++;
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

void FuseGemmKernels(Graph* graph) {  
  while(1) {
    bool graph_changed =
        ReorderReshapeAndBiasAdd(graph) ||
        RemoveReshapeBetweenMatMuls(graph) ||
        FuseMatMuls(graph) ||
        FuseBiasAddsAfterBatchMatMulUnpack(graph) ||
        RemoveShapeAfterReshape(graph) ||
        ReorderReshapeAndUnpack(graph) ||
        FuseMatMulsAfterUnpack(graph) ||
        ReorderTransposeAndUnpack(graph) ||
        RemoveUnpackBeforeShape(graph) ||
        ReorderUnaryOpAndUnpack(graph) ||
        ReorderBinaryOpAndUnpack(graph);
        // RemoveUnpackPackPairs(graph);
    if (!graph_changed) break;
  }
}
}  // end namespace

Status GemmOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  static int pass = 0;
  LOG(INFO) << "GemmOptimizer";
  std::fstream f;
  f.open("before_gemm." + std::to_string(pass) + ".pbtxt", std::fstream::out);
  f << item.graph.DebugString();
  f.close();
  f.open("before_gemm." + std::to_string(pass) + ".pb", std::fstream::out | std::fstream::binary);
  f << item.graph.SerializeAsString();
  f.close();

  // convert graphdef to graph
  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());
  Graph graph(flib);
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(GraphConstructorOptions(),
                                            item.graph, &graph));
  FuseGemmKernels(&graph);

  // convert graph to graphdef
  graph.ToGraphDef(optimized_graph);

  f.open("after_gemm." + std::to_string(pass) + ".pbtxt", std::fstream::out);
  f << optimized_graph->DebugString();
  f.close();
  f.open("after_gemm." + std::to_string(pass) + ".pb", std::fstream::out | std::fstream::binary);
  f << optimized_graph->SerializeAsString();
  f.close();
  LOG(INFO) << "GemmOptimizer";
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
