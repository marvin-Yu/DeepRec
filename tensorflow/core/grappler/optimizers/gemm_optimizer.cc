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
      LOG(INFO) << "matmul_1: " << node->DebugString();
      LOG(INFO) << "current: " << current->DebugString();
      while (op == "Reshape" || (ops_with_unchanged_shape.find(op) !=
                                 ops_with_unchanged_shape.end())) {
        if (op != "Reshape") non_reshape_before_matmul_1 = current;
        Node* temp = nullptr;
        current->input_node(0, &temp);
        current = temp;
        op = current->type_string();
        LOG(INFO) << "current: " << current->DebugString();
      }
      if (op == "MatMul") {
        Node* matmul_1 = node;
        Node* matmul_0 = current;
        LOG(INFO) << "matmul_0: " << matmul_0->DebugString();
        // TODO(ylxu): check shape compatibility using matmul weights
        if (non_reshape_before_matmul_1 == nullptr) {
          non_reshape_before_matmul_1 = matmul_0;
        }
        LOG(INFO) << "non_reshape_before_matmul_1: " << non_reshape_before_matmul_1->DebugString();
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

// Change MatMul->Reshape->BiasAdd->others to MatMul->BiasAdd->Reshape->others
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
      std::vector<Node*> bias_out_nodes;
      std::vector<int> bias_out_indices;
      for (const Edge* e : bias->out_edges()) {
        LOG(INFO) << "e->DebugString(): " << e->DebugString();
        bias_out_nodes.push_back(e->dst());
        bias_out_indices.push_back(e->dst_input());
      }
      int num = bias->out_edges().size();
      for (int i = 0; i < num; i++) {
        graph->UpdateEdge(reshape, 0, bias_out_nodes[i], bias_out_indices[i]);
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
    std::vector<Node*> weights;
    for (Node* n : in->out_nodes()) {
      if (n->type_string() == "MatMul") {
        matmuls.push_back(n);
        LOG(INFO) << "matmul->DebugString(): " << n->DebugString();
        Node* w = nullptr;
        n->input_node(1, &w);
        weights.push_back(w);
        LOG(INFO) << "w->DebugString(): " << w->DebugString();
      }
    }
    if (matmuls.size() < 2) continue;
    // TODO(ylxu): check the compatibility of shapes (weights)
    // and attrs (transpose_a and transpose_b).

    LOG(INFO) << "FuseMatMuls: found pattern";
    // Add a Pack node to group weights
    string pack_name;
    std::vector<NodeDefBuilder::NodeOut> pack_inputs;
    DataType dtype = weights[0]->output_type(0);
    for (Node* w : weights) {
      pack_name += w->name();
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

    // Add a new node BatchMatMulV2
    string matmul_name;
    for (Node* m : matmuls) {
      matmul_name += m->name();
    }
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
      LOG(ERROR) << "BatchMatMulV2 node construction failed with" << status;
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
    string unpack_name = "Unpack/" + matmul_name;
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
      std::vector<Node*> out_nodes;
      std::vector<int> out_ports;
      for (const Edge* e : m->out_edges()) {
        out_nodes.push_back(e->dst());
        out_ports.push_back(e->dst_input());
      }
      int num = out_nodes.size();
      for (int i = 0; i < num; i++) {
        graph->UpdateEdge(unpack, index, out_nodes[i], out_ports[i]);
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
bool FuseBiasAddsAfterUnpack(Graph* graph) {
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
    if (matmul->type_string() != "BatchMatMulV2") continue;
    std::vector<Node*> biasadds;
    std::vector<Node*> biases;
    for (Node* out : node->out_nodes()) {
      if (out->type_string() == "BiasAdd") {
        biasadds.push_back(out);
        Node* bias = nullptr;
        out->input_node(1, &bias);
        biases.push_back(bias);
      }
    }
    if (biasadds.size() < 2) continue;

    LOG(INFO) << "FuseBiasAdds: found pattern";
 
    // Add a Pack node to group biases
    string pack_name;
    std::vector<NodeDefBuilder::NodeOut> pack_inputs;
    DataType dtype = biases[0]->output_type(0);
    for (Node* b : biases) {
      pack_name += b->name();
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
    string dim_name = "ExpandDims/" + pack_name + "/axis";
    NodeDefBuilder dim_builder(dim_name, "Const");
    NodeDef dim_node;
    Tensor axis((int)1);
    status =
        dim_builder
            .Attr("dtype", axis.dtype())
            .Attr("value", axis)
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

    string expand_name = "ExpandDims/" + pack_name;
    std::vector<NodeDefBuilder::NodeOut> expand_inputs;
    expand_inputs.emplace_back(pack_name, 0, dtype);
    expand_inputs.emplace_back(dim_name, 0, axis.dtype());
    NodeDefBuilder expand_builder(expand_name, "ExpandDims");
    expand_builder.Input(expand_inputs[0]);
    expand_builder.Input(expand_inputs[1]);
    NodeDef expand_node;
    status =
        expand_builder
            .Attr("T", dtype)
            .Attr("Tdim", axis.dtype())
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
    string biasadd_name;
    for (Node* b : biasadds) {
      biasadd_name += b->name();
    }
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
    string unpack_name = "Unpack/" + biasadd_name;
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
      std::vector<int> out_ports;
      for (const Edge* e : b->out_edges()) {
        out_nodes.push_back(e->dst());
        out_ports.push_back(e->dst_input());
      }
      int num = out_nodes.size();
      for (int i = 0; i < num; i++) {
        graph->UpdateEdge(unpack, index, out_nodes[i], out_ports[i]);
      }
      graph->RemoveNode(b);
      index++;
    }
    changed = true;
  }
  return changed;
}

// ->Unpack--->MatMul  to ->BatchMatMul->Unpack
//          |->MatMul
//          |->MatMul
//             ...
bool FuseMatMulsAfterUnpack(Graph* graph) {
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
    std::vector<Node*> matmuls;
    std::vector<Node*> weights;
    for (Node* out : node->out_nodes()) {
      if (out->type_string() == "MatMul") {
        matmuls.push_back(out);
        Node* w = nullptr;
        out->input_node(1, &w);
        weights.push_back(w);
      }
    }
    if (matmuls.size() < 2) continue;

    LOG(INFO) << "FuseMatMulsAfterUnpack: found pattern";
 
    // Add a Pack node to group weights
    string pack_name;
    std::vector<NodeDefBuilder::NodeOut> pack_inputs;
    DataType dtype = weights[0]->output_type(0);
    for (Node* w : weights) {
      pack_name += w->name();
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

    // Add a new node BatchMatMulV2
    string matmul_name;
    for (Node* m : matmuls) {
      matmul_name += m->name();
    }
    std::vector<NodeDefBuilder::NodeOut> matmul_inputs;
    const Edge* e;
    node->input_edge(0, &e);
    Node* in = e->src();
    int src_output = e->src_output();
    matmul_inputs.emplace_back(in->name(), src_output, dtype);
    matmul_inputs.emplace_back(pack_name, 0, dtype);
    NodeDefBuilder matmul_builder(matmul_name, "BatchMatMulV2");
    matmul_builder.Input(matmul_inputs[0]);
    matmul_builder.Input(matmul_inputs[1]);
    NodeDef matmul_node;
    status =
        matmul_builder
            .Attr("T", dtype)
            .Finalize(&matmul_node);
    if (!status.ok()) {
      LOG(ERROR) << "BiasAdd node construction failed with" << status;
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
    string unpack_name = "Unpack/" + matmul_name;
    NodeDefBuilder::NodeOut unpack_input(matmul_name, 0, dtype);
    NodeDefBuilder unpack_builder(unpack_name, "Unpack");
    unpack_builder.Input(unpack_input);
    NodeDef unpack_node;
    status =
        unpack_builder
            .Attr("num", (int)matmuls.size())
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
      std::vector<Node*> out_nodes;
      std::vector<int> out_ports;
      for (const Edge* e : m->out_edges()) {
        out_nodes.push_back(e->dst());
        out_ports.push_back(e->dst_input());
      }
      int num = out_nodes.size();
      for (int i = 0; i < num; i++) {
        graph->UpdateEdge(unpack, index, out_nodes[i], out_ports[i]);
      }
      graph->RemoveNode(m);
      index++;
    }
    changed = true;
  }
  return changed;
}

void FuseGemmKernels(Graph* graph) {  
  while(ReorderReshapeAndBiasAdd(graph) ||
        RemoveReshapeBetweenMatMuls(graph) ||
        FuseMatMuls(graph) ||
        FuseBiasAddsAfterUnpack(graph) ||
        FuseMatMulsAfterUnpack(graph)) {
    // static int round = 0;
    // GraphDef temp_graph;
    // graph->ToGraphDef(&temp_graph);
    // std::fstream f;
    // string filename = "gemm_round" + std::to_string(round++);
    // f.open(filename + ".pbtxt", std::fstream::out);
    // f << temp_graph.DebugString();
    // f.close();
    // f.open(filename + ".pb", std::fstream::out | std::fstream::binary);
    // f << temp_graph.SerializeAsString();
    // f.close();
  }
}
}  // end namespace

Status GemmOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  LOG(INFO) << "GemmOptimizer";
  std::fstream f;
  f.open("before_gemm.pbtxt", std::fstream::out);
  f << item.graph.DebugString();
  f.close();
  f.open("before_gemm.pb", std::fstream::out | std::fstream::binary);
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

  f.open("after_gemm.pbtxt", std::fstream::out);
  f << optimized_graph->DebugString();
  f.close();
  f.open("after_gemm.pb", std::fstream::out | std::fstream::binary);
  f << optimized_graph->SerializeAsString();
  f.close();
  LOG(INFO) << "GemmOptimizer";
  return Status::OK();
}

void GemmOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
