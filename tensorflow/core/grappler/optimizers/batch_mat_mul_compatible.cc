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

#include "tensorflow/core/grappler/optimizers/batch_mat_mul_compatible.h"

#include <fstream>
#include <queue>
#include <map>

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"


namespace tensorflow {
namespace grappler {

namespace {

Status ReplaceNewNode(Graph *graph, Node *new_node, Node *old_node) {
  std::vector<Node*> src_nodes;
  std::vector<int> src_outputs;
  std::vector<Node*> dst_nodes;
  std::vector<int> dst_inputs;
  for (const Edge* e : old_node->in_edges()) {
    src_nodes.push_back(e->src());
    src_outputs.push_back(e->src_output());
    dst_nodes.push_back(e->dst());
    dst_inputs.push_back(e->dst_input());
  }
  for (unsigned int i = 0; i < src_nodes.size(); i++) {
    graph->AddEdge(src_nodes[i], src_outputs[i], new_node, dst_inputs[i]);
  }
  src_nodes.clear();
  src_outputs.clear();
  dst_nodes.clear();
  dst_inputs.clear();
  for (const Edge* e : old_node->out_edges()) {
    src_nodes.push_back(e->src());
    src_outputs.push_back(e->src_output());
    dst_nodes.push_back(e->dst());
    dst_inputs.push_back(e->dst_input());
  }
  for (unsigned int i = 0; i < dst_nodes.size(); i++) {
    TF_RETURN_IF_ERROR(
        graph->UpdateEdge(new_node, src_outputs[i], dst_nodes[i], dst_inputs[i]));
  }
  graph->RemoveNode(old_node);
  VLOG(1) << "replace to BatchMatMulV2 node:" << new_node->DebugString();
  return Status::OK();
}

bool OptimizeBatchMatMul(Graph *graph) {
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  Status status;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  for (Node* node : nodes) {
    if (node->type_string() != "BatchMatMul") continue;
    Node *batch_matmul = node;
    VLOG(1) << "find BatchMatMul node:" << batch_matmul->DebugString();
    NodeDef v2_node;
    v2_node.CopyFrom(batch_matmul->def());
    v2_node.set_op("BatchMatMulV2");
    Node* v2 = graph->AddNode(v2_node, &status);
    if (!status.ok()) {
      LOG(ERROR) << "Adding BatchMatMulV2 node failed " << status;
      return false;
    }
    v2->set_assigned_device_name(batch_matmul->assigned_device_name());
    ReplaceNewNode(graph, v2, batch_matmul);
  }
  return true;
}

}  // end namespace

Status BatchMatMulCompatibleOptimizer::Optimize(Cluster* cluster,
                                                const GrapplerItem& item,
                                                GraphDef* optimized_graph) {
  bool replace = false;
  ReadBoolFromEnvVar("TF_ENABLE_NATIVE_DELIVERY_OPTIMIZE", false, &replace);
  if (!replace) {
    *optimized_graph = item.graph;
    return Status::OK();
  }
  VLOG(0) << "BatchMatMulCompatibleOptimizer is on.";

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

  if(!OptimizeBatchMatMul(&graph)) {
    LOG(WARNING) << "OptimizeBatchMatMul failed!";
    *optimized_graph = item.graph;
    return Status::OK();
  }

  // convert graph to graphdef
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

  return Status::OK();
}

void BatchMatMulCompatibleOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
