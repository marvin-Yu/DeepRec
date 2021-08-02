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

#include "tensorflow/core/grappler/optimizers/check_nan_optimizer.h"

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

bool IsNodeNeedCheckNan(Node* node) {
  LOG(INFO) << "node need check nan " << node->name() ;
  // 1.Node is neither source nor sink
  if (!node->IsOp()) return false;
  // 2.Node change value
  static std::unordered_set<std::string> skip_set = {
           "CheckInputFinite",
           "Shape",
           "Reshape",
           "Transpose",
           "Identity",
           "_Send",
           "_HostSend",
           "_Recv",
           "_HostRecv",
           "_Arg",
           "_Retval"
           // other nodes going to add
};
  if (skip_set.count(node->type_string())) return false;
  // 3.Node outputs dtype are float or half
  bool out_has_float = false;
  bool out_has_check = false;
  for (const Edge* e : node->out_edges()) {
    Node* out = e->dst();
    if (out->type_string() == "CheckInputFinite") out_has_check = true;
    int port = e->src_output();
    DataType dtype = node->output_type(port);
    if (dtype == DT_FLOAT || dtype == DT_HALF) out_has_float = true;
  }
  if (out_has_check || !out_has_float) return false;
  return true;
}

bool InsertCheckNanNode(Graph* graph, Node* in_node, int in_port,
                        Node* out_node, int out_port) {

  // Prepare op name
  string prefix = "CheckNanOptimizer/" + in_node->name();
  string op_name = prefix + "_node_output_" + std::to_string(in_port)
                   + "_to_" + out_node->name() + "_node_input_" + std::to_string(out_port);

  // Build NodeDef
  std::vector<NodeDefBuilder::NodeOut> check_finite_inputs;
  Node *input_nodes[1] = {in_node};
  int input_idx[1] = {in_port};
  LOG(INFO) << "InsertCheckNanNode 111";
  if (!input_nodes[0]) return false;
  DataType dtype = input_nodes[0]->output_type(in_port);
  check_finite_inputs.emplace_back(input_nodes[0]->name(), input_idx[0], dtype);
  NodeDefBuilder check_finite_builder(op_name, "CheckInputFinite");
  check_finite_builder.Input(check_finite_inputs[0]);

  LOG(INFO) << "InsertCheckNanNode 222";
  NodeDef check_finite_def;
  Status status = check_finite_builder
      .Attr("dump_input", false)
      .Attr("T", dtype)
      .Device(in_node->def().device())
      .Finalize(&check_finite_def);
  if (!status.ok()) {
    LOG(ERROR) << "CheckInputFinite node construction failed with " << status;
    return false;
  }
  LOG(INFO) << "InsertCheckNanNode 333";

  // Insert CheckInputFinite node
  Node *check_finite_node = graph->AddNode(check_finite_def, &status);
  if (!status.ok()) {
    LOG(ERROR) << "Adding node failed " << status;
    return false;
  }
  LOG(INFO) << "InsertCheckNanNode 444";
  // Update input edge
  graph->AddEdge(input_nodes[0], input_idx[0], check_finite_node, 0);
  LOG(INFO) << "InsertCheckNanNode 555";
  // Update output edge add first, then remove
  graph->UpdateEdge(check_finite_node, 0, out_node, out_port);
  LOG(INFO) << "InsertCheckNanNode 666";
  return true;
}

void RemoveOriginEdge(Graph* graph, Node* node) {
  for (const Edge* e : node->out_edges()) {
    if (e->IsControlEdge()) continue;
    Node* out_node = e->dst();
    LOG(INFO) << "removeing edge " << node->name() << " to " << out_node->name();
    if (out_node->type_string() != "CheckInputFinite") {
      graph->RemoveEdge(e);
    }
  }
}

bool AddCheckNanNode(Graph* graph) {
  bool changed = false;
  std::vector<Node*> nodes(graph->num_nodes());
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }

  for (Node* node : nodes) {
    if (!graph->IsValidNode(node).ok()) continue;
    if (!IsNodeNeedCheckNan(node)) continue;
    LOG(INFO) << "handling node:" << node->name() << ", out_deges:" << node->out_edges().size();
    changed = false;
    std::vector<const Edge*> edges(node->out_edges().size());
    int idx = 0;
    for (const Edge* e : node->out_edges()) {
      edges[idx++] = e;
    }
    for (const Edge* e : edges) {
      if (e->IsControlEdge()) continue;
      
      LOG(INFO) << "handling node edge";
      Node* in_node = e->src();
      LOG(INFO) << "handling node edge from " << in_node->name();
      Node* out_node = e->dst();
      LOG(INFO) << "handling node edge to " << out_node->name();
      int src_out_port = e->src_output();
      LOG(INFO) << "handling node edge src output " << src_out_port;
      int dst_in_port = e->dst_input();
      LOG(INFO) << "handling node edge dst input " << dst_in_port;
      LOG(INFO) << "handling node out size: " << node->num_outputs();
      DataType dtype = node->output_type(src_out_port);
      LOG(INFO) << "handling node out dtype: " << dtype;
      LOG(INFO) << "insert node:" << node->name() << ":" << src_out_port <<  " to node:" << out_node->name() << ":" << dst_in_port;
      if (dtype == DT_FLOAT || dtype == DT_HALF) {
        if(!InsertCheckNanNode(graph, in_node, src_out_port, out_node, dst_in_port)) continue;
        else changed = true;
      }
    }
    LOG(INFO) << "handle done node:" << node->name();
    //if (changed) RemoveOriginEdge(graph, node);
    // LOG(INFO) << "remove edge done:" << node->name();
  }
  LOG(INFO) << "AddCheckNanNode all node done";
  return true;
}

void SetCPUDevice(Graph* graph) {
  std::string cpu_device = "/device:CPU:" + std::to_string(0);
  
  for (Node* node : graph->nodes()) {
    node->set_requested_device(cpu_device);
  }
}

}  // end namespace

Status CheckNanOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  VLOG(1) << "CheckNanOptimizer is on.";
  static int pass = 0;
  if (VLOG_IS_ON(2)) {
    std::fstream f;
    f.open("before_check_nan_" + std::to_string(pass) + ".pb",
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

  ReadBoolFromEnvVar("TF_CHECK_NAN_GRAPH_REWRITE_DUMP_ALL", false, &dump_all_node);
  std::string to_add;
  TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_CHECK_NAN_GRAPH_REWRITE_DUMP_LIST", "", &to_add));
  //for (auto x : str_util::Split(to_add, ",")) {
  //  list->insert(x);
  //}
  LOG(INFO) << "starting AddCheckNanNode";
  SetCPUDevice(&graph);
  if (!AddCheckNanNode(&graph)) {
    LOG(WARNING) << "Add check nan node failed ";
    *optimized_graph = item.graph;
    return Status::OK();
  }

  LOG(INFO) << "AddCheckNanNode done!!!!!!!";
  // convert graph to graphdef
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

  //if (VLOG_IS_ON(2)) {
    std::fstream f;
    f.open("after_check_nan_" + std::to_string(pass) + ".pb",
           std::fstream::out | std::fstream::binary);
    f << optimized_graph->SerializeAsString();
    f.close();
  //}
  pass++;
  return Status::OK();
}

void CheckNanOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
