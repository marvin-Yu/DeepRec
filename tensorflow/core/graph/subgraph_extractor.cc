// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/18
// Description:
// Implement of subgraph extractor optimizer

#include "tensorflow/core/graph/subgraph_extractor.h"

#include <unordered_set>
#include <unordered_map>

#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

bool GetNodesByName(Graph* g, const std::vector<std::string>& node_names, std::vector<Node*>& nodes) {
  nodes.reserve(node_names.size());
  for (int i = 0; i < node_names.size(); ++i) {
    Node* node = g->FindNodeByName(node_names[i]);
    if (nullptr == node) {
      LOG(INFO) << "Failed to get node, node name " << node_names[i] << " not found in graph.";
      return false;
    }
    nodes.push_back(node);
  }
  return true;
}

bool ConstructPlaceholderByTensor(Graph* g, const OutputTensor& input_tensor, Node** placeholder, const std::string& node_name) {
  int idx = input_tensor.index;
  Node* node = input_tensor.node;
  if (node->def().attr().find("_output_shapes") != node->def().attr().end()) {
    auto shape = node->def().attr().at("_output_shapes").list().shape(idx);
    auto builder = NodeBuilder(node_name, "Placeholder")
                        .Attr("dtype", node->output_type(idx))
                        .Attr("shape", shape);
    builder.Finalize(g, placeholder);
    return true;
  } else {
    LOG(INFO) << "Cannot find tensor attr _output_shapes";
    return false;
  }
}

bool GetInputTypesFromNodes(const std::vector<Node*> nodes, std::vector<DataType>& tensor_types) {
  for (auto node : nodes) {
    for (int i = 0; i < node->num_inputs(); ++i) {
      tensor_types.push_back(node->input_type(i));
    }
  } 
  return true;
}

bool GetOutputTypesFromNodes(const std::vector<Node*> nodes, std::vector<DataType>& tensor_types) {
  for (auto node : nodes) {
    for (int i = 0; i < node->num_outputs(); ++i) {
      tensor_types.push_back(node->output_type(i));
    }
  }
  return true;
}

bool ExtractSubgraph(Graph* g, 
                     const std::vector<std::string>& input_node_names, 
                     const std::vector<std::string>& output_node_names) {
  // step1. find all input nodes and output node
  LOG(INFO) << "Extract subgraph step1";
  if (input_node_names.size() == 0) {
    LOG(INFO) << "Failed to extract subgraph, subgraph input node names size is 0.";
    return false;
  }
  if (output_node_names.size() == 0) {
    LOG(INFO) << "Failed to extract subgraph, subgraph output node names size is 0.";
    return false;
  } 
    
  std::vector<Node*> input_nodes, output_nodes;
  bool succ = GetNodesByName(g, input_node_names, input_nodes);
  if (!succ) {
    LOG(INFO) << "Get input nodes failed, please check graph and node name config.";
    return false;
  }
  succ = GetNodesByName(g, output_node_names, output_nodes);
  if (!succ) {
    LOG(INFO) << "Get output nodes failed, please check graph and node name config.";
    return false;
  }

  // step2. collect all input edges and replace input nodes with placeholders.
  LOG(INFO) << "Extract subgraph step2";
  for (Node* node : input_nodes) {
    for (int i = 0; i < node->num_inputs(); ++i) {
      OutputTensor tensor;
      const Edge* input_edge;
      node->input_tensor(i, &tensor);
      node->input_edge(i, &input_edge);

      Node* ph_node = nullptr;
      if (!ConstructPlaceholderByTensor(g, tensor, &ph_node, "Placeholder")) {
        return false;
      }
      g->AddEdge(ph_node, 0, node, i);
      g->RemoveEdge(input_edge);
    }
  }

  // step3. place origin input with placeholder output
  // addEdge and removeEdges
  LOG(INFO) << "Extract subgraph step3";
  std::unordered_set<const Node*> output_nodes_set(output_nodes.begin(), output_nodes.end());
  PruneForReverseReachability(g, output_nodes_set);

  return true;
}

bool ReplaceSubgraph(Graph* g,
                     const std::vector<std::string>& input_node_names,
                     const std::vector<std::string>& output_node_names,
                     const std::string& replace_node_name) {
  // step1. find all input nodes and output node
  if (input_node_names.size() == 0) {
    LOG(INFO) << "Failed to extract subgraph, subgraph input node names size is 0.";
    return false;
  }
  if (output_node_names.size() == 0) {
    LOG(INFO) << "Failed to extract subgraph, subgraph output node names size is 0.";
    return false;
  } 
    
  std::vector<Node*> input_nodes, output_nodes;
  bool succ = GetNodesByName(g, input_node_names, input_nodes);
  if (!succ) {
    LOG(INFO) << "Get input nodes failed, please check graph and node name config.";
    return false;
  }
  succ = GetNodesByName(g, output_node_names, output_nodes);
  if (!succ) {
    LOG(INFO) << "Get output nodes failed, please check graph and node name config.";
    return false;
  }

  // step2. make a cudagraphop replace subgraph
  std::vector<DataType> input_types, output_types;
  GetInputTypesFromNodes(input_nodes, input_types);
  GetOutputTypesFromNodes(output_nodes, output_types);
  auto builder = NodeBuilder(replace_node_name, "CudaGraphOp")
                      .Attr("T1", input_types)
                      .Attr("T2", output_types);
                  //    .Attr("num_input", input_types.size())
                  //    .Attr("num_output", output_types.size());
  Node** cuda_graph_node;
  builder.Finalize(g, cuda_graph_node);

  int input_idx = 0;
  for (auto node : input_nodes) {
    for (int i = 0; i < node->num_inputs(); ++i) {
      const Edge* input_edge;
      OutputTensor tensor;
      node->input_tensor(i, &tensor);
      node->input_edge(i, &input_edge);
      g->AddEdge(tensor.node, tensor.index, *cuda_graph_node, input_idx);
      g->RemoveEdge(input_edge);
      ++input_idx;
    }
  }

  int output_idx = 0;
  for (auto node : output_nodes) {
    for (int i = 0; i < node->num_outputs(); ++i) {
      const Edge* output_edge;
      InputTensor tensor;
     // node->output_tensor(i, &tensor);
     // node->output_edge(i, &input_edge);
    }
  }

  // step3. place origin input with placeholder output
  // todo: get original output node name
  std::unordered_set<const Node*> output_nodes_set(output_nodes.begin(), output_nodes.end());
  PruneForReverseReachability(g, output_nodes_set);
  
  return true;
  
}

}
