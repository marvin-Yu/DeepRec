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

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

bool GetNodesByName(Graph* g, const std::vector<std::string>& node_names, std::vector<Node*>& nodes) {
  nodes.reserve(node_names.size());
  for (int i = 0; i < node_names.size(); ++i) {
    Node* node = graph->FindNodeByName(node_names[i]);
    if (nullptr == node) {
      VLOG(1) << "Failed to get node, node name " << input_node_names[i] << " not found in graph.";
      return false;
    }
    nodes.push_back(node);
  }
  return true;
}

bool ConstructPlaceholderByTensor(Graph* g, const InputTensor& input_tensor, Node** placeholder, const std::string& node_name) {
  int idx = input_tensor.index;
  Node* origin_node = input_tensor.node;
  if (node->def().attr().find("_output_shapes") != node->def().attr().end()) {
    auto shape = node->def().attr().at("_output_shapes").list().shape(idx);
    auto builder = NodeBuilder(node_name, "Placeholder")
                        .Attr("dtype", node->output_type(idx))
                        .Attr("shape", shape);
    builder.Finalize(g, placeholder);
    return true;
  } else {
    VLOG(1) << "Cannot find tensor attr _output_shapes";
    return false;
  }
}

/*
 bool GetNodeTypesAndNamesFromNodes(const vector<Node*> nodes, vector<DataType>& input_tensor_types, vector<std::string>& node_names) {
   for (auto node : nodes) {
     if (node->def().attr().find("T"))
   }
 }
 */

bool ExtractSubgraph(Graph* g, 
                     const std::vector<std::string>& input_node_names, 
                     const std::vector<std::string>& output_node_names) {
  // step1. find all input nodes and output node
  if (input_node_names.size() == 0) {
    VLOG(1) << "Failed to extract subgraph, subgraph input node names size is 0.";
    return false;
  }
  if (output_node_names.size() == 0) {
    VLOG(1) << "Failed to extract subgraph, subgraph output node names size is 0.";
    return false;
  } 
    
  std::vector<Node*> input_nodes, output_nodes;
  bool succ = GetNodesByName(g, input_node_names, input_nodes);
  if (!succ) {
    VLOG(1) << "Get input nodes failed, please check graph and node name config.";
    return false;
  }
  succ = GetNodesByName(g, output_node_names, output_nodes);
  if (!succ) {
    VLOG(1) << "Get output nodes failed, please check graph and node name config.";
    return false;
  }

  // step2. collect all input edges and replace input nodes with placeholders.
  std::unordered_map<InputTensor, std::vector<Node*>> input_tensor_consumer_map;
  for (Node* node : input_nodes) {
    for (int i = 0; i < node->num_inputs(); ++i) {
      InputTensor tensor;
      node->input_tensor(&tensor);
      if (input_tensor_consumer_map.find(tensor) == input_tensor_consumer_map.end()) {
        input_tensor_consumer_map.emplace(tensor, std::vector<Node*>());
      }
      input_tensor_consumer_map[tensor].push_back(node);
    }
  }
  std::unordered_map<InputTensor, Node*> input_tensor_placeholder_map;
  for (auto iter = input_tensor_consumer_map.begin(); iter != input_tensor_consumer_map.end(); ++iter) {
    InputTensor& tensor = iter->first;
    // todo: 
    Node* ph_node;
    ConstructPlaceholderByTensor(g, tensor, &ph_node, "Placeholder");
    input_tensor_placeholder_map.emplace(tensor, ph_node);
  }

  // step3. place origin input with placeholder output
  // addEdge and removeEdges

  return true;
}

bool ReplaceSubgraph(Graph* g,
                     const std::vector<std::string>& input_node_names,
                     const std::vector<std::string>& output_node_names,
                     const std::string& replace_node_name) {
  return true;
  
}

}