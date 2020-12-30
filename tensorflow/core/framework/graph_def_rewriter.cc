// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/23
// Description:
// An util to rewrite graph def.
// Generate a new graph, original graph def object wont be modified.

#include "tensorflow/core/framework/graph_def_rewriter.h"

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

static const std::string PLACEHOLDER = "Placeholder";

namespace tensorflow {

void GraphDefRewriter::InitNodeMap(const GraphDef& origin_graph_def) {
  for (int i = 0; i < origin_graph_def.node_size(); ++i) {
    const NodeDef& node = origin_graph_def.node(i);
    const std::string& node_name = node.name();
    if (node_map_.find(node_name) != node_map_.end()) {
      LOG(ERROR) << "Node in original graph whose name is " << node_name << " has been in node map, ignore it.";
    } else {
      // init consumer info
      for (int j = 0; j < node.input_size(); ++j) {
        std::vector<std::string> provider_parts = absl::StrSplit(node.input(j), ':');
        int slot = 0;
        if (provider_parts.size() == 2) {
          absl::SimpleAtoi(provider_parts[1], &slot);
        } else if (provider_parts.size() == 1) {
          // do nothing
        } else {
          LOG(ERROR) << "Node " << node_name << "'s input " << node.input(j) << " is invalid.";
        }
        
        if (provider_consumer_info_map_.find(provider_parts[0]) == provider_consumer_info_map_.end()) {
          provider_consumer_info_map_.emplace(provider_parts[0], std::vector<ConsumerInfo>());
        }
        ConsumerInfo info;
        info.consumer_name_ = node_name;
        info.consumer_slot_ = j;
        info.provider_slot_ = slot;
        LOG(INFO) << "[jieluo] Provider name " << provider_parts[0] << " index " << info.provider_slot_ << " consumer name " << node_name << " consumer index " << info.consumer_slot_ << " origin input str " << node.input(j);
        provider_consumer_info_map_[provider_parts[0]].emplace_back(info);
      }
      node_map_.emplace(node_name, node);
    }
  }
}

bool GraphDefRewriter::AddPlaceholder(const std::string& ph_name, const DataType& dtype, const PartialTensorShape& shape) {
  // step0. check if name is duplicated
  if (node_map_.find(ph_name) != node_map_.end()) {
    LOG(ERROR) << "Node name " << ph_name << " for new placeholder is found in graph.";
    return false;
  }
  
  // step1. build an node def
  NodeDef ph_node;
  NodeDefBuilder builder(ph_name, PLACEHOLDER);
  TF_CHECK_OK(builder.Attr("dtype", dtype)
         .Attr("shape", shape)
         .Attr("_output_shape", shape)
         .Finalize(&ph_node));
  
  // step2. put new ph_node into node map
  node_map_.emplace(ph_name, ph_node);
  return true;
}

bool GraphDefRewriter::ReplaceEdgesForGivenConsumer(const std::string& origin_provider_name,
                                    const int origin_provider_slot,
                                    const std::string& replacer_provider_name,
                                    const int replacer_provider_slot,
                                    const std::unordered_set<std::string>& consumers_for_replace) {
  // step1. Check if provider exist
  if (node_map_.find(origin_provider_name) == node_map_.end()) {
    LOG(ERROR) << "Node " << origin_provider_name << " not found in graph.";
    return false; 
  }
  if (node_map_.find(replacer_provider_name) == node_map_.end()) {
    LOG(ERROR) << "Node " << replacer_provider_name << " not found in graph.";
    return false; 
  }

  // step2. Get all origin consumer
  auto consumer_iter = provider_consumer_info_map_.find(origin_provider_name);
  if (consumer_iter == provider_consumer_info_map_.end()) {

    return false;
  }
  for (int i = 0; i < consumer_iter->second.size(); ++i) {
    ConsumerInfo info = consumer_iter->second[i];
    // if matched
    if (info.provider_slot_ == origin_provider_slot) {
      // step3. modify consumer node def
      if (node_map_.find(info.consumer_name_) == node_map_.end()) {
      
        continue;
      }
      std::string replaced_input_name = strings::StrCat(replacer_provider_name, ":", replacer_provider_slot);
      node_map_[info.consumer_name_].set_input(info.consumer_slot_, replaced_input_name);

      // step4. modify map info
      info.provider_slot_ = replacer_provider_slot;
      if (provider_consumer_info_map_.find(replacer_provider_name) == provider_consumer_info_map_.end()) {
        provider_consumer_info_map_.emplace(replacer_provider_name, std::vector<ConsumerInfo>());
      }
      provider_consumer_info_map_[replacer_provider_name].emplace_back(info);
    }
  }
  return true;                      
}

bool GraphDefRewriter::GenerateGraphDefFromTop(GraphDef& output_graph_def,
                               const std::vector<std::string> top_nodes,
                               std::vector<std::string>& input_names) {
  // BFS gen graph from top
  // step1. clear output_graph_def nodes
  output_graph_def.clear_node();
  for (int i = 0; i < top_nodes.size(); ++i) {
    if (node_map_.find(top_nodes[i]) == node_map_.end()) {
      LOG(ERROR) << "Top node "<< top_nodes[i] << " not found in graph";
      return false;
    } else {
      NodeDef* new_node = output_graph_def.add_node();
      *new_node = node_map_[top_nodes[i]];
    }
  }

  // step2. bfs graph, add input node into new graph
  int travel_idx = 0;
  while (travel_idx < output_graph_def.node_size()) {
    const NodeDef& curr_node = output_graph_def.node(travel_idx);
    LOG(INFO) << "[jieluo] visit node " << curr_node.name() << " index is " << travel_idx;
    // check node type
    if (curr_node.op() == PLACEHOLDER) {
      input_names.push_back(curr_node.name());
    } 
    for (int i = 0; i < curr_node.input_size(); ++i) {
      int separator_pos = curr_node.input(i).find(':');
      if (separator_pos == std::string::npos) {
        separator_pos = curr_node.input(i).size();
      }
      std::string input_node_name = curr_node.input(i).substr(0, separator_pos);
      if (node_map_.find(input_node_name) == node_map_.end()) {
        LOG(ERROR) << "Node " << input_node_name << " not found in graph.";
      } else {
        NodeDef* new_node = output_graph_def.add_node();
        *new_node = node_map_[input_node_name];
      }
    }
    ++travel_idx;
  }

  return true;
}

void SubgraphGenerator::CopyCommonField(const GraphDef& origin_graph, GraphDef& output_graph) {
  output_graph.mutable_versions()->CopyFrom(origin_graph.versions());
  output_graph.set_version(origin_graph.version());
  output_graph.mutable_library()->CopyFrom(origin_graph.library());
  return;
}

bool SubgraphGenerator::GenerateSubgraph(const GraphDef& origin_graph, 
                                         GraphDef& output_graph, 
                                         const SubgraphDescription& subgraph_desc,
                                         std::vector<std::string>& subgraph_final_inputs,
                                         std::vector<std::string>& subgraph_final_outputs) {
  // step0. copy other fields in graph
  CopyCommonField(origin_graph, output_graph);
  
  GraphDefRewriter rewriter(origin_graph);
  std::unordered_set<std::string> empty_set;
  // step1. gen new placeholder and replace edge
  for (int i = 0; i < subgraph_desc.input_tensors_size(); ++i) {
    PartialTensorShape shape;
    for (int j = 0; j < subgraph_desc.input_tensors(i).shape_size(); ++j) {
      shape.AddDim(subgraph_desc.input_tensors(i).shape(j));
    }
    std::string ph_name = subgraph_desc.input_tensors(i).ph_name();
    if (rewriter.AddPlaceholder(ph_name, 
                                subgraph_desc.input_tensors(i).type(), shape)) {
      bool succ = rewriter.ReplaceEdgesForGivenConsumer(subgraph_desc.input_tensors(i).tensor_provider_name(),
                                                        subgraph_desc.input_tensors(i).tensor_provider_slot(),
                                                        ph_name, 0, empty_set);
      if (!succ) {
        LOG(ERROR) << "Replace edge in origin graph with new placehold failed.";
        return false;
      }
    } else {
      LOG(ERROR) << "Generate subgraph new placeholder failed.";
      return false;
    }
  }

  // step2. output
  subgraph_final_outputs.clear();
  subgraph_final_outputs.reserve(subgraph_desc.output_node_names_size());
  for (int i = 0; i < subgraph_desc.output_node_names_size(); ++i) {
    subgraph_final_outputs.emplace_back(subgraph_desc.output_node_names(i));
  }
  subgraph_final_inputs.clear();
  bool succ = rewriter.GenerateGraphDefFromTop(output_graph, subgraph_final_outputs, subgraph_final_inputs);
  if (!succ) {
    LOG(ERROR) << "Generate subgraph failed.";
  }
  return true;
}

bool SubgraphGenerator::ReplaceSubgraph(const GraphDef& origin_graph, 
                                        GraphDef& output_graph, 
                                        const SubgraphDescription& subgraph_desc) {
  return true;
}

} // tensorflow
