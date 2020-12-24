// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/23
// Description:
// An util to rewrite graph def.
// Generate a new graph, original graph def object wont be modified.

#include "tensorflow/core/framework/graph_def_rewriter.h"

#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/logging.h"

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
        if (input_parts.size() == 2) {
          absl::SimpleAtoi(input_parts[1], &slot);
        } else if (input_parts.size() == 1) {
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
        provider_consumer_info_map_[provider_parts[0]].emplace_back(info);
      }
      node_map_.emplace(node_name, node);
    }
  }
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
  if (consumer_iter == provider_consumer_info_map_.end) {

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
                               const std::unordered_map<std::string> terminal_ops) {
  // BFS gen graph from top
  // step1. clear output_graph_def nodes
  output_graph_def.clear_node();
  for (int i = 0; i < top_nodes.size(); ++i) {
    if (node_map_.find(top_nodes[i]) == node_map_.end()) {
      LOG(ERROR) << "Top node not found in graph";
    } else {
      NodeDef* new_node = output_graph_def.add_node();
      *new_node = node_map_[top_nodes[i]];
    }
  }

  // step2. bfs graph, add input node into new graph
  int travel_idx = 0;
  while (traval_idx < output_graph_def.node_size()) {
    const NodeDef& curr_node = output_graph_def.node(travel_idx);
    for (int i = 0; i < curr_node.input_size(); ++i) {
      int separator_pos = curr_node.input(i).find(':');
      if (separator_pos == std::string::npos) {
        separator_pos = curr_node.input(i).size();
      }
      std::string input_node_name = curr_node.input(i).substr(0, separator_pos);
      if (node_map_.find(input_node_name) == node_map_.end()) {
        LOG(ERROR) << "Node not found in graph.";
      } else {
        NodeDef* new_node = output_graph_def.add_node();
        *new_node = node_map_[input_node_name];
      }
    }
  }

  return true;
}

bool GraphDefRewriter::ReplaceProviderByAPlaceholder(const std::string& origin_provider_name,
                                     const int origin_provider_slot,
                                     const std::unordered_set<std::string>& consumers_for_replace) {
  // step1. get provider tensor shape and type
  if (provider_consumer_info_map_.find(origin_provider_name) == provider_consumer_info_map_.end()) {
    LOG(WARN) << "No node's input is " << origin_provider_name;
    return false;
  }
  const ConsumerInfo& info = provider_consumer_info_map_[origin_provider_name][0];
  if (node_map_.find(info.consumer_name_) == node_map_.end()) {
    LOG(ERROR) << "Graph error";
    return false;
  }

  return true;
}

} // tensorflow