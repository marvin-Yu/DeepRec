// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/23
// Description:
// An util to rewrite graph def.
// Generate a new graph, original graph def object wont be modified.

#include "tensorflow/core/framework/graph_def_rewriter.h"

#include <set>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

static const std::string PLACEHOLDER = "Placeholder";
static const std::string CUDA_GRAPH = "CudaGraph";
static const std::string IDENTITY = "Identity";
static const std::string OUTPUT_IDENTITY_SUFFIX = "/output_";
static const std::string CPU_DEVICE = "/device:CPU:0";
static const std::string GPU_DEVICE = "/device:GPU:0";

namespace tensorflow {

bool GraphDefRewriter::ExtractInputNodeAndSlot(const std::string& input, std::string& node, int& slot, bool& is_control) {
  std::vector<std::string> provider_parts = absl::StrSplit(input, ':');
  if (provider_parts.size() == 2) {
    absl::SimpleAtoi(provider_parts[1], &slot);
  } else if (provider_parts.size() != 1) {
    LOG(ERROR) << "input " << input << " is invalid.";
    return false;
  }
  if (provider_parts[0].size() > 0 && provider_parts[0][0] == '^') {
    is_control = true;
    node = provider_parts[0].substr(1);
  } else {
    is_control = false;
    node = provider_parts[0];
  }
  return true;
}

bool GraphDefRewriter::ExtractConsumerInfo(const NodeDef& node) {
  const std::string& node_name = node.name();
  for (int j = 0; j < node.input_size(); ++j) {
    int slot = 0;
    std::string input_node;
    bool is_control = false;
    if (!ExtractInputNodeAndSlot(node.input(j), input_node, slot, is_control)) {
      return false;
    }

    if (provider_consumer_info_map_.find(input_node) ==
        provider_consumer_info_map_.end()) {
      provider_consumer_info_map_.emplace(input_node,
                                          std::vector<ConsumerInfo>());
    }
    ConsumerInfo info;
    info.consumer_name_ = node_name;
    info.consumer_slot_ = j;
    info.provider_slot_ = slot;
    info.control_edge_ = is_control;
    provider_consumer_info_map_[input_node].emplace_back(info);
  }
  return true;
}

void GraphDefRewriter::InitNodeMap(const GraphDef& origin_graph_def) {
  for (int i = 0; i < origin_graph_def.node_size(); ++i) {
    const NodeDef& node = origin_graph_def.node(i);
    const std::string& node_name = node.name();
    if (node_map_.find(node_name) != node_map_.end()) {
      LOG(ERROR) << "Node in original graph whose name is " << node_name << " has been in node map, ignore it.";
    } else {
      // init consumer info
      ExtractConsumerInfo(node);
      node_map_.emplace(node_name, node);
    }
  }
}

bool GraphDefRewriter::GetNodeConsumedTensorInfo(const std::string& provider_node_name, 
                                 std::vector<int>& consumed_index,
                                 std::vector<DataType>& data_types) {
  // step1. check node existance
  auto iter = provider_consumer_info_map_.find(provider_node_name);
  if (iter == provider_consumer_info_map_.end()) {
    LOG(ERROR) << "Node with name " << provider_node_name << " not found in graph";
    return false;
  }
  
  // step2. collect all consumed index
  std::set<int> slot_set;
  auto& consumers = iter->second;
  for (ConsumerInfo& info : consumers) {
    slot_set.emplace(info.provider_slot_);
  }

  // step3. extract info needed
  NodeDef& provider = node_map_[provider_node_name];
  const OpDef* provider_op_def;
  TF_CHECK_OK(global_op_registry_->LookUpOpDef(string(provider.op()), &provider_op_def));
  for (auto it = slot_set.begin(); it != slot_set.end(); ++it) {
    DataType type;
    consumed_index.emplace_back(*it);
    TF_CHECK_OK(OutputTypeForNode(provider, *provider_op_def, *it, &type));
    data_types.emplace_back(type);
  }
  return true;
}

bool GraphDefRewriter::AddNode(NodeDef&& node) {
  const std::string& node_name = node.name();
  ExtractConsumerInfo(node);
  node_map_.emplace(node_name, node);
  return true;
}

bool GraphDefRewriter::AddPlaceholder(const std::string& ph_name, const DataType& dtype, const std::vector<int64>& shape) {
  // step0. check if name is duplicated
  if (node_map_.find(ph_name) != node_map_.end()) {
    LOG(ERROR) << "Node name " << ph_name << " for new placeholder is found in graph.";
    return false;
  }
  
  // step1. build an node def
  NodeDef ph_node;
  NodeDefBuilder builder(ph_name, PLACEHOLDER);
  PartialTensorShape s(gtl::ArraySlice<tensorflow::int64>(shape.data(), shape.size()));
  TF_CHECK_OK(builder.Attr("dtype", dtype)
         .Attr("shape", s)
         .Attr("_output_shape", s)
         .Finalize(&ph_node));
  
  // step2. put new ph_node into node map
  node_map_.emplace(ph_name, ph_node);
  return true;
}

bool GraphDefRewriter::AddIdentityNode(const std::string& origin_node_name, 
                                       const int origin_slot, 
                                       std::string& identity_node_name) {
  // step1. check node existance
  auto iter = node_map_.find(origin_node_name);
  if (iter == node_map_.end()) {
    LOG(ERROR) << "Node with name " << origin_node_name << " not found in graph";
    return false;  
  }
  NodeDef& node = iter->second;

  // step2. gen identity node
  const OpDef* origin_op_def;
  DataType identity_type;
  TF_CHECK_OK(global_op_registry_->LookUpOpDef(string(node.op()), &origin_op_def));
  TF_CHECK_OK(OutputTypeForNode(node, *origin_op_def, origin_slot, &identity_type));
  identity_node_name = strings::StrCat(origin_node_name, OUTPUT_IDENTITY_SUFFIX, origin_slot);

  NodeDef id_node;
  NodeDefBuilder builder(identity_node_name, IDENTITY);
  TF_CHECK_OK(builder.Input(origin_node_name, origin_slot, identity_type).Finalize(&id_node));
   
  // step3. put ot node map
  node_map_.emplace(id_node.name(), id_node);
  if (provider_consumer_info_map_.find(origin_node_name) ==
      provider_consumer_info_map_.end()) {
    provider_consumer_info_map_.emplace(origin_node_name,
                                        std::vector<ConsumerInfo>());
  }
  ConsumerInfo info;
  info.consumer_name_ = id_node.name();
  info.consumer_slot_ = 0;
  info.provider_slot_ = origin_slot;
  provider_consumer_info_map_[origin_node_name].emplace_back(info);
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
  std::vector<int> remove_idx;
  for (int i = 0; i < consumer_iter->second.size(); ++i) {
    ConsumerInfo info = consumer_iter->second[i];
    // if matched
    if (info.provider_slot_ == origin_provider_slot) {
      // step3. modify consumer node def
      if (node_map_.find(info.consumer_name_) == node_map_.end()) {
        LOG(ERROR) << "Consumer node " << info.consumer_name_ << " not found in graph";
        continue;
      }
      std::string replaced_input_name = strings::StrCat(replacer_provider_name, ":", replacer_provider_slot);
      // for control edge
      if (info.control_edge_) {
        replaced_input_name = "^" + replaced_input_name;
      }
      node_map_[info.consumer_name_].set_input(info.consumer_slot_, replaced_input_name);

      // step4. add new edge to map info
      info.provider_slot_ = replacer_provider_slot;
      if (provider_consumer_info_map_.find(replacer_provider_name) == provider_consumer_info_map_.end()) {
        provider_consumer_info_map_.emplace(replacer_provider_name, std::vector<ConsumerInfo>());
      }
      provider_consumer_info_map_[replacer_provider_name].emplace_back(info);

      // step5. remove old edge
      remove_idx.push_back(i);
    }
  }
  // todo: optimize, switch to tail and remove once
  for (int i = remove_idx.size() - 1; i >= 0; --i) {
    consumer_iter->second.erase(consumer_iter->second.begin() + remove_idx[i]);
  }
  return true;                      
}

bool GraphDefRewriter::GenerateGraphDefFromTop(GraphDef& output_graph_def,
                               const std::vector<std::string> top_nodes,
                               std::vector<std::string>& input_names) {
  // BFS gen graph from top
  // step1. clear output_graph_def nodes
  std::unordered_set<std::string> visited_nodes;
  output_graph_def.clear_node();
  for (int i = 0; i < top_nodes.size(); ++i) {
    if (node_map_.find(top_nodes[i]) == node_map_.end()) {
      LOG(ERROR) << "Top node "<< top_nodes[i] << " not found in graph";
      return false;
    } else {
      NodeDef* new_node = output_graph_def.add_node();
      *new_node = node_map_[top_nodes[i]];
      visited_nodes.emplace(top_nodes[i]);
    }
  }

  // step2. bfs graph, add input node into new graph
  int travel_idx = 0;
  while (travel_idx < output_graph_def.node_size()) {
    const NodeDef& curr_node = output_graph_def.node(travel_idx);
    ++travel_idx;

    if (curr_node.op() == PLACEHOLDER) {
      // LOG(INFO) << "[Jieluo] add placeholder " << curr_node.name();
      input_names.push_back(curr_node.name());
    }

    // check node type
    for (int i = 0; i < curr_node.input_size(); ++i) {
      int separator_pos = curr_node.input(i).find(':');
      if (separator_pos == std::string::npos) {
        separator_pos = curr_node.input(i).size();
      }
      // for control edge
      int start_pos = 0;
      if (separator_pos > 0 && curr_node.input(i)[0] == '^') {
        start_pos = 1;
      }
      std::string input_node_name = curr_node.input(i).substr(start_pos, separator_pos);
      if (node_map_.find(input_node_name) == node_map_.end()) {
        LOG(ERROR) << "Node " << input_node_name << " not found in graph.";
        return false;
      } else if (visited_nodes.find(input_node_name) == visited_nodes.end()) {
        NodeDef* new_node = output_graph_def.add_node();
        *new_node = node_map_[input_node_name];
        visited_nodes.emplace(input_node_name);
      }
    }
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
    std::vector<int64> shape;
    for (int j = 0; j < subgraph_desc.input_tensors(i).shape_size(); ++j) {
      shape.push_back(subgraph_desc.input_tensors(i).shape(j));
    }
    std::string ph_name = subgraph_desc.input_tensors(i).ph_name();
    if (rewriter.AddPlaceholder(ph_name, 
                                subgraph_desc.input_tensors(i).type(),
                                shape)) {
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
  std::vector<int> consumed_idx;
  std::vector<DataType> types;
  for (int i = 0; i < subgraph_desc.output_node_names_size(); ++i) {
    consumed_idx.clear();
    const std::string& output_name = subgraph_desc.output_node_names(i);
    bool succ = rewriter.GetNodeConsumedTensorInfo(output_name, consumed_idx, types);
    for (int j = 0; j < consumed_idx.size(); ++j) {
      std::string identity_name;
      if (!rewriter.AddIdentityNode(output_name, consumed_idx[j], identity_name)) {
        LOG(ERROR) << "Add Identity node failed, identity input " << output_name << " slot " << consumed_idx[j];
        return false;
      }
      subgraph_final_outputs.emplace_back(identity_name);
    }
  }

  subgraph_final_inputs.clear();
  bool succ = rewriter.GenerateGraphDefFromTop(output_graph, subgraph_final_outputs, subgraph_final_inputs);
  if (!succ) {
    LOG(ERROR) << "Generate subgraph failed.";
  }

  // step3. set default device gpu
  for (int i = 0; i < output_graph.node_size(); ++i) {
    auto node = output_graph.mutable_node(i);
    if (node->device().empty()) {
      node->set_device(GPU_DEVICE);
    }
  }
  return true;
}

bool SubgraphGenerator::ReplaceSubgraph(const GraphDef& origin_graph, 
                                        GraphDef& output_graph, 
                                        const std::string& group_name,
                                        const std::vector<const SubgraphDescription*>& subgraph_descriptions,
                                        const std::vector<std::string>& output_nodes) {
  CopyCommonField(origin_graph, output_graph);

  GraphDefRewriter rewriter(origin_graph);

  // step1. fetch attr for cuda graph 
  for (auto subgraph_desc : subgraph_descriptions) {
    std::vector<std::string> feed_names;
    std::vector<std::string> fetch_names;
    std::vector<DataType> T1;
    std::vector<DataType> T2;

    std::vector<int> fetch_index;
    std::vector<std::string> fetch_nodes;
    std::vector<NodeDefBuilder::NodeOut> node_out;
    node_out.reserve(subgraph_desc->input_tensors_size());
    NodeDefBuilder builder(subgraph_desc->subgraph_name(), CUDA_GRAPH);
    for (int i = 0; i < subgraph_desc->input_tensors_size(); ++i) {
      feed_names.emplace_back(subgraph_desc->input_tensors(i).ph_name());
      T1.emplace_back(subgraph_desc->input_tensors(i).type());
      node_out.emplace_back(NodeDefBuilder::NodeOut(subgraph_desc->input_tensors(i).tensor_provider_name(),
                    subgraph_desc->input_tensors(i).tensor_provider_slot(),
                    subgraph_desc->input_tensors(i).type()));      
    }
    for (int i = 0; i < subgraph_desc->output_node_names_size(); ++i) {
      rewriter.GetNodeConsumedTensorInfo(subgraph_desc->output_node_names(i),
                                         fetch_index,
                                         T2);
      for (int j = fetch_names.size(); j < fetch_index.size(); ++j) {
        fetch_names.emplace_back(strings::StrCat(subgraph_desc->output_node_names(i), ":", fetch_index[j]));
        fetch_nodes.emplace_back(subgraph_desc->output_node_names(i));
      }
    }

    std::vector<int> buckets;
    buckets.reserve(subgraph_desc->cuda_graph_batch_sizes_size());
    for (int j = 0; j < subgraph_desc->cuda_graph_batch_sizes_size(); ++j) {
      buckets.emplace_back(subgraph_desc->cuda_graph_batch_sizes(j));
    }

    // step2. build cudagraph node 
    NodeDef cudagraph_node;
    TF_CHECK_OK(builder.Input(node_out)
          .Attr("feed_names", feed_names)
          .Attr("fetch_names", fetch_names)
          .Attr("T1", T1)
          .Attr("T2", T2)
          .Attr("buckets", buckets)
          .Attr("graph_name", subgraph_desc->subgraph_name())
          .Device(GPU_DEVICE)
          .Finalize(&cudagraph_node));
    rewriter.AddNode(std::move(cudagraph_node));

    // step3. reroute cuda graph op output edge
    const std::unordered_set<std::string> empty;
    for (int i = 0; i < fetch_names.size(); ++i) {
      rewriter.ReplaceEdgesForGivenConsumer(fetch_nodes[i], fetch_index[i], subgraph_desc->subgraph_name(), i, empty);
    }
  }
  // step4. 
  std::vector<std::string> input_nodes;
  if (!rewriter.GenerateGraphDefFromTop(output_graph, output_nodes, input_nodes)) {
    LOG(ERROR) << "Generate subgraph failed.";
  }
  return true;
}

} // tensorflow
