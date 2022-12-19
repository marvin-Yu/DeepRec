// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/23
// Description:
// An util to rewrite graph def.
// Generate a new graph, original graph def object wont be modified.

#ifndef TENSORFLOW_CORE_FRAMEWORK_GRAPH_DEF_REWRITER_H_
#define TENSORFLOW_CORE_FRAMEWORK_GRAPH_DEF_REWRITER_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// Forward declare proto so that it's symbols can be removed from .so exports
class GraphDef;
class NodeDef;

class GraphDefRewriter {
private:
  struct ConsumerInfo {
    std::string consumer_name_;
    int consumer_slot_;
    int provider_slot_;
    bool control_edge_ = false;
  }; 

private:
  std::unordered_map<std::string, NodeDef> node_map_;
  std::unordered_map<std::string, std::vector<ConsumerInfo>> provider_consumer_info_map_;
  OpRegistryInterface* global_op_registry_;

public:
  GraphDefRewriter(const GraphDef& origin_graph_def);
  void InitNodeMap(const GraphDef& origin_graph_def);
  bool AddNode(NodeDef&& new_node);
  bool AddPlaceholder(const std::string& ph_name, const DataType& dtype, const std::vector<int64>& shape);
  bool AddIdentityNode(const std::string& origin_node_name, const int origin_slot, std::string& identity_node_name);
  bool GetNodeConsumedTensorInfo(const std::string& provider_node_name, 
                                 std::vector<int>& consumed_index,
                                 std::vector<DataType>& data_types);
  bool ReplaceEdgesForGivenConsumer(const std::string& origin_provider_name,
                                    const int origin_provider_slot,
                                    const std::string& replacer_provider_name,
                                    const int replacer_provider_slot,
                                    const std::unordered_set<std::string>& consumers_for_replace);
  // during bfs to get all reachable nodes, recording all ph node name. 
  bool GenerateGraphDefFromTop(GraphDef& output_graph_def,
                               const std::vector<std::string> top_nodes,
                               std::vector<std::string>& input_names);

private:
  bool ExtractConsumerInfo(const NodeDef& node); 
  bool ExtractInputNodeAndSlot(const std::string& input, std::string& node, int& slot, bool& is_control);
  
};

inline GraphDefRewriter::GraphDefRewriter(const GraphDef& origin_graph_def) {
  global_op_registry_ = OpRegistry::Global();
  InitNodeMap(origin_graph_def);
}

class SubgraphGenerator {
public:
  // cut subgraph from origin graph, subgraph's boundary info is recorded in subgraph_desc
  // subgraph final input and output nodes' name is collected as result
  // which are used during capturing cuda graph. 
  bool static GenerateSubgraph(const GraphDef& origin_graph, 
                               GraphDef& output_graph, 
                               const SubgraphDescription& subgraph_desc,
                               std::vector<std::string>& subgraph_final_inputs,
                               std::vector<std::string>& subgraph_final_outputs);
  bool static ReplaceSubgraph(const GraphDef&  origin_graph, 
                              GraphDef& output_graph, 
                              const std::string& group_name,
                              const std::vector<const SubgraphDescription*>& subgraph_descriptions,
                              const std::vector<std::string>& output_nodes);
private:
  void static CopyCommonField(const GraphDef& origin_graph, GraphDef& output_graph);
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_GRAPH_DEF_REWRITER_H_
