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
#include "tensorflow/core/framework/tensor_shape.h"

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
  }; 

private:
  std::unordered_map<std::string, NodeDef> node_map_;
  std::unordered_map<std::string, std::vector<ConsumerInfo>> provider_consumer_info_map_;

public:
  GraphDefRewriter(const GraphDef& origin_graph_def);
  void InitNodeMap(const GraphDef& origin_graph_def);
  bool AppendNode(NodeDef new_node);
  bool AddPlaceholder(const std::string& ph_name, const DataType& dtype, const PartialTensorShape& shape);
  bool ReplaceEdgesForGivenConsumer(const std::string& origin_provider_name,
                                    const int origin_provider_slot,
                                    const std::string& replacer_provider_name,
                                    const int replacer_provider_slot,
                                    const std::unordered_set<std::string>& consumers_for_replace);
  bool GenerateGraphDefFromTop(GraphDef& output_graph_def,
                               const std::vector<std::string> top_nodes,
                               const std::unordered_set<std::string> terminal_ops);
  
};

GraphDefRewriter::GraphDefRewriter(const GraphDef& origin_graph_def) {
  InitNodeMap(origin_graph_def);
}

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_GRAPH_DEF_REWRITER_H_
