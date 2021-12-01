// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/18
// Description:
// An optimizer to extract subgraph.
// According to input node names and output node name to define subgraph's scope.

#ifndef TENSORFLOW_CORE_GRAPH_SUBGRAPH_EXTRACTOR_H_
#define TENSORFLOW_CORE_GRAPH_SUBGRAPH_EXTRACTOR_H_

#include <string>
#include <vector>

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Return true if and only if 'g' is mutated.
extern bool ExtractSubgraph(Graph* g, 
                            const std::vector<std::string>& input_node_names, const std::vector<std::string>& output_node_names);

extern bool ReplaceSubgraph(Graph* g,
                            const std::vector<std::string>& input_node_names,
                            const std::vector<std::string>& output_node_names,
                            const std::string& replace_node_name);

}


#endif  // TENSORFLOW_CORE_GRAPH_SUBGRAPH_EXTRACTOR_H_