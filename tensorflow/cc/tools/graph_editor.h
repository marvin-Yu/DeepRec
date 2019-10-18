//
// Created by qiaoxj on 2019-10-14.
//

#ifndef TENSORFLOW_GRAPH_EDITOR_H
#define TENSORFLOW_GRAPH_EDITOR_H

#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

Status ReplaceOp(const string& old_name, const string& new_name,
                 GraphDef* graphDef);

Status ReadGraphDef(const string& model_path, GraphDef* graphDef);

Status SaveGraphDef(const string& model_path, const GraphDef& graphDef,
                    bool as_binary = true, bool as_txt = false);

Status BypassGather(const std::unordered_set<string>& bypass_op_names,
                    const std::unordered_set<string>& bypass_blacklist,
                    GraphDef* graphDef);

Status ReplaceWithBlazeGRU(const string& custom_op,
                           const std::unordered_map<string, string>& inputs,
                           GraphDef* graphDef);

Status InsertConcatOp(const string& concat_op, const string& valid_mask,
                      const std::unordered_set<string>& valid_inputs,
                      string* temp_concat_op, GraphDef* graphDef);

Status TagCPUDevice(const std::vector<string>& inputs,
                    const std::vector<string>& outputs,
                    const std::vector<string>& split_nodes, GraphDef* graphDef);

Status OptimizeDien(GraphDef* graphDef);
}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_EDITOR_H
