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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_KERNEL_FUSION_CORE_GRAPH_REWRITER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_KERNEL_FUSION_CORE_GRAPH_REWRITER_H_

#include "tensorflow/core/grappler/optimizers/kernel_fusion/fusion_pattern.h"

namespace tensorflow {
namespace grappler {

struct Node {
  Node() : remove(false) { }

  struct Input {
    // parent node id
    int parent_node_id;
    // position of parent output
    int output_pos;
  };
  struct Output {
    // child node id
    int child_node_id;
    // position of output
    int output_pos;
  };

  const std::string& name() const { return node_def->name(); }
  const std::string& op_name() const { return node_def->op(); }
  std::string output_name(int pos) const {
    if (pos == 0) return node_def->name();
    return node_def->name() + ":" + std::to_string(pos);
  }
  std::vector<Input> inputs;
  std::vector<Output> outputs;
  NodeDef* node_def;
  bool remove;
};

class GraphRewriter {
 public:
  explicit GraphRewriter(GraphDef* graph);
  virtual ~GraphRewriter() = default;

  /*
  bool FuseRewrite(GraphRusePattern& pattern);

  bool ModifyOp(GraphOpModifier& modifier);*/

  std::string GetParentName(const std::string& name, int* out_pos = nullptr) const;
  const Node* GetNodeByName(const std::string& name) const;
  const Node* GetNodeByInputName(const std::string& name, int* out_pos = nullptr) const {
    std::string node_name = GetParentName(name, out_pos);
    return GetNodeByName(node_name);
  }

  // NodeDef* NewFusedNodeDef(bool new_name = true);
  std::string GetOutputName(const std::string& op_name) const;

  template <typename T>
  void set_attr(const std::string& key, const T value) {
    attr_[key] = value;
  }

  template <typename T>
  T get_attr(const std::string& key) {
    auto iter = attrs_find(key);
    // TODO
  }

 protected:
  // for debug
  void DumpGraph();

  void InitNodes(const std::unordered_map<std::string, int>& idx_map);

  bool VerifyFusionPattern(const FusionPattern& pattern);

  bool RewriteGraph(FusionPattern& pattern);

  bool BFS(int root_id, FusionPattern& pattern);

  std::unordered_map<std::string, int> idx_map_;
  std::vector<Node> nodes_;
  GraphDef raw_graph_def_;
  GraphDef* fused_graph_def_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_KERNEL_FUSION_CORE_GRAPH_REWRITER_H_
