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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_REWRITER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_REWRITER_H_

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/optimizers/fusion_pattern.h"

namespace tensorflow {
namespace grappler {

class GraphRewriter {
 public:
  explicit GraphRewriter(GraphDef* graph);
  virtual ~GraphRewriter() = default;

  // Do fuse
  bool FuseRewrite(FusionPattern& pattern);

 protected:
  // Search pattern
  bool BFS(Node* root, FusionPattern& pattern);

  // Remove marked nodes
  void Finalize();

  // For debug
  void DumpGraph();

  std::shared_ptr<Graph> graph_;
  //Graph graph_;
  GraphDef raw_graph_def_;
  GraphDef* fused_graph_def_;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_REWRITER_H_
