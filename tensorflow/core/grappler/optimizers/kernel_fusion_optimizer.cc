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

#include "tensorflow/core/grappler/optimizers/kernel_fusion_optimizer.h"

#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"

namespace tensorflow {
namespace grappler {

Status KernelFusionOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                       GraphDef* optimized_graph) {
  GrapplerItem optimized_item(item);

  std::vector<std::shared_ptr<FusionPattern>>& pattern =
      FusionPatternRegisterer::Get()->pattern;

  bool finished;
  do {
    finished = true;
    for (auto& p : pattern) {
      bool rewrite = false;
      do {
        GraphRewriter graph(&optimized_item.graph);
        rewrite = graph.FuseRewrite(*p);
        if (rewrite) {
          finished = false;
        }
      } while (rewrite);
    }
  } while (!finished);

  optimized_graph->Swap(&optimized_item.graph);

  return Status::OK();
}

void KernelFusionOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                                     const tensorflow::grappler::GrapplerItem &item,
                                     const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
