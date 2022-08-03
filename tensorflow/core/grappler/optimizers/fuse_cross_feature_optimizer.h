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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUSE_CROSS_FEATURE_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUSE_CROSS_FEATURE_OPTIMIZER_H_

#include "tensorflow/tools/graph_transforms/transform_utils.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"

using namespace tensorflow::graph_transforms;

namespace tensorflow {
namespace grappler {

namespace {

static const OpTypePattern cross_feature_pattern = 
      {"ConcatV2|Concat",                // concat, base node
        { // input
          {"Sum",
            { // input.input
              {"Tanh",
                { // input.input.input
                  {"BatchMatMul|BatchMatMulV2",
                    { // input.input.input.input
                      {"StridedSlice", // same 1, need
                        { // input.input.input.input.input
                          {"GatherV2", // need
                            { // input.input.input.input.input.input
                              {"*"}, // need
                              {"Placeholder|Tile"}, // need
                              {"Const"} // need
                            }
                          },
                          {"Const"}, // need
                          {"Const"}, // need
                          {"Const"}, // need
                        }
                      },
                      {"Reshape", // need
                        { // input.input.input.input.input
                          {"StridedSlice"}, // ad stridedslice, need
                          {"Const"},        // shape const, [batch, 5, 4] -> [batch, 1, 5, 4]
                        }
                      },
                    }
                  },
                }
              },
              {"Const"},  // sum input const, not need
            }
          },
          {"Sum",
            {
              {"Tanh",
                {
                  {"BatchMatMul|BatchMatMulV2",
                    {
                      {"Mul|Square",
                        {
                          {"StridedSlice"}, // same 2
                          {"StridedSlice"}, // same 3
                        }
                      },
                      {"Reshape"}, // need
                    }
                  },
                }
              },
              {"Const"}, // not need
            }
          },
          {"Const"}, // not need
        }
      };
}  // end namespace

class FuseCrossFeatureOptimizer : public GraphOptimizer {
 public:
  FuseCrossFeatureOptimizer() {}
  ~FuseCrossFeatureOptimizer() override {}

  string name() const override { return "fuse_cross_feature"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUSE_CROSS_FEATURE_OPTIMIZER_H_
