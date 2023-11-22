#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CUSTOM_MATMUL_BN_FUSION_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CUSTOM_MATMUL_BN_FUSION_H_

#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Custom MatMul BN Fusion optimization for a graph.

// Target original subgraph
//             ┌─────────────┐
//             │             │
//             │    Matmul   │
//             │             │
//             └──────┬──────┘
//        ┌───────────┴────────────┐
//        │                 ┌──────▼──────┐
//        │                 │             │
//        │                 │   Reshape   │
//        │                 │             │
//        │                 └──────┬──────┘
//        │             ┌──────────▼────────┐
// ┌──────▼──────┐      │                   │
// │             │      │                   │
// │    Shape    │      │ FusedBatchNormV3  │
// │             │      │                   │
// └──────┬──────┘      │                   │
//        │             └──────────┬────────┘
//        └───────────┬────────────┘
//             ┌──────▼──────┐
//             │             │
//             │   Reshape   │
//             │             │
//             └──────┬──────┘
//             ┌──────▼──────┐
//             │             │
//             │  LeakyRelu  │
//             │             │
//             └─────────────┘

// Actual target subgraph (origin optimized by other optimizer)
//             ┌─────────────┐
//             │             │
//             │    Matmul   │
//             │             │
//             └──────┬──────┘
//                    │
//        ┌───────────┴────────────┐
//        │                 ┌──────▼──────┐
//        │                 │             │
//        │                 │   Reshape   │
//        │                 │             │
//        │                 └──────┬──────┘
//        │                 ┌──────▼──────┐
//        │                 │             │
// ┌──────▼──────┐          │     Mul     │
// │             │          │             │
// │    Shape    │          └──────┬──────┘
// │             │                 │
// └──────┬──────┘          ┌──────▼──────┐
//        │                 │             │
//        │                 │     Add     │
//        │                 │             │
//        │                 └──────┬──────┘
//        └───────────┬────────────┘
//             ┌──────▼──────┐
//             │             │
//             │   Reshape   │
//             │             │
//             └──────┬──────┘
//             ┌──────▼──────┐
//             │             │
//             │  LeakyRelu  │
//             │             │
//             └─────────────┘

class CustomMatMulBNFusion : public GraphOptimizer {
 public:
  CustomMatMulBNFusion() = default;
  explicit CustomMatMulBNFusion(RewriterConfig::Toggle opt_level) {}
  ~CustomMatMulBNFusion() override{};

  string name() const override { return "matmul_custom_fusion"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CUSTOM_MATMUL_BN_FUSION_H_
