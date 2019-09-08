#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_PARALLEL_GEMM_FUSION_PATTERN_IMPL_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_PARALLEL_GEMM_FUSION_PATTERN_IMPL_H_

#include "tensorflow/core/grappler/optimizers/fusion_pattern.h"

namespace tensorflow {
namespace grappler {

class ParallelGemmFusionPatternImpl : public FusionPatternImpl {
 public:
  virtual void Init() override;

  virtual bool Match(std::vector<Node*>& nodes, Graph* graph) override;

  virtual bool GraphRewrite(std::vector<Node*>& nodes, Graph* graph) override;
};


}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_PARALLEL_GEMM_FUSION_PATTERN_IMPL_H_