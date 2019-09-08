#include "tensorflow/core/grappler/optimizers/fusion_pattern.h"

#include "tensorflow/core/grappler/optimizers/gemm_fusion_pattern_impl.h"
#include "tensorflow/core/grappler/optimizers/parallel_gemm_fusion_pattern_impl.h"

namespace tensorflow {
namespace grappler {

REGISTER_FUSION_PATTERN(FusedGemm)
    .FusionOpTypeString("Gemm")
    .BfsPatternNodes({{"BiasAdd", {0}},
                      {"Reshape", {0}},
                      {"MatMul", {}}})
    .SetFusionPatternImpl(new GemmFusionPatternImpl());

REGISTER_FUSION_PATTERN(ParallelFusedGemm)
    .FusionOpTypeString("ParallelGemm")
    .BfsPatternNodes({{"Gemm", {}}})
    .SetFusionPatternImpl(new ParallelGemmFusionPatternImpl());

}  // namespace grappler
}  // namespace tensorflow


