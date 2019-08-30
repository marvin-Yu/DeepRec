#include "tensorflow/core/grappler/optimizers/kernel_fusion/graph_rewriter.h"

#include <vector>

namespace tensorflow {
namespace grappler {

class FusionPattern;

class FusionPatternImpl {
 public:
  // Init fusion pattern impl
  virtual void Init() { }

  // If the fusion pattern is matched, return true
  virtual bool Match(std::vector<Node*>& nodes, GraphRewriter* graph_rewriter) { return false; }

  // Do graph rewrite
  virtual void GraphRewrite(const std::vector<Node*>& nodes, GraphRewriter* graph_rewriter) { };

  FusionPattern* pattern;
};

// The fusion pattern definition, which must be Fully Connected Graph.
//
// for example: we define Slice,Slice -> Slice
//
// REGISTER_FUSION_PATTERN(SliceSlice)
//    .Type(kInOrder)
//    .Name("ParalellMatMul")
//    .OpNameSet({ "MatMul" })
//    .FusionOpName("Slice")
//    .SetFusionPatternImpl(new SliceSliceFusionPatternImpl())
//    .Init();
//
class FusionPattern {
 public:
  virtual ~FusionPattern();

  FusionPattern &Name(const std::string &name);

  inline const std::string &name() const { return name_; }

  FusionPattern &FusionOpName(std::string fusion_op_name);

  inline const std::string &fusion_op_name() const { return fusion_op_name_; }

  FusionPattern &OpNameSet(const std::vector<std::string> &op_name_set);

  inline const std::vector<std::string> &op_name_set() const { return op_name_set_; }

 protected:
  // The fusion pattern name
  std::string name_;
  // The original op name set
  std::vector<std::string> op_name_set_;
  // The fusion op name
  std::string fusion_op_name_;
};

}  // namespace grappler
}  // namespace tensorflow