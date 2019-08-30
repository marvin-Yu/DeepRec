#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUSION_PATTERN_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUSION_PATTERN_H_

#include "tensorflow/core/grappler/optimizers/common_defines.h"
#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"

namespace tensorflow {
namespace grappler {

class FusionPatternImpl {
 public:
  // Init fusion pattern impl
  virtual void Init() { }

  // If the fusion pattern is matched, return true
  virtual bool Match(std::vector<Node*>& nodes, GraphRewriter* graph_rewriter) { return false; }

  // Do graph rewrite
  virtual void GraphRewrite(std::vector<Node*>& nodes, GraphRewriter* graph_rewriter) { }
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
  struct PatternNode {
    // Op name
    std::string op_name;
    // Output pos is used for pattern graph BFS
    std::vector<int> output_pos;
  };

  virtual ~FusionPattern();

  FusionPattern& Name(const std::string& name);
  inline const std::string& name() const { return name_; }

  FusionPattern& FusionOpName(const std::string& fusion_op_name);
  inline const std::string &fusion_op_name() const { return fusion_op_name_; }

  FusionPattern& BfsPatternNodes(const std::vector<PatternNode>& bfs_pattern_nodes);
  inline const std::vector<PatternNode>& bfs_pattern_nodes() const { return bfs_pattern_nodes_; }

  FusionPattern& SetFusionPatternImpl(FusionPatternImpl* fusion_pattern_impl);
  FusionPatternImpl* fusion_pattern_impl() { return fusion_pattern_impl_; }

  // Initialize the pattern
  void Init();

  // Match sub graph
  bool Match(std::vector<Node*>& nodes, GraphRewriter* graph_rewriter);

  // Rewrite graph
  void GraphRewrite(std::vector<Node*>& nodes, GraphRewriter* graph_rewriter);

  // Check valid
  bool CheckValid();

  // Get pattern root name
  const std::string& GetFusionPatternRootName() const;

 protected:
  // The fusion pattern name
  std::string name_;
  // The bfs nodes subgraph
  std::vector<PatternNode> bfs_pattern_nodes_;
  // The fusion op name
  std::string fusion_op_name_;
  // The fusion pattern impl
  FusionPatternImpl* fusion_pattern_impl_ = nullptr;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUSION_PATTERN_H_