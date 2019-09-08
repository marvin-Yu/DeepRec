#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUSION_PATTERN_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUSION_PATTERN_H_

#include "tensorflow/core/grappler/optimizers/common_defines.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace grappler {

class FusionPattern;

class FusionPatternImpl {
 public:
  // Init fusion pattern impl
  virtual void Init() { }

  // If the fusion pattern is matched, return true
  virtual bool Match(std::vector<Node*>& nodes, Graph* graph) { return false; }

  // Do graph rewrite
  virtual bool GraphRewrite(std::vector<Node*>& nodes, Graph* graph) { }

  FusionPattern* pattern;
};

// The fusion pattern definition, which must be Fully Connected Graph.
//
// for example: we define Slice,Slice -> Slice
//
// REGISTER_FUSION_PATTERN(SliceSlice)
//    .Type(kInOrder)
//    .Name("ParalellMatMul")
//    .FusionOpTypeSting("Slice")
//    .BfsPatternNodes({}{})
//    .SetFusionPatternImpl(new SliceSliceFusionPatternImpl())
//
class FusionPattern {
 public:
  struct PatternNode {
    // Op type string
    std::string op_type_string;
    // Input pos of edge, which is used for pattern graph BFS
    std::vector<int> input_pos;
  };

  FusionPattern() : fusion_pattern_impl_(nullptr)  {}
  virtual ~FusionPattern() = default;

  FusionPattern& Name(const std::string& name);
  inline const std::string& name() const { return name_; }

  FusionPattern& FusionOpTypeString(const std::string& fusion_op_type_string);
  inline const std::string &fusion_op_type_string() const { return fusion_op_type_string_; }

  FusionPattern& BfsPatternNodes(const std::vector<PatternNode>& bfs_pattern_nodes);
  inline const std::vector<PatternNode>& bfs_pattern_nodes() const { return bfs_pattern_nodes_; }

  FusionPattern& SetFusionPatternImpl(FusionPatternImpl* fusion_pattern_impl);
  FusionPatternImpl* fusion_pattern_impl() { return fusion_pattern_impl_; }

  // Check valid
  bool CheckValid();

  // Initialize the pattern
  void Init();

  // Match sub graph
  bool Match(std::vector<Node*>& nodes, Graph* graph);

  // Rewrite graph
  bool GraphRewrite(std::vector<Node*>& nodes, Graph* graph);

  // Get pattern root type
  const std::string& GetFusionPatternRootType() const;

 protected:
  // The fusion pattern name
  std::string name_;
  // The bfs nodes subgraph
  std::vector<PatternNode> bfs_pattern_nodes_;
  // The fusion type
  std::string fusion_op_type_string_;
  // The fusion pattern impl
  FusionPatternImpl* fusion_pattern_impl_;
};

// Pattern register
struct FusionPatternRegisterer {
  static FusionPatternRegisterer* Get() {
    static std::shared_ptr<FusionPatternRegisterer> inst(new FusionPatternRegisterer());
    return inst.get();
  }
  FusionPattern& Register() {
    size_t idx = pattern.size();
    pattern.resize(idx + 1);
    pattern[idx].reset(new FusionPattern());
    return *(pattern[idx].get());
  }

  std::vector<std::shared_ptr<FusionPattern>> pattern;
};

#define REGISTER_FUSION_PATTERN(name)                             \
    static FusionPattern& ANONYMOUS_VARIABLE(name) =              \
      FusionPatternRegisterer::Get()->Register().Name(#name)

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUSION_PATTERN_H_