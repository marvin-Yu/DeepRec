#include "tensorflow/core/grappler/optimizers/fusion_pattern.h"

namespace tensorflow {
namespace grappler {

FusionPattern& FusionPattern::Name(const std::string& name) {
  this->name_ = name;
  return *this;
}

FusionPattern& FusionPattern::FusionOpName(const std::string& fusion_op_name) {
  this->fusion_op_name_ = fusion_op_name;
  return *this;
}

FusionPattern& FusionPattern::BfsPatternNodes(const std::vector<PatternNode>& bfs_pattern_node) {
  this->bfs_pattern_nodes_ = bfs_pattern_node;
  return *this;
}

FusionPattern& FusionPattern::SetFusionPatternImpl(FusionPatternImpl* fusion_pattern_impl) {
  this->fusion_pattern_impl_ = fusion_pattern_impl;
  return *this;
}

bool FusionPattern::CheckValid() {
  // TODO check pattern valid
  return true;
}

void FusionPattern::Init() {
  if (this->fusion_pattern_impl_)
    fusion_pattern_impl_->Init();
}

bool FusionPattern::Match(std::vector<Node*>& nodes,
                          GraphRewriter* graph_rewriter) {
  if (this->fusion_pattern_impl_)
    return fusion_pattern_impl_->Match(nodes, graph_rewriter);
  return false;
}

void FusionPattern::GraphRewrite(std::vector<Node *> &nodes,
                                 GraphRewriter *graph_rewriter) {
  if (this->fusion_pattern_impl_)
    fusion_pattern_impl_->GraphRewrite(nodes, graph_rewriter);
}

const std::string& FusionPattern::GetFusionPatternRootName() const {
  if (bfs_pattern_nodes_.size() > 0) {
    return bfs_pattern_nodes_[0].op_name;
  } else {
    return kEmptyString;
  }
}

}  // namespace grappler
}  // namespace tensorflow

