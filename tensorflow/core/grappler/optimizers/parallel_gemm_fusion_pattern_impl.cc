#include "tensorflow/core/grappler/optimizers/parallel_gemm_fusion_pattern_impl.h"

namespace tensorflow {
namespace grappler {

void ParallelGemmFusionPatternImpl::Init() {
  candidate_gemm_nodes_.clear();
}

bool ParallelGemmFusionPatternImpl::Match(std::vector<Node *> &nodes, Graph *graph) {
  Node* gemm_node = nodes[0];
  candidate_gemm_nodes_.push_back(gemm_node);

  // find parallel gemm node with same B & bias shape
  const Edge* in_edge = nullptr;
  Status status = gemm_node->input_edge(0, &in_edge);
  if (status != Status::OK()) {
    LOG(ERROR) << "ParallelGemmFusion get gemm input edge failed! " << status.ToString();
    return false;
  }
  Node* parent_node = in_edge->src();
  for (Node* node : parent_node->out_nodes()) {
    if (node == gemm_node) continue;
    if (node->type_string() == "Gemm") {
      // TODO check shape

      candidate_gemm_nodes_.push_back(node);
    }
  }
  if (candidate_gemm_nodes_.size() == 1)
    return false;

  return true;
}

bool ParallelGemmFusionPatternImpl::GraphRewrite(std::vector<Node *> &nodes, Graph *graph) {
  NodeDef parallel_gemm_node_def;
  parallel_gemm_node_def.set_op(this->pattern->fusion_op_type_string());

  // TODO make parallel gemm input B & bias constant
  NodeDef b_param, bias_param;
  b_param.set_op("Constant");
  bias_param.set_op("Constant");

  // TODO add parallel gemm node

  // TODO remove pattern node

  return false;
}

}  // namespace grappler
}  // namespace tensorflow
