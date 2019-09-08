#include "tensorflow/core/grappler/optimizers/gemm_fusion_pattern_impl.h"

namespace tensorflow {
namespace grappler {

void GemmFusionPatternImpl::Init() {
  LOG(INFO) << "GemmFusion begin";
}

bool GemmFusionPatternImpl::Match(std::vector<Node *> &nodes, Graph *graph) {
  // TODO add strict verification here
  return true;
}

bool GemmFusionPatternImpl::GraphRewrite(std::vector<Node *> &nodes, Graph *graph) {
  // add Gemm node
  NodeDef gemm_node_def;
  gemm_node_def.set_op(this->pattern->fusion_op_type_string());
  // TODO set attribute & op name
  // temp code need to remove
  /* copy matmul attrs
  Node* matmul_op = nodes[2];
  for (auto iter = matmul_op->attrs().begin(); iter != matmul_op->attrs().end(); ++iter) {
    LOG(INFO) << "iter->first=" << iter->first;
    (*gemm_node_def.mutable_attr())[iter->first] = iter->second;
  }
  */

  Status status;
  Node* gemm_node = graph->AddNode(gemm_node_def, &status);
  if (status != Status::OK()) {
    LOG(ERROR) << "GemmFusion add gemm node failed! " << status.ToString();
    return false;
  }

  // change output edge
  Node* bias_node = nodes[0];
  for (const Edge* out_edge : bias_node->out_edges()) {
    Node* dst_node = out_edge->dst();
    int dst_input_idx = out_edge->dst_input();
    const Edge* new_edge = graph->AddEdge(gemm_node, 0, dst_node, dst_input_idx);
    if (nullptr == new_edge) {
      LOG(ERROR) << "GemmFusion add new output edge failed!";
      return false;
    }
  }

  // add matmul input edge
  Node* matmul_node = nodes[2];
  for (int i = 0; i < matmul_node->num_inputs(); ++i) {
    const Edge* in_edge = nullptr;
    status = matmul_node->input_edge(i, &in_edge);
    if (status != Status::OK() || nullptr == in_edge) {
      LOG(ERROR) << "GemmFusion get old matmul input edge failed!";
      return false;
    }
    Node* src_node = in_edge->src();
    int src_output_idx = in_edge->src_output();
    const Edge* new_edge = graph->AddEdge(src_node, src_output_idx, gemm_node, i);
    if (nullptr == new_edge) {
      LOG(ERROR) << "GemmFusion add new matmul input edge failed!";
      return false;
    }
  }

  // add bias input edge
  const Edge* in_edge = nullptr;
  status = bias_node->input_edge(1, &in_edge);
  if (status != Status::OK() || nullptr == in_edge) {
    LOG(ERROR) << "GemmFusion get old bias input edge failed!";
    return false;
  }
  Node* src_node = in_edge->src();
  int src_output_idx = in_edge->src_output();
  const Edge* new_edge = graph->AddEdge(src_node, src_output_idx, gemm_node, 1);
  if (nullptr == new_edge) {
    LOG(ERROR) << "GemmFusion add new bias input edge failed!";
    return false;
  }

  // remove original pattern nodes
  for (size_t i = 0; i < nodes.size(); ++i) {
    graph->RemoveNode(nodes[i]);
  }

  return true;
}

}  // namespace grappler
}  // namespace tensorflows