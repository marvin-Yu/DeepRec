#include "tensorflow/core/grappler/optimizers/custom_matmul_bn_fusion.h"

#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/evaluation_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"

namespace tensorflow {
namespace grappler {
namespace {

struct Pattern {
  int matmul_id = -1;
  int shape_id = -1;
  int reshape_1_id = -1;
  int mul_id = -1;
  int add_id = -1;
  int reshape_2_id = -1;
  int leakyrelu_id = -1;
};

inline bool HasControlFaninOrFanout(const utils::MutableNodeView& node_view) {
  return node_view.NumControllingFanins() > 0 ||
         node_view.NumControlledFanouts() > 0;
}

int GetOutputNodeIndex(const utils::MutableGraphView& graph_view,
                       int node_index, int output_index) {
  const auto* node_view = graph_view.GetNode(node_index);
  const auto& fanouts = node_view->GetRegularFanout(0);
  int size = fanouts.size();

  if (output_index < 0 || output_index >= size) {
    return -1;
  }
  return fanouts[output_index].node_index();
}

void ListOutputNodeIndex(const utils::MutableGraphView& graph_view,
                         int node_index, std::vector<int>& output) {
  const auto* node_view = graph_view.GetNode(node_index);
  const auto& fanouts = node_view->GetRegularFanout(0);
  int size = fanouts.size();

  output.clear();
  for (int i = 0; i < size; ++i) {
    output.emplace_back(fanouts[i].node_index());
  }

  return;
}

bool IsNodeViewMatch(const utils::MutableGraphView& graph_view, int node_index,
                     int num_outputs, bool (*func)(const NodeDef& node)) {
  if (node_index < 0) return false;
  const auto* node_view = graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  if (!func(*node_def)) return false;
  if (HasControlFaninOrFanout(*node_view)) return false;

  if (num_outputs >= 0) {
    if (node_view->NumRegularFanouts() != num_outputs) return false;
    const auto& fanouts = node_view->GetRegularFanout(0);
    if (fanouts.size() != num_outputs) return false;
  }

  return true;
}

bool IsNodeViewMatchMatMul(const utils::MutableGraphView& graph_view,
                           int node_index, Pattern* pattern) {
  if (!IsNodeViewMatch(graph_view, node_index, 2, IsMatMul)) return false;

  std::vector<int> output;
  ListOutputNodeIndex(graph_view, node_index, output);
  if (output.size() != 2) return false;
  for (int j = 0; j < 2; j++) {
    if (IsNodeViewMatch(graph_view, output.at(j), 1, IsShape) &&
        pattern->shape_id == -1) {
      pattern->shape_id = output.at(j);
    } else if (IsNodeViewMatch(graph_view, output.at(j), 1, IsReshape) &&
               pattern->reshape_1_id == -1) {
      pattern->reshape_1_id = output.at(j);
    } else {
      return false;
    }
  }

  return true;
}

bool FindDPattern(const utils::MutableGraphView& graph_view, int node_index,
                  Pattern* matched) {
  Pattern pattern;
  //   std::vector<int> outs;
  auto* node_view = graph_view.GetNode(node_index);
  if (HasControlFaninOrFanout(*node_view)) return false;

  pattern.matmul_id = node_index;
  // Match matmul, shape and reshape
  if (!IsNodeViewMatchMatMul(graph_view, node_index, &pattern)) return false;

  pattern.mul_id = GetOutputNodeIndex(graph_view, pattern.reshape_1_id, 0);
  if (!IsNodeViewMatch(graph_view, pattern.mul_id, 1, IsMul)) return false;

  pattern.add_id = GetOutputNodeIndex(graph_view, pattern.mul_id, 0);
  if (!IsNodeViewMatch(graph_view, pattern.add_id, 1, IsAdd)) return false;

  pattern.reshape_2_id = GetOutputNodeIndex(graph_view, pattern.shape_id, 0);
  if (pattern.reshape_2_id != GetOutputNodeIndex(graph_view, pattern.add_id, 0))
    return false;

  pattern.leakyrelu_id =
      GetOutputNodeIndex(graph_view, pattern.reshape_2_id, 0);
  if (!IsNodeViewMatch(graph_view, pattern.leakyrelu_id, -1, IsLeakyRelu))
    return false;

  *matched = pattern;
  return true;
}
}  // namespace

Status CustomMatMulBNFusion::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* output) {
  Status status;
  *output = item.graph;
  utils::MutableGraphView graph_view(output, &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  const int num_nodes = item.graph.node_size();
  // invalidated_nodes - nodes that have been changed into a fused op
  // nodes_to_delete -  nodes that were fused into a fused op and are not needed
  // anymore
  std::vector<bool> invalidated_nodes(num_nodes);
  std::vector<bool> nodes_to_delete(num_nodes);
  const GraphDef* graph = graph_view.graph();

  VLOG(3) << "Before Dice fusion rewrites: " << graph->DebugString();

  for (int i = 0; i < num_nodes; ++i) {
    if (invalidated_nodes[i] || nodes_to_delete[i]) continue;

    Pattern base;
    if (FindDPattern(graph_view, i, &base)) {
      const auto& fused_node = graph->node(i);
      VLOG(2) << "Optimizing fused custom matmul node "
              << SummarizeNodeDef(fused_node);

      std::vector<int> dead_nodes = {base.shape_id, base.reshape_1_id,
                                     base.reshape_2_id};
      for (auto& id : dead_nodes) nodes_to_delete[id] = true;
      invalidated_nodes[base.matmul_id] = true;
      invalidated_nodes[base.mul_id] = true;
      invalidated_nodes[base.add_id] = true;
      invalidated_nodes[base.leakyrelu_id] = true;

      utils::Mutation* mutation = graph_view.GetMutationBuilder();

      auto add_tensor_id = ParseTensorName(graph_view.GetNode(base.reshape_2_id)->node()->input(0));
      auto* leakyrule_view = graph_view.GetNode(base.leakyrelu_id);
      mutation->AddOrUpdateRegularFanin(leakyrule_view,0, add_tensor_id);

      auto matmul_tensor_id = ParseTensorName(graph_view.GetNode(base.reshape_1_id)->node()->input(0));
      auto* add_view = graph_view.GetNode(base.add_id);
      mutation->AddOrUpdateRegularFanin(add_view,0, matmul_tensor_id);
      mutation->UpdateNodeOp(add_view, "BiasAdd");

      auto mul_tensor_id = ParseTensorName(add_view->node()->input(0));
      auto* matmul_view = graph_view.GetNode(base.matmul_id);
      mutation->AddOrUpdateRegularFanin(matmul_view,1, mul_tensor_id);

      auto weight_tensor_id = ParseTensorName(matmul_view->node()->input(1));
      auto* mul_view = graph_view.GetNode(base.mul_id);
      // Mul's input is Scale and Input. Input usually is Fanin 0, but in custom case, it's 1
      mutation->AddOrUpdateRegularFanin(mul_view, 1, weight_tensor_id);
      // mutation->AddOrUpdateRegularFanin(mul_view,0, weight_tensor_id);

      TF_RETURN_IF_ERROR(mutation->Apply());
    }
  }

  // Remove useless node
  utils::Mutation* mutation = graph_view.GetMutationBuilder();
  for (int i = 0; i < num_nodes; ++i) {
    if (nodes_to_delete[i]) {
      mutation->RemoveNode(graph_view.GetNode(i));
    }
  }
  TF_RETURN_IF_ERROR(mutation->Apply());
  *output = *graph_view.graph();

  VLOG(3) << "After custom matmul fusion rewrites: " << output->DebugString();

  return Status::OK();
}

void CustomMatMulBNFusion::Feedback(Cluster* cluster, const GrapplerItem& item,
                                  const GraphDef& optimize_output,
                                  double result) {
  // Nothing to do for CustomMatMulBNFusion.
}

}  // namespace grappler
}  // namespace tensorflow