#include "tensorflow/core/grappler/optimizers/tile_optimizer.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {

Status TileOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  ArithmeticOptimizer::SimplifyArithmeticOpsRtp(optimized_graph,
      PeepHoleFun(&TileOptimizer::TileEqual));
  return Status::OK();
}

bool IsGpuDevice(const NodeDef* node) {
    return (node->device().find("GPU") != std::string::npos
        || node->device().find("gpu") != std::string::npos);
}

string TileOptimizer::TileEqual(const NodeDef* node,
        GraphDef* graph_def, NodeMap* node_map, std::vector<const NodeDef*>* new_nodes)
{
    if (IsGpuDevice(node)) {
      return "";
    }
    if (node->op() != "Equal") {
        return "";
    }

    //for tile + equal_to ---> euqal
    const NodeDef *tile = node_map->GetNode(node->input(0));
    const NodeDef *equal_to = node_map->GetNode(node->input(1));
    if (tile->op() != "Tile" && equal_to->op() != "Tile") {
      return "";
    }
    if (equal_to->op() == "Tile") {
      const NodeDef* tmp = tile;
      tile = equal_to;
      equal_to = tmp;
    }

    const NodeDef *tile_i0 = node_map->GetNode(tile->input(0));
    if (!CheckType(node) || !CheckType(tile)) {
      return "";
    }

    const NodeDef *tile_i1 = node_map->GetNode(tile->input(1));
      //generate new node for tile_euqal
      NodeDef* out = AddNode("TileEqual",  node->name() + "_TileEqual",
          node->device(), {tile_i0->name(),  equal_to->name(), tile_i1->name()},
          graph_def, node_map, new_nodes, node->attr().at("T").type());
      AddNodeAttr("Tmultiples", tile->attr().at("Tmultiples").type(), out);
    return out->name();
}

NodeDef *TileOptimizer::AddNode(
        const std::string &op, const std::string &name, const std::string &device,
        const std::vector<std::string> &inputs,
        GraphDef* graph_def, NodeMap* node_map, std::vector<const NodeDef*>* new_nodes,
        DataType type)
{
    NodeDef *node = graph_def->add_node();
    string new_name;
    int counter = 0;
    do {
        new_name = name + "_" + std::to_string(counter++);
    } while (node_map->GetNode(new_name));
    node->set_name(new_name);
    node->set_op(op);
    node->set_device(device);
    node_map->AddNode(node->name(), node);
    new_nodes->push_back(node);
    for (const string &input : inputs) {
        node->add_input(input);
        node_map->AddOutput(input, node->name());
    }
    AddNodeAttr("T", type, node);
    return node;
}

bool TileOptimizer::CheckType(const NodeDef* node) {
  if (node == nullptr) {
    return false;
  }

  DataType type = node->attr().at("T").type();
  if (type == DT_INT32 || type == DT_FLOAT || type == DT_DOUBLE) {
    return true;
  }
  return false;
}

void TileOptimizer::Feedback(Cluster* /*cluster*/,
                              const GrapplerItem& /*item*/,
                              const GraphDef& /*optimized_graph*/,
                              double /*result*/) {
  // Nothing to do for LoopOptimizer.
}

}  // end namespace grappler
}  // namespace tensorflow
