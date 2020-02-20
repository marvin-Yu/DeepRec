#include "tensorflow/core/grappler/optimizers/unfuse_batch_norm_optimizer.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"

namespace tensorflow {
namespace grappler {

Status UnfuseBatchNormOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
        GraphDef* optimized_graph)
{
  *optimized_graph = item.graph;
  ArithmeticOptimizer::SimplifyArithmeticOpsRtp(optimized_graph,
          PeepHoleFun(&UnfuseBatchNormOptimizer::UnfuseFusedBatchNormAndReshape));
  auto s = doConstantFolding(cluster, item, optimized_graph);
  if (!s.ok()) {
    return s;
  }

  ArithmeticOptimizer::SimplifyArithmeticOpsRtp(optimized_graph,
          PeepHoleFun(&ArithmeticOptimizer::FuseMatMulAndMul));
  s = doConstantFolding(cluster, item, optimized_graph);
  if (!s.ok()) {
    return s;
  }

  ArithmeticOptimizer::SimplifyArithmeticOpsRtp(optimized_graph,
          PeepHoleFun(&ArithmeticOptimizer::FuseMatMulBiasAndMulBias));
  s = doConstantFolding(cluster, item, optimized_graph);
  if (!s.ok()) {
    return s;
  }

  return Status::OK();
}

void UnfuseBatchNormOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                    const GraphDef& optimized_graph, double result)
{
}

Status UnfuseBatchNormOptimizer::doConstantFolding(Cluster* cluster, const GrapplerItem& item,
        GraphDef* optimized_graph) {
    GrapplerItem inner_item = item;
    inner_item.graph = *optimized_graph;
    return ConstantFolding(cpu_device_).Optimize(cluster, inner_item, optimized_graph);
}

// 1. Reshape + FusedBatchNorm + Reshape --> Mul + Add
string UnfuseBatchNormOptimizer::UnfuseFusedBatchNormAndReshape(const NodeDef* node,
        GraphDef* graph_def, NodeMap* node_map, std::vector<const NodeDef*>* new_nodes)
{
    if (node->op() != "Reshape") {
        return "";
    }
    const NodeDef *shape = node_map->GetNode(node->input(1));
    const NodeDef *fused_batch_norm_node = node_map->GetNode(node->input(0));
    if (fused_batch_norm_node->op() != "FusedBatchNorm") {
        return "";
    }
    const NodeDef *reshape_input = node_map->GetNode(fused_batch_norm_node->input(0));
    if (reshape_input->op() != "Reshape") {
        return "";
    }
    return UnfuseFusedBatchNorm(fused_batch_norm_node, graph_def, node_map, new_nodes,
                                reshape_input->input(0));
}

string UnfuseBatchNormOptimizer::UnfuseFusedBatchNorm(const NodeDef* node,
        GraphDef* graph_def, NodeMap* node_map, std::vector<const NodeDef*>* new_nodes, const string &input)
{
    const NodeDef *scale = node_map->GetNode(node->input(1));
    const NodeDef *offset = node_map->GetNode(node->input(2));
    const NodeDef *mean = node_map->GetNode(node->input(3));
    const NodeDef *variance = node_map->GetNode(node->input(4));
    float epsilon = node->attr().at("epsilon").f();
    bool is_training = node->attr().at("is_training").b();
    string data_format = node->attr().at("data_format").s();
    if (!is_training &&
        data_format == "NHWC" &&
        scale->op() == "Const" &&
        offset->op() == "Const" &&
        mean->op() == "Const" &&
        variance->op() == "Const")
    {
        return ConstructBatchNorm(graph_def, node_map, new_nodes,
                node->name(), node->device(), input, scale, offset, mean, variance, epsilon);
    }
    return "";
}

string UnfuseBatchNormOptimizer::ConstructBatchNorm(
        GraphDef* graph_def, NodeMap *node_map,
        std::vector<const NodeDef*>* new_nodes,
        const string &name,
        const string &device,
        const string &input,
        const NodeDef *scale,
        const NodeDef *offset,
        const NodeDef *mean,
        const NodeDef *variance,
        float epsilon_value)
{
    NodeDef *epsilon = ArithmeticOptimizer::AddNode("Const", name + "_epsilon", device, {},
                               graph_def, node_map, new_nodes, "dtype");
    TensorProto value;
    Tensor v(DT_FLOAT, {});
    v.scalar<float>()() = epsilon_value;
    v.AsProtoTensorContent(&value);
    AddNodeAttr("value", value, epsilon);
    NodeDef *add = ArithmeticOptimizer::AddNode("Add", name + "_add", device, {epsilon->name(), variance->name()},
                           graph_def, node_map, new_nodes);
    NodeDef *rsqrt = ArithmeticOptimizer::AddNode("Rsqrt", name + "_rsqrt", device, {add->name()},
                             graph_def, node_map, new_nodes);
    NodeDef *scaling_factor = ArithmeticOptimizer::AddNode("Mul", name + "_scaling_factor", device,
            {rsqrt->name(), scale->name()},
            graph_def, node_map, new_nodes);

    NodeDef *mul = ArithmeticOptimizer::AddNode("Mul", name + "_mul", device, {input, scaling_factor->name()},
                           graph_def, node_map, new_nodes);

    NodeDef *mul_bias = ArithmeticOptimizer::AddNode("Mul", name + "_mul_bias", device, {mean->name(), scaling_factor->name()},
                                graph_def, node_map, new_nodes);
    NodeDef *neg = ArithmeticOptimizer::AddNode("Neg", name + "_neg", device, {mul_bias->name()},
                           graph_def, node_map, new_nodes);
    NodeDef *bias = ArithmeticOptimizer::AddNode("Add", name + "_final_bias", device, {neg->name(), offset->name()},
                            graph_def, node_map, new_nodes);
    NodeDef *bias_add = ArithmeticOptimizer::AddNode("BiasAdd", name + "_bias_add", device, {mul->name(), bias->name()},
                                graph_def, node_map, new_nodes);
    AddNodeAttr("data_format", "NHWC", bias_add);
    return bias_add->name();
}

}  // end namespace grappler
}  // end namespace tensorflow
