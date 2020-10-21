#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_UNFUSE_BATCH_NORM_OPTIMIZER_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_UNFUSE_BATCH_NORM_OPTIMIZER_H_

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

class UnfuseBatchNormOptimizer : public GraphOptimizer {
 public:
  UnfuseBatchNormOptimizer(DeviceBase* cpu_device) :
      cpu_device_(cpu_device) {}
  ~UnfuseBatchNormOptimizer() override {}

  string name() const override { return "unfuse_batchnorm"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;
  bool UsesFunctionLibrary() const override { return false; }
 private:
  static std::string UnfuseFusedBatchNormAndReshape(
      const NodeDef* node, GraphDef* graph_def, NodeMap* node_map,
      std::vector<const NodeDef*>* new_nodes);
  static std::string UnfuseFusedBatchNorm(
      const NodeDef* node, GraphDef* graph_def, NodeMap* node_map,
      std::vector<const NodeDef*>* new_nodes, const std::string &input);
  static std::string ConstructBatchNorm(
      GraphDef* graph_def, NodeMap *node_map,
      std::vector<const NodeDef*>* new_nodes,
      const string &name,
      const string &device,
      const string &input,
      const NodeDef *scale,
      const NodeDef *offset,
      const NodeDef *mean,
      const NodeDef *variance,
      float epsilon_value);
  Status doConstantFolding(Cluster* cluster, const GrapplerItem& item,
                           GraphDef* optimized_graph);
private:
  DeviceBase* const cpu_device_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_UNFUSE_BATCH_NORM_OPTIMIZER_H_
