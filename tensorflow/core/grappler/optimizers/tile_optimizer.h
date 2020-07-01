#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_TILE_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_TILE_OPTIMIZER_H_

#include <string>
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

class TileOptimizer : public GraphOptimizer {
 public:
  ~TileOptimizer() override {}

  string name() const override { return "tile_optimizer"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* pruned_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& pruned_graph, double result) override;

 private:
  static std::string TileEqual(
      const NodeDef* node, GraphDef* graph_def, NodeMap* node_map,
      std::vector<const NodeDef*>* new_nodes);

  static NodeDef *AddNode(const std::string &op, const std::string &name, const std::string &device,
                          const std::vector<std::string> &inputs,
                          GraphDef* graph_def, NodeMap* node_map, std::vector<const NodeDef*>* new_nodes,
                          DataType type);
  static bool CheckType(const NodeDef* node);
	static bool TypeEqual(const NodeDef* node1, const NodeDef* node2, const std::string& key);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_TILE_OPTIMIZER_H_
