#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"

#include <queue>
#include <set>
#include <fstream>

#include "tensorflow/core/graph/graph_constructor.h"

namespace tensorflow {
namespace grappler {

GraphRewriter::GraphRewriter(GraphDef* graph_def)
    : fused_graph_def_(graph_def),
      graph_(new Graph(OpRegistry::Global())) {
  // TODO ifdef debug
  raw_graph_def_ = *fused_graph_def_;
  // TODO end if
  ConvertGraphDefToGraph({}, *graph_def, graph_.get());
}

bool GraphRewriter::FuseRewrite(FusionPattern& pattern) {
  bool valid = pattern.CheckValid();
  if (!valid) return false;

  bool graph_fused = false;
  const std::string& root_op_type = pattern.GetFusionPatternRootType();
  for (size_t i = 0; i < graph_->num_node_ids(); ++i) {
    Node* node = graph_->FindNodeId(i);
    // subgraph matching
    if (node->type_string() != root_op_type) continue;
    if (BFS(node, pattern)) {
      graph_fused = true;
      break;
    }
  }

  Finalize();
  return graph_fused;
}

bool GraphRewriter::BFS(Node* root, FusionPattern& pattern) {
  std::queue<int> queue;
  std::set<int> visited;
  std::vector<Node*> candidate_fuse_nodes;
  queue.push(root->id());

  size_t bfs_node_iter = 0;
  while (!queue.empty()) {
    auto node_id = queue.front();
    queue.pop();
    if (node_id < 0 || node_id >= graph_->num_node_ids()) {
      LOG(ERROR) << "Invalid node_id: " << node_id
          << " node_size:" << graph_->num_node_ids();
      break;
    }
    Node* node = graph_->FindNodeId(node_id);
    if (nullptr == node) continue;

    if (!visited.count(node_id)) {
      auto& fuse_node = pattern.bfs_pattern_nodes()[bfs_node_iter];
      // match op name
      if (node->type_string() != fuse_node.op_type_string) {
        LOG(ERROR) << "nodes->type_string()=" << node->type_string()
            << " fuse_node.op_type_string=" << fuse_node.op_type_string;
        break;
      }
      // TODO: check single node here

      for (const auto& input_pos : fuse_node.input_pos) {
        if (input_pos < 0 || input_pos >= node->num_inputs()) {
          LOG(ERROR) << "Invalid input pos, input_pos=" << input_pos
              << " node->num_inputs()=", node->num_inputs();
          break;
        }
        const Edge* input_edge = nullptr;
        Status status = node->input_edge(input_pos, &input_edge);
        if (status != Status::OK() || nullptr == input_edge) {
          LOG(ERROR) << "Get input edge failed, input_pos=" << input_pos;
          break;
        }

        Node* parent_node = input_edge->src();
        queue.push(parent_node->id());
        candidate_fuse_nodes.push_back(parent_node);
        visited.insert(parent_node->id());
        ++bfs_node_iter;
      }
    }
  }

  if (bfs_node_iter == pattern.bfs_pattern_nodes().size() && pattern.Match(candidate_fuse_nodes, graph_.get())) {
    pattern.GraphRewrite(candidate_fuse_nodes, graph_.get());
    return true;
  }
  return false;
}

void GraphRewriter::Finalize() {
  // TODO
  fused_graph_def_->Clear();
  graph_->ToGraphDef(fused_graph_def_);

  // TODO ifdef debug
  DumpGraph();
  // TODO end if
}

void GraphRewriter::DumpGraph() {
  static int step = 0;
  std::fstream fs1("/tmp/old." + std::to_string(step) + ".txt", std::ios::out);
  std::fstream fs2("/tmp/new." + std::to_string(step) + ".txt", std::ios::out);
  fs1 << raw_graph_def_.DebugString();
  fs2 << fused_graph_def_->DebugString();
  fs1.close();
  fs2.close();
  step++;
  LOG(INFO) << "step=" << step
      << " before node size:" << raw_graph_def_.node_size()
      << " after node size:" << fused_graph_def_->node_size();
}

}  // namespace grappler
}  // namespace tensorflow