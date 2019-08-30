#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"

#include <queue>
#include <set>
#include <fstream>

#include "tensorflow/core/grappler/optimizers/fusion_pattern.h"

namespace tensorflow {
namespace grappler {

GraphRewriter::GraphRewriter(tensorflow::GraphDef *graph_def)
    : fused_graph_def_(graph_def) {
  // TODO ifdef debug
  raw_graph_def_ = *fused_graph_def_;
  // TODO end if
  nodes_.resize(fused_graph_def_->node_size());
  for (auto i = 0; i < fused_graph_def_->node_size(); ++i) {
    idx_map_[fused_graph_def_->node(i).name()] = i;
    nodes_[i].node_def = fused_graph_def_->mutable_node(i);
    nodes_[i].remove = false;
  }
  // Build directed graph
  InitNodes(idx_map_);
}

void GraphRewriter::InitNodes(const std::unordered_map<std::string, int> &idx_map) {
  int out_pos = 0;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    auto& node = nodes_[i];
    node.inputs.resize(node.node_def->input_size());
    for (auto j = 0; j < node.node_def->input_size(); ++j) {
      const auto& input_name = node.node_def->input(i);
      const auto& parent_name = GetParentName(input_name, &out_pos);
      auto iter = idx_map_.find(parent_name);
      if (iter != idx_map_.end()) {
        // set current node input
        node.inputs[i].parent_node_id = iter->second;
        node.inputs[i].output_pos = out_pos;
        // set parent node output
        Node::Output output;
        output.child_node_id = i;
        output.output_pos = out_pos;
        nodes_[iter->second].outputs.emplace_back(output);
      } else {
        node.inputs[i].parent_node_id = -1;
      }
    }
  }
}

bool GraphRewriter::FuseRewrite(FusionPattern& pattern) {
  bool valid = pattern.CheckValid();
  if (!valid) return false;

  bool graph_fused = false;
  const std::string& root_op_name = pattern.GetFusionPatternRootName();
  for (size_t i = 0; i < nodes_.size(); ++i) {
    // subgraph matching
    if (nodes_[i].op_name() != root_op_name) continue;
    if (BFS(i, pattern)) {
      graph_fused = true;
      break;
    }
  }

  Finalize();
  return graph_fused;
}

bool GraphRewriter::BFS(int root_id, FusionPattern& pattern) {
  std::queue<int> queue;
  std::set<int> visited;
  std::vector<Node*> candidate_fuse_nodes;
  queue.push(root_id);

  size_t bfs_node_iter = 0;
  while (!queue.empty()) {
    auto node_id = queue.front();
    queue.pop();
    if (node_id < 0 || node_id >= nodes_.size()) {
      LOG(ERROR) << "Invalid node_id: " << node_id
          << " node_size:" << nodes_.size();
      break;
    }
    if (!visited.count(node_id)) {
      auto& fuse_node = pattern.bfs_pattern_nodes()[bfs_node_iter];
      // match op name
      if (nodes_[node_id].op_name() != fuse_node.op_name) {
        LOG(ERROR) << "nodes_[node_id].name=" << nodes_[node_id].op_name()
            << " fuse_node.op_name=" << fuse_node.op_name;
        break;
      }
      // TODO: check single node here

      for (const auto& output_pos : fuse_node.output_pos) {
        if (output_pos < 0 || output_pos >= nodes_[node_id].outputs.size()) {
          LOG(ERROR) << "Invalid ";
          break;
        }
        queue.push(nodes_[node_id].outputs[output_pos].child_node_id);
        candidate_fuse_nodes.push_back(&(nodes_[node_id]));
        visited.insert(node_id);
        ++bfs_node_iter;
      }
    }
  }

  if (bfs_node_iter == pattern.bfs_pattern_nodes().size() && pattern.Match(candidate_fuse_nodes, this)) {
    pattern.GraphRewrite(candidate_fuse_nodes, this);
    return true;
  }
  return false;
}

void GraphRewriter::Finalize() {
  auto graph = *fused_graph_def_;
  fused_graph_def_->clear_node();
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].remove) {
      LOG(INFO) << "remove node:" << graph.node(i).DebugString();
      continue;
    }
    *(fused_graph_def_->add_node()) = graph.node(i);
  }
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