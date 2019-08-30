#include "tensorflow/core/grappler/optimizers/kernel_fusion/graph_rewriter.h"

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

bool GraphRewriter::VerifyFusionPattern(const FusionPattern &pattern) {
  if (pattern.fusion_op_name().empty()) {
    LOG(ERROR) << "fuse_op_name is empty!";
    return false;
  }
  // TOOD
  //if (pattern.bfs_node)
  return true;
}

bool GraphRewriter::RewriteGraph(FusionPattern& pattern) {
  bool valid = VerifyFusionPattern(pattern);
  if (!valid) return false;

  bool graph_fused = false;
  const std::string& root_op_name = GetFusePatternRootName(pattern);
  for (size_t i = 0; i < nodes.size(); ++i) {
    // subgraph matching
    if (nodes[i].op_name() != root_op_name) continue;
    if (BFS(i, pattern)) {
      graph_fused = true;
      break;
    }
  }
  // TODO
  //Finalize();
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
      auto& fuse_node = pattern.bfs_nodes[bfs_node_iter];
      // match op name
      if (nodes_[node_id].op_name() != fuse_node.op_name) {
        LOG(ERROR) << "nodes_[node_id].name=" << nodes_[node_id].op_name()
            << " fuse_node.op_name=" << fuse_node.op_name;
        break;
      }
      // TOOD node check
      /*if (!fuse_node.node_check(&(nodes_[node_id]), &pattern, this)); {
        LOG(ERROR) << "node check failed: " << nodes_[node_id].op_name();
        break;
      }*/
      for (const auto& intput_id : fuse_node.input_id) { // ? input_id 哪里来？
        if(input_id < 0 || input_id >= nodes_[node_id].inputs.size()) {
          LOG(ERROR) << "Invalid ";
          break;
        }
      }
      queue.push(nodes_[node_id].inputs[input_id].node_id);
      candidate_fuse_nodes.push_back(&(nodes_[node_id]));
      visited.insert(node_id);
      ++bfs_node_iter;
    }
  }

  if (bfs_node_iter == pattern.bfs_nodes.size() && pattern.total_node_checker(andidate_fuse_nodes, &pattern, this)) {
    auto node_def = NewFusedNodeDef(false);
    node_def->set_name(candidate_fuse_nodes[0]->name());
    node_def->set_op(pattern.fuse_op_name);
    // copy op inputs and attrs
    for (size_t k = 0; k < pattern.bfs_nodes.size(); ++k) {
      const auto& fuse_node = pattern.bfs_nodes[k];
      const auto& candidate_fuse_node = candidate_fuse_nodes[k];
      // copy inputs
      for (auto input_id : fuse_node.keep_input_id) {
        CHECK(input_id >= 0 && input_id < candidate_fuse_node->node_def->input_size())
            << "input_id=" << input_id << " candidate_fuse_node->node_def->input_size()="
            << candidate_fuse_node->node_def->input_size();
        node_def->add_input(candidate_fuse_node->node_def->input(input_id));
      }

      // copy attrs
      for (const auto& attr_name : fuse_node.keep_attr_names) {
        auto iter = candidate_fuse_node->node_def->mutable_attr()->find(attr_name);
        if (iter == candidate_fuse_node->node_def->mutable_attr()->end()) continue;
        (*node_def->mutable_attr())[attr_name] = iter->second;
      }
    }

    // remove candidate node for fused graph generation
    for (size_t i = 0; i < candidate_fuse_nodes.size(); ++i) {
      if (i == 0 || IsOutputRemoved(candidate_fuse_nodes[i])) {
        candidate_fuse_nodes[i]->remove = true;
      }
    }

    pattern.node_fuser(candidate_fuse_nodes, &pattern, this, node_def);
  }

}


}  // namespace grappler
}  // namespace tensorflow