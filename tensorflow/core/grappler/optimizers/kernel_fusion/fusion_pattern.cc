#include "tensorflow/core/grappler/optimizers/kernel_fusion/fusion_pattern.h"

namespace tensorflow {
namespace grappler {
namespace {

FusionPattern& FusionPattern::Name(const std::string& name) {
  this->name_ = name;
  return *this;
}

FusionPattern& FusionPattern::Type(FusionPatternType type) {
  this->type_ = type;
  return *this;
}

FusionPattern& FusionPattern::FusionOpName(std::string fusion_op_name) {
  this->fusion_op_name_ = fusion_op_name;
  return *this;
}

FusionPattern& FusionPattern::SetFusionPatternImpl(FusionPatternImpl* fusion_pattern_impl) {
  this->fusion_pattern_impl_ = fusion_pattern_impl;
  this->fusion_pattern_impl_->pattern = this;
  return *this;
}

FusionPattern& FusionPattern::Option(const std::string option_name, bool flag) {
  option_[option_name] = flag;
  return *this;
}

FusionPattern& FusionPattern::AddOpNode(
    const std::string& node_name, const std::string& op_name) {
  auto pattern_node_def = pattern_def_.add_node();
  pattern_node_def->set_name(node_name);
  pattern_node_def->set_type(op_name);
  return *this;
}

FusionPattern& FusionPattern::AddConnect(
    const std::string& edge_from, const std::string& edge_to) {
  size_t from_id = 0, to_id = 0;
  for (size_t i = 0; i < pattern_def_.node_size(); ++i) {
    if (pattern_def_.node(i).name() == edge_from) {
      from_id = i;
      continue;
    }
    if (pattern_def_.node(i).name() == edge_to) {
      to_id = i;
      continue;
    }
  }
  pattern_def_.mutable_node(from_id)->add_output(edge_from);
  pattern_def_.mutable_node(to_id)->add_input(edge_from);
  return *this;
}

FusionPattern& FusionPattern::AddOpNode(
    const std::string& node_name, const std::string& op_name,
    const std::vector<std::string>& iname,
    const std::vector<std::string>& oname) {
  auto pattern_node_def = pattern_def_.add_node();
  pattern_node_def->set_name(node_name);
  pattern_node_def->set_type(op_name);
  for (const auto& name : iname) {
    pattern_node_def->add_input(name);
  }
  for (const auto& name : oname) {
    pattern_node_def->add_output(name);
  }
  return *this;
}

FusionPattern& FusionPattern::Init() {
  fp_graph_.reset(new FPGraph(pattern_def_));
  CheckValid();
  return *this;
}

void FusionPattern::CheckValid() {
  switch (type_) {
    case kInOrder:
      CheckInOrderValid();
      break;
    case kInParallel:
      CheckInParallelValid();
      break;
  }
}

void FusionPattern::CheckInOrderValid() {
  CHECK_TRUE(input().size() == 1,
             "input is not one %u %s",
             input().size(), name_.c_str());

  CHECK_TRUE(output().size() == 1,
             "output is not one %u %s",
             input().size(), name_.c_str());

  CHECK_TRUE(fp_graph_->nodes.size() <= 2,
             "fp_graph_->nodes.size()=%u %s",
             fp_graph_->nodes.size(), name_.c_str());
}

void FusionPattern::CheckInParallelValid() {
  CHECK_TRUE(input().size() == 1,
             "input is not one %u %s",
             input().size(), name_.c_str());

  CHECK_TRUE(output().size() == 1,
             "output is not one %u %s",
             output().size(), name_.c_str());

  CHECK_TRUE(fp_graph_->nodes.size() <= 2,
             "fp_graph_->nodes.size()=%u %s",
             fp_graph_->nodes.size(), name_.c_str());
}

std::unordered_map<int, FMatch> g_fusion_fmatch_map;

bool FusionPattern::Match(Graph* graph) {
  BLAZE_CONDITION_THROW(g_fusion_fmatch_map.find(this->type()) != g_fusion_fmatch_map.end(),
                        "PatternType this->type()=",
                        this->type(),
                        " is not regisitered");
  switch (this->type()) {
    case kInOrder:
      return g_fusion_fmatch_map[this->type()](this, graph);
    case kInParallel:
      return g_fusion_fmatch_map[this->type()](this, graph);
  }
  return false;
}

void SaveGraph(const std::string& name, const NetDef& def) {
  static int dump_id = 0;
  std::ostringstream file;
  file << "dump/dump-" << std::setfill('0') << std::setw(4) << dump_id++ << "-" << name;
  std::ofstream out(file.str(), std::ios::binary);
  if (out) {
    std::ostringstream ser;
    def.SerializeToOstream(&ser);
    out << ser.str();
    out.close();
  }
}

NetDef FusionPatternPass(const NetDef& net_def, bool dump) {
  NetDef def = net_def;

  std::vector<std::shared_ptr<FusionPattern>>& pattern =
      FusionPatternRegisterer::Get()->pattern;
  FusionPatternType pattern_types[] = { kInOrder, kInParallel };

  bool finished;
  do {
    finished = true;
    for (auto type : pattern_types) {
      for (auto& p : pattern) {
        if (p->type() == type) {  // pass according to one by one pattern type.
          bool ct = true;
          do {
            Graph graph(def);
            LOG_DEBUG("probe name=%s", p->name().c_str());
            ct = p->Match(&graph);
            if (ct) {
              finished = false;
              LOG_DEBUG("p->name()=%s", p->name().c_str());
            }
            def = graph.GetNetDef();
            if (dump) { SaveGraph(p->name(), def); }
          } while (ct);
        }
      }
    }
  } while (!finished);

  return def;
}

}  // namespace blaze

