//
// Created by qiaoxj on 2019-10-14.
//
#include "tensorflow/cc/tools/graph_editor.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_graph.h"

namespace tensorflow {

Status ReplaceInput(const string& old_input, const string& new_input,
                    GraphDef* graphDef) {
  for (NodeDef& node : *(graphDef->mutable_node())) {
    for (string& input : *(node.mutable_input())) {
      if (input == old_input) {
        LOG(INFO) << absl::StrFormat("Replace node %s\'s input %s with %s",
                                     node.name(), old_input, new_input);
        input = new_input;
      }
    }
  }
  return Status::OK();
}

Status ReplaceOp(const string& old_name, const string& new_name,
                 GraphDef* graphDef) {
  for (auto& node : *(graphDef->mutable_node())) {
    if (node.op() == old_name) {
      node.set_op(new_name);
      LOG(INFO) << old_name << " is replaced to " << new_name;
    }
  }
  return Status::OK();
}

Status ReadGraphDef(const string& model_path, GraphDef* graphDef) {
  if (Env::Default()->FileExists(model_path).ok()) {
    return ReadBinaryProto(Env::Default(), model_path, graphDef);
  } else {
    return Status(
        errors::NotFound("Could not find SavedModel .pb or .pbtxt at supplied "
                         "export directory path: " +
                         model_path));
  }
}

Status SaveGraphDef(const string& model_path, const GraphDef& graphDef,
                    bool as_binary, bool as_txt) {
  if (as_binary) {
    TF_RETURN_IF_ERROR(WriteBinaryProto(Env::Default(), model_path, graphDef));
  }
  if (as_txt) {
    TF_RETURN_IF_ERROR(WriteStringToFile(Env::Default(),
                                         strings::StrCat(model_path, ".txt"),
                                         graphDef.DebugString()));
  }
  return Status::OK();
}

Status BypassGather(const std::unordered_set<string>& bypass_op_names,
                    const std::unordered_set<string>& bypass_blacklist,
                    GraphDef* graphDef) {
  std::unordered_map<string, string> bypass_map;
  for (auto& node : graphDef->node()) {
    if (bypass_op_names.find(node.name()) != bypass_op_names.end()) {
      for (auto& input : node.input()) {
        if (input.find("Placeholder") == string::npos &&
            input.find("axis") == string::npos) {
          bypass_map[node.name()] = input;
          VLOG(INFO) << "bypass_map: " << node.name() << ", " << input;
        }
      }
    }
  }
  if (bypass_map.empty()) {
    return errors::NotFound("bypass_op_names nod found.");
  }
  for (NodeDef& node : *(graphDef->mutable_node())) {
    for (string& input : *(node.mutable_input())) {
      if (bypass_op_names.find(input) != bypass_op_names.end() &&
          bypass_blacklist.find(node.name()) == bypass_blacklist.end()) {
        LOG(INFO) << "Replace Node \"" << node.name() << "\" input \"" << input
                  << "\" with \"" << bypass_map[input] << "\"";
        input = bypass_map[input];
      }
    }
  }
  return Status::OK();
}

Status ReplaceWithBlazeGRU(const string& custom_op,
                           const std::unordered_map<string, string>& inputs,
                           GraphDef* graphDef) {
  std::unordered_set<string> node_names;
  for (auto& node : graphDef->node()) {
    node_names.insert(node.name());
  }
  //check
  if (node_names.count(custom_op) == 0){
    return errors::NotFound("gru output nod found.");
  }
  if (inputs.empty()){
    return errors::NotFound("No custom gru input.");
  }
  for (auto& i : inputs) {
    if(node_names.count(i.second) == 0) {
      return errors::NotFound("gru input " + i.second + " not found");
    }
  }
  NodeDef* blaze_gru_node = graphDef->add_node();
  // find valid names
  string gru_op_name = "BlazeGRU";
  int64 counter = 1;
  while (!node_names.insert(gru_op_name).second) {
    gru_op_name = strings::StrCat("BlazeGRU_", counter++);
  }
  blaze_gru_node->set_name(gru_op_name);
  blaze_gru_node->set_op("BlazeGRU");
  blaze_gru_node->add_input(inputs.at("x"));
  blaze_gru_node->add_input(inputs.at("h2h"));
  blaze_gru_node->add_input(inputs.at("i2h"));
  blaze_gru_node->add_input(inputs.at("h2h_bias"));
  blaze_gru_node->add_input(inputs.at("i2h_bias"));
  (*blaze_gru_node->mutable_attr())["T"].set_type(DataType::DT_FLOAT);
  LOG(INFO) << "BlazeGRU op: " << blaze_gru_node->ShortDebugString();
  TF_RETURN_IF_ERROR(ReplaceInput(custom_op, gru_op_name, graphDef));
  return Status::OK();
}

Status InsertConcatOp(const string& old_concat_op_name,
                      const string& valid_mask,
                      const std::unordered_set<string>& valid_inputs,
                      string* temp_concat_op, GraphDef* graphDef) {
  std::unordered_set<string> node_names;
  NodeDef* old_concat_op = nullptr;
  NodeDef* old_concat_axis_op = nullptr;
  for (auto& node : *(graphDef->mutable_node())) {
    if (node.name() == old_concat_op_name) {
      old_concat_op = &node;
    }
    if (node.name() == old_concat_op_name + "/axis") {
      old_concat_axis_op = &node;
    }
    node_names.insert(node.name());
  }
  if(old_concat_op == nullptr || old_concat_axis_op == nullptr) {
    return errors::NotFound("old concat op nod found.");
  }
  if(valid_inputs.empty() && valid_mask.empty()) {
    return errors::NotFound("valid_inputs nod found.");
  }
  // function to create concat node
  auto create_concat = [&](std::vector<string> inputs) -> string {
    // create name
    string concat_op_name = "concat";
    int64 counter = 1;
    while (!node_names.insert(concat_op_name).second) {
      concat_op_name = strings::StrCat("concat_", counter++);
    }

    // create copy attr function
    auto copy_attribute = [](const string& attribute_name, const NodeDef& from,
                             NodeDef* to_node) {
      (*to_node->mutable_attr())[attribute_name] =
          from.attr().at(attribute_name);
    };

    // create const axis node
    auto const_axis_node = graphDef->add_node();
    string const_axis_node_name = concat_op_name + "/axis";
    const_axis_node->set_name(const_axis_node_name);
    const_axis_node->set_op("Const");
    copy_attribute("_output_shapes", *old_concat_axis_op, const_axis_node);
    copy_attribute("dtype", *old_concat_axis_op, const_axis_node);
    copy_attribute("value", *old_concat_axis_op, const_axis_node);

    // create concat node
    auto new_concat = graphDef->add_node();
    new_concat->set_name(concat_op_name);
    new_concat->set_op("ConcatV2");
    for (auto& input : inputs) {
      new_concat->add_input(input);
    }
    new_concat->add_input(const_axis_node_name);
    (*new_concat->mutable_attr())["N"].set_i(inputs.size());
    copy_attribute("T", *old_concat_op, new_concat);
    copy_attribute("Tidx", *old_concat_op, new_concat);
    return new_concat->name();
  };

  std::vector<string> concat1_inputs;
  std::vector<string> concat2_inputs = {"concat1"};
  for (auto& input : old_concat_op->input()) {
    if (input == old_concat_axis_op->name()) {
      continue;
    }
    if (input.find(valid_mask) != string::npos ||
        valid_inputs.count(input) > 0) {
      concat1_inputs.push_back(input);
    } else {
      concat2_inputs.push_back(input);
    }
  }
  auto concat1 = create_concat(concat1_inputs);
  *temp_concat_op = concat1;
  concat2_inputs[0] = concat1;
  auto concat2 = create_concat(concat2_inputs);
  TF_RETURN_IF_ERROR(ReplaceInput(old_concat_op_name, concat2, graphDef));
  return Status::OK();
}

Status TagCPUDevice(const std::vector<string>& inputs,
                    const std::vector<string>& outputs,
                    const std::vector<string>& split_nodes,
                    GraphDef* graphDef) {
  if (inputs.empty() || outputs.empty() || split_nodes.empty()) {
    return errors::InvalidArgument("TagCPUDevice error.");
  }
  string transforms =
      "strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics)";
  GraphDef copy_graph = *graphDef;
  graph_transforms::TransformParameters parameters;
  TF_RETURN_IF_ERROR(
      graph_transforms::ParseTransformParameters(transforms, &parameters));
  TF_RETURN_IF_ERROR(graph_transforms::TransformGraph(inputs, split_nodes,
                                                      parameters, &copy_graph));
  TF_RETURN_IF_ERROR(
      graph_transforms::TransformGraph(inputs, outputs, parameters, graphDef));
  std::unordered_set<string> all_cpu_nodes;
  for (auto& node : copy_graph.node()) {
    all_cpu_nodes.insert(node.name());
  }
  for (NodeDef& node : *(graphDef->mutable_node())) {
    if (all_cpu_nodes.find(node.name()) != all_cpu_nodes.end()) {
      node.set_device("/device:CPU:0");
    }
  }
  return Status::OK();
}

Status OptimizeDien(GraphDef* graphDef) {
  // Replace
  TF_RETURN_IF_ERROR(ReplaceOp("BatchMatMul", "BatchMatMulV2", graphDef));
  // Bypass Gather
  std::unordered_set<string> need_bypass_nodes{"GatherV2", "GatherV2_1",
                                               "GatherV2_2", "GatherV2_3"};
  std::unordered_set<string> donot_bypass_nodes{"Sum_4", "Sum_5"};
  TF_RETURN_IF_ERROR(
      BypassGather(need_bypass_nodes, donot_bypass_nodes, graphDef));

  // Replace global_gru
  std::unordered_map<string, string> global_gru_inputs = {
      {"x", "TakeAxis"},
      {"h2h", "global_gru/rnn/mx_gru_cell/h2h_weight"},
      {"i2h", "global_gru/rnn/mx_gru_cell/i2h_weight"},
      {"h2h_bias", "global_gru/rnn/mx_gru_cell/h2h_bias"},
      {"i2h_bias", "global_gru/rnn/mx_gru_cell/i2h_bias"}};
  string global_gru_output = "global_gru/rnn/transpose_1";
  TF_RETURN_IF_ERROR(
      ReplaceWithBlazeGRU(global_gru_output, global_gru_inputs, graphDef));

  // Replace cnxh_gru
  std::unordered_map<string, string> cnxh_gru_inputs = {
      {"x", "TakeAxis_1"},
      {"h2h", "cnxh_gru/rnn/mx_gru_cell/h2h_weight"},
      {"i2h", "cnxh_gru/rnn/mx_gru_cell/i2h_weight"},
      {"h2h_bias", "cnxh_gru/rnn/mx_gru_cell/h2h_bias"},
      {"i2h_bias", "cnxh_gru/rnn/mx_gru_cell/i2h_bias"}};
  string cnxh_gru_output = "cnxh_gru/rnn/transpose_1";
  TF_RETURN_IF_ERROR(
      ReplaceWithBlazeGRU(cnxh_gru_output, cnxh_gru_inputs, graphDef));

  // Insert concat
  string concat_op = "concat_5";
  std::unordered_set<string> valid_ops = {"item", "cate", "shop",
                                          "node", "prod", "brand"};
  string valid_mask = "a_";
  string temp_concat_op;
  TF_RETURN_IF_ERROR(InsertConcatOp(concat_op, valid_mask, valid_ops,
                                    &temp_concat_op, graphDef));

  // TagCPUDevice
  std::vector<string> inputs = {
      "a_141_1",     "a_201",     "a_203",     "a_204",     "item",
      "cate",        "shop",      "node",      "a_210",     "a_212",
      "a_213",       "a_214",     "prod",      "brand",     "a_219",
      "a_301",       "a_601",     "a_652",     "a_653",     "a_514",
      "a_515",       "a_516",     "a_508",     "a_509",     "a_510",
      "a_564",       "a_565",     "a_701",     "a_702",     "a_707",
      "a_708",       "a_805",     "a_806",     "a_807",     "a_825",
      "a_826",       "a_827",     "a_850",     "a_853",     "a_6100_3",
      "a_6100_14",   "a_6100_30", "a_6100_60", "a_6200_3",  "a_6200_14",
      "a_6200_30",   "a_6200_60", "a_6101_3",  "a_6101_14", "a_6101_30",
      "a_6101_60",   "a_6201_3",  "a_6201_14", "a_6201_30", "a_6201_60",
      "a_6102_3",    "a_6102_14", "a_6102_30", "a_6102_60", "a_6202_3",
      "a_6202_14",   "a_6202_30", "a_6202_60", "a_6103_3",  "a_6103_14",
      "a_6103_30",   "a_6103_60", "a_6203_3",  "a_6203_14", "a_6203_30",
      "a_6203_60",   "a_861",     "a_862",     "a_507",     "a_507_1",
      "u_109_14",    "u_110_14",  "u_113_14",  "u_117_14",  "u_126_14",
      "u_127_14",    "u_119",     "u_160_3",   "u_161_3",   "u_150_14",
      "u_1100_3",    "u_1100_14", "u_1100_30", "u_1100_60", "u_1101_3",
      "u_1101_14",   "u_1101_30", "u_1101_60", "u_1102_3",  "u_1102_14",
      "u_1102_30",   "u_1102_60", "u_1103_3",  "u_1103_14", "u_1103_30",
      "u_1103_60",   "u_121",     "u_122",     "item_1",    "cate_1",
      "node_1",      "shop_1",    "brand_1",   "prod_1",    "item_2",
      "cate_2",      "node_2",    "shop_2",    "brand_2",   "prod_2",
      "Placeholder", "u_124",     "u_125",     "u_126",     "u_127",
      "u_128",       "u_129"};
  std::vector<string> split_nodes = {"concat", "concat_1", "concat_3",
                                     "concat_4"};
  split_nodes.push_back(temp_concat_op);
  std::vector<string> outputs = {"add_1"};
  TF_RETURN_IF_ERROR(TagCPUDevice(inputs, outputs, split_nodes, graphDef));
  return Status::OK();
}
}  // namespace tensorflow
