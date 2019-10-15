//
// Created by qiaoxj on 2019-10-14.
//
#include "tensorflow/cc/tools/graph_editor.h"
#include <vector>

int main(int argc, char** argv) {
  using namespace tensorflow;
  std::string base_path =
      "/home/xianjie.qxj/dien_model/dien_1006_private/frozen_graph.pb";
  std::string edit_path =
      "/home/xianjie.qxj/dien_model/dien_1006_private/new_model.pb";
  tensorflow::GraphDef graphDef;
  tensorflow::ReadGraphDef(base_path, &graphDef);
  //  tensorflow::ReplaceOp("BatchMatMul", "BatchMatMulV2", &graphDef);
  //  tensorflow::ReplaceOp("BatchMatMulV2", "BatchMatMul", &graphDef);

  // Bypass Gather
  std::unordered_set<string> need_bypass_nodes{"GatherV2", "GatherV2_1",
                                               "GatherV2_2", "GatherV2_3"};
  std::unordered_set<string> donot_bypass_nodes{"Sum_4", "Sum_5"};
  BypassGather(need_bypass_nodes, donot_bypass_nodes, &graphDef);

  std::unordered_map<string, string> global_gru_inputs = {
      {"x", "TakeAxis:0"},
      {"h2h", "global_gru/rnn/mx_gru_cell/h2h_weight:0"},
      {"i2h", "global_gru/rnn/mx_gru_cell/i2h_weight:0"},
      {"h2h_bias", "global_gru/rnn/mx_gru_cell/h2h_bias:0"},
      {"i2h_bias", "global_gru/rnn/mx_gru_cell/i2h_bias:0"}};
  string global_gru_output = "global_gru/rnn/transpose_1";
  ReplaceWithBlazeGRU(global_gru_output, global_gru_inputs, &graphDef);

  std::unordered_map<string, string> cnxh_gru_inputs = {
      {"x", "TakeAxis_1:0"},
      {"h2h", "cnxh_gru/rnn/mx_gru_cell/h2h_weight:0"},
      {"i2h", "cnxh_gru/rnn/mx_gru_cell/i2h_weight:0"},
      {"h2h_bias", "cnxh_gru/rnn/mx_gru_cell/h2h_bias:0"},
      {"i2h_bias", "cnxh_gru/rnn/mx_gru_cell/i2h_bias:0"}};
  string cnxh_gru_output = "cnxh_gru/rnn/transpose_1";
  ReplaceWithBlazeGRU(cnxh_gru_output, cnxh_gru_inputs, &graphDef);

  string concat_op = "concat_5";
  std::unordered_set<string> valid_ops = {"item", "cate", "shop",
                                          "node", "prod", "brand"};
  string valid_mask = "a_";
  string temp_concat_op;
  InsertConcatOp(concat_op, valid_mask, valid_ops, &temp_concat_op, &graphDef);

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
  TagCPUDevice(inputs, outputs, split_nodes, &graphDef);

  tensorflow::SaveGraphDef(edit_path, graphDef, true, true);
  return 0;
}
