//
// Created by qiaoxj on 2019-10-16.
//

#include "tensorflow/c/c_graph_editor_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/cc/tools/graph_editor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

using tensorflow::BypassGather;
using tensorflow::GraphDef;
using tensorflow::InsertConcatOp;
using tensorflow::MessageToBuffer;
using tensorflow::OptimizeDien;
using tensorflow::ReplaceWithBlazeGRU;
using tensorflow::SaveGraphDef;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::TagCPUDevice;
using tensorflow::errors::InvalidArgument;

extern "C" {

void TF_WriteGraphDefToFile(const char* graph_def_path,
                            const TF_Buffer* graph_def, TF_Status* status,
                            bool as_text) {
  GraphDef graphDef;
  if (!tensorflow::ParseProtoUnlimited(&graphDef, graph_def->data,
                                       graph_def->length)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }
  status->status =
      SaveGraphDef(string(graph_def_path), graphDef, true, as_text);
}
void TF_BypassGather(const char** bypass_op_names, int n_opnames,
                     const char** bypass_blacklist, int n_blacklist,
                     TF_Buffer* graph_def, TF_Status* status) {
  std::unordered_set<string> bypass_op_name_strings;
  for (int i = 0; i < n_opnames; i++) {
    bypass_op_name_strings.insert(bypass_op_names[i]);
  }
  std::unordered_set<string> bypass_blacklist_strings;
  for (int i = 0; i < n_blacklist; i++) {
    bypass_blacklist_strings.insert(bypass_blacklist[i]);
  }
  GraphDef graphDef;
  if (!tensorflow::ParseProtoUnlimited(&graphDef, graph_def->data,
                                       graph_def->length)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }
  status->status =
      BypassGather(bypass_op_name_strings, bypass_blacklist_strings, &graphDef);
  if (TF_GetCode(status) != TF_OK) return;
  status->status = MessageToBuffer(graphDef, graph_def);
}
void TF_ReplaceWithBlazeGRU(const char* custom_op, const char** input_keys,
                            const char** input_values, int n_inputs,
                            TF_Buffer* graph_def, TF_Status* status) {
  std::unordered_map<string, string> input_map;
  for (int i = 0; i < n_inputs; i++) {
    input_map.insert(std::make_pair(input_keys[i], input_values[i]));
  }
  GraphDef graphDef;
  if (!tensorflow::ParseProtoUnlimited(&graphDef, graph_def->data,
                                       graph_def->length)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }
  status->status = ReplaceWithBlazeGRU(custom_op, input_map, &graphDef);
  if (TF_GetCode(status) != TF_OK) return;
  status->status = MessageToBuffer(graphDef, graph_def);
}

void TF_InsertConcatOp(const char* concat_op, const char* valid_mask,
                       const char** valid_inputs, int n_valid_inputs,
                       char* temp_concat_op, TF_Buffer* graph_def,
                       TF_Status* status) {
  std::unordered_set<string> valid_input_string;
  for (int i = 0; i < n_valid_inputs; i++) {
    valid_input_string.insert(valid_inputs[i]);
  }
  GraphDef graphDef;
  if (!tensorflow::ParseProtoUnlimited(&graphDef, graph_def->data,
                                       graph_def->length)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }
  std::string temp_comcat;
  status->status = InsertConcatOp(concat_op, valid_mask, valid_input_string,
                                  &temp_comcat, &graphDef);
  if (TF_GetCode(status) != TF_OK) return;
  status->status = MessageToBuffer(graphDef, graph_def);
  if (TF_GetCode(status) == TF_OK) {
    temp_concat_op = new char[temp_comcat.size() + 1];
    memcpy(temp_concat_op, temp_comcat.c_str(), temp_comcat.size() + 1);
  }
}
void TF_TagCPUDevice(const char** inputs, int n_inputs, const char** outputs,
                     int n_outputs, const char** split_nodes, int n_split_nodes,
                     TF_Buffer* graph_def, TF_Status* status) {
  std::vector<string> input_names;
  for (int i = 0; i < n_inputs; i++) {
    input_names[i] = inputs[i];
  }
  std::vector<string> output_names;
  for (int i = 0; i < n_outputs; i++) {
    output_names[i] = outputs[i];
  }
  std::vector<string> splits_names;
  for (int i = 0; i < n_split_nodes; i++) {
    splits_names[i] = split_nodes[i];
  }
  GraphDef graphDef;
  if (!tensorflow::ParseProtoUnlimited(&graphDef, graph_def->data,
                                       graph_def->length)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }
  status->status =
      TagCPUDevice(input_names, output_names, splits_names, &graphDef);
  if (TF_GetCode(status) != TF_OK) return;
  status->status = MessageToBuffer(graphDef, graph_def);
}

void TF_OptimizeDienMode(TF_Buffer* graph_def, TF_Status* status) {
  GraphDef graphDef;
  if (!tensorflow::ParseProtoUnlimited(&graphDef, graph_def->data,
                                       graph_def->length)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }
  status->status = OptimizeDien(&graphDef);
  if (TF_GetCode(status) != TF_OK) return;
  status->status = MessageToBuffer(graphDef, graph_def);
}
}