//
// Created by qiaoxj on 2019-10-16.
//

#ifndef TENSORFLOW_C_GRAPH_EDITOR_API_H
#define TENSORFLOW_C_GRAPH_EDITOR_API_H

#include "tensorflow/c/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

TF_CAPI_EXPORT extern void TF_WriteGraphDefToFile(const char* graph_def_path,
                                                  const TF_Buffer* graph_def,
                                                  TF_Status* status,
                                                  bool as_text = false);

TF_CAPI_EXPORT extern void TF_BypassGather(
    const char** bypass_op_names, int n_opnames, const char** bypass_blacklist,
    int n_blacklist, TF_Buffer* graph_def, TF_Status* status);

TF_CAPI_EXPORT extern void TF_ReplaceWithBlazeGRU(
    const char* custom_op, const char** input_keys, const char** input_values,
    int n_inputs, TF_Buffer* graph_def, TF_Status* status);
TF_CAPI_EXPORT extern void TF_InsertConcatOp(
    const char* concat_op, const char* valid_mask, const char** valid_inputs,
    int n_valid_inputs, char* temp_concat_op, TF_Buffer* graph_def,
    TF_Status* status);
TF_CAPI_EXPORT extern void TF_TagCPUDevice(const char** inputs, int n_inputs,
                                           const char** outputs, int n_outputs,
                                           const char** split_nodes,
                                           int n_split_nodes,
                                           TF_Buffer* graph_def,
                                           TF_Status* status);
TF_CAPI_EXPORT extern void TF_OptimizeDienMode(TF_Buffer* graph_def,
                                               TF_Status* status);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_GRAPH_EDITOR_API_H
