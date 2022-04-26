/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_C_C_API_BLAZE_H_
#define TENSORFLOW_C_C_API_BLAZE_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/tf_attrtype.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

// --------------------------------------------------------------------------
// C API for TensorFlow.
//
// The API leans towards simplicity and uniformity instead of convenience
// since most usage will be by language specific wrappers.
//
// Conventions:
// * We use the prefix TF_ for everything in the API.
// * Objects are always passed around as pointers to opaque structs
//   and these structs are allocated/deallocated via the API.
// * TF_Status holds error information.  It is an object type
//   and therefore is passed around as a pointer to an opaque
//   struct as mentioned above.
// * Every call that has a TF_Status* argument clears it on success
//   and fills it with error info on failure.
// * unsigned char is used for booleans (instead of the 'bool' type).
//   In C++ bool is a keyword while in C99 bool is a macro defined
//   in stdbool.h. It is possible for the two to be inconsistent.
//   For example, neither the C99 nor the C++11 standard force a byte
//   size on the bool type, so the macro defined in stdbool.h could
//   be inconsistent with the bool keyword in C++. Thus, the use
//   of stdbool.h is avoided and unsigned char is used instead.
// * size_t is used to represent byte sizes of objects that are
//   materialized in the address space of the calling process.
// * int is used as an index into arrays.
// * Deletion functions are safe to call on nullptr.
//
// Questions left to address:
// * Might at some point need a way for callers to provide their own Env.
// * Maybe add TF_TensorShape that encapsulates dimension info.
//
// Design decisions made:
// * Backing store for tensor memory has an associated deallocation
//   function.  This deallocation function will point to client code
//   for tensors populated by the client.  So the client can do things
//   like shadowing a numpy array.
// * We do not provide TF_OK since it is not strictly necessary and we
//   are not optimizing for convenience.
// * We make assumption that one session has one graph.  This should be
//   fine since we have the ability to run sub-graphs.
// * We could allow NULL for some arguments (e.g., NULL options arg).
//   However since convenience is not a primary goal, we don't do this.
// * Devices are not in this API.  Instead, they are created/used internally
//   and the API just provides high level controls over the number of
//   devices of each type.

// Macro to control visibility of exported symbols in the shared library (.so,
// .dylib, .dll).
// This duplicates the TF_EXPORT macro definition in
// tensorflow/core/platform/macros.h in order to keep this .h file independent
// of any other includes.
#ifdef SWIG
#define TF_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TF_COMPILE_LIBRARY
#define TF_CAPI_EXPORT __declspec(dllexport)
#else
#define TF_CAPI_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TF_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TF_ProfStats {
  unsigned long long flops;
  unsigned long long tao_op_calls;
  bool dump_shapes = false;
} TF_ProfStats;

TF_CAPI_EXPORT extern TF_Buffer* TF_ReadGraphDefFromFile(
    const char* graph_def_path,
    TF_Status* status);
TF_CAPI_EXPORT extern TF_Buffer* TF_ReadMetaGraphDefFromFile(
    const char* graph_def_path,
    TF_Status* status);

TF_CAPI_EXPORT extern void TF_UpdateHugeConstPath(
    TF_Graph* graph,
    const char* directory);

// Sets the device attrs of nodes in graph to `device`.
TF_CAPI_EXPORT extern void TF_GraphSetDevice(TF_Graph* graph,
                                             int cpu_id, int gpu_id);
// This function creates a new TF_Session (which is created on success) using
// `session_options`, and then initializes state (restoring tensors and other
// assets) using `run_options`.
//
// Any NULL and non-NULL value combinations for (`run_options,
// `meta_graph_def`) are valid.
//
// - `export_dir` must be set to the path of the exported checkpoint.
// - `graph` must be a graph newly allocated with TF_NewGraph().
//
// If successful, populates `graph` with the contents of the Graph and
// `meta_graph_def` with the MetaGraphDef of the loaded model.
TF_CAPI_EXPORT extern TF_Session* TF_LoadSessionFromCheckpoint(
    const TF_SessionOptions* session_options, const TF_Buffer* run_options,
    const char* export_dir, TF_Graph* graph, TF_Buffer* meta_graph_def,
    TF_Status* status);

// Get input and output names in MetaGraphDef
// - `meta_graph_def` serialized MetaGraphDef protobuf message buffer
// - `method_name` use mthod name to lookup signature map
// - `ninput` input number
// - `input_names` input names, memory are managered by function caller
// - `noutput` output number
// - `output_names` output names, memory are managered by function caller
TF_CAPI_EXPORT extern void TF_GetIONamesFromMetaGraphDef(
    const TF_Buffer* meta_graph_def,
    bool use_method_name, const char* method_name,
    int* ninput, char*** input_names,
    int* noutput, char*** output_names, TF_Status* status);

TF_CAPI_EXPORT extern void TF_EnableSoftDevicePlacement(
    TF_SessionOptions* opt,
    unsigned char enable);
TF_CAPI_EXPORT extern void TF_EnableGemmOptimization(
    TF_SessionOptions* opt,
    unsigned char enable);
TF_CAPI_EXPORT extern void TF_EnableXlaAutoPadding(
    TF_SessionOptions* opt,
    unsigned char enable,
    unsigned char padding_type);
TF_CAPI_EXPORT extern void TF_EnableVirtualGPUDevices(
    TF_SessionOptions* opt,
    int num_virtual_gpus_per_device,
    int memory_limit_mb_per_virtual_gpus,
    int num_phisical_gpus);
TF_CAPI_EXPORT extern void TF_SetCPUDeviceCount(
    TF_SessionOptions* opt,
    int num_cpus);
TF_CAPI_EXPORT extern void TF_SetThreadPoolOptions(
    TF_SessionOptions* opt,
    int num_inter_op_threads,
    int num_intra_op_threads);
TF_CAPI_EXPORT extern void TF_SetGPUMemoryOptions(
    TF_SessionOptions* opt,
    unsigned char allow_growth,
    unsigned char force_gpu_compatible);
TF_CAPI_EXPORT extern bool TF_InitSessionOptionsFromPB(const char* pb_char,
    TF_SessionOptions* options);
TF_CAPI_EXPORT extern void TF_EnableAutoMixedPrecision(
    TF_SessionOptions* opt,
    unsigned char enable);
TF_CAPI_EXPORT extern void TF_EnableCudaGraph(
    TF_Buffer* run_options, unsigned char enable,
    unsigned char init, int count, TF_Status* status);
TF_CAPI_EXPORT extern void TF_EnableSingleThreadedExecutor(
    TF_SessionOptions* opt,
    unsigned char enable);

typedef int64_t TF_CallableHandle;

TF_CAPI_EXPORT extern void TF_SessionMakeCallable(
    TF_Session* tf_sess, TF_CallableHandle* callable_handle,
    const char* const* feed_names, int feed_count,
    const char* const* fetch_names, int fetch_count,
    bool adapt_device, const char* device_name, TF_Status* status);
TF_CAPI_EXPORT extern void TF_SessionRunCallable(
    TF_Session* tf_sess, TF_CallableHandle callable_handle,
    TF_Tensor* const* input_values, int ninputs,
    TF_Tensor** output_values, int noutputs,
    TF_Buffer* run_metadata, TF_Status* status,
    //[DYNAMIC-SHAPE]
    uint64_t before_padding = 0, uint64_t after_padding = 0,
    //[PROF-STATS]
    TF_ProfStats* prof_stats = nullptr
    );

TF_CAPI_EXPORT extern void TF_SessionReleaseCallable(
    TF_Session* tf_sess, TF_CallableHandle callable_handle,
    TF_Status* status);

TF_CAPI_EXPORT extern bool TF_CudaMemAlloc(
    int virtual_device_id,
    void** gpu_ptr,
    size_t length);
TF_CAPI_EXPORT extern bool TF_CudaMemDealloc(
    int virtual_device_id,
    void* gpu_ptr);
TF_CAPI_EXPORT extern bool TF_CudaMemCopyHostToDevice(
    int virtual_device_id,
    void* device_ptr,
    const void* host_ptr,
    size_t length);
TF_CAPI_EXPORT extern bool TF_CudaMemCopyDeviceToHost(
    int virtual_device_id,
    void* host_ptr,
    const void* device_ptr,
    size_t length);

TF_CAPI_EXPORT extern void TF_SetPaddingInfo(
    TF_Buffer* run_options, unsigned long long before_padding,
    unsigned long long after_padding, TF_Status* status);
TF_CAPI_EXPORT extern void TF_SaveRunMetadata(const TF_Buffer* run_metadata,
                                              const char* save_path,
                                              const char* file_name);
TF_CAPI_EXPORT extern bool TF_IsXlaFalseNode(TF_Graph* graph,
                                             const char* node_name);
#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_C_API_BLAZE_H_
