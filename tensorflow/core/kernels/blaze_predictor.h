#ifndef TENSORFLOW_CORE_KERNELS_BLAZE_PREDICOTR_H_
#define TENSORFLOW_CORE_KERNELS_BLAZE_PREDICOTR_H_

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/blaze.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/gpu_utils.h"
#endif

namespace tensorflow {

//Base blaze predictor, for normal run/(mlir)
class BlazePredictor {
 public:
  explicit BlazePredictor(OpKernelConstruction* ctx);
  virtual ~BlazePredictor() {}

  virtual void Compute(OpKernelContext* ctx);

 protected:
  // read from tensor proto
  std::string graph_def_str_;
  std::string blaze_option_path_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  //runtime options
  GraphDef graph_def_;
  Session* session_;
  BlazeRunOptions blaze_run_options_;
  Session::CallableHandle handle_;

 private:
  //session must created in constructor function, otherwise in compute function
  //it will cost lots of time the first time
  Status InitSession(OpKernelConstruction* ctx);

  void SetDeviceInGraphDef(const std::string device_name, GraphDef* graph_def);
};
}
#endif
