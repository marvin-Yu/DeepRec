#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_blaze.h"
#include "tensorflow/c/c_api_experimental.h"
#include <chrono>

using namespace std;
using namespace tensorflow;
//bazel build --copt=-mavx2 --copt='-DINTEL_MKL_GEMM_ONLY' --config=mkl_gemm_only -c opt --config=cuda  --copt -mfpmath=both --copt -mfma --copt -msse4.2 --copt -DGOOGLE_CUDA=1 --copt -D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/demo:demo


void InitSessionOptions(TF_SessionOptions* opt) {
  TF_EnableSoftDevicePlacement(opt, 1);
  TF_EnableXLACompilation(opt, true);
  TF_SetGPUMemoryOptions(opt, /*allow_growth=*/1, /*force_gpu_compatible=*/1);

  TF_EnableXlaAutoPadding(opt, true, 0);

  TF_EnableAutoMixedPrecision(opt, true);

  TF_EnableSingleThreadedExecutor(opt, false);
}

void Deallocator(void* data, size_t, void* arg) {
  tensorflow::cpu_allocator()->DeallocateRaw(data);
  *reinterpret_cast<bool*>(arg) = true;
}

int run(TF_Graph* graph, TF_Session* tf_sess, int batch) {
  TF_Status* s = TF_NewStatus();
  int64_t dims[] = {1, 252};
  const int num_bytes = 252 * sizeof(Eigen::half);
  Eigen::half* values =
      reinterpret_cast<Eigen::half*>(tensorflow::cpu_allocator()->AllocateRaw(
          EIGEN_MAX_ALIGN_BYTES, num_bytes));

  bool deallocator_called = false;
  TF_Tensor* feed = TF_NewTensor(TF_HALF, dims, 2, values, num_bytes, &Deallocator, &deallocator_called); 


  int64_t dims1[] = {batch, 756};
  const int num_bytes1 = batch * 756 * sizeof(Eigen::half);
  Eigen::half* values1 =
      reinterpret_cast<Eigen::half*>(tensorflow::cpu_allocator()->AllocateRaw(
          EIGEN_MAX_ALIGN_BYTES, num_bytes1));

  bool deallocator_called1 = false;
  TF_Tensor* feed1 = TF_NewTensor(TF_HALF, dims1, 2, values1, num_bytes1, &Deallocator, &deallocator_called1); 

  std::vector<TF_Output> inputs_;
  std::vector<TF_Tensor*> input_values_;
  inputs_.emplace_back(TF_Output{TF_GraphOperationByName(graph, "comm"), 0});
  inputs_.emplace_back(TF_Output{TF_GraphOperationByName(graph, "ncomm"), 0});
  input_values_.emplace_back(feed);
  input_values_.emplace_back(feed1);


  std::vector<TF_Tensor*> fetch_tensors;

  TF_Output output{TF_GraphOperationByName(graph, "add"), 0};
  TF_Tensor* ret;
  TF_SessionRun(tf_sess, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ &inputs_[0], /*input_values*/ &input_values_[0], /*ninputs*/ 2,
                // output related parameters
                /*outputs*/ &output, /*output_values*/ &ret,
                /*noutputs*/ 1,
                /*targets*/ nullptr, /*ntargets*/ 0,
                /*run_metadata*/ nullptr, s);
  if (TF_OK != TF_GetCode(s)) {
    cout << "============== run error" << endl;
  }
  TF_DeleteStatus(s);
  TF_DeleteTensor(feed1);
  TF_DeleteTensor(feed);
  TF_DeleteTensor(ret);
}

void run_thread(TF_Graph* graph, TF_Session* tf_sess) {
  run(graph, tf_sess, 100);
  run(graph, tf_sess, 99);
  int count = 0;
  while (true) {
    count++;
    auto start = chrono::system_clock::now();
    run(graph, tf_sess, 99);
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    if (duration > 700) {
      cout << "inference cost " << duration << " us" << " current count=" << count << endl;
    }
  }
}

void compile_thread(TF_Graph* graph, TF_Session* tf_sess) {
  for (int i = 0; i < 10000; i++) {
    auto start = chrono::system_clock::now();
    run(graph, tf_sess, i + 1);
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
	cout << i + 1 <<" inference cost " << duration << " us" << endl;
  }
}

int main() {
  TF_Status* s = TF_NewStatus();
  const std::string tf_graph_path = "/home/yuxing.hqb/python/cnxh_cvr_hash/tf/tf_frozen_graph";
  const std::string device = "/device:GPU:0";
  if (tf_graph_path.empty()) {
    return false;
  }
  TF_Buffer* graph_def = TF_ReadGraphDefFromFile(tf_graph_path.c_str(), s);
  if (TF_GetCode(s) != TF_OK) {
    return false;
  }

  TF_Graph* graph = TF_NewGraph();
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, opts, s);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(graph_def);
  if (TF_GetCode(s) != TF_OK) {
    return false;
  }
  TF_SessionOptions* opt = TF_NewSessionOptions();
  InitSessionOptions(opt);
  TF_Session* tf_sess = TF_NewSession(graph, opt, s);
  TF_DeleteSessionOptions(opt);
  if (TF_OK != TF_GetCode(s)) {
    return false;
  }
  //std::thread t1(run_thread, graph, tf_sess);
  std::thread t2(compile_thread, graph, tf_sess);

  //t1.join();
  t2.join();
  std::cout << "=================";
  TF_DeleteStatus(s);
  return 0;
}
