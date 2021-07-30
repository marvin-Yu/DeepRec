#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/cuda_graph_session.h"

#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime_api.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class CudaGraphSessionTest : public ::testing::Test {
 public:
  void init(std::unique_ptr<CudaGraphSession>& session,
            const std::vector<std::string> outputNames,
            const std::map<int, int>& batchs,
            bool outputOnCPU = false) {
    SessionOptions options;
    options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
    options.config.mutable_gpu_options()->set_allow_growth(false);
    options.target = CUDA_GRAPH_TARGET_NAME;
    for (auto& str : outputNames) {
      options.config.mutable_cuda_graph_options()->add_outputs(str);
    }
    for (auto& it : batchs) {
      options.config.mutable_cuda_graph_options()->add_batchs(it.first);
      options.config.mutable_cuda_graph_options()->add_copies(it.second);
    }
    if (outputOnCPU) {
        options.config.mutable_cuda_graph_options()->set_output_on_cpu(outputOnCPU);
    }
    session.reset(static_cast<CudaGraphSession*>(NewSession(options)));
  }

  Tensor copyFromGPU(const Tensor& t) {
    Tensor cpu_tensor(t.dtype(), t.shape());
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(cpu_tensor.base<void>(), t.base<void>(), t.TotalBytes(),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return cpu_tensor;
  }

  Tensor copyToGPU(const Tensor& t, Allocator* gpu_allocator_) {
    Tensor gpu_tensor(gpu_allocator_, t.dtype(), t.shape());
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(gpu_tensor.base<void>(), t.base<void>(), t.TotalBytes(),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return gpu_tensor;
  }

  CudaGraphSession::CudaGraphContextPtr GetCudaGraphContext(int batch_size) {
    return std::move(cuda_session->GetCudaGraphContext(batch_size));
  }
  Allocator* GetGPUAllocator() { return cuda_session->gpu_allocator_; }
  std::unique_ptr<CudaGraphSession> cuda_session;
};

void TFRun(Session* sess, int infer_num,
           std::vector<std::pair<std::string, Tensor>>* inputs,
           std::vector<std::string>* output_names,
           std::vector<Tensor>* output_tensors) {
  for (int i = 0; i < infer_num; i++) {
    TF_CHECK_OK(sess->Run(*inputs, *output_names, {}, output_tensors));
  }
}

void CudaGraphRun(CudaGraphSession* sess, int infer_num,
                  std::vector<std::pair<std::string, Tensor>>* inputs,
                  std::vector<std::string>* output_names,
                  std::vector<Tensor>* output_tensors) {
  for (int i = 0; i < infer_num; i++) {
    TF_CHECK_OK(sess->Run(*inputs, *output_names, {}, output_tensors));
  }
}

void CudaGraphCreate(CudaGraphSession* sess, GraphDef* def) {
  TF_CHECK_OK(sess->Create(*def));
}

// y = tf.square(x)
GraphDef CreateGraphForYEqualsXSquared() {
  GraphDef graph_def;
  const char* text_proto = R"EOF(
node {
  name: "x"
  op: "Placeholder"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { dim { size: -1 } } } }
}
node {
  name: "y"
  op: "Square"
  device: "/device:GPU:0"
  input: "x"
  attr { key: "T" value { type: DT_FLOAT } }
}
versions {
  producer: 26
}
  )EOF";

  QCHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph_def));
  return graph_def;
}

// result1 = x + y
// result2 = result1 * z
GraphDef CreateGraphForXPlusYThenMultiplyByZ() {
  GraphDef graph_def;
  const char* text_proto = R"EOF(
node {
  name: "x"
  op: "Placeholder"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { dim { size: -1 } } } }
}
node {
  name: "y"
  op: "Placeholder"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { dim { size: -1 } } } }
}
node {
  name: "result1"
  op: "Add"
  input: "x"
  input: "y"
  device: "/device:GPU:0"
  attr { key: "T" value { type: DT_FLOAT } }
}
node {
  name: "const1"
  op: "Const"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 1 } } float_val: 1 } } }
}
node {
  name: "result"
  op: "Mul"
  input: "result1"
  input: "const1"
  device: "/device:GPU:0"
  attr { key: "T" value { type: DT_FLOAT } }
}

node {
  name: "z"
  op: "Placeholder"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { dim { size: -1 } } } }
}
node {
  name: "result2"
  op: "Mul"
  input: "result"
  input: "z"
  device: "/device:GPU:0"
  attr { key: "T" value { type: DT_FLOAT } }
}
versions {
  producer: 26
}
  )EOF";

  QCHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph_def));
  return graph_def;
}

GraphDef CreateGraphForYEqualsXSquaredUnknownRank() {
  GraphDef graph_def;
  const char* text_proto = R"EOF(
node {
  name: "x"
  op: "Placeholder"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { unknown_rank: true } } }
}
node {
  name: "y"
  op: "Square"
  input: "x"
  device: "/device:GPU:0"
  attr { key: "T" value { type: DT_FLOAT } }
}
versions {
  producer: 26
}
  )EOF";

  QCHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph_def));
  return graph_def;
}

void ModifyType(GraphDef& graph, DataType type) {
  for (size_t i = 0; i < graph.node_size(); ++i) {
    auto* node = graph.mutable_node(i);
    if (node->mutable_attr()->count("T") > 0) {
      (*node->mutable_attr())["T"].set_type(type);
    }
    if (node->mutable_attr()->count("dtype") > 0) {
      (*node->mutable_attr())["dtype"].set_type(type);
    }
  }
}

void ModifyNodeDevice(GraphDef& graph, const std::string& device) {
  for (size_t i = 0; i < graph.node_size(); ++i) {
    auto* node = graph.mutable_node(i);
    node->set_device(device);
  }
}

TEST_F(CudaGraphSessionTest, TestSimple) {
  std::map<int, int> batchs;
  batchs[3] = 3;
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForYEqualsXSquared()));
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  float* data = input.flat<float>().data();
  data[0] = 1.0f;
  data[1] = 2.2f;
  data[2] = 3.7f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  TF_CHECK_OK(cuda_session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = copyFromGPU(outputs[0]);
  data = result.flat<float>().data();
  EXPECT_FLOAT_EQ(1.0, data[0]);
  EXPECT_FLOAT_EQ(4.84, data[1]);
  EXPECT_FLOAT_EQ(13.69, data[2]);
}

TEST_F(CudaGraphSessionTest, TestGPUInput) {
  std::map<int, int> batchs;
  batchs[3] = 3;
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForYEqualsXSquared()));
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  float* data = input.flat<float>().data();
  data[0] = 1.0f;
  data[1] = 2.2f;
  data[2] = 3.7f;
  auto gpu_input = copyToGPU(input, GetGPUAllocator());
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", gpu_input));
  TF_CHECK_OK(cuda_session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = copyFromGPU(outputs[0]);
  data = result.flat<float>().data();
  EXPECT_FLOAT_EQ(1.0, data[0]);
  EXPECT_FLOAT_EQ(4.84, data[1]);
  EXPECT_FLOAT_EQ(13.69, data[2]);
}

TEST_F(CudaGraphSessionTest, TestMultiInputAndOutput) {
  std::map<int, int> batchs;
  batchs[2] = 1;
  init(cuda_session, {"result1", "result2"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForXPlusYThenMultiplyByZ()));
  Tensor x(DT_FLOAT, TensorShape({2}));
  Tensor y(DT_FLOAT, TensorShape({2}));
  Tensor z(DT_FLOAT, TensorShape({2}));
  std::vector<Tensor> outputs;
  float* dx = x.flat<float>().data();
  float* dy = y.flat<float>().data();
  float* dz = z.flat<float>().data();
  dx[0] = 1.0f;
  dx[1] = 2.0f;
  dy[0] = 3.0f;
  dy[1] = 4.0f;
  dz[0] = 5.0f;
  dz[1] = 6.0f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  inputs_map.push_back(std::make_pair("y", y));
  inputs_map.push_back(std::make_pair("z", z));
  TF_CHECK_OK(
      cuda_session->Run(inputs_map, {"result1", "result2"}, {}, &outputs));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(2, outputs[0].NumElements());
  ASSERT_EQ(2, outputs[1].NumElements());
  auto out1 = copyFromGPU(outputs[0]);
  auto out2 = copyFromGPU(outputs[1]);
  float* result1 = out1.flat<float>().data();
  float* result2 = out2.flat<float>().data();
  EXPECT_FLOAT_EQ(4.0, result1[0]);
  EXPECT_FLOAT_EQ(20.0, result2[0]);
  EXPECT_FLOAT_EQ(6.0, result1[1]);
  EXPECT_FLOAT_EQ(36.0, result2[1]);
}

TEST_F(CudaGraphSessionTest, TestInt32TypeFailed) {
  std::map<int, int> batchs;
  batchs[3] = 3;
  GraphDef graph = CreateGraphForYEqualsXSquared();
  ModifyType(graph, DT_INT32);
  init(cuda_session, {"y"}, batchs);
  ASSERT_FALSE(cuda_session->Create(graph).ok());
}

TEST_F(CudaGraphSessionTest, TestInt64Type) {
  std::map<int, int> batchs;
  batchs[3] = 3;
  GraphDef graph = CreateGraphForYEqualsXSquared();
  ModifyType(graph, DT_INT64);
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(graph));
  Tensor input(DT_INT64, TensorShape({3}));
  std::vector<Tensor> outputs;
  int64* data = input.flat<int64>().data();
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  TF_CHECK_OK(cuda_session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = copyFromGPU(outputs[0]);
  data = result.flat<int64>().data();
  EXPECT_EQ(1, data[0]);
  EXPECT_EQ(4, data[1]);
  EXPECT_EQ(9, data[2]);
}

TEST_F(CudaGraphSessionTest, TestDoubleType) {
  std::map<int, int> batchs;
  batchs[3] = 3;
  GraphDef graph = CreateGraphForYEqualsXSquared();
  ModifyType(graph, DT_DOUBLE);
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(graph));
  Tensor input(DT_DOUBLE, TensorShape({3}));
  std::vector<Tensor> outputs;
  double* data = input.flat<double>().data();
  data[0] = 1.0;
  data[1] = 2.0;
  data[2] = 3.0;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  TF_CHECK_OK(cuda_session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = copyFromGPU(outputs[0]);
  data = result.flat<double>().data();
  EXPECT_FLOAT_EQ(1.0, data[0]);
  EXPECT_FLOAT_EQ(4.0, data[1]);
  EXPECT_FLOAT_EQ(9.0, data[2]);
}

TEST_F(CudaGraphSessionTest, TestCPUNodeFailed) {
  std::map<int, int> batchs;
  batchs[3] = 3;
  GraphDef graph = CreateGraphForYEqualsXSquared();
  ModifyNodeDevice(graph, "/device:CPU:0");
  init(cuda_session, {"y"}, batchs);
  ASSERT_FALSE(cuda_session->Create(graph).ok());
}

TEST_F(CudaGraphSessionTest, TestEmptyBatchs) {
  std::map<int, int> batchs;
  GraphDef graph = CreateGraphForYEqualsXSquared();
  init(cuda_session, {"y"}, batchs);
  ASSERT_FALSE(cuda_session->Create(graph).ok());
}

TEST_F(CudaGraphSessionTest, TestShapeUnknownRank) {
  std::map<int, int> batchs;
  batchs[3] = 3;
  GraphDef graph = CreateGraphForYEqualsXSquaredUnknownRank();
  init(cuda_session, {"y"}, batchs);
  ASSERT_FALSE(cuda_session->Create(graph).ok());
}

TEST_F(CudaGraphSessionTest, TestRunAndCudaGraphRunParallel) {
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  std::map<int, int> batchs;
  batchs[3] = 3;
  std::unique_ptr<Session> session(NewSession(options));
  GraphDef graph = CreateGraphForYEqualsXSquared();
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(graph));
  TF_CHECK_OK(session->Create(graph));
  Tensor input(DT_FLOAT, TensorShape({3}));
  float* data = input.flat<float>().data();
  data[0] = 1.0;
  data[1] = 2.0;
  data[2] = 3.0;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  std::vector<std::thread> threads;
  int num_threads = 100;
  std::vector<Tensor> outputs1[num_threads];
  std::vector<Tensor> outputs2[num_threads];
  std::vector<Tensor> outputs3[num_threads];
  std::vector<std::string> output_names = {"y"};
  const int num_infers_per_thread = 100;

  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(TFRun, session.get(), num_infers_per_thread,
                                  &inputs_map, &output_names, &outputs1[i]));
  }
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(CudaGraphRun, cuda_session.get(),
                                  num_infers_per_thread, &inputs_map,
                                  &output_names, &outputs2[i]));
  }
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(TFRun, session.get(), num_infers_per_thread,
                                  &inputs_map, &output_names, &outputs3[i]));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(CudaGraphSessionTest, TestRunAndCaptureParallel) {
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(false);
  std::map<int, int> batchs;
  batchs[3] = 3;
  std::unique_ptr<Session> session(NewSession(options));
  GraphDef graph = CreateGraphForYEqualsXSquared();
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(graph));
  TF_CHECK_OK(session->Create(graph));
  Tensor input(DT_FLOAT, TensorShape({3}));
  float* data = input.flat<float>().data();
  data[0] = 1.0;
  data[1] = 2.0;
  data[2] = 3.0;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  std::vector<std::thread> threads;
  int num_threads = 100;
  std::vector<Tensor> outputs1[num_threads];
  std::vector<Tensor> outputs2[num_threads];
  std::vector<Tensor> outputs3[num_threads];
  std::vector<std::string> output_names = {"y"};
  const int num_infers_per_thread = 1000;
  std::vector<std::unique_ptr<CudaGraphSession>> cuda_sessions(num_threads);
  for (int i = 0; i < num_threads; i++) {
    init(cuda_sessions[i], output_names, batchs);
  }
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(TFRun, session.get(), num_infers_per_thread,
                                  &inputs_map, &output_names, &outputs1[i]));
  }
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(
        std::thread(CudaGraphCreate, cuda_sessions[i].get(), &graph));
  }
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(TFRun, session.get(), num_infers_per_thread,
                                  &inputs_map, &output_names, &outputs3[i]));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(CudaGraphSessionTest, TestBatchSizePadding) {
  std::map<int, int> batchs;
  batchs[10] = 3;
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForYEqualsXSquared()));
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  float* data = input.flat<float>().data();
  data[0] = 1.0f;
  data[1] = 2.2f;
  data[2] = 3.7f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  TF_CHECK_OK(cuda_session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = copyFromGPU(outputs[0]);
  data = result.flat<float>().data();
  EXPECT_FLOAT_EQ(1.0, data[0]);
  EXPECT_FLOAT_EQ(4.84, data[1]);
  EXPECT_FLOAT_EQ(13.69, data[2]);
}

TEST_F(CudaGraphSessionTest, TestBatchDispatcher) {
  std::map<int, int> batchs;
  batchs[10] = 1;
  batchs[50] = 1;
  batchs[30] = 1;
  batchs[100] = 1;
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForYEqualsXSquared()));
  ASSERT_NE((void*)GetCudaGraphContext(25).get(), nullptr);
  ASSERT_NE((void*)GetCudaGraphContext(10).get(), nullptr);
  ASSERT_NE((void*)GetCudaGraphContext(31).get(), nullptr);
  ASSERT_NE((void*)GetCudaGraphContext(80).get(), nullptr);
  ASSERT_EQ((void*)GetCudaGraphContext(101).get(), nullptr);
}

TEST_F(CudaGraphSessionTest, TestForgetToCreate) {
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  std::map<int, int> batchs;
  init(cuda_session, {}, batchs);
  ASSERT_FALSE(cuda_session->Run(inputs_map, {"y"}, {}, &outputs).ok());
}

TEST_F(CudaGraphSessionTest, TestCreateTwice) {
  std::map<int, int> batchs;
  batchs[10] = 1;
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForYEqualsXSquared()));
  ASSERT_FALSE(cuda_session->Create(CreateGraphForYEqualsXSquared()).ok());
}

TEST_F(CudaGraphSessionTest, TestInputsBatchSizeNotConsist) {
  std::map<int, int> batchs;
  batchs[2] = 1;
  init(cuda_session, {"result1", "result2"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForXPlusYThenMultiplyByZ()));
  Tensor x(DT_FLOAT, TensorShape({2}));
  Tensor y(DT_FLOAT, TensorShape({1}));
  Tensor z(DT_FLOAT, TensorShape({1}));
  std::vector<Tensor> outputs;
  float* dx = x.flat<float>().data();
  float* dy = y.flat<float>().data();
  float* dz = z.flat<float>().data();
  dx[0] = 1.0f;
  dx[1] = 2.0f;
  dy[0] = 3.0f;
  dz[0] = 5.0f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  inputs_map.push_back(std::make_pair("y", y));
  inputs_map.push_back(std::make_pair("z", y));
  ASSERT_FALSE(
      cuda_session->Run(inputs_map, {"result1", "result2"}, {}, &outputs).ok());
}

TEST_F(CudaGraphSessionTest, TestLackInput) {
  std::map<int, int> batchs;
  batchs[2] = 1;
  init(cuda_session, {"result1", "result2"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForXPlusYThenMultiplyByZ()));
  Tensor x(DT_FLOAT, TensorShape({2}));
  std::vector<Tensor> outputs;
  float* dx = x.flat<float>().data();
  dx[0] = 1.0f;
  dx[1] = 2.0f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  ASSERT_FALSE(
      cuda_session->Run(inputs_map, {"result1", "result2"}, {}, &outputs).ok());
}

TEST_F(CudaGraphSessionTest, TestInputTypeNotConsist) {
  std::map<int, int> batchs;
  batchs[2] = 1;
  init(cuda_session, {"result1", "result2"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForXPlusYThenMultiplyByZ()));
  Tensor x(DT_FLOAT, TensorShape({2}));
  Tensor y(DT_INT64, TensorShape({2}));
  Tensor z(DT_FLOAT, TensorShape({2}));
  std::vector<Tensor> outputs;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  inputs_map.push_back(std::make_pair("y", y));
  inputs_map.push_back(std::make_pair("z", y));
  ASSERT_FALSE(
      cuda_session->Run(inputs_map, {"result1", "result2"}, {}, &outputs).ok());
}

TEST_F(CudaGraphSessionTest, TestBatchSizeUnable) {
  SessionOptions options;
  // std::unique_ptr<CudaGraphSession> cuda_session(new
  // CudaGraphSession(options));
  std::map<int, int> batchs;
  // batch size for capture does not satisfy.
  batchs[1] = 3;
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForYEqualsXSquared()));
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  float* data = input.flat<float>().data();
  data[0] = 1.0f;
  data[1] = 2.2f;
  data[2] = 3.7f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  ASSERT_FALSE(cuda_session->Run(inputs_map, {"y"}, {}, &outputs).ok());
}

TEST_F(CudaGraphSessionTest, TestBackUp) {
  SessionOptions options;
  // std::unique_ptr<CudaGraphSession> cuda_session(new
  // CudaGraphSession(options));
  std::map<int, int> batchs;
  // batch size for capture does not satisfy.
  batchs[1] = 3;
  init(cuda_session, {"y"}, batchs);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForYEqualsXSquared()));
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  float* data = input.flat<float>().data();
  data[0] = 1.0f;
  data[1] = 2.2f;
  data[2] = 3.7f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  RunMetadata runMetadata;
  RunOptions runOptions;
  runOptions.set_cuda_graph_use_back_up(true);
  TF_CHECK_OK(cuda_session->Run(runOptions, inputs_map, {"y"}, {}, &outputs,
                                &runMetadata));
  ASSERT_TRUE(runMetadata.cuda_graph_fallback_used());
}

TEST_F(CudaGraphSessionTest, TestOutputOnCPU) {
  std::map<int, int> batchs;
  batchs[3] = 3;
  init(cuda_session, {"y"}, batchs, true);
  TF_CHECK_OK(cuda_session->Create(CreateGraphForYEqualsXSquared()));
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  float* data = input.flat<float>().data();
  data[0] = 1.0f;
  data[1] = 2.2f;
  data[2] = 3.7f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  TF_CHECK_OK(cuda_session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = outputs[0];
  data = result.flat<float>().data();
  EXPECT_FLOAT_EQ(1.0, data[0]);
  EXPECT_FLOAT_EQ(4.84, data[1]);
  EXPECT_FLOAT_EQ(13.69, data[2]);
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
