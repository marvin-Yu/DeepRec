#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/blaze_xla_predictor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {
class BlazeXlaPredictorTest {
 public:
  BlazeXlaPredictorTest(std::string& graph_path, std::string& blaze_opt_path):
      graph_def_path_(graph_path), blaze_options_path_(blaze_opt_path) {
  }

  ~BlazeXlaPredictorTest() {
    if (constr_) {
      delete constr_;
    }
  }

  static NodeDef MakeBlazeNodeDef(std::initializer_list<DataType> t1,
                      std::initializer_list<DataType> t2,
                      std::vector<std::string> input_names,
                      std::vector<std::string> output_names,
                      const std::string &graph_def,
                      const std::string &blaze_option_path);

  Status GeneOpKernelConstruction(DeviceType device_type, NodeDef& node_def);

  OpKernelConstruction* GetConstruction() {
    return constr_;
  }
 private:
  std::string graph_def_path_;
  std::string blaze_options_path_;

  OpKernelConstruction *constr_;
};

NodeDef BlazeXlaPredictorTest::MakeBlazeNodeDef(
    std::initializer_list<DataType> t1,
    std::initializer_list<DataType> t2,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    const std::string &graph_def,
    const std::string &blaze_option_path) {
  NodeDefBuilder builder("BlazeXlaOp", "BlazeXlaOp");
  NodeDef node_def;
  builder
      .Input(FakeInput(t1))
      .Attr("InT", t1)
      .Attr("OutT", t2)
      .Attr("input_names", input_names)
      .Attr("output_names", output_names)
      .Attr("graph_def", graph_def)
      .Attr("blaze_option_path", blaze_option_path)
      .Finalize(&node_def);
  node_def.set_device("/CPU:0");
  return node_def;
}

Status BlazeXlaPredictorTest::GeneOpKernelConstruction(DeviceType device_type,
                                                    NodeDef& node_def) {
  const OpDef* op_def = nullptr;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def));
  // Validate node_def against OpDef.
  TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, *op_def));

  DataTypeVector inputs;
  DataTypeVector outputs;
  Status s;
  s.Update(InOutTypesForNode(node_def, *op_def, &inputs, &outputs));
  TF_RETURN_IF_ERROR(s);

  // We are creating a kernel for an op registered in
  // OpRegistry::Global(), we consult the kernel registry to decide
  // the kernel's input and output memory types.
  MemoryTypeVector input_memory_types;
  MemoryTypeVector output_memory_types;
  TF_RETURN_IF_ERROR(MemoryTypesForNode(OpRegistry::Global(), device_type,
                                        node_def, &input_memory_types,
                                        &output_memory_types));

  Status status;
  DeviceBase device(Env::Default());
  constr_ = new OpKernelConstruction(device_type, &device, cpu_allocator(),
                           &node_def, op_def, nullptr, inputs, input_memory_types,
                           outputs, output_memory_types, TF_GRAPH_DEF_VERSION, &status);
  return status;
}

Status TestWithMutConf(std::string& pb, std::string& options) {
  BlazeXlaPredictorTest test(pb, options);

  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), pb);
  string blaze_options = io::JoinPath(testing::TensorFlowSrcRoot(), options);

  GraphDef gdef;
  (ReadTextProto(Env::Default(), filename, &gdef));
  NodeDef node_def = BlazeXlaPredictorTest::MakeBlazeNodeDef({DT_INT32, DT_INT32}, {DT_INT32},
                                                          {"x", "y"}, {"result"}, gdef.DebugString(),
                                                          blaze_options);
  TF_RETURN_IF_ERROR(test.GeneOpKernelConstruction(DEVICE_CPU, node_def));

  BlazeXlaPredictor predictor(test.GetConstruction());
  return predictor.InitSession();
}

TEST(TestBlazeXlaPredictor, TestErrorInit) {
  {
    // no warmup conf
    std::string succ_pb = "core/kernels/blaze_test_data/aplusb.pbtxt";
    std::string options = "core/kernels/blaze_test_data/options";

    ASSERT_FALSE(TestWithMutConf(succ_pb, options).ok());
  }

  {
    //negative warmup value
    std::string succ_pb = "core/kernels/blaze_test_data/aplusb.pbtxt";
    std::string options = "core/kernels/blaze_test_data/negative_options";

    auto st = TestWithMutConf(succ_pb, options);

    ASSERT_FALSE(st.ok());
  }

  {
    // error input
    std::string succ_pb = "core/kernels/blaze_test_data/error.pbtxt";
    std::string options = "core/kernels/blaze_test_data/succ_options";

    auto st = TestWithMutConf(succ_pb, options);

    ASSERT_FALSE(st.ok());
  }
}

TEST(TestBlazeXlaPredictor, TestSuccInit) {
  {
    // error input
    std::string succ_pb = "core/kernels/blaze_test_data/aplusb.pbtxt";
    std::string options = "core/kernels/blaze_test_data/succ_options";

    auto st = TestWithMutConf(succ_pb, options);

    ASSERT_TRUE(st.ok());
  }
}

TEST(TestBlazeXlaPredictor, TestRun) {
  std::string pb = "core/kernels/blaze_test_data/aplusb.pbtxt";
  std::string options = "core/kernels/blaze_test_data/succ_options";
  BlazeXlaPredictorTest test(pb, options);

  string filename = io::JoinPath(testing::TensorFlowSrcRoot(), pb);
  string blaze_options = io::JoinPath(testing::TensorFlowSrcRoot(), options);

  GraphDef gdef;
  (ReadTextProto(Env::Default(), filename, &gdef));
  NodeDef node_def = BlazeXlaPredictorTest::MakeBlazeNodeDef({DT_INT32, DT_INT32}, {DT_INT32},
                                                          {"x", "y"}, {"result"}, gdef.DebugString(),
                                                          blaze_options);
  TF_ASSERT_OK(test.GeneOpKernelConstruction(DEVICE_CPU, node_def));

  BlazeXlaPredictor predictor(test.GetConstruction());
  TF_ASSERT_OK(predictor.InitSession());
 {
    Status status;
    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

   // thread::ThreadPool threadpool(Env::Default(), "test", num_threads);
   // Eigen::ThreadPoolDevice eigen_cpu_device(threadpool.AsEigenThreadPool(),
   //                                          num_threads);
   //  device->set_eigen_cpu_device(&eigen_cpu_device);
    std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                                cpu_allocator(), node_def,
                                                TF_GRAPH_DEF_VERSION, &status));
    TF_ASSERT_OK(status);

    OpKernelContext::Params params;
    params.op_kernel = op.get();
    gtl::InlinedVector<TensorValue, 4> inputs;
    int bs = 8;
    TensorShape shape1({bs});
    Tensor input1(DT_INT32, shape1);
    test::FillIota<int>(&input1, 1);
    inputs.push_back({nullptr, &input1});
   // params.op_kernel->input_memory_types()[0] = HOST_MEMORY;

    Tensor input2(DT_INT32, shape1);
    test::FillIota<int>(&input2, 2);
    inputs.push_back({nullptr, &input2});
   // params.op_kernel->input_memory_types()[1] = HOST_MEMORY;
    params.device = device.get();
    params.frame_iter = FrameAndIter(0, 0);
    params.inputs = &inputs;
    params.op_device_context = new DeviceContext;
   // params->op_kernel->output_memory_types()[0] = HOST_MEMORY;
    std::vector<AllocatorAttributes> attrs;
    test::SetOutputAttrs(&params, &attrs);

    std::unique_ptr<OpKernelContext> predictor_context(
        new OpKernelContext(&params));
    /*
    predictor.Compute(predictor_context.get());
    TF_ASSERT_OK(predictor_context->status());
    ASSERT_EQ(predictor_context->num_outputs(), 1);
    auto output = predictor_context->mutable_output(0);
    ASSERT_NE(nullptr, output);
    Tensor expected(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&expected, {3, 5, 7, 9, 11, 13, 15, 17});
    test::ExpectTensorEqual<int32>(expected, *output); */
  } 
}
}
}
