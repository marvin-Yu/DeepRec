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
#include "tensorflow/core/kernels/blaze_predictor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {
NodeDef MakeBlazeNodeDef(std::initializer_list<DataType> t1,
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

TEST(BlazePredictorCPUTest, CPUTest) {
  // Test constructor
  string filename = io::JoinPath(testing::TensorFlowSrcRoot(),
                                 "core/kernels/blaze_test_data/aplusb.pbtxt");
  string blaze_options = io::JoinPath(testing::TensorFlowSrcRoot(),
                                 "core/kernels/blaze_test_data/options");

  std::cout << "caixukun\n";
  std::cout << filename << std::endl;
  GraphDef gdef;
  TF_ASSERT_OK(ReadTextProto(Env::Default(), filename, &gdef));
  NodeDef node_def = MakeBlazeNodeDef({DT_INT32, DT_INT32}, {DT_INT32},
                                      {"x", "y"}, {"z"}, gdef.DebugString(),
                                      blaze_options);
  // Look up the Op registered for this op name.
  const OpDef* op_def = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def);
  TF_ASSERT_OK(s);

  // Validate node_def against OpDef.
  s = ValidateNodeDef(node_def, *op_def);
  TF_ASSERT_OK(s);

  DataTypeVector inputs;
  DataTypeVector outputs;
  s.Update(InOutTypesForNode(node_def, *op_def, &inputs, &outputs));
  TF_ASSERT_OK(s);

  // We are creating a kernel for an op registered in
  // OpRegistry::Global(), we consult the kernel registry to decide
  // the kernel's input and output memory types.
  MemoryTypeVector input_memory_types;
  MemoryTypeVector output_memory_types;
  TF_ASSERT_OK(MemoryTypesForNode(OpRegistry::Global(), DEVICE_CPU,
                                        node_def, &input_memory_types,
                                        &output_memory_types));

  Status status;
  DeviceBase device(Env::Default());
  OpKernelConstruction ctx(DEVICE_CPU, &device, cpu_allocator(),
                           &node_def, op_def, nullptr, inputs, input_memory_types,
                           outputs, output_memory_types, TF_GRAPH_DEF_VERSION, &status);
  TF_ASSERT_OK(status);
  
  BlazePredictor predictor(&ctx);

  //computing test
  {
    int num_threads = 2;
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

    gtl::InlinedVector<TensorValue, 4> inputs;
    int bs = 10;
    TensorShape shape1({bs});
    Tensor input1(DT_INT32, shape1);
    test::FillIota<int>(&input1, 1);
    inputs.push_back({nullptr, &input1});

    Tensor input2(DT_INT32, shape1);
    test::FillIota<int>(&input2, 2);
    OpKernelContext::Params params;
    params.device = device.get();
    params.frame_iter = FrameAndIter(0, 0);
    params.inputs = &inputs;
    params.op_kernel = op.get();
    std::vector<AllocatorAttributes> attrs;
    test::SetOutputAttrs(&params, &attrs);
  }
}
}
}
