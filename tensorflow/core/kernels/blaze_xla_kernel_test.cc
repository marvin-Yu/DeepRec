#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/env_var.h"


namespace tensorflow {
namespace {
string cpu = "/CPU:0";
string gpu = "/GPU:0";
class BlazeXlaKernelTest : public OpsTestBase {
 protected:
  void MakeOp(
    std::initializer_list<DataType> t1,
    std::initializer_list<DataType> t2,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    const std::string &graph_def,
    const std::string &blaze_option_path,
    string device = "/CPU:0") {
#if GOOGLE_CUDA
    if (device == gpu) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                        "GPU", {}, "")));
    }
#endif
    TF_ASSERT_OK(NodeDefBuilder("BlazeXlaOp", "BlazeXlaOp")
      .Input(FakeInput(t1))
      .Attr("InT", t1)
      .Attr("OutT", t2)
      .Attr("input_names", input_names)
      .Attr("output_names", output_names)
      .Attr("graph_def", graph_def)
      .Attr("blaze_option_path", blaze_option_path)
      .Finalize(node_def()));
    node_def()->set_device(device);
    std::cout << node_def()->DebugString() << std::endl;
  TF_ASSERT_OK(InitOp());
}

//std::initializer_list<DataType> t1({DT_INT32, DT_INT32});
//std::initializer_list<DataType> t2({DT_INT32});
std::vector<std::string> input_names = {"x", "y"};
std::vector<std::string> output_names = {"result"};

  template <typename T>
  void APLUSBNORM(string device="/CPU:0") {
    std::string succ_pb = "core/kernels/blaze_test_data/gpu_aplusb.pbtxt";
    std::string options = "core/kernels/blaze_test_data/options";

    string filename = io::JoinPath(testing::TensorFlowSrcRoot(), succ_pb);
    string blaze_options = io::JoinPath(testing::TensorFlowSrcRoot(), options);

    GraphDef gdef;
    TF_ASSERT_OK(ReadTextProto(Env::Default(), filename, &gdef));

    MakeOp({DT_INT64, DT_INT64}, {DT_INT64}, input_names,
           output_names, filename, blaze_options, device);
    AddInputFromArray<T>(TensorShape({2, 2}),
                         {1, 2, 3, 4});
    AddInputFromArray<T>(TensorShape({2, 2}), {5, 6, 7, 8});

    TF_ASSERT_OK(RunOpKernel());

    // Check the new state of the input
    Tensor* params_tensor = GetOutput(0);
    Tensor expected(allocator(), DataTypeToEnum<T>::value,
                    TensorShape({2, 2}));
    // Should become
    // [[[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]]
    //  [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]]
    test::FillValues<T>(&expected,
                        {6, 8, 10, 12});
    test::ExpectTensorEqual<T>(expected, *params_tensor);
  }

const char* const kCacheKey = "TF_XLA_PTX_CACHE_DIR";
const char* const kValue = "/tmp";

  template <typename T>
  void APLUSBXLA(string device="/CPU:0") {
    ASSERT_EQ(0, setenv(kCacheKey, kValue, 0));

    string ptx_cache_dir;
    ReadStringFromEnvVar("TF_XLA_PTX_CACHE_DIR", "",
                         &ptx_cache_dir);
    ASSERT_FALSE(ptx_cache_dir.empty());
    std::string succ_pb = "core/kernels/blaze_test_data/gpu_aplusb.pbtxt";
    std::string options = "core/kernels/blaze_test_data/succ_options";

    string filename = io::JoinPath(testing::TensorFlowSrcRoot(), succ_pb);
    string blaze_options = io::JoinPath(testing::TensorFlowSrcRoot(), options);

    GraphDef gdef;
    TF_ASSERT_OK(ReadTextProto(Env::Default(), filename, &gdef));

    MakeOp({DT_INT64, DT_INT64}, {DT_INT64}, input_names,
           output_names, filename, blaze_options);
    AddInputFromArray<T>(TensorShape({2, 2}),
                         {1, 2, 3, 4});
    AddInputFromArray<T>(TensorShape({2, 2}), {5, 6, 7, 8});

    std::shared_ptr<UserTracedInfos> ptr = std::make_shared<UserTracedInfos>(true, true);
    TF_ASSERT_OK(RunOpKernel(ptr));
    ASSERT_NE(0, ptr->prof_stats->blaze_latency_ms);
    ASSERT_NE(nullptr, ptr->traced_tensors);
    ASSERT_EQ(3, ptr->traced_tensors->name_tensors_size());
    
    // Check the new state of the input
    Tensor* params_tensor = GetOutput(0);
    Tensor expected(allocator(), DataTypeToEnum<T>::value,
                    TensorShape({2, 2}));
    // Should become
    // [[[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]]
    //  [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]]
    test::FillValues<T>(&expected,
                        {6, 8, 10, 12});
    test::ExpectTensorEqual<T>(expected, *params_tensor);

    Tensor i1(allocator(), DataTypeToEnum<T>::value,
                    TensorShape({2, 2}));
    test::FillValues<T>(&i1,
                        {1, 2, 3, 4});

    Tensor i2(allocator(), DataTypeToEnum<T>::value,
                    TensorShape({2, 2}));
    test::FillValues<T>(&i2,
                        {5, 6, 7, 8});

    std::vector<Tensor> tensors = {i1, i2, expected};

    std::vector<std::string> trace_names = {"x", "y", "result"};
    for (int i = 0; i < ptr->traced_tensors->name_tensors_size(); ++i) {
      const auto& name_tensor = ptr->traced_tensors->name_tensors(i);
      const auto& name = name_tensor.name();
      ASSERT_EQ(name, trace_names[i]);
      Tensor tensor;
      tensor.FromProto(name_tensor.tensor());
       test::ExpectTensorEqual<T>(tensor, tensors[i]);   
    }
  }
};

TEST_F(BlazeXlaKernelTest, NormRun) { APLUSBNORM<int64>(); }
TEST_F(BlazeXlaKernelTest, PaddingRun) { APLUSBXLA<int64>(); }

#if GOOGLE_CUDA
//fixme, i do not know why  this core dump
//TEST_F(BlazeXlaKernelTest, NormRunGPU) { APLUSBNORM<int64>(gpu); }
TEST_F(BlazeXlaKernelTest, PaddingRunGPU) { APLUSBXLA<int64>(gpu); }
#endif
}
}

