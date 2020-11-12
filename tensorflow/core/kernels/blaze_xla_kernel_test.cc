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


namespace tensorflow {
namespace {

class BlazeXlaKernelTest : public OpsTestBase {
 protected:
  void MakeOp(
    std::initializer_list<DataType> t1,
    std::initializer_list<DataType> t2,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    const std::string &graph_def,
    const std::string &blaze_option_path) {
    TF_ASSERT_OK(NodeDefBuilder("BlazeXlaOp", "BlazeXlaOp")
      .Input(FakeInput(t1))
      .Attr("InT", t1)
      .Attr("OutT", t2)
      .Attr("input_names", input_names)
      .Attr("output_names", output_names)
      .Attr("graph_def", graph_def)
      .Attr("blaze_option_path", blaze_option_path)
      .Finalize(node_def()));
    node_def()->set_device("/CPU:0");
    std::cout << node_def()->DebugString() << std::endl;
  TF_ASSERT_OK(InitOp());
}

//std::initializer_list<DataType> t1({DT_INT32, DT_INT32});
//std::initializer_list<DataType> t2({DT_INT32});
std::vector<std::string> input_names = {"x", "y"};
std::vector<std::string> output_names = {"result"};

  template <typename T>
  void APLUSBNORM() {
    std::string succ_pb = "core/kernels/blaze_test_data/aplusb.pbtxt";
    std::string options = "core/kernels/blaze_test_data/options";

    string filename = io::JoinPath(testing::TensorFlowSrcRoot(), succ_pb);
    string blaze_options = io::JoinPath(testing::TensorFlowSrcRoot(), options);

    GraphDef gdef;
    TF_ASSERT_OK(ReadTextProto(Env::Default(), filename, &gdef));

    MakeOp({DT_INT32, DT_INT32}, {DT_INT32}, input_names,
           output_names, gdef.DebugString(), blaze_options);
    // Feed and run
    // [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    //  [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
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

  template <typename T>
  void APLUSBXLA() {
    std::string succ_pb = "core/kernels/blaze_test_data/aplusb.pbtxt";
    std::string options = "core/kernels/blaze_test_data/succ_options";

    string filename = io::JoinPath(testing::TensorFlowSrcRoot(), succ_pb);
    string blaze_options = io::JoinPath(testing::TensorFlowSrcRoot(), options);

    GraphDef gdef;
    TF_ASSERT_OK(ReadTextProto(Env::Default(), filename, &gdef));

    MakeOp({DT_INT32, DT_INT32}, {DT_INT32}, input_names,
           output_names, gdef.DebugString(), blaze_options);
    // Feed and run
    // [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    //  [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
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
};

TEST_F(BlazeXlaKernelTest, NormRun) { APLUSBNORM<int32>(); }
TEST_F(BlazeXlaKernelTest, PaddingRun) { APLUSBXLA<int32>(); }
}
}

