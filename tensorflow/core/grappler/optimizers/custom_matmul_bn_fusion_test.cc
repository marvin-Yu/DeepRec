#include "tensorflow/core/grappler/optimizers/custom_matmul_bn_fusion.h"

#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace tensorflow {
namespace grappler {

class CustomMatMulBNFusionTest : public GrapplerTest {};

TEST_F(CustomMatMulBNFusionTest, CustomMatMulBNFusionOp) {
  using test::function::NDef;

  CustomMatMulBNFusion optimizer;

  const Tensor input = test::AsTensor<float>({1.1, 0.9, 1.2, 0.8}, {2, 2});
  const Tensor weight = test::AsTensor<float>({1.1, 0.9, 1.2, 0.8}, {2, 2});
  const Tensor toshape = test::AsTensor<int>({-1, 1, 1, 2});
  const Tensor scaler = test::AsTensor<float>({1.2, 0.8});
  const Tensor offset = test::AsTensor<float>({1.2, 0.8});

  //   const auto scalar = PartialTensorShape({-1, 2});

  GrapplerItem item;
  item.graph = test::function::GDef({

      NDef("input", "Const", {}, {{"T", DT_FLOAT}, {"value", input}}),
      NDef("weight", "Const", {}, {{"T", DT_FLOAT}, {"value", weight}}),
      NDef("scaler", "Const", {}, {{"T", DT_FLOAT}, {"value", scaler}}),
      NDef("offset", "Const", {}, {{"T", DT_FLOAT}, {"value", offset}}),
      NDef("toshape", "Const", {}, {{"T", DT_INT32}, {"value", toshape}}),

      NDef("matmul", "MatMul", {"input", "weight"},
           {{"T", DT_FLOAT}, {"transpose_a", false}, {"transpose_b", false}}),
      NDef("shape", "Shape", {"matmul"},
           {{"T", DT_FLOAT}, {"out_type", DT_INT32}}),
      NDef("reshape_1", "Reshape", {"matmul", "toshape"},
           {{"T", DT_FLOAT}, {"Tshape", DT_INT32}}),
      NDef("mul", "Mul", {"reshape_1", "scaler"}, {{"T", DT_FLOAT}}),
      NDef("add", "Add", {"mul", "offset"}, {{"T", DT_FLOAT}}),
      NDef("reshape_2", "Reshape", {"add", "shape"},
           {{"T", DT_FLOAT}, {"Tshape", DT_INT32}}),
      NDef("leakyrelu", "LeakyRelu", {"reshape_2"},
           {{"alpha", 0.2}, {"T", DT_FLOAT}}),
  });

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(/*cluster=*/nullptr, item, &output));

  GraphDef expected = test::function::GDef({
      NDef("input", "Const", {}, {{"T", DT_FLOAT}, {"value", input}}),
      NDef("weight", "Const", {}, {{"T", DT_FLOAT}, {"value", weight}}),
      NDef("scaler", "Const", {}, {{"T", DT_FLOAT}, {"value", scaler}}),
      NDef("offset", "Const", {}, {{"T", DT_FLOAT}, {"value", offset}}),
      NDef("toshape", "Const", {}, {{"T", DT_INT32}, {"value", toshape}}),


      NDef("matmul", "MatMul", {"input", "mul"},
           {{"T", DT_FLOAT}, {"transpose_a", false}, {"transpose_b", false}}),
      NDef("mul", "Mul", {"weight", "scaler"}, {{"T", DT_FLOAT}}),
      NDef("add", "BiasAdd", {"matmul", "offset"}, {{"T", DT_FLOAT}}),
      NDef("leakyrelu", "LeakyRelu", {"add"},
           {{"alpha", 0.2}, {"T", DT_FLOAT}}),

  });

  CompareGraphs(expected, output);
}

}  // namespace grappler
}  // namespace tensorflow