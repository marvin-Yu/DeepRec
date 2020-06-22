#include "tensorflow/core/grappler/optimizers/unfuse_batch_norm_optimizer.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include <iostream>

namespace tensorflow {
namespace grappler {
namespace {

class UnfuseBatchNormOptimizerTest : public ::testing::Test {};

TEST_F(UnfuseBatchNormOptimizerTest, FusedBatchNorm) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output inputs = ops::Placeholder(s.WithOpName("inputs"), DT_FLOAT,
            ops::Placeholder::Shape({150, 20}));
    Output weights1 = ops::Const(s.WithOpName("weights1"), 5.0f, {20, 30});
    Output nn_output = ops::MatMul(s.WithOpName("matmul"), inputs, weights1);
    Output reshaped = ops::Reshape(s.WithOpName("reshape1"), nn_output,
                                   ops::Const(s.WithOpName("shape"), Input::Initializer({150, 1, 1, 30}, {4})));

    Output scale = ops::Const(s.WithOpName("scale"), Input::Initializer(1.0f, {30}));
    Output offset = ops::Const(s.WithOpName("offset"), Input::Initializer(2.0f, {30}));
    Output mean = ops::Const(s.WithOpName("mean"), Input::Initializer(3.0f, {30}));
    Output variance = ops::Const(s.WithOpName("variance"), Input::Initializer(4.0f, {30}));
    ops::FusedBatchNorm::Attrs attrs;
    attrs.is_training_ = false;
    attrs.epsilon_ = 0.03;
    Output bn_output = ops::FusedBatchNorm(s.WithOpName("bn"), reshaped, scale, offset, mean, variance, attrs).y;
    Output reshaped2 = ops::Reshape(s.WithOpName("reshape2"), bn_output,
                                    ops::Const(s.WithOpName("shape"), Input::Initializer({150, 30}, {4})));

    Output outputs = ops::Identity(s.WithOpName("outputs"), reshaped2);

    GrapplerItem item;
    item.fetch = {"outputs"};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));
    GraphDef output;
    TF_EXPECT_OK(UnfuseBatchNormOptimizer(nullptr).Optimize(nullptr, item, &output));
    item.graph = output;
    TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));
    item.graph = output;
    std::cout << output.DebugString() << std::endl;
    NodeMap node_map(&output);

    auto matmul = node_map.GetNode("weights1_matmul_0");
    ASSERT_TRUE(matmul);
    CHECK_EQ(matmul->op(), "MatMul");
    CHECK_EQ(matmul->input(0), "inputs");
    CHECK_EQ(matmul->input(1), "weights1_mul_0");
    auto biasadd = node_map.GetNode("bn_bias_add_0");
    ASSERT_TRUE(biasadd);
    CHECK_EQ(biasadd->op(), "BiasAdd");
    CHECK_EQ(biasadd->input(0), "weights1_matmul_0");
    CHECK_EQ(biasadd->input(1), "bn_final_bias_0");
    float scale_value = 1 / sqrt(4 + 0.03) * 1.0;
    EXPECT_FLOAT_EQ(5 * scale_value, node_map.GetNode("weights1_mul_0")->attr().at("value").tensor().float_val(0));
    EXPECT_FLOAT_EQ(2.0 - scale_value * 3, node_map.GetNode("bn_final_bias_0")->attr().at("value").tensor().float_val(0));

}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
