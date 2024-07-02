/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/remapper.h"

#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/util.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace grappler {

class RemapperTest : public GrapplerTest {
 protected:
  void SetUp() override {
    // This is a requirement for fusing FusedBatchNorm + SideInput + Activation.
    setenv("TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT", "1", 1 /* replace */);
  }
};

TEST_F(RemapperTest, FusedBatchNorm) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output dflt = ops::Const(s.WithOpName("dflt"), {3.14f, 2.7f}, {2, 1, 1, 1});
  Output x = ops::PlaceholderWithDefault(s.WithOpName("x"), dflt, {2, 1, 1, 1});
  Output scale = ops::Const(s.WithOpName("scale"), {0.3f}, {1});
  Output offset = ops::Const(s.WithOpName("offset"), {0.123f}, {1});
  Output mean = ops::Const(s.WithOpName("mean"), {7.3f}, {1});
  Output variance = ops::Const(s.WithOpName("variance"), {0.57f}, {1});
  ops::FusedBatchNorm::Attrs attr;
  attr = attr.IsTraining(false);
  ops::FusedBatchNorm bn(s.WithOpName("batch_norm"), x, scale, offset, mean,
                         variance, attr);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"batch_norm"};

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(RemapperTest, FusedBatchNormNCHW) {
#if !GOOGLE_CUDA
  GTEST_SKIP() << "CUDA is not enabled";
#endif  // !GOOGLE_CUDA
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output dflt =
      ops::Const(s.WithOpName("dflt"), {3.14f, 2.7f, 1.0f, 2.0f, 3.0f, 100.0f},
                 {1, 3, 1, 2});
  Output x = ops::PlaceholderWithDefault(s.WithOpName("x"), dflt, {1, 3, 1, 2});
  Output scale = ops::Const(s.WithOpName("scale"), {0.3f, 7.0f, 123.0f}, {3});
  Output offset =
      ops::Const(s.WithOpName("offset"), {0.123f, 2.1f, 0.55f}, {3});
  Output mean = ops::Const(s.WithOpName("mean"), {7.3f, 8.3f, 3.1f}, {3});
  Output variance =
      ops::Const(s.WithOpName("variance"), {0.57f, 1.0f, 2.0f}, {3});
  ops::FusedBatchNorm::Attrs attr;
  attr = attr.IsTraining(false);
  attr = attr.DataFormat("NCHW");
  ops::FusedBatchNorm bn(s.WithOpName("batch_norm").WithDevice("/device:GPU:0"),
                         x, scale, offset, mean, variance, attr);

  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"batch_norm"};

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;

  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  if (GetNumAvailableGPUs() > 0) {
    // NCHW batch norm is only supported on GPU.
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch);
    ASSERT_EQ(tensors.size(), 1);
    test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-3);
  }
}

TEST_F(RemapperTest, FuseBatchNormWithRelu) {
  if (IsMKLEnabled()) GTEST_SKIP() << "Fusion not available with oneDNN.";
  using ::tensorflow::ops::Placeholder;

  for (bool is_training : {true, false}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

#if !defined(GOOGLE_CUDA) || !(CUDNN_VERSION >= 7402)
    if (is_training) {
      LOG(INFO) << "Skip FuseBatchNormWithRelu"
                << "[is_training=" << is_training << "] "
                << "test. It requires CUDNN_VERSION >= 7402.";
      continue;
    }
#endif

#if !defined(GOOGLE_CUDA)
    if (!is_training) {
      LOG(INFO) << "Skip FuseBatchNormWithRelu"
                << "[is_training=" << is_training << "]";
      continue;
    }
#endif

    const int num_channels = 24;

    TensorShape channel_shape({num_channels});
    TensorShape empty_shape({0});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT,
                             ops::Placeholder::Shape({2, 8, 8, num_channels}));
    auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_HALF);
    auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT);
    auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT);
    auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT);
    auto var = Placeholder(s.WithOpName("var"), DT_FLOAT);

    float epsilon = 0.1f;
    auto fbn = ops::FusedBatchNormV3(
        s.WithOpName("fused_batch_norm"), input_cast, scale, offset, mean, var,
        ops::FusedBatchNormV3::IsTraining(is_training)
            .Epsilon(epsilon)
            .DataFormat("NHWC"));
    auto relu = ops::Relu(s.WithOpName("relu"), fbn.y);
    auto fetch = ops::Identity(s.WithOpName("fetch"), relu);

    auto input_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});
    auto scale_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
    auto offset_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
    auto mean_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                             : channel_shape);
    auto var_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                            : channel_shape);

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t},
                 {"scale", scale_t},
                 {"offset", offset_t},
                 {"mean", mean_t},
                 {"var", var_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on GPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:GPU:0");
    }

    Remapper optimizer(RewriterConfig::AGGRESSIVE);  // trust placeholders shape
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "relu") {
        EXPECT_EQ(node.op(), "Identity");
        ASSERT_EQ(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "fused_batch_norm");
        found++;
      }
      if (node.name() == "fused_batch_norm") {
        EXPECT_EQ(node.op(), "_FusedBatchNormEx");
        ASSERT_EQ(node.input_size(), 5);
        EXPECT_EQ(node.input(0), "input_cast");
        EXPECT_EQ(node.input(1), "scale");
        EXPECT_EQ(node.input(2), "offset");
        EXPECT_EQ(node.input(3), "mean");
        EXPECT_EQ(node.input(4), "var");

        auto attr = node.attr();
        EXPECT_EQ(attr["num_side_inputs"].i(), 0);
        EXPECT_EQ(attr["activation_mode"].s(), "Relu");
        found++;
      }
    }
    EXPECT_EQ(found, 2);

    if (GetNumAvailableGPUs() > 0) {
      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, /*rtol=*/1e-2);
    }
  }
}

TEST_F(RemapperTest, FuseBatchNormWithAddAndRelu) {
  if (IsMKLEnabled()) GTEST_SKIP() << "Fusion not available with oneDNN.";
  using ::tensorflow::ops::Placeholder;

  for (bool is_training : {true, false}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

#if !defined(GOOGLE_CUDA) || !(CUDNN_VERSION >= 7402)
    if (is_training) {
      LOG(INFO) << "Skip FuseBatchNormWithAddAndRelu"
                << "[is_training=" << is_training << "] "
                << "test. It requires CUDNN_VERSION >= 7402.";
      continue;
    }
#endif

#if !defined(GOOGLE_CUDA)
    if (!is_training) {
      LOG(INFO) << "Skip FuseBatchNormWithAddAndRelu"
                << "[is_training=" << is_training << "]";
      continue;
    }
#endif

    const int num_channels = 24;

    TensorShape input_shape({2, 8, 8, num_channels});
    TensorShape channel_shape({num_channels});
    TensorShape empty_shape({0});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT,
                             ops::Placeholder::Shape(input_shape));
    auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_HALF);
    auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT);
    auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT);
    auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT);
    auto var = Placeholder(s.WithOpName("var"), DT_FLOAT);
    auto side_input = Placeholder(s.WithOpName("side_input"), DT_FLOAT,
                                  ops::Placeholder::Shape(input_shape));
    auto side_input_cast =
        ops::Cast(s.WithOpName("side_input_cast"), side_input, DT_HALF);

    float epsilon = 0.1f;
    auto fbn = ops::FusedBatchNormV3(
        s.WithOpName("fused_batch_norm"), input_cast, scale, offset, mean, var,
        ops::FusedBatchNormV3::IsTraining(is_training)
            .Epsilon(epsilon)
            .DataFormat("NHWC"));
    auto add = ops::Add(s.WithOpName("add"), fbn.y, side_input_cast);
    auto relu = ops::Relu(s.WithOpName("relu"), add);
    auto fetch = ops::Identity(s.WithOpName("fetch"), relu);

    auto input_t = GenerateRandomTensor<DT_FLOAT>(input_shape);
    auto scale_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
    auto offset_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
    auto mean_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                             : channel_shape);
    auto var_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                            : channel_shape);
    auto side_input_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t},   {"scale", scale_t},
                 {"offset", offset_t}, {"mean", mean_t},
                 {"var", var_t},       {"side_input", side_input_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on GPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:GPU:0");
    }

    Remapper optimizer(RewriterConfig::AGGRESSIVE);  // trust placeholders shape
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "relu") {
        EXPECT_EQ(node.op(), "Identity");
        ASSERT_EQ(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "fused_batch_norm");
        found++;
      }
      if (node.name() == "fused_batch_norm") {
        EXPECT_EQ(node.op(), "_FusedBatchNormEx");
        ASSERT_EQ(node.input_size(), 6);
        EXPECT_EQ(node.input(0), "input_cast");
        EXPECT_EQ(node.input(1), "scale");
        EXPECT_EQ(node.input(2), "offset");
        EXPECT_EQ(node.input(3), "mean");
        EXPECT_EQ(node.input(4), "var");
        EXPECT_EQ(node.input(5), "side_input_cast");

        auto attr = node.attr();
        EXPECT_EQ(attr["num_side_inputs"].i(), 1);
        EXPECT_EQ(attr["activation_mode"].s(), "Relu");
        found++;
      }
    }
    EXPECT_EQ(found, 2);

    if (GetNumAvailableGPUs() > 0) {
      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, /*rtol=*/1e-2);
    }
  }
}

TEST_F(RemapperTest, FuseConv2DWithBias) {
  using ::tensorflow::ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
  auto bias_shape = ops::Placeholder::Shape({128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv = ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);
  auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "bias_add") {
      EXPECT_EQ(node.op(), "_FusedConv2D");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 1);
      EXPECT_EQ(node.input(2), "bias");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      found++;
    }
  }
  EXPECT_EQ(found, 1);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

class RemapperFuseMatMulWithBiasTest : public RemapperTest {
 public:
  template <DataType DTYPE>
  void RunTest() {
    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto lhs_shape = ops::Placeholder::Shape({8, 32});
    auto rhs_shape = ops::Placeholder::Shape({32, 64});
    auto bias_shape = ops::Placeholder::Shape({64});

    auto lhs = Placeholder(s.WithOpName("lhs"), DTYPE, lhs_shape);
    auto rhs = Placeholder(s.WithOpName("rhs"), DTYPE, rhs_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

    auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);
    auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);

    auto lhs_t = GenerateTensorWithSetRandom<DTYPE>({8, 32});
    auto rhs_t = GenerateTensorWithSetRandom<DTYPE>({32, 64});
    auto bias_t = GenerateTensorWithSetRandom<DTYPE>({64});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}, {"bias", bias_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "bias_add") {
        EXPECT_EQ(node.op(), "_FusedMatMul");
        ASSERT_GE(node.input_size(), 3);
        EXPECT_EQ(node.input(0), "lhs");
        EXPECT_EQ(node.input(1), "rhs");

        EXPECT_EQ(node.attr().at("num_args").i(), 1);
        EXPECT_EQ(node.input(2), "bias");

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(fused_ops.size(), 1);
        EXPECT_EQ(fused_ops[0], "BiasAdd");
        found++;
      }
    }
    EXPECT_EQ(1, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    float atol = 1e-6, rtol = 1e-6;
    if (DTYPE == DT_BFLOAT16 || DTYPE == DT_HALF) {
      atol = 1e-2;
      rtol = 1e-2;
    }
    test::ExpectClose(tensors[0], tensors_expected[0], atol, rtol);
  }
};

TEST_F(RemapperFuseMatMulWithBiasTest, F32) { RunTest<DT_FLOAT>(); }

TEST_F(RemapperFuseMatMulWithBiasTest, Bf16) {
  if (!IsMKLEnabled() || !IsDataTypeSupportedByOneDNNOnThisCPU(DT_BFLOAT16))
    GTEST_SKIP() << "Intel oneDNN with bfloat16 support is not supported, "
                    "skipping FuseMatMulWithBias with bfloat16.";
  RunTest<DT_BFLOAT16>();  // NOLINT
}

// TODO(b/161005848): Fix flaky test.
TEST_F(RemapperTest, DISABLED_FuseConv2DWithBiasAndActivationOnGPU) {
#if !(GOOGLE_CUDA)
  GTEST_SKIP() << "No CUDA, skip FuseConv2DWithBiasAndActivation on GPU";
#endif  // !GOOGLE_CUDA
  using ::tensorflow::ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto lhs_shape = ops::Placeholder::Shape({8, 32});
  auto rhs_shape = ops::Placeholder::Shape({32, 64});
  auto bias_shape = ops::Placeholder::Shape({64});

  auto lhs = Placeholder(s.WithOpName("lhs"), DT_FLOAT, lhs_shape);
  auto rhs = Placeholder(s.WithOpName("rhs"), DT_FLOAT, rhs_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);
  auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);

  auto lhs_t = GenerateRandomTensor<DT_FLOAT>({8, 32});
  auto rhs_t = GenerateRandomTensor<DT_FLOAT>({32, 64});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({64});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}, {"bias", bias_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "bias_add") {
      EXPECT_EQ(node.op(), "_FusedMatMul");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "lhs");
      EXPECT_EQ(node.input(1), "rhs");

      EXPECT_EQ(node.attr().at("num_args").i(), 1);
      EXPECT_EQ(node.input(2), "bias");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      found++;
    }
  }
  EXPECT_EQ(1, found);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(RemapperTest, FuseConv2DWithBiasAndActivation) {
  using ::tensorflow::ops::Placeholder;

  for (const string& activation : {"Relu", "Relu6", "Elu", "LeakyRelu"}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = Placeholder::Shape({8, 32, 32, 3});
    auto filter_shape = Placeholder::Shape({1, 1, 3, 128});
    auto bias_shape = Placeholder::Shape({128});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
    auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

    std::vector<int> strides = {1, 1, 1, 1};
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);

    ops::Identity fetch = [&]() -> ops::Identity {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");

      if (activation == "Relu") {
        return ops::Identity(fetch, ops::Relu(activate, bias_add));
      } else if (activation == "Relu6") {
        return ops::Identity(fetch, ops::Relu6(activate, bias_add));
      } else if (activation == "Elu") {
        return ops::Identity(fetch, ops::Elu(activate, bias_add));
      } else if (activation == "LeakyRelu") {
        auto attr = ops::internal::LeakyRelu::Alpha(0.5);
        return ops::Identity(
            fetch, ops::internal::LeakyRelu(activate, bias_add, attr));
      }

      return ops::Identity(fetch, bias);
    }();

    auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
    auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
    auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "activation") {
        EXPECT_EQ(node.op(), "_FusedConv2D");
        ASSERT_GE(node.input_size(), 3);
        EXPECT_EQ(node.input(0), "input");
        EXPECT_EQ(node.input(1), "filter");

        EXPECT_EQ(node.attr().at("num_args").i(), 1);
        EXPECT_EQ(node.input(2), "bias");

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(fused_ops.size(), 2);
        EXPECT_EQ(fused_ops[0], "BiasAdd");
        EXPECT_EQ(fused_ops[1], activation);

        if (activation == "LeakyRelu") {
          EXPECT_EQ(node.attr().at("leakyrelu_alpha").f(), 0.5);
        }
        found++;
      }
    }
    EXPECT_EQ(found, 1);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  }
}

class RemapperFuseMatMulWithBiasAndActivationTest : public RemapperTest {
 public:
  template <DataType DTYPE>
  void RunTest() {
    if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to oneDNN.";
    if (!IsDataTypeSupportedByOneDNNOnThisCPU(DTYPE))
      GTEST_SKIP() << "Intel oneDNN with " << DataType_Name(DTYPE)
                   << " is not supported, skipping "
                   << " RemapperFuseMatMulWithBiasAndActivationTest test.";
    using ::tensorflow::ops::Placeholder;

    for (const string& activation : {"Relu", "Relu6", "Elu"}) {
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();

      auto lhs_shape = ops::Placeholder::Shape({8, 32});
      auto rhs_shape = ops::Placeholder::Shape({32, 64});
      auto bias_shape = ops::Placeholder::Shape({64});

      auto lhs = Placeholder(s.WithOpName("lhs"), DTYPE, lhs_shape);
      auto rhs = Placeholder(s.WithOpName("rhs"), DTYPE, rhs_shape);
      auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

      auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
      auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);

      ops::Identity fetch = [&]() -> ops::Identity {
        auto activate = s.WithOpName("activation");
        auto fetch = s.WithOpName("fetch");

        if (activation == "Relu") {
          return ops::Identity(fetch, ops::Relu(activate, bias_add));
        } else if (activation == "Relu6") {
          return ops::Identity(fetch, ops::Relu6(activate, bias_add));
        } else if (activation == "Elu") {
          return ops::Identity(fetch, ops::Elu(activate, bias_add));
        }

        return ops::Identity(fetch, bias);
      }();

      auto lhs_t = GenerateTensorWithSetRandom<DTYPE>({8, 32});
      auto rhs_t = GenerateTensorWithSetRandom<DTYPE>({32, 64});
      auto bias_t = GenerateTensorWithSetRandom<DTYPE>({64});

      GrapplerItem item;
      item.fetch = {"fetch"};
      item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}, {"bias", bias_t}};
      TF_ASSERT_OK(s.ToGraphDef(&item.graph));

      // Place all nodes on CPU.
      for (int i = 0; i < item.graph.node_size(); ++i) {
        item.graph.mutable_node(i)->set_device("/device:CPU:0");
      }

      Remapper optimizer(RewriterConfig::ON);
      GraphDef output;
      TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

      int found = 0;
      for (const NodeDef& node : output.node()) {
        if (node.name() == "activation") {
          EXPECT_EQ(node.op(), "_FusedMatMul");
          ASSERT_GE(node.input_size(), 3);
          EXPECT_EQ(node.input(0), "lhs");
          EXPECT_EQ(node.input(1), "rhs");

          EXPECT_EQ(node.attr().at("num_args").i(), 1);
          EXPECT_EQ(node.input(2), "bias");

          const auto fused_ops = node.attr().at("fused_ops").list().s();
          ASSERT_EQ(fused_ops.size(), 2);
          EXPECT_EQ(fused_ops[0], "BiasAdd");
          EXPECT_EQ(fused_ops[1], activation);
          found++;
        }
      }
      EXPECT_EQ(1, found);

      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      float atol = 1e-6, rtol = 1e-6;
      if (DTYPE == DT_BFLOAT16 || DTYPE == DT_HALF) {
        atol = 1e-2;
        rtol = 1e-2;
      }
      test::ExpectClose(tensors[0], tensors_expected[0], atol, rtol);
    }
  }
};

TEST_F(RemapperFuseMatMulWithBiasAndActivationTest, F32) {
  RunTest<DT_FLOAT>();
}

TEST_F(RemapperFuseMatMulWithBiasAndActivationTest, BF16) {
  RunTest<DT_BFLOAT16>();
}

TEST_F(RemapperFuseMatMulWithBiasAndActivationTest, F16) { RunTest<DT_HALF>(); }

TEST_F(RemapperTest, FuseMklLayerNorm) {
  if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to MKL.";
  using ::tensorflow::ops::Placeholder;
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  TensorShape input_shape = TensorShape({2, 4});
  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT,
                           ops::Placeholder::Shape(input_shape));
  auto add_const = ops::Const(s.WithOpName("add_const"), 1.0f, {2, 4});
  auto add = ops::Add(s.WithOpName("b_add"), add_const, input);
  auto r_indices = ops::Const(s.WithOpName("r_indices"), {1}, {1});
  ops::Mean::Attrs attrs;
  attrs = attrs.KeepDims(true);
  auto mean = ops::Mean(s.WithOpName("mean"), add, r_indices, attrs);
  auto s_diff = ops::SquaredDifference(s.WithOpName("s_diff"), mean, add);
  auto variance = ops::Mean(s.WithOpName("variance"), s_diff, r_indices, attrs);
  auto e_const = ops::Const(s.WithOpName("e_const"), {0.001f}, {});
  auto add_1 = ops::Add(s.WithOpName("add_1"), e_const, variance);
  auto rsqrt = ops::Rsqrt(s.WithOpName("rsqrt"), add_1);
  auto g_const = ops::Const(s.WithOpName("g_const"), 1.0f, {4});
  auto mul = ops::Mul(s.WithOpName("mul"), rsqrt, g_const);
  auto mul_1 = ops::Mul(s.WithOpName("mul_1"), mul, add);
  auto mul_2 = ops::Mul(s.WithOpName("mul_2"), mul, mean);
  auto b_const = ops::Const(s.WithOpName("b_const"), 0.0f, {4});
  auto sub = ops::Sub(s.WithOpName("sub"), b_const, mul_2);
  auto add_2 = ops::Add(s.WithOpName("add_2"), mul_1, sub);
  auto fetch = ops::Identity(s.WithOpName("fetch"), add_2);

  auto input_t = GenerateTensorWithSetRandom<DT_FLOAT>({2, 4});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "add_2") {
      EXPECT_EQ(node.op(), "_MklLayerNorm");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "b_add");
      EXPECT_EQ(node.input(1), "g_const");
      EXPECT_EQ(node.input(2), "b_const");
      found++;
    }
  }
  EXPECT_EQ(found, 1);
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-4);
}

class FuseMklLayerNormPattern : public RemapperTest {
 public:
  template <DataType DTYPE>
  void RunTest() {
    if (!IsMKLEnabled()) GTEST_SKIP() << "Test only applicable to MKL.";
    using ::tensorflow::ops::Placeholder;
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    TensorShape input_shape = TensorShape({2, 4});
    auto input = Placeholder(s.WithOpName("input"), DTYPE,
                             ops::Placeholder::Shape(input_shape));
    auto add_const = ops::Const(s.WithOpName("add_const"), 1.0f, {2, 4});
    auto add = ops::Add(s.WithOpName("b_add"), add_const, input);
    auto r_indices = ops::Const(s.WithOpName("r_indices"), {1}, {1});
    ops::Mean::Attrs attrs;
    attrs = attrs.KeepDims(true);
    auto mean = ops::Mean(s.WithOpName("mean"), add, r_indices, attrs);
    auto sub = ops::Sub(s.WithOpName("sub"), add, mean);
    auto s_diff = ops::SquaredDifference(s.WithOpName("s_diff"), mean, add);
    auto variance =
        ops::Mean(s.WithOpName("variance"), s_diff, r_indices, attrs);
    auto e_const = ops::Const(s.WithOpName("e_const"), {0.001f}, {});
    auto add_1 = ops::AddV2(s.WithOpName("add_1"), e_const, variance);
    auto rsqrt = ops::Rsqrt(s.WithOpName("rsqrt"), add_1);
    auto mul = ops::Mul(s.WithOpName("mul"), sub, rsqrt);
    auto g_const = ops::Const(s.WithOpName("g_const"), 1.0f, {4});
    auto mul_1 = ops::Mul(s.WithOpName("mul_1"), g_const, mul);
    auto b_const = ops::Const(s.WithOpName("b_const"), 0.0f, {4});
    auto add_2 = ops::AddV2(s.WithOpName("add_2"), mul_1, b_const);
    auto fetch = ops::Identity(s.WithOpName("fetch"), add_2);

    auto input_t = GenerateTensorWithSetRandom<DTYPE>({2, 4});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "add_2") {
        EXPECT_EQ(node.op(), "_MklLayerNorm");
        ASSERT_GE(node.input_size(), 3);
        EXPECT_EQ(node.input(0), "b_add");
        EXPECT_EQ(node.input(1), "g_const");
        EXPECT_EQ(node.input(2), "b_const");
        found++;
      }
    }

    EXPECT_EQ(found, 1);
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-4);
  }
};

TEST_F(FuseMklLayerNormPattern, F32) { RunTest<DT_FLOAT>(); }

#ifndef INTEL_MKL
TEST_F(RemapperTest, FuseConv2DWithBatchNorm) {
  using ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
  auto scale_shape = ops::Placeholder::Shape({128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT, scale_shape);
  auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT, scale_shape);
  auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT, scale_shape);
  auto variance = Placeholder(s.WithOpName("variance"), DT_FLOAT, scale_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv = ops::Conv2D(
      s.WithOpName("conv"), input, filter, strides, "EXPLICIT",
      ops::Conv2D::Attrs().ExplicitPaddings({0, 0, 1, 2, 3, 4, 0, 0}));
  ops::FusedBatchNorm::Attrs attrs;
  attrs = attrs.IsTraining(false);
  auto batch_norm = ops::FusedBatchNorm(s.WithOpName("batch_norm"), conv, scale,
                                        offset, mean, variance, attrs);
  auto fetch = ops::Identity(s.WithOpName("fetch"), batch_norm.y);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
  auto scale_t = GenerateRandomTensor<DT_FLOAT>({128});
  auto offset_t = GenerateRandomTensor<DT_FLOAT>({128});
  auto mean_t = GenerateRandomTensor<DT_FLOAT>({128});
  auto variance_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}, {"filter", filter_t},
               {"scale", scale_t}, {"offset", offset_t},
               {"mean", mean_t},   {"variance", variance_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "batch_norm") {
      EXPECT_EQ(node.op(), "_FusedConv2D");
      ASSERT_GE(node.input_size(), 6);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 4);
      EXPECT_EQ(node.input(2), "scale");
      EXPECT_EQ(node.input(3), "offset");
      EXPECT_EQ(node.input(4), "mean");
      EXPECT_EQ(node.input(5), "variance");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "FusedBatchNorm");
      found++;
    }
  }
  EXPECT_EQ(found, 1);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(RemapperTest, FuseConv2DWithBatchNormAndActivation) {
  using ops::Placeholder;

  for (const string& activation : {"Relu", "Relu6", "Elu", "LeakyRelu"}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = ops::Placeholder::Shape({8, 32, 32, 3});
    auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
    auto scale_shape = ops::Placeholder::Shape({128});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
    auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
    auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT, scale_shape);
    auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT, scale_shape);
    auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT, scale_shape);
    auto variance =
        Placeholder(s.WithOpName("variance"), DT_FLOAT, scale_shape);

    std::vector<int> strides = {1, 1, 1, 1};
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");
    ops::FusedBatchNorm::Attrs attrs;
    attrs = attrs.IsTraining(false);
    auto batch_norm = ops::FusedBatchNorm(s.WithOpName("batch_norm"), conv,
                                          scale, offset, mean, variance, attrs);

    ops::Identity fetch = [&]() -> ops::Identity {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");

      if (activation == "Relu") {
        return ops::Identity(fetch, ops::Relu(activate, batch_norm.y));
      } else if (activation == "Relu6") {
        return ops::Identity(fetch, ops::Relu6(activate, batch_norm.y));
      } else if (activation == "Elu") {
        return ops::Identity(fetch, ops::Elu(activate, batch_norm.y));
      } else if (activation == "LeakyRelu") {
        auto attr = ops::internal::LeakyRelu::Alpha(0.5);
        return ops::Identity(
            fetch, ops::internal::LeakyRelu(activate, batch_norm.y, attr));
      }

      return ops::Identity(fetch, batch_norm.y);
    }();

    auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});
    auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
    auto scale_t = GenerateRandomTensor<DT_FLOAT>({128});
    auto offset_t = GenerateRandomTensor<DT_FLOAT>({128});
    auto mean_t = GenerateRandomTensor<DT_FLOAT>({128});
    auto variance_t = GenerateRandomTensor<DT_FLOAT>({128});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"filter", filter_t},
                 {"scale", scale_t}, {"offset", offset_t},
                 {"mean", mean_t},   {"variance", variance_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "activation") {
        EXPECT_EQ(node.op(), "_FusedConv2D");
        ASSERT_GE(node.input_size(), 6);
        EXPECT_EQ(node.input(0), "input");
        EXPECT_EQ(node.input(1), "filter");

        EXPECT_EQ(node.attr().at("num_args").i(), 4);
        EXPECT_EQ(node.input(2), "scale");
        EXPECT_EQ(node.input(3), "offset");
        EXPECT_EQ(node.input(4), "mean");
        EXPECT_EQ(node.input(5), "variance");

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(fused_ops.size(), 2);
        EXPECT_EQ(fused_ops[0], "FusedBatchNorm");
        EXPECT_EQ(fused_ops[1], activation);

        if (activation == "LeakyRelu") {
          EXPECT_EQ(node.attr().at("leakyrelu_alpha").f(), 0.5);
        }
        found++;
      }
    }
    EXPECT_EQ(found, 1);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 1);
    test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  }
}

// Disabling for now since it will be enabled when adding PR #51702
TEST_F(RemapperTest, FuseConv2DWithSqueezeAndBias) {
  using ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({8, 32, 1, 3});
  auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
  auto bias_shape = ops::Placeholder::Shape({128});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  std::vector<int> strides = {1, 1, 1, 1};
  auto conv = ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME");

  ops::Squeeze::Attrs attrs;
  attrs = attrs.Axis({2});
  auto squeeze = ops::Squeeze(s.WithOpName("squeeze"), conv, attrs);

  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), squeeze, bias);
  auto fetch = ops::Identity(s.WithOpName("fetch"), bias_add);

  auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 1, 3});
  auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 128});
  auto bias_t = GenerateRandomTensor<DT_FLOAT>({128});

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "conv") {
      EXPECT_EQ(node.op(), "_FusedConv2D");
      ASSERT_GE(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_EQ(node.input(1), "filter");

      EXPECT_EQ(node.attr().at("num_args").i(), 1);
      EXPECT_EQ(node.input(2), "bias");

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      ASSERT_EQ(fused_ops.size(), 1);
      EXPECT_EQ(fused_ops[0], "BiasAdd");
      found++;
    } else if (node.name() == "bias_add") {
      EXPECT_EQ(node.op(), "Squeeze");
      ASSERT_GE(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "conv");
      found++;
    }
  }
  EXPECT_EQ(found, 2);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}
#endif  // !INTEL_MKL

}  // namespace grappler
}  // namespace tensorflow
