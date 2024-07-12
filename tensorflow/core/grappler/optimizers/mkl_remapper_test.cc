/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/remapper.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {
namespace grappler {

class MklRemapperTest : public GrapplerTest {
 public:
  const string kAddNOp = "AddN";
  const string kAddOp = "Add";
  const string kAddV2Op = "AddV2";

 protected:
  void FuseConv2DWithBiasAndAddNOrAdd(const string& data_format,
                                      const string& activation, string add_op,
                                      bool add_with_bcast) {
    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = (data_format == "NHWC")
                           ? ops::Placeholder::Shape({8, 32, 32, 3})
                           : ops::Placeholder::Shape({8, 3, 32, 32});
    auto input_shape_addn = ops::Placeholder::Shape({});
    if (data_format == "NHWC") {
      if (add_with_bcast)
        input_shape_addn = ops::Placeholder::Shape({128});
      else
        input_shape_addn = ops::Placeholder::Shape({8, 32, 32, 128});
    } else {
      if (add_with_bcast)
        input_shape_addn = ops::Placeholder::Shape({32});
      else
        input_shape_addn = ops::Placeholder::Shape({8, 128, 32, 32});
    }
    auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
    auto bias_shape = ops::Placeholder::Shape({128});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
    auto input_addn =
        Placeholder(s.WithOpName("input_addn"), DT_FLOAT, input_shape_addn);
    auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

    std::vector<int> strides = {1, 1, 1, 1};
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME",
                    ops::Conv2D::Attrs().DataFormat(data_format));
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias,
                                 ops::BiasAdd::Attrs().DataFormat(data_format));

    auto addfetch = [&](::tensorflow::Input addop) {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");
      if (activation == "Relu") {
        ops::Identity(fetch, ops::Relu(activate, addop));
      } else if (activation == "Relu6") {
        ops::Identity(fetch, ops::Relu6(activate, addop));
      } else if (activation == "Elu") {
        ops::Identity(fetch, ops::Elu(activate, addop));
      } else if (activation == "LeakyRelu") {
        auto attr = ops::internal::LeakyRelu::Alpha(0.5);
        ops::Identity(fetch, ops::internal::LeakyRelu(activate, addop, attr));
      } else {
        DCHECK(activation == "None");
        ops::Identity(fetch, addop);
      }
    };

    if (add_op == kAddNOp) {
      auto addn = ops::AddN(s.WithOpName(add_op),
                            std::initializer_list<Input>{input_addn, bias_add});
      addfetch(addn);
    } else if (add_op == kAddV2Op) {
      auto add = ops::AddV2(s.WithOpName(add_op), input_addn, bias_add);
      addfetch(add);
    } else {
      auto add = ops::Add(s.WithOpName(add_op), input_addn, bias_add);
      addfetch(add);
    }
    auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(input_shape.shape_.dim_sizes()));
    auto input_addn_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(input_shape_addn.shape_.dim_sizes()));
    auto filter_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(filter_shape.shape_.dim_sizes()));
    auto bias_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(bias_shape.shape_.dim_sizes()));

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_tensor},
                 {"filter", filter_tensor},
                 {"bias", bias_tensor},
                 {"input_addn", input_addn_tensor}};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    // Set Rewriter config to AGGRESSIVE so that we can use Placeholder shape
    // to test that Add with both inputs having same shape get fused with
    // Conv2D. Setting this config to AGGRESSIVE is not required for the feature
    // though.
    Remapper optimizer(RewriterConfig::AGGRESSIVE);
    GraphDef output;
    TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

    bool check_fusion = !add_with_bcast;
    int found = 0;
    for (const NodeDef& node : output.node()) {
      auto fetch_node_name = activation != "None" ? "activation" : add_op;
      if (node.name() == fetch_node_name) {
        if (check_fusion) {
          EXPECT_EQ("_FusedConv2D", node.op());
          EXPECT_EQ("input", node.input(0));
          EXPECT_EQ("filter", node.input(1));

          EXPECT_EQ(2, node.attr().at("num_args").i());
          EXPECT_EQ("bias", node.input(2));
          EXPECT_EQ("input_addn", node.input(3));

          const auto fused_ops = node.attr().at("fused_ops").list().s();
          if (activation != "None") {
            EXPECT_EQ(3, fused_ops.size());
            EXPECT_EQ("BiasAdd", fused_ops[0]);
            EXPECT_EQ("Add", fused_ops[1]);
            EXPECT_EQ(activation, fused_ops[2]);
          } else {
            EXPECT_EQ(2, fused_ops.size());
            EXPECT_EQ("BiasAdd", fused_ops[0]);
            EXPECT_EQ("Add", fused_ops[1]);
          }
        } else {
          if (activation != "None") {
            EXPECT_EQ(node.op(), activation);
            ASSERT_EQ(node.input_size(), 1);
            EXPECT_EQ(node.input(0), add_op);
          } else {
            EXPECT_EQ(node.op(), add_op);
            ASSERT_EQ(node.input_size(), 2);
          }
        }
        found++;
      }
    }
    EXPECT_EQ(1, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    EXPECT_EQ(1, tensors_expected.size());
    EXPECT_EQ(1, tensors.size());
    // Using relative tolerance since oneDNN could produce different results
    // when float32 numbers need to be rounded during accumulation.
    test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-6);
  }
};

#define CREATE_CONV2DFUSION_TEST(data_format, addop, activation, bcast)                          \
  TEST_F(                                                                                        \
      MklRemapperTest,                                                                           \
      FuseConv2DWithBiasAnd##addop##_##data_format##_activation##activation##_addbcast##bcast) { \
    FuseConv2DWithBiasAndAddNOrAdd(#data_format, #activation, #addop, bcast);                    \
  }

#define CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(data_format, addop, bcast) \
  CREATE_CONV2DFUSION_TEST(data_format, addop, Relu, bcast);               \
  CREATE_CONV2DFUSION_TEST(data_format, addop, Relu6, bcast);              \
  CREATE_CONV2DFUSION_TEST(data_format, addop, Elu, bcast);                \
  CREATE_CONV2DFUSION_TEST(data_format, addop, LeakyRelu, bcast);          \
  CREATE_CONV2DFUSION_TEST(data_format, addop, None, bcast);

#define CREATE_CONV2DFUSION_ADD_NOBCAST_TEST(addop)            \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NHWC, addop, false); \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NCHW, addop, false);

CREATE_CONV2DFUSION_ADD_NOBCAST_TEST(AddN);

#define CREATE_CONV2DFUSION_ADD_BCAST_TEST(addop)              \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NHWC, addop, false); \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NCHW, addop, false); \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NHWC, addop, true);  \
  // CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NCHW, addop, true); // flaky test

CREATE_CONV2DFUSION_ADD_BCAST_TEST(Add);
CREATE_CONV2DFUSION_ADD_BCAST_TEST(AddV2);

#undef CREATE_CONV2DFUSION_ADD_NOBCAST_TEST
#undef CREATE_CONV2DFUSION_ADD_BCAST_TEST
#undef CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST
#undef CREATE_CONV2DFUSION_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklRemapperTest, NAME##_##T) {                                       \
    using ::tensorflow::ops::Placeholder;                                     \
                                                                              \
    for (const string& activation : {"Relu", "Relu6", "Elu", "None"}) {       \
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();                \
                                                                              \
      auto input_shape = Placeholder::Shape({8, 32, 32, 3});                  \
      auto filter_shape = Placeholder::Shape({1, 1, 3, 1});                   \
      auto bias_shape = Placeholder::Shape({3});                              \
                                                                              \
      auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape); \
      auto filter =                                                           \
          Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);        \
      auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);    \
                                                                              \
      std::vector<int> strides = {1, 1, 1, 1};                                \
      auto conv = ops::DepthwiseConv2dNative(s.WithOpName("depthwise_conv"),  \
                                             input, filter, strides, "SAME"); \
      auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);     \
                                                                              \
      ops::Identity fetch = [&]() -> ops::Identity {                          \
        auto activate = s.WithOpName("activation");                           \
        auto fetch = s.WithOpName("fetch");                                   \
                                                                              \
        if (activation == "Relu") {                                           \
          return ops::Identity(fetch, ops::Relu(activate, bias_add));         \
        } else if (activation == "Relu6") {                                   \
          return ops::Identity(fetch, ops::Relu6(activate, bias_add));        \
        } else if (activation == "Elu") {                                     \
          return ops::Identity(fetch, ops::Elu(activate, bias_add));          \
        }                                                                     \
                                                                              \
        DCHECK(activation == "None");                                         \
        return ops::Identity(fetch, bias_add);                                \
      }();                                                                    \
                                                                              \
      auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});          \
      auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 1});           \
      auto bias_t = GenerateRandomTensor<DT_FLOAT>({3});                      \
                                                                              \
      GrapplerItem item;                                                      \
      item.fetch = {"fetch"};                                                 \
      item.feed = {                                                           \
          {"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};        \
      TF_CHECK_OK(s.ToGraphDef(&item.graph));                                 \
                                                                              \
      for (int i = 0; i < item.graph.node_size(); ++i) {                      \
        item.graph.mutable_node(i)->set_device("/device:CPU:0");              \
      }                                                                       \
                                                                              \
      Remapper optimizer(RewriterConfig::ON);                                 \
      GraphDef output;                                                        \
      TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));                \
                                                                              \
      int found = 0;                                                          \
      for (const NodeDef& node : output.node()) {                             \
        if (node.name() != "bias_add" && node.name() != "activation")         \
          continue;                                                           \
                                                                              \
        EXPECT_EQ(node.op(), "_FusedDepthwiseConv2dNative");                  \
        ASSERT_EQ(node.input_size(), 3);                                      \
        EXPECT_EQ(node.input(0), "input");                                    \
        EXPECT_EQ(node.input(1), "filter");                                   \
                                                                              \
        EXPECT_EQ(node.attr().at("num_args").i(), 1);                         \
        EXPECT_EQ(node.input(2), "bias");                                     \
                                                                              \
        const auto fused_ops = node.attr().at("fused_ops").list().s();        \
        if (node.name() == "bias_add") {                                      \
          ASSERT_EQ(fused_ops.size(), 1);                                     \
          EXPECT_EQ(fused_ops[0], "BiasAdd");                                 \
          found++;                                                            \
        }                                                                     \
        if (node.name() == "activation") {                                    \
          ASSERT_EQ(fused_ops.size(), 2);                                     \
          EXPECT_EQ(fused_ops[0], "BiasAdd");                                 \
          EXPECT_EQ(fused_ops[1], activation);                                \
          found++;                                                            \
        }                                                                     \
      }                                                                       \
      EXPECT_EQ(found, 1);                                                    \
                                                                              \
      auto tensors_expected =                                                 \
          EvaluateNodes(item.graph, item.fetch, item.feed);                   \
      ASSERT_EQ(tensors_expected.size(), 1);                                  \
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);            \
      ASSERT_EQ(tensors.size(), 1);                                           \
      test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);   \
    }                                                                         \
  }
REGISTER_TEST_ALL_TYPES(FuseDepthwiseConv2DWithBiasAndActivation);
#undef REGISTER_TEST

#ifndef ENABLE_ONEDNN_V3
TEST_F(MklRemapperTest, FuseBatchNormWithRelu) {
  using ::tensorflow::ops::Placeholder;

  for (bool is_training : {true, false}) {
    for (bool has_side_input : {true, false}) {
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();

      const int num_channels = 24;

      TensorShape channel_shape({num_channels});
      TensorShape empty_shape({0});

      auto input =
          Placeholder(s.WithOpName("input"), DT_FLOAT,
                      ops::Placeholder::Shape({2, 8, 8, num_channels}));
      auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_FLOAT);
      auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT);
      auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT);
      auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT);
      auto var = Placeholder(s.WithOpName("var"), DT_FLOAT);

      float epsilon = 0.1f;
      auto fbn =
          ops::FusedBatchNormV3(s.WithOpName("fused_batch_norm"), input_cast,
                                scale, offset, mean, var,
                                ops::FusedBatchNormV3::IsTraining(is_training)
                                    .Epsilon(epsilon)
                                    .DataFormat("NHWC"));

      if (has_side_input) {
        auto side_input =
            Placeholder(s.WithOpName("side_input"), DT_FLOAT,
                        ops::Placeholder::Shape({2, 8, 8, num_channels}));
        auto side_input_cast =
            ops::Cast(s.WithOpName("side_input_cast"), side_input, DT_FLOAT);
        auto add = ops::Add(s.WithOpName("add"), fbn.y, side_input_cast);
        auto relu = ops::Relu(s.WithOpName("relu"), add);
      } else {
        auto relu = ops::Relu(s.WithOpName("relu"), fbn.y);
      }

      auto input_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});
      auto scale_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
      auto offset_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
      auto mean_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                               : channel_shape);
      auto var_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                              : channel_shape);
      auto side_input_t =
          GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});

      GrapplerItem item;
      item.fetch = {"relu"};
      if (has_side_input)
        item.feed = {{"input", input_t},   {"scale", scale_t},
                     {"offset", offset_t}, {"mean", mean_t},
                     {"var", var_t},       {"side_input", side_input_t}};
      else
        item.feed = {{"input", input_t},
                     {"scale", scale_t},
                     {"offset", offset_t},
                     {"mean", mean_t},
                     {"var", var_t}};
      TF_ASSERT_OK(s.ToGraphDef(&item.graph));

      // Place all nodes on CPU.
      for (int i = 0; i < item.graph.node_size(); ++i) {
        item.graph.mutable_node(i)->set_device("/device:CPU:0");
      }

      Remapper optimizer(RewriterConfig::AGGRESSIVE);
      GraphDef output;
      TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

      int found = 0;
      if (has_side_input) {
        for (const NodeDef& node : output.node()) {
          if (node.name() == "add") {
            EXPECT_EQ(node.op(), "Add");
            ASSERT_EQ(node.input_size(), 2);
            EXPECT_EQ(node.input(0), "fused_batch_norm");
            EXPECT_EQ(node.input(1), "side_input_cast");
            found++;
          }
          if (node.name() == "relu") {
            EXPECT_EQ(node.op(), "Relu");
            ASSERT_EQ(node.input_size(), 1);
            EXPECT_EQ(node.input(0), "add");
            found++;
          }
          if (node.name() == "fused_batch_norm") {
            EXPECT_EQ(node.op(), "FusedBatchNormV3");
            ASSERT_EQ(node.input_size(), 5);
            EXPECT_EQ(node.input(0), "input_cast");
            EXPECT_EQ(node.input(1), "scale");
            EXPECT_EQ(node.input(2), "offset");
            EXPECT_EQ(node.input(3), "mean");
            EXPECT_EQ(node.input(4), "var");
            found++;
          }
        }
        EXPECT_EQ(found, 3);
      } else {
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
      }

      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
    }
  }
}

TEST_F(MklRemapperTest, FuseMatMulWithBiasAndGelu) {
  using ::tensorflow::ops::Placeholder;

  for (const string& activation : {"Gelu_tanh", "Gelu_erf"}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto lhs_shape = ops::Placeholder::Shape({8, 32});
    auto rhs_shape = ops::Placeholder::Shape({32, 64});
    auto bias_shape = ops::Placeholder::Shape({64});

    auto lhs = Placeholder(s.WithOpName("lhs"), DT_FLOAT, lhs_shape);
    auto rhs = Placeholder(s.WithOpName("rhs"), DT_FLOAT, rhs_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

    auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);

    ops::Identity fetch = [&]() -> ops::Identity {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");

      return ops::Identity(
          fetch, ops::Gelu(activate, bias_add,
                           ops::Gelu::Approximate(activation == "Gelu_tanh")));
    }();
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
       if(activation == "Gelu_tanh") {
         EXPECT_EQ(fused_ops[1], "Gelu");
       } else {
         EXPECT_EQ(fused_ops[1], "Gelu_erf");
       }
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
}

TEST_F(MklRemapperTest, FuseMatMulWithBiasAddAndAdd) {
  using ::tensorflow::ops::Placeholder;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto input_shape = ops::Placeholder::Shape({4, 32});
  auto input_shape_add = ops::Placeholder::Shape({4, 8});
  auto filter_shape = ops::Placeholder::Shape({32, 8});
  auto bias_shape = ops::Placeholder::Shape({8});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto input_add =
      Placeholder(s.WithOpName("input_add"), DT_FLOAT, input_shape_add);
  auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  auto matmul = ops::MatMul(s.WithOpName("matmul"), input, filter);
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);

  auto fetch = s.WithOpName("fetch");
  auto add = ops::Add(s.WithOpName("add"), bias_add, input_add);

  ops::Identity(fetch, add);

  auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
      TensorShape(input_shape.shape_.dim_sizes()));
  auto input_add_tensor = GenerateRandomTensor<DT_FLOAT>(
      TensorShape(input_shape_add.shape_.dim_sizes()));
  auto filter_tensor = GenerateRandomTensor<DT_FLOAT>(
      TensorShape(filter_shape.shape_.dim_sizes()));
  auto bias_tensor = GenerateRandomTensor<DT_FLOAT>(
      TensorShape(bias_shape.shape_.dim_sizes()));

  GrapplerItem item;
  item.fetch = {"fetch"};
  item.feed = {{"input", input_tensor},
               {"filter", filter_tensor},
               {"bias", bias_tensor},
               {"input_add", input_add_tensor}};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  Remapper optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    auto fetch_node_name = "add";
    if (node.name() == fetch_node_name) {
      EXPECT_EQ("_FusedMatMul", node.op());
      EXPECT_EQ("input", node.input(0));
      EXPECT_EQ("filter", node.input(1));

      EXPECT_EQ(2, node.attr().at("num_args").i());
      EXPECT_EQ("bias", node.input(2));
      EXPECT_EQ("input_add", node.input(3));

      const auto fused_ops = node.attr().at("fused_ops").list().s();
      EXPECT_EQ(2, fused_ops.size());
      EXPECT_EQ("BiasAdd", fused_ops[0]);
      EXPECT_EQ("Add", fused_ops[1]);
      found++;
    }
  }
  EXPECT_EQ(1, found);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-6);
}
#endif  // !ENABLE_ONEDNN_V3

class MklFusedBatchMatMul : public MklRemapperTest {
 public:
  template <DataType DTYPE>
  void VerifyFused(bool adjx, bool adjy) {
    if (!IsDataTypeSupportedByOneDNNOnThisCPU(DTYPE))
      GTEST_SKIP() << "Intel oneDNN with " << DataType_Name(DTYPE)
                   << " is not supported, skipping MklFusedBatchMatMul test.";
    using ::tensorflow::ops::Placeholder;

    int b0 = 2;
    int b1 = 2;
    int m = 32;
    int k = 16;
    int n = 64;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape =
        adjx ? TensorShape({b0, b1, k, m}) : TensorShape({b0, b1, m, k});
    auto weight_shape =
        adjy ? TensorShape({b0, b1, n, k}) : TensorShape({b0, b1, k, n});
    auto add_shape = TensorShape({b0, 1, m, n});

    auto input_placeholder_shape = ops::Placeholder::Shape(input_shape);
    auto weight_placeholder_shape = ops::Placeholder::Shape(weight_shape);
    auto add_placeholder_shape = ops::Placeholder::Shape(add_shape);

    auto input =
        Placeholder(s.WithOpName("input"), DTYPE, input_placeholder_shape);
    auto weight =
        Placeholder(s.WithOpName("weight"), DTYPE, weight_placeholder_shape);
    auto addend =
        Placeholder(s.WithOpName("addend"), DTYPE, add_placeholder_shape);

    auto batchmatmul =
        ops::BatchMatMulV2(s.WithOpName("batchmatmul"), input, weight,
                           ops::BatchMatMulV2::Attrs().AdjX(adjx).AdjY(adjy));
    auto scale_const = ops::Const(s.WithOpName("scale_const"), {0.1f});
    auto scale = ops::Cast(s.WithOpName("scale"), scale_const, DTYPE);
    auto mul = ops::Multiply(s.WithOpName("mul"), batchmatmul, scale);
    auto add = ops::AddV2(s.WithOpName("add"), mul, addend);
    auto fetch = ops::Identity(s.WithOpName("fetch"), add);

    Tensor input_tensor = GenerateTensorWithSetRandom<DTYPE>(input_shape);
    Tensor weight_tensor = GenerateTensorWithSetRandom<DTYPE>(weight_shape);
    Tensor add_tensor = GenerateTensorWithSetRandom<DTYPE>(add_shape);

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_tensor},
                 {"weight", weight_tensor},
                 {"addend", add_tensor}};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "add") {
        EXPECT_EQ("_MklFusedBatchMatMulV2", node.op());
        EXPECT_EQ("input", node.input(0));
        EXPECT_EQ("weight", node.input(1));
        EXPECT_EQ("scale", node.input(2));
        EXPECT_EQ("addend", node.input(3));
        const auto fused_ops = node.attr().at("fused_ops").list().s();
        EXPECT_EQ(2, fused_ops.size());
        EXPECT_EQ("Mul", fused_ops[0]);
        found++;
        EXPECT_EQ("Add", fused_ops[1]);
        found++;
      }
    }
    EXPECT_EQ(2, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    float atol = 1e-6, rtol = 1e-6;
    if (DTYPE == DT_BFLOAT16 || DTYPE == DT_HALF) {
      atol = 1e-2;
      rtol = 1e-2;
    }
    test::ExpectClose(tensors_expected[0], tensors[0], atol, rtol);
  }

  template <DataType DTYPE>
  void VerifyPreceedingScalarMul(bool adjx, bool adjy) {
    if (!IsDataTypeSupportedByOneDNNOnThisCPU(DTYPE))
      GTEST_SKIP() << "Intel oneDNN with " << DataType_Name(DTYPE)
                   << " is not supported, skipping MklFusedBatchMatMul test.";
    using ::tensorflow::ops::Placeholder;

    int b0 = 2;
    int b1 = 2;
    int m = 32;
    int k = 16;
    int n = 64;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape =
        adjx ? TensorShape({b0, b1, k, m}) : TensorShape({b0, b1, m, k});
    auto weight_shape =
        adjy ? TensorShape({b0, b1, n, k}) : TensorShape({b0, b1, k, n});
    auto add_shape = TensorShape({b0, 1, m, n});

    auto input_placeholder_shape = ops::Placeholder::Shape(input_shape);
    auto weight_placeholder_shape = ops::Placeholder::Shape(weight_shape);
    auto add_placeholder_shape = ops::Placeholder::Shape(add_shape);

    auto input =
        Placeholder(s.WithOpName("input"), DTYPE, input_placeholder_shape);
    auto weight =
        Placeholder(s.WithOpName("weight"), DTYPE, weight_placeholder_shape);
    auto addend =
        Placeholder(s.WithOpName("addend"), DTYPE, add_placeholder_shape);

    auto scale_const = ops::Const(s.WithOpName("scale_const"), {0.1f});
    auto scale = ops::Cast(s.WithOpName("scale"), scale_const, DTYPE);
    auto mul = ops::Multiply(s.WithOpName("mul"), input, scale);
    auto batchmatmul =
        ops::BatchMatMulV2(s.WithOpName("batchmatmul"), mul, weight,
                           ops::BatchMatMulV2::Attrs().AdjX(adjx).AdjY(adjy));
    auto add = ops::Add(s.WithOpName("add"), batchmatmul, addend);
    auto fetch = ops::Identity(s.WithOpName("fetch"), add);

    Tensor input_tensor = GenerateTensorWithSetRandom<DTYPE>(input_shape);
    Tensor weight_tensor = GenerateTensorWithSetRandom<DTYPE>(weight_shape);
    Tensor add_tensor = GenerateTensorWithSetRandom<DTYPE>(add_shape);

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_tensor},
                 {"weight", weight_tensor},
                 {"addend", add_tensor}};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "add") {
        EXPECT_EQ("_MklFusedBatchMatMulV2", node.op());
        EXPECT_EQ("input", node.input(0));
        EXPECT_EQ("weight", node.input(1));
        EXPECT_EQ("scale", node.input(2));
        EXPECT_EQ("addend", node.input(3));
        const auto fused_ops = node.attr().at("fused_ops").list().s();
        EXPECT_EQ(2, fused_ops.size());
        EXPECT_EQ("Mul", fused_ops[0]);
        found++;
        EXPECT_EQ("Add", fused_ops[1]);
        found++;
      }
    }
    EXPECT_EQ(2, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    float atol = 1e-6, rtol = 1e-6;
    if (DTYPE == DT_BFLOAT16 || DTYPE == DT_HALF) {
      atol = 1e-2;
      rtol = 1e-2;
    }
    test::ExpectClose(tensors_expected[0], tensors[0], atol, rtol);
  }
};

TEST_F(MklFusedBatchMatMul, MulAndAdd) {
  for (const auto adjx : {false, true})
    for (const auto adjy : {false, true}) {
      this->VerifyFused<DT_FLOAT>(adjx, adjy);
      this->VerifyFused<DT_BFLOAT16>(adjx, adjy);
      this->VerifyFused<DT_HALF>(adjx, adjy);
    }
}

TEST_F(MklFusedBatchMatMul, MulAndAdd2) {
  for (const auto adjx : {false, true})
    for (const auto adjy : {false, true}) {
      this->VerifyPreceedingScalarMul<DT_FLOAT>(adjx, adjy);
      this->VerifyPreceedingScalarMul<DT_BFLOAT16>(adjx, adjy);
      this->VerifyPreceedingScalarMul<DT_HALF>(adjx, adjy);
    }
}

class MklFuseInstanceNormTest : public GrapplerTest {
 protected:
  template <DataType DTYPE>
  void FuseMklInstanceNorm5D_Runner(string FORMAT, string activation) {
    using ::tensorflow::ops::Placeholder;
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    TensorShape input_shape, filter_shape, add_shape, scale_shift_shape;
    if (FORMAT == "NCDHW") {
      input_shape = TensorShape({4, 8, 3, 3, 4});
      add_shape = TensorShape({1, 16, 1, 1, 1});
      scale_shift_shape = TensorShape({1, 16, 1, 1, 1});
    } else {
      input_shape = TensorShape({4, 3, 3, 4, 8});
      add_shape = TensorShape({1, 1, 1, 1, 16});
      scale_shift_shape = TensorShape({1, 1, 1, 1, 16});
    }

    filter_shape = TensorShape({1, 1, 1, 8, 16});

    auto input_t = GenerateTensorWithSetRandom<DTYPE>(input_shape);
    auto filter_t = GenerateTensorWithSetRandom<DTYPE>(filter_shape);
    auto add_input = GenerateTensorWithSetRandom<DTYPE>(scale_shift_shape);
    auto scale_input = GenerateTensorWithSetRandom<DTYPE>(scale_shift_shape);
    auto shift_input = GenerateTensorWithSetRandom<DTYPE>(scale_shift_shape);

    typedef typename EnumToDataType<DTYPE>::Type T;
    T epsilon_value = static_cast<T>(0.00001);
    auto epsilon_tensor =
        GenerateConstantTensor<DTYPE>(TensorShape({}), epsilon_value);

    std::vector<int> strides = {1, 1, 1, 1, 1};
    auto input = Placeholder(s.WithOpName("input"), DTYPE,
                             ops::Placeholder::Shape(input_shape));
    auto filter = Placeholder(s.WithOpName("filter"), DTYPE,
                              ops::Placeholder::Shape(filter_shape));
    ops::Conv3D::Attrs conv_attrs = ops::Conv3D::Attrs().DataFormat(FORMAT);
    auto conv = ops::Conv3D(s.WithOpName("conv"), input, filter, strides,
                            "SAME", conv_attrs);

    auto add_const =
        ops::Const(s.WithOpName("add_const"), Input::Initializer(add_input));
    auto add = ops::Add(s.WithOpName("conv_add"), add_const, conv);

    auto r_indices =
        (FORMAT == "NCDHW")
            ? ops::Const(s.WithOpName("r_indices"), {2, 3, 4}, {3})
            : ops::Const(s.WithOpName("r_indices"), {1, 2, 3}, {3});
    ops::Mean::Attrs mean_attrs = ops::Mean::Attrs().KeepDims(true);
    auto mean = ops::Mean(s.WithOpName("mean"), add, r_indices, mean_attrs);
    auto s_diff = ops::SquaredDifference(s.WithOpName("s_diff"), add, mean);
    auto variance =
        ops::Mean(s.WithOpName("variance"), s_diff, r_indices, mean_attrs);

    auto epsilon =
        ops::Const(s.WithOpName("epsilon"), Input::Initializer(epsilon_tensor));
    auto add_1 = ops::AddV2(s.WithOpName("add_1"), variance, epsilon);
    auto rsqrt = ops::Rsqrt(s.WithOpName("rsqrt"), add_1);
    auto g_const =
        ops::Const(s.WithOpName("g_const"), Input::Initializer(scale_input));
    auto mul = ops::Mul(s.WithOpName("mul"), rsqrt, g_const);
    auto mul_1 = ops::Mul(s.WithOpName("mul_1"), add, mul);
    auto mul_2 = ops::Mul(s.WithOpName("mul_2"), mean, mul);
    auto b_const =
        ops::Const(s.WithOpName("b_const"), Input::Initializer(shift_input));
    auto sub = ops::Sub(s.WithOpName("sub"), b_const, mul_2);
    auto add_2 = ops::AddV2(s.WithOpName("add_2"), mul_1, sub);

    ops::Identity fetch = [&]() -> ops::Identity {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");
      if (activation == "None") {
        return ops::Identity(fetch, add_2);
      } else if (activation == "Relu") {
        return ops::Identity(fetch, ops::Relu(activate, add_2));
      } else if (activation == "LeakyRelu") {
        auto attr = ops::internal::LeakyRelu::Alpha(0.3f);
        return ops::Identity(fetch,
                             ops::internal::LeakyRelu(activate, add_2, attr));
      }
      return ops::Identity(fetch, add_2);
    }();

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"filter", filter_t}};

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    string fused_node_name = (activation == "None") ? "add_2" : "activation";
    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == fused_node_name) {
        EXPECT_EQ(node.op(), "_MklFusedInstanceNorm");
        ASSERT_EQ(node.input_size(), 3);
        EXPECT_EQ(node.input(0), "conv_add");
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
    if (DTYPE == DT_BFLOAT16) {
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
    } else {
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-5, 1e-5);
    }
  }

  template <DataType DTYPE>
  void FuseMklInstanceNorm4D_Runner(string FORMAT, string activation) {
    using ::tensorflow::ops::Placeholder;
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    TensorShape input_shape, filter_shape, add_shape, scale_shift_shape;
    if (FORMAT == "NCHW") {
      input_shape = TensorShape({4, 3, 32, 32});
      add_shape = TensorShape({1, 3, 1, 1});
      scale_shift_shape = TensorShape({1, 3, 1, 1});
    } else {
      input_shape = TensorShape({4, 32, 32, 3});
      add_shape = TensorShape({1, 1, 1, 3});
      scale_shift_shape = TensorShape({1, 1, 1, 3});
    }
    filter_shape = TensorShape({2, 2, 3, 3});

    auto input_t = GenerateTensorWithSetRandom<DTYPE>(input_shape);
    auto filter_t = GenerateTensorWithSetRandom<DTYPE>(filter_shape);
    auto add_input = GenerateTensorWithSetRandom<DTYPE>(scale_shift_shape);
    auto scale_input = GenerateTensorWithSetRandom<DTYPE>(scale_shift_shape);
    auto shift_input = GenerateTensorWithSetRandom<DTYPE>(scale_shift_shape);

    typedef typename EnumToDataType<DTYPE>::Type T;
    T epsilon_value = static_cast<T>(0.00001);
    auto epsilon_tensor =
        GenerateConstantTensor<DTYPE>(TensorShape({}), epsilon_value);

    std::vector<int> strides = {1, 1, 1, 1};
    auto input = Placeholder(s.WithOpName("input"), DTYPE,
                             ops::Placeholder::Shape(input_shape));
    auto filter = Placeholder(s.WithOpName("filter"), DTYPE,
                              ops::Placeholder::Shape(filter_shape));
    ops::Conv2D::Attrs conv_attrs = ops::Conv2D::Attrs().DataFormat(FORMAT);
    auto conv = ops::Conv2D(s.WithOpName("conv"), input, filter, strides,
                            "SAME", conv_attrs);

    auto add_const =
        ops::Const(s.WithOpName("add_const"), Input::Initializer(add_input));
    auto add = ops::Add(s.WithOpName("conv_add"), add_const, conv);

    auto r_indices = (FORMAT == "NCHW")
                         ? ops::Const(s.WithOpName("r_indices"), {2, 3}, {2})
                         : ops::Const(s.WithOpName("r_indices"), {1, 2}, {2});
    ops::Mean::Attrs mean_attrs = ops::Mean::Attrs().KeepDims(true);
    auto mean = ops::Mean(s.WithOpName("mean"), add, r_indices, mean_attrs);
    auto s_diff = ops::SquaredDifference(s.WithOpName("s_diff"), add, mean);
    auto variance =
        ops::Mean(s.WithOpName("variance"), s_diff, r_indices, mean_attrs);

    auto epsilon =
        ops::Const(s.WithOpName("epsilon"), Input::Initializer(epsilon_tensor));
    auto add_1 = ops::AddV2(s.WithOpName("add_1"), variance, epsilon);
    auto rsqrt = ops::Rsqrt(s.WithOpName("rsqrt"), add_1);
    auto g_const =
        ops::Const(s.WithOpName("g_const"), Input::Initializer(scale_input));
    auto mul = ops::Mul(s.WithOpName("mul"), rsqrt, g_const);
    auto mul_1 = ops::Mul(s.WithOpName("mul_1"), add, mul);
    auto mul_2 = ops::Mul(s.WithOpName("mul_2"), mean, mul);
    auto b_const =
        ops::Const(s.WithOpName("b_const"), Input::Initializer(shift_input));
    auto sub = ops::Sub(s.WithOpName("sub"), b_const, mul_2);
    auto add_2 = ops::AddV2(s.WithOpName("add_2"), mul_1, sub);

    ops::Identity fetch = [&]() -> ops::Identity {
      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");
      if (activation == "None") {
        return ops::Identity(fetch, add_2);
      } else if (activation == "Relu") {
        return ops::Identity(fetch, ops::Relu(activate, add_2));
      } else if (activation == "LeakyRelu") {
        auto attr = ops::internal::LeakyRelu::Alpha(0.3f);
        return ops::Identity(fetch,
                             ops::internal::LeakyRelu(activate, add_2, attr));
      }
      return ops::Identity(fetch, add_2);
    }();

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"filter", filter_t}};

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    string fused_node_name = (activation == "None") ? "add_2" : "activation";
    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == fused_node_name) {
        EXPECT_EQ(node.op(), "_MklFusedInstanceNorm");
        ASSERT_EQ(node.input_size(), 3);
        EXPECT_EQ(node.input(0), "conv_add");
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
    if (DTYPE == DT_BFLOAT16) {
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
    } else {
      test::ExpectClose(tensors[0], tensors_expected[0], 1e-5, 1e-5);
    }
  }

  template <DataType DTYPE>
  void FuseMklInstanceNorm4D(string FORMAT, bool add_activation = false) {
    if (!add_activation) {
      return FuseMklInstanceNorm4D_Runner<DTYPE>(FORMAT, "None");
    }
    for (const string& activation : {"Relu", "LeakyRelu"}) {
      FuseMklInstanceNorm4D_Runner<DTYPE>(FORMAT, activation);
    }
  }

  template <DataType DTYPE>
  void FuseMklInstanceNorm5D(string FORMAT, bool add_activation = false) {
    if (!add_activation) {
      return FuseMklInstanceNorm5D_Runner<DTYPE>(FORMAT, "None");
    }
    for (const string& activation : {"Relu", "LeakyRelu"}) {
      FuseMklInstanceNorm5D_Runner<DTYPE>(FORMAT, activation);
    }
  }
};

TEST_F(MklFuseInstanceNormTest, FuseMklInstanceNorm5D_FP32_NDHWC) {
  FuseMklInstanceNorm5D<DT_FLOAT>("NDHWC");
}
TEST_F(MklFuseInstanceNormTest, FuseMklInstanceNorm5D_FP32_NCDHW) {
  FuseMklInstanceNorm5D<DT_FLOAT>("NCDHW");
}
TEST_F(MklFuseInstanceNormTest, FuseMklInstanceNorm4D_FP32_NHWC) {
  FuseMklInstanceNorm4D<DT_FLOAT>("NHWC");
}
TEST_F(MklFuseInstanceNormTest, FuseMklInstanceNorm4D_FP32_NCHW) {
  FuseMklInstanceNorm4D<DT_FLOAT>("NCHW");
}
TEST_F(MklFuseInstanceNormTest,
       FuseMklInstanceNormWithActivation5D_FP32_NDHWC) {
  FuseMklInstanceNorm5D<DT_FLOAT>("NDHWC", true);
}
TEST_F(MklFuseInstanceNormTest,
       FuseMklInstanceNormWithActivation5D_FP32_NCDHW) {
  FuseMklInstanceNorm5D<DT_FLOAT>("NCDHW", true);
}
TEST_F(MklFuseInstanceNormTest, FuseMklInstanceNormWithActivation4D_FP32_NHWC) {
  FuseMklInstanceNorm4D<DT_FLOAT>("NHWC", true);
}
TEST_F(MklFuseInstanceNormTest, FuseMklInstanceNormWithActivation4D_FP32_NCHW) {
  FuseMklInstanceNorm4D<DT_FLOAT>("NCHW", true);
}

}  // namespace grappler
}  // namespace tensorflow
#endif  // INTEL_MKL
