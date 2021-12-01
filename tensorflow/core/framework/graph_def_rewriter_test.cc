// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/24
// Description:
// Graph def rewriter unit test

#include "tensorflow/core/framework/graph_def_rewriter.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/protobuf.h"

const static std::string SIMPLE_MODE_PATH = "core/framework/testdata/simple_model_test.pbtxt";

namespace tensorflow {
namespace {

void ReadFileToStringOrDie(Env* env, const string& filename, string* output) {
  TF_CHECK_OK(ReadFileToString(env, filename, output));
}

void DumpGraph(const GraphDef& graph_def) {
  return;
}

class GraphDefRewriterTest : public ::testing::Test {
protected:
  GraphDefRewriterTest() {};
  void Setup(const std::string& graph_path) {
    std::string proto_string;
    std::string filename =
        io::JoinPath(testing::TensorFlowSrcRoot(), graph_path);
    ReadFileToStringOrDie(Env::Default(), filename, &proto_string);
    protobuf::TextFormat::ParseFromString(proto_string, &graph_def_);
  }

  GraphDef graph_def_;
};

TEST_F(GraphDefRewriterTest, InitSimple) {
  Setup(SIMPLE_MODE_PATH);
  GraphDefRewriter rewriter(graph_def_);
  std::vector<std::string> output;
  std::unordered_set<std::string> empty_set;
  PartialTensorShape shape({-1, 512});
  rewriter.AddPlaceholder("new_ph", DataType::DT_FLOAT, shape);
  rewriter.ReplaceEdgesForGivenConsumer("MatMul_1", 0, "new_ph", 0, empty_set);
  std::vector<std::string> top_node = {"MatMul_3"};
  GraphDef new_def;
  rewriter.GenerateGraphDefFromTop(new_def, top_node, output);
}

TEST_F(GraphDefRewriterTest, GenSubSimple) {
  Setup(SIMPLE_MODE_PATH);
  GraphDef output_graph;

  SubgraphDescription desc;
  desc.set_subgraph_name("simple");
  desc.add_output_node_names("MatMul_3");
  SubgraphInputTensor* input = desc.add_input_tensors();
  input->set_tensor_provider_name("MatMul_1");
  input->set_tensor_provider_slot(0);
  input->set_ph_name("ph");
  input->set_type(DataType::DT_FLOAT);
  input->add_shape(-1);
  input->add_shape(512);
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  SubgraphGenerator::GenerateSubgraph(graph_def_, output_graph, desc, inputs, outputs);
}


TEST_F(GraphDefRewriterTest, RepSubSimple) {
  Setup(SIMPLE_MODE_PATH);
  GraphDef output_graph;

  SubgraphDescription desc;
  desc.set_subgraph_name("simple");
  desc.add_output_node_names("MatMul_3");
  SubgraphInputTensor* input = desc.add_input_tensors();
  input->set_tensor_provider_name("MatMul_1");
  input->set_tensor_provider_slot(0);
  input->set_ph_name("ph");
  input->set_type(DataType::DT_FLOAT);
  input->add_shape(-1);
  input->add_shape(512);
  std::vector<SubgraphDescription*> descs = {&desc};
  SubgraphGenerator::ReplaceSubgraph(graph_def_, output_graph, descs, {64}, {"output"});
}

}
} // tensorflow
