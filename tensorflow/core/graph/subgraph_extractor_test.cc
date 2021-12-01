// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/22
// Description:
// Subraph extractor unit test 

#include "tensorflow/core/graph/subgraph_extractor.h"

#include <string>
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/protobuf.h"

const static std::string SIMPLE_MODE_PATH = "core/graph/testdata/simple_model_test.pbtxt";

namespace tensorflow {
namespace {

void ReadFileToStringOrDie(Env* env, const string& filename, string* output) {
  TF_CHECK_OK(ReadFileToString(env, filename, output));
}

class SubgraphExtractorTest : public ::testing::Test {
 protected:
  SubgraphExtractorTest() : 
      graph_(OpRegistry::Global()) {};
  void Setup(const std::string& graph_path) {
    std::string proto_string;
    std::string filename =
        io::JoinPath(testing::TensorFlowSrcRoot(), graph_path);
    ReadFileToStringOrDie(Env::Default(), filename, &proto_string);
    protobuf::TextFormat::ParseFromString(proto_string, &graph_def_);
    
    TF_CHECK_OK(ConvertGraphDefToGraph({}, graph_def_, &graph_));
    LOG(INFO) << "Setup finish";
  }

  void Reset() { LOG(INFO) << "Reset test."; }

  GraphDef graph_def_;
  Graph graph_;
};

TEST_F(SubgraphExtractorTest, ExtractSubgraphSimple) {
  Setup(SIMPLE_MODE_PATH);
  std::vector<std::string> input_node_names = {"MatMul_1"};
  std::vector<std::string> output_node_names = {"MatMul_3"};
  bool succ = ExtractSubgraph(&graph_, input_node_names, output_node_names);
  CHECK_EQ(true, succ);
}

} // namespace
} // namespace tensorflow
