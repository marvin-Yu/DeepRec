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
  std::vector<std::string> top_node = {"MatMul_3"};
  std::unordered_set<std::string> term_op;
  GraphDef new_def;
  rewriter.GenerateGraphDefFromTop(new_def, top_node, term_op);
}

}
} // tensorflow
