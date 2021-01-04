// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/29
// Description:
// Direct Session test for cudagraph support
// May be the root entrance for cudagraph testing

#include "tensorflow/core/common_runtime/direct_session.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/protobuf.h"

const static std::string SIMPLE_MODE_PATH = "core/framework/testdata/simple_model_test.pbtxt";

namespace tensorflow {
namespace {

class DirectSessionCudaGraphTest : public ::testing::Test {
public: 
  DirectSessionCudaGraphTest() {};
  void Setup(const std::string& graph_path) {
    std::string proto_string;
    std::string filename =
        io::JoinPath(testing::TensorFlowSrcRoot(), graph_path);
    ReadFileToStringOrDie(Env::Default(), filename, &proto_string);
    protobuf::TextFormat::ParseFromString(proto_string, &graph_def_);
  }

  void InitSessionConfig(SessionOptions& options, bool cudagraph_enable = false, bool try_capture = false);

  GraphDef graph_def_;
}

void DirectSessionCudaGraphTest::InitSessionConfig(SessionOptions& options, bool cudagraph_enable, bool try_capture) {
  
}

TEST_F(DirectSessionCudaGraphTest, SimpleGraphConfig) {
  SessionOptions options;
  InitSessionConfig(options);
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def_));
}

}
} // tensorflow