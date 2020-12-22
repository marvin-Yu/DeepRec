// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/22
// Description:
// Subraph extractor unit test 

#include "tensorflow/core/graph/subgraph_extractor.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class SubgraphExtractorTest : public ::testing::Test {
 protected:
  SubgraphExtractorTest() {};
  void Reset() { LOG(INFO) << "Reset test."; }
}

TEST_F(SubgraphExtractorTest, Basic) {
  LOG(INFO) << "Subgraph Extractor basic test.";
}

} // namespace
} // namespace tensorflow