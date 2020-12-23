// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/22
// Description:
// Subraph extractor unit test 

#include "tensorflow/core/graph/subgraph_extractor.h"

#include <iostream>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class SubgraphExtractorTest : public ::testing::Test {
 protected:
  SubgraphExtractorTest() {;};
  void Reset() { std::cout << "Reset test." << std::endl; }
};

TEST_F(SubgraphExtractorTest, Basic) {
  LOG(INFO) << "Subgraph Extractor basic test.";
  ASSERT_EQ("1", "1");
}

} // namespace
} // namespace tensorflow
