#include "tensorflow/core/grappler/optimizers/tile_optimizer.h"
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
class TileOptimizerTest : public ::testing::Test {};

GraphDef output;
TEST_F(TileOptimizerTest, FusedBatchNorm) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output inputs = ops::Placeholder(s.WithOpName("inputs"), DT_FLOAT,
            ops::Placeholder::Shape({1, 10, 64}));

    Output multiplies = ops::Const(s.WithOpName("multiplies1"), {60, 2, 1}, {3});
    Output tile = ops::Tile(s.WithOpName("tile"), inputs, multiplies);

    Output equal_to = ops::Placeholder(s.WithOpName("equal_to"), DT_FLOAT,
            ops::Placeholder::Shape({60, 1, 64}));

    Output eq = ops::Equal(s.WithOpName("eq"), tile, equal_to);
 
    Output out = ops::Identity(s.WithOpName("identity"), eq);

    GrapplerItem item;
    item.fetch = {"identytp"};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));
    std::cout << item.graph.DebugString() << std::endl;
//    GraphDef output;
    TF_EXPECT_OK(TileOptimizer().Optimize(nullptr, item, &output));
  //  std::cout << output.DebugString() << std::endl;
    NodeMap node_map(&output);

    auto identity = node_map.GetNode("identity");
    ASSERT_TRUE(identity);
    ASSERT_EQ(identity->input(0), "eq_TileEqual_01");
}
}  // namespace grappler
}  // namespace tensorflow
