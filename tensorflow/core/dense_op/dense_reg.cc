#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace tensorflow;

REGISTER_OP("DenseOp")
.Input("feed: T1")
.Output("fetch: T2")
.Attr("T1: list(type) >= 0")
.Attr("T2: list(type) >= 0")
.Attr("feed_names: list(string) >= 0")
.Attr("fetch_names: list(string) >= 0")
.Attr("graph: string")
.SetIsStateful(); // avoid constantfolding
