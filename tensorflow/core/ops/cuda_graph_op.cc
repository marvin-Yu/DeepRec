#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

REGISTER_OP("CudaGraph")
.Input("feed: T1")
.Output("fetch: T2")
.Attr("T1: list(type) >= 0")
.Attr("T2: list(type) >= 0")
.Attr("feed_names: list(string) >=0")
.Attr("fetch_names: list(string) >=0")
.Attr("buckets: list(int) >=0")
.Attr("graph_name: string")
.Doc(R"doc(
Get cuda graph exec instance from cudaGraphMgr if captured,
Launch cuda graph and fetch output tensor)doc");

}  // namespace tensorflow