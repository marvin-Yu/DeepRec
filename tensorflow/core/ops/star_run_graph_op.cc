#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

REGISTER_OP("StarRunGraph")
.Input("feed: T1")
.Output("fetch: T2")
.Attr("T1: list(type) >= 0")
.Attr("T2: list(type) >= 0")
.Attr("feed_names: list(string) >=0")
.Attr("fetch_names: list(string) >=0")
.Attr("loc: string")
.Attr("graph_handle: string")
.Attr("ps_graph_count: int")
.Doc(R"doc(
convert the input tensor to string tensor, the string output content is
in-tensor's serialized proto.)doc");

}  // namespace tensorflow

