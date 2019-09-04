#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("TakeAxis")
    .Input("input: T")
    .Input("begin: Index")
    .Output("output: T")
    .Attr("size: int")
    .Attr("axis: int")
    .Attr("reverse: bool")
    .Attr("T: realnumbertype")
    .Attr("Index: {int32,int64}")
    .SetShapeFn([](InferenceContext* c) {
      int32 axis;
      int32 size;
      TF_RETURN_IF_ERROR(c->GetAttr("size", &size));
      TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));
      ShapeHandle in = c->input(0);
      ShapeHandle out;
      DimensionHandle size_dim;
      TF_RETURN_IF_ERROR(c->ReplaceDim(in, axis, c->MakeDim(size), &out));
      c->set_output(0, out);
      return Status::OK();
    });
