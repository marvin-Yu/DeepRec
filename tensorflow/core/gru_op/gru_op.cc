#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BlazeGRU")
  .Input("x: T")              //[batch_size, rounds, elts]
  .Input("h2h: T")            //[elts, 3elts]
  .Input("i2h: T")            //[elts, 3elts]
  .Input("h2h_bias: T")        //[3elts]
  .Input("i2h_bias: T")        //[3elts]
  .Output("y: T")             //[batch_size, rounds, elts]
  .Attr("T: {float}")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    //check param shapes
    c->set_output(0, c->input(0));
    return Status::OK();
  });
