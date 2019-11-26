#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;
/* values N * (scences, unit_size)
 * coords N * (scences, 2)
 * output_shape (2, ) [user's sessions, max scenes in one session]
 * output (users' sessions, max scenes in one session, unit_size)
 */

REGISTER_OP("Take")
    .Input("values: N * T")
    .Input("coords: N * Tindices")
    .Input("output_shape: Tindices")
    .Output("output: T")
    .Attr("N: int")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(shape_inference::UnknownShape);

/* values N * (scences, unit_size)
 * coords N * (scences, 2)
 * output_shape (2, ) [user's sessions, max scenes in one session]
 * output (users' sessions, max scenes in one session, unit_size)
 */

REGISTER_OP("TakeGrad")
    .Input("grad: T")
    .Input("coords: N * Tindices")
    .Output("grad_values: N * T")
    .Attr("N: int")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(shape_inference::UnknownShape);
