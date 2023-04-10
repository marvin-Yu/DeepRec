#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/util/tensor_slice_reader.h"


namespace tensorflow {

template <typename TIndex>
class RecordSparseIndicesOp: public OpKernel {
 public:
   explicit RecordSparseIndicesOp(OpKernelConstruction* context) : OpKernel(context) {
     OP_REQUIRES_OK(context,
         context->GetAttr("var_name", &sparse_incr_res_name_));
   }

  void Compute(OpKernelContext* ctx) override {
    /*
    IndicesIncrRecorder<TIndex>* sparse_incr_res = nullptr;

    auto rm = ctx->resource_manager();

    OP_REQUIRES_OK(
        ctx,
        rm->LookupOrCreate<IndicesIncrRecorder<TIndex>>(
            "", sparse_incr_res_name_ + "_sparse_incr", &sparse_incr_res,
            [this](IndicesIncrRecorder<TIndex>** ptr) {
              *ptr = new IndicesIncrRecorder<TIndex>();
              VLOG(2) << "sparse_incr_res created, name:" << sparse_incr_res_name_;
              return Status::OK();
            }));

    sparse_incr_res->UpdateIndices(ctx->input(0));
    */

  }
 private:
  string sparse_incr_res_name_;
};

REGISTER_KERNEL_BUILDER(Name("RecordSparseIndices")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("TIndex"),
    RecordSparseIndicesOp<int32>);

REGISTER_KERNEL_BUILDER(Name("RecordSparseIndices")
    .Device(DEVICE_CPU)
    .TypeConstraint<int64>("TIndex"),
    RecordSparseIndicesOp<int64>);

}