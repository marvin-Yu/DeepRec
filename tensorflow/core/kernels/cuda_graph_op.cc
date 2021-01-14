// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/30
// Description:
// CudaGraphOp, fetch a cudagraph instance and launch cudagraph async

#include <algorithm>

#include "tensorflow/core/common_runtime/cuda_graph_meta.h"
#include "tensorflow/core/common_runtime/cuda_graph_mgr.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef std::function<void()> Callback;

typedef struct CudaGraphCbArgs {
  OpKernelContext* ctx_;
  CudaGraphMeta* meta_;
  int origin_batch_size_;
  Callback done_;

  CudaGraphCbArgs(OpKernelContext* ctx, CudaGraphMeta* meta, int origin_batch_size, Callback done) :
      ctx_(ctx),
      meta_(meta),
      origin_batch_size_(origin_batch_size),
      done_(done) {};
} CudaGraphCbArgs;

class CudaGraphOp : public AsyncOpKernel {
public:
  explicit CudaGraphOp(OpKernelConstruction* ctx);

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

private:
  std::string graph_name_;
  std::vector<std::string> feed_names_;
  std::vector<std::string> fetch_names_;
  std::vector<DataType> data_type_;
  std::vector<int> buckets_;
};

#define GET_ATTR(k, v) {                           \
  const NodeDef &def = ctx->def();                 \
  if (def.attr().find(#k) != def.attr().end()) {   \
    OP_REQUIRES_OK(ctx, ctx->GetAttr(#k, &v));     \
  }                                                \
}

void CUDART_CB CudaGraphCallback(cudaStream_t stream, 
                                 cudaError_t status, 
                                 void* data) {
  CudaGraphCbArgs* args = (CudaGraphCbArgs*)data;
  // todo: copy output tensor;
  OpKernelContext* ctx = args->ctx_;
  CudaGraphMeta* meta = args->meta_;
  CudaGraphMgr& mgr = CudaGraphMgr::Singleton();
  for (int i = 0; i < meta->output_tensors_.size(); ++i) {
    Tensor *output = nullptr;
    TensorShape shape = meta->output_tensors_[i].shape();
    shape.set_dim(0, args->origin_batch_size_);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(i, shape, &output));
    output->CopyFrom(meta->output_tensors_[i], shape); 
  }
  mgr.ReturnCudaGraphMeta( meta);
  args->done_();
  delete args;
}

CudaGraphOp::CudaGraphOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx)  {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("graph_name", &graph_name_));

  GET_ATTR(feed_names, feed_names_);
  GET_ATTR(fetch_names, fetch_names_);
  GET_ATTR(T1, data_type_);
  GET_ATTR(buckets, buckets_);
}

void CudaGraphOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  OP_REQUIRES_ASYNC(ctx, ctx->num_inputs() == feed_names_.size(),
              errors::Internal("Op input size must equal to feed_names size, ",
              ctx->num_inputs(), " .vs ", feed_names_.size()), done);

  OP_REQUIRES_ASYNC(ctx, ctx->num_outputs() == fetch_names_.size(),
              errors::Internal("Op input size must equal to fetch_names size, ",
              ctx->num_inputs(), " .vs ", fetch_names_.size()), done);

  int req_id = ctx->step_id(); // rtp will set session id as step id.
  const Tensor& input_0 = ctx->input(0);
  int batch_size = input_0.dim_size(0);
  auto upper_iter = std::upper_bound(buckets_.begin(), buckets_.end(), batch_size);
  // todo: optimize
  if (upper_iter != buckets_.begin() && *(upper_iter - 1) == batch_size) {
    --upper_iter;
  }
  OP_REQUIRES_ASYNC(ctx, upper_iter != buckets_.end(),
              errors::Internal("Batch size ", batch_size, " is exceed max bucket ",
              buckets_.back()), done);

  // step1. fetch metas
  cudaStream_t stream;
  CudaGraphMeta* meta;
  CudaGraphMgr& mgr = CudaGraphMgr::Singleton();
  OP_REQUIRES_OK_ASYNC(ctx, mgr.GetCudagraphMeta(graph_name_, *upper_iter, meta), done);
  OP_REQUIRES_OK_ASYNC(ctx, mgr.GetCudaStream(req_id, stream), done);

  // do h2d copies first
  // do not padding explictly
  for (int i = 0; i < feed_names_.size(); ++i) {
    const Tensor& input = ctx->input(i);
    const void* host_buffer;
    size_t ele_size = 1;
    if (input.dtype() == DT_HALF) {
      ele_size = 2;
      host_buffer = reinterpret_cast<const void*>(input.flat<Eigen::half>().data());
    } else if (input.dtype() == DT_FLOAT) {
      ele_size = 4;
      host_buffer = reinterpret_cast<const void*>(input.flat<float>().data());
    } else if (input.dtype() == DT_INT32) {
      ele_size = 4;
      host_buffer = reinterpret_cast<const void*>(input.flat<int>().data());
    } else if (input.dtype() == DT_BOOL) {
      ele_size = 1;
      host_buffer = reinterpret_cast<const void*>(input.flat<bool>().data());
    } else if (input.dtype() == DT_INT64) {
      ele_size = 8;
      host_buffer = reinterpret_cast<const void*>(input.flat<int64>().data());
    } else {
      std::cout << "Unsupported data type!" << std::endl;
      exit(1);
    }

    size_t num_elements = input.NumElements();
    size_t num_bytes = num_elements * ele_size;
    void* device_buffer = meta->src_dst_mapping_[i].second;
    OP_REQUIRES_ASYNC(ctx, cudaMemcpyAsync(device_buffer, host_buffer, num_bytes,
        cudaMemcpyHostToDevice, stream) == cudaSuccess, 
        errors::Internal("CudaMemCpy to ", i, " st tensor failed, device addr: ", device_buffer),
        done);
  }

  // run cuda graph instance
  cudaError_t ret = cudaGraphLaunch(meta->cuda_graph_instance_, stream);
  OP_REQUIRES_ASYNC(ctx, ret == cudaSuccess, 
        errors::Internal("cudagraph launch faild: ", ret), done);

  CudaGraphCbArgs* args = new CudaGraphCbArgs(ctx, meta, batch_size, done);
  ret = cudaStreamAddCallback(stream, CudaGraphCallback, (void *)(args),0); 
  OP_REQUIRES_ASYNC(ctx, ret == cudaSuccess, 
        errors::Internal("Add cuda callback failed: ", ret), done);
/*
  cudaEvent_t event;
   cudaEventCreateWithFlags(&event, cudaEventBlockingSync);
   cudaEventRecord(event, stream);
   cudaEventSynchronize(event);
   cudaEventDestroy(event);
  for (int i = 0; i < meta->output_tensors_.size(); ++i) {
    Tensor *output = nullptr;
    TensorShape shape = meta->output_tensors_[i].shape();
    shape.set_dim(0, batch_size);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(i, shape, &output));
    output->CopyFrom(meta->output_tensors_[i], shape); 
  }
  mgr.ReturnCudaGraphMeta( meta);
  done();
*/
  return;
}

REGISTER_KERNEL_BUILDER(Name("CudaGraph").Device(DEVICE_CPU), CudaGraphOp);

}
