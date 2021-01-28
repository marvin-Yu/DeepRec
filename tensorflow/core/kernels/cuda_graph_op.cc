// Copyright (c) 2020, Alibaba Inc.
// All right reserved.
//
// Author: Zexin YAN <zexin.yzx@alibaba-inc.com>
// Created: 2020/12/30
// Description:
// CudaGraphOp, fetch a cudagraph instance and launch cudagraph async

#include <algorithm>
#include <mutex>

#include "tensorflow/core/common_runtime/cuda_graph_meta.h"
#include "tensorflow/core/common_runtime/cuda_graph_mgr.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/tools/traffic/traffic.h"

namespace tensorflow {

typedef std::function<void()> Callback;

typedef struct CudaGraphCbArgs {
  OpKernelContext* ctx_;
  int origin_batch_size_;
  int waiting_slice_num_; // atomic
  Callback done_;
  bool single_slice_;
  std::mutex mtx_;
  std::vector<std::vector<Tensor>> slice_output_tensor_;  // new row-

  CudaGraphCbArgs(OpKernelContext* ctx, int origin_batch_size, int slice_num, Callback done, int output_num) :
      ctx_(ctx),
      origin_batch_size_(origin_batch_size),
      waiting_slice_num_(slice_num),
      done_(done) {
    single_slice_ = (slice_num == 1);
    if (!single_slice_) {
      slice_output_tensor_.resize(output_num);
      for (int i = 0; i < output_num; ++i) {
        slice_output_tensor_[i].resize(slice_num);
      }
    }
  };
} CudaGraphCbArgs;

typedef struct CudaGraphCbSliceArgs {
  int slice_batch_;
  int slice_idx_;
  CudaGraphMeta* meta_;
  CudaGraphCbArgs* cb_args_;

  CudaGraphCbSliceArgs(int slice_batch, int slice_idx, CudaGraphMeta* meta, CudaGraphCbArgs* args) :
      slice_batch_(slice_batch),
      slice_idx_(slice_idx),
      meta_(meta),
      cb_args_(args) {};

} CudaGraphCbSliceArgs;

class CudaGraphOp : public AsyncOpKernel {
public:
  explicit CudaGraphOp(OpKernelConstruction* ctx);

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

private:
  void RecordTraffic(int batch_size);

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

void CudaGraphOp::RecordTraffic(int batch_size) {
  ::Traffic::Instance()->Record("CgBatchSize", "CudaGraph");
  if (batch_size <= 32) {
    ::Traffic::Instance()->Record("CgBatchSize-32", "CudaGraph");
  } else if (batch_size <= 64) {
    ::Traffic::Instance()->Record("CgBatchSize33-64", "CudaGraph");
  } else if (batch_size <= 96) {
    ::Traffic::Instance()->Record("CgBatchSize64-96", "CudaGraph");
  } else if (batch_size <= 96) {
    ::Traffic::Instance()->Record("CgBatchSize64-96", "CudaGraph");
  } else if (batch_size <= 128) {
    ::Traffic::Instance()->Record("CgBatchSize97-128", "CudaGraph");
  } else if (batch_size <= 160) {
    ::Traffic::Instance()->Record("CgBatchSize129-160", "CudaGraph");
  } else if (batch_size <= 192) {
    ::Traffic::Instance()->Record("CgBatchSize161-192", "CudaGraph");
  } else {
    ::Traffic::Instance()->Record("CgBatchSize192-", "CudaGraph");
  }
  return;
}

void CopyRetAndReturnMeta(CudaGraphCbSliceArgs* args) {
  OpKernelContext* ctx = args->cb_args_->ctx_;
  CudaGraphMeta* meta = args->meta_;
  CudaGraphMgr& mgr = CudaGraphMgr::Singleton();

  if (args->cb_args_->single_slice_) {
    for (int i = 0; i < meta->output_tensors_.size(); ++i) {
      Tensor* output = nullptr;
      TensorShape shape = meta->output_tensors_[i].shape();
      shape.set_dim(0, args->cb_args_->origin_batch_size_);
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, shape, &output));
      output->CopyFrom(meta->output_tensors_[i], shape);
    }
  } else {
    for (int i = 0; i < meta->output_tensors_.size(); ++i) {
      TensorShape shape = meta->output_tensors_[i].shape();
      shape.set_dim(0, args->slice_batch_);
      args->cb_args_->slice_output_tensor_[i][args->slice_idx_] output->CopyFrom(meta->output_tensors_[i], shape);
    }
    // if not all slice finished
    do {
      std::lock_guard<std::mutex> lock(args->mtx_);
      --args->waiting_slice_num_;
      if (args->waiting_slice_num_ > 0) {
        mgr.ReturnCudaGraphMeta(meta);
        delete args;
        return;
      }
    } while (0);

    // if all slice finished 
    for (int i = 0; i < fetch_names_.size(); ++i) {
      Tensor* output = nullptr;
      TensorShape shape = args->slice_metas_[0]->output_tensors_[i].shape();
      shape.set_dim(0, args->cb_args_->origin_batch_size_);
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, shape, &output));
      Concat(args->cb_args_->slice_output_tensor_[i], output);
    }
  }
  mgr.ReturnCudaGraphMeta(meta);
  args->cb_args_->done_();
  delete args->cb_args_;
  delete args;
  return;
}

void CUDART_CB CudaGraphCallback(cudaStream_t stream, 
                                 cudaError_t status, 
                                 void* data) {
  CudaGraphCbSliceArgs* args = (CudaGraphCbSliceArgs*)data;
  OpKernelContext* ctx = args->ctx_;

  const DeviceBase::CpuWorkerThreads* threads = ctx->device()->tensorflow_cpu_worker_threads();
  if (threads == nullptr) {
    CopyRetAndReturnMeta(args);
  } else {
    threads->workers->Schedule(std::bind(&CudaGraphCbSliceArgs, args));
  }
  return;
}

CudaGraphOp::CudaGraphOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx)  {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("graph_name", &graph_name_));

  GET_ATTR(feed_names, feed_names_);
  GET_ATTR(fetch_names, fetch_names_);
  GET_ATTR(T1, data_type_);
  GET_ATTR(buckets, buckets_);
}

void CudaGraphOp::ComputeAsyncSlice(OpKernelContext* ctx, 
                                    DoneCallback done, 
                                    size_t begin, 
                                    size_t end,
                                    size_t origin_batch_size,
                                    int req_id,
                                    CudaGraphCbArgs* args,
                                    int slice_idx) {
  int batch_size = end - begin;

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


  for (int i = 0; i < feed_names_.size(); ++i) {
    const Tensor& input = ctx->input(i);
    size_t dim0 = input.dim_size(0);
    size_t num_elements = input.NumElements();
    size_t ele_num_per_dim0 = num_elements / dim0;
    bool need_split = (dim0 == origin_batch_size);
    int copy_offset = 0;
    if (need_split) {
      copy_offset = ele_num_per_dim0 * begin;
    }

    const void* host_buffer;
    size_t ele_size = 1;
    if (input.dtype() == DT_HALF) {
      ele_size = 2;
      host_buffer = reinterpret_cast<const void*>(input.flat<Eigen::half>().data() + copy_offset);
    } else if (input.dtype() == DT_FLOAT) {
      ele_size = 4;
      host_buffer = reinterpret_cast<const void*>(input.flat<float>().data() + copy_offset);
    } else if (input.dtype() == DT_INT32) {
      ele_size = 4;
      host_buffer = reinterpret_cast<const void*>(input.flat<int>().data() + copy_offset);
    } else if (input.dtype() == DT_BOOL) {
      ele_size = 1;
      host_buffer = reinterpret_cast<const void*>(input.flat<bool>().data() + copy_offset);
    } else if (input.dtype() == DT_INT64) {
      ele_size = 8;
      host_buffer = reinterpret_cast<const void*>(input.flat<int64>().data() + copy_offset);
    } else {
      mgr.ReturnCudaGraphMeta(meta);
      OP_REQUIRES_ASYNC(ctx, false,
              errors::Internal("Unsupported data type ", input.dtype()), done); // todo: if callback, return meta
      return;
    }

    size_t num_elements = input.NumElements();
    size_t num_bytes = ele_num_per_dim0 * ele_size * (end - begin);
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

  CudaGraphCbSliceArgs* slice_args = new CudaGraphCbSliceArgs(end - begin, slice_idx, meta, args);
  ret = cudaStreamAddCallback(stream, CudaGraphCallback, (void *)(slice_args), 0); 
  OP_REQUIRES_ASYNC(ctx, ret == cudaSuccess, 
        errors::Internal("Add cuda callback failed: ", ret), done);
  return;

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
  RecordTraffic(batch_size);
  int slice_num = (batch_size - 1) / buckets_.back() + 1;
  OP_REQUIRES_ASYNC(ctx, slice_num > 0,
              errors::Internal("slice num should above 0"), done);
  int slice_size = batch_size / slice_num;

  // todo: init cb args
  CudaGraphCbArgs* args = new CudaGraphCbArgs(ctx, batch_size, slice_num, done, fetch_names_.size());

  for (int i = 0; i < slice_num; ++i) {
    int begin = i * slice_size;
    int end = std::min(batch_size, (i + 1) * slice_size);
    ComputeAsyncSlice(ctx, done, begin, end, batch_size, req_id + i, args, i);
  }
  return;
}

REGISTER_KERNEL_BUILDER(Name("CudaGraph").Device(DEVICE_CPU), CudaGraphOp);

}
