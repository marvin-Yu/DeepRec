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
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/tools/traffic/traffic.h"

namespace tensorflow {

typedef std::function<void()> Callback;

typedef struct CudaGraphCbArgs {
  OpKernelContext* ctx_;
  int waiting_slice_num_; // atomic
  Callback done_;
  cudaEvent_t pre_event_;
  bool single_slice_;
  std::mutex mtx_;
  std::vector<Tensor*> output_tensors_;
  bool has_failed_slice_;

  CudaGraphCbArgs(OpKernelContext* ctx, int slice_num, Callback done, cudaEvent_t pre_event, int output_num) :
      ctx_(ctx),
      waiting_slice_num_(slice_num),
      done_(done),
      pre_event_(pre_event) {
    single_slice_ = (slice_num == 1);
    output_tensors_.reserve(output_num);
    has_failed_slice_ = false;
  };
} CudaGraphCbArgs;

typedef struct CudaGraphCbSliceArgs {
  CudaGraphMeta* meta_;
  CudaGraphCbArgs* cb_args_;

  CudaGraphCbSliceArgs(CudaGraphCbArgs* args) :
      cb_args_(args) {};

} CudaGraphCbSliceArgs;

class CudaGraphOp : public AsyncOpKernel {
public:
  explicit CudaGraphOp(OpKernelConstruction* ctx);

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

private:
  void RecordTraffic(int batch_size);
  void ComputeAsyncSlice(OpKernelContext* ctx, DoneCallback done, size_t begin,
                         size_t end, size_t origin_batch_size, int req_id,
                         CudaGraphCbArgs* args, int slice_idx);

 private:
  std::string graph_name_;
  std::vector<std::string> feed_names_;
  std::vector<std::string> fetch_names_;
  std::vector<DataType> data_type_;
  std::vector<int> buckets_;
  bool empty_bucket_;
  int stream_num_;
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

bool CopyOutputTensorContent(const CudaGraphOutputInfo& meta_output_info,
                             Tensor* opTensor,
                             size_t batch_offset, size_t batch_size,
                             cudaStream_t stream) {
  if (meta_output_info.dtype_ != opTensor->dtype()) {
    return false;
  }

  size_t copy_ele_offset = meta_output_info.ele_num_per_dim0_ * batch_offset;
  size_t copy_eles = meta_output_info.ele_num_per_dim0_ * batch_size;

  size_t ele_size = 1;
  void* op_buffer = nullptr;
  if (opTensor->dtype() == DT_HALF) {
    ele_size = 2;
    op_buffer = reinterpret_cast<void*>(opTensor->flat<Eigen::half>().data() + copy_ele_offset);
  } else if (opTensor->dtype() == DT_FLOAT) {
    ele_size = 4;
    op_buffer = reinterpret_cast<void*>(opTensor->flat<float>().data() + copy_ele_offset);
  } else if (opTensor->dtype() == DT_INT32) {
    ele_size = 4;
    op_buffer = reinterpret_cast<void*>(opTensor->flat<int>().data() + copy_ele_offset);
  } else if (opTensor->dtype() == DT_BOOL) {
    ele_size = 1;
    op_buffer = reinterpret_cast<void*>(opTensor->flat<bool>().data() + copy_ele_offset);
  } else if (opTensor->dtype() == DT_INT64) {
    ele_size = 8;
    op_buffer = reinterpret_cast<void*>(opTensor->flat<int64>().data() + copy_ele_offset);
  } else {
    LOG(ERROR) << "Unsupported data type "
               << opTensor->dtype();  // todo: if callback, return meta
    return false;
  }

  size_t num_bytes = ele_size * copy_eles;
  if (cudaMemcpyAsync(op_buffer, meta_output_info.device_buffer_, num_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
    LOG(ERROR) << "Copy tensor context from meta to op output failed";
    return false;
  }
  return true;
}

void CopyRetAndReturnMetaWhenFail(CudaGraphCbSliceArgs* args) {
  args->cb_args_->has_failed_slice_ = true;

  if (!args->cb_args_->single_slice_) {
    do {
      std::lock_guard<std::mutex> lock(args->cb_args_->mtx_);
      --args->cb_args_->waiting_slice_num_;
      if (args->cb_args_->waiting_slice_num_ > 0) {
        delete args;
        return;
      }
    } while (0);
  }

  // if all slice finished
  args->cb_args_->done_();
  cudaEventDestroy(args->cb_args_->pre_event_);
  delete args->cb_args_;
  delete args;
  return;
}

void CopyRetAndReturnMeta(CudaGraphCbSliceArgs* args) {
  // if not all slice finished
  if (!args->cb_args_->single_slice_) {
    do {
      std::lock_guard<std::mutex> lock(args->cb_args_->mtx_);
      --args->cb_args_->waiting_slice_num_;
      if (args->cb_args_->waiting_slice_num_ > 0) {
        delete args;
        return;
      }
    } while (0);
  }

  // if all slice finished
  args->cb_args_->done_();
  cudaEventDestroy(args->cb_args_->pre_event_);
  delete args->cb_args_;
  delete args;
  return;
}

void CUDART_CB CudaGraphCallback(cudaStream_t stream, 
                                 cudaError_t status, 
                                 void* data) {
  CudaGraphCbSliceArgs* args = (CudaGraphCbSliceArgs*)data;
  OpKernelContext* ctx = args->cb_args_->ctx_;

  const DeviceBase::CpuWorkerThreads* threads = ctx->device()->tensorflow_cpu_worker_threads();
  if (threads == nullptr) {
    CopyRetAndReturnMeta(args);
  } else {
    threads->workers->Schedule(std::bind(&CopyRetAndReturnMeta, args));
  }
  return;
}

CudaGraphOp::CudaGraphOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx)  {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("graph_name", &graph_name_));

  GET_ATTR(feed_names, feed_names_);
  GET_ATTR(fetch_names, fetch_names_);
  GET_ATTR(T1, data_type_);
  GET_ATTR(buckets, buckets_);
  empty_bucket_ = (buckets_.size() == 0);
  CudaGraphMgr& mgr = CudaGraphMgr::Singleton();
  stream_num_ = mgr.GetStreamNum();
}

void CudaGraphOp::ComputeAsyncSlice(OpKernelContext* ctx, DoneCallback done,
                                    size_t begin, size_t end,
                                    size_t origin_batch_size, int req_id,
                                    CudaGraphCbArgs* args, int slice_idx) {
  int batch_size = end - begin;
  CudaGraphCbSliceArgs* slice_args = new CudaGraphCbSliceArgs(args);
  if (args->has_failed_slice_) {
    CopyRetAndReturnMetaWhenFail(slice_args);
    return;
  }

  auto upper_iter = std::upper_bound(buckets_.begin(), buckets_.end(), batch_size);
  // todo: optimize
  if (upper_iter != buckets_.begin() && *(upper_iter - 1) == batch_size) {
    --upper_iter;
  }
  OP_REQUIRES_ASYNC_WITH_ARGS(ctx, upper_iter != buckets_.end(), 
          errors::Internal("Batch size ", batch_size, " exceed to largest bucket"),
          CopyRetAndReturnMetaWhenFail, slice_args);

  // step1. fetch metas
  cudaStream_t stream;
  CudaGraphMeta* meta;
  CudaGraphMgr& mgr = CudaGraphMgr::Singleton();
  OP_REQUIRES_OK_ASYNC_WITH_ARGS(ctx, mgr.GetCudaStream(req_id, stream), 
          CopyRetAndReturnMetaWhenFail, slice_args); 
  OP_REQUIRES_OK_ASYNC_WITH_ARGS(ctx, mgr.GetCudagraphMeta(req_id, graph_name_, *upper_iter, meta), 
          CopyRetAndReturnMetaWhenFail, slice_args);
  slice_args->meta_ = meta;

  // allocate output when first slice 
  if (slice_idx == 0) {
    for (int i = 0; i < meta->output_infos_.size(); ++i) {
      Tensor* output = nullptr;
      TensorShape shape = meta->output_infos_[i].shape_;
      shape.set_dim(0, origin_batch_size);
      OP_REQUIRES_OK_ASYNC_WITH_ARGS(ctx, ctx->allocate_output(i, shape, &output),
          CopyRetAndReturnMetaWhenFail, slice_args);
      args->output_tensors_.push_back(output);
    }
  }

  std::lock_guard<std::mutex> lock(meta->mutex_);
  // need wait event
  if (slice_idx < stream_num_) {
    cudaError_t ret = cudaStreamWaitEvent(stream, args->pre_event_, 0);
    OP_REQUIRES_ASYNC_WITH_ARGS(ctx, ret == cudaSuccess,
              errors::Internal("synchronize event failed.", cudaGetErrorString(ret)), 
              CopyRetAndReturnMetaWhenFail, slice_args);
  }

  for (int i = 0; i < feed_names_.size(); ++i) {
    int origin_dim0 = meta->input_dim0_[i];
    const Tensor& input = ctx->input(i);
    size_t dim0 = input.dim_size(0);
    size_t num_elements = input.NumElements();
    size_t ele_num_per_dim0 = num_elements / dim0;
    // if origin_dim0 is above 0, means input with fixed batch, do not slice
    int copy_offset = 0;
    if (origin_dim0 <= 0) {
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
      LOG(ERROR) << "Unsupported data type " <<  input.dtype(); // todo: if callback, return meta
      ctx->CtxFailureWithWarning(__FILE__, __LINE__, 
            errors::Internal("Unsupported data type ", input.dtype()));
      CopyRetAndReturnMetaWhenFail(slice_args);
      return;
    }

    // if origin_dim0 is above 0, means input with fixed batch, do not slice
    size_t num_bytes = 0;
    if (origin_dim0 > 0) {
      size_t real_batch = std::min((size_t)origin_dim0, dim0);
      num_bytes = ele_num_per_dim0 * ele_size * real_batch;
    } else {
      num_bytes = ele_num_per_dim0 * ele_size * batch_size;
    }

    void* device_buffer = meta->src_dst_mapping_[i].second;
    OP_REQUIRES_ASYNC_WITH_ARGS(ctx, cudaMemcpyAsync(device_buffer, host_buffer, num_bytes,
        cudaMemcpyDeviceToDevice, stream) == cudaSuccess, 
        errors::Internal("CudaMemCpy to ", i, " st tensor failed, device addr: ", device_buffer),
        CopyRetAndReturnMetaWhenFail, slice_args);
 }

  // run cuda graph instance
  cudaError_t ret = cudaGraphLaunch(meta->cuda_graph_instance_, stream);
  OP_REQUIRES_ASYNC_WITH_ARGS(ctx, ret == cudaSuccess, 
        errors::Internal("cudagraph launch faild: ", ret), 
        CopyRetAndReturnMetaWhenFail, slice_args);
  for (int i = 0; i < meta->output_infos_.size(); ++i) {
    OP_REQUIRES_ASYNC_WITH_ARGS(ctx, 
        CopyOutputTensorContent(meta->output_infos_[i], args->output_tensors_[i], begin, batch_size, stream),
        errors::Internal("cudagraph copy output content failed"),
        CopyRetAndReturnMetaWhenFail, slice_args);
  }
  ret = cudaStreamAddCallback(stream, CudaGraphCallback, (void *)(slice_args), 0); 
  OP_REQUIRES_ASYNC_WITH_ARGS(ctx, ret == cudaSuccess, 
        errors::Internal("Add cuda callback failed: ", ret), 
        CopyRetAndReturnMetaWhenFail, slice_args);
  return;
}

void CudaGraphOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  OP_REQUIRES_ASYNC(ctx, ctx->num_inputs() == feed_names_.size(),
              errors::Internal("Op input size must equal to feed_names size, ",
              ctx->num_inputs(), " .vs ", feed_names_.size()), done);

  OP_REQUIRES_ASYNC(ctx, ctx->num_outputs() == fetch_names_.size(),
              errors::Internal("Op input size must equal to fetch_names size, ",
              ctx->num_inputs(), " .vs ", fetch_names_.size()), done);
 
  OP_REQUIRES_ASYNC(ctx, !empty_bucket_,
              errors::Internal("buckets for cuda graph is empty, check config."), done);

  // get compute stream and wait
  cudaStream_t* tf_raw_compute_stream = reinterpret_cast<cudaStream_t*>(ctx->op_device_context()->stream()->implementation()->GpuStreamMemberHack());
  OP_REQUIRES_ASYNC(ctx, tf_raw_compute_stream != nullptr,
              errors::Internal("get gpu compute stream failed."), done);
  cudaEvent_t event;
  cudaError_t ret = cudaEventCreateWithFlags(&event, cudaEventBlockingSync);
  OP_REQUIRES_ASYNC(ctx, ret == cudaSuccess,
              errors::Internal("create event failed, ", cudaGetErrorString(ret)), done);
  ret = cudaEventRecord(event, *tf_raw_compute_stream);
  OP_REQUIRES_ASYNC(ctx, ret == cudaSuccess,
              errors::Internal("record event failed, ", cudaGetErrorString(ret)), done);

  int req_id = ctx->step_id(); // rtp will set session id as step id.
  const Tensor& input_0 = ctx->input(0);
  int batch_size = input_0.dim_size(0);
 //  RecordTraffic(batch_size);
  OP_REQUIRES_ASYNC(ctx, batch_size > 0,
              errors::Internal("slice num should above 0"), done);

  int slice_num = (batch_size - 1) / buckets_.back() + 1;
  int slice_size = buckets_.back();
  CudaGraphCbArgs* args = new CudaGraphCbArgs(ctx, slice_num, done, event, fetch_names_.size());

  int begin = 0;
  int end = slice_size;
  for (int i = 0; i < slice_num - 1; ++i) {
    ComputeAsyncSlice(ctx, done, begin, end, batch_size, req_id + i, args, i);
    begin = end;
    end += slice_size;
  }
  end = batch_size;
  ComputeAsyncSlice(ctx, done, begin, end, batch_size, req_id + slice_num, args, slice_num - 1);
  return;
}

REGISTER_KERNEL_BUILDER(Name("CudaGraph").Device(DEVICE_GPU), CudaGraphOp);
}
