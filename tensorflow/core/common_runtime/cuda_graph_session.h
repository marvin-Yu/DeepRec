#if GOOGLE_CUDA

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_CUDA_GRAPH_SESSION_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_CUDA_GRAPH_SESSION_H_

#include <cuda_runtime.h>

#include <condition_variable>
#include <mutex>
#include <stack>

#include "tensorflow/core/common_runtime/direct_session.h"

namespace tensorflow {

class CudaGraphSessionTest;
class CudaGraphSessionFactory;

class CudaGraphSession : public Session {
 public:
  CudaGraphSession(const SessionOptions& options, const DeviceMgr* device_mgr,
                   CudaGraphSessionFactory* factory);
  ~CudaGraphSession() override;
  Status Create(const GraphDef& graph) override;
  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_names,
             const std::vector<string>& target_nodes,
             std::vector<Tensor>* outputs) override;
  Status Run(const ::tensorflow::RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_names,
             const std::vector<string>& target_nodes,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override;
  Status Extend(const GraphDef& graph) override {
    return errors::Unimplemented(
        "Extend(const RunOptions& run_options, const GraphDef& graph) is not "
        "supported for this session.");
  }
  Status ListDevices(std::vector<DeviceAttributes>* response) override {
    return errors::Unimplemented(
        "Extend(const RunOptions& run_options, const GraphDef& graph) is not "
        "supported for this session.");
  }

  Status Close() override;

 private:
  Status InitCudaGraphInputs(int batch_size, std::vector<Tensor>& inputs, const cudaStream_t& stream);
  Status InitCallableInputs(
      const std::vector<std::pair<string, Tensor>>& inputs,
      std::vector<Tensor>& callable_inputs);
  Status InitInOutInfos(const GraphDef& graph, std::vector<std::string>& inputs,
                        const std::vector<std::string>& outputs);
  Status InitGPUInfo(const DeviceMgr* device_mgr);
  Status InitCallableOptions(CallableOptions& opts,
                             const std::vector<std::string>& inputs,
                             const std::vector<std::string>& outputs);
  Status CaptureCudaGraph(int batch_size,
                          const Session::CallableHandle& handle);
  Status CopyCudaGraphInput(
      const std::vector<Tensor>& cuda_graph_inputs,
      const std::vector<std::pair<std::string, Tensor>>& inputs,
      const cudaStream_t& stream);
  Status CopyTensorData(const Tensor* from, Tensor* to, size_t dataSize,
                        const cudaStream_t& stream);
  Status CopyCudaOutput(const std::vector<Tensor>& cuda_outputs,
                        const std::vector<string>& output_tensor_names,
                        std::vector<Tensor>* outputs, int batch_size,
                        const cudaStream_t& stream);
  template <class Shape>
  Status CheckShape(const Shape& fromShape, const TensorShape& toShape);
  bool IsCUDATensor(const Tensor* t);
  Status CheckInputsInfo(const std::vector<std::pair<string, Tensor>>& inputs,
                         int& batch_size);

 private:
  friend class CudaGraphSessionTest;

  std::unique_ptr<DirectSession> session_;
  struct CudaGraphContext {
    cudaGraph_t cuda_graph;
    cudaGraphExec_t cuda_graph_exec;
    std::map<std::string, std::pair<void*, size_t>> dst_map;
    std::vector<Tensor> outputs;
    std::vector<Tensor> inputs;
    cudaStream_t stream;
    ~CudaGraphContext() {
      cudaGraphExecDestroy(cuda_graph_exec);
      cudaGraphDestroy(cuda_graph);
      cudaStreamDestroy(stream);
    }
  };
  using CudaGraphContextPtr = std::unique_ptr<CudaGraphContext>;

  struct CudaGraphDispatcher {
   private:
    std::stack<CudaGraphContextPtr> ctxs_;
    std::mutex mu_;
    std::condition_variable cv_;

   public:
    CudaGraphContextPtr GetContext() {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this]() { return ctxs_.size() > 0; });
      CudaGraphContextPtr top = std::move(ctxs_.top());
      ctxs_.pop();
      assert(top.get() != nullptr);
      return std::move(top);
    }
    void PutContext(CudaGraphContextPtr& ctx) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        assert(ctx.get() != nullptr);
        ctxs_.push(std::move(ctx));
      }
      cv_.notify_one();
    }
  };
  using CudaGraphDispatcherPtr = std::unique_ptr<CudaGraphDispatcher>;
  static const std::set<DataType> unsupported_types;
  std::map<int, CudaGraphDispatcherPtr> ctx_dispatcher_map_;
  std::map<std::string, int> output_idxs_;
  std::map<std::string, int> input_idxs_;
  cudaStream_t capturing_stream_ = nullptr;
  std::map<std::string, std::pair<PartialTensorShape, DataType>> inputs_info_;
  std::unique_ptr<TensorHolder> tensor_holder_;
  Allocator* host_allocator_ = nullptr;
  Allocator* gpu_allocator_ = nullptr;
  std::string gpu_device_name_;
  std::atomic_flag has_inited_ = ATOMIC_FLAG_INIT;
  bool inited_succ_ = false;
  CudaGraphOptions options_;
  CudaGraphSessionFactory* const factory_;  // not owned
  // Device structures.
  const std::unique_ptr<const DeviceMgr> device_mgr_;
  // true if the Session has been Closed.
  mutex closed_lock_;
  Session::CallableHandle feed_gpu_fetch_gpu_;
  bool closed_ GUARDED_BY(closed_lock_) = false;

 private:
  CudaGraphContextPtr GetCudaGraphContext(int batch_size);
  void PutBackCudaGraphContext(int batch_size, CudaGraphContextPtr& ctx);
  Status RunCudaGraph(const std::vector<std::pair<string, Tensor>>& inputs,
                      const std::vector<string>& output_tensor_names,
                      std::vector<Tensor>* outputs, const CudaGraphContext* ctx,
                      int batch_size);
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_CUDA_GRAPH_SESSION_H_

#endif  // GOOGLE_CUDA
