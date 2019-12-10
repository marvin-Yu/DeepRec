//
// Created by qiaoxj on 2019-12-10.
//

#include "tensorflow/core/kernels/indicator_matmul_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace {
template <typename T>
inline se::DeviceMemory<T> AsDeviceMemory(const T* gpu_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(gpu_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

template <typename Scalar>
void RunGemmStridedBatched(OpKernelContext* context, bool trans_a, bool trans_b,
                           int64 m, int64 n, int64 k, Scalar alpha,
                           const se::DeviceMemory<Scalar>& a, int64 stride_a,
                           const se::DeviceMemory<Scalar>& b, int64 stride_b,
                           Scalar beta, se::DeviceMemory<Scalar>* c,
                           int64 stride_c, int64 batch_count) {
  int lda = trans_a ? m : k;
  int ldb = trans_b ? k : n;
  int ldc = n;
  auto trans_a_tf = trans_a ? se::blas::Transpose::kTranspose
                            : se::blas::Transpose::kNoTranspose;
  auto trans_b_tf = trans_b ? se::blas::Transpose::kTranspose
                            : se::blas::Transpose::kNoTranspose;
  auto* stream = context->op_device_context()->stream();
  bool blas_launch_status =
      stream
          ->ThenBlasGemmStridedBatched(trans_b_tf, trans_a_tf, n, m, k, alpha,
                                       b, ldb, stride_b, a, lda, stride_a, beta,
                                       c, ldc, stride_c, batch_count)
          .ok();
  if (!blas_launch_status) {
    context->SetStatus(errors::Internal(
        "Blas GemmStridedBatched launch failed : m=", m, ", n=", n, ", k=", k));
  }
}

template <typename Scalar>
void RunGemmBatched(OpKernelContext* context, bool trans_a, bool trans_b,
                    int64 m, int64 n, int64 k, Scalar alpha,
                    std::vector<se::DeviceMemory<Scalar>*>& a_ptrs,
                    std::vector<se::DeviceMemory<Scalar>*>& b_ptrs, Scalar beta,
                    std::vector<se::DeviceMemory<Scalar>*>& c_ptrs,
                    int64 batch_count) {
  int lda = trans_a ? m : k;
  int ldb = trans_b ? k : n;
  int ldc = n;
  auto trans_a_tf = trans_a ? se::blas::Transpose::kTranspose
                            : se::blas::Transpose::kNoTranspose;
  auto trans_b_tf = trans_b ? se::blas::Transpose::kTranspose
                            : se::blas::Transpose::kNoTranspose;
  auto* stream = context->op_device_context()->stream();
  bool blas_launch_status =
      stream
          ->ThenBlasGemmBatched(trans_b_tf, trans_a_tf, n, m, k, alpha, b_ptrs,
                                ldb, a_ptrs, lda, beta, c_ptrs, ldc,
                                batch_count)
          .ok();

  if (!blas_launch_status) {
    context->SetStatus(errors::Internal(
        "Blas GemmBatched launch failed : m=", m, ", n=", n, ", k=", k));
  }
}

template <typename Scalar>
void LaunchIndicatorMatmul<GPUDevice, Scalar>::operator()(
    OpKernelContext* context, bool trans_a, bool trans_b, int64 m, int64 n,
    int64 k, const Tensor& in_a, const Tensor& in_b, const Tensor& indicator,
    Tensor* out, int64 batch_a, int64 batch_b) {
  LOG(INFO) << "LaunchIndicatorMatmul GPU entry";
  LOG(INFO) << indicator.DeviceSafeDebugString();
  auto a_ptr = AsDeviceMemory(in_a.template flat<Scalar>().data());
  auto b_ptr = AsDeviceMemory(in_b.template flat<Scalar>().data());
  auto out_ptr = AsDeviceMemory(out->template flat<Scalar>().data());
  if (batch_a == 1) {
    RunGemmStridedBatched<Scalar>(context, trans_a, trans_b, m, n, k,
                                  Scalar(1.0), a_ptr, 0, b_ptr, k * n,
                                  Scalar(0.0), &out_ptr, m * n, batch_b);
  } else {
    typedef se::DeviceMemory<Scalar> DeviceMemoryType;
    std::vector<DeviceMemoryType> a_device_memory;
    std::vector<DeviceMemoryType> b_device_memory;
    std::vector<DeviceMemoryType> c_device_memory;
    std::vector<DeviceMemoryType*> a_ptrs;
    std::vector<DeviceMemoryType*> b_ptrs;
    std::vector<DeviceMemoryType*> c_ptrs;
    a_ptrs.reserve(batch_b);
    b_ptrs.reserve(batch_b);
    c_ptrs.reserve(batch_b);
    auto* a_base_ptr = in_a.template flat<Scalar>().data();
    auto* b_base_ptr = in_b.template flat<Scalar>().data();
    auto* c_base_ptr = out->template flat<Scalar>().data();
    auto ind_data = indicator.flat<int>();
//    auto* ind_base_ptr = indicator.template flat<int>().data();
    for (int64 i = 0; i < batch_b; i++) {
      LOG(INFO) << "ind: " << ind_data(i);
      a_device_memory.push_back(
          AsDeviceMemory(a_base_ptr + ind_data(i) * m * k));
      b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
      c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
      a_ptrs.push_back(&a_device_memory.back());
      b_ptrs.push_back(&b_device_memory.back());
      c_ptrs.push_back(&c_device_memory.back());
    }
    LOG(INFO) << m << " "<< n << " " << k << " ";
    RunGemmBatched(context, trans_a, trans_b, m, n, k, Scalar(1.0), a_ptrs,
                   b_ptrs, Scalar(0.0), c_ptrs, batch_b);
  }
}

template struct LaunchIndicatorMatmul<GPUDevice, float>;
template struct LaunchIndicatorMatmul<GPUDevice, double>;
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
