#include "gru_func.h"

#include "vml.h"

#define EIGEN_USE_THREADS
#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename T>
inline void SetOutput(const size_t n, const T* z, const T* h,
                      const T* last_output, T* output) {
  for (int i = 0; i < n; i++) {
    output[i] = h[i] + z[i] * (last_output[i] - h[i]);
  }
}

template <typename T>
void Gemm(const CPUDevice& d, size_t m, size_t n, size_t k, const T* a,
          const T* b, T* c) {
  typename tensorflow::TTypes<const T>::Matrix a_matrix(a, m, k);
  typename tensorflow::TTypes<const T>::Matrix b_matrix(b, k, n);
  typename tensorflow::TTypes<T>::Matrix c_matrix(c, m, n);

  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
  dim_pair[0].first = 1;
  dim_pair[0].second = 0;
  c_matrix.device(d) = a_matrix.contract(b_matrix, dim_pair);
}

template <typename T>
void GRUKernel(const CPUDevice& d, int batch_size, int rounds, int elts, T* y,
               const T* x, const T* h2h, const T* i2h, const T* h2h_bias,
               const T* i2h_bias, T* act_p, T* preact_p) {
  if (rounds <= 0) {
    return;
  }
  memset(y, 0, batch_size * rounds * elts * sizeof(T));
  for (int b = 0; b < batch_size; b++) {
    Gemm<T>(d, rounds, elts * 3, elts, x + b * rounds * elts,
            i2h + b * elts * elts * 3, preact_p + b * rounds * elts * 3);
    T* y_batch = &y[b * rounds * elts];
    T* preact = &preact_p[b * rounds * elts * 3];
    bool preact_nonzero = false;
    for (int i = 0; i < rounds; i++) {
      if (!preact_nonzero) {
        for (int k = 0; k < elts; k++) {
          if (preact[k] != 0) {
            preact_nonzero = true;
            break;
          }
        }
      }
      if (preact_nonzero) {
        if (i > 0) {
          const T* prev_y = y_batch - elts;
          Gemm<T>(d, 1, elts * 3, elts, prev_y, h2h + b * elts * elts * 3,
                  act_p);
          VML_Add(elts * 3, act_p, h2h_bias + b * elts * 3, act_p);
          VML_Add(elts * 3, preact, i2h_bias + b * elts * 3, preact);
          VML_Add(elts * 2, preact, act_p, preact);
          VML_Sigmoid(elts * 2, preact, preact);
          VML_AddMul(elts, &preact[elts * 2], preact, &act_p[elts * 2],
                     &preact[elts * 2]);
          VML_Tanh(elts, &preact[elts * 2], &preact[elts * 2]);
          SetOutput(elts, &preact[elts], &preact[elts * 2], prev_y, y_batch);
        } else {
          VML_Add(elts * 3, preact, i2h_bias + b * elts * 3, preact);
          VML_Add(elts * 2, preact, h2h_bias + b * elts * 3, preact);
          VML_Sigmoid(elts * 2, preact, preact);
          VML_AddMul(elts, &preact[elts * 2], preact,
                     &h2h_bias[elts * 2 + b * elts * 3], &preact[elts * 2]);
          VML_Tanh(elts, &preact[elts * 2], &preact[elts * 2]);
          SetOutput(elts, &preact[elts], &preact[elts * 2], y_batch, y_batch);
        }
      } else {
        memset(y_batch, 0, elts * sizeof(T));
      }
      preact += elts * 3;
      y_batch += elts;
    }
  }
}

template <typename T>
struct GRUFunctor<CPUDevice, T> {
  Status operator()(const CPUDevice& d, OpKernelContext* context,
                    int batch_size, int rounds, int elts, T* y, const T* x,
                    const T* h2h, const T* i2h, const T* h2hBias,
                    const T* i2hBias);
};

template <typename T>
Status GRUFunctor<CPUDevice, T>::operator()(
    const CPUDevice& d, OpKernelContext* context, int batch_size, int rounds,
    int elts, T* y, const T* x, const T* h2h, const T* i2h, const T* h2hBias,
    const T* i2hBias) {
  VLOG(2) << "=== CPU GRUFunctor ===";

  Tensor act, preact;
  if (std::is_same<float, T>::value) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_FLOAT, TensorShape({batch_size, rounds * 3, elts}), &preact));
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_FLOAT,
        TensorShape({batch_size, elts *
                                     (alignN(elts, gru_weights_per_thread) /
                                      gru_weights_per_thread) *
                                     3 * 2}),
        &act));
  } else if (std::is_same<double, T>::value) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_DOUBLE, TensorShape({batch_size, rounds * 3, elts}), &preact));
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_DOUBLE,
        TensorShape({batch_size, elts *
                                     (alignN(elts, gru_weights_per_thread) /
                                      gru_weights_per_thread) *
                                     3 * 2}),
        &act));
  } else {
    return errors::InvalidArgument("Unsupported Datatype");
  }
  T* act_p = act.flat<T>().data();
  T* preact_p = preact.flat<T>().data();

  GRUKernel(d, batch_size, rounds, elts, y, x, h2h, i2h, h2hBias, i2hBias,
            act_p, preact_p);
  return Status::OK();
};

template struct GRUFunctor<CPUDevice, float>;

}  // namespace tensorflow
