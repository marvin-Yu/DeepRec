/*
 * \file vml.h
 * \brief The VML routine on CPU Architecture
 */
#pragma once
#include <cstdint>


namespace tensorflow {

#ifndef DECLARE_VML_FUNCTION
#define DECLARE_VML_FUNCTION(name)                                \
  template <typename T>                            \
  void name(const int N, const T* x, T* y)
#endif

DECLARE_VML_FUNCTION(VML_Exp);
DECLARE_VML_FUNCTION(VML_Log);
DECLARE_VML_FUNCTION(VML_Cos);
DECLARE_VML_FUNCTION(VML_Acos);
DECLARE_VML_FUNCTION(VML_Sin);
DECLARE_VML_FUNCTION(VML_Asin);
DECLARE_VML_FUNCTION(VML_Tan);
DECLARE_VML_FUNCTION(VML_Atan);
DECLARE_VML_FUNCTION(VML_Tanh);
DECLARE_VML_FUNCTION(VML_Abs);
DECLARE_VML_FUNCTION(VML_Sqrt);

template <typename T>
void VML_Powx(const int N, const T* a, T b, T* y);

#ifndef DECLARE_VML_BINARY_OP_FUNCTION
#define DECLARE_VML_BINARY_OP_FUNCTION(name)                      \
  template <typename T>                            \
  void name(const int N, const T* a, const T* b, T* y)
#endif

DECLARE_VML_BINARY_OP_FUNCTION(VML_Add);
DECLARE_VML_BINARY_OP_FUNCTION(VML_Sub);
DECLARE_VML_BINARY_OP_FUNCTION(VML_Mul);
DECLARE_VML_BINARY_OP_FUNCTION(VML_Div);

template <typename T>
void VML_Set(const int N, T* a, T v);

template <typename DstT, class SrcT>
void VML_Set(const int N, DstT* dst, const SrcT* src);

template <typename T1, typename T2>
void VML_Where(const int N, const T1* condition, const T2* a, const T2* b, T2* y);

template <typename T>
inline void VML_Sigmoid(const int n, const T* x, T* y) {
  for (int i = 0; i < n; i++) { y[i] = -x[i]; }
  VML_Exp<T>(n, y, y);
  for (int i = 0; i < n; i++) { y[i] = 1.0 / (1 + y[i]); }
}

template <typename T>
inline void VML_AddMul(const int n, const T* a, const T* b,
                       const T* c, T* z) {
  for (int i = 0; i < n; i++) { z[i] = a[i] + b[i] * c[i]; }
}

}  // namespace blaze
