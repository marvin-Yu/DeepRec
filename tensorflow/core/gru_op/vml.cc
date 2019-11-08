/*
 * \file vml.cc
 * \brief The VML routine on CPU Architecture
 */
#include "vml.h"
#include <cmath>

namespace tensorflow {

#ifndef DECLARE_VML_FUNCTION_IMPL
#define DECLARE_VML_FUNCTION_IMPL(T, FuncName, OriginalFunc)                 \
  template <>                                                                     \
  void FuncName<T>(const int N, const T* x, T* y) {      \
    for (auto i = 0; i < N; ++i) { \
      y[i] = OriginalFunc(x[i]);                                         \
    } \
  }
#endif

DECLARE_VML_FUNCTION_IMPL(float, VML_Exp, std::exp)
DECLARE_VML_FUNCTION_IMPL(double, VML_Exp, std::exp)

DECLARE_VML_FUNCTION_IMPL(float, VML_Log, std::log)
DECLARE_VML_FUNCTION_IMPL(double, VML_Log, std::log)

DECLARE_VML_FUNCTION_IMPL(float, VML_Cos, std::cos)
DECLARE_VML_FUNCTION_IMPL(double, VML_Cos, std::cos)

DECLARE_VML_FUNCTION_IMPL(float, VML_Acos, std::acos)
DECLARE_VML_FUNCTION_IMPL(double, VML_Acos, std::acos)

DECLARE_VML_FUNCTION_IMPL(float, VML_Sin, std::sin)
DECLARE_VML_FUNCTION_IMPL(double, VML_Sin, std::sin)

DECLARE_VML_FUNCTION_IMPL(float, VML_Asin, std::asin)
DECLARE_VML_FUNCTION_IMPL(double, VML_Asin, std::asin)

DECLARE_VML_FUNCTION_IMPL(float, VML_Tan, std::tan)
DECLARE_VML_FUNCTION_IMPL(double, VML_Tan, std::tan)

DECLARE_VML_FUNCTION_IMPL(float, VML_Tanh, std::tanh)
DECLARE_VML_FUNCTION_IMPL(double, VML_Tanh, std::tanh)

DECLARE_VML_FUNCTION_IMPL(float, VML_Atan, std::atan)
DECLARE_VML_FUNCTION_IMPL(double, VML_Atan, std::atan)

DECLARE_VML_FUNCTION_IMPL(float, VML_Abs, std::abs)
DECLARE_VML_FUNCTION_IMPL(double, VML_Abs, std::abs)

DECLARE_VML_FUNCTION_IMPL(float, VML_Sqrt, std::sqrt)
DECLARE_VML_FUNCTION_IMPL(double, VML_Sqrt, std::sqrt)

#undef DECLARE_VML_FUNCTION_IMPL

#ifndef DECLARE_VML_POWX_FUNCTION_IMPL
#define DECLARE_VML_POWX_FUNCTION_IMPL(T, OriginalFunc)                        \
  template <>                                                                  \
  void VML_Powx<T>(const int N, const T* a, T b, T* y) {  \
    for (auto i = 0; i < N; ++i) {                        \
      y[i] = OriginalFunc(a[i], b);                       \
    } \
  }
#endif

DECLARE_VML_POWX_FUNCTION_IMPL(float, std::pow)
DECLARE_VML_POWX_FUNCTION_IMPL(double, std::pow)

#undef DECLARE_VML_POWX_FUNCTION_IMPL

#ifndef DECLARE_VML_BINARY_OP_FUNCTION_IMPL
#define DECLARE_VML_BINARY_OP_FUNCTION_IMPL(T, FuncName, Operand)                        \
  template <>                                                                            \
  void FuncName<T>(const int N, const T* a, const T* b, T* y) { \
    for (auto i = 0; i < N; ++i) {                                                       \
      y[i] = a[i] Operand b[i];                                                          \
    }                                                                                    \
  }
#endif

DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Add, +)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Add, +)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Sub, -)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Sub, -)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Mul, *)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Mul, *)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Div, /)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Div, /)

#undef DECLARE_VML_BINARY_OP_FUNCTION_IMPL

#ifndef DECLARE_VML_SET_FUNCTION_IMPL
#define DECLARE_VML_SET_FUNCTION_IMPL(T)                                  \
  template <>                                                             \
  void VML_Set<T>(const int N, T* a, T v) {      \
    for (int i = 0; i < N; ++i) {                                         \
      a[i] = v;                                                           \
    }                                                                     \
  }
#endif

DECLARE_VML_SET_FUNCTION_IMPL(float)
DECLARE_VML_SET_FUNCTION_IMPL(double)

#undef DECLARE_VML_SET_FUNCTION_IMPL

#ifndef DECLARE_VML_SET2_FUNCTION_IMPL
#define DECLARE_VML_SET2_FUNCTION_IMPL(DstT, SrcT)                                             \
  template <>                                                                                  \
  void VML_Set<DstT, SrcT>(const int N, DstT* dst, const SrcT* src) { \
    for (int i = 0; i < N; ++i) {                                                              \
      dst[i] = src[i];                                                                         \
    }                                                                                          \
  }
#endif

DECLARE_VML_SET2_FUNCTION_IMPL(float, float)

#undef DECLARE_VML_SET2_FUNCTION_IMPL

#ifndef DECLARE_VML_WHERE_FUNCTION_IMPL
#define DECLARE_VML_WHERE_FUNCTION_IMPL(T1, T2)                                       \
  template <>                                                                         \
  void VML_Where<T1, T2>(const int N, const T1* condition,                \
                                     const T2* a, const T2* b, T2* y) {  \
    for (int i = 0; i < N; ++i) {                                                     \
      y[i] = condition[i] > 0 ? a[i] : b[i];                                          \
    }                                                                                 \
  }
#endif

DECLARE_VML_WHERE_FUNCTION_IMPL(int32_t, float)
DECLARE_VML_WHERE_FUNCTION_IMPL(int32_t, double)
DECLARE_VML_WHERE_FUNCTION_IMPL(int64_t, float)
DECLARE_VML_WHERE_FUNCTION_IMPL(int64_t, double)

#undef DECLARE_VML_WHERE_FUNCTION_IMPL


}  // namespace blaze

