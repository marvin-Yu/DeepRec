/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Utilities for dealing with XLA primitive types.

#ifndef NLO_INTERFACE_NLO_PRIMITIVE_UTIL_H_
#define NLO_INTERFACE_NLO_PRIMITIVE_UTIL_H_

#include <type_traits>
#include <cstdint>
#include <string>

#include "nlo_primitive_type.h"

namespace sinian {
namespace nlo_primitive_util {
bool PrimitiveType_IsValid(int value);
std::string PrimitiveType_Name(NloPrimitiveType type);
const NloPrimitiveType NLO_PrimitiveType_MIN = NLO_PRIMITIVE_TYPE_INVALID;
const NloPrimitiveType NLO_PrimitiveType_MAX = NLO_OPAQUE;
const int NLO_PrimitiveType_ARRAYSIZE = NLO_PrimitiveType_MAX + 1;

// Returns the XLA primitive type (eg, F32) corresponding to the given
// template parameter native type (eg, float).
template <typename NativeT>
NloPrimitiveType NativeToPrimitiveType() {
  // Make the expression depend on the template parameter NativeT so
  // that this compile-time error only apperas if this function is
  // instantiated with some concrete type that is not specialized
  // below.
  static_assert(!std::is_same<NativeT, NativeT>::value,
                "Cannot map native type to primitive type.");
  return NLO_PRIMITIVE_TYPE_INVALID;
}

// Declarations of specializations for each native type which correspond to a
// XLA primitive type.
template <>
NloPrimitiveType NativeToPrimitiveType<bool>();

// Unsigned integer
template <>
NloPrimitiveType NativeToPrimitiveType<uint8_t>();

template <>
NloPrimitiveType NativeToPrimitiveType<uint16_t>();

template <>
NloPrimitiveType NativeToPrimitiveType<uint32_t>();

template <>
NloPrimitiveType NativeToPrimitiveType<uint64_t>();

// Signed integer
template <>
NloPrimitiveType NativeToPrimitiveType<int8_t>();

template <>
NloPrimitiveType NativeToPrimitiveType<int16_t>();

template <>
NloPrimitiveType NativeToPrimitiveType<int32_t>();

template <>
NloPrimitiveType NativeToPrimitiveType<int64_t>();

// Floating point
template <>
NloPrimitiveType NativeToPrimitiveType<float>();
template <>
NloPrimitiveType NativeToPrimitiveType<double>();

bool IsFloatingPointType(NloPrimitiveType type);

bool IsSignedIntegralType(NloPrimitiveType type);

bool IsUnsignedIntegralType(NloPrimitiveType type);

bool IsIntegralType(NloPrimitiveType type);

// Returns the number of bits in the representation for a given type.
int BitWidth(NloPrimitiveType type);

// Returns the native type (eg, float) corresponding to the given template
// parameter XLA primitive type (eg, F32).
template <NloPrimitiveType>
struct PrimitiveTypeToNative;

// Declarations of specializations for each native type which correspond to a
// XLA primitive type.
template <>
struct PrimitiveTypeToNative<NLO_PRED> {
  using type = bool;
};

// Unsigned integer
template <>
struct PrimitiveTypeToNative<NLO_U8> {
  using type = uint8_t;
};

template <>
struct PrimitiveTypeToNative<NLO_U16> {
  using type = uint16_t;
};

template <>
struct PrimitiveTypeToNative<NLO_U32> {
  using type = uint32_t;
};

template <>
struct PrimitiveTypeToNative<NLO_U64> {
  using type = uint64_t;
};

// Signed integer
template <>
struct PrimitiveTypeToNative<NLO_S8> {
  using type = int8_t;
};

template <>
struct PrimitiveTypeToNative<NLO_S16> {
  using type = int16_t;
};

template <>
struct PrimitiveTypeToNative<NLO_S32> {
  using type = int32_t;
};

template <>
struct PrimitiveTypeToNative<NLO_S64> {
  using type = int64_t;
};

// Floating point
template <>
struct PrimitiveTypeToNative<NLO_F32> {
  using type = float;
};
template <>
struct PrimitiveTypeToNative<NLO_F64> {
  using type = double;
};
}  // namespace nlo_primitive_util
}  // namespace sinian

#endif  // NLO_INTERFACE_NLO_PRIMITIVE_UTIL_H_
