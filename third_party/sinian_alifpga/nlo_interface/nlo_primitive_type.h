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
#ifndef NLO_INTERFACE_NLO_PRIMITIVE_TYPE_H_
#define NLO_INTERFACE_NLO_PRIMITIVE_TYPE_H_
namespace sinian {
// Primitive types are the individual values that can be held in rectangular
// multidimensional arrays. A description of the rectangular multidimensional
// array dimensions / primitive type is given by Shape, below.
enum NloPrimitiveType {
  // Invalid primitive type to serve as default.
  NLO_PRIMITIVE_TYPE_INVALID = 0,

  // Predicates are two-state booleans.
  NLO_PRED = 1,

  // Signed integral values of fixed width.
  NLO_S8 = 2,
  NLO_S16 = 3,
  NLO_S32 = 4,
  NLO_S64 = 5,

  // Unsigned integral values of fixed width.
  NLO_U8 = 6,
  NLO_U16 = 7,
  NLO_U32 = 8,
  NLO_U64 = 9,

  // Floating-point values of fixed width.
  //
  // Note: if f16s are not natively supported on the device, they will be
  // converted to f16 from f32 at arbirary points in the computation.
  // F16 = 10,
  NLO_F32 = 11,
  NLO_F64 = 12,

  // A tuple is a polymorphic sequence; e.g. a shape that holds different
  // sub-shapes. They are used for things like returning multiple values from a
  // computation; e.g. a computation that returns weights and biases may have a
  // signature that results in a tuple like (f32[784x2000], f32[2000])
  //
  // If a shape proto has the tuple element type, it may not have any entries
  // in the dimensions field.
  NLO_TUPLE = 13,

  // An opaque type used for passing context specific data to a custom
  // operation.
  NLO_OPAQUE = 14,
};
} // namespace sinian
#endif // NLO_INTERFACE_NLO_PRIMITIVE_TYPE_H_

