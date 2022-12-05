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

// Utilities for dealing with NloLiteral protobufs.

#ifndef NLO_INTERFACE_NLO_LITERAL_UTIL_H_
#define NLO_INTERFACE_NLO_LITERAL_UTIL_H_

#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "nlo_primitive_util.h"
#include "nlo_shape_util.h"

namespace sinian {

// Utility class for dealing with XLA literal values.  Most methods are
// templated by native (host) type which corresponds to a unique XLA
// NloPrimitiveType. See ComputationBuilder for details.  Not all primitive types
// defined in xla_data.proto have a corresponding native type or even have a
// storage location in the NloLiteral proto yet (for example, primitive type F16).
class NloLiteral {
 public:
  NloLiteral() {}

  NloLiteral(const NloLiteral& other) = default;
  NloLiteral(NloLiteral&&) = default;

  NloLiteral& operator=(const NloLiteral& other) = default;
  NloLiteral& operator=(NloLiteral&&) = default;

  // Literals are equal if they have compatible shapes and the same data
  // values. Layout is not checked.
  bool operator==(const NloLiteral& other) const;
  bool operator!=(const NloLiteral& other) const { return !(*this == other); }

  bool has_shape() const {
    return shape_.element_type() != NLO_PRIMITIVE_TYPE_INVALID;
  }

  // Return the nested literal at the given shape index.
  const NloLiteral& GetSubliteral(const NloShapeIndex& index) const;
  NloLiteral& GetSubliteral(const NloShapeIndex& index);

  void Clear() {
    shape_.Clear();
    u8s_.clear();
    s16s_.clear();
    s32s_.clear();
    s64s_.clear();
    u16s_.clear();
    u32s_.clear();
    u64s_.clear();
    f32s_.clear();
    f64s_.clear();
    tuple_literals_.clear();
  }

  int preds_size() const { return u8s().size(); }
  const std::vector<uint8_t>& preds() const {
    static_assert(sizeof(uint8_t) == sizeof(bool),
                  "The uint8_t and bool types should be the same size");
    return u8s_;
  }
  std::vector<uint8_t>* mutable_preds() {
    static_assert(sizeof(uint8_t) == sizeof(bool),
                  "The uint8_t and bool types should be the same size");
    return &u8s_;
  }

  int s16s_size() const { return s16s().size(); }
  int32_t s16s(int i) const { return s16s_[i]; }
  const std::vector<int16_t>& s16s() const { return s16s_; }
  std::vector<int16_t>* mutable_s16s() { return &s16s_; }

  int s32s_size() const { return s32s().size(); }
  int32_t s32s(int i) const { return s32s_[i]; }
  const std::vector<int32_t>& s32s() const { return s32s_; }
  std::vector<int32_t>* mutable_s32s() { return &s32s_; }
  void set_s32s(const std::vector<int32_t>& value) { s32s_ = value; }

  int s64s_size() const { return s64s().size(); }
  void add_s64s(int64_t value) { s64s_.push_back(value); }
  const std::vector<int64_t>& s64s() const { return s64s_; }
  std::vector<int64_t>* mutable_s64s() { return &s64s_; }

  int u16s_size() const { return u16s().size(); }
  uint32_t u16s(int i) const { return u16s_[i]; }
  const std::vector<uint16_t>& u16s() const { return u16s_; }
  std::vector<uint16_t>* mutable_u16s() { return &u16s_; }

  int u32s_size() const { return u32s().size(); }
  uint32_t u32s(int i) const { return u32s_[i]; }
  const std::vector<uint32_t>& u32s() const { return u32s_; }
  std::vector<uint32_t>* mutable_u32s() { return &u32s_; }
  void set_u32s(const std::vector<uint32_t>& value) { u32s_ = value; }

  int u64s_size() const { return u64s().size(); }
  const std::vector<uint64_t>& u64s() const { return u64s_; }
  std::vector<uint64_t>* mutable_u64s() { return &u64s_; }

  int f32s_size() const { return f32s().size(); }
  float f32s(int i) const { return f32s_[i]; }
  void set_f32s(const std::vector<float>& value) { f32s_ = value; }

  void add_f32s(float value) { f32s_.push_back(value); }
  const std::vector<float>& f32s() const { return f32s_; }
  std::vector<float>& f32s() { return f32s_; }
  std::vector<float>* mutable_f32s() { return &f32s_; }

  int f64s_size() const { return f64s().size(); }
  const std::vector<double>& f64s() const { return f64s_; }
  std::vector<double>* mutable_f64s() { return &f64s_; }

  int tuple_literals_size() const { return tuple_literals().size(); }
  const NloLiteral& tuple_literals(int i) const { return tuple_literals_[i]; }
  NloLiteral* add_tuple_literals() {
    tuple_literals_.push_back(NloLiteral());
    return &tuple_literals_.back();
  }
  std::vector<NloLiteral>* mutable_tuple_literals() { return &tuple_literals_; }
  const std::vector<NloLiteral>& tuple_literals() const { return tuple_literals_; }

  int u8s_size() const { return u8s().size(); }
  const std::vector<uint8_t>& u8s() const { return u8s_; }
  void set_u8s(const std::vector<uint8_t>& value) { u8s_ = value; }

  std::string u8s_string() const { return std::string(u8s().begin(), u8s().end()); }

  std::vector<uint8_t>* mutable_u8s() { return &u8s_; }

  const NloShape& shape() const { return shape_; }
  NloShape* mutable_shape() { return &shape_; }

  // Creates a new literal of a given rank. To minimize ambiguity (for users
  // and the compiler) these CreateR[0-2] methods should explicitly specify the
  // native type. For example:
  //
  //  CreateR1<float>({1.0, 42.0});
  //  CreateR2<uint32_t>({{1, 2}, {3, 4}});
  //
  // The variants not ending with WithLayout use the default XLA layout for the
  // literal's linear representation in memory.
  template <typename NativeT>
  static std::unique_ptr<NloLiteral> CreateR0(NativeT value);
  template <typename NativeT>
  static std::unique_ptr<NloLiteral> CreateR1(
      std::vector<NativeT> values);
  template <typename NativeT>
  static std::unique_ptr<NloLiteral> CreateR2(
      std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  static std::unique_ptr<NloLiteral> CreateR3(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          values);
  template <typename NativeT>
  static std::unique_ptr<NloLiteral> CreateR4(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values);
  // Creates a new NloLiteral object with the shape specified as parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  static std::unique_ptr<NloLiteral> CreateFromShape(const NloShape& shape);

  // Creates a new NloLiteral object with its values havings the primitive_type
  // type, and with dimensions defined by the dimensions parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  static std::unique_ptr<NloLiteral> CreateFromDimensions(
      NloPrimitiveType primitive_type,
      std::vector<int64_t> dimensions);


  // Creates a new literal by reordering the dimensions of this literal.
  // The given `permutation` must be a permutation of the dimension numbers
  // in the original literal, and it specifies the order of the new dimensions
  // in the result literal (i.e., new_order[i] = old_order[permutation[i]]).
  // For example, a transpose call on a literal of shape [3 x 8 x 4] and
  // `permutation` = {2, 0, 1} returns a new literal of shape [4 x 3 x 8].
  std::unique_ptr<NloLiteral> Transpose(
      std::vector<int64_t> permutation) const;

  // Creates a sub-array from this literal by extracting the indices
  // [start_index, limit_index) of each dimension. The result literal has the
  // same rank and layout as for the given literal. The number of indices in
  // start_indices and limit_indices must be the rank of the literal, and the
  // indices follow the order of the dimensions.
  std::unique_ptr<NloLiteral> Slice(
      std::vector<int64_t> start_indices,
      std::vector<int64_t> limit_indices) const;

  // Creates a literal with a prepended dimension with bound "times"; e.g. a
  // f32[3x2] with times=4 will produce a f32[4x3x2] with the 3x2 from this
  // literal replicated four times.
  template <typename NativeT>
  std::unique_ptr<NloLiteral> Replicate(int64_t times) const;


  // Creates a literal value zero of the given primitive type.
  static NloLiteral Zero(NloPrimitiveType primitive_type);

  // Creates a literal value one of the given primitive type.
  static NloLiteral One(NloPrimitiveType primitive_type);

  // Creates a literal value containing the minimum value of the given
  // primitive type. For floating-point types, returns -inf.
  static NloLiteral MinValue(NloPrimitiveType primitive_type);

  // Creates a literal value containing the maximum value of the given
  // primitive type. For floating-point types, returns inf.
  static NloLiteral MaxValue(NloPrimitiveType primitive_type);

  // Creates a literal of the given shape where each element is `value`.
  template <typename NativeT>
  static std::unique_ptr<NloLiteral> CreateFullWithMonotonicDim0MajorLayout(
      std::vector<int64_t> dimensions, NativeT value);


  // Creates a literal that projects the (x, y) dimensions given in values into
  // the z dimension given by "projection".
  template <typename NativeT>
  static std::unique_ptr<NloLiteral> CreateR3Projected(
      std::initializer_list<std::initializer_list<NativeT>> values,
      int64_t projection);

  // Creates a literal that projects the (x, y) dimensions given in values into
  // the z and p dimensions given.
  template <typename NativeT>
  static std::unique_ptr<NloLiteral> CreateR4Projected(
      std::initializer_list<std::initializer_list<NativeT>> values,
      int64_t projection_p, int64_t projection_z);

  // Clones this literal into an owned unique_ptr version.
  std::unique_ptr<NloLiteral> CloneToUnique() const;

  // Returns the linear index of the given index within this literal's
  // element_type repeated field.
  int64_t LinearIndex(std::vector<int64_t> multi_index) const;

  // Gets or sets an element in the literal at the given index. The index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  NativeT Get(std::vector<int64_t> multi_index) const;
  template <typename NativeT>
  void Set(std::vector<int64_t> multi_index, NativeT value);

  // Returns a (Mutable)ArraySlice view of the array for this literal for the
  // given NativeT (e.g., float). These functions map native type to XLA
  // NloPrimitiveType via template specialization. The unspecialized forms below
  // aborts to handle the error case where the given native type does not map to
  // an XLA primitive type.
  template <typename NativeT>
  std::vector<NativeT> GetArraySlice() const {
    static_assert(!std::is_same<NativeT, NativeT>::value,
                  "Cannot map native type to primitive type.");
  }

  // Returns the element value at index (0, ..., 0), however many zeroes are
  // required for that index.
  template <typename NativeT>
  NativeT GetFirstElement() const {
    return Get<NativeT>(0);
  }

  // As Get(), but determines the correct type and converts the value
  // into text.
  std::string GetAsString(std::vector<int64_t> multi_index) const;
  std::string GetAsString(int64_t index) const;
  // Returns an identity matrix (rank 2) with the given row and column count.
  template <typename NativeT>
  static std::unique_ptr<NloLiteral> MakeIdentityR2(int64_t size);

  // Returns a tuple literal composed of given literals.
  static std::unique_ptr<NloLiteral> MakeTuple(
      std::vector<const NloLiteral*> elements);

  // As above, but intended to be invoked with move semantics; i.e.
  //
  //  std::vector<std::unique_ptr<NloLiteral>> elements = ...;
  //  auto result = NloLiteral::MakeTupleOwned(std::move(elements));
  //
  // This would have been declared as an overload, but there is ambiguity
  // in invocation between the above signature and this one.
  static std::unique_ptr<NloLiteral> MakeTupleOwned(
      std::vector<std::unique_ptr<NloLiteral>> elements);


  // Returns a string representation of the literal value.
  std::string ToString() const;

  // Invokes the "per cell" callback for each element in the provided
  // literal with the element's indices and a string representation of
  // the element's value.
  //
  // This function is useful if you want a polymorphic representation
  // of the tensor's elements (turning it to a string for something
  // like representation in a protobuf).
  void EachCellAsString(
      const std::function<void(std::vector<int64_t> indices,
                               const std::string& value)>& per_cell) const;
  template <typename NativeT>
  void EachCell(std::function<void(std::vector<int64_t> indices,
                                   NativeT value)>
                    per_cell) const;

  // Templated methods which populate the given repeated field in this literal
  // with the given value(s). The NloShape field of this literal is set
  // to match the array dimensions and type. Examples:
  //
  //   // Populate with floats.
  //   Array2D<float> float_values = ...
  //   literal.PopulateR2FromArray2D(values);
  //
  //   // Populate with int32s.
  //   literal.PopulateR2({{1, 2}, {3, 4}});
  //
  template <typename NativeT>
  void PopulateR0(NativeT values);
  template <typename NativeT>
  void PopulateR1(std::vector<NativeT> values);

  // Returns a pointer to the underlying vector corresponding to the NloLiteral's
  // shape.
  const void* InternalData() const;
  void* MutableInternalData();

  // Allocates space in the underlying vector of this literal sufficient to hold
  // num_elements of this literal's primitive type. Values in the vector are set
  // to zero. num_elements must equal the number of elements in the literal's
  // shape.
  void Reserve(int64_t num_elements);

  // Allocates space in the underlying vector of this literal sufficient to hold
  // num_elements of this literal's primitive type and sets each element in this
  // literal to the given value. num_elements must equal the number of elements
  // in this literal's shape.
  template <typename NativeT>
  void Resize(int64_t num_elements, NativeT value);

  // Returns whether every element in this literal is equal to value.
  //
  // value is an int8_t because we expect this to be called with small
  // compile-time constants (0, -1, etc.) and so that whatever value you pass
  // can be represented exactly by floating-point types as small as 16 bits.
  //
  // If value doesn't fit in this literal's type, returns false.  Values of 1/0
  // are considered equal to true/false; other values are not considered equal
  // to true.
  bool IsAll(int8_t value) const;

  // Like IsAll(const NloLiteral&, int8_t), except we check whether the literal is
  // equal to a particular floating-point number.
  //
  // If the literal is not a floating-point value, this always returns false.
  //
  // This casts value to the type of literal, then compares using ==.  The usual
  // admonishments about floating-point equality checks apply.  We expect you to
  // use this to check for values that can be expressed precisely as a float,
  // e.g. -0.5.
  bool IsAllFloat(float value) const;

  // Returns whether this literal is zero at the specified index. This literal
  // must be an array.
  bool IsZero(std::vector<int64_t> indices) const;

  // Gets or sets an element in the literal at the given index. The index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  NativeT Get(int64_t index) const;

 private:
  NloShape shape_;
  std::vector<uint8_t> u8s_;
  std::vector<int16_t> s16s_;
  std::vector<int32_t> s32s_;
  std::vector<int64_t> s64s_;
  std::vector<uint16_t> u16s_;
  std::vector<uint32_t> u32s_;
  std::vector<uint64_t> u64s_;
  std::vector<float> f32s_;
  std::vector<double> f64s_;
  std::vector<NloLiteral> tuple_literals_;
};


// Returns the linear index of the given index within this literal's
// element_type repeated field.
inline int64_t
NloLiteral::LinearIndex(std::vector<int64_t> multi_index) const {
  assert(multi_index.size() == (size_t)shape_.dimensions_size());
  // Scale factor holding the growing product of D{L(i)} terms.
  int64_t scale = 1;
  int64_t linear_index = 0;
  bool first = true;
  for (auto dimension : shape_.dimensions()) {
    if (first) {
      // Avoid two multiplies on the first loop iteration
      linear_index = multi_index[dimension];
      scale = shape_.dimensions(dimension);
      first = false;
    } else {
      linear_index += scale * multi_index[dimension];
      scale *= shape_.dimensions(dimension);
    }
  }
  return linear_index;
}

template <>
inline bool NloLiteral::Get<bool>(
    int64_t index) const {
  assert(shape().element_type() == NLO_PRED);
  assert(u8s().size() >= (size_t)index);
  return (bool)u8s()[index];
}

template <>
inline uint8_t NloLiteral::Get<uint8_t>(
    int64_t index) const {
  assert(shape().element_type() == NLO_U8);
  assert(u8s().size() >= (size_t)index);
  return u8s()[index];
}

template <>
inline int8_t NloLiteral::Get<int8_t>(
    int64_t index) const {
  assert(shape().element_type() == NLO_S8);
  assert(u8s().size() >= (size_t)index);
  return u8s()[index];
}

template <>
inline int16_t NloLiteral::Get<int16_t>(
    int64_t index) const {
  assert(shape().element_type() == NLO_S16);
  assert(s16s().size() >= (size_t)index);
  return s16s()[index];
}

template <>
inline int32_t NloLiteral::Get<int32_t>(
    int64_t index) const {
  assert(shape().element_type() == NLO_S32);
  assert(s32s().size() >= (size_t)index);
  return s32s()[index];
}

template <>
inline int64_t NloLiteral::Get<int64_t>(
    int64_t index) const {
  assert(shape().element_type() == NLO_S64);
  assert(s64s().size() >= (size_t)index);
  return s64s()[index];
}

template <>
inline uint16_t NloLiteral::Get<uint16_t>(
    int64_t index) const {
  assert(shape().element_type() == NLO_U16);
  assert(u16s().size() >= (size_t)index);
  return u16s()[index];
}

template <>
inline uint32_t NloLiteral::Get<uint32_t>(
    int64_t index) const {
  assert(shape().element_type() == NLO_U32);
  assert(u32s().size() >= (size_t)index);
  return u32s()[index];
}


template <>
inline uint64_t NloLiteral::Get<uint64_t>(
    int64_t index) const {
  assert(shape().element_type() == NLO_U64);
  assert(u64s().size() >= (size_t)index);
  return u64s()[index];
}

template <>
inline float NloLiteral::Get<float>(
    int64_t index) const {
  assert(shape().element_type() == NLO_F32);
  assert(f32s().size() >= (size_t)index);
  return f32s()[index];
}

template <>
inline double NloLiteral::Get<double>(
    int64_t index) const {
  assert(shape().element_type() == NLO_F64);
  assert(f64s().size() >= (size_t)index);
  return f64s()[index];
}

template <>
inline bool NloLiteral::Get<bool>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_U8);
  int64_t index = LinearIndex(multi_index);
  assert(u8s().size() >= (size_t)index);
  return (bool)u8s()[index];
}

template <>
inline uint8_t NloLiteral::Get<uint8_t>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_U8);
  int64_t index = LinearIndex(multi_index);
  assert(u8s().size() >= (size_t)index);
  return u8s()[index];
}

template <>
inline int8_t NloLiteral::Get<int8_t>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_S8);
  int64_t index = LinearIndex(multi_index);
  assert(u8s().size() >= (size_t)index);
  return u8s()[index];
}

template <>
inline int16_t NloLiteral::Get<int16_t>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_S16);
  int64_t index = LinearIndex(multi_index);
  assert(s16s().size() >= (size_t)index);
  return s16s()[index];
}

template <>
inline int32_t NloLiteral::Get<int32_t>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_S32);
  int64_t index = LinearIndex(multi_index);
  assert(s32s().size() >= (size_t)index);
  return s32s()[index];
}

template <>
inline int64_t NloLiteral::Get<int64_t>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_S64);
  int64_t index = LinearIndex(multi_index);
  assert(s64s().size() >= (size_t)index);
  return s64s()[index];
}

template <>
inline uint16_t NloLiteral::Get<uint16_t>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_U16);
  int64_t index = LinearIndex(multi_index);
  assert(u16s().size() >= (size_t)index);
  return u16s()[index];
}

template <>
inline uint32_t NloLiteral::Get<uint32_t>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_U32);
  int64_t index = LinearIndex(multi_index);
  assert(u32s().size() >= (size_t)index);
  return u32s()[index];
}


template <>
inline uint64_t NloLiteral::Get<uint64_t>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_U64);
  int64_t index = LinearIndex(multi_index);
  assert(u64s().size() >= (size_t)index);
  return u64s()[index];
}

template <>
inline float NloLiteral::Get<float>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_F32);
  int64_t index = LinearIndex(multi_index);
  assert(f32s().size() >= (size_t)index);
  return f32s()[index];
}

template <>
inline double NloLiteral::Get<double>(
    std::vector<int64_t> multi_index) const {
  assert(shape().element_type() == NLO_F64);
  int64_t index = LinearIndex(multi_index);
  assert(f64s().size() >= (size_t)index);
  return f64s()[index];
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         bool value) {
  int64_t linear_index = LinearIndex(multi_index);
  (*mutable_u8s())[linear_index] = value;
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         uint8_t value) {
  int64_t linear_index = LinearIndex(multi_index);
  (*mutable_u8s())[linear_index] = value;
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         int8_t value) {
  return Set<uint8_t>(multi_index, value);
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         int16_t value) {
  int64_t linear_index = LinearIndex(multi_index);
  (*mutable_s16s())[linear_index] = value;
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         int32_t value) {
  int64_t linear_index = LinearIndex(multi_index);
  (*mutable_s32s())[linear_index] = value;
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         int64_t value) {
  int64_t linear_index = LinearIndex(multi_index);
  (*mutable_s64s())[linear_index] = value;
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         uint16_t value) {
  int64_t linear_index = LinearIndex(multi_index);
  (*mutable_u16s())[linear_index] = value;
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         uint32_t value) {
  int64_t linear_index = LinearIndex(multi_index);
  (*mutable_u32s())[linear_index] = value;
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         uint64_t value) {
  int64_t linear_index = LinearIndex(multi_index);
  (*mutable_u64s())[linear_index] = value;
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         float value) {
  int64_t linear_index = LinearIndex(multi_index);
  (*mutable_f32s())[linear_index] = value;
}

template <>
inline void NloLiteral::Set(std::vector<int64_t> multi_index,
                         double value) {
  int64_t linear_index = LinearIndex(multi_index);
  (*mutable_f64s())[linear_index] = value;
}

std::ostream& operator<<(std::ostream& out, const NloLiteral& literal);

template <>
void NloLiteral::Resize<bool>(int64_t num_elements, bool value);

template <>
void NloLiteral::Resize<int8_t>(int64_t num_elements, int8_t value);

template <>
void NloLiteral::Resize<uint8_t>(int64_t num_elements, uint8_t value);

template <>
void NloLiteral::Resize<int32_t>(int64_t num_elements, int32_t value);

template <>
void NloLiteral::Resize<uint32_t>(int64_t num_elements, uint32_t value);

template <>
void NloLiteral::Resize<int64_t>(int64_t num_elements, int64_t value);

template <>
void NloLiteral::Resize<uint64_t>(int64_t num_elements, uint64_t value);

template <>
void NloLiteral::Resize<float>(int64_t num_elements, float value);

template <>
void NloLiteral::Resize<double>(int64_t num_elements, double value);
}  // namespace sinian

#endif  // NLO_INTERFACE_NLO_LITERAL_UTIL_H_
