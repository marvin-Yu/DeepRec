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

// Shapes are protobuf messages, so this utility header offers a bunch of
// functionality for querying / poking at them.

#ifndef NLO_INTERFACE_NLO_SHAPE_UTIL_H_
#define NLO_INTERFACE_NLO_SHAPE_UTIL_H_

#include <initializer_list>
#include <string>
#include <vector>
#include <assert.h>

#include "nlo_primitive_type.h"
#include "nlo_shape.h"

namespace sinian {

// An index for specifying a particular nested subshape within a shape. Used in
// NloShapeUtil::GetSubshape and other interfaces. Shapes are recursive data
// structures (trees) and NloShapeIndex defines a path through the tree where each
// element of NloShapeIndex indexes into a tuple (or nested tuple) within the
// shape. For a non-nested tuple, an index has a single element. For example,
// given a 3-element tuple (a, b, c) containing arrays a, b, and c, the index
// {1} corresponds to array b. For a nested tuple, the index can have more than
// one element. For the nested tuple (a, (b, c, d), e) below are the values
// corresponding to the given indices:
//
//   index {0}    : array a
//   index {1, 2} : array d
//   index {2}    : array e
//   index {0, 0} : invalid index (element at {0} is an array not a tuple)
//
// For indexing into array shapes, the index is always trivially empty, ie {}.
//
// NloShapeIndex is a trivial wrapper around std::vector with a minimum number of
// methods implemented.
class NloShapeIndex {
 public:
  NloShapeIndex() = default;
  NloShapeIndex(std::initializer_list<int64_t> init) : indices_(init) {}

  bool empty() const { return indices_.empty(); }
  size_t size() const { return indices_.size(); }
  void push_back(int64_t value) { indices_.push_back(value); }
  void pop_back() { indices_.pop_back(); }

  std::vector<int64_t>::const_iterator begin() const { return indices_.begin(); }
  std::vector<int64_t>::const_iterator end() const { return indices_.end(); }
  std::vector<int64_t>::iterator begin() { return indices_.begin(); }
  std::vector<int64_t>::iterator end() { return indices_.end(); }

  const int64_t* data() const { return indices_.data(); }

  const int64_t& operator[](size_t i) const { return indices_[i]; }
  int64_t& operator[](size_t i) { return indices_[i]; }

  bool operator==(const NloShapeIndex& other) const {
    return indices_ == other.indices_;
  }
  bool operator!=(const NloShapeIndex& other) const { return !(*this == other); }
  bool operator<(const NloShapeIndex& other) const {
    return indices_ < other.indices_;
  }

  std::string ToString() const;

 private:
  std::vector<int64_t> indices_;
};

// A view into a NloShapeIndex as above, with the cheap/easy ability to consume the
// value at the front of the view.
//
// NB! NloShapeIndexView does not own the memory backing the index array.
// The memory backing the index array should be owned by an object
// that lives longer than the NloShapeIndexView instances pointing into
// it.
class NloShapeIndexView {
 public:
  NloShapeIndexView(const NloShapeIndex& shape_index, int64_t offset = 0)
      : NloShapeIndexView(shape_index.data() + offset,
                       shape_index.data() + shape_index.size()) {
    assert((size_t)offset <= shape_index.size());
  }
  NloShapeIndexView(std::initializer_list<int64_t> indices)
      : NloShapeIndexView(indices.begin(), indices.end()) {}
  NloShapeIndexView(const NloShapeIndexView& other) = default;

  using iterator = const int64_t*;

  iterator begin() const { return begin_; }
  iterator end() const { return end_; }
  int64_t size() const { return std::distance(begin_, end_); }
  bool empty() const { return begin_ == end_; }
  int64_t front() const {
    assert(!empty());
    return *begin_;
  }
  NloShapeIndexView ConsumeFront() const {
    assert(!empty());
    auto new_begin = begin_;
    ++new_begin;
    return NloShapeIndexView(new_begin, end_);
  }

  std::string ToString() const;

 private:
  NloShapeIndexView(iterator begin, iterator end) : begin_(begin), end_(end) {}

  iterator begin_;
  iterator end_;
};

std::ostream& operator<<(std::ostream& out, const NloShapeIndex& shape_index);

// Namespaced collection of (static) shape utilities.
//
// These are all effectively convenience functions for testing/tweaking proto
// properties, which do invariant checks before / after the operation.
class NloShapeUtil {
 public:
  // Returns the number of elements are contained within the provided shape;
  // e.g. for rank 0 (scalars) the result is always 1.
  // Precondition: !IsTuple(shape)
  static int64_t ElementsIn(const NloShape& shape);

  // Returns true if 'shape' has zero elements.
  static bool HasZeroElements(const NloShape& shape);

  // Returns the number of bytes required for an allocation of shape.  The
  // |pointer_size| parameter is used for calculating the size of tuple
  // shapes. This includes only the size of the top-level buffer. For example, a
  // tuple is stored as an array of pointers to other buffers. In this case,
  // this method only returns the size of the pointer array.
  // Precondition: (!NloShapeUtil::IsTuple(shape) || pointer_size > 0) &&
  //               !NloShapeUtil::IsOpaque(shape)
  static int64_t ByteSizeOf(const NloShape& shape, int64_t pointer_size = -1);

  // Returns the number of bytes used to store the primitive_type.
  //
  // Precondition: !NloShapeUtil::IsOpaque(shape) && !NloShapeUtil::IsTuple(shape)
  static int64_t ByteSizeOfPrimitiveType(NloPrimitiveType primitive_type);

  // Returns a human-readable std::string that represents the given shape, with or
  // without layout. e.g. "f32[42x12] {0, 1}" or "f32[64]".
  static std::string HumanString(const NloShape& shape);

  // Returns whether the LHS and RHS shapes have the same dimensions; note: does
  // not check element type.
  static bool SameDimensions(const NloShape& lhs, const NloShape& rhs);

  // Returns whether the lhs and rhs shapes have the same element type.
  static bool SameElementType(const NloShape& lhs, const NloShape& rhs) {
    return lhs.element_type() == rhs.element_type();
  }

  // Returns true if the rank, dimension sizes, and element type are
  // identical. Layout is ignored. Tuple elements are compared recursively for
  // compatibility.
  static bool Compatible(const NloShape& lhs, const NloShape& rhs);

  // Returns the rank (number of dimensions) of the given shape.
  // Precondition: !IsTuple(shape)
  static int64_t Rank(const NloShape& shape);

  // Returns the number of dimensions for which the dimension is not (trivially)
  // 1. e.g., f32[2x1x1] has a true rank of 1D, the other dimensions are just
  // fluff. Note that zero dimensions are included in the true rank, e.g.,
  // f32[3,0,1] has a true rank of 2D.
  static int64_t TrueRank(const NloShape& shape);

  ////////////////////
  // Scalar-specific

  static bool IsScalar(const NloShape& shape) {
    return !IsTuple(shape) && !IsOpaque(shape) && Rank(shape) == 0;
  }
  static bool IsEffectiveScalar(const NloShape& shape) {
    return !IsTuple(shape) && !IsOpaque(shape) && TrueRank(shape) == 0;
  }
  static bool IsScalarF32(const NloShape& shape);

  // Extracts the size of the shape's dimension at dimension number
  // GetDimensionNumber(dimension_number).
  static int64_t GetDimension(const NloShape& shape, int64_t dimension_number);

  // Resolves a dimension number, supporting negative indexing.
  //
  // Negative indexing has similar semantics to Python. For an N-dimensional
  // array, dimension -1 is equivalent to dimension N-1, -2 is equivalent to
  // N-2, and so on.
  //
  // This function always returns a positive dimension number for any given
  // dimension_number (which itself can be negative).
  static int64_t GetDimensionNumber(const NloShape& shape, int64_t dimension_number);

  // Returns a shape with the same dimensions as the original, but with the
  // element type changed to type.
  static NloShape ChangeElementType(const NloShape& original, NloPrimitiveType type);

  // Creates a tuple shape from a slice of element shapes within the tuple.
  static NloShape MakeTupleShape(std::vector<NloShape> shapes);

  // Creates an opaque shape. These are generally used for threading a context
  // into a custom operation.
  static NloShape MakeOpaqueShape();

  // Appends a shape to the given tuple.
  static void AppendShapeToTuple(const NloShape& shape, NloShape* tuple_shape);

  // Appends a major dimension to the shape with the given bound.
  static void AppendMajorDimension(int bound, NloShape* shape);

  // Returns an empty tuple shape. Can be used to indicate side-effects.
  static NloShape MakeNil() { return MakeTupleShape({}); }

  // Constructs a new shape with the given element type and sequence of
  // dimensions.
  static NloShape MakeShape(NloPrimitiveType element_type,
                         std::vector<int64_t> dimensions);

  // for multiple way shape
  static NloShape MakeShape(NloPrimitiveType element_type,
                         std::vector<int64_t> dimensions,
                         std::vector<int64_t> way_numbers);

  // Constructs a new shape with the given minor_to_major order in its Layout.
  // Returns a value shape such that shape.has_layout().
  static NloShape MakeShapeWithLayout(
      NloPrimitiveType element_type, std::vector<int64_t> dimensions,
      std::vector<int64_t> minor_to_major);

  // Constructs a new shape with major-first layout.
  static NloShape MakeShapeWithMonotonicDim0MajorLayout(
      NloPrimitiveType element_type,
      std::vector<int64_t> dimensions);

  // Returns a new shape with major-first layout that has the same layout of
  // elements with a different shape.
  static NloShape NormalizeShapeToMonotonicDim0MajorLayout(const NloShape& shape);

  // As MakeShape, but the object to write to is passed in.
  static void PopulateShape(NloPrimitiveType element_type,
                            std::vector<int64_t> dimensions,
                            NloShape* shape);

  // for multiple way shape
  static void PopulateShape(NloPrimitiveType element_type,
                            std::vector<int64_t> dimensions,
                            std::vector<int64_t> way_numbers,
                            NloShape* shape);

  // Returns whether the element type of the shape is integral (signed or
  // unsigned). Note that predicates are not considered integral here, since
  // they are logical values.
  static bool ElementIsIntegral(const NloShape& shape);

  // Returns whether the element type of the shape is floating point.
  static bool ElementIsFloating(const NloShape& shape);

  // Returns whether the element type has the given bit width.
  static bool ElementHasBitWidth(const NloShape& shape, int bits);

  // Returns whether the element type of the shape is integral and has
  // the specified number of bits.
  static bool ElementIsIntegralWithBits(const NloShape& shape, int bits);

  // Returns whether the element type of the shape is signed. Note
  // that floating point numbers are signed.
  static bool ElementIsSigned(const NloShape& shape);

  // Returns whether the shape is a tuple.
  static bool IsTuple(const NloShape& shape) {
    return shape.element_type() == NLO_TUPLE;
  }

  // Returns whether the shape is an opaque value (i.e. an 'existential' typed
  // value that is passed to CustomCall operations).
  static bool IsOpaque(const NloShape& shape) {
    return shape.element_type() == NLO_OPAQUE;
  }

  // Returns whether the shape is an array.
  static bool IsArray(const NloShape& shape) {
    return !IsTuple(shape) && !IsOpaque(shape);
  }

  // Returns whether the shape is a tuple with at least one element which is
  // also a tuple.
  static bool IsNestedTuple(const NloShape& shape);

  // Returns true if shape is an empty tuple.
  static bool IsEmptyTuple(const NloShape& shape);

  // Returns true if shape is an empty tuple, or is an array with no elements.
  static bool IsNil(const NloShape& shape);

  // Returns the number of elements in the given tuple shape.
  // Precondition: IsTuple(shape)
  static int64_t TupleElementCount(const NloShape& shape);

  // Returns the tuple element shape at given index.
  // Precondition: IsTuple(shape) && TupleElementCount(shape) > index
  static const NloShape& GetTupleElementShape(const NloShape& shape, int64_t index);

  // Slices tuple elements in the range [start, limit) and returns a new tuple
  // shape. E.g. a tuple like (f32, s32, u32) would slice via 1,3 to (s32, u32).
  static NloShape SliceTuple(const NloShape& tuple, int64_t start, int64_t limit);

  // Shorthand for testing whether a shape is of a given element type and
  // sequence of dimensions.
  //
  // DEPRECATED: Use Equal() instead.
  static bool ShapeIs(const NloShape& shape, NloPrimitiveType element_type,
                      std::initializer_list<int64_t> dimensions);

  static bool Equal(const NloShape& lhs, const NloShape& rhs);
};

}  // namespace sinian

#endif  // NLO_INTERFACE_NLO_SHAPE_UTIL_H_
