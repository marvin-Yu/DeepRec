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
#ifndef NLO_INTERFACE_NLO_SHAPE_H_
#define NLO_INTERFACE_NLO_SHAPE_H_
#include <vector>
#include "nlo_primitive_type.h"
namespace sinian {
class NloShape {
 public:
  NloShape() {
    type_ = NLO_PRIMITIVE_TYPE_INVALID;
  }
  NloShape(const std::vector<int64_t>& dims) : way_numbers_(dims.size(), 1) {
    type_ = NLO_PRIMITIVE_TYPE_INVALID;
    dims_ = dims;
  }
  ~NloShape() {}

  void Clear() {
    type_ = NLO_PRIMITIVE_TYPE_INVALID;
    dims_.clear();
    way_numbers_.clear();
  }

  int dimensions_size() const {
    return dims_.size();
  }

  int64_t dimensions(int index) const {
    assert((size_t)index <= dims_.size());
    return dims_[index];
  }

  const std::vector<int64_t>& dimensions() const {
    return dims_;
  }

  void set_dimensions(int index, int64_t value) {
    assert((size_t)index <= dims_.size());
    dims_[index] = value;
  }

  void add_dimensions(int64_t value) {
    dims_.emplace_back(value);
  }

  NloPrimitiveType element_type() const {
    return type_;
  }

  void set_element_type(NloPrimitiveType type) {
    type_ = type;
  }

  void add_tuple_shapes(const NloShape& shape) {
    tuple_shapes_.emplace_back(shape);
  }

  const NloShape& tuple_shapes(int index) const {
    assert((size_t)index <= tuple_shapes_.size());
    return tuple_shapes_[index];
  }

  const std::vector<NloShape>& tuple_shapes() const {
    return tuple_shapes_;
  }

  int tuple_shapes_size() const {
    return tuple_shapes_.size();
  }

  void set_way_number(int index, int64_t n) {
    assert((size_t)index < way_numbers_.size());
    way_numbers_[index] = n;
  }

  void add_way_number(int64_t n) {
    way_numbers_.emplace_back(n);
  }

  int64_t way_number(int index) const {
    assert((size_t)index < way_numbers_.size());
    return way_numbers_[index];
  }

  const std::vector<int64_t>& way_numbers() const {
    return way_numbers_;
  }

 private:
  NloPrimitiveType type_;
  std::vector<int64_t> dims_;
  std::vector<NloShape> tuple_shapes_;
  // for multiple way tensors
  std::vector<int64_t> way_numbers_;

  // const std::vector<bool> dynamics_;
};
} // namespace sinian
#endif  // NLO_INTERFACE_NLO_SHAPE_H_
