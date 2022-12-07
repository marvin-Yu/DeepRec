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

#ifndef NLO_INTERFACE_NLO_ITERATOR_UTIL_H_
#define NLO_INTERFACE_NLO_ITERATOR_UTIL_H_

#include <iterator>
#include <utility>

namespace sinian {
template <typename NestedIter>
class NloUnwrappingIterator
    : public std::iterator<std::input_iterator_tag,
                           decltype(std::declval<NestedIter>()->get())> {
 private:
  NestedIter iter_;

 public:
  explicit NloUnwrappingIterator(NestedIter iter) : iter_(std::move(iter)) {}

  auto operator*() -> decltype(iter_->get()) { return iter_->get(); }
  auto operator-> () -> decltype(iter_->get()) { return iter_->get(); }
  NloUnwrappingIterator& operator++() {
    ++iter_;
    return *this;
  }
  NloUnwrappingIterator operator++(int) {
    NloUnwrappingIterator temp(iter_);
    operator++();
    return temp;
  }

  friend bool operator==(const NloUnwrappingIterator& a,
                         const NloUnwrappingIterator& b) {
    return a.iter_ == b.iter_;
  }

  friend bool operator!=(const NloUnwrappingIterator& a,
                         const NloUnwrappingIterator& b) {
    return !(a == b);
  }
};

template <typename NestedIter>
NloUnwrappingIterator<NestedIter> MakeNloUnwrappingIterator(NestedIter iter) {
  return NloUnwrappingIterator<NestedIter>(std::move(iter));
}

}  // namespace sinian

#endif  // NLO_INTERFACE_NLO_ITERATOR_UTIL_H_
