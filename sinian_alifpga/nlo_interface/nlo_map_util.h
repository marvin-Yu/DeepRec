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

#ifndef NLO_INTERFACE_NLO_MAP_UTIL_H_
#define NLO_INTERFACE_NLO_MAP_UTIL_H_

#include <assert.h>

namespace sinian {

// Returns a const reference to the value associated with the given key if it
// exists, otherwise returns a const reference to the provided default value.
//
// WARNING: If a temporary object is passed as the default "value,"
// this function will return a reference to that temporary object,
// which will be destroyed at the end of the statement. A common
// example: if you have a map with string values, and you pass a char*
// as the default "value," either use the returned value immediately
// or store it in a string (not string&).
template <class Collection>
const typename Collection::value_type::second_type& FindWithDefault(
    const Collection& collection,
    const typename Collection::value_type::first_type& key,
    const typename Collection::value_type::second_type& value) {
  typename Collection::const_iterator it = collection.find(key);
  if (it == collection.end()) {
    return value;
  }
  return it->second;
}

// FindOrDie returns a const reference to the value associated with
// the given key if it exists. Crashes otherwise.
//
// This is intended as a replacement for operator[] as an rvalue (for reading)
// when the key is guaranteed to exist.
template <class Collection>
const typename Collection::value_type::second_type& NloMapFindOrDie(
    const Collection& collection,
    const typename Collection::value_type::first_type& key) {
  typename Collection::const_iterator it = collection.find(key);
  assert(it != collection.end());
  return it->second;
}

// Same as above, but returns a non-const reference.
template <class Collection>
typename Collection::value_type::second_type& NloMapFindOrDie(
    Collection& collection,  // NOLINT
    const typename Collection::value_type::first_type& key) {
  typename Collection::iterator it = collection.find(key);
  assert(it != collection.end());
  return it->second;
}

// Inserts the key-value pair into the collection. Dies if key was already
// present.
template <class Collection>
void NloMapInsertOrDie(Collection* const collection,
                 const typename Collection::value_type::first_type& key,
                 const typename Collection::value_type::second_type& data) {
  auto p = collection->insert(std::make_pair(key, data));
  assert(p.second);
}

// Returns true if and only if the given collection contains the given key.
template <class Collection, class Key>
bool NloMapContainsKey(const Collection& collection, const Key& key) {
  return collection.find(key) != collection.end();
}

}  // namespace sinian

#endif  // NLO_INTERFACE_NLO_MAP_UTIL_H_
