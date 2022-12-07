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

#ifndef NLO_INTERFACE_NLO_REACHABILITY_H_
#define NLO_INTERFACE_NLO_REACHABILITY_H_
#include <cstddef>
#include <cstdint>
#include <list>
#include <vector>
#include <map>

#include "nlo_map_util.h"
#include "nlo_status.h"

namespace sinian {

class NloInstruction;

// A class for computing and representing reachability between NloInstructions.
class NloReachabilityMap {
 public:
  // Sets up an empty reachable matrix for the full set of instructions
  // specified in 'instructions'.
  explicit NloReachabilityMap(const std::list<NloInstruction*>& instructions);

  // Set the reachability set of 'instruction' to the union of the reachability
  // sets of 'inputs'. Upon return, IsReachable(x, instruction) where
  // 'x' is not 'instruction' will return true iff IsReachable(x, input) is true
  // for some 'input' in 'inputs'. Also sets 'instruction' to be reachable from
  // itself. Returns whether the reachability set of 'instruction' changed.
  bool SetReachabilityToUnion(
      std::vector<const NloInstruction*> inputs,
      const NloInstruction* instruction);

  // Sets entry so that IsReachable(a, b) will return true
  void SetReachable(const NloInstruction* a, const NloInstruction* b);

  // Returns true if "b" is reachable from "a"
  bool IsReachable(const NloInstruction* a, const NloInstruction* b) const;

  // Returns true if "b" is reachable from "a" or "a" is reachable from "b"
  bool IsConnected(const NloInstruction* a, const NloInstruction* b) const;

 private:
  // A bit-vector implementation specialized for this use case which provides a
  // fast bitwise OR operation not available in tensorflow::gtl::BitMap.
  class BitVector {
   public:
    BitVector() = default;
    BitVector(size_t size)
        : size_(size), vector_((size + kBits - 1) / kBits, 0) {}

    // Return the bit at the given index.
    bool Get(size_t index) const {
      assert(index >= 0 && index < size_);
      return vector_[index / kBits] & (1ull << (index % kBits));
    }

    // Set the bit at the given index.
    void Set(size_t index) {
      assert(index >= 0 && index < size_);
      vector_[index / kBits] |= 1ull << (index % kBits);
    }

    // Set this bitvector to the Logical OR of this bitvector and 'other'.
    void OrWith(const BitVector& other) {
      for (size_t i = 0; i < vector_.size(); ++i) {
        vector_[i] |= other.vector_[i];
      }
    }

    // Set the bitvector to all zeros.
    void SetToZero() { std::fill(vector_.begin(), vector_.end(), 0); }

    bool operator==(const BitVector& other) const {
      return vector_ == other.vector_;
    }
    bool operator!=(const BitVector& other) const {
      return vector_ != other.vector_;
    }

   private:
    using Word = uint64_t;
    static const size_t kBits = 64;

    // Number of bits in the bitvector.
    size_t size_;

    std::vector<Word> vector_;
  };

  // Return the bitvector storing the reachability-to of the given instruction.
  const BitVector& GetBitVector(const NloInstruction* instruction) const {
    return bit_vectors_[GetIndex(instruction)];
  }
  BitVector& GetBitVector(const NloInstruction* instruction) {
    return bit_vectors_[GetIndex(instruction)];
  }

  // Return the index of the given instruction. The value is used to index into
  // the vector of BitVectors and the BitVectors themselves.
  int GetIndex(const NloInstruction* instruction) const {
    return NloMapFindOrDie(indices_, instruction);
  }

  // The number of instructions in the reachability map.
  const size_t size_;

  // Dense assignment from NloInstruction* to number. These numbers index
  // into the bit_vectors_ vector and into the bits within a BitVector.
  std::map<const NloInstruction*, int> indices_;

  // Bitvectors holding the reachability to each instruction. The bit vector for
  // instruction X includes ones for each instruction which X is reachable from.
  std::vector<BitVector> bit_vectors_;

  // A temporary used by SetReachabilityToUnion to avoid an allocation with each
  // call to the method.
  BitVector tmp_bit_vector_;
};

}  // namespace sinian

#endif  // NLO_INTERFACE_NLO_REACHABILITY_H_
