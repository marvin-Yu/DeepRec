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

#ifndef NLO_INTERFACE_NLO_MODULE_H_
#define NLO_INTERFACE_NLO_MODULE_H_

#include <list>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "nlo_status.h"
#include "nlo_iterator_util.h"
#include "nlo_computation.h"
#include "nlo_instruction.h"
#include "nlo_name_uniquer.h"
#include "nlo_iterator_range.h"

namespace sinian {
namespace fpga {
class SinianFpgaTarget;
}

using std::string;
// Describes a compilation unit at the HLO level.
//
// A HLO module contains one or more HLO computations. The module contains one
// "entry" computation which produces the result. The module also includes any
// embedded computations used by instructions such as "map" and "reduce". All
// computations are owned by the module.
class NloModule {
 public:
  // Constructor without a versioned computation handle. This constructor should
  // only be used for NloModules used outside of the XLA service (eg
  // tests). The versioned handle is used by the service in the compilation
  // cache. A default configuration is created for this module.
  explicit NloModule(const string& name);

  // Adds an entry computation to the module. A module can only have one entry
  // computation. Returns a pointer to the newly added computation.
  NloComputation* AddEntryComputation(
      std::unique_ptr<NloComputation> computation);

  // Adds an embedded computation to the module.
  NloComputation* AddEmbeddedComputation(
      std::unique_ptr<NloComputation> computation);

  // Removes an embedded computation.
  NloStatus RemoveEmbeddedComputation(NloComputation* to_remove);

  NloStatus RemoveAllComputation();

  // Replaces all uses of computations that are keys of 'replacements' with
  // the corresponding values in 'replacements'. Replaces the entry computation,
  // if applicable.
  //
  // This function iterates over all instructions in the module to find
  // computations to replace. We could speed it up by keeping track of users of
  // computations.
  void ReplaceComputations(
      const std::unordered_map<NloComputation*, NloComputation*>& replacements);

  const string& name() const { return name_; }

  // Returns a deep copy of this module including all computations.
  std::unique_ptr<NloModule> Clone(const string& suffix = "clone") const;

  // Return a pointer to the entry computation of the module..
  NloComputation* entry_computation() const {
    assert(nullptr != entry_computation_);
    return entry_computation_;
  }

  // Gets the computations in this module.
  //
  // Returns a view of NloComputation*s, so you can iterate over this in the
  // natural way:
  //
  //   for (NloComputation* c : module->computations()) { ... }
  //
  nlo_iterator_range<NloUnwrappingIterator<
      std::vector<std::unique_ptr<NloComputation>>::const_iterator>>
  computations() const {
    return {MakeNloUnwrappingIterator(computations_.begin()),
            MakeNloUnwrappingIterator(computations_.end())};
  }
  nlo_iterator_range<NloUnwrappingIterator<
      std::vector<std::unique_ptr<NloComputation>>::iterator>>
  computations() {
    return {MakeNloUnwrappingIterator(computations_.begin()),
            MakeNloUnwrappingIterator(computations_.end())};
  }

  // Gets the number of computations in this module.
  int64_t computation_count() const { return computations_.size(); }

  // Compute and return a post order of all computations in the module. The sort
  // is defined like so: if computation A has an instruction which calls
  // computation B, then A will appear after B in the sort.
  std::list<NloComputation*> MakeComputationPostOrder() const;

  // Gets the computations in this module which aren't for fusion nodes.
  //
  // Postcondition: All computations in the returned list have
  // !IsFusionComputation().
  //
  // Note: Callers can and do rely on the return value here being a *snapshot*
  // of the module's non-fusion computations -- that is, it's OK to add or
  // remove computations from a module while iterating over
  // MakeNonfusionComputations().
  std::vector<NloComputation*> MakeNonfusionComputations() const;


  string ToString() const;

  // Outlines the given expression from the given computation.
  // instructions_to_outline contains the instructions that form the expression.
  //
  // Precondition: instructions in instructions_to_outline are in topological
  // order (root of outlined instructions last). TODO(jingyue): takes a set of
  // instructions and topologically sorts them.
  NloInstruction* OutlineExpressionFromComputation(
      std::vector<NloInstruction*> instructions_to_outline,
      const string& outlined_computation_name, NloComputation* computation);


  // Returns the unique name for a computation in this module.
  string GetUniqueCompuationName(const string& prefix) {
    return computation_name_uniquer_.GetUniqueName(prefix);
  }

  // Returns the NameUniquer for uniquing instruction names in this module.
  NloNameUniquer& instruction_name_uniquer() { return instruction_name_uniquer_; }

  // Assign a new unique dense id for an instruction
  int NewUniqueInstructionId() {
    int result = next_unique_id_;
    next_unique_id_++;
    return result;
  }

  // Returns the number of unique intruction ids given out.  All ids up to
  // this point are guaranteed to be in the range [0..NumUniqueInstructionIds())
  int NumUniqueInstructionIds() const { return next_unique_id_; }
  ::sinian::fpga::SinianFpgaTarget *GetTarget() { return target; };

 private:
  NloComputation* AddComputationInternal(
      std::unique_ptr<NloComputation> computation);
  void InitTarget();

  const string name_;
  NloComputation* entry_computation_ = nullptr;
  std::vector<std::unique_ptr<NloComputation>> computations_;

  // Unique name generator for computation and instruction names, which are
  // unique per module.
  NloNameUniquer computation_name_uniquer_{/*separator=*/"."};
  NloNameUniquer instruction_name_uniquer_{/*separator=*/"."};
  int next_unique_id_ = 0;
  ::sinian::fpga::SinianFpgaTarget *target;
};

}  // namespace sinian

#endif  // NLO_INTERFACE_NLO_MODULE_H_
