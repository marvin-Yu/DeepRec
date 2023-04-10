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

#ifndef NLO_INTERFACE_NLO_COMPUTATION_H_
#define NLO_INTERFACE_NLO_COMPUTATION_H_

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nlo_iterator_util.h"
#include "dfs_nlo_visitor.h"
#include "dfs_nlo_visitor_with_default.h"
#include "nlo_instruction.h"
#include "nlo_reachability.h"

namespace sinian {
using std::string;
class NloModule;

// Describes a computation at the HLO level.
//
// An NloComputation contains a directed acyclic graph of HLO instructions. The
// computation has a single root instruction which produces the output of the
// computation.
class NloComputation {
 public:
  // Builder class for NloComputation.
  class Builder {
   public:
    explicit Builder(const string& name,
                     NloInstruction* fusion_instruction = nullptr)
        : name_(name),
          last_added_instruction_(nullptr),
          fusion_instruction_(fusion_instruction) {}

    // Build and return an NloComputation. The parameter root_instruction
    // specifies the already-added instruction to use as the root. If
    // root_instruction is nullptr then use the last added instruction as the
    // root.
    std::unique_ptr<NloComputation> Build(
        NloInstruction* root_instruction = nullptr);
    NloComputation* BuildNloComputation(
        NloInstruction* root_instruction = nullptr);

    NloInstruction* AddInstruction(
        std::unique_ptr<NloInstruction> instruction) {
      instructions_.push_back(std::move(instruction));
      last_added_instruction_ = instructions_.back().get();
      return last_added_instruction_;
    }

   private:
    const string name_;
    NloInstruction* last_added_instruction_;
    NloInstruction* fusion_instruction_;
    std::vector<std::unique_ptr<NloInstruction>> instructions_;
  };

  // Add an instruction to the computation. The computation takes ownership of
  // the instruction.
  NloInstruction* AddInstruction(std::unique_ptr<NloInstruction> instruction);

  // Remove the param_no'th parameter from the computation.
  // Note this is only applicatable to the computation for the fusion
  // instruction.
  NloStatus RemoveParameter(int64_t param_no);

  // Add new parameter instruction to the computation.
  // This should be a new parameter. Instruction will be appended to parameters
  // and inserted to the instruction list.
  NloInstruction* AddParameter(std::unique_ptr<NloInstruction> instruction);

  // Remove an instruction from the computation. The instruction must have no
  // users. Instruction is deallocated with this call.
  NloStatus RemoveInstruction(NloInstruction* instruction);

  // Remove an instruction from the computation and also transitively any
  // operand that has no users post removing an instruction. The instruction
  // must have no users. Instruction is deallocated with this call.
  NloStatus RemoveInstructionAndUnusedOperands(NloInstruction* instruction);

  // Set the root of the computation to the given instruction. The instruction
  // must have already been added to the computation and have the same shape as
  // the result of the computation for non fusion computations.
  void set_root_instruction(NloInstruction* new_root_instruction);

  // Return the root instruction of the computation. The root instruction is the
  // instruction which produces the output of the computation.
  NloInstruction* root_instruction() const { return root_instruction_; }

  // Returns the number of parameters for this computation.
  int64_t num_parameters() const { return param_instructions_.size(); }

  // Returns the parameter instruction for the given parameter number.
  NloInstruction* parameter_instruction(int64_t param_no) const {
    assert(param_no >= 0);
    assert(param_no < static_cast<int64_t>(param_instructions_.size()));
    return param_instructions_[param_no];
  }

  const std::vector<NloInstruction*>& parameter_instructions() const {
    return param_instructions_;
  }

  const string& name() const { return name_; }

  // Use the given NameUniquer to select a unique name for the computation based
  // on the computation's existing name.
  void UniquifyName(NloNameUniquer* name_uniquer);

  // Return a string representation of the computation.
  string ToString(int nested_level = 0) const;


  // Gets the instructions in this computation.
  //
  // The returned type is a range of NloInstruction*s, so you can iterate over
  // it using a range-based for loop in the natural way:
  //
  //   for (NloInstruction* instr : computation->instructions()) { ... }
  //
  nlo_iterator_range<NloUnwrappingIterator<
      std::list<std::unique_ptr<NloInstruction>>::const_iterator>>
  instructions() const {
    return {MakeNloUnwrappingIterator(instructions_.begin()),
            MakeNloUnwrappingIterator(instructions_.end())};
  }
  nlo_iterator_range<
      NloUnwrappingIterator<std::list<std::unique_ptr<NloInstruction>>::iterator>>
  instructions() {
    return {MakeNloUnwrappingIterator(instructions_.begin()),
            MakeNloUnwrappingIterator(instructions_.end())};
  }

  // Compute and return a post-order of the instructions in the computation. In
  // this order, definitions of values always appear before their uses.
  std::list<NloInstruction*> MakeInstructionPostOrder() const;

  // Computes and returns the reachability between HLO instructions in the
  // computation. The returned NloReachabilityMap is constructed such that
  // NloReachabilityMap::IsReachable(a, b) returns true iff there exists a
  // directed path (from producer to consumer) from 'a' to 'b'. Both data
  // dependencies (operands) and control dependencies are considered for
  // reachability. Trivially an instruction is reachable from itself.
  std::unique_ptr<NloReachabilityMap> ComputeReachability() const;

  // Updates the given reachability map after the immediate predecessor set
  // (operands and control predecessors) of 'instruction' has changed.
  void UpdateReachabilityThroughInstruction(
      const NloInstruction* instruction, NloReachabilityMap* reachability_map);

  int64_t instruction_count() const { return instructions_.size(); }

  int64_t param_instruction_count() const { return param_instructions_.size(); }

  // Creates and returns a list of the embedded computations called by this
  // computation. This includes all embedded computations called directly or
  // transitively. The embedded computations are sorted such that if computation
  // A calls computation B (eg, via a map instruction) then A will appear after
  // B in the list.
  std::list<NloComputation*> MakeEmbeddedComputationsList() const;

  // Creates a fusion instruction containing the given instructions.
  // `fusion_kind` indicates the type of the fusion, e.g., loop fusion or fusion
  // into a library call. Instructions must be in reverse topological order
  // (root of the fused expression first). Replaces all uses of the original
  // root instruction with the fusion instruction. The original instructions are
  // removed if they have no uses after fusion (this is necessarily true for at
  // least the root).
  NloInstruction* CreateFusionInstruction(
      std::vector<NloInstruction*> instructions_to_fuse,
      NloInstruction::FusionKind fusion_kind);

  // Return whether `*this` and `other` are functionally equivalent.
  bool operator==(const NloComputation& other) const;

  // Replaces old instruction with newly created instruction. Removes old
  // instruction from computation. Updates uses and root instruction.
  NloStatus ReplaceWithNewInstruction(
      NloInstruction* old_instruction,
      std::unique_ptr<NloInstruction> new_instruction);

  // Replace old instruction with new instruction.  Updates uses and root
  // instruction. Removes old instruction from computation. Precondition:
  // old_instruction and new_instruction must have the compatible shapes.
  NloStatus ReplaceInstruction(NloInstruction* old_instruction,
                            NloInstruction* new_instruction);

  // Set/get the module containing this computation.
  void set_parent(NloModule* module) { parent_ = module; }
  const NloModule* parent() const { return parent_; }
  NloModule* parent() { return parent_; }

  // Visit every node in the computation in DFS post-order with the given
  // visitor. This is similar to calling NloInstruction::Accept on the root of
  // the computation except this method also visits instructions not reachable
  // via the root. The root instruction of the computation is visited last, and
  // the visitor's FinishVisit method is called once upon completion (with the
  // root instruction as the argument).
  NloStatus Accept(DfsNloVisitor* visitor) const;

  // Same as Accept() above, but the order of operand and control predecessor
  // visitation is determined by the given operand order; if compare(A, B) ==
  // true, A is visited before B.
  NloStatus AcceptWithOperandOrder(
      DfsNloVisitor* visitor,
      const NloInstruction::CompareFunction& operand_order) const;

  // Visit every node in the computation in the given order. 'order' must
  // be a topological sort of all instructions in the computation.
  NloStatus AcceptOrdered(DfsNloVisitor* visitor,
                       const std::vector<const NloInstruction*>& order) const;

  // Same as Accept() above, but the visitor is given as a function.
  NloStatus Accept(const NloFunctionVisitor::VisitorFunction& visitor_func) const;

  // Returns a deep copy of this computation including all instructions.
  std::unique_ptr<NloComputation> Clone(const string& suffix = "clone");

  // Returns true if the given instruction can be removed from the
  // computation. Instructions such as parameters and send/receive instructions
  // cannot be removed without violating invariants of the HLO computation or
  // module with the exception of fusion computation.  A parameter instruction
  // is removable for a fusion computation.
  bool IsRemovable(const NloInstruction* instruction);

  // Returns true if this computation has a side effect. A computation has a
  // side effect if it contains one or more instructions with a side effect.
  bool HasSideEffect() const;

  // Returns if this computation is a fusion computation.
  bool IsFusionComputation() const { return fusion_instruction_ != nullptr; }

  // Returns the owning fusion instruction, or nullptr if this is not a fusion
  // computation.
  NloInstruction* FusionInstruction() const { return fusion_instruction_; }

 private:
  explicit NloComputation(
      const string& name, int parameter_count,
      std::vector<std::unique_ptr<NloInstruction>>* instructions,
      NloInstruction* root_instruction,
      NloInstruction* fusion_instruction = nullptr);

  // Internal helper for adding instructions.
  NloInstruction* AddInstructionInternal(
      std::unique_ptr<NloInstruction> instruction);

  // Helper for setting the parent of instructions that are added to this
  // computation.
  void Reparent(NloInstruction* instruction);

  // Fuses HLOs in instructions_to_fuse into fusion_instruction.
  //
  // Pre-condition: fusion_instruction's opcode is kFusion.
  void FuseInstructionsInto(
      std::vector<NloInstruction*> instructions_to_fuse,
      NloInstruction* fusion_instruction);

  // Internal helper to collect unreachable roots.
  std::vector<NloInstruction*> CollectUnreachableRoots() const;

  string name_;
  NloInstruction* root_instruction_;

  // If this computation is a fusion computation, this field points to the
  // corresponding fusion instruction.  Otherwise, this is null.
  NloInstruction* fusion_instruction_;

  // Module containing this computation.
  NloModule* parent_ = nullptr;

  // Store instructions in std::list as they can be added and removed
  // arbitrarily and we want a stable iteration order. Keep a map from
  // instruction pointer to location in the list for fast lookup.
  using InstructionList = std::list<std::unique_ptr<NloInstruction>>;
  InstructionList instructions_;
  std::unordered_map<const NloInstruction*, InstructionList::iterator>
      instruction_iterators_;

  std::vector<NloInstruction*> param_instructions_;

};

}  // namespace sinian

#endif  // NLO_INTERFACE_NLO_COMPUTATION_H_
