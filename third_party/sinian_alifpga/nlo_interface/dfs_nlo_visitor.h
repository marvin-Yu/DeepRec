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

#ifndef NLO_INTERFACE_DFS_NLO_VISITOR_H_
#define NLO_INTERFACE_DFS_NLO_VISITOR_H_

#include <vector>
#include <assert.h>

#include "nlo_status.h"
#include "nlo_literal_util.h"
#include "nlo_opcode.h"

namespace sinian {

class NloComputation;
class NloInstruction;

// A postorder depth-first NloInstruction visitor. When Handle* is called on an
// instruction, all its operands were already visited. User code can subclass
// this to iterate over an NloInstruction DAG. The Handle* routines have
// operands / data unpacked for ease of use in the visitor subclass.
//
// No instruction will ever be visited twice; however, the root instruction will
// be reported again when the traversal is done via a call to FinishVisit.
//
// A subclass must override at least
// (either HandleElementwiseUnary or all the Handle methods for unary ops) and
// (either HandleElementwiseBinary or all the Handle methods for binary ops)).
// The default Handle methods for (unary, binary) ops call
// (HandleElementwiseUnary, HandleElementwiseBinary).
// The default (HandleElementwiseUnary, HandleElementwiseBinary) return an
// "unimplemented" error status.
//
// Note: this may change to an iterator in the future for flexibility purposes.
//
// TODO(b/26548304): Stop passing in information about the visited
// instruction that is accessible from the instruction object itself.
class DfsNloVisitor {
 public:
  DfsNloVisitor() {}
  virtual ~DfsNloVisitor() {}

  // These routines are self-descriptive, see class comment for usage
  // information.

  virtual NloStatus HandleElementwiseUnary(NloInstruction* hlo);
  virtual NloStatus HandleElementwiseBinary(NloInstruction* hlo);
  virtual NloStatus HandleClamp(NloInstruction* clamp, NloInstruction* min,
                             NloInstruction* arg, NloInstruction* max) = 0;
  virtual NloStatus HandleSelect(NloInstruction* select, NloInstruction* pred,
                              NloInstruction* on_true,
                              NloInstruction* on_false) = 0;
  virtual NloStatus HandleMaximum(NloInstruction* maximum) {
    return HandleElementwiseBinary(maximum);
  }
  virtual NloStatus HandleMinimum(NloInstruction* minimum) {
    return HandleElementwiseBinary(minimum);
  }
  virtual NloStatus HandleConcatenate(
      NloInstruction* concatenate,
      std::vector<NloInstruction*> operands) = 0;
  virtual NloStatus HandleConvert(NloInstruction* convert) {
    return HandleElementwiseUnary(convert);
  }
  virtual NloStatus HandleCopy(NloInstruction* copy) {
    return HandleElementwiseUnary(copy);
  }
  virtual NloStatus HandleMultiply(NloInstruction* multiply, NloInstruction* lhs,
                                NloInstruction* rhs) {
    return HandleElementwiseBinary(multiply);
  }
  virtual NloStatus HandleDot(NloInstruction* dot, NloInstruction* lhs,
                           NloInstruction* rhs) = 0;
  virtual NloStatus HandlePower(NloInstruction* power, NloInstruction* lhs,
                             NloInstruction* rhs) {
    return HandleElementwiseBinary(power);
  }
  virtual NloStatus HandleCrossReplicaSum(NloInstruction* crs) = 0;
  virtual NloStatus HandleCompare(NloInstruction* compare, NloOpcode opcode,
                               NloInstruction* lhs, NloInstruction* rhs) {
    return HandleElementwiseBinary(compare);
  }
  virtual NloStatus HandleAdd(NloInstruction* add, NloInstruction* lhs,
                           NloInstruction* rhs) {
    return HandleElementwiseBinary(add);
  }
  virtual NloStatus HandleDivide(NloInstruction* divide, NloInstruction* lhs,
                              NloInstruction* rhs) {
    return HandleElementwiseBinary(divide);
  }
  virtual NloStatus HandleRemainder(NloInstruction* remainder, NloInstruction* lhs,
                                 NloInstruction* rhs) {
    return HandleElementwiseBinary(remainder);
  }
  virtual NloStatus HandleSubtract(NloInstruction* subtract, NloInstruction* lhs,
                                NloInstruction* rhs) {
    return HandleElementwiseBinary(subtract);
  }
  virtual NloStatus HandleAbs(NloInstruction* abs, NloInstruction* operand) {
    return HandleElementwiseUnary(abs);
  }
  virtual NloStatus HandleRound(NloInstruction* round) {
    return HandleElementwiseUnary(round);
  }
  virtual NloStatus HandleSign(NloInstruction* sign, NloInstruction* operand) {
    return HandleElementwiseUnary(sign);
  }
  virtual NloStatus HandleNegate(NloInstruction* negate, NloInstruction* operand) {
    return HandleElementwiseUnary(negate);
  }
  virtual NloStatus HandleExp(NloInstruction* exp, NloInstruction* operand) {
    return HandleElementwiseUnary(exp);
  }
  virtual NloStatus HandleFloor(NloInstruction* floor, NloInstruction* operand) {
    return HandleElementwiseUnary(floor);
  }
  virtual NloStatus HandleCeil(NloInstruction* ceil, NloInstruction* operand) {
    return HandleElementwiseUnary(ceil);
  }
  virtual NloStatus HandleLog(NloInstruction* log, NloInstruction* operand) {
    return HandleElementwiseUnary(log);
  }
  virtual NloStatus HandleCos(NloInstruction* cos, NloInstruction* operand) {
    return HandleElementwiseUnary(cos);
  }
  virtual NloStatus HandleSin(NloInstruction* sin, NloInstruction* operand) {
    return HandleElementwiseUnary(sin);
  }
  virtual NloStatus HandleTanh(NloInstruction* tanh, NloInstruction* operand) {
    return HandleElementwiseUnary(tanh);
  }
  virtual NloStatus HandleIsFinite(NloInstruction* is_finite,
                                NloInstruction* operand) {
    return HandleElementwiseUnary(is_finite);
  }
  virtual NloStatus HandleLogicalAnd(NloInstruction* logical_and,
                                  NloInstruction* lhs, NloInstruction* rhs) {
    return HandleElementwiseBinary(logical_and);
  }
  virtual NloStatus HandleLogicalNot(NloInstruction* logical_not,
                                  NloInstruction* operand) {
    return HandleElementwiseUnary(logical_not);
  }
  virtual NloStatus HandleLogicalOr(NloInstruction* logical_or,
                                 NloInstruction* lhs, NloInstruction* rhs) {
    return HandleElementwiseBinary(logical_or);
  }
  virtual NloStatus HandleReducePrecision(NloInstruction* reduce_precision) {
    return HandleElementwiseUnary(reduce_precision);
  }

  virtual NloStatus HandleInfeed(NloInstruction* infeed) = 0;
  virtual NloStatus HandleOutfeed(NloInstruction* outfeed) = 0;
  virtual NloStatus HandleReverse(NloInstruction* reverse,
                               NloInstruction* operand) = 0;
  virtual NloStatus HandleSort(NloInstruction* sort, NloInstruction* operand) = 0;
  virtual NloStatus HandleConstant(NloInstruction* constant,
                                const NloLiteral& literal) = 0;
  virtual NloStatus HandleGetTupleElement(NloInstruction* get_tuple_element,
                                       NloInstruction* operand) = 0;
  virtual NloStatus HandleReduce(NloInstruction* reduce, NloInstruction* arg,
                              NloInstruction* init_value,
                              std::vector<int64_t> dimensions,
                              NloComputation* function) = 0;
  virtual NloStatus HandleBitcast(NloInstruction* bitcast) = 0;
  virtual NloStatus HandleBroadcast(NloInstruction* broadcast) = 0;
  virtual NloStatus HandleReshape(NloInstruction* reshape) = 0;
  virtual NloStatus HandleTranspose(NloInstruction* transpose) = 0;
  virtual NloStatus HandleParameter(NloInstruction* parameter) = 0;
  virtual NloStatus HandleFusion(NloInstruction* fusion) = 0;
  virtual NloStatus HandleCall(NloInstruction* call) = 0;
  virtual NloStatus HandleCustomCall(
      NloInstruction* custom_call,
      std::vector<NloInstruction*> operands,
      std::string custom_call_target) = 0;
  virtual NloStatus HandleSlice(NloInstruction* slice,
                             NloInstruction* operand) = 0;
  virtual NloStatus HandleDynamicSlice(NloInstruction* dynamic_slice,
                                    NloInstruction* operand,
                                    NloInstruction* start_indices) = 0;
  virtual NloStatus HandleDynamicUpdateSlice(NloInstruction* dynamic_update_slice,
                                          NloInstruction* operand,
                                          NloInstruction* update,
                                          NloInstruction* start_indices) = 0;
  virtual NloStatus HandleTuple(
      NloInstruction* tuple,
      std::vector<NloInstruction*> operands) = 0;
  virtual NloStatus HandleMap(
      NloInstruction* map,
      std::vector<NloInstruction*> operands,
      NloComputation* function,
      std::vector<NloInstruction*> static_operands) = 0;
  virtual NloStatus HandleSelectAndScatter(NloInstruction* instruction) = 0;
  virtual NloStatus HandleWhile(NloInstruction* xla_while) = 0;

  virtual NloStatus HandlePad(NloInstruction* pad) = 0;

  virtual NloStatus HandleSend(NloInstruction* send) = 0;

  virtual NloStatus HandleRecv(NloInstruction* recv) = 0;

  virtual NloStatus HandleBatchNormTraining(
      NloInstruction* batch_norm_training) = 0;

  virtual NloStatus HandleBatchNormInference(
      NloInstruction* batch_norm_inference) = 0;

  virtual NloStatus HandleBatchNormGrad(NloInstruction* batch_norm_grad) = 0;

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  virtual NloStatus FinishVisit(NloInstruction* root) = 0;

  // 3 possible visitation states of HLO instructions. Each instruction's
  // state only flows one way: kNotVisited -> kVisiting -> kVisited.
  enum VisitState {
    kNotVisited = 0,
    kVisiting = 1,
    kVisited = 2,
  };

  VisitState GetVisitState(int id) { return visit_state_.GetState(id); }
  VisitState GetVisitState(const NloInstruction& instruction);

  // Resize internal state if necessary to hold state for ids <= num.
  // This call is purely a performance hint and can be omitted without
  // affecting correctness.
  void ReserveVisitStates(int num) { visit_state_.Reserve(num); }

  void SetVisitState(int id, VisitState state) {
    visit_state_.SetState(id, state);
  }

  // Sets the visitation state of the given instruction as kVisiting.
  //
  // Precondition: current state must be kNotVisited.
  void SetVisiting(const NloInstruction& instruction);

  // Sets the visitation state of the given instruction as kVisited.
  //
  // Precondition: current state must be either kNotVisited or kVisiting.
  void SetVisited(const NloInstruction& instruction);

  // Returns whether the state of the given instruction is kVisiting.
  bool IsVisiting(const NloInstruction& instruction) {
    return GetVisitState(instruction) == kVisiting;
  }

  // Returns whether the state of the given instruction is kVisited.
  bool DidVisit(const NloInstruction& instruction) {
    return GetVisitState(instruction) == kVisited;
  }

  // Returns whether the state of the given instruction is kNotVisited.
  bool NotVisited(const NloInstruction& instruction) {
    return GetVisitState(instruction) == kNotVisited;
  }

  // This method should be overridden by subclasses that wish to run some
  // operation on an op before its Handle* visitor method is called.
  //
  // For any HLO op, the order of calls is:
  //
  //   Preprocess(op);
  //   Handle/OpType/(op);
  //   Postprocess(op);
  //
  // Overriding methods should call DfsNloVisitor::Preprocess before doing their
  // own preprocessing.
  virtual NloStatus Preprocess(NloInstruction* hlo);

  // This method should be overridden by subclasses that wish to run some
  // operation on an op after its Handle* visitor method is called. See
  // Preprocess for more details.
  //
  // Overriding methods should call DfsNloVisitor::Postprocess after doing their
  // own postprocessing.
  virtual NloStatus Postprocess(NloInstruction* visited);

 private:
  class DFSVisitStates {
   public:
    DFSVisitStates() {}
    void Reserve(uint64_t num) {
      states_.reserve((num + kStatesPerWord - 1) / kStatesPerWord);
    }
    VisitState GetState(uint64_t id) {
      uint64_t word_index = id / kStatesPerWord;
      if (word_index >= states_.size()) {
        return VisitState::kNotVisited;
      }
      static_assert(static_cast<int>(VisitState::kVisited) < 3,
                    "VisitState must fit in two bits");
      uint64_t w = states_[word_index];
      uint32_t shift = 2 * (id % kStatesPerWord);  // 2 bits per state
      return static_cast<VisitState>((w >> shift) & 0x3);
    }
    void SetState(uint64_t id, VisitState state) {
      uint64_t word_index = id / kStatesPerWord;
      if (word_index >= states_.size()) {
        states_.resize(word_index + 1, 0);
      }
      uint64_t* w = &states_[word_index];
      uint32_t shift = 2 * (id % kStatesPerWord);  // 2 bits per state
      uint64_t mask = 0x3ull << shift;
      *w = (*w & ~mask) | (static_cast<uint64_t>(state) << shift);
      assert(GetState(id) == state);
    }

   private:
    static const uint32_t kStatesPerWord = sizeof(uint64_t) / 2 /*bits per entry*/;
    // Map from id to two-bit states.  We store 32 such states per 64-bit
    // value
    std::vector<uint64_t> states_;
  };

  DFSVisitStates visit_state_;
};

}  // namespace sinian

#endif  // NLO_INTERFACE_DFS_NLO_VISITOR_H_
