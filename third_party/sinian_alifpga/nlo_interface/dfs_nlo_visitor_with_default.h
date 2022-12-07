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

#ifndef NLO_INTERFACE_DFS_NLO_VISITOR_WITH_DEFAULT_H_
#define NLO_INTERFACE_DFS_NLO_VISITOR_WITH_DEFAULT_H_

#include "nlo_literal_util.h"
#include "dfs_nlo_visitor.h"
#include "nlo_opcode.h"

namespace sinian {

class NloComputation;
class NloInstruction;

// DfsNloVisitor with default action based on the NloInstruction being visited.
class DfsNloVisitorWithDefault : public DfsNloVisitor {
 public:
  DfsNloVisitorWithDefault() {}
  ~DfsNloVisitorWithDefault() override {}

  // Default action performed on NloInstruction.
  virtual NloStatus DefaultAction(NloInstruction* hlo_instruction) = 0;

  NloStatus HandleElementwiseUnary(NloInstruction* hlo) override {
    return DefaultAction(hlo);
  }
  NloStatus HandleElementwiseBinary(NloInstruction* hlo) override {
    return DefaultAction(hlo);
  }

  NloStatus HandleBatchNormTraining(NloInstruction* hlo) override {
    return DefaultAction(hlo);
  }

  NloStatus HandleBatchNormInference(NloInstruction* hlo) override {
    return DefaultAction(hlo);
  }

  NloStatus HandleBatchNormGrad(NloInstruction* hlo) override {
    return DefaultAction(hlo);
  }

  NloStatus HandleClamp(NloInstruction* clamp, NloInstruction* /*min*/,
                     NloInstruction* /*arg*/,
                     NloInstruction* /*max*/) override {
    return DefaultAction(clamp);
  }
  NloStatus HandleConcatenate(
      NloInstruction* concatenate,
      std::vector<NloInstruction*> /*operands*/) override {
    return DefaultAction(concatenate);
  }
  NloStatus HandleConvert(NloInstruction* convert) override {
    return DefaultAction(convert);
  }
  NloStatus HandleCopy(NloInstruction* copy) override {
    return DefaultAction(copy);
  }
  NloStatus HandleSelect(NloInstruction* select, NloInstruction* /*pred*/,
                      NloInstruction* /*on_true*/,
                      NloInstruction* /*on_false*/) override {
    return DefaultAction(select);
  }
  NloStatus HandleDot(NloInstruction* dot, NloInstruction* /*lhs*/,
                   NloInstruction* /*rhs*/) override {
    return DefaultAction(dot);
  }
  NloStatus HandleCrossReplicaSum(NloInstruction* crs) override {
    return DefaultAction(crs);
  }
  NloStatus HandleCompare(NloInstruction* compare, NloOpcode /*opcode*/,
                       NloInstruction* /*lhs*/,
                       NloInstruction* /*rhs*/) override {
    return DefaultAction(compare);
  }
  NloStatus HandleInfeed(NloInstruction* infeed) override {
    return DefaultAction(infeed);
  }
  NloStatus HandleOutfeed(NloInstruction* outfeed) override {
    return DefaultAction(outfeed);
  }
  NloStatus HandleReverse(NloInstruction* reverse,
                       NloInstruction* /*operand*/) override {
    return DefaultAction(reverse);
  }
  NloStatus HandleSort(NloInstruction* sort,
                    NloInstruction* /*operand*/) override {
    return DefaultAction(sort);
  }
  NloStatus HandleConstant(NloInstruction* constant,
                        const NloLiteral& /*literal*/) override {
    return DefaultAction(constant);
  }
  NloStatus HandleGetTupleElement(NloInstruction* get_tuple_element,
                               NloInstruction* /*operand*/) override {
    return DefaultAction(get_tuple_element);
  }
  NloStatus HandleParameter(NloInstruction* parameter) override {
    return DefaultAction(parameter);
  }
  NloStatus HandleFusion(NloInstruction* fusion) override {
    return DefaultAction(fusion);
  }
  NloStatus HandleCall(NloInstruction* call) override {
    return DefaultAction(call);
  }
  NloStatus HandleCustomCall(
      NloInstruction* custom_call,
      std::vector<NloInstruction*> /*operands*/,
      std::string /*custom_call_target*/) override {
    return DefaultAction(custom_call);
  }
  NloStatus HandleSlice(NloInstruction* slice,
                     NloInstruction* /*operand*/) override {
    return DefaultAction(slice);
  }
  NloStatus HandleDynamicSlice(NloInstruction* dynamic_slice,
                            NloInstruction* /*operand*/,
                            NloInstruction* /*start_indices*/) override {
    return DefaultAction(dynamic_slice);
  }
  NloStatus HandleDynamicUpdateSlice(NloInstruction* dynamic_update_slice,
                                  NloInstruction* /*operand*/,
                                  NloInstruction* /*update*/,
                                  NloInstruction* /*start_indices*/) override {
    return DefaultAction(dynamic_update_slice);
  }
  NloStatus HandleTuple(
      NloInstruction* tuple,
      std::vector<NloInstruction*> /*operands*/) override {
    return DefaultAction(tuple);
  }
  NloStatus HandleMap(
      NloInstruction* map,
      std::vector<NloInstruction*> /*operands*/,
      NloComputation* /*function*/,
      std::vector<NloInstruction*> /*static_operands*/)
      override {
    return DefaultAction(map);
  }
  NloStatus HandleReduce(NloInstruction* reduce, NloInstruction* /*arg*/,
                      NloInstruction* /*init_value*/,
                      std::vector<int64_t> /*dimensions*/,
                      NloComputation* /*function*/) override {
    return DefaultAction(reduce);
  }
  NloStatus HandleSelectAndScatter(NloInstruction* select_and_scatter) override {
    return DefaultAction(select_and_scatter);
  }
  NloStatus HandleBitcast(NloInstruction* bitcast) override {
    return DefaultAction(bitcast);
  }
  NloStatus HandleBroadcast(NloInstruction* broadcast) override {
    return DefaultAction(broadcast);
  }
  NloStatus HandlePad(NloInstruction* pad) override { return DefaultAction(pad); }
  NloStatus HandleReshape(NloInstruction* reshape) override {
    return DefaultAction(reshape);
  }
  NloStatus HandleTranspose(NloInstruction* transpose) override {
    return DefaultAction(transpose);
  }
  NloStatus HandleWhile(NloInstruction* xla_while) override {
    return DefaultAction(xla_while);
  }
  NloStatus HandleSend(NloInstruction* send) override {
    return DefaultAction(send);
  }
  NloStatus HandleRecv(NloInstruction* recv) override {
    return DefaultAction(recv);
  }

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  NloStatus FinishVisit(NloInstruction* /*root*/) override { return NloStatus::OK(); }
};

// Helper class for Accept(VisitorFunction) which visits instructions in DFS
// order calling the given function at each instruction.
class NloFunctionVisitor : public DfsNloVisitorWithDefault {
 public:
  using VisitorFunction = std::function<NloStatus(NloInstruction*)>;
  explicit NloFunctionVisitor(VisitorFunction visitor_func)
      : visitor_func_(std::move(visitor_func)) {}

  NloStatus DefaultAction(NloInstruction* hlo_instruction) override {
    return visitor_func_(hlo_instruction);
  }

 private:
  VisitorFunction visitor_func_;
};

}  // namespace sinian

#endif  // NLO_INTERFACE_DFS_NLO_VISITOR_WITH_DEFAULT_H_
