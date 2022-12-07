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

#include "third_party/sinian_alifpga/nlo_interface/nlo_instruction.h"
#include "third_party/sinian_alifpga/nlo_interface/dfs_nlo_visitor_with_default.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_computation.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_module.h"

namespace sinian {
/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateParameter(
    int64_t /* parameter_number */, const NloShape& /* shape */, const string& /* name */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateConstant(
    std::unique_ptr<NloLiteral> /* literal */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateNary(
    const NloShape& /* shape */, NloOpcode /* opcode */,
    std::vector<NloInstruction*> /* operands */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateUnary(
    const NloShape& /* shape */, NloOpcode /* opcode */, NloInstruction* /* operand */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateReduce(
  const NloShape& shape, NloInstruction* arg, NloInstruction* init_value,
  std::vector<int64_t> dimensions_to_reduce,
  NloComputation* reduce_computation) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateBinary(
    const NloShape& /* shape */, NloOpcode /* opcode */, NloInstruction* /* lhs */,
    NloInstruction* /* rhs */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateTernary(
    const NloShape& /* shape */, NloOpcode /* opcode */, NloInstruction* /* lhs */,
    NloInstruction* /* rhs */, NloInstruction* /* ehs */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateSlice(
    const NloShape& /* shape */, NloInstruction* /* operand */,
    std::vector<int64_t> /* start_indices */,
    std::vector<int64_t> /* limit_indices */,
    std::vector<int64_t> /* strides */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateConcatenate(
    const NloShape& /* shape */, std::vector<NloInstruction*> /* operands */,
    int64_t /* dimension */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction>
NloInstruction::CreateBatchNormTraining(const NloShape& /* shape */,
                                        NloInstruction* /* operand */,
                                        NloInstruction* /* scale */,
                                        NloInstruction* /* offset */, float /* epsilon */,
                                        int64_t /* feature_index */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateBroadcast(
    const NloShape& /* shape */, NloInstruction* /* operand */,
    std::vector<int64_t> /* broadcast_dimensions */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction>
NloInstruction::CreateBatchNormInference(
    const NloShape& /* shape */, NloInstruction* /* operand */, NloInstruction* /* scale */,
    NloInstruction* /* offset */, NloInstruction* /* mean */, NloInstruction* /* variance */,
    float /* epsilon */, int64_t /* feature_index */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateReshape(
    const NloShape& /* shape */, NloInstruction* /* operand */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateTranspose(
    const NloShape& /* shape */, NloInstruction* /* operand */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateCustomCall(
    const NloShape& /* shape */, std::vector<NloInstruction*> /* operands */,
    std::string /* custom_call_target */) {
  return nullptr;
}

/* static */ std::unique_ptr<NloInstruction> NloInstruction::CreateTuple(
    std::vector<NloInstruction*> /* elements */) {
  return nullptr;
}

NloInstruction::~NloInstruction() {}

NloInstruction* NloInstruction::fused_expression_root() const {
  return nullptr;
}

const NloLiteral& NloInstruction::literal() const {
  return *literal_;
}

const NloInstruction* NloInstruction::operand(int64_t /* i */) const {
  return nullptr;
}

NloInstruction* NloInstruction::mutable_operand(int64_t /* i */) {
  return nullptr;
}

const string& NloInstruction::custom_call_target() const {
  return custom_call_target_;
}

const NloShape& NloInstruction::shape() const {
  return shape_;
}
}  // namespace sinian
