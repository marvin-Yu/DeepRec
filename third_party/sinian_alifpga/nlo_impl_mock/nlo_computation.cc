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

#include "third_party/sinian_alifpga/nlo_interface/nlo_computation.h"
#include "third_party/sinian_alifpga/nlo_interface/dfs_nlo_visitor_with_default.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_module.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_opcode.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_shape_util.h"

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <list>
#include <queue>
#include <set>
#include <sstream>


namespace sinian {

std::unique_ptr<NloComputation> NloComputation::Builder::Build(
    NloInstruction*  /* root_instruction */) {
  return nullptr;
}
NloComputation* NloComputation::Builder::BuildNloComputation(
    NloInstruction*  /* root_instruction */) {
  return nullptr;
}

NloComputation::NloComputation(
    const string& name, int /*  parameter_count */,
    std::vector<std::unique_ptr<NloInstruction>>* /*  instructions */,
    NloInstruction* root_instruction, NloInstruction* fusion_instruction)
    : name_(name),
      root_instruction_(root_instruction),
      fusion_instruction_(fusion_instruction) {
}

NloInstruction* NloComputation::AddInstruction(
    std::unique_ptr<NloInstruction> /* instruction */) {
  return nullptr;
}

NloInstruction* NloComputation::AddParameter(
    std::unique_ptr<NloInstruction> /* instruction */) {
  return nullptr;
}

NloStatus NloComputation::RemoveParameter(int64_t /* param_no */) {
  return NloStatus::OK();
}

void NloComputation::set_root_instruction(
    NloInstruction* /* new_root_instruction */) {
}

std::list<NloInstruction*> NloComputation::MakeInstructionPostOrder() const {
  std::list<NloInstruction*> post_order;
  return post_order;
}

std::list<NloComputation*> NloComputation::MakeEmbeddedComputationsList()
    const {
  std::list<NloComputation*> post_order;
  return post_order;
}

NloStatus NloComputation::Accept(DfsNloVisitor* /* visitor */) const {
  return NloStatus::OK();
}

NloStatus NloComputation::Accept(
    const NloFunctionVisitor::VisitorFunction& visitor_func) const {
  NloFunctionVisitor visitor(visitor_func);
  return this->Accept(&visitor);
}

void NloComputation::UniquifyName(NloNameUniquer*  name_uniquer) {
  name_ = name_uniquer->GetUniqueName(name_);
}

}  // namespace sinian
