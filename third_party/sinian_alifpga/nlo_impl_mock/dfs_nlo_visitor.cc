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

#include "third_party/sinian_alifpga/nlo_interface/dfs_nlo_visitor.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_status.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_instruction.h"
namespace sinian {
NloStatus DfsNloVisitor::HandleElementwiseUnary(NloInstruction* /* hlo */) {
  return NloStatus::Unimplemented();
}

NloStatus DfsNloVisitor::HandleElementwiseBinary(NloInstruction* /* hlo */) {
  return NloStatus::Unimplemented();
}

NloStatus DfsNloVisitor::Preprocess(NloInstruction* /* hlo */) {
  return NloStatus::OK();
}

NloStatus DfsNloVisitor::Postprocess(NloInstruction* /* visited*/) {
  return NloStatus::OK();
}
}  // namespace sinian
