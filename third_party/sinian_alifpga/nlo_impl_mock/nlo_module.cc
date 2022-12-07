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

#include "third_party/sinian_alifpga/nlo_interface/nlo_module.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_shape_util.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_map_util.h"
namespace sinian {
NloModule::NloModule(const string& name) : name_(name) {}

NloComputation* NloModule::AddEntryComputation(
    std::unique_ptr<NloComputation> /*  computation */) {
  return nullptr;
}

NloStatus NloModule::RemoveAllComputation() {
  return NloStatus::OK();
}

std::list<NloComputation*> NloModule::MakeComputationPostOrder() const {
  std::list<NloComputation*> result;
  return result;
}
}  // namespace sinian
