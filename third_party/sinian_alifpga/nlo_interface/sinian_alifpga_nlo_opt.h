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

#ifndef NLO_INTERFACE_SINIAN_ALIFPGA_NLO_OPT_H_
#define NLO_INTERFACE_SINIAN_ALIFPGA_NLO_OPT_H_

#include "nlo_computation.h"
#include "nlo_instruction.h"
#include "nlo_module.h"

#include "../fpga_stream.h"

namespace sinian {
class SinianAliFPGANloOpt {
 public:
  explicit SinianAliFPGANloOpt()  {}
  ~SinianAliFPGANloOpt() {}
  static bool RunNloPasses(NloModule* module,
    std::vector<std::unique_ptr<::sinian_alifpga::SinianAliFPGAStream>>* fpga_stream_vec);
  static bool BuildNloGruCell(NloModule* module, const int batch_size,
                              const int input_size, const int hidden_size);
};
  void DumpInputShape(NloModule* nlomodule);
}  // namespace sinian

#endif  // NLO_INTERFACE_SINIAN_ALIFPGA_NLO_OPT_H_
