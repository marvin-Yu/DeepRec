/* Copyright 2018 The Sinian FPGA Compiler Authors. All Rights Reserved.*/
#ifndef THIRD_PARTY_SINIAN_ALIFPGA_SINIAN_ALIFPGA_NLO_PASS_H_
#define THIRD_PARTY_SINIAN_ALIFPGA_SINIAN_ALIFPGA_NLO_PASS_H_

#include "fpga_stream.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace sinian_alifpga {
class SinianAliFPGANloPass : public HloModulePass {
 public:
  explicit SinianAliFPGANloPass(
    std::vector<std::unique_ptr<::sinian_alifpga::SinianAliFPGAStream>>* fpga_stream_vec) :
    fpga_stream_vec_(fpga_stream_vec) {}
  ~SinianAliFPGANloPass() override = default;

  tensorflow::StringPiece name() const override { return name_; }

  // Returns whether the computation was changed.
  StatusOr<bool> Run(HloModule* module) override;
 protected:
  string name_="neuralnetwork-layer optimize passes";
  std::vector<std::unique_ptr<::sinian_alifpga::SinianAliFPGAStream>>* fpga_stream_vec_;
  TF_DISALLOW_COPY_AND_ASSIGN(SinianAliFPGANloPass);
};
}  // namespace sinian_alifpga
}  // namespace xla

#endif  // THIRD_PARTY_SINIAN_ALIFPGA_SINIAN_ALIFPGA_NLO_PASS_H_
