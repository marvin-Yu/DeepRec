//
// Created by qiaoxj on 2019-11-09.
//

#ifndef TENSORFLOW_SLICE_FUSION_H
#define TENSORFLOW_SLICE_FUSION_H

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {
class SliceFusion : public HloModulePass {
 public:
  absl::string_view name() const override { return "slice_fusion"; }
  StatusOr<bool> Run(HloModule* module) override;
};
}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_SLICE_FUSION_H
