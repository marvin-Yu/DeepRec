//
// Created by qiaoxj on 2019-11-09.
//

#include "tensorflow/compiler/xla/service/gpu/slice_fusion.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

namespace m = match;

static bool SliceDim(const HloInstruction* slice, const int64 dim,
                     const int64 i) {
  auto& shape = slice->shape();
  auto slice_starts = slice->slice_starts();
  auto slice_limits = slice->slice_limits();
  for (int64 j = 0; j < shape.dimensions_size(); j++) {
    if (j == dim) {
      if (slice_starts[j] != i * shape.dimensions(dim) ||
          slice_limits[j] != (i + 1) * shape.dimensions(dim)) {
        return false;
      }
    } else {
      if (slice_starts[j] != 0 || slice_limits[j] != shape.dimensions(j)) {
        return false;
      }
    }
  }
  return true;
}

static bool EqualShape(const HloInstruction* inst1,
                       const HloInstruction* inst2) {
  return ShapeUtil::Equal(inst1->shape(), inst2->shape());
}

static bool AllUsersSameConcat(const HloInstruction* multiply,
                               const HloInstruction* slice,
                               const HloInstruction* reshape,
                               const HloInstruction* concat) {
  for (int64 i = 0; i < concat->operand_count(); i++) {
    auto operand1 = concat->operand(i);
    if (operand1->opcode() != HloOpcode::kReshape ||
        !EqualShape(operand1, reshape)) {
      return false;
    }
    auto operand2 = operand1->operand(0);
    if (operand2->opcode() != HloOpcode::kSlice ||
        !EqualShape(operand2, slice) || !SliceDim(operand2, 0, i)) {
      return false;
    }
    if (operand2->shape().dimensions_size() !=
        operand1->shape().dimensions_size() + 1) {
      return false;
    }
    if (operand2->shape().dimensions(0) != 1) {
      return false;
    }
    for (int64 j = 0; j < operand1->shape().dimensions_size(); j++) {
      if (operand1->shape().dimensions(j) !=
          operand2->shape().dimensions(j + 1)) {
        return false;
      }
    }
    if (operand2->operand(0) != multiply) {
      return false;
    }
    if (multiply->shape().dimensions(0) != multiply->user_count()) {
      return false;
    }
  }
  return true;
}

class SliceFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit SliceFusionVisitor(HloComputation* computation)
      : computation_(computation) {}

  Status HandleReduce(HloInstruction* reduce) override {
    auto concat = reduce->mutable_operand(0);
    if (concat->opcode() != HloOpcode::kConcatenate) {
      return Status::OK();
    }
    auto reshape = concat->mutable_operand(0);
    if (reshape->opcode() != HloOpcode::kReshape) {
      return Status::OK();
    }
    HloInstruction *slice, *multiply;
    if (Match(reshape, m::Reshape(m::Slice(&slice, m::Op(&multiply))))) {
      VLOG(10) << "Matched";
      // check only concat the last dimension
      auto concat_dimensions = concat->dimensions();
      if (concat_dimensions.size() != 1 ||
          concat_dimensions.front() != concat->shape().dimensions_size() - 1) {
        VLOG(10) << "Not concat the last Dimension";
        return Status::OK();
      }

      if (!AllUsersSameConcat(multiply, slice, reshape, concat)) {
        VLOG(10) << "Multiply's all users do not access to the same concat.";
        return Status::OK();
      }

      VLOG(10) << "Matched mul->slice->reshape->concat: "
               << multiply->ToString();

      // Currently only support multiply with 4-dim shape and concat with 3-dim
      // shape.
      if (multiply->shape().dimensions_size() != 4 ||
          concat->shape().dimensions_size() != 3) {
        return Status::OK();
      }

      auto reduce_dim = reduce->dimensions(0);
      if (reduce_dim == 1) {
        auto new_reduce_shape_dim = {multiply->shape().dimensions(0),
                                     multiply->shape().dimensions(1),
                                     multiply->shape().dimensions(3)};
        auto new_reduce_shape = ShapeUtil::MakeShape(
            reduce->shape().element_type(), new_reduce_shape_dim);
        auto new_reduce =
            computation_->AddInstruction(HloInstruction::CreateReduce(
                new_reduce_shape, multiply, reduce->mutable_operand(1), {2},
                reduce->called_computations()[0]));
        VLOG(10) << "new_reduce: " << new_reduce->ToString();

        // Create transpose
        TF_ASSIGN_OR_RETURN(auto transpose,
                            MakeTransposeHlo(new_reduce, {1, 0, 2}));
        VLOG(10) << "transpose: " << transpose->ToString();

        // Create reshape
        TF_ASSIGN_OR_RETURN(auto new_reshape,
                            MakeReshapeHlo(reduce->shape(), transpose));
        VLOG(10) << "new_reshape: " << new_reshape->ToString();

        changed_ = true;
        return computation_->ReplaceInstruction(reduce, new_reshape);
      } else if (reduce_dim == 0) {
        LOG(INFO) << "Need to implement the reduce_dim part";
      } else {
        LOG(ERROR) << "Invalid reduce dimension";
      }
      return Status::OK();
    }
    return Status::OK();
  }

 private:
  HloComputation* computation_;
};

static StatusOr<bool> RunOnComputation(HloComputation* computation) {
  SliceFusionVisitor visitor(computation);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

StatusOr<bool> SliceFusion::Run(xla::HloModule* module) {
  bool changed = false;
  int64 iter = 0;
  bool local_changed;
  do {
    local_changed = false;
    iter++;
    for (HloComputation* computation : module->MakeNonfusionComputations()) {
      TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
      changed |= result;
      local_changed |= result;
    }
  } while (local_changed && iter <= 10);
  return changed;
}
}  // namespace gpu
}  // namespace xla
