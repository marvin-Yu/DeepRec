/* Copyright 2018 The Sinian FPGA Compiler Authors. All Rights Reserved.*/

#include "third_party/sinian_alifpga/sinian_alifpga_nlo_pass.h"

#include "third_party/sinian_alifpga/nlo_interface/nlo_computation.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_instruction.h"
#include "third_party/sinian_alifpga/nlo_interface/nlo_module.h"
#include "third_party/sinian_alifpga/nlo_interface/sinian_alifpga_nlo_opt.h"

namespace xla {
namespace sinian_alifpga {
using namespace sinian;
#if !SINIAN_ALIFPGA_MOCK
static NloOpcode HloOpcode2NloOpcode(HloOpcode code) {
  switch (code) {
#define CONVERT(CODE)                       \
  case HloOpcode::CODE:                    \
    return NloOpcode::CODE;

    CONVERT(kAbs)
    CONVERT(kAdd)
    CONVERT(kBatchNormGrad)
    CONVERT(kBatchNormInference)
    CONVERT(kBatchNormTraining)
    CONVERT(kBitcast)
    CONVERT(kBroadcast)
    CONVERT(kCall)
    CONVERT(kCeil)
    CONVERT(kClamp)
    CONVERT(kConcatenate)
    CONVERT(kConstant)
    CONVERT(kConvert)
    CONVERT(kConvolution)
    CONVERT(kCopy)
    CONVERT(kCos)
    CONVERT(kCrossReplicaSum)
    CONVERT(kCustomCall)
    CONVERT(kDivide)
    CONVERT(kDot)
    CONVERT(kDynamicSlice)
    CONVERT(kDynamicUpdateSlice)
    CONVERT(kEq)
    CONVERT(kExp)
    CONVERT(kFloor)
    CONVERT(kFusion)
    CONVERT(kGe)
    CONVERT(kGetTupleElement)
    CONVERT(kGt)
    CONVERT(kInfeed)
    CONVERT(kIsFinite)
    CONVERT(kLe)
    CONVERT(kLog)
    CONVERT(kLt)
    CONVERT(kMap)
    CONVERT(kMaximum)
    CONVERT(kMinimum)
    CONVERT(kMultiply)
    CONVERT(kNe)
    CONVERT(kNegate)
    CONVERT(kOutfeed)
    CONVERT(kPad)
    CONVERT(kParameter)
    CONVERT(kPower)
    CONVERT(kRecv)
    CONVERT(kReduce)
    CONVERT(kReducePrecision)
    CONVERT(kReduceWindow)
    CONVERT(kRemainder)
    CONVERT(kReshape)
    CONVERT(kReverse)
    CONVERT(kRng)
    CONVERT(kRoundNearestAfz)
    CONVERT(kSelect)
    CONVERT(kSelectAndScatter)
    CONVERT(kSend)
    CONVERT(kSign)
    CONVERT(kSin)
    CONVERT(kSlice)
    CONVERT(kSort)
    CONVERT(kSubtract)
    CONVERT(kTanh)
    CONVERT(kTrace)
    CONVERT(kTranspose)
    CONVERT(kTuple)
    CONVERT(kWhile)
#undef CONVERT
    case HloOpcode::kAnd:
      return NloOpcode::kLogicalAnd;
    case HloOpcode::kNot:
      return NloOpcode::kLogicalNot;
    case HloOpcode::kOr:
      return NloOpcode::kLogicalOr;
    default:
      // Unspported;
      LOG(FATAL) << "Unspported Opcode!";
      return NloOpcode::kAdd;
  }
}

static NloPrimitiveType HloType2NloType(PrimitiveType type) {
  switch(type) {
    case PRIMITIVE_TYPE_INVALID:
      return NLO_PRIMITIVE_TYPE_INVALID;
    case PRED:
      return NLO_PRED;
    case S8:
      return NLO_S8;
    case S16:
      return NLO_S16;
    case S32:
      return NLO_S32;
    case S64:
      return NLO_S64;
    case U8:
      return NLO_U8;
    case U32:
      return NLO_U32;
    case U64:
      return NLO_U64;
    case F32:
      return NLO_F32;
    case F64:
      return NLO_F64;
    case TUPLE:
      return NLO_TUPLE;
    case OPAQUE:
      return NLO_OPAQUE;
    default:
      // Unspported;
      LOG(FATAL) << "Unspported PrimitiveType!";
      return NLO_PRIMITIVE_TYPE_INVALID;
  }
}

static PrimitiveType NloType2HloType(NloPrimitiveType type) {
  switch(type) {
    case NLO_PRIMITIVE_TYPE_INVALID:
      return PRIMITIVE_TYPE_INVALID;
    case NLO_PRED:
      return PRED;
    case NLO_S8:
      return S8;
    case NLO_S16:
      return S16;
    case NLO_S32:
      return S32;
    case NLO_S64:
      return S64;
    case NLO_U8:
      return U8;
    case NLO_U32:
      return U32;
    case NLO_U64:
      return U64;
    case NLO_F32:
      return F32;
    case NLO_F64:
      return F64;
    case NLO_TUPLE:
      return TUPLE;
    case NLO_OPAQUE:
      return OPAQUE;
    default:
      // Unspported;
      LOG(FATAL) << "Unspported PrimitiveType!";
      return PRIMITIVE_TYPE_INVALID;
  }
}

static NloShape HloShape2NloShape(const Shape& shape) {
  auto type = shape.element_type();
  auto dims = shape.dimensions();
  auto new_type = HloType2NloType(type);
  std::vector<int64_t> new_dims;
  for (auto d : dims) {
    new_dims.emplace_back(d);
  }
  return NloShapeUtil::MakeShape(new_type, new_dims);
}

static Shape NloShape2HloShape(const NloShape& shape) {
  auto type = shape.element_type();
  auto dims = shape.dimensions();
  auto new_type = NloType2HloType(type);
  std::vector<int64> new_dims;
  for (uint32_t i = 0; i < dims.size(); i++) {
    new_dims.emplace_back(dims[i] * shape.way_number(i));
  }
  return ShapeUtil::MakeShape(new_type, new_dims);
}

static std::unique_ptr<NloLiteral> HloLiteral2NloLiteral(const Literal& l) {
  auto nlo_shape = HloShape2NloShape(l.shape());
  auto nlo_l = NloLiteral::CreateFromShape(nlo_shape);

  if (ShapeUtil::ElementsIn(l.shape()) <= 0) {
    VLOG(0) << "empty literal";
    return std::move(nlo_l);
  }
  switch (nlo_shape.element_type()) {
    case NLO_PRED: {
      std::vector<uint8_t> new_u8s;
      new_u8s.emplace_back(l.GetFirstElement<uint8_t>());
      nlo_l->set_u8s(new_u8s);
      break;
    }
    case NLO_S32:
    {
      std::vector<int32_t> new_s32s;
      new_s32s.emplace_back(l.GetFirstElement<int32_t>());
      nlo_l->set_s32s(new_s32s);
      break;
    }
    case NLO_U32:
    {
      std::vector<uint32_t> new_u32s;
      new_u32s.emplace_back(l.GetFirstElement<uint32_t>());
      nlo_l->set_u32s(new_u32s);
      break;
    }
    case NLO_F32: {
      std::vector<float> new_f32s;
      new_f32s.emplace_back(l.GetFirstElement<float>());
      nlo_l->set_f32s(new_f32s);
      break;
    }
    default: {
      LOG(FATAL) << "Unspported PrimitiveType!";
    }
  }
  return std::move(nlo_l);
}

static void Visit(
        HloInstruction* hlo,
        std::unordered_map<HloInstruction*, NloInstruction*>* instructions, 
        NloComputation::Builder* nlo_builder) {
  auto add_instruction = [&](std::unique_ptr<NloInstruction> instruction) {
    NloInstruction* nlo = nlo_builder->AddInstruction(std::move(instruction));
    return nlo;
  };
  auto lookup_instruction = [&](HloInstruction* hlo) {
    return instructions->at(hlo);
  };

  NloInstruction* nlo_instruction;
  std::vector<NloInstruction*> new_operands;
  if (hlo->operand_count() > 0) {
    for (auto operand : hlo->operands()) {
      auto new_operand = lookup_instruction(operand);
      new_operands.push_back(new_operand);
    }
  }

  switch (hlo->opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kIsFinite:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSort:
    case HloOpcode::kTanh: {
      auto nlo = NloInstruction::CreateUnary(
          HloShape2NloShape(hlo->shape()), HloOpcode2NloOpcode(hlo->opcode()), new_operands[0]);
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kReduce: {
      // (const NloShape& shape, NloInstruction* arg, NloInstruction* init_value,
      // std::vector<int64_t> dimensions_to_reduce,
      // NloComputation* reduce_computation)
      // hlo->dimensions_
      auto hlo_computation = hlo->called_computations()[0];
      auto root = hlo_computation->root_instruction();
      auto nlo_builder = (NloComputation::Builder)(hlo_computation->name());
      std::unordered_map<HloInstruction*, NloInstruction*> hlo2nlomap;
      //std::cout << "HloOpcode::kReduce=" << hlo_computation << std::endl;
      for (auto* instruction : hlo_computation->MakeInstructionPostOrder()) {
        //std::cout << "instruction=" << instruction->ToShortString() << std::endl;
        Visit(instruction, &hlo2nlomap, &nlo_builder);
        if (hlo2nlomap.find(instruction) == hlo2nlomap.end()) {
            VLOG(0) << "not support now." << HloOpcodeString(hlo->opcode());
            return;
        }
      }
      //std::cout << "root instruction=" << root->ToShortString() << std::endl;
      NloComputation* nlocpt = nlo_builder.BuildNloComputation(hlo2nlomap.at(root));
      auto dims = hlo->dimensions();
      std::vector<int64_t> new_dims;
      for (auto d : dims) {
        new_dims.emplace_back(d);
      }
      //std::cout << "new_operands.size()=" << new_operands.size() << std::endl;
      //std::cout << "new_operands[0]=" << new_operands[0]->ToString() << std::endl;
      //std::cout << "new_operands[1]=" << new_operands[1]->ToString() << std::endl;
      auto nlo = NloInstruction::CreateReduce(
          HloShape2NloShape(hlo->shape()), new_operands[0], new_operands[1], new_dims, nlocpt);
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    // Binary ops.
    case HloOpcode::kAdd:
    case HloOpcode::kDivide:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kNe:
    case HloOpcode::kDot:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kAnd:
    case HloOpcode::kOr: {
      auto nlo = NloInstruction::CreateBinary(
          HloShape2NloShape(hlo->shape()), HloOpcode2NloOpcode(hlo->opcode()), new_operands[0], new_operands[1]);
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect: {
      auto nlo = NloInstruction::CreateTernary(
                  HloShape2NloShape(hlo->shape()), HloOpcode2NloOpcode(hlo->opcode()),
                  new_operands[0], new_operands[1], new_operands[2]);
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    // Other supported ops.
    case HloOpcode::kBroadcast: {
      auto dims = hlo->dimensions();
      std::vector<int64_t> new_dmis;
      for (auto d : dims) {
        new_dmis.emplace_back(d);
      }
      auto nlo = NloInstruction::CreateBroadcast(
          HloShape2NloShape(hlo->shape()), new_operands[0], new_dmis);
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kCustomCall: {
      auto nlo = NloInstruction::CreateCustomCall(
          HloShape2NloShape(hlo->shape()), new_operands, hlo->custom_call_target());
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kConcatenate: {
      auto nlo = NloInstruction::CreateConcatenate(
          HloShape2NloShape(hlo->shape()), new_operands, hlo->dimensions(0));
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kReshape: {
      auto nlo = NloInstruction::CreateReshape(
          HloShape2NloShape(hlo->shape()), new_operands[0]);
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kTranspose: {
      auto nlo = NloInstruction::CreateTranspose(
          HloShape2NloShape(hlo->shape()), new_operands[0]);
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kSlice: {
      auto starts = hlo->slice_starts();
      auto limits = hlo->slice_limits();
      auto strides = hlo->slice_strides();
      std::vector<int64_t> new_starts;
      std::vector<int64_t> new_limits;
      std::vector<int64_t> new_strides;

      for (auto i : starts) {
        new_starts.emplace_back(i);
      }
      for (auto i : limits) {
        new_limits.emplace_back(i);
      }
      for (auto i : strides) {
        new_strides.emplace_back(i);
      }

      auto nlo = NloInstruction::CreateSlice(
          HloShape2NloShape(hlo->shape()), new_operands[0], new_starts,
          new_limits, new_strides);
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kTuple: {
      auto nlo = NloInstruction::CreateTuple(new_operands);
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kConstant: {
      auto nlo = NloInstruction::CreateConstant(HloLiteral2NloLiteral(hlo->literal()));
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kParameter: {
      auto nlo = NloInstruction::CreateParameter(
          hlo->parameter_number(), HloShape2NloShape(hlo->shape()), hlo->name());
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kBatchNormTraining: {
      auto nlo = NloInstruction::CreateBatchNormTraining(
          HloShape2NloShape(hlo->shape()), new_operands[0], new_operands[1],
          new_operands[2], hlo->epsilon(), hlo->feature_index());
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    case HloOpcode::kBatchNormInference: {
      auto nlo = NloInstruction::CreateBatchNormInference(
          HloShape2NloShape(hlo->shape()), new_operands[0], new_operands[1],
          new_operands[2], new_operands[3], new_operands[4],
          hlo->epsilon(), hlo->feature_index());
      nlo_instruction = add_instruction(std::move(nlo));
      break;
    }
    default:
      VLOG(0) << "not support now hlo::opcode = " << HloOpcodeString(hlo->opcode());
      return;
  }
  (*instructions)[hlo] = nlo_instruction;
}

static void Visit(
        NloInstruction* nlo,
        std::unordered_map<NloInstruction*, HloInstruction*>* instructions,
        HloComputation::Builder* hlo_builder) {
  auto add_instruction = [&](std::unique_ptr<HloInstruction> instruction) {
    HloInstruction* hlo_add = hlo_builder->AddInstruction(std::move(instruction));
    return hlo_add;
  };
  auto lookup_instruction = [&](NloInstruction* nlo) {
    return instructions->at(nlo);
  };

  HloInstruction* hlo_instruction;
  std::vector<HloInstruction*> new_operands;
  if (nlo->operand_count() > 0) {
    for (auto operand : nlo->operands()) {
      auto new_operand = lookup_instruction(operand);
      new_operands.push_back(new_operand);
    }
  }

  switch (nlo->opcode()) {
    case NloOpcode::kTuple: {
      auto hlo = HloInstruction::CreateTuple(new_operands);
      hlo_instruction = add_instruction(std::move(hlo));
      break;
    }
    case NloOpcode::kFusion:
      if (nlo->fusion_kind() == NloInstruction::FusionKind::kCustom ||
        nlo->fusion_kind() == NloInstruction::FusionKind::kCustomCluster) {
        if (nlo->IsMultiOutputFusion()) {
          auto old_tuple = nlo->fused_expression_root();
          /* FusionCluster's root node may be fusion node, should update old_tuple. */
          if (old_tuple->opcode() == NloOpcode::kFusion) {
            old_tuple = nlo->fused_expression_root()->fused_expression_root();
          }
          std::vector<HloInstruction*> new_tuple_operands;

          for (int64_t i = 0; i < old_tuple->operand_count() - 1; i++) {
            auto old_tuple_operand = old_tuple->operand(i);
            std::string t_name = "__xla_fpga_runtime_AliFPGA_Dummy_";
            t_name += std::to_string(i);
            auto deadbeef_call = HloInstruction::CreateCustomCall(
                NloShape2HloShape(old_tuple_operand->shape()),
                new_operands,
                t_name);
            new_tuple_operands.push_back(add_instruction(std::move(deadbeef_call)));
          }
          std::string t_name = "__xla_fpga_runtime_AliFPGA_End_";
          t_name += std::to_string(old_tuple->operand_count());
          std::vector<HloInstruction*> t_operands = new_tuple_operands;
          for (auto operand : new_operands) {
            t_operands.push_back(operand);
          }
          auto last_deadbeef_call = HloInstruction::CreateCustomCall(
                NloShape2HloShape(old_tuple->operand(old_tuple->operand_count() - 1)->shape()),
                t_operands,
                t_name);
          new_tuple_operands.push_back(add_instruction(
            std::move(last_deadbeef_call)));
          auto new_tuple = HloInstruction::CreateTuple(new_tuple_operands);
          hlo_instruction = add_instruction(std::move(new_tuple));
        } else {
          std::string t_name = "__xla_fpga_runtime_AliFPGA_End_1";
          auto deadbeef_call = HloInstruction::CreateCustomCall(
                NloShape2HloShape(nlo->shape()),
                new_operands,
                t_name);
          hlo_instruction = add_instruction(std::move(deadbeef_call));
        }
      } else {
        VLOG(0) << "non-custom fusion not support now.";
        return;
      }
      break;
    case NloOpcode::kParameter: {
      auto hlo = HloInstruction::CreateParameter(
          nlo->parameter_number(), NloShape2HloShape(nlo->shape()), nlo->parameter_name());
      hlo_instruction = add_instruction(std::move(hlo));
      break;
    }
    default:
      VLOG(3) << "nlo opcode:" << nlo->opcode() <<", not support now.";
      return;
  }
  (*instructions)[nlo] = hlo_instruction;
}

static bool ConvHloModu2NloModu(xla::HloModule* hlomodule, NloModule* nlomodule) {
  std::ostringstream s;
  auto hlo_computation = hlomodule->entry_computation();
  auto root = hlo_computation->root_instruction();
  auto nlo_builder = (NloComputation::Builder)(hlo_computation->name());
  std::unordered_map<HloInstruction*, NloInstruction*> hlo2nlomap;

  for (auto* instruction : hlo_computation->MakeInstructionPostOrder()) {
    Visit(instruction, &hlo2nlomap, &nlo_builder);
    if (hlo2nlomap.find(instruction) == hlo2nlomap.end()) {
      VLOG(0) << "don't support this instruction now.";
      return false;
    }
  }
  auto nlo_root = hlo2nlomap.at(root);
  // add a tuple for nlo module root
  if (nlo_root->opcode() != NloOpcode::kTuple) {
    std::vector<NloInstruction*> new_operands;
    new_operands.push_back(nlo_root);
    auto new_tuple = NloInstruction::CreateTuple(new_operands);
    nlo_root = nlo_builder.AddInstruction(std::move(new_tuple));
  }
  nlomodule->AddEntryComputation(nlo_builder.Build(nlo_root));
  return true;
}

static bool ConvNloModu2HloModu(NloModule* nlomodule, xla::HloModule* hlomodule) {
  bool hlo_has_tuple = hlomodule->entry_computation()->
                       root_instruction()->opcode() == HloOpcode::kTuple;
  std::ostringstream s;
  auto nlo_computation = nlomodule->entry_computation();
  auto root = nlo_computation->root_instruction();
  auto hlo_builder = (HloComputation::Builder)(nlo_computation->name());
  std::unordered_map<NloInstruction*, HloInstruction*> nlo2hlomap;

  for (auto* instruction : nlo_computation->MakeInstructionPostOrder()) {
    // do not add tuple if hlo does not
    if (!hlo_has_tuple && instruction == root &&
        instruction->opcode() == NloOpcode::kTuple) {
      root = root->mutable_operand(0);
      continue;
    }
    Visit(instruction, &nlo2hlomap, &hlo_builder);
    if (nlo2hlomap.find(instruction) == nlo2hlomap.end()) {
      VLOG(0) << "don't support this instruction now.";
      return false;
    }
  }
  hlomodule->RemoveAllComputation();
  hlomodule->AddEntryComputation(hlo_builder.Build(nlo2hlomap.at(root)));
  return true;
}

static void ConvertModuleToCpu(xla::HloModule* hlomodule) {
  auto hlo_computation = hlomodule->entry_computation();
  auto name = hlo_computation->name();
  auto root = hlo_computation->root_instruction();
  auto hlo_builder = (HloComputation::Builder)(name);

  std::vector<HloInstruction*> new_ops;
  for (auto op : root->operands()) {
    auto lit = Literal::CreateFromShape(op->shape());
    auto hlo = HloInstruction::CreateConstant(std::move(lit));
    new_ops.push_back(hlo_builder.AddInstruction(std::move(hlo)));
  }

  auto tuple = HloInstruction::CreateTuple(new_ops);
  HloInstruction* new_root = hlo_builder.AddInstruction(std::move(tuple));

  hlomodule->RemoveAllComputation();
  hlomodule->AddEntryComputation(hlo_builder.Build(new_root));
}

#else
static bool ConvHloModu2NloModu(xla::HloModule* hlomodule, NloModule* nlomodule) {
  return true;
}

static bool ConvNloModu2HloModu(NloModule* nlomodule, xla::HloModule* hlomodule) {
  return true;
}

static void ConvertModuleToCpu(xla::HloModule* hlomodule) {

}

#endif

static int64 cache_size = 0;
static std::mutex cache_size_mutex;

StatusOr<bool> SinianAliFPGANloPass::Run(HloModule* module) {
  {
    std::lock_guard<std::mutex> guard(cache_size_mutex);
    cache_size++;
  }
  VLOG(0) << "Compilation cache missing, "
          << "Increase compilation cache size to : " << cache_size;

  VLOG(2) << "Before conv nlo:";
  NloModule nlo_module(module->name());
  if (!ConvHloModu2NloModu(module, &nlo_module)) {
    VLOG(0) << "ConvHloModu2NloModu failed.";
    return true;
  }
  VLOG(2) << "After conv nlo:";
  if (!SinianAliFPGANloOpt::RunNloPasses(&nlo_module, fpga_stream_vec_)) {
    VLOG(0) << "Can not support, roll back to CPU." << std::endl;
    ConvertModuleToCpu(module);
    return true;
  }
  if (!ConvNloModu2HloModu(&nlo_module, module)) {
    VLOG(0) << "ConvNloModu2HloModu failed.";
  }

  VLOG(0) << "FPGA Compilation " << cache_size << " th " << "Pass! ";
  return true;
}
}  // namespace sinian_alifpga
}  // namespace xla
