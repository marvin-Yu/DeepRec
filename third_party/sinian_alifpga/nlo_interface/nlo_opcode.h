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

#ifndef NLO_INTERFACE_NLO_OPCODE_H_
#define NLO_INTERFACE_NLO_OPCODE_H_

#include <iosfwd>
#include <string>

namespace sinian {
using std::string;
// High-level optimizer instruction opcodes -- these are linear-algebra level
// opcodes. They are a flattened form of the UnaryOp, BinaryOp, ... opcodes
// present in the XLA service protobuf.
//
// See the XLA documentation for the semantics of each opcode.
enum class NloOpcode {
  kAbs,
  kAdd,
  kBatchNormGrad,
  kBatchNormInference,
  kBatchNormTraining,
  kBitcast,
  kBroadcast,
  kCall,
  kCeil,
  kClamp,
  kConcatenate,
  kConstant,
  kConvert,
  kConvolution,
  kCopy,
  kCos,
  kCrossReplicaSum,
  kCustomCall,
  kDivide,
  kDot,
  kDynamicSlice,
  kDynamicUpdateSlice,
  kEq,
  kExp,
  kFloor,
  kFusion,
  kGe,
  kGetTupleElement,
  kGt,
  kIndex,  //deprecated in 1.12
  kInfeed,
  kIsFinite,
  kLe,
  kLog,
  kLogicalAnd,  //kAnd
  kLogicalNot,  //kNot
  kLogicalOr,   //kOr
  kLt,
  kMap,
  kMaximum,
  kMinimum,
  kMultiply,
  kNe,
  kNegate,
  kOutfeed,
  kPad,
  kParameter,
  kPower,
  kRecv,
  kReduce,
  kReducePrecision,
  kReduceWindow,
  kRemainder,
  kReshape,
  kReverse,
  kRng,
  kRoundNearestAfz,
  kSelect,
  kSelectAndScatter,
  kSend,
  kSign,
  kSin,
  kSlice,
  kSort,
  kSubtract,
  kTanh,
  kTrace,
  kTranspose,
  kTuple,
  kUpdate,   //deprecated in 1.12
  kWhile,
};

// Returns a string representation of the opcode.
string NloOpcodeString(NloOpcode opcode);

inline std::ostream& operator<<(std::ostream& os, NloOpcode opcode) {
  return os << NloOpcodeString(opcode);
}

// Returns true iff the given opcode is a comparison operation.
bool NloOpcodeIsComparison(NloOpcode opcode);

// Returns true iff the given opcode has variadic operands.
bool NloOpcodeIsVariadic(NloOpcode opcode);

// Returns the number of HloOpcode values.
inline const uint32_t NloOpcodeCount() {
  return static_cast<uint32_t>(NloOpcode::kWhile) + 1;
}

}  // namespace sinian

#endif  // NLO_INTERFACE_NLO_OPCODE_H_
