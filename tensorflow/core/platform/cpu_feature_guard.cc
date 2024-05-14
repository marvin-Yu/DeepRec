/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cpu_feature_guard.h"

#include <mutex>
#include <string>

#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace port {
namespace {

// If the CPU feature isn't present, log a fatal error.
void CheckFeatureOrDie(CPUFeature feature, const string& feature_name) {
  if (!TestCPUFeature(feature)) {
#ifdef __ANDROID__
    // Some Android emulators seem to indicate they don't support SSE, so to
    // avoid crashes when testing, switch this to a warning.
    LOG(WARNING)
#else
    LOG(FATAL)
#endif
        << "The TensorFlow library was compiled to use " << feature_name
        << " instructions, but these aren't available on your machine.";
  }
}

// Check if CPU feature is included in the TensorFlow binary.
void CheckIfFeatureUnused(CPUFeature feature, const string& feature_name,
                          string& missing_instructions) {
  if (TestCPUFeature(feature)) {
    missing_instructions.append(" ");
    missing_instructions.append(feature_name);
  }
}

// Raises an error if the binary has been compiled for a CPU feature (like AVX)
// that isn't available on the current machine. It also warns of performance
// loss if there's a feature available that's not being used.
// Depending on the compiler and initialization order, a SIGILL exception may
// occur before this code is reached, but this at least offers a chance to give
// a more meaningful error message.
class CPUFeatureGuard {
 public:
  CPUFeatureGuard() {
#ifdef __SSE__
    CheckFeatureOrDie(CPUFeature::SSE, "SSE");
#endif  // __SSE__
#ifdef __SSE2__
    CheckFeatureOrDie(CPUFeature::SSE2, "SSE2");
#endif  // __SSE2__
#ifdef __SSE3__
    CheckFeatureOrDie(CPUFeature::SSE3, "SSE3");
#endif  // __SSE3__
#ifdef __SSE4_1__
    CheckFeatureOrDie(CPUFeature::SSE4_1, "SSE4.1");
#endif  // __SSE4_1__
#ifdef __SSE4_2__
    CheckFeatureOrDie(CPUFeature::SSE4_2, "SSE4.2");
#endif  // __SSE4_2__
#ifdef __AVX__
    CheckFeatureOrDie(CPUFeature::AVX, "AVX");
#endif  // __AVX__
#ifdef __AVX2__
    CheckFeatureOrDie(CPUFeature::AVX2, "AVX2");
#endif  // __AVX2__
#ifdef __AVX512F__
    CheckFeatureOrDie(CPUFeature::AVX512F, "AVX512F");
#endif  // __AVX512F__
#ifdef __AVX512VNNI__
    CheckFeatureOrDie(CPUFeature::AVX512_VNNI, "AVX512_VNNI");
#endif  // __AVX512VNNI__
#ifdef __AVX512BF16__
    CheckFeatureOrDie(CPUFeature::AVX512_BF16, "AVX512_BF16");
#endif  // __AVX512BF16__
#ifdef __AVX512FP16__
    CheckFeatureOrDie(CPUFeature::AVX512_FP16, "AVX512_FP16");
#endif  // __AVX512FP16__
#ifdef __AVXVNNI__
    CheckFeatureOrDie(CPUFeature::AVX_VNNI, "AVX_VNNI");
#endif  // __AVXVNNI__
#ifdef __AVXVNNIINT8__
    CheckFeatureOrDie(CPUFeature::AVX_VNNI_INT8, "AVX_VNNI_INT8");
#endif  // __AVXVNNIINT8__
#ifdef __AVXNECONVERT__
    CheckFeatureOrDie(CPUFeature::AVX_NE_CONVERT, "AVX_NE_CONVERT");
#endif  // __AVXNECONVERT__
#ifdef __AMXTILE__
    CheckFeatureOrDie(CPUFeature::AMX_TILE, "AMX_TILE");
#endif  // __AMXTILE__
#ifdef __AMXINT8__
    CheckFeatureOrDie(CPUFeature::AMX_INT8, "AMX_INT8");
#endif  // __AMXINT8__
#ifdef __AMXBF16__
    CheckFeatureOrDie(CPUFeature::AMX_BF16, "AMX_BF16");
#endif  // __AMXBF16__
#ifdef __AMXFP16__
    CheckFeatureOrDie(CPUFeature::AMX_FP16, "AMX_FP16");
#endif  // __AMXFP16__
#ifdef __FMA__
    CheckFeatureOrDie(CPUFeature::FMA, "FMA");
#endif  // __FMA__
  }
};

CPUFeatureGuard g_cpu_feature_guard_singleton;

std::once_flag g_cpu_feature_guard_warn_once_flag;

}  // namespace

void InfoAboutUnusedCPUFeatures() {
  std::call_once(g_cpu_feature_guard_warn_once_flag, [] {
    string missing_instructions;
#if defined(_MSC_VER) && !defined(__clang__)

#ifndef __AVX__
    CheckIfFeatureUnused(CPUFeature::AVX, "AVX", missing_instructions);
#endif  // __AVX__
#ifndef __AVX2__
    CheckIfFeatureUnused(CPUFeature::AVX2, "AVX2", missing_instructions);
#endif  // __AVX2__

#else  // if defined(_MSC_VER) && !defined(__clang__)

#ifndef __SSE__
    CheckIfFeatureUnused(CPUFeature::SSE, "SSE", missing_instructions);
#endif  // __SSE__
#ifndef __SSE2__
    CheckIfFeatureUnused(CPUFeature::SSE2, "SSE2", missing_instructions);
#endif  // __SSE2__
#ifndef __SSE3__
    CheckIfFeatureUnused(CPUFeature::SSE3, "SSE3", missing_instructions);
#endif  // __SSE3__
#ifndef __SSE4_1__
    CheckIfFeatureUnused(CPUFeature::SSE4_1, "SSE4.1", missing_instructions);
#endif  // __SSE4_1__
#ifndef __SSE4_2__
    CheckIfFeatureUnused(CPUFeature::SSE4_2, "SSE4.2", missing_instructions);
#endif  // __SSE4_2__
#ifndef __AVX__
    CheckIfFeatureUnused(CPUFeature::AVX, "AVX", missing_instructions);
#endif  // __AVX__
#ifndef __AVX2__
    CheckIfFeatureUnused(CPUFeature::AVX2, "AVX2", missing_instructions);
#endif  // __AVX2__
#ifndef __AVX512F__
    CheckIfFeatureUnused(CPUFeature::AVX512F, "AVX512F", missing_instructions);
#endif  // __AVX512F__
#ifndef __AVX512VNNI__
    CheckIfFeatureUnused(CPUFeature::AVX512_VNNI, "AVX512_VNNI",
                         missing_instructions);
#endif  // __AVX512VNNI__
#ifndef __AVX512BF16__
    CheckIfFeatureUnused(CPUFeature::AVX512_BF16, "AVX512_BF16",
                         missing_instructions);
#endif  // __AVX512BF16__
#ifndef __AVX512FP16__
    CheckIfFeatureUnused(CPUFeature::AVX512_FP16, "AVX512_FP16",
                         missing_instructions);
#endif  // __AVX512FP16__
#ifndef __AVXVNNI__
    CheckIfFeatureUnused(CPUFeature::AVX_VNNI, "AVX_VNNI",
                         missing_instructions);
#endif  // __AVXVNNI__
#ifndef __AVXVNNIINT8__
    CheckIfFeatureUnused(CPUFeature::AVX_VNNI_INT8, "AVX_VNNI_INT8",
                         missing_instructions);
#endif  // __AVXVNNIINT8__
#ifndef __AVXNECONVERT__
    CheckIfFeatureUnused(CPUFeature::AVX_NE_CONVERT, "AVX_NE_CONVERT",
                         missing_instructions);
#endif  // __AVXNECONVERT__
#ifndef __AMX_TILE__
    CheckIfFeatureUnused(CPUFeature::AMX_TILE, "AMX_TILE",
                         missing_instructions);
#endif  // __AMX_TILE__
#ifndef __AMX_INT8__
    CheckIfFeatureUnused(CPUFeature::AMX_INT8, "AMX_INT8",
                         missing_instructions);
#endif  // __AMX_INT8__
#ifndef __AMX_BF16__
    CheckIfFeatureUnused(CPUFeature::AMX_BF16, "AMX_BF16",
                         missing_instructions);
#endif  // __AMX_BF16__
#ifndef __AMX_FP16__
    CheckIfFeatureUnused(CPUFeature::AMX_FP16, "AMX_FP16",
                         missing_instructions);
#endif  // __AMX_FP16__
#ifndef __FMA__
    CheckIfFeatureUnused(CPUFeature::FMA, "FMA", missing_instructions);
#endif  // __FMA__
#endif  // else of if defined(_MSC_VER) && !defined(__clang__)
    if (!missing_instructions.empty()) {
#ifndef INTEL_MKL
      LOG(INFO) << "Your CPU supports instructions that this TensorFlow "
                << "binary was not compiled to use:" << missing_instructions;
#else
      LOG(INFO) << "This TensorFlow binary is optimized "
                << "to use available CPU instructions in performance-"
                << "critical operations: " << std::endl
                << "To enable the following instructions:"
                << missing_instructions << ", in other operations, rebuild "
                << "TensorFlow with the appropriate compiler flags.";
#endif
    }
  });
}

}  // namespace port
}  // namespace tensorflow
