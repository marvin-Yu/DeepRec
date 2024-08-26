/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/port.h"

#include "absl/base/call_once.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

bool IsGoogleCudaEnabled() {
#if GOOGLE_CUDA
  return true;
#else
  return false;
#endif
}

bool IsBuiltWithROCm() {
#if TENSORFLOW_USE_ROCM
  return true;
#else
  return false;
#endif
}

bool GpuSupportsHalfMatMulAndConv() {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  return true;
#else
  return false;
#endif
}

bool IsMklEnabled() {
#if defined(INTEL_MKL) && defined(ENABLE_MKL)
  return true;
#else
  return false;
#endif  // INTEL_MKL && ENABLE_MKL
}

// TODO(intel-tf): Use IsDataTypeSupportedByOneDNNOnThisCPU() instead
bool IsBF16SupportedByOneDNNOnThisCPU() {
  bool result = false;
#ifdef INTEL_MKL
  using port::CPUFeature;
  using port::TestCPUFeature;
  result = (TestCPUFeature(CPUFeature::AVX512F) ||
            TestCPUFeature(CPUFeature::AVX_NE_CONVERT));
  if (result)
    VLOG(2) << "CPU supports BF16";
  else
    VLOG(2) << "CPU does not support BF16";
#endif  // INTEL_MKL
  return result;
}

bool UseCpuAdvancedOps() {
  static absl::once_flag once;
  static bool use_adv_cpu_ops = false;
  absl::call_once(once, [&] {
    auto status = ReadBoolFromEnvVar("TF_USE_ADVANCED_CPU_OPS", use_adv_cpu_ops,
                                     &use_adv_cpu_ops);
    if (!status.ok()) {
      LOG(WARNING)
          << "TF_USE_ADVANCED_CPU_OPS is not set to either '0', 'false',"
          << " '1', or 'true'. Using the default setting: " << use_adv_cpu_ops;
    }
    if (use_adv_cpu_ops) {
      LOG(INFO) << "TF advanced CPU ops are enabled.";
    }
  });
  return use_adv_cpu_ops;
}

}  // end namespace tensorflow
