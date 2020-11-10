/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/blaze_predictor.h"

namespace tensorflow {
class BlazeXlaOp : public OpKernel {
 public:
  explicit BlazeXlaOp(OpKernelConstruction* context);
  ~BlazeXlaOp() {
    if (predictor_) {
      delete predictor_;
    }
  }

  void Compute(OpKernelContext* context) override;

 private:
  BlazePredictor* predictor_;
};

BlazeXlaOp::BlazeXlaOp(OpKernelConstruction* context)
    : OpKernel(context) {
  predictor_ = new BlazePredictor(context);
}


void BlazeXlaOp::Compute(OpKernelContext* ctx) {
  predictor_->Compute(ctx);
}

REGISTER_KERNEL_BUILDER(Name("BlazeXlaOp").Device(DEVICE_CPU), BlazeXlaOp);
REGISTER_KERNEL_BUILDER(Name("BlazeXlaOp").Device(DEVICE_GPU), BlazeXlaOp);
}  // namespace tensorflow
