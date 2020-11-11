#ifndef TENSORFLOW_CORE_KERNELS_BLAZE_XLA_PREDICTOR_H_
#define TENSORFLOW_CORE_KERNELS_BLAZE_XLA_PREDICTOR_H_

#include "tensorflow/core/kernels/blaze_predictor.h"

namespace tensorflow {
typedef std::map<std::string, std::vector<NodeDef>> InputNodeMap;

class BlazeXlaPredictor : public BlazePredictor {
 public:
  BlazeXlaPredictor(OpKernelConstruction* ctx) : BlazePredictor(ctx) {}
  ~BlazeXlaPredictor() {}

 private:
  Status FindBlackPaddingInputs();
  InputNodeMap ToInputNodeMap();

  Status PrepareData(OpKernelConstruction* ctx) override;

  Status PadToStatic(const std::vector<Tensor>& inputs,
                     std::vector<Tensor>* padded_inputs,
                     int batchsize, int pad_to_batchsize,
                     OpKernelContext* ctx);

  Status SliceToDynamic(const std::vector<Tensor>& padded_outputs,
                        std::vector<Tensor*>* outputs,
                        int batchsize, int pad_to_batchsize,
                        OpKernelContext* ctx);
  int InferBatchSize(const std::vector<Tensor>& tensors);
  
  Status InitXlaWarmup();
  std::vector<int32> batch_sizes_;
  std::vector<bool> skip_padding_;
};
}
#endif
