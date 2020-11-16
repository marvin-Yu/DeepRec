#ifndef TENSORFLOW_CORE_KERNELS_BLAZE_XLA_PREDICTOR_H_
#define TENSORFLOW_CORE_KERNELS_BLAZE_XLA_PREDICTOR_H_

#include "tensorflow/core/kernels/blaze_predictor.h"

namespace tensorflow {
typedef std::map<std::string, std::vector<NodeDef>> InputNodeMap;
typedef std::map<std::string, NodeDef> NodeMap;

class BlazeXlaPredictor : public BlazePredictor {
 public:
  using BlazePredictor::BlazePredictor;
  ~BlazeXlaPredictor() {}

  void Compute(OpKernelContext* ctx) override;
 private:
  Status FindBlackPaddingInputs();
  InputNodeMap ToInputNodeMap();

  Status PrepareData() override;

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
  Status Warmup() override;
  Status CheckShape(const TensorShapeProto& shape);

  std::vector<int32> batch_sizes_;
  std::vector<bool> skip_padding_;
  NodeMap node_map_;
};
}
#endif
