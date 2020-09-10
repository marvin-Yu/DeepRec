#include <string>
#include <fstream>

#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/protobuf.h"

using namespace std;
namespace tensorflow {
namespace {

class DenseOp : public OpKernel {
 public:
  explicit DenseOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("graph", &graph_));
    OP_REQUIRES_OK(context, context->GetAttr("feed_names", &feeds_));
    OP_REQUIRES_OK(context, context->GetAttr("fetch_names", &fetches_));
    auto value = ::tensorflow::protobuf::TextFormat::ParseFromString(graph_, &graph_def_);
    OP_REQUIRES(context, value, errors::InvalidArgument("parse graph failed"));

    // session options
    auto options = SessionOptions();
    tensorflow::ConfigProto* config = &options.config;
    options.config.mutable_gpu_options()->set_allow_growth(true);
    OP_REQUIRES_OK(context, NewSession(options, &session_));
    OP_REQUIRES_OK(context, session_->Create(graph_def_));
//    run_options_.set_trace_level(RunOptions::SOFTWARE_TRACE);
  }

  void Compute(OpKernelContext* context) override {
  auto begin = std::chrono::high_resolution_clock::now();
    OP_REQUIRES(context, context->num_inputs() == feeds_.size(),
        errors::InvalidArgument("input_num: ", context->num_inputs(),
          ", feed_count: ", feeds_.size()));

    OP_REQUIRES(context, context->num_outputs() == fetches_.size(),
        errors::InvalidArgument("output_num: ", context->num_outputs(),
          ", fetch_count: ", fetches_.size()));

    vector<pair<string, Tensor>> inputs;
    for (int i = 0; i < feeds_.size(); ++i) {
      inputs.emplace_back(make_pair(feeds_[i], context->input(i)));
    }

    vector<Tensor> outputs;
    RunMetadata metadata;
    OP_REQUIRES_OK(context, session_->Run(run_options_,
          inputs, fetches_, {}, &outputs, &metadata));

 /*   auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::ofstream dump;
    dump.open("desen_op_runmeta." + std::to_string(dur));
    if (metadata.SerializeToOstream(&dump)) {
      std::cerr << "dump error\n";
    }
    dump.close(); */
    for (int i = 0; i < fetches_.size(); ++i) {
      context->set_output(i, outputs[i]);
    }
  }

 private:
  string graph_;
  std::vector<string> feeds_;
  std::vector<string> fetches_;

  GraphDef graph_def_;
  // 后面用无锁队列实现vgpu
  Session* session_;
  RunOptions run_options_;
};

REGISTER_KERNEL_BUILDER(Name("DenseOp").Device(DEVICE_CPU),
                        DenseOp);

}  // namespace
}  // namespace tensorflow
