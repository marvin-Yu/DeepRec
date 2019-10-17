//
// Created by qiaoxj on 2019-10-14.
//
#include "tensorflow/cc/tools/graph_editor.h"
#include <vector>

int main(int argc, char** argv) {
  using namespace tensorflow;
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <input_frozen_graph_path> <output_model_path>" << std::endl;
    return -1;
  }
  std::string frozen_model_path = argv[1];
  std::string opt_model_path = argv[2];
  tensorflow::GraphDef graphDef;
  auto status = tensorflow::ReadGraphDef(frozen_model_path, &graphDef);
  if (status != Status::OK()) {
    LOG(ERROR) << status.error_message();
    return -1;
  }

  status = tensorflow::OptimizeDien(&graphDef);
  if (status != Status::OK()) {
    LOG(ERROR) << status.error_message();
  }

  status = tensorflow::SaveGraphDef(opt_model_path, graphDef, true, true);
  if (status != Status::OK()) {
    LOG(ERROR) << status.error_message();
    return -1;
  }
  return 0;
}
