//
// Created by qiaoxj on 2019-10-14.
//
#include "tensorflow/cc/tools/graph_editor.h"
#include <vector>

int main(int argc, char** argv) {
  using namespace tensorflow;
  std::string base_path =
      "/home/xianjie.qxj/dien_model/dien_1006_private/frozen_graph.pb";
  std::string edit_path =
      "/home/xianjie.qxj/dien_model/dien_1006_private/new_model.pb";
  tensorflow::GraphDef graphDef;
  tensorflow::ReadGraphDef(base_path, &graphDef);

  tensorflow::OptimizeDien(&graphDef);

  tensorflow::SaveGraphDef(edit_path, graphDef, true, true);
  return 0;
}
