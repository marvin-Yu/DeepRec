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

#include "tensorflow/core/grappler/optimizers/fold_continuous_fc.h"
#include "tensorflow/core/grappler/optimizers/original_delivery_common.h"

#include <fstream>
#include <queue>
#include <map>
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace grappler {

namespace {

bool OptimizeContinuousFCPattern(GraphDef &input_graph_def, GraphDef* output_graph_def,
                                 bool& is_changed, int& count) {
  VLOG(1) << "start to optimize continuous fc pattern, " << continuous_fc_pattern.DebugString();
  Status status = ReplaceMatchingOpTypes(
      input_graph_def,
      continuous_fc_pattern,
      [&is_changed, &count](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        const NodeDef& batch_matmul = match.node;
        const NodeDef& bias_add2 = match.inputs[0].node;
        const NodeDef& preserved_node = match.inputs[1].node;
        const NodeDef& reshape2 = match.inputs[0].inputs[0].node;
        const NodeDef& matmul2 = match.inputs[0].inputs[0].inputs[0].node;
        const NodeDef& reshape_shape = match.inputs[0].inputs[0].inputs[1].node;
        const NodeDef& input = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& w1 = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& b1 = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& w2 = match.inputs[0].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& b2 = match.inputs[0].inputs[1].node;
        VLOG(1) << match.DebugString();
        
        Status status;
        DataType output_type;
        output_type = w1.attr().at("dtype").type();
        // fold weight
        NodeDef matmul_w;
        matmul_w.CopyFrom(matmul2);
        *(matmul_w.mutable_input(0)) = w1.name();
        *(matmul_w.mutable_input(1)) = w2.name();
        matmul_w.set_name(matmul2.name() + "_fold_weight");
        NodeDef new_matmul2;
        new_matmul2.CopyFrom(matmul2);
        *(new_matmul2.mutable_input(0)) = input.name();
        *(new_matmul2.mutable_input(1)) = matmul_w.name();
        // 构建新的bias
        NodeDef dim_const;
        Tensor t_dim(DT_INT64, TensorShape({1}));
        auto dim_data = t_dim.tensor<int64, 1>();
        dim_data(0) = 0;
        TF_RETURN_IF_ERROR(CreateConstNodeDef(dim_const,
                           b1.name() + "_expand_dim", t_dim, b1));
        NodeDef expand_dims;
        NodeDefBuilder::NodeOut expand_input(b1.name(), 0, output_type);
        NodeDefBuilder::NodeOut expand_dim(dim_const.name(), 0, DT_INT64);
        TF_RETURN_IF_ERROR(ConstructExpandDimsNodeDef(expand_dims, b1,
                                    b1.name() + "_expand", expand_input,
                                    expand_dim, output_type, DT_INT64));
        NodeDef matmul_b;
        matmul_b.CopyFrom(matmul2);
        *(matmul_b.mutable_input(0)) = expand_dims.name();
        *(matmul_b.mutable_input(1)) = w2.name();
        matmul_b.set_name(b2.name() + "_matmul_weight");

        std::vector<NodeDefBuilder::NodeOut> add_inputs;
        add_inputs.emplace_back(matmul_b.name(), 0, output_type);
        add_inputs.emplace_back(b2.name(), 0, output_type);
        NodeDef add_b;
        TF_RETURN_IF_ERROR(ConstuctAddNodeDef(add_b, matmul2, b2.name() + "_fold_bias",
                                              add_inputs, output_type));
        NodeDef squeeze;
        NodeDefBuilder::NodeOut squeeze_input(add_b.name(), 0, output_type);
        TF_RETURN_IF_ERROR(ConstructSqueezeNodeDef(squeeze, b1,
                                    add_b.name() + "_squeeze",
                                    squeeze_input, output_type));

        NodeDef new_bias_add2;
        new_bias_add2.CopyFrom(bias_add2);
        *(new_bias_add2.mutable_input(1)) = squeeze.name();
        // 5. 保留匹配的节点
        new_nodes->push_back(input);
        new_nodes->push_back(new_matmul2);
        new_nodes->push_back(w1);
        new_nodes->push_back(w2);
        new_nodes->push_back(b1);
        new_nodes->push_back(b2);
        new_nodes->push_back(matmul_w);
        new_nodes->push_back(reshape2);
        new_nodes->push_back(reshape_shape);
        new_nodes->push_back(expand_dims);
        new_nodes->push_back(dim_const);
        new_nodes->push_back(matmul_b);
        new_nodes->push_back(add_b);
        new_nodes->push_back(squeeze);
        new_nodes->push_back(new_bias_add2);
        new_nodes->push_back(batch_matmul);
        new_nodes->push_back(preserved_node);

        is_changed = true;
        count++;
        return Status::OK();
      },
      {}, output_graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "optimize continuous fc failed " << status;
    return false;
  }
  return true;
}

bool OptimizeContinuousFC(GraphDef& input_graph, GraphDef* optimized_graph) {
  
  int count = 0;
  while(1) {
    bool graph_changed = false;
    bool result = OptimizeContinuousFCPattern(input_graph, optimized_graph,
                                              graph_changed, count);
    if (!result) return false;
    if (!graph_changed) break;
    std::swap(input_graph, *optimized_graph);
  }
  VLOG(0) << "Fold " << count << " continuous fully connected layer";
  return true;
}

}  // end namespace

Status FoldContinuousFCOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool optimize = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE", true, &optimize);
  if (!optimize) {
    *optimized_graph = item.graph;
    return Status::OK();
  }

  VLOG(0) << "FoldContinuousFCOptimizer is on.";

  GraphDef input_graph_def = item.graph;
  if(!OptimizeContinuousFC(input_graph_def, optimized_graph)) {
    LOG(INFO) << "optimize continuous fully conntected layer failed";
    *optimized_graph = item.graph;
    return Status::OK();
  }
  *optimized_graph->mutable_versions() = item.graph.versions();
  return Status::OK();
}

void FoldContinuousFCOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
