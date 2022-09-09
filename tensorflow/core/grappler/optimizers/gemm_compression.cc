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

#include "tensorflow/core/grappler/optimizers/gemm_compression.h"

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

bool CreateConstNode(NodeDef& def, string const_name, Tensor &t_const, const NodeDef& base) {
  NodeDefBuilder const_builder(const_name, "Const");
  Status status = const_builder
                  .Attr("dtype", t_const.dtype())
                  .Attr("value", t_const)
                  .Finalize(&def);
  if (!status.ok()) {
    LOG(ERROR) << "Const node construction failed with" << status;
    return false;
  }
  def.set_device(base.device());
  return true;
}

bool ConstuctSliceOp(const NodeDef& input, Tensor& t_begin, Tensor& t_size,
                     string prefix, DataType output_type,
                     NodeDef& begin_const, NodeDef& size_const, NodeDef& slice) {
  // 构建begin const
  string begin_name = prefix + "/slice_begin";
  if (CreateConstNode(begin_const, begin_name, t_begin, input)) {
    return false;
  }
  // 构建size const
  string size_name = prefix + "/slice_size";
  if (CreateConstNode(size_const, size_name, t_size, input)) {
    return false;
  }
  // 构建Slice
  string slice_name = prefix + "/slice";
  std::vector<NodeDefBuilder::NodeOut> slice_inputs;
  slice_inputs.emplace_back(input.name(), 0, output_type);
  slice_inputs.emplace_back(begin_const.name(), 0, DT_INT64);
  slice_inputs.emplace_back(size_const.name(), 0, DT_INT64);
  Status status = NodeDefBuilder(slice_name, "Slice")
                                .Input(slice_inputs[0])
                                .Input(slice_inputs[1])
                                .Input(slice_inputs[2])
                                .Attr("T", output_type)
                                .Attr("Index", DT_INT64)
                                .Finalize(&slice);
  if (!status.ok()) {
    LOG(ERROR) << "Adding slice nodedef build failed " << status;
    return false;
  }
  slice.set_device(input.device());
  VLOG(1) << slice.DebugString();
  return true;
}

void GetAllMatchNodes(std::vector<NodeDef>& nodes, std::set<string>& node_set, const NodeMatch& match) {
  if (!node_set.count(match.node.name())) {
    nodes.push_back(match.node);
    node_set.insert(match.node.name());
  }
  for (const NodeMatch& input : match.inputs) {
    GetAllMatchNodes(nodes, node_set, input);
  }
  return;
}

bool OptimizeGatherConcatPattern(GraphDef &input_graph_def, GraphDef* output_graph_def,
                                 bool& is_changed) {
  VLOG(1) << "start to optimize gather pattern, " << gemm_compression_pattern.DebugString();
  Status status = ReplaceMatchingOpTypes(
      input_graph_def,
      gemm_compression_pattern,
      [&is_changed](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        // 1. 匹配到pattern
        // 2. 获取有用的节点
        const NodeDef& matmul_node = match.node;
        const NodeDef& concat_node = match.inputs[0].node;
        const NodeDef& weight_node = match.inputs[1].node;
        const NodeDef& gather_node = match.inputs[0].inputs[0].node;
        const NodeDef& gather_input_node = match.inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_ph_node = match.inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_ind_node = match.inputs[0].inputs[0].inputs[1].node;
        const NodeDef& gather_axis_node = match.inputs[0].inputs[0].inputs[2].node;
        VLOG(1) << match.DebugString();
        
        Status status;
        std::vector<NodeDef> match_nodes;
        std::set<string> node_set;
        GetAllMatchNodes(match_nodes, node_set, match);
        VLOG(1) << "match nodes number:" << match_nodes.size();

        // 3. 检查placeholder和gather axis const 值
        bool invalid = false;
        if (gather_ind_node.name().find("user_creative_indicator") == string::npos) {
          LOG(WARNING) << "gather input indicator placeholder not match:" << gather_ind_node.name();
          invalid = true;
        }
        Tensor gather_axis_tensor = GetNodeTensorAttr(gather_axis_node, "value");
        auto const_dtype = gather_axis_node.attr().at("dtype").type();
        if (const_dtype == DT_INT32) {
          auto gather_axis_value = gather_axis_tensor.flat<int32>();
          if (gather_axis_value(0) != 0) {
            LOG(WARNING) << "gather axis const value not valid:"
                    << gather_axis_node.DebugString();
            invalid = true;
          }
        } else if (const_dtype == DT_INT64) {
          auto gather_axis_value = gather_axis_tensor.flat<int64>();
          if (gather_axis_value(0) != 0) {
            LOG(WARNING) << "gather axis const value not valid:"
                    << gather_axis_node.DebugString();
            invalid = true;
          }
        } else {
          LOG(WARNING) << "gather axis const dtype is not int:" << const_dtype;
          invalid = true;
        }
        if (concat_node.op() != "ConcatV2") {
          LOG(WARNING) << "concat node is not ConcatV2:" << concat_node.DebugString();
          invalid = true;
        }
        // 4. 将concat进行分拆，后续MatMul的权重也要分拆
        DataType output_type = DT_FLOAT;
        string type_key = "T";
        if (weight_node.op() == "Const") type_key = "dtype";
        if (weight_node.attr().count(type_key) != 0) {
          output_type = weight_node.attr().at(type_key).type();
        } else {
          invalid = true;
        }
        if (invalid) {
          // 不替换时，直接返回，会导致之后每次都匹配到这个不满足条件的pattern，
          // 其余pattern无法继续匹配，因此给这部分子图增加一个Identity，改变图结构
          NodeDef new_identity_node;
          new_identity_node.set_name(concat_node.name() + "/identity");
          new_identity_node.set_op("Identity");
          new_identity_node.clear_attr();
          (*new_identity_node.mutable_attr())["T"].set_type(output_type);
          *(new_identity_node.mutable_input()->Add()) = gather_node.name();

          NodeDef new_concat_node;
          new_concat_node.CopyFrom(concat_node);
          *(new_concat_node.mutable_input(0)) = new_identity_node.name();
          VLOG(1) << "insert Identity before Gather" << new_identity_node.DebugString();
          // 5. 保留匹配的节点
          new_nodes->push_back(matmul_node);
          new_nodes->push_back(weight_node);
          new_nodes->push_back(new_concat_node);
          new_nodes->push_back(new_identity_node);
          new_nodes->push_back(gather_node);
          new_nodes->push_back(gather_ph_node);
          new_nodes->push_back(gather_input_node);
          new_nodes->push_back(gather_ind_node);
          new_nodes->push_back(gather_axis_node);

          is_changed = true;
          return Status::OK();
        }
        // 获取输入的shape
        int size = gather_ph_node.attr().at("shape").shape().dim(1).size();
        VLOG(1) << "get gaterh input dim 1 size:" << size;
        
        // 构建split，拆分权重
        Tensor t_begin(DT_INT64, TensorShape({2}));
        auto begin_data = t_begin.tensor<int64, 1>();
        begin_data(0) = 0;
        begin_data(1) = 0;
        Tensor t_size(DT_INT64, TensorShape({2}));
        auto size_data = t_size.tensor<int64, 1>();
        size_data(0) = size;
        size_data(1) = -1;
        NodeDef begin_const_part1;
        NodeDef size_const_part1;
        NodeDef slice_part1;
        // 构建Slice
        if (ConstuctSliceOp(weight_node, t_begin, t_size, weight_node.name() + "_part1",
              output_type, begin_const_part1, size_const_part1, slice_part1)) {
          return false;
        }
        begin_data(0) = size;
        size_data(0) = -1;
        NodeDef begin_const_part2;
        NodeDef size_const_part2;
        NodeDef slice_part2;
        // 构建Slice
        if (ConstuctSliceOp(weight_node, t_begin, t_size, weight_node.name() + "_part2",
                        output_type, begin_const_part2, size_const_part2, slice_part2)) {
          return false;
        }

        // 构建新的ConcatV2
        NodeDef new_concat;
        std::vector<NodeDefBuilder::NodeOut> concat_inputs;
        DataType tidx;
        string idx_name;
        int idx = 0;
        for (string input : concat_node.input()) {
          if (idx == 0) {
            idx++;
            continue;
          }
          if (idx == (concat_node.input().size() - 1)) {
            tidx = concat_node.attr().at("Tidx").type();
            idx_name = input;
          } else {
            DataType type = concat_node.attr().at("T").type();
            concat_inputs.emplace_back(input, 0, type);
          }
          idx++;
        }
        int concat_n = concat_node.attr().at("N").i();
        // ConcatV2 input和index要分开传入
        status = NodeDefBuilder(matmul_node.name() + "/concat", "ConcatV2")
                               .Input(concat_inputs)
                               .Input({idx_name, 0, tidx})
                               .Attr("N", concat_n - 1)
                               .Attr("T", concat_node.attr().at("T").type())
                               .Attr("Tidx", concat_node.attr().at("Tidx").type())
                               .Finalize(&new_concat);
        if (!status.ok()) {
          LOG(ERROR) << "Adding ConcatV2 nodedef build failed " << status;
          return status;
        }
        new_concat.set_device(matmul_node.device());
        // 构建新的MatMul
        NodeDef matmul_part1;
        matmul_part1.CopyFrom(matmul_node);
        *(matmul_part1.mutable_input(0)) = gather_input_node.name();
        *(matmul_part1.mutable_input(1)) = slice_part1.name();
        matmul_part1.set_name(matmul_node.name() + "_part1");
        NodeDef matmul_part2;
        matmul_part2.CopyFrom(matmul_node);
        *(matmul_part2.mutable_input(0)) = new_concat.name();
        *(matmul_part2.mutable_input(1)) = slice_part2.name();
        matmul_part2.set_name(matmul_node.name() + "_part2");
        // 结果做加法
        std::vector<NodeDefBuilder::NodeOut> add_inputs;
        add_inputs.emplace_back(matmul_part1.name(), 0, output_type);
        add_inputs.emplace_back(matmul_part2.name(), 0, output_type);
        NodeDef add_node;
        status = NodeDefBuilder(matmul_node.name(), "Add")
                                      .Input(add_inputs[0])
                                      .Input(add_inputs[1])
                                      .Attr("T", output_type)
                                      .Finalize(&add_node);
        if (!status.ok()) {
          LOG(ERROR) << "Adding add nodedef build failed " << status;
          return status;
        }
        add_node.set_device(matmul_node.device());
        VLOG(1) << add_node.DebugString();

        // 5. 保留匹配的节点
        new_nodes->push_back(add_node);
        new_nodes->push_back(matmul_part1);
        new_nodes->push_back(matmul_part2);
        new_nodes->push_back(new_concat);
        // 为了使concat的其他输入能再次匹配到这个pattern，保留原gather和concat节点
        // 最终会在图中留下无后续依赖的concat节点
        new_nodes->push_back(concat_node);
        new_nodes->push_back(gather_node);
        new_nodes->push_back(slice_part1);
        new_nodes->push_back(slice_part2);
        new_nodes->push_back(begin_const_part1);
        new_nodes->push_back(size_const_part1);
        new_nodes->push_back(begin_const_part2);
        new_nodes->push_back(size_const_part2);
        new_nodes->push_back(weight_node);
        new_nodes->push_back(gather_ph_node);
        new_nodes->push_back(gather_input_node);
        new_nodes->push_back(gather_ind_node);
        new_nodes->push_back(gather_axis_node);

        is_changed = true;
        return Status::OK();
      },
      {}, output_graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "optimize gather concat failed " << status;
    return false;
  }
  return true;
}

bool OptimizeGemmCompression(GraphDef& input_graph, GraphDef* optimized_graph) {
  
  while(1) {
    bool graph_changed = false;
    bool result = OptimizeGatherConcatPattern(input_graph, optimized_graph, graph_changed);
    if (!result) return false;
    if (!graph_changed) break;
    input_graph = *optimized_graph;
  }
  return true;
}

}  // end namespace

Status GemmCompressionOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool optimize = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE", true, &optimize);
  if (!optimize) {
    *optimized_graph = item.graph;
    return Status::OK();
  }

  VLOG(0) << "GemmCompressionOptimizer is on.";

  GraphDef input_graph_def = item.graph;
  if(!OptimizeGemmCompression(input_graph_def, optimized_graph)) {
    LOG(INFO) << "optimize gemm compression failed";
    *optimized_graph = item.graph;
    return Status::OK();
  }
  *optimized_graph->mutable_versions() = item.graph.versions();
  if (VLOG_IS_ON(1)) {
    std::fstream f;
    static int pass = 0;
    f.open("after_gemm_compression_" + std::to_string(pass) + ".pb",
           std::fstream::out);
    f << optimized_graph->DebugString();
    f.close();
    pass++;
  }
  return Status::OK();
}

void GemmCompressionOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
