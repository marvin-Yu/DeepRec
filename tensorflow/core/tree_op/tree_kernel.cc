#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <algorithm>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


class GetChildren_ParentIndicator: public OpKernel {
 public:
  explicit GetChildren_ParentIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const auto& nodes = context->input(0).vec<int>();

    //an indicator showing the parent of each node, must be monotonically increasing
    // e.g. a tree such as
    // 0
    // 1      2    3   4      
    // 5 6 7  8 9  10  11 12
    // , gives  -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, ...
    const auto& tree = context->input(1).vec<int>();

    int num_nodes = nodes.dimension(0);
    std::vector<int> parents(nodes.data(), nodes.data()+num_nodes);
    std::sort(parents.begin(), parents.end());

    int avg_node_degree = 1024;
    std::vector<int> children;
    children.reserve(num_nodes*avg_node_degree);

    int child_node = 0;
    for (int i = 0; i < parents.size(); ++i) {
      int parent = parents[i];
      while (tree(child_node) <= parent) {
        if (tree(child_node) == parent) {
          children.push_back(child_node);
        }
        child_node++;
      }
    }
 
    //Allocate Output
    TensorShape output_shape({children.size()});
    Tensor *output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<int>();

    std::copy(children.begin(), children.end(), output.data());
  };
};

class FirstLevel_ParentIndicator: public OpKernel {
 public:
  explicit FirstLevel_ParentIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const auto& tree = context->input(0).vec<int>();

    int avg_node_degree = 1024;
    std::vector<int> first_level;
    first_level.reserve(avg_node_degree);

    for (int i = 0; i < tree.dimension(0); ++i) {
      if (tree(i) < 0) {
        first_level.push_back(i);
      } else {
        break;
      }
    }
 
    //Allocate Output
    TensorShape output_shape({first_level.size()});
    Tensor *output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<int>();

    std::copy(first_level.begin(), first_level.end(), output.data());
  };
};





class GetChildren_SplitIndicator: public OpKernel {
 public:
  explicit GetChildren_SplitIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const auto& nodes = context->input(0).vec<int>();

    // an indicator showing the splits of level order traversal of a complete tree
    // e.g. a tree such as
    // 0
    // 1      2    3   4      
    // 5 6 7  8 9  10  11 12
    // , whose level order traversal is  0 | 1 2 3 4 | 5 6 7 ; 8 9 ; 10 ; 11 12 | ...
    // will be represented as  1, 5, 8, 10, 11, 13 ..., such that [a_i, a_i+i) is the children of i-th node
    // it also represents a bunch of trees (a forest, or start from mid layer), e.g.
    // 0         1         2
    // 3     4   5         6      7   8
    // 9 10  11  12 13 14  15 16  17  18 19 20
    // represents as  3, 5, 6, 9, 11, 12, 15, 17, 18, 21
    // thous we know that the "first level" (roots of trees) is [0,3)=0, 1, 2, since the first element is 3.
    const auto& tree = context->input(1).vec<int>();

    int num_nodes = nodes.dimension(0);
    int num_children = 0;
    for (int i = 0; i < num_nodes; ++i) {
      int node = nodes(i);
      num_children += tree(node+1) - tree(node);
    }
    
    std::vector<int> children;
    children.reserve(num_children);

    for (int i = 0; i < num_nodes; ++i) {
      int node = nodes(i);
      for (int j = tree(node); j < tree(node+1); ++j) {
        children.push_back(j);
      }
    }
 
    //Allocate Output
    TensorShape output_shape({children.size()});
    Tensor *output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<int>();

    std::copy(children.begin(), children.end(), output.data());
  };
};

class FirstLevel_SplitIndicator: public OpKernel {
 public:
  explicit FirstLevel_SplitIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const auto& tree = context->input(0).vec<int>();

    int num_nodes = tree(0);
 
    //Allocate Output
    TensorShape output_shape({num_nodes});
    Tensor *output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<int>();

    for (int i = 0; i < num_nodes; ++i) {
      output(i) = i;
    }
  };
};

REGISTER_KERNEL_BUILDER(Name("GetChildren").Device(DEVICE_CPU), GetChildren_SplitIndicator);
REGISTER_KERNEL_BUILDER(Name("FirstLevel").Device(DEVICE_CPU), FirstLevel_SplitIndicator);
