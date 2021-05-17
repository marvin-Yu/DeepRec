#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <algorithm>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.

class GetChildren_ParentIndicator: public OpKernel {
 public:
  explicit GetChildren_ParentIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const Tensor & nodes_tensor = context->input(0);
    const auto& nodes = nodes_tensor.vec<int>();

    //an indicator showing the parent of each node, must be monotonically increasing
    // e.g. a tree such as
    // 0
    // 1      2    3   4      
    // 5 6 7  8 9  10  11 12
    // , gives  -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, ...
    const Tensor & tree_tensor = context->input(1);
    const auto& tree = tree_tensor.vec<int>();

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

class GetChildren_SplitIndicator: public OpKernel {
 public:
  explicit GetChildren_SplitIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const Tensor & nodes_tensor = context->input(0);
    const auto& nodes = nodes_tensor.vec<int>();

    //an indicator showing the splits of level order traversal of a complete tree
    // e.g. a tree such as
    // 0
    // 1      2    3   4      
    // 5 6 7  8 9  10  11 12
    // , whose level order traversal is  0 | 1 2 3 4 | 5 6 7 ; 8 9 ; 10 ; 11 12 | ...
    // will be represented as  1, 5, 8, 10, 11, 13 ..., such that [a_i, a_i+i) is the children of i-th node 
    const Tensor & tree_tensor = context->input(1);
    const auto& tree = tree_tensor.vec<int>();

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
      for (int j = tree(node); j < tree(node); ++j) {
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

REGISTER_KERNEL_BUILDER(Name("GetChildren").Device(DEVICE_CPU), GetChildren_SplitIndicator);
