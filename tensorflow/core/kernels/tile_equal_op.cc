#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/tile_fuse_base.h"

namespace tensorflow {
  template <typename Device, typename Functor, typename T, typename OT>
    class TileFuseEqualOp : public TileFuseBase<Device, Functor, T, OT> {
      public:
        explicit TileFuseEqualOp(OpKernelConstruction* context) :
          TileFuseBase<Device, Functor, T, OT>(context) {}

      private:
        bool CheckValid(OpKernelContext* context) override {
          const auto& equal_to = context->input(0);
          const auto& tile_in = context->input(1);
          const auto& tile_multis = context->input(2).tensor<int32, 1>();
          OP_REQUIRES_TRUE(context, equal_to.dims() == tile_in.dims(),
              errors::Internal("dims: " ,equal_to.dims(), " != ", tile_in.dims()));

          OP_REQUIRES_TRUE(context, equal_to.dims() == context->input(2).dim_size(0),
              errors::Internal("dims: " ,equal_to.dims(), " != ", context->input(2).dim_size(0)));

          for (int i = 0; i < equal_to.dims(); ++i) {
            auto left = tile_in.dim_size(i) * tile_multis(i);
            OP_REQUIRES_TRUE(context, (left == equal_to.dim_size(i)
                  || left == 1 || equal_to.dim_size(i) == 1),
                errors::Internal(left, " != ", equal_to.dim_size(i)));
          }
          return true;
        }

        bool CanDoBroadcast(OpKernelContext* context) override {
          const auto& equal_to = context->input(0);
          const auto& tile_in = context->input(1);
          const auto& tile_multis = context->input(2).tensor<int32, 1>();

          for (int i = 0; i < equal_to.dims(); ++i) {
            if (equal_to.dim_size(i) != 1) {
              if (tile_in.dim_size(i) != 1 && tile_multis(i) != 1) {
                return false;
              }
            } else {
              if (tile_multis(i) != 1) {
                return false;
              }
            }
          }
          return true;
        }

        TensorShape GenerateOutShape(OpKernelContext* context) override {
          const auto& equal_to = context->input(0);
          const auto& tile_in = context->input(1);
          const auto& tile_multis = context->input(2).tensor<int32, 1>();

          TensorShape shape;
          for (int i = 0; i < equal_to.dims(); ++i) {
            if (equal_to.dim_size(i) != 1) {
              shape.AddDim(equal_to.dim_size(i));
            } else {
              shape.AddDim(tile_in.dim_size(i) * tile_multis(i));
            }
          }
          return shape;
        }
    };

#define REGISTER_TILE_FUSE_EQUAL(T)                     \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("TileFuseEqual")                                                  \
      .Device(DEVICE_CPU)                                  \
      .TypeConstraint<T>("T"),                              \
      TileFuseEqualOp<CPUDevice, functor::tile_equal_to<T>, T, bool>);

  REGISTER_TILE_FUSE_EQUAL(float);
  REGISTER_TILE_FUSE_EQUAL(double);
  REGISTER_TILE_FUSE_EQUAL(int32);

#undef REGISTER_TILE_FUSE_EQUAL
}
