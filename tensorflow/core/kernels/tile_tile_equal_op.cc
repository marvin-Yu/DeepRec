#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/tile_fuse_base.h"

namespace tensorflow {
  const std::string empty_str = "";
  template <typename Device, typename Functor, typename T, typename OT>
    class TileTileEqualOp : public TileFuseBase<Device, Functor, T, OT> {
      public:
        explicit TileTileEqualOp(OpKernelConstruction* context) :
          TileFuseBase<Device, Functor, T, OT>(context) {}

      private:
        const string& GetName(OpKernelContext* context, int i) {
          if (i >= context->num_inputs()) {
            return empty_str;
          }
          return context->op_kernel().requested_input(i);
        }

        bool CheckValid(OpKernelContext* context) override {
          const auto& tile_0 = context->input(0);
          const auto& tile_1 = context->input(1);
          const auto& tile_multis0 = context->input(2).tensor<int32, 1>();
          const auto& tile_multis1 = context->input(3).tensor<int32, 1>();

          OP_REQUIRES_TRUE(context, tile_0.dims() == tile_1.dims(),
              errors::Internal(GetName(context, 0), " dims: ", tile_0.dims(), " != ",
                GetName(context, 1), " dims: ", tile_1.dims()));

          auto dims_0_2 = context->input(2).dim_size(0);
          auto dims_0_3 = context->input(3).dim_size(0);
          OP_REQUIRES_TRUE(context, dims_0_2 == dims_0_3,
              errors::Internal(GetName(context, 2), " dims size: ", dims_0_2, " != ",
                GetName(context, 3), " dims: ", dims_0_3));

          for (int i = 0; i < tile_0.dims(); ++i) {
            auto left = tile_0.dim_size(i) * tile_multis0(i);
            auto right = tile_1.dim_size(i) * tile_multis1(i);

            OP_REQUIRES_TRUE(context, (left == right || left == 1 || right == 1),
                errors::Internal(tile_0.dim_size(i), " * ", tile_multis0(i), " != ",
                  tile_1.dim_size(i), tile_multis1(i)));
          }
          return true;
        }

        bool CanDoBroadcast(OpKernelContext* context) override {
          const auto& tile_0 = context->input(0);
          const auto& tile_1 = context->input(1);
          const auto& tile_multis0 = context->input(2).tensor<int32, 1>();
          const auto& tile_multis1 = context->input(3).tensor<int32, 1>();

          for (int i = 0; i < tile_0.dims(); ++i) {
            if ((tile_0.dim_size(i) != 1 && tile_multis0(i) != 1) || 
                (tile_1.dim_size(i) != 1 && tile_multis1(i) != 1)) {
              return false;
            }

            if (tile_0.dim_size(i) == 1 && tile_1.dim_size(i) == 1 && tile_multis0(i) != 1) {
              return false;
            }
          }
          return true;
        }

        TensorShape GenerateOutShape(OpKernelContext* context) override {
          const auto& tile_0 = context->input(0);
          const auto& tile_1 = context->input(1);
          const auto& tile_multis0 = context->input(2).tensor<int32, 1>();
          const auto& tile_multis1 = context->input(3).tensor<int32, 1>();

          TensorShape shape;
          for (int i = 0; i < tile_0.dims(); ++i) {
            auto left = tile_0.dim_size(i) * tile_multis0(i);
            auto right = tile_1.dim_size(i) * tile_multis1(i);

            shape.AddDim(left == 1 ? right : left);
          }
          return shape;
        }
    };

#define REGISTER_TILE_TILE_EQUAL(T)                                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("TileTileEqual")                                              \
      .Device(DEVICE_CPU)                                            \
      .TypeConstraint<T>("T"),                                       \
      TileTileEqualOp<CPUDevice, functor::tile_equal_to<T>, T, bool>);

  REGISTER_TILE_TILE_EQUAL(float);
  REGISTER_TILE_TILE_EQUAL(double);
  REGISTER_TILE_TILE_EQUAL(int32);

#undef REGISTER_TILE_FUSE_EQUAL
}
