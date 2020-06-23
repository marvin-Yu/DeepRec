#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
  template <class T>
    class TileFuseOp : public OpKernel {
      public:
        explicit TileFuseOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
          auto& input0 = context->input(0);
          auto& input1 = context->input(1);
          auto& input2 = context->input(2);
          const auto* p = input1.flat<int32>().data();

          auto nelem = input1.NumElements();

          OP_REQUIRES(context, input0.dims() == nelem,
              errors::Internal("input0 dimsize: ", input0.dims(), " != ", nelem)); 
          OP_REQUIRES(context, input0.dims() == input2.dims(),
              errors::Internal("input0 dimsize: ", input0.dims(), " != ", input2.dims()));

          TensorShape shape;
          for (int i = 0; i < input0.dims(); ++i) {
            auto val = p[i] * input0.dim_size(i);
            if (val != 1 && input2.dim_size(i) != 1) {
              OP_REQUIRES(context, val == input2.dim_size(i),
                  errors::Internal(val, " not equal to ", input2.dim_size(i)));
              shape.AddDim(val);
            } else {
              shape.AddDim(val == 1 ? input2.dim_size(i) : val);
            }
          }

          Tensor* output;

          OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
          auto ret = Compute(input0, input1, input2, output, shape);
          OP_REQUIRES(context, ret, errors::Internal("TileEqual compute failed"));
        }

      private:
        bool Compute(const Tensor& input0, const Tensor& input1,
            const Tensor& input2, Tensor* output, const TensorShape& shape) {
          switch (input0.dims()) {
            case 1 : {
                       auto i_data0 = input0.flat<T>().data();
                       auto i_data2 = input2.flat<T>().data();
                       auto out = output->flat<bool>().data();

                       for (int i = 0; i < shape.dim_size(0); ++i) {
                         out[i] = (i_data0[i % input0.dim_size(0)] == i_data2[i % input2.dim_size(0)]);
                       }
                       return true;
                     }
            case 2 : {
                       auto i_data0 = input0.flat<T>().data();
                       auto i_data2 = input2.flat<T>().data();
                       auto out = output->flat<bool>().data();
                       std::vector<int> idx0;
                       std::vector<int> idx2;
                       idx0.reserve(shape.dim_size(1));
                       idx2.reserve(shape.dim_size(1));
                       for (int i = 0; i < shape.dim_size(1); ++i) {
                         idx0.push_back(i % input0.dim_size(1));
                         idx2.push_back(i % input2.dim_size(1));
                       }

                       int idx = 0;
                       for (int i = 0; i < shape.dim_size(0); ++i) {
                         auto d_0_0 = (i % input0.dim_size(0)) * input0.dim_size(1);
                         auto d_2_0 = (i % input2.dim_size(0)) * input2.dim_size(1);
                         for (int j = 0; j < shape.dim_size(1); ++j) {
                           out[idx++] = (i_data0[d_0_0 + idx0[j]] == i_data2[d_2_0 + idx2[j]]);
                         }
                       }
                       return true;
                     }
            case 3 : {
                       auto i_data0 = input0.flat<T>().data();
                       auto i_data2 = input2.flat<T>().data();
                       auto out = output->flat<bool>().data();
                       auto special = (
                           input0.dim_size(0) == 1 && input2.dim_size(0) != 1 && 
                           input2.dim_size(1) == 1 && input0.dim_size(2) == input2.dim_size(2));
                       if (special) {
                         std::vector<int> d_0_1;
                         d_0_1.reserve(shape.dim_size(1));

                         for (int i = 0; i < shape.dim_size(1); ++i) {
                           d_0_1.push_back((i % input0.dim_size(1)) * input0.dim_size(2));
                         }
                         auto count = 0;

                         for (int i = 0; i < shape.dim_size(0); ++i) {
                           auto idx = (i % input2.dim_size(0)) * input2.dim_size(2);
                           for (int j = 0; j < shape.dim_size(1); ++j) {
                             for (int k = 0; k < shape.dim_size(2); ++k) {
                               out[count++] = (i_data0[d_0_1[j] + k] == i_data2[idx + k]);
                             }
                           }
                         }
                       } else {
                         std::vector<int> d_0_1;
                         std::vector<int> d_0_2;
                         std::vector<int> d_2_1;
                         std::vector<int> d_2_2;
                         d_0_1.reserve(shape.dim_size(1));
                         d_2_1.reserve(shape.dim_size(1));
                         d_0_2.reserve(shape.dim_size(2));
                         d_2_2.reserve(shape.dim_size(2));
                         for (int i = 0; i < shape.dim_size(1); ++i) {
                           d_0_1.push_back((i % shape.dim_size(1)) * shape.dim_size(2));
                           d_2_1.push_back((i % shape.dim_size(1)) * shape.dim_size(2));
                         }
                         for (int i = 0; i < shape.dim_size(2); ++i) {
                           d_0_2.push_back(i % shape.dim_size(2));
                           d_2_2.push_back(i % shape.dim_size(2));
                         }
                         auto count = 0;
                         for (int i = 0; i < shape.dim_size(0); ++i) {
                           auto d_0_0 = (i % input0.dim_size(0)) * input0.dim_size(1) * input0.dim_size(2);
                           auto d_2_0 = (i % input2.dim_size(0)) * input2.dim_size(1) * input2.dim_size(2);
                           for (int j = 0; j < shape.dim_size(1); ++j) {
                             for (int k = 0; k < shape.dim_size(2); ++k) {
                               out[count++] = (i_data0[d_0_0 + d_0_1[j] + d_0_2[k]]
                                   == i_data2[d_2_0 + d_2_1[j] + d_2_2[k]]);
                             }
                           }
                         }
                       }
                       return true;
                     }
            default:
                     return false;
          }
          return false;
        }
    };

#define REGISTER(T) \
  REGISTER_KERNEL_BUILDER(Name("TileEqual") \
      .Device(DEVICE_CPU) \
      .TypeConstraint<T>("T"), \
      TileFuseOp<T>);
  REGISTER(int);
  REGISTER(float);
#undef REGISTER
}
