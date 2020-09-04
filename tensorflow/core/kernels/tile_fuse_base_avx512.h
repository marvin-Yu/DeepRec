#ifndef TENSORFLOW_CORE_KERNELS_TILE_FUSE_BASE_H
#define TENSORFLOW_CORE_KERNELS_TILE_FUSE_BASE_H

#define EIGEN_USE_THREADS

#ifdef TILE_VECTORIZE_AVX512
#include <immintrin.h>
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
  template <typename Device, typename Functor, typename T, typename OT>
    class TileFuseBase : public BinaryOp<Device, Functor> {
      public:
        explicit TileFuseBase(OpKernelConstruction* context) :
          BinaryOp<Device, Functor>(context, false) {
            ReadBoolFromEnvVar("USE_TILE_DIRECT", false, &use_direct_);
          }

        void Compute(OpKernelContext* context) override {
          OP_REQUIRES(context, CheckValid(context), errors::Internal("check valid false"));
          if (use_direct_ && CanDoBroadcast(context)) {
            BinaryOp<Device, Functor>::Compute(context);
          } else {
            TensorShape shape = GenerateOutShape(context);
            Tensor* output;
            OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
            const auto& input0 = context->input(0);
            const auto& input1 = context->input(1);
            auto ret = Compute(input0, input1, output, shape, f_);
            OP_REQUIRES(context, ret, errors::Internal("TileEqual compute failed"));
          }
        }

      private:
        void Compute(const float* input0, const float* input1, OT* out, int len) {
          #ifdef TILE_VECTORIZE_AVX512
          int packed = (len % 16);
          if (packed != 0) {
            for (int i = 0; i < len; ++i) {
              out[i] = f_(input0[i], input1[i]);
            }
          } else {
            int idx = 0;
            int times = len / 16;
            int total = 0;
            for (int i = 0; i < times; ++i) {
              __mm512i * i0 = (__mm512i*)(input0 + idx);
              __mm512i * i1 = (__mm512i*)(input1 + idx);
              auto ret =  _mm512_cmpeq_epi32_mask(*i0, *i1);
              for (int j = 0; j < 16; ++j) {
                out[total++] = (((ret) & (1 << j)) != 0);
              }
              idx += 16;
            }
            return;
          }
          #endif
          for (int i = 0; i < len; ++i) {
            out[i] = f_(input0[i], input1[i]);
          }
        }

        template<typename TIN>
        void Compute(const TIN* input0, const TIN* input1, OT* out, int len) {
          for (int i = 0; i < len; ++i) {
            out[i] = f_(input0[i], input1[i]);
          }
        }

        bool Compute(const Tensor& input0, const Tensor& input1,
            Tensor* output, const TensorShape& shape, Functor f) {
          switch (input0.dims()) {
            case 1 : {
                       auto i_data0 = input0.flat<T>().data();
                       auto i_data2 = input1.flat<T>().data();
                       auto out = output->flat<OT>().data();
                       for (int i = 0; i < shape.dim_size(0); ++i) {
                         out[i] = f_(i_data0[i % input0.dim_size(0)], i_data2[i % input1.dim_size(0)]);
                       }
                       return true;
                     }
            case 2 : {
                       auto i_data0 = input0.flat<T>().data();
                       auto i_data2 = input1.flat<T>().data();
                       auto out = output->flat<OT>().data();
                       std::vector<int> idx0;
                       std::vector<int> idx2;
                       idx0.reserve(shape.dim_size(1));
                       idx2.reserve(shape.dim_size(1));
                       for (int i = 0; i < shape.dim_size(1); ++i) {
                         idx0.push_back(i % input0.dim_size(1));
                         idx2.push_back(i % input1.dim_size(1));
                       }

                       int idx = 0;
                       for (int i = 0; i < shape.dim_size(0); ++i) {
                         auto d_0_0 = (i % input0.dim_size(0)) * input0.dim_size(1);
                         auto d_2_0 = (i % input1.dim_size(0)) * input1.dim_size(1);
                         for (int j = 0; j < shape.dim_size(1); ++j) {
                           out[idx++] = f_(i_data0[d_0_0 + idx0[j]], i_data2[d_2_0 + idx2[j]]);
                         }
                       }
                       return true;
                     }
            case 3 : {
                       auto i_data0 = input0.flat<T>().data();
                       auto i_data2 = input1.flat<T>().data();
                       auto out = output->flat<OT>().data();
                       auto special = (
                           input0.dim_size(0) == 1 && input1.dim_size(0) != 1 && 
                           input1.dim_size(1) == 1 && input0.dim_size(2) == input1.dim_size(2));
                       if (special) {
                         std::vector<int> d_0_1;
                         d_0_1.reserve(shape.dim_size(1));

                         for (int i = 0; i < shape.dim_size(1); ++i) {
                           d_0_1.push_back((i % input0.dim_size(1)) * input0.dim_size(2));
                         }
                         auto count = 0;

                         if (input0.dim_size(2) == input1.dim_size(2)) {
                           for (int i = 0; i < shape.dim_size(0); ++i) {
                             auto idx = (i % input1.dim_size(0)) * input1.dim_size(2);
                             for (int j = 0; j < shape.dim_size(1); ++j) {
                               Compute(i_data0 + d_0_1[j], i_data2 + idx, out + count, shape.dim_size(2));
                               count += shape.dim_size(2);
                             }
                           }
                         } else {
                           for (int i = 0; i < shape.dim_size(0); ++i) {
                             auto idx = (i % input1.dim_size(0)) * input1.dim_size(2);
                             for (int j = 0; j < shape.dim_size(1); ++j) {
                               for (int k = 0; k < shape.dim_size(2); ++k) {
                                 out[count++] = f_(i_data0[d_0_1[j] + k], i_data2[idx + k]);
                               }
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
                         if (input0.dim_size(2) == input1.dim_size(2)) {
                           for (int i = 0; i < shape.dim_size(0); ++i) {
                             auto d_0_0 = (i % input0.dim_size(0)) * input0.dim_size(1) * input0.dim_size(2);
                             auto d_2_0 = (i % input1.dim_size(0)) * input1.dim_size(1) * input1.dim_size(2);
                             for (int j = 0; j < shape.dim_size(1); ++j) {
                               for (int k = 0; k < shape.dim_size(2); ++k) {
                                 out[count++] = f_(i_data0[d_0_0 + d_0_1[j] + d_0_2[k]],  i_data2[d_2_0 + d_2_1[j] + d_2_2[k]]);
                               }
                             }
                           }
                         } else {
                           for (int i = 0; i < shape.dim_size(0); ++i) {
                             auto d_0_0 = (i % input0.dim_size(0)) * input0.dim_size(1) * input0.dim_size(2);
                             auto d_2_0 = (i % input1.dim_size(0)) * input1.dim_size(1) * input1.dim_size(2);
                             for (int j = 0; j < shape.dim_size(1); ++j) {
                               Compute(i_data0 + d_0_0 + d_0_1[j], i_data2 + d_2_0 + d_2_1[j], out + count, shape.dim_size(2));
                               count += shape.dim_size(2);
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
      private:
        virtual bool CanDoBroadcast(OpKernelContext* context) = 0;
        virtual TensorShape GenerateOutShape(OpKernelContext* context) = 0;
        virtual bool CheckValid(OpKernelContext* context) = 0;
      private:
        bool use_direct_;
        Functor f_;
    };
}

#endif //end TENSORFLOW_CORE_KERNELS_TILE_FUSE_BASE_H
