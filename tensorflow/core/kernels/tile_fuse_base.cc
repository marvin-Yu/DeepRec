#include "tensorflow/core/kernels/tile_fuse_base.h"

namespace tensorflow {
	/*template<typename Device, typename Functor, typename T, typename OT>
		bool TileFuseBase<Device, Functor, T, OT>::Compute(const Tensor& input0, const Tensor& input1, Tensor* output, const TensorShape& shape, Functor f) {
			switch (input0.dims()) {
				case 1 : {
									 auto i_data0 = input0.flat<T>().data();
									 auto i_data2 = input1.flat<T>().data();
									 auto out = output->flat<OT>().data();

									 for (int i = 0; i < shape.dim_size(0); ++i) {
										 out[i] = f(i_data0[i % input0.dim_size(0)], i_data2[i % input1.dim_size(0)]);
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
											 out[idx++] = f(i_data0[d_0_0 + idx0[j]], i_data2[d_2_0 + idx2[j]]);
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

										 for (int i = 0; i < shape.dim_size(0); ++i) {
											 auto idx = (i % input1.dim_size(0)) * input1.dim_size(2);
											 for (int j = 0; j < shape.dim_size(1); ++j) {
												 for (int k = 0; k < shape.dim_size(2); ++k) {
													 out[count++] = f(i_data0[d_0_1[j] + k], i_data2[idx + k]);
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
											 auto d_2_0 = (i % input1.dim_size(0)) * input1.dim_size(1) * input1.dim_size(2);
											 for (int j = 0; j < shape.dim_size(1); ++j) {
												 for (int k = 0; k < shape.dim_size(2); ++k) {
													 out[count++] = f(i_data0[d_0_0 + d_0_1[j], d_0_2[k]]
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
		} */
} //end namespace tensorflow
