/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/tensor_util.h"

#include <cmath>
#include <vector>
#include <fstream>

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensor {

static const float EPSILON = 0.0001;

Tensor DeepCopy(const Tensor& other) {
  Tensor tmp = Tensor(other.dtype(), other.shape());
  DeepCopy(other, &tmp);
  return tmp;
}

void DeepCopy(const Tensor& input, Tensor* output) {
  if (DataTypeCanUseMemcpy(input.dtype())) {
    if (input.NumElements() > 0) {
      StringPiece input_data = input.tensor_data();

      // We use StringPiece as a convenient map over the tensor buffer,
      // but we cast the type to get to the underlying buffer to do the
      // copy.
      StringPiece output_data = output->tensor_data();
      memcpy(const_cast<char*>(output_data.data()), input_data.data(),
             input_data.size());
    }
  } else if (input.dtype() == DT_STRING) {
    output->unaligned_flat<tstring>() = input.unaligned_flat<tstring>();
  } else {
    CHECK_EQ(DT_VARIANT, input.dtype());
    output->unaligned_flat<Variant>() = input.unaligned_flat<Variant>();
  }
}

void PrintTensorData(const Tensor& t) {
  const void* data;
  /* if (t.dtype() == DT_HALF) {
    data = static_cast<const void*>(t.flat<Eigen::half>().data());
  } else  */ 
  if (t.dtype() == DT_FLOAT) {
    data = static_cast<const void*>(t.flat<float>().data());
  } else if (t.dtype() == DT_BOOL) {
    data = static_cast<const void*>(t.flat<bool>().data());
  } else if (t.dtype() == DT_INT32) {
    data = static_cast<const void*>(t.flat<int>().data());
  } else {
    LOG(INFO) << "Print Tensor: Unsupported data type!" << std::endl;
    return;
  }

  int dims = t.dims();
  std::ostringstream tensor_string;
  tensor_string << "shape: " << std::endl;
  for (int i = 0; i < dims; i++) {
    tensor_string << t.dim_size(i) << ", ";
  }
  tensor_string << std::endl;

  int size = t.NumElements();
  size = size > 32 ? 32 : size;

  for (int i = 0; i < size; i++) {
    float value;
    /* if (t.dtype() == DT_HALF) {
      value = __half2float(static_cast<const __half*>(data)[i]);
    } else */
    if (t.dtype() == DT_INT32) {
      value = static_cast<const int*>(data)[i];
    } else if (t.dtype() == DT_BOOL) {
      value = static_cast<const bool*>(data)[i];
    } else {
      value = static_cast<const float*>(data)[i];
    }
    tensor_string << value << ",";
  }
  LOG(INFO) << tensor_string.str();
}

bool CheckTensorEquality(const Tensor& a, const Tensor& b) {
  if (a.dtype() != b.dtype()) {
    LOG(ERROR) << "Tensor type not equal, tensor a is " << a.dtype() 
               << " tensor b is " << b.dtype();
    return false;
  }
  if (a.dtype() != DT_FLOAT && a.dtype() != DT_INT32) {
    LOG(ERROR) << "Check Tensor Equality: Unsupported data type " << a.dtype();
    return false;
  }
  if (a.NumElements() != b.NumElements()) {
        LOG(ERROR) << "Tensor num elememts not equal, tensor a is " << a.NumElements() 
                   << " tensor b is " << b.NumElements();
    return false;
  }

  if (a.dtype() == DT_INT32) {
    const int* a_data = a.flat<int>().data();
    const int* b_data = b.flat<int>().data();

    for (int i = 0; i < a.NumElements(); ++i) {
      if (a_data[i] != b_data[i]) {
        LOG(ERROR) << "Tensor content not equal, index " << i 
                   << " tensor a is " << a_data[i]
                   << " tensor b is " << b_data[i];
        return false;
      }
    }
  } else {
    const float* a_data = a.flat<float>().data();
    const float* b_data = b.flat<float>().data();

    for (int i = 0; i < a.NumElements(); ++i) {
      if (fabs(a_data[i] - b_data[i]) > EPSILON) {
        LOG(ERROR) << "Tensor content not equal, index " << i 
                   << " tensor a is " << a_data[i]
                   << " tensor b is " << b_data[i];
        return false;
      }
    }  
  }
  return true;
}

Status Concat(const gtl::ArraySlice<Tensor>& tensors, Tensor* result) {
  if (tensors.empty()) {
    return errors::InvalidArgument("Cannot concatenate zero tensors");
  }
  int64 total_dim0_size = 0;
  for (const Tensor& tensor : tensors) {
    if (tensor.dims() == 0) {
      return errors::InvalidArgument(
          "Cannot concatenate a zero-dimensional tensor");
    }
    total_dim0_size += tensor.dim_size(0);
  }
  TensorShape shape = tensors[0].shape();
  shape.set_dim(0, total_dim0_size);

  const DataType dtype = tensors[0].dtype();
  for (int i = 1; i < tensors.size(); ++i) {
    if (tensors[i].dtype() != dtype) {
      return errors::InvalidArgument(
          "Cannot concatenate tensors that have different data types");
    }
  }
  *result = Tensor(dtype, shape);

  // We use StringPiece as a convenient map over the tensor buffer,
  // but we cast the type to get to the underlying buffer to do the
  // copy.
  StringPiece to_data = result->tensor_data();

  if (DataTypeCanUseMemcpy(dtype)) {
    int64 offset = 0;
    for (const Tensor& tensor : tensors) {
      StringPiece from_data = tensor.tensor_data();
      CHECK_LE(offset + from_data.size(), to_data.size());
      memcpy(const_cast<char*>(to_data.data()) + offset, from_data.data(),
             from_data.size());

      offset += from_data.size();
    }
  } else {
    if (dtype != DT_STRING) {
      return errors::Internal("Unexpected data type");
    }
    tstring* to_strings =
        reinterpret_cast<tstring*>(const_cast<char*>(to_data.data()));

    int64 offset = 0;
    for (const Tensor& tensor : tensors) {
      auto from_strings = tensor.flat<tstring>();
      CHECK_LE(offset + tensor.NumElements(), result->NumElements());
      for (int i = 0; i < tensor.NumElements(); ++i) {
        to_strings[offset + i] = from_strings(i);
      }

      offset += tensor.NumElements();
    }
  }

  return Status::OK();
}

Status Split(const Tensor& tensor, const gtl::ArraySlice<int64>& sizes,
             std::vector<Tensor>* result) {
  if (tensor.dims() == 0) {
    return errors::InvalidArgument("Cannot split a zero-dimensional tensor");
  }
  int64 total_size = 0;
  for (int64 size : sizes) {
    total_size += size;
  }
  if (total_size != tensor.dim_size(0)) {
    return errors::InvalidArgument(
        "The values in 'sizes' do not sum to the zeroth-dimension size of "
        "'tensor'");
  }

  StringPiece from_data = tensor.tensor_data();

  if (DataTypeCanUseMemcpy(tensor.dtype())) {
    int64 offset = 0;
    for (int64 size : sizes) {
      TensorShape shape = tensor.shape();
      shape.set_dim(0, size);
      result->emplace_back(tensor.dtype(), shape);
      Tensor* split = &(*result)[result->size() - 1];

      // We use StringPiece as a convenient map over the tensor buffer,
      // but we cast the type to get to the underlying buffer to do the
      // copy.
      StringPiece to_data = split->tensor_data();
      CHECK_LE(offset + to_data.size(), from_data.size());
      memcpy(const_cast<char*>(to_data.data()), from_data.data() + offset,
             to_data.size());

      offset += to_data.size();
    }
  } else {
    if (tensor.dtype() != DT_STRING) {
      return errors::Internal("Unexpected data type");
    }
    auto from_strings = tensor.flat<tstring>();

    int64 offset = 0;
    for (int64 size : sizes) {
      TensorShape shape = tensor.shape();
      shape.set_dim(0, size);
      result->emplace_back(tensor.dtype(), shape);
      Tensor& split = (*result)[result->size() - 1];
      tstring* to_strings = reinterpret_cast<tstring*>(
          const_cast<char*>(split.tensor_data().data()));

      CHECK_LE(offset + split.NumElements(), tensor.NumElements());
      for (int i = 0; i < split.NumElements(); ++i) {
        to_strings[i] = from_strings(offset + i);
      }

      offset += split.NumElements();
    }
  }

  return Status::OK();
}

namespace internal {
void SetTensorProtoShape(std::vector<size_t> shape,
                         TensorShapeProto* shape_proto) {
  for (auto dim : shape) {
    shape_proto->mutable_dim()->Add()->set_size(dim);
  }
}

template <typename T>
bool CompressTensorContent(float min_compression_ratio,
                           const TensorShape& shape, TensorProto* tensor) {
  using TypeHelper = internal::TensorProtoHelper<T>;
  using FieldType = typename internal::TensorProtoHelper<T>::FieldType;
  const int64 num_tensor_values = shape.num_elements();
  const int64 num_bytes = tensor->tensor_content().size();
  const int64 num_raw_values = num_bytes / sizeof(T);
  if (num_raw_values != num_tensor_values) {
    // Invalid or too small.
    return false;
  }
  int64 last_offset = num_bytes - 1;
  int64 prev_offset = last_offset - sizeof(T);
  // Inspect individual raw bytes sizeof(T) bytes apart in adjacent elements,
  // starting from the end, to find the last pair of elements that are not
  // identical.
  while (prev_offset >= 0) {
    if (tensor->tensor_content()[prev_offset] !=
        tensor->tensor_content()[last_offset]) {
      break;
    }
    --last_offset;
    --prev_offset;
  }
  // Round up to the next whole number of element of type T.
  const int64 new_num_values = last_offset / sizeof(T) + 1;
  if (new_num_values * (is_complex<T>::value ? 2 : 1) * sizeof(FieldType) >
      static_cast<int64>(num_bytes / min_compression_ratio)) {
    return false;
  }
  // Copy values to truncated repeated field.
  if (sizeof(FieldType) == sizeof(T)) {
    FieldType* dst_ptr =
        TypeHelper::AppendUninitialized(new_num_values, tensor);
    port::CopySubrangeToArray(tensor->tensor_content(), 0,
                              new_num_values * sizeof(T),
                              reinterpret_cast<char*>(dst_ptr));
    tensor->clear_tensor_content();
  } else if (sizeof(T) > 1) {
    // Copy raw bytes to temp array first, then cast.
    gtl::InlinedVector<T, 64> tmp(new_num_values);
    port::CopySubrangeToArray(tensor->tensor_content(), 0,
                              new_num_values * sizeof(T),
                              reinterpret_cast<char*>(tmp.data()));
    tensor->clear_tensor_content();
    const T* begin = tmp.begin();
    const T* end = tmp.end();
    TypeHelper::AddValues(begin, end, tensor);
  } else {
    // Copy and cast, one byte at a time.
    for (int64 i = 0; i < new_num_values; ++i) {
      char c = tensor->tensor_content()[i];
      TypeHelper::AddValue(static_cast<T>(c), tensor);
    }
    tensor->clear_tensor_content();
  }
  return true;
}

template <typename T>
inline bool PackedValuesNotEqual(T a, T b) {
  return a != b;
}
template <>
inline bool PackedValuesNotEqual(float a, float b) {
  return reinterpret_cast<int32_t&>(a) != reinterpret_cast<int32_t&>(b);
}
template <>
inline bool PackedValuesNotEqual(double a, double b) {
  return reinterpret_cast<int64_t&>(a) != reinterpret_cast<int64_t&>(b);
}
template <typename RealType>
inline bool PackedValuesNotEqual(const std::complex<RealType>& a,
                                 const std::complex<RealType>& b) {
  return PackedValuesNotEqual(a.real(), b.real()) ||
         PackedValuesNotEqual(a.imag(), b.imag());
}

template <typename T>
bool CompressRepeatedField(float min_compression_ratio,
                           const TensorShape& shape, TensorProto* tensor) {
  using TypeHelper = internal::TensorProtoHelper<T>;
  using FieldType = typename internal::TensorProtoHelper<T>::FieldType;
  const int64 num_tensor_values = shape.num_elements();
  // Notice that for complex types the tensor is stored as an array of up to
  // 2 * num_tensor_values real values (real and imaginary parts), possibly
  // truncated.
  const int64 num_proto_values = TypeHelper::NumValues(*tensor);
  if (num_proto_values != num_tensor_values) {
    // Already compressed or invalid.
    return false;
  }
  const T last_value = TypeHelper::GetValue(num_proto_values - 1, *tensor);
  int64 last_index = 0;
  for (int64 i = num_proto_values - 2; i >= 0 && last_index == 0; --i) {
    const T cur_value = TypeHelper::GetValue(i, *tensor);
    if (PackedValuesNotEqual(cur_value, last_value)) {
      last_index = i + 1;
    }
  }
  const int64 num_truncated_proto_values = last_index + 1;
  const int64 num_bytes_as_field =
      num_truncated_proto_values * sizeof(FieldType);
  const int64 num_bytes_as_tensor_content = num_tensor_values * sizeof(T);
  const int64 num_bytes_before = num_proto_values * sizeof(FieldType);
  if (std::min(num_bytes_as_field, num_bytes_as_tensor_content) >
      static_cast<int64>(num_bytes_before / min_compression_ratio)) {
    return false;
  }
  if (num_bytes_as_field <= num_bytes_as_tensor_content) {
    TypeHelper::Truncate(num_truncated_proto_values, tensor);
  } else {
    gtl::InlinedVector<T, 64> tmp(num_tensor_values);
    TypeHelper::CopyValues(tmp.begin(), *tensor);
    TypeHelper::Truncate(0, tensor);
    port::CopyFromArray(tensor->mutable_tensor_content(),
                        reinterpret_cast<const char*>(tmp.data()),
                        num_bytes_as_tensor_content);
  }
  return true;
}

template <typename T>
bool CompressTensorProtoInPlaceImpl(int64 min_num_elements,
                                    float min_compression_ratio,
                                    TensorProto* tensor) {
  const TensorShape shape(tensor->tensor_shape());
  const int64 num_tensor_values = shape.num_elements();
  if (num_tensor_values < min_num_elements) {
    return false;
  }
  if (tensor->tensor_content().empty()) {
    return CompressRepeatedField<T>(min_compression_ratio, shape, tensor);
  } else {
    return CompressTensorContent<T>(min_compression_ratio, shape, tensor);
  }
  return true;
}

}  // namespace internal

#define HANDLE_COMPRESS_CASE(TF_TYPE)                                  \
  case TF_TYPE:                                                        \
    return internal::CompressTensorProtoInPlaceImpl<                   \
        EnumToDataType<TF_TYPE>::Type>(min_num_elements,               \
                                       min_compression_ratio, tensor); \
    break

bool CompressTensorProtoInPlace(int64 min_num_elements,
                                float min_compression_ratio,
                                TensorProto* tensor) {
  switch (tensor->dtype()) {
    HANDLE_COMPRESS_CASE(DT_FLOAT);
    HANDLE_COMPRESS_CASE(DT_DOUBLE);
    HANDLE_COMPRESS_CASE(DT_COMPLEX64);
    HANDLE_COMPRESS_CASE(DT_COMPLEX128);
    HANDLE_COMPRESS_CASE(DT_UINT8);
    HANDLE_COMPRESS_CASE(DT_INT8);
    HANDLE_COMPRESS_CASE(DT_UINT16);
    HANDLE_COMPRESS_CASE(DT_INT16);
    HANDLE_COMPRESS_CASE(DT_UINT32);
    HANDLE_COMPRESS_CASE(DT_INT32);
    HANDLE_COMPRESS_CASE(DT_UINT64);
    HANDLE_COMPRESS_CASE(DT_INT64);
    HANDLE_COMPRESS_CASE(DT_BOOL);
    HANDLE_COMPRESS_CASE(DT_QUINT8);
    HANDLE_COMPRESS_CASE(DT_QINT8);
    HANDLE_COMPRESS_CASE(DT_QUINT16);
    HANDLE_COMPRESS_CASE(DT_QINT16);
    HANDLE_COMPRESS_CASE(DT_QINT32);
    HANDLE_COMPRESS_CASE(DT_HALF);
    HANDLE_COMPRESS_CASE(DT_BFLOAT16);
    default:
      return false;
  }
}
#undef HANDLE_COMPRESS_CASE
inline const string  DumpOneElement(const strings::AlphaNum& a,
                                                bool print_v2) {
  return StrCat(a);
}
inline string DumpOneElement(const tstring& a, bool print_v2) {
  if (print_v2) {
    return "\"" + absl::CEscape(a) + "\"";
  } else {
    return absl::CEscape(a);
  }
}
inline float DumpOneElement(const Eigen::half& h, bool print_v2) {
  return static_cast<float>(h);
}

// Dump from left dim to right dim recursively.
template <typename T>
void DumpOneDim(int dim_index, const gtl::InlinedVector<int64, 4>& shape,
                 int64 limit, int shape_size, const T* data, int64* data_index,
                 std::ofstream &ofs) {
  if (*data_index >= limit) return;
  int64 element_count = shape[dim_index];
  // We have reached the right-most dimension of the tensor.
  if (dim_index == shape_size - 1) {
    for (int64 i = 0; i < element_count; i++) {
      if (*data_index >= limit) {
        // If not enough elements has been printed, append "...".
        if (dim_index != 0) {
          ofs << "...";
        }
        return;
      }
      if (i > 0) ofs << " ";
      ofs << DumpOneElement(data[(*data_index)++], false);
    }
    return;
  }
  // Loop every element of one dim.
  for (int64 i = 0; i < element_count; i++) {
    bool flag = false;
    if (*data_index < limit) {
      ofs << "[";
      flag = true;
    }
    // As for each element, print the sub-dim.
    DumpOneDim(dim_index + 1, shape, limit, shape_size, data, data_index,
                ofs);
    if (*data_index < limit || flag) {
      ofs << "]";
      flag = false;
    }
  }
}

// Appends the spacing between elements for a given dim onto a result string
void DumpDimSpacing(int dim_index, int num_dims, std::ofstream &ofs) {
  if (dim_index == num_dims - 1) {
    ofs << " ";
    return;
  }
  for (int j = 0; j < num_dims - dim_index - 1; j++) {
    ofs << "\n";
  }
  for (int j = 0; j <= dim_index; j++) {
    ofs << " ";
  }
}

// Dump from left dim to right dim recursively.
template <typename T>
void DumpOneDimV2(int dim_index, const gtl::InlinedVector<int64, 4>& shape,
                   int64 num_elts_at_ends, int num_dims, const T* data,
                   int64 data_index, std::ofstream &ofs) {
  // We have recursed beyond all the dimensions into a single element
  // of the tensor.
  if (dim_index == num_dims) {
    ofs << DumpOneElement(data[data_index], true);
    return;
  }

  ofs << "[";
  int64 element_count = shape[dim_index];
  int64 start_of_end =
      std::max(num_elts_at_ends, element_count - num_elts_at_ends);

  // Loop every element of one dim.
  int64 elements_per_iter = 1;
  for (int i = dim_index + 1; i < num_dims; i++) {
    elements_per_iter *= shape[i];
  }
  for (int64 i = 0; (i < num_elts_at_ends) && (i < element_count); i++) {
    if (i > 0) {
      DumpDimSpacing(dim_index, num_dims, ofs);
    }

    // As for each element, print the sub-dim.
    DumpOneDimV2(dim_index + 1, shape, num_elts_at_ends, num_dims, data,
                  data_index + elements_per_iter * i, ofs);
  }
  if (element_count > 2 * num_elts_at_ends) {
    DumpDimSpacing(dim_index, num_dims, ofs);
    ofs << "...";
  }
  for (int64 i = start_of_end; i < element_count; i++) {
    // As for each element, print the sub-dim.
    DumpDimSpacing(dim_index, num_dims, ofs);
    DumpOneDimV2(dim_index + 1, shape, num_elts_at_ends, num_dims, data,
                  data_index + elements_per_iter * i, ofs);
  }

  ofs << "]";
}

template <typename T>
void DumpTensor(std::ofstream &ofs, int64 limit, int64 num_elts,
                      const TensorShape& tensor_shape, const char* data,
                      const bool print_v2) {
  const T* array = reinterpret_cast<const T*>(data);

  const gtl::InlinedVector<int64, 4> shape = tensor_shape.dim_sizes();
  if (shape.empty()) {
    for (int64 i = 0; i < limit; ++i) {
      if (i > 0) ofs << " ";
      ofs << DumpOneElement(array[i], print_v2);
    }
    if (num_elts > limit) ofs << "...";
    return;
  }
  if (print_v2) {
    const int num_dims = tensor_shape.dims();
    DumpOneDimV2(0, shape, limit, num_dims, array, 0, ofs);
  } else {
    int64 data_index = 0;
    const int shape_size = tensor_shape.dims();
    DumpOneDim(0, shape, limit, shape_size, array, &data_index, ofs);

    if (num_elts > limit) ofs << "...";
  }
}

void DumpTensorToFile(std::ofstream &ofs, const Tensor& tensor, bool print_v2) {
  ofs << "Tensor<type: " << DataTypeString(tensor.dtype());
  ofs << " shape: " << tensor.shape().DebugString();
  ofs << " values: " << std::endl;
  const int64 num_elts = tensor.NumElements();
  if (!tensor.IsInitialized()) {
    ofs << "uninitialized Tensor of " << num_elts << " elements of type " << tensor.dtype() << std::endl;
    return;
  }
  const char* data = tensor.tensor_data().data();
  int64 limit = num_elts;
  switch (tensor.dtype()) {
    case DT_HALF:
      DumpTensor<Eigen::half>(ofs, limit, num_elts, tensor.shape(), data,
                                         print_v2);
      break;
    case DT_FLOAT:
      DumpTensor<float>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_DOUBLE:
      DumpTensor<double>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_UINT32:
      DumpTensor<uint32>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_INT32:
      DumpTensor<int32>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_UINT8:
    case DT_QUINT8:
      DumpTensor<uint8>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_UINT16:
    case DT_QUINT16:
      DumpTensor<uint16>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_INT16:
    case DT_QINT16:
      DumpTensor<int16>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_INT8:
    case DT_QINT8:
      DumpTensor<int8>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_UINT64:
      DumpTensor<uint64>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_INT64:
      DumpTensor<int64>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_BOOL:
      // TODO(tucker): Is it better to emit "True False..."?  This
      // will emit "1 0..." which is more compact.
      DumpTensor<bool>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    case DT_STRING:
      DumpTensor<tstring>(ofs, limit, num_elts, tensor.shape(), data, print_v2);
      break;
    default: {
      // All irregular cases
      if (print_v2) {
        ofs << "[";
      }
      // TODO(irving): Don't call flat every time around this
      // loop.
      for (size_t i = 0; i < num_elts; ++i) {
        if (i > 0) ofs << " ";
        switch (tensor.dtype()) {
          case DT_VARIANT: {
            const Variant& v = tensor.flat<Variant>()(i);
            ofs << v.DebugString();
          } break;
          default:
          // TODO(zhifengc, josh11b): Pretty-print other types (bool,
          // complex64, quantized).
          ofs << "?";
        }
      }
      if (print_v2) {
        ofs << "]";
      }
    }
  }
  ofs << ">" << std::endl;
}

}  // namespace tensor
}  // namespace tensorflow
