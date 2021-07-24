#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/fbs_tensor_generated.h"

namespace tensorflow {
namespace fbs {
class TensorFB;
}

#define TENSOR_FB_BASE_SIZE 88
#define DIM_FB_STORE_SIZE 16
#define DEFAULT_BUFFER_SIZE 1024

class TensorFBUtil {
public:
    static flatbuffers::Offset<fbs::TensorFB> CreateTensorFB(
            flatbuffers::FlatBufferBuilder &fbb,
            fbs::DataType dt,
            flatbuffers::Offset<fbs::TensorShapeFB> shape_offset,
            flatbuffers::Offset<flatbuffers::Vector<int32_t>> offset)
    {
        switch(dt) {
        case fbs::DataType_DT_HALF:
        case fbs::DataType_DT_BFLOAT16:        
            return fbs::CreateTensorFB(fbb, fbs::DataType(dt), shape_offset, 0, 0, offset);
        case fbs::DataType_DT_INT32:
        case fbs::DataType_DT_UINT8:
        case fbs::DataType_DT_UINT16:
        case fbs::DataType_DT_INT16:
        case fbs::DataType_DT_INT8:
        case fbs::DataType_DT_QINT8:
        case fbs::DataType_DT_QUINT8:        
        case fbs::DataType_DT_QINT16:
        case fbs::DataType_DT_QUINT16:
        case fbs::DataType_DT_QINT32:
            return fbs::CreateTensorFB(fbb, dt, shape_offset, 0, 0, 0, 0, 0, offset);
        default:
            LOG(FATAL) << "fbs::DataType " << dt << " cant not use int32";
            return 0;
        }
    }

    static flatbuffers::Offset<fbs::TensorFB> CreateTensorFB(
            flatbuffers::FlatBufferBuilder &fbb,
            fbs::DataType dt,
            flatbuffers::Offset<fbs::TensorShapeFB> shape_offset,
            flatbuffers::Offset<flatbuffers::Vector<float>> offset)
    {
        switch(dt) {
        case fbs::DataType_DT_FLOAT:
            return fbs::CreateTensorFB(fbb, dt, shape_offset, 0, 0, 0, offset);
        case fbs::DataType_DT_COMPLEX64:
            return fbs::CreateTensorFB(fbb, dt, shape_offset, 0, 0, 0, 0, 0, 0, 0, offset);
        default:
            LOG(FATAL) << "fbs::DataType " << dt << " cant not use float";
            return 0;
        }
    }

    static flatbuffers::Offset<fbs::TensorFB> CreateTensorFB(
            flatbuffers::FlatBufferBuilder &fbb,
            fbs::DataType dt,
            flatbuffers::Offset<fbs::TensorShapeFB> shape_offset,
            flatbuffers::Offset<flatbuffers::Vector<double>> offset)
    {
        switch(dt) {
        case fbs::DataType_DT_DOUBLE:
            return fbs::CreateTensorFB(fbb, dt, shape_offset, 0, 0, 0, 0, offset);
        case fbs::DataType_DT_COMPLEX128:
            return fbs::CreateTensorFB(fbb, dt, shape_offset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, offset);
        default:
            LOG(FATAL) << "fbs::DataType " << dt << " cant not use float";
            return 0;
        }
    }

    static flatbuffers::Offset<fbs::TensorFB> CreateTensorFB(
            flatbuffers::FlatBufferBuilder &fbb, fbs::DataType dt,
            flatbuffers::Offset<fbs::TensorShapeFB> shape_offset,
            flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> offset)
    {
        switch(dt) {
        case fbs::DataType_DT_STRING:
            return fbs::CreateTensorFB(fbb, dt, shape_offset, 0, 0, 0, 0, 0, 0, offset);
        case fbs::DataType_DT_VARIANT:
            return fbs::CreateTensorFB(
                    fbb, dt, shape_offset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, offset.o);
        default:
            LOG(FATAL) << "fbs::DataType " << dt << " cant not use string";
            return 0;
        }
    }

    static flatbuffers::Offset<fbs::TensorFB> CreateTensorFB(
            flatbuffers::FlatBufferBuilder &fbb, fbs::DataType dt,
            flatbuffers::Offset<fbs::TensorShapeFB> shape_offset,
            flatbuffers::Offset<flatbuffers::Vector<int64_t>> offset)
    {
        return fbs::CreateTensorFB(fbb, dt, shape_offset, 0, 0, 0, 0, 0, 0, 0, 0, offset);
    }

    static flatbuffers::Offset<fbs::TensorFB> CreateTensorFB(
            flatbuffers::FlatBufferBuilder &fbb, fbs::DataType dt,
            flatbuffers::Offset<fbs::TensorShapeFB> shape_offset,
            flatbuffers::Offset<flatbuffers::Vector<uint8_t>> offset)
    {
        return fbs::CreateTensorFB(fbb, dt, shape_offset, 0, 0, 0, 0, 0, 0, 0, 0, 0, offset);
    }

    static flatbuffers::Offset<fbs::TensorFB> CreateTensorFB(
            flatbuffers::FlatBufferBuilder &fbb, fbs::DataType dt,
            flatbuffers::Offset<fbs::TensorShapeFB> shape_offset,
            flatbuffers::Offset<flatbuffers::Vector<bool>> offset)
    {
        return fbs::CreateTensorFB(fbb, dt, shape_offset, 0, 0, 0, 0, 0, 0, 0, 0, 0, offset.o);
    }

    static flatbuffers::Offset<fbs::TensorFB> CreateTensorFB(
            flatbuffers::FlatBufferBuilder &fbb, fbs::DataType dt,
            flatbuffers::Offset<fbs::TensorShapeFB> shape_offset,
            flatbuffers::Offset<flatbuffers::Vector<uint32_t>> offset)
    {
        return fbs::CreateTensorFB(
                fbb, dt, shape_offset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, offset);
    }

    static flatbuffers::Offset<fbs::TensorFB> CreateTensorFB(
            flatbuffers::FlatBufferBuilder &fbb, fbs::DataType dt,
            flatbuffers::Offset<fbs::TensorShapeFB> shape_offset,
            flatbuffers::Offset<flatbuffers::Vector<uint64>> offset)
    {
        return fbs::CreateTensorFB(
                fbb, dt, shape_offset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, offset.o);
    }

    static flatbuffers::Offset<fbs::TensorFB> CreateTensorFB(
            flatbuffers::FlatBufferBuilder &fbb, fbs::DataType dt,
            flatbuffers::Offset<fbs::TensorShapeFB> shape_offset,
            flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<fbs::VariantTensorDataFB>>> offset)
    {
        return fbs::CreateTensorFB(
                fbb, dt, shape_offset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, offset.o);
    }

    static size_t calFlatbufferSize(const Tensor &t);

private:
    template<typename T>
    static size_t calFlatbufferSizeTyped(T* data, size_t num_elements, size_t dim_size);
    
};

template <typename T>
struct FBHelper {};

#define FLAT_BUFFERS_TRAITS(T, F, N)                                    \
    template <>                                                         \
    struct FBHelper<T> {                                                \
        typedef F Type;                                                 \
        static const Type * Begin(const fbs::TensorFB& tensorFB) {      \
            return (Type*)(tensorFB.N##_val()->Data());                 \
        }                                                               \
        static size_t NumElements(const fbs::TensorFB& tensorFB) {      \
            return tensorFB.N##_val()->size();                          \
        }                                                               \
        static size_t CalFlatBufferSize(T* data, size_t num_elements, size_t dim_size) {\
            return DEFAULT_BUFFER_SIZE;                                 \
        }                                                               \
    };

FLAT_BUFFERS_TRAITS(Eigen::half, int32, half);
FLAT_BUFFERS_TRAITS(bfloat16, int32, half);
FLAT_BUFFERS_TRAITS(float, float, float);
FLAT_BUFFERS_TRAITS(double, double, double);
FLAT_BUFFERS_TRAITS(int32, int32, int);
FLAT_BUFFERS_TRAITS(uint8, int32, int);
FLAT_BUFFERS_TRAITS(uint16, int32, int);
FLAT_BUFFERS_TRAITS(int16, int32, int);
FLAT_BUFFERS_TRAITS(int8, int32, int);
FLAT_BUFFERS_TRAITS(qint8, int32, int);
FLAT_BUFFERS_TRAITS(quint8, int32, int);
FLAT_BUFFERS_TRAITS(qint16, int32, int);
FLAT_BUFFERS_TRAITS(quint16, int32, int);
FLAT_BUFFERS_TRAITS(qint32, int32, int);
FLAT_BUFFERS_TRAITS(int64, int64, int64);
FLAT_BUFFERS_TRAITS(bool, bool, bool);
FLAT_BUFFERS_TRAITS(uint32, uint32, uint32);
FLAT_BUFFERS_TRAITS(uint64, uint64, uint64);


#undef FLAT_BUFFERS_TRAITS

template <>
struct FBHelper<std::string> {
    static const std::string * Begin(const fbs::TensorFB& tensorFB) {
        return (std::string*)(tensorFB.string_val()->Data());
    }
    static size_t NumElements(const fbs::TensorFB& tensorFB) {
        return tensorFB.string_val()->size();
    }
    static size_t CalFlatBufferSize(std::string* data, size_t num_elements, size_t dim_size) {
        size_t char_size = 0;
        for (auto i = 0; i < num_elements; i++) {
            char_size += 4 * (3 + (data + i)->size() / 4);
        }
        return TENSOR_FB_BASE_SIZE + char_size + dim_size * DIM_FB_STORE_SIZE;
    }
};

template <>
struct FBHelper<Variant> {
    static const std::string * Begin(const fbs::TensorFB& tensorFB) {
        return (std::string*)(tensorFB.variant_val()->Data());
    }
    static size_t NumElements(const fbs::TensorFB& tensorFB) {
        return tensorFB.variant_val()->size();
    }
    static size_t CalFlatBufferSize(Variant* data, size_t num_elements, size_t dim_size) {
        return DEFAULT_BUFFER_SIZE;
    }
};

template <>
struct FBHelper<complex64> {
    static const complex64 *Begin(const fbs::TensorFB& tensorFB) {
        return reinterpret_cast<const complex64*>(tensorFB.scomplex_val()->Data());
    }
    static size_t NumElements(const fbs::TensorFB& tensorFB) {
        return tensorFB.scomplex_val()->size() / 2;
    }
    static size_t CalFlatBufferSize(complex64* data, size_t num_elements, size_t dim_size) {
        return DEFAULT_BUFFER_SIZE;
    }
};

template <>
struct FBHelper<complex128> {
    static const complex128 *Begin(const fbs::TensorFB& tensorFB) {
        return reinterpret_cast<const complex128*>(tensorFB.dcomplex_val()->Data());
    }
    static size_t NumElements(const fbs::TensorFB& tensorFB) {
        return tensorFB.dcomplex_val()->size() / 2;
    }
    static size_t CalFlatBufferSize(complex128* data, size_t num_elements, size_t dim_size) {
        return DEFAULT_BUFFER_SIZE;
    }
};

template <>
struct FBHelper<ResourceHandle> {
    static const ResourceHandle *Begin(const fbs::TensorFB& tensorFB) {
        LOG(FATAL) << "ResourceHandle for fb not support";
        return nullptr;
    }
    static size_t NumElements(const fbs::TensorFB& tensorFB) {
        LOG(FATAL) << "ResourceHandle for fb not support";
        return 0;
    }
    static size_t CalFlatBufferSize(ResourceHandle* data, size_t num_elements, size_t dim_size) {
        LOG(FATAL) << "ResourceHandle for fb not support";
        return 0;
    }
};

}
