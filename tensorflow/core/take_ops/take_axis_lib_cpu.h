#ifndef TENSORFLOW_CORE_USER_OPS_TAKE_AXIS_LIB_CPU_H_
#define TENSORFLOW_CORE_USER_OPS_TAKE_AXIS_LIB_CPU_H_

#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/register_types.h"
#include "take_axis_lib.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

template <typename T, typename Index, bool reverse>
void take_axis_cpu(
    const T* val,
    const Index* beg,
    int64 unit_size, int64 col_size, int64 row_size, int64 axis_size, 
    T* out) {
  for (int64 i = 0; i < col_size*row_size; ++i) {
    int64 y = i / row_size;
    int64 x = i % row_size; 
    Index b = beg[y];
    //printf("y=%lld x=%lld b=%lld\n", y, x, b);
    if (reverse) {
      if (x < b || x-b >= axis_size) {
        memset(&out[y*row_size*unit_size+x*unit_size], 0, 
               unit_size*sizeof(T));
      } else {
        memcpy(&out[y*row_size*unit_size+x*unit_size],
               &val[y*axis_size*unit_size+(x-b)*unit_size],
               unit_size*sizeof(T));
      }
    } else {
      if (x+b >= axis_size) {
        memset(&out[y*row_size*unit_size+x*unit_size], 0, 
               unit_size*sizeof(T));
      } else {
        memcpy(&out[y*row_size*unit_size+x*unit_size],
               &val[y*axis_size*unit_size+(x+b)*unit_size],
               unit_size*sizeof(T));
      }
    }
  }
}

// ElementCopier must be a struct with a single Copy function, which is passed
// the output pointer, input pointer, input index, and number of elements to
// copy from input to output.
template <typename T, typename Index>
void TakeAxisCPUImpl(
    DeviceBase* d,
    const typename TTypes<T, 3>::ConstTensor& input,
    const typename TTypes<Index, 1>::ConstTensor& begin,
    int64 cost_per_unit,
    bool reserve,
    typename TTypes<T, 3>::Tensor* output) {
  int64 unit_size = output->dimension(2);
  CHECK(input.dimension(2) == unit_size);

  int64 col_size = output->dimension(0);
  CHECK(input.dimension(0) == col_size);

  int64 row_size = output->dimension(1);
  int64 axis_size = input.dimension(1);

  auto worker_threads = d->tensorflow_cpu_worker_threads();
  int num_threads = std::min(4, worker_threads->num_threads);
  num_threads = 0;

  // Single threaded mode.
  if (num_threads == 0) {
    if (reserve) {
      take_axis_cpu<T, Index, true>(input.data(), begin.data(),
                                     unit_size, col_size, row_size, axis_size, 
                                     output->data());
    } else {
      take_axis_cpu<T, Index, false>(input.data(), begin.data(),
                                      unit_size, col_size, row_size, axis_size, 
                                      output->data());
    }
    /*
    if (row_size < axis_size) {
      for (int64 i = 0; i < col_size*row_size; ++i) {
        int64 y = i / row_size;
        int64 x = i % row_size; 
        Index b = beg[y];
        //printf("y=%lld x=%lld b=%lld\n", y, x, b);
        DCHECK(x+b < axis_size);
        memcpy(&out[y*row_size*unit_size+x*unit_size],
               &val[y*axis_size*unit_size+(x+b)*unit_size],
               unit_size*sizeof(T));
      }
    } else {
      for (int64 i = 0; i < col_size*axis_size; ++i) {
        int64 y = i / axis_size;
        int64 x = i % axis_size; 
        Index b = beg[y];
        //printf("y=%lld x=%lld b=%lld\n", y, x, b);
        DCHECK(x+b < row_size);
        memcpy(&out[y*row_size*unit_size+(x+b)*unit_size],
               &val[y*axis_size*unit_size+x*unit_size],
               unit_size*sizeof(T));
      }
    }
    */
    return;
  }

  /*
  // Sharded mode.
  auto work = [&row_size, &sizes, &inputs, &output, &copier, &num_inputs](
                  int64 start, int64 end) {
    int64 skipped_rows = start / row_size;
    T* out = output->data() + skipped_rows * row_size;
    T* out_start = output->data() + start;
    T* out_end = output->data() + end;

    // Handle partial row at start
    if (out < out_start) {
      for (size_t j = 0; j < num_inputs; ++j) {
        ptrdiff_t size = sizes[j];
        ptrdiff_t offset = out_start - out;
        if (size <= offset) {
          out += size;
          continue;
        }
        const T* inp = &(*inputs[j])(skipped_rows, 0);
        if (offset > 0) {
          out += offset;
          inp += offset;
          size -= offset;
        }
        size = std::min(size, out_end - out);
        if (size <= 0) break;
        copier.Copy(out, inp, j, size);
        out += size;
      }
      ++skipped_rows;
    }
    if (out == out_end) return;
    CHECK(out >= out_start);
    CHECK(out < out_end);

    // Copy remaining data.
    std::vector<const T*> inp;
    inp.reserve(num_inputs);
    for (const auto& input : inputs) {
      inp.push_back(&(*input)(skipped_rows, 0));
    }
    const int64 dim0 = output->dimension(0);
    for (int64 i = skipped_rows; i < dim0; ++i) {
      for (int64 j = 0; j < num_inputs; ++j) {
        ptrdiff_t size = std::min(sizes[j], out_end - out);
        copier.Copy(out, inp[j], j, size);
        out += size;
        inp[j] += size;
        if (out == out_end) return;
      }
    }
  };
  Shard(worker_threads->num_threads, worker_threads->workers, output->size(),
        cost_per_unit, work);
  */
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_USER_OPS_TAKE_AXIS_LIB_CPU_H_
