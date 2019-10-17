#ifndef TENSORFLOW_CORE_USER_OPS_TAKE_LIB_CPU_H_
#define TENSORFLOW_CORE_USER_OPS_TAKE_LIB_CPU_H_

#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/register_types.h"
#include "take_lib.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

// ElementCopier must be a struct with a single Copy function, which is passed
// the output pointer, input pointer, input index, and number of elements to
// copy from input to output.
template <typename T, typename Index>
void TakeCPUImpl(
    DeviceBase* d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        values,
    const std::vector<std::unique_ptr<typename TTypes<Index, 2>::ConstMatrix>>&
        coords,
    int64 cost_per_unit,
    typename TTypes<T, 3>::Tensor* output) {
  size_t N = values.size();
  int64 unit_size = output->dimension(2);
  for (const auto& value: values) {
    CHECK(value->dimension(1) == unit_size);
  }

  int64 col_size = output->dimension(0);
  int64 row_size = output->dimension(1);

  auto worker_threads = d->tensorflow_cpu_worker_threads();
  int num_threads = std::min(4, worker_threads->num_threads);
  num_threads = 0;

  // Single threaded mode.
  if (num_threads == 0) {
    T* out = &(*output)(0, 0, 0);
    std::vector<const T*> vals;
    std::vector<const Index*> coos;
    vals.reserve(N);
    coos.reserve(N);
    for (size_t n = 0; n < N; ++n) {
      const auto& value = values[n];
      const auto& coord = coords[n];
      const T* val = &(*value)(0, 0);
      const Index *coo = &(*coord)(0, 0);
      int64 size = value->dimension(0);
      CHECK(size == coord->dimension(0));
      CHECK(2 == coord->dimension(1));
      for (int64 i = 0; i < size; ++i) {
        Index y = coo[i*2];
        Index x = coo[i*2+1];
        CHECK(y < col_size);
        memcpy(&out[y*row_size*unit_size+x*unit_size], &val[i*unit_size], unit_size*sizeof(T));
      }
    }
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

#endif  // TENSORFLOW_CORE_USER_OPS_TAKE_LIB_CPU_H_
