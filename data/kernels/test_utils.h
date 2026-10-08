#ifndef XLA_SERVING_DATA_KERNELS_TEST_UTILS_H_
#define XLA_SERVING_DATA_KERNELS_TEST_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "data/kernels/macros.h"  // from @xla_serving
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::kernel {

template <typename T>
xla_serving::HostBuffer CreateBuffer(absl::Span<const int64_t> dims,
                             const std::vector<T>& data) {
  xla::ifrt::Shape shape(dims);
  xla_serving::HostBuffer buffer(xla::ifrt::DType(NativeAsDTypeKind<T>::value), shape);
  T* base = static_cast<T*>(buffer.base());
  for (size_t i = 0; i < data.size(); ++i) {
    base[i] = data[i];
  }
  return buffer;
}

template <typename T>
std::vector<T> GetData(const xla_serving::HostBuffer& buffer) {
  const T* base = static_cast<const T*>(buffer.base());
  return std::vector<T>(base, base + buffer.shape().num_elements());
}

}  // namespace xla_serving::data::kernel

#endif  // XLA_SERVING_DATA_KERNELS_TEST_UTILS_H_
