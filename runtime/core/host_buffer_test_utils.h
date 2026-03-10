#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_TEST_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_TEST_UTILS_H_

#include <type_traits>
#include <utility>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/core/allocator.h"  // from @neptune
#include "runtime/core/default_allocator.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/host_buffer_type_utils.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace jsv {

// Builds a `HostBuffer` from a list of values and a shape with the given
// allocator. It will pick a `xla::ifrt::Dtype` that has the same storage width
// as the native C++ T type. Examples include int32, float, etc. The default
// allocator will be used if not specified.
template <typename T>
HostBuffer AsHostBuffer(
    absl::Span<const T> values, xla::ifrt::Shape shape,
    Allocator* allocator = DefaultAllocator::GetInstance()) {
  CHECK_EQ(values.size(), shape.num_elements())
      << "Values size " << values.size() << " does not match the shape "
      << shape;
  HostBuffer host_buffer(xla::ifrt::DType(NativeAsDTypeKind<T>::value),
                         std::move(shape), allocator);
  auto flat = host_buffer.flat<T>();
  for (int i = 0; i < values.size(); ++i) {
    flat(i) = values[i];
  }
  return host_buffer;
}

// Same as above, but allows the source type to be different from the target
// type and will perform a static_cast.
template <typename T, typename SrcT,
          typename = std::enable_if_t<!std::is_same<T, SrcT>::value>>
HostBuffer AsHostBuffer(
    absl::Span<const SrcT> values, xla::ifrt::Shape shape,
    Allocator* allocator = DefaultAllocator::GetInstance()) {
  CHECK_EQ(values.size(), shape.num_elements())
      << "Values size " << values.size() << " does not match the shape "
      << shape;
  HostBuffer host_buffer(xla::ifrt::DType(NativeAsDTypeKind<T>::value),
                         std::move(shape), allocator);
  auto flat = host_buffer.flat<T>();
  for (int i = 0; i < values.size(); ++i) {
    flat(i) = static_cast<T>(values[i]);
  }
  return host_buffer;
}

// Builds a scalar `HostBuffer` from a single value with the given allocator.
// The default allocator will be used if not specified.
template <typename T>
HostBuffer AsHostBuffer(
    const T& value, Allocator* allocator = DefaultAllocator::GetInstance()) {
  HostBuffer host_buffer(xla::ifrt::DType(NativeAsDTypeKind<T>::value),
                         xla::ifrt::Shape({}), allocator);
  host_buffer.flat<T>()(0) = value;
  return host_buffer;
}

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_TEST_UTILS_H_
