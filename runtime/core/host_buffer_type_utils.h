#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_TYPE_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_TYPE_UTILS_H_

#include <cstdint>

#include "xla/python/ifrt/dtype.h"  // from @xla
#include "tsl/platform/bfloat16.h"  // from @tsl
#include "tsl/platform/ml_dtypes.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

namespace neptune {

// TODO: Fancy data types such as f8exxx are not supported yet.
// Add them when needed. String and token are not supported.
//
// This macor invokes the given macro with the corresponding C++ host storage
// type for each supported ifrt::DType::Kind enum.
//
// A typical usage is to define a macro to process the data type:

// #define CASE_TYPE(T, DTYPE_KIND)                  \
//  case DTYPE_KIND: {                              \
//    process<T>(host_buffer.flat<T>());                \
//  }  break;                                        \
//
//   APPLY_POD_CASE(CASE_TYPE)
// #undef CASE_TYPE
// default:
//   LOG(FATAL) << "Unsupported dtype: " << dtype;
// }

#define APPLY_POD_CASE(CASE_TYPE)                     \
  CASE_TYPE(int8_t, xla::ifrt::DType::Kind::kS8);     \
  CASE_TYPE(uint8_t, xla::ifrt::DType::Kind::kU8);    \
  CASE_TYPE(int16_t, xla::ifrt::DType::Kind::kS16);   \
  CASE_TYPE(uint16_t, xla::ifrt::DType::Kind::kU16);  \
  CASE_TYPE(int32_t, xla::ifrt::DType::Kind::kS32);   \
  CASE_TYPE(uint32_t, xla::ifrt::DType::Kind::kU32);  \
  CASE_TYPE(int64_t, xla::ifrt::DType::Kind::kS64);  \
  CASE_TYPE(uint64_t, xla::ifrt::DType::Kind::kU64);  \
  CASE_TYPE(float, xla::ifrt::DType::Kind::kF32);     \
  CASE_TYPE(double, xla::ifrt::DType::Kind::kF64);    \
  CASE_TYPE(tsl::int4, xla::ifrt::DType::Kind::kS4);  \
  CASE_TYPE(tsl::uint4, xla::ifrt::DType::Kind::kU4); \
  CASE_TYPE(tsl::bfloat16, xla::ifrt::DType::Kind::kBF16); \
  CASE_TYPE(bool, xla::ifrt::DType::Kind::kPred);     \
  CASE_TYPE(tsl::tstring, xla::ifrt::DType::Kind::kString);

// Forward declaration.
// NativeAsDTypeKind<T>::value is the corresponding xla::ifrt::DType::Kind of T.
// It will fail compilation if T is not supported.
// Example:
//   xla::ifrt::DType::Kind kind = NativeAsDTypeKind<int32_t>::value;
//  will assign kS32 to kind.
//   xla::ifrt::DType::Kind kind = NativeAsDTypeKind<custom_type>::value;
//  will fail compilation.
template <typename T>
struct NativeAsDTypeKind;

// A macro to define NativeAsDTypeKind<T> for a given type.
#define DEFINE_NATIVE_AS_DTYPE_KIND(T, KIND)              \
  template <>                                             \
  struct NativeAsDTypeKind<T> {                           \
    static constexpr xla::ifrt::DType::Kind value = KIND; \
  }

// Specialization for supported numerical types.
APPLY_POD_CASE(DEFINE_NATIVE_AS_DTYPE_KIND);

// Specialization for unsupported types. Adding a new type requires adding a
// specialization of NativeAsDTypeKind.
template <typename T>
struct NativeAsDTypeKind {
  static_assert(false, "Unsupported type");
};

#undef DEFINE_NATIVE_AS_DTYPE_KIND

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_TYPE_UTILS_H_
