// Copyright 2026 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLA_SERVING_DATA_KERNELS_MACROS_H_
#define XLA_SERVING_DATA_KERNELS_MACROS_H_

#include <complex>
#include <cstdint>

#include "Eigen/Core"  // IWYU pragma: keep by recommendation
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "tsl/platform/ml_dtypes.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

// These macros are shameless copies of those in host_buffer_type_utils.h.
// TODO: Merge them later.
namespace xla_serving::data::kernel {

// This macro invokes the given macro with the corresponding C++ host storage
// type for each supported ifrt::DType::Kind enum.
//
// A typical usage is to define a macro to process the data type:
// switch (dtype) {
// #define CASE_TYPE(T, DTYPE_KIND)               \
//   case DTYPE_KIND: {                           \
//     process<T>(host_buffer.flat<T>());         \
//   }  break;
//
//   ND_APPLY_POD_CASE(CASE_TYPE)
// #undef CASE_TYPE
// default:
//   LOG(FATAL) << "Unsupported dtype: " << dtype;
// }

// Apply a given macro over standard POD data types.
#define ND_APPLY_POD_CASE(CASE_TYPE)                            \
  CASE_TYPE(int8_t, xla::ifrt::DType::Kind::kS8);               \
  CASE_TYPE(uint8_t, xla::ifrt::DType::Kind::kU8);              \
  CASE_TYPE(int16_t, xla::ifrt::DType::Kind::kS16);             \
  CASE_TYPE(uint16_t, xla::ifrt::DType::Kind::kU16);            \
  CASE_TYPE(int32_t, xla::ifrt::DType::Kind::kS32);             \
  CASE_TYPE(uint32_t, xla::ifrt::DType::Kind::kU32);            \
  CASE_TYPE(int64_t, xla::ifrt::DType::Kind::kS64);             \
  CASE_TYPE(uint64_t, xla::ifrt::DType::Kind::kU64);            \
  CASE_TYPE(float, xla::ifrt::DType::Kind::kF32);               \
  CASE_TYPE(double, xla::ifrt::DType::Kind::kF64);              \
  CASE_TYPE(tsl::int4, xla::ifrt::DType::Kind::kS4);            \
  CASE_TYPE(tsl::uint4, xla::ifrt::DType::Kind::kU4);           \
  CASE_TYPE(Eigen::half, xla::ifrt::DType::Kind::kF16);         \
  CASE_TYPE(Eigen::bfloat16, xla::ifrt::DType::Kind::kBF16);    \
  CASE_TYPE(bool, xla::ifrt::DType::Kind::kPred);               \
  CASE_TYPE(tsl::tstring, xla::ifrt::DType::Kind::kString);     \
  CASE_TYPE(std::complex<float>, xla::ifrt::DType::Kind::kC64); \
  CASE_TYPE(std::complex<double>, xla::ifrt::DType::Kind::kC128);

// Apply a given macro over all floating point and integer numeric types.
#define ND_APPLY_INTEGRAL_CASE(CASE_TYPE)            \
  CASE_TYPE(int8_t, xla::ifrt::DType::Kind::kS8);    \
  CASE_TYPE(uint8_t, xla::ifrt::DType::Kind::kU8);   \
  CASE_TYPE(int16_t, xla::ifrt::DType::Kind::kS16);  \
  CASE_TYPE(uint16_t, xla::ifrt::DType::Kind::kU16); \
  CASE_TYPE(int32_t, xla::ifrt::DType::Kind::kS32);  \
  CASE_TYPE(uint32_t, xla::ifrt::DType::Kind::kU32); \
  CASE_TYPE(int64_t, xla::ifrt::DType::Kind::kS64);  \
  CASE_TYPE(uint64_t, xla::ifrt::DType::Kind::kU64);

#define ND_APPLY_FLOAT_CASE(CASE_TYPE)                  \
  CASE_TYPE(float, xla::ifrt::DType::Kind::kF32);       \
  CASE_TYPE(double, xla::ifrt::DType::Kind::kF64);      \
  CASE_TYPE(Eigen::half, xla::ifrt::DType::Kind::kF16); \
  CASE_TYPE(Eigen::bfloat16, xla::ifrt::DType::Kind::kBF16);

#define ND_APPLY_COMPLEX_CASE(CASE_TYPE)                        \
  CASE_TYPE(std::complex<float>, xla::ifrt::DType::Kind::kC64); \
  CASE_TYPE(std::complex<double>, xla::ifrt::DType::Kind::kC128);

#define ND_APPLY_NUMBER_CASE(CASE_TYPE) \
  ND_APPLY_INTEGRAL_CASE(CASE_TYPE)     \
  ND_APPLY_FLOAT_CASE(CASE_TYPE)

#define ND_APPLY_ALL_NUMBER_CASE(CASE_TYPE) \
  ND_APPLY_NUMBER_CASE(CASE_TYPE)           \
  ND_APPLY_COMPLEX_CASE(CASE_TYPE)

// Apply a given macro over GPU-supported floating point types.
#define ND_APPLY_GPU_NUMBER_CASE(CASE_TYPE)             \
  CASE_TYPE(float, xla::ifrt::DType::Kind::kF32);       \
  CASE_TYPE(double, xla::ifrt::DType::Kind::kF64);      \
  CASE_TYPE(Eigen::half, xla::ifrt::DType::Kind::kF16); \
  CASE_TYPE(Eigen::bfloat16, xla::ifrt::DType::Kind::kBF16);

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

// Specialization for unsupported types. Adding a new type requires adding a
// specialization of NativeAsDTypeKind.
template <typename T>
struct NativeAsDTypeKind {
  static_assert(false, "Unsupported type");
};

// Specialization for supported numerical types.
ND_APPLY_POD_CASE(DEFINE_NATIVE_AS_DTYPE_KIND);

#undef DEFINE_NATIVE_AS_DTYPE_KIND
}  // namespace xla_serving::data::kernel

#endif  // XLA_SERVING_DATA_KERNELS_MACROS_H_
