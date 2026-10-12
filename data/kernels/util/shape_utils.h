/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2026 Google LLC. All Rights Reserved.

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

#ifndef XLA_SERVING_DATA_KERNELS_UTIL_SHAPE_UTILS_H_
#define XLA_SERVING_DATA_KERNELS_UTIL_SHAPE_UTILS_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::kernel {

// Static helper routines for `xla::ifrt::Shape`. Includes common predicates on
// a tensor shape, shape creation, and dimension arithmetic.
class ShapeUtils {
 public:
  static bool IsScalar(const xla::ifrt::Shape& shape) {
    return shape.dims().empty();
  }

  static bool IsVector(const xla::ifrt::Shape& shape) {
    return shape.dims().size() == 1;
  }

  static bool IsVectorOrHigher(const xla::ifrt::Shape& shape) {
    return !shape.dims().empty();
  }

  static bool IsMatrix(const xla::ifrt::Shape& shape) {
    return shape.dims().size() == 2;
  }

  static bool IsSquareMatrix(const xla::ifrt::Shape& shape) {
    return shape.dims().size() == 2 && shape.dims()[0] == shape.dims()[1];
  }

  static bool IsMatrixOrHigher(const xla::ifrt::Shape& shape) {
    return shape.dims().size() >= 2;
  }

  // Returns a `xla::ifrt::Shape` whose dimensions are dims[0], ..., dims[n-1].
  static absl::StatusOr<xla::ifrt::Shape> MakeShape(const int32_t* dims,
                                                    int64_t n);
  static absl::StatusOr<xla::ifrt::Shape> MakeShape(const int64_t* dims,
                                                    int64_t n);
  static absl::StatusOr<xla::ifrt::Shape> MakeShape(
      absl::Span<const int32_t> shape);
  static absl::StatusOr<xla::ifrt::Shape> MakeShape(
      absl::Span<const int64_t> shape);

  // Returns a string representation of a list of shapes, e.g. "[[1, 2], [3]]".
  static std::string ShapeListString(absl::Span<const xla::ifrt::Shape> shapes);

  // Returns true iff `shape` starts with `prefix`.
  static bool StartsWith(const xla::ifrt::Shape& shape,
                         const xla::ifrt::Shape& prefix);

  // Returns true iff `shape` ends with `suffix`.
  static bool EndsWith(const xla::ifrt::Shape& shape,
                       const xla::ifrt::Shape& suffix);

  // Returns the product of values in an int64 array, or a failing Status if the
  // array represents an invalid size or overflow.
  static absl::StatusOr<int64_t> NumElements(absl::Span<const int64_t> shape);
};

}  // namespace xla_serving::data::kernel

#endif  // XLA_SERVING_DATA_KERNELS_UTIL_SHAPE_UTILS_H_
