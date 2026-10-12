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

#include "data/kernels/util/shape_utils.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <string>

#include "data/kernels/bounds_check.h"  // from @xla_serving
#include "data/kernels/util/overflow.h"  // from @xla_serving
#include "absl/base/optimization.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::kernel {

using ::xla::ifrt::Shape;

bool ShapeUtils::StartsWith(const Shape& shape, const Shape& prefix) {
  if (shape.dims().size() < prefix.dims().size()) return false;
  for (size_t i = 0; i < prefix.dims().size(); ++i) {
    if (shape.dims()[i] != prefix.dims()[i]) return false;
  }
  return true;
}

bool ShapeUtils::EndsWith(const Shape& shape, const Shape& suffix) {
  const size_t suffix_size = suffix.dims().size();
  if (shape.dims().size() < suffix_size) return false;
  for (size_t i = 0; i < suffix_size; ++i) {
    if (shape.dims()[shape.dims().size() - suffix_size + i] !=
        suffix.dims()[i]) {
      return false;
    }
  }
  return true;
}

namespace {

template <typename T>
absl::StatusOr<Shape> MakeShapeHelper(const T* dims, int64_t n) {
  // JAX Data does not support dynamic shapes unlike in TensorFlow (i.e.,
  // unknown rank or unknown (negative) dimensions).
  if (n < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Negative number of dimensions ", n));
  }
  absl::InlinedVector<int64_t, 6> dim_vec;
  dim_vec.reserve(n);
  int64_t num_elements = 1;
  for (int64_t i = 0; i < n; ++i) {
    T dim = internal::SubtleMustCopy(dims[i]);
    if (dim < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Dimension sizes must be non-negative, got ", dim));
    }
    dim_vec.push_back(dim);
    num_elements = MultiplyWithoutOverflow(num_elements, dim);
    if (ABSL_PREDICT_FALSE(num_elements < 0)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Shape [", absl::StrJoin(absl::MakeConstSpan(dims, n), ","),
          "] would have more than 2**63 - 1 elements"));
    }
  }
  return Shape(dim_vec);
}

}  // namespace

absl::StatusOr<Shape> ShapeUtils::MakeShape(const int32_t* dims, int64_t n) {
  return MakeShapeHelper(dims, n);
}

absl::StatusOr<Shape> ShapeUtils::MakeShape(const int64_t* dims, int64_t n) {
  return MakeShapeHelper(dims, n);
}

absl::StatusOr<Shape> ShapeUtils::MakeShape(absl::Span<const int32_t> shape) {
  return MakeShapeHelper(shape.data(), shape.size());
}

absl::StatusOr<Shape> ShapeUtils::MakeShape(absl::Span<const int64_t> shape) {
  return MakeShapeHelper(shape.data(), shape.size());
}

std::string ShapeUtils::ShapeListString(absl::Span<const Shape> shapes) {
  std::string result = "[";
  bool first = true;
  for (const auto& shape : shapes) {
    absl::StrAppend(&result, first ? "" : ", ", shape);
    first = false;
  }
  absl::StrAppend(&result, "]");
  return result;
}

absl::StatusOr<int64_t> ShapeUtils::NumElements(
    absl::Span<const int64_t> shape) {
  for (auto dim : shape) {
    if (dim < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid dimension size ", dim, " in shape [",
                       absl::StrJoin(shape, ","), "]"));
    }
  }
  for (auto dim : shape) {
    if (dim == 0) {
      return 0;
    }
  }
  int64_t n = 1;
  for (auto dim : shape) {
    n = MultiplyWithoutOverflow(n, dim);
    if (n < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Can't compute total size of shape [", absl::StrJoin(shape, ","),
          "]; product would overflow int64"));
    }
  }
  return n;
}

}  // namespace xla_serving::data::kernel
