// Copyright 2026 Google LLC
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

#ifndef XLA_SERVING_DATA_TEST_UTILS_SHAPE_H_
#define XLA_SERVING_DATA_TEST_UTILS_SHAPE_H_

#include <utility>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/shape.h"  // from @xla_serving

namespace xla_serving::data {
namespace test_utils {

// Returns PartialShape::Create(dims), and crashes if it fails. Tests use it
// to write shapes: PartialShapeOrDie({Dim::Unknown(), Dim::Static(3)}).
inline composite::api::PartialShape PartialShapeOrDie(
    absl::Span<const composite::api::Dim> dims) {
  absl::StatusOr<composite::api::PartialShape> shape =
      composite::api::PartialShape::Create(dims);
  CHECK_OK(shape);
  return *std::move(shape);
}

}  // namespace test_utils
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_TEST_UTILS_SHAPE_H_
