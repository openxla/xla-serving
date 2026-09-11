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

#ifndef XLA_SERVING_RUNTIME_LIBS_MLIR_UTILS_H_
#define XLA_SERVING_RUNTIME_LIBS_MLIR_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving::sharding_internal {

// Converts an MLIR type to IFRT DType.
absl::StatusOr<xla::ifrt::DType> ToIfrtDType(const mlir::Type& type);

// Represents the shape of an MLIR type. Any dimension that is not set is
// assumed to be dynamic.
using MlirShape = absl::InlinedVector<std::optional<int64_t>, 4>;

// Get the shape of an MLIR type. If there are any dynamic shapes (a nullopt)
// the program will have to be refined before compilation.
absl::StatusOr<MlirShape> ToMlirShape(const mlir::Type& type);

}  // namespace xla_serving::sharding_internal

#endif  // XLA_SERVING_RUNTIME_LIBS_MLIR_UTILS_H_
