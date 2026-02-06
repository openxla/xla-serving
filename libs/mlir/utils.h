#ifndef THIRD_PARTY_NEPTUNE_MLIR_UTILS_H_
#define THIRD_PARTY_NEPTUNE_MLIR_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "third_party/absl/container/inlined_vector.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Builders.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Types.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

namespace jsv::sharding_internal {

// Converts an MLIR type to IFRT DType.
absl::StatusOr<xla::ifrt::DType> ToIfrtDType(const mlir::Type& type);

// Represents the shape of an MLIR type. Any dimension that is not set is
// assumed to be dynamic.
using MlirShape = absl::InlinedVector<std::optional<int64_t>, 4>;

// Get the shape of an MLIR type. If there are any dynamic shapes (a nullopt)
// the program will have to be refined before compilation.
absl::StatusOr<MlirShape> ToMlirShape(const mlir::Type& type);

}  // namespace jsv::sharding_internal

#endif  // THIRD_PARTY_NEPTUNE_MLIR_UTILS_H_
