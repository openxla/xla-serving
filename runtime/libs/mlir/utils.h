#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_MLIR_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_MLIR_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

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

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_MLIR_UTILS_H_
