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

#include "runtime/libs/mlir/utils.h"  // from @neptune

#include <cstdint>
#include <optional>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "mlir/include/mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/mlir/utils/type_util.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving::sharding_internal {

absl::StatusOr<xla::ifrt::DType> ToIfrtDType(const mlir::Type& type) {
  xla::PrimitiveType primitive_type;
  if (llvm::isa<mlir::stablehlo::TokenType>(type)) {
    primitive_type = xla::PrimitiveType::TOKEN;
  } else if (llvm::isa<mlir::TupleType>(type)) {
    primitive_type = xla::PrimitiveType::TUPLE;
  } else if (auto stype = mlir::dyn_cast<mlir::ShapedType>(type)) {
    primitive_type =
        xla::ConvertMlirTypeToPrimitiveType(stype.getElementType());
  } else {
    primitive_type = xla::ConvertMlirTypeToPrimitiveType(type);
  }
  return xla::ifrt::ToDType(primitive_type);
}

absl::StatusOr<MlirShape> ToMlirShape(const mlir::Type& type) {
  if (mlir::isa<mlir::UnrankedTensorType>(type)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported type ", mlir::debugString(type)));
  }

  if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    llvm::ArrayRef<int64_t> shape = tensor_type.getShape();
    MlirShape shape_vec;
    shape_vec.reserve(shape.size());
    for (const auto& dim : shape) {
      if (mlir::ShapedType::isDynamic(dim)) {
        shape_vec.push_back(std::nullopt);
      } else {
        shape_vec.push_back(dim);
      }
    }
    return shape_vec;
  }

  // If type is not a RankedTensor or UnrankedTensor, it must be a scalar.
  return MlirShape{};
}

}  // namespace xla_serving::sharding_internal
