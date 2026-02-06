#include "third_party/neptune/libs/mlir/utils.h"

#include <cstdint>
#include <optional>

#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/Casting.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Builders.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinAttributes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinTypes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Types.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/DebugStringHelper.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/LLVM.h"
#include "third_party/stablehlo/stablehlo/dialect/StablehloOps.h"
#include "third_party/tensorflow/compiler/xla/mlir/utils/type_util.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

namespace jsv::sharding_internal {

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

}  // namespace jsv::sharding_internal
