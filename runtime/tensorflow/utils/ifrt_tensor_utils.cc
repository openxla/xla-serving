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

#include "runtime/tensorflow/utils/ifrt_tensor_utils.h"  // from @xla_serving

#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "mlir/include/mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/shape.h"  // from @xla
#include "xla/shape_util.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow

namespace xla_serving {
namespace {
// TODO: delete all these after tensor usage is removed.
absl::Status DataTypeToPrimitiveType(tensorflow::DataType data_type,
                                     xla::PrimitiveType* type) {
  switch (data_type) {
    case tensorflow::DT_BOOL:
      *type = xla::PRED;
      return absl::OkStatus();
    case tensorflow::DT_INT4:
      *type = xla::S4;
      return absl::OkStatus();
    case tensorflow::DT_INT8:
    case tensorflow::DT_QINT8:
      *type = xla::S8;
      return absl::OkStatus();
    case tensorflow::DT_INT16:
    case tensorflow::DT_QINT16:
      *type = xla::S16;
      return absl::OkStatus();
    case tensorflow::DT_INT32:
    case tensorflow::DT_QINT32:
      *type = xla::S32;
      return absl::OkStatus();
    case tensorflow::DT_INT64:
      *type = xla::S64;
      return absl::OkStatus();
    case tensorflow::DT_UINT4:
      *type = xla::U4;
      return absl::OkStatus();
    case tensorflow::DT_UINT8:
    case tensorflow::DT_QUINT8:
      *type = xla::U8;
      return absl::OkStatus();
    case tensorflow::DT_UINT16:
    case tensorflow::DT_QUINT16:
      *type = xla::U16;
      return absl::OkStatus();
    case tensorflow::DT_UINT32:
      *type = xla::U32;
      return absl::OkStatus();
    case tensorflow::DT_UINT64:
      *type = xla::U64;
      return absl::OkStatus();
    case tensorflow::DT_FLOAT8_E5M2:
      *type = xla::F8E5M2;
      return absl::OkStatus();
    case tensorflow::DT_FLOAT8_E4M3FN:
      *type = xla::F8E4M3FN;
      return absl::OkStatus();
    case tensorflow::DT_FLOAT4_E2M1FN:
      *type = xla::F4E2M1FN;
      return absl::OkStatus();
    case tensorflow::DT_BFLOAT16:
      *type = xla::BF16;
      return absl::OkStatus();
    case tensorflow::DT_HALF:
      *type = xla::F16;
      return absl::OkStatus();
    case tensorflow::DT_FLOAT:
      *type = xla::F32;
      return absl::OkStatus();
    case tensorflow::DT_DOUBLE:
      *type = xla::F64;
      return absl::OkStatus();
    case tensorflow::DT_COMPLEX64:
      *type = xla::C64;
      return absl::OkStatus();
    case tensorflow::DT_COMPLEX128:
      *type = xla::C128;
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported type in DataTypeToPrimitiveType", data_type));
  }
}

absl::StatusOr<tensorflow::DataType> EncodePrimitiveTypeAsDataType(
    xla::PrimitiveType type) {
  static const absl::flat_hash_map<xla::PrimitiveType, tensorflow::DataType>&
      data_type_map =
          *new absl::flat_hash_map<xla::PrimitiveType, tensorflow::DataType>({
              {xla::PRED, tensorflow::DT_BOOL},
              {xla::F8E5M2, tensorflow::DT_FLOAT8_E5M2},
              {xla::F8E4M3FN, tensorflow::DT_FLOAT8_E4M3FN},
              {xla::F4E2M1FN, tensorflow::DT_FLOAT4_E2M1FN},
              {xla::F8E8M0FNU, tensorflow::DT_UINT8},
              {xla::BF16, tensorflow::DT_BFLOAT16},
              {xla::F16, tensorflow::DT_HALF},
              {xla::F32, tensorflow::DT_FLOAT},
              {xla::F64, tensorflow::DT_DOUBLE},
              {xla::C64, tensorflow::DT_COMPLEX64},
              {xla::S4, tensorflow::DT_INT4},
              {xla::S8, tensorflow::DT_INT8},
              {xla::S16, tensorflow::DT_INT16},
              {xla::S32, tensorflow::DT_INT32},
              {xla::S64, tensorflow::DT_INT64},
              {xla::U4, tensorflow::DT_UINT4},
              {xla::U8, tensorflow::DT_UINT8},
              {xla::U16, tensorflow::DT_UINT16},
              {xla::U32, tensorflow::DT_UINT32},
              {xla::U64, tensorflow::DT_UINT64},
              {xla::C128, tensorflow::DT_COMPLEX128},
          });

  auto it = data_type_map.find(type);
  if (it == data_type_map.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported type in PrimitiveTypeToDataType ", type));
  }
  return it->second;
}

absl::Status ConvertScalarTypeToDataType(mlir::Type type,
                                         tensorflow::DataType* dtype) {
  if (type.isF16()) {
    *dtype = tensorflow::DT_HALF;
    return absl::OkStatus();
  } else if (type.isF32()) {
    *dtype = tensorflow::DT_FLOAT;
    return absl::OkStatus();
  } else if (type.isF64()) {
    *dtype = tensorflow::DT_DOUBLE;
    return absl::OkStatus();
  } else if (type.isBF16()) {
    *dtype = tensorflow::DT_BFLOAT16;
    return absl::OkStatus();
  } else if (mlir::isa<mlir::Float8E4M3FNType>(type)) {
    *dtype = tensorflow::DT_FLOAT8_E4M3FN;
    return absl::OkStatus();
  } else if (mlir::isa<mlir::Float4E2M1FNType>(type)) {
    *dtype = tensorflow::DT_FLOAT4_E2M1FN;
    return absl::OkStatus();
  } else if (mlir::isa<mlir::Float8E5M2Type>(type)) {
    *dtype = tensorflow::DT_FLOAT8_E5M2;
    return absl::OkStatus();
  } else if (auto itype = mlir::dyn_cast<mlir::IntegerType>(type)) {
    switch (itype.getWidth()) {
      case 1:
        *dtype = tensorflow::DT_BOOL;
        return absl::OkStatus();
      case 4:
        *dtype =
            itype.isUnsigned() ? tensorflow::DT_UINT4 : tensorflow::DT_INT4;
        return absl::OkStatus();
      case 8:
        *dtype =
            itype.isUnsigned() ? tensorflow::DT_UINT8 : tensorflow::DT_INT8;
        return absl::OkStatus();
      case 16:
        *dtype =
            itype.isUnsigned() ? tensorflow::DT_UINT16 : tensorflow::DT_INT16;
        return absl::OkStatus();
      case 32:
        *dtype =
            itype.isUnsigned() ? tensorflow::DT_UINT32 : tensorflow::DT_INT32;
        return absl::OkStatus();
      case 64:
        *dtype =
            itype.isUnsigned() ? tensorflow::DT_UINT64 : tensorflow::DT_INT64;
        return absl::OkStatus();
      default:
        return absl::UnimplementedError(absl::StrCat(
            "Converting ", mlir::debugString(type), " to DataType"));
    }
  } else if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type)) {
    auto etype = complex_type.getElementType();
    if (etype.isF32()) {
      *dtype = tensorflow::DT_COMPLEX64;
      return absl::OkStatus();
    } else if (etype.isF64()) {
      *dtype = tensorflow::DT_COMPLEX128;
      return absl::OkStatus();
    }
  }
  return absl::UnimplementedError(
      absl::StrCat("Converting ", mlir::debugString(type), " to DataType"));
}
void ConvertToTensorShapeProto(mlir::ArrayRef<int64_t> shape,
                               tensorflow::TensorShapeProto* output_shape) {
  for (auto d : shape) {
    output_shape->add_dim()->set_size(mlir::ShapedType::isDynamic(d) ? -1 : d);
  }
}
}  // namespace

absl::StatusOr<tensorflow::DataType> ToTensorDataType(
    xla::ifrt::DType ifrt_dtype) {
  if (ifrt_dtype.kind() == xla::ifrt::DType::kString) {
    return tensorflow::DataType::DT_STRING;
  }
  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(ifrt_dtype));
  return EncodePrimitiveTypeAsDataType(primitive_type);
}

absl::StatusOr<xla::ifrt::DType> ToIfrtDType(
    tensorflow::DataType tensor_dtype) {
  if (tensor_dtype == tensorflow::DataType::DT_STRING) {
    return xla::ifrt::DType(xla::ifrt::DType::kString);
  }
  xla::PrimitiveType primitive_type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tensor_dtype, &primitive_type));
  return xla::ifrt::ToDType(primitive_type);
}

absl::StatusOr<xla::PrimitiveType> ToXlaPrimitiveType(
    tensorflow::DataType tensor_dtype) {
  if (tensor_dtype == tensorflow::DataType::DT_STRING) {
    return absl::InvalidArgumentError(
        "DT_STRING is not supported in xla primitive types.");
  }
  xla::PrimitiveType primitive_type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tensor_dtype, &primitive_type));
  return primitive_type;
}

xla::ifrt::Shape ToIfrtShape(const tensorflow::TensorShape& shape) {
  return xla::ifrt::Shape(shape.dim_sizes());
}

tensorflow::TensorShape ToTensorShape(const xla::ifrt::Shape& shape) {
  return tensorflow::TensorShape(shape.dims());
}

xla::Shape TensorShapeToXLAShape(
    xla::PrimitiveType type,
    const tensorflow::PartialTensorShape& tensor_shape) {
  if (tensor_shape.unknown_rank()) {
    // For unknown shape, create a rank 1 size 0 tensor.
    return xla::ShapeUtil::MakeShapeWithDenseLayout(type, {0}, {0});
  }
  int rank = tensor_shape.dims();
  std::vector<int64_t> dimensions(rank);
  std::vector<int64_t> layout(rank);
  for (int d = 0; d < rank; ++d) {
    dimensions[d] = tensor_shape.dim_size(d);
    if (dimensions[d] < 0) {
      LOG(WARNING) << "Unable to convert TF shape with dynamic size to XLA "
                      "shape; returning unknown sentinel value";
      return xla::ShapeUtil::MakeShapeWithDenseLayout(type, {0}, {0});
    }
  }
  // XLA uses minor-to-major; Tensorflow uses major-to-minor.
  std::iota(layout.rbegin(), layout.rend(), 0);
  xla::Shape result =
      xla::ShapeUtil::MakeShapeWithDenseLayout(type, dimensions, layout);
  return result;
}

// Convert a TensorShape into the equivalent XLA Shape proto.
absl::Status TensorShapeToXLAShape(tensorflow::DataType dtype,
                                   const tensorflow::TensorShape& tensor_shape,
                                   xla::Shape* shape) {
  xla::PrimitiveType type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype, &type));
  *shape = TensorShapeToXLAShape(type, tensor_shape);
  return absl::OkStatus();
}

absl::StatusOr<xla::Shape> TensorShapeToXLAShape(
    tensorflow::DataType dtype, const tensorflow::TensorShape& tensor_shape) {
  xla::Shape out;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype, tensor_shape, &out));
  return out;
}

absl::Status MlirTypeToDataType(mlir::Type type, tensorflow::DataType* dtype) {
  if (auto stype = mlir::dyn_cast<mlir::ShapedType>(type)) {
    TF_RETURN_IF_ERROR(
        ConvertScalarTypeToDataType(stype.getElementType(), dtype));
  } else {
    TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, dtype));
  }
  return absl::OkStatus();
}

tensorflow::PartialTensorShape MlirTypeToTensorShape(const mlir::Type& type) {
  if (mlir::isa<mlir::UnrankedTensorType>(type)) {
    // An empty PartialTensorShape indicates an unranked tensor.
    return tensorflow::PartialTensorShape();
  }

  if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    tensorflow::TensorShapeProto tensor_shape_proto;
    ConvertToTensorShapeProto(tensor_type.getShape(), &tensor_shape_proto);
    return tensorflow::PartialTensorShape(tensor_shape_proto);
  }

  // If type is not a RankedTensor or UnrankedTensor, it must be a scalar.
  // Empty TensorShape indicates a scalar.
  return tensorflow::TensorShape();
}

}  // namespace xla_serving
