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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_IFRT_TENSOR_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_IFRT_TENSOR_UTILS_H_

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/shape.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow

namespace neptune {

// Converts an IFRT DType to a Tensorflow DataType.
absl::StatusOr<tensorflow::DataType> ToTensorDataType(
    xla::ifrt::DType ifrt_dtype);

// Converts a Tensorflow DataType to an XLA PrimitiveType.
absl::StatusOr<xla::PrimitiveType> ToXlaPrimitiveType(
    tensorflow::DataType tensor_dtype);

// Converts a Tensorflow DataType to an IFRT DType.
absl::StatusOr<xla::ifrt::DType> ToIfrtDType(tensorflow::DataType tensor_dtype);

// Converts a Tensorflow TensorShape to an IFRT Shape.
xla::ifrt::Shape ToIfrtShape(const tensorflow::TensorShape& shape);

// Converts an IFRT Shape to a Tensorflow TensorShape.
tensorflow::TensorShape ToTensorShape(const xla::ifrt::Shape& shape);

// Converts a TensorShape into the equivalent XLA Shape proto.
absl::StatusOr<xla::Shape> TensorShapeToXLAShape(
    tensorflow::DataType dtype, const tensorflow::TensorShape& tensor_shape);

// Converts a MLIR Type to a Tensorflow DataType.
absl::Status MlirTypeToDataType(mlir::Type type, tensorflow::DataType* dtype);

// Converts a MLIR Type to a Tensorflow PartialTensorShape.
tensorflow::PartialTensorShape MlirTypeToTensorShape(const mlir::Type& type);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_IFRT_TENSOR_UTILS_H_
