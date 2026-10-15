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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow

namespace xla_serving {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(IfrtTensorUtilsTest, ToTensorDataTypeF4AndF8E8M0) {
  EXPECT_THAT(ToTensorDataType(xla::ifrt::DType(xla::ifrt::DType::kF4E2M1FN)),
              IsOkAndHolds(tensorflow::DT_FLOAT4_E2M1FN));
  EXPECT_THAT(ToTensorDataType(xla::ifrt::DType(xla::ifrt::DType::kF8E8M0FNU)),
              IsOkAndHolds(tensorflow::DT_UINT8));
}

TEST(IfrtTensorUtilsTest, ToXlaPrimitiveTypeF4) {
  EXPECT_THAT(ToXlaPrimitiveType(tensorflow::DT_FLOAT4_E2M1FN),
              IsOkAndHolds(xla::F4E2M1FN));
}

TEST(IfrtTensorUtilsTest, ToIfrtDTypeF4) {
  EXPECT_THAT(
      ToIfrtDType(tensorflow::DT_FLOAT4_E2M1FN),
      IsOkAndHolds(xla::ifrt::DType(xla::ifrt::DType::kF4E2M1FN)));
}

TEST(IfrtTensorUtilsTest, MlirTypeToDataTypeF4) {
  mlir::MLIRContext context;
  tensorflow::DataType dtype;
  EXPECT_TRUE(
      MlirTypeToDataType(mlir::Float4E2M1FNType::get(&context), &dtype).ok());
  EXPECT_EQ(dtype, tensorflow::DT_FLOAT4_E2M1FN);
}

}  // namespace
}  // namespace xla_serving
