// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "data/kernels/bucket_pad/bucket_pad_transform.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/bucket_pad/bucket_pad_transform.pb.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/utils/default_value.pb.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::composite::transforms {
namespace {

using ::absl_testing::StatusIs;

TEST(BucketPadTransformTest, Basic2DPadSuccess) {
  // Buckets: [4, 8, 16]. Input shape: [2, 3] -> Expect: [2, 4]
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      BucketPadTransform::Create(/*target_dimensions=*/{4, 8, 16}, /*axis=*/1));

  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kF32),
                        xla::ifrt::Shape({2, 3}));
  float* in_ptr = static_cast<float*>(input.base());
  in_ptr[0] = 1.0f;
  in_ptr[1] = 2.0f;
  in_ptr[2] = 3.0f;
  in_ptr[3] = 4.0f;
  in_ptr[4] = 5.0f;
  in_ptr[5] = 6.0f;

  ASSERT_OK_AND_ASSIGN(auto output, (*transform)(input));

  EXPECT_EQ(output.shape().dims(), (std::vector<int64_t>{2, 4}));
  const float* out_ptr = static_cast<const float*>(output.base());

  // Row 0
  EXPECT_FLOAT_EQ(out_ptr[0], 1.0f);
  EXPECT_FLOAT_EQ(out_ptr[1], 2.0f);
  EXPECT_FLOAT_EQ(out_ptr[2], 3.0f);
  EXPECT_FLOAT_EQ(out_ptr[3], 0.0f);  // Padded

  // Row 1
  EXPECT_FLOAT_EQ(out_ptr[4], 4.0f);
  EXPECT_FLOAT_EQ(out_ptr[5], 5.0f);
  EXPECT_FLOAT_EQ(out_ptr[6], 6.0f);
  EXPECT_FLOAT_EQ(out_ptr[7], 0.0f);  // Padded
}

TEST(BucketPadTransformTest, ExactBucketHitNoExtraPadding) {
  // Input shape: [2, 4] matching bucket 4 exactly
  ASSERT_OK_AND_ASSIGN(auto transform,
                       BucketPadTransform::Create(
                           /*target_dimensions=*/{4, 8, 16}, /*axis=*/-1));

  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kS32),
                        xla::ifrt::Shape({2, 4}));
  int32_t* in_ptr = static_cast<int32_t*>(input.base());
  for (int i = 0; i < 8; ++i) {
    in_ptr[i] = i + 10;
  }

  ASSERT_OK_AND_ASSIGN(auto output, (*transform)(input));

  EXPECT_EQ(output.shape().dims(), (std::vector<int64_t>{2, 4}));
  const int32_t* out_ptr = static_cast<const int32_t*>(output.base());
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(out_ptr[i], i + 10);
  }
}

TEST(BucketPadTransformTest, CustomPadValue) {
  utils::DefaultValue pad_val_proto;
  pad_val_proto.set_float_value(-1.0f);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       BucketPadTransform::Create(/*target_dimensions=*/{5},
                                                  /*axis=*/1, pad_val_proto));

  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kF32),
                        xla::ifrt::Shape({1, 2}));
  float* in_ptr = static_cast<float*>(input.base());
  in_ptr[0] = 7.0f;
  in_ptr[1] = 8.0f;

  ASSERT_OK_AND_ASSIGN(auto output, (*transform)(input));

  EXPECT_EQ(output.shape().dims(), (std::vector<int64_t>{1, 5}));
  const float* out_ptr = static_cast<const float*>(output.base());
  EXPECT_FLOAT_EQ(out_ptr[0], 7.0f);
  EXPECT_FLOAT_EQ(out_ptr[1], 8.0f);
  EXPECT_FLOAT_EQ(out_ptr[2], -1.0f);
  EXPECT_FLOAT_EQ(out_ptr[3], -1.0f);
  EXPECT_FLOAT_EQ(out_ptr[4], -1.0f);
}

TEST(BucketPadTransformTest, ExceedsMaxBucketFails) {
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      BucketPadTransform::Create(/*target_dimensions=*/{4, 8}, /*axis=*/1));

  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kF32),
                        xla::ifrt::Shape({1, 10}));

  EXPECT_THAT((*transform)(input),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BucketPadTransformTest, HigherRank3DTensor) {
  // Input: [2, 3, 2] padded along axis=1 with buckets [4, 6] -> Output: [2, 4,
  // 2]
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      BucketPadTransform::Create(/*target_dimensions=*/{4, 6}, /*axis=*/1));

  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kS32),
                        xla::ifrt::Shape({2, 3, 2}));
  int32_t* in_ptr = static_cast<int32_t*>(input.base());
  for (int i = 0; i < 12; ++i) {
    in_ptr[i] = i + 1;
  }

  ASSERT_OK_AND_ASSIGN(auto output, (*transform)(input));

  EXPECT_EQ(output.shape().dims(), (std::vector<int64_t>{2, 4, 2}));
  const int32_t* out_ptr = static_cast<const int32_t*>(output.base());

  // Batch 0: rows 0, 1, 2 copied; row 3 is zero-padded
  EXPECT_EQ(out_ptr[0], 1);
  EXPECT_EQ(out_ptr[1], 2);
  EXPECT_EQ(out_ptr[2], 3);
  EXPECT_EQ(out_ptr[3], 4);
  EXPECT_EQ(out_ptr[4], 5);
  EXPECT_EQ(out_ptr[5], 6);
  EXPECT_EQ(out_ptr[6], 0);
  EXPECT_EQ(out_ptr[7], 0);  // Padded row

  // Batch 1: rows 0, 1, 2 copied; row 3 is zero-padded
  EXPECT_EQ(out_ptr[8], 7);
  EXPECT_EQ(out_ptr[9], 8);
  EXPECT_EQ(out_ptr[10], 9);
  EXPECT_EQ(out_ptr[11], 10);
  EXPECT_EQ(out_ptr[12], 11);
  EXPECT_EQ(out_ptr[13], 12);
  EXPECT_EQ(out_ptr[14], 0);
  EXPECT_EQ(out_ptr[15], 0);  // Padded row
}

TEST(BucketPadTransformTest, SerializationAndDeserialization) {
  ASSERT_OK_AND_ASSIGN(
      auto orig, BucketPadTransform::Create(/*target_dimensions=*/{16, 32, 64},
                                            /*axis=*/1));

  ASSERT_OK_AND_ASSIGN(auto proto_config, orig->ToProto());
  EXPECT_EQ(proto_config.op_name(), "BucketPadTransform");

  ASSERT_OK_AND_ASSIGN(
      auto deserialized_base,
      TransformBase::FromProto(proto_config, /*resources=*/{}));
  auto* deserialized =
      dynamic_cast<BucketPadTransform*>(deserialized_base.get());
  ASSERT_NE(deserialized, nullptr);

  EXPECT_EQ(deserialized->target_dimensions(),
            (std::vector<int64_t>{16, 32, 64}));
  EXPECT_EQ(deserialized->axis(), 1);
}

TEST(BucketPadTransformTest, ShapeInference) {
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      BucketPadTransform::Create(/*target_dimensions=*/{16, 32}, /*axis=*/1));

  api::ShapeInferenceContext context;
  api::HostBufferSpec in_spec;
  in_spec.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32);
  in_spec.shape = xla::ifrt::Shape({64, 10});
  std::vector<api::ShapeInferenceValue> inputs = {
      api::ShapeInferenceValue(in_spec)};
  context.inputs = absl::MakeConstSpan(inputs);

  ASSERT_OK(transform->PropagateSignature(context));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsSpec());
  EXPECT_EQ(context.outputs[0].spec().shape.dims(),
            (std::vector<int64_t>{64, 16}));
}

TEST(BucketPadTransformTest, RejectInvalidDimensions) {
  EXPECT_THAT(BucketPadTransform::Create(/*target_dimensions=*/{}, /*axis=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      BucketPadTransform::Create(/*target_dimensions=*/{-1, 16}, /*axis=*/0),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      BucketPadTransform::Create(/*target_dimensions=*/{0, 16}, /*axis=*/0),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BucketPadTransformTest, RejectScalarTensor) {
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      BucketPadTransform::Create(/*target_dimensions=*/{4}, /*axis=*/0));

  xla_serving::HostBuffer scalar_input(xla::ifrt::DType(xla::ifrt::DType::kF32),
                               xla::ifrt::Shape({}));
  EXPECT_THAT((*transform)(scalar_input),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BucketPadTransformTest, RejectOutOfBoundsAxis) {
  ASSERT_OK_AND_ASSIGN(
      auto transform_pos,
      BucketPadTransform::Create(/*target_dimensions=*/{4}, /*axis=*/3));
  ASSERT_OK_AND_ASSIGN(
      auto transform_neg,
      BucketPadTransform::Create(/*target_dimensions=*/{4}, /*axis=*/-4));

  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kF32),
                        xla::ifrt::Shape({2, 3}));
  EXPECT_THAT((*transform_pos)(input),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT((*transform_neg)(input),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BucketPadTransformTest, ExecuteContextSuccessAndErrorHandling) {
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      BucketPadTransform::Create(/*target_dimensions=*/{4}, /*axis=*/1));

  // Invalid: Empty inputs
  api::ExecuteContext empty_context;
  EXPECT_THAT(transform->Execute(empty_context),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Invalid: Multiple inputs
  xla_serving::HostBuffer input1(xla::ifrt::DType(xla::ifrt::DType::kF32),
                         xla::ifrt::Shape({1, 2}));
  xla_serving::HostBuffer input2(xla::ifrt::DType(xla::ifrt::DType::kF32),
                         xla::ifrt::Shape({1, 2}));
  std::vector<api::Value> multi_inputs;
  multi_inputs.push_back(api::Value(std::move(input1)));
  multi_inputs.push_back(api::Value(std::move(input2)));
  api::ExecuteContext multi_context;
  multi_context.inputs = absl::MakeConstSpan(multi_inputs);
  EXPECT_THAT(transform->Execute(multi_context),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Valid: Exactly 1 input
  xla_serving::HostBuffer valid_input(xla::ifrt::DType(xla::ifrt::DType::kF32),
                              xla::ifrt::Shape({1, 2}));
  float* valid_ptr = static_cast<float*>(valid_input.base());
  valid_ptr[0] = 5.0f;
  valid_ptr[1] = 6.0f;

  std::vector<api::Value> valid_inputs;
  valid_inputs.push_back(api::Value(std::move(valid_input)));
  api::ExecuteContext valid_context;
  valid_context.inputs = absl::MakeConstSpan(valid_inputs);
  ASSERT_OK(transform->Execute(valid_context));
  ASSERT_EQ(valid_context.outputs.size(), 1);
  const auto& out_buf = valid_context.outputs[0].Get<xla_serving::HostBuffer>();
  EXPECT_EQ(out_buf.shape().dims(), (std::vector<int64_t>{1, 4}));
}

TEST(BucketPadTransformTest, ConfigUnpackError) {
  ::xla_serving::data::composite::TransformConfig invalid_config;
  invalid_config.set_op_name("BucketPadTransform");
  EXPECT_THAT(BucketPadTransform::CreateFromTransformConfig(invalid_config,
                                                            /*resources=*/{}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BucketPadTransformTest, Float64Support) {
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      BucketPadTransform::Create(/*target_dimensions=*/{4}, /*axis=*/1));

  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kF64),
                        xla::ifrt::Shape({1, 2}));
  double* in_ptr = static_cast<double*>(input.base());
  in_ptr[0] = 3.14159;
  in_ptr[1] = 2.71828;

  ASSERT_OK_AND_ASSIGN(auto output, (*transform)(input));
  EXPECT_EQ(output.shape().dims(), (std::vector<int64_t>{1, 4}));
  const double* out_ptr = static_cast<const double*>(output.base());
  EXPECT_DOUBLE_EQ(out_ptr[0], 3.14159);
  EXPECT_DOUBLE_EQ(out_ptr[1], 2.71828);
  EXPECT_DOUBLE_EQ(out_ptr[2], 0.0);
  EXPECT_DOUBLE_EQ(out_ptr[3], 0.0);
}

TEST(BucketPadTransformTest, Int64Support) {
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      BucketPadTransform::Create(/*target_dimensions=*/{4}, /*axis=*/1));

  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kS64),
                        xla::ifrt::Shape({1, 2}));
  int64_t* in_ptr = static_cast<int64_t*>(input.base());
  in_ptr[0] = 100000000000LL;
  in_ptr[1] = 200000000000LL;

  ASSERT_OK_AND_ASSIGN(auto output, (*transform)(input));
  EXPECT_EQ(output.shape().dims(), (std::vector<int64_t>{1, 4}));
  const int64_t* out_ptr = static_cast<const int64_t*>(output.base());
  EXPECT_EQ(out_ptr[0], 100000000000LL);
  EXPECT_EQ(out_ptr[1], 200000000000LL);
  EXPECT_EQ(out_ptr[2], 0LL);
  EXPECT_EQ(out_ptr[3], 0LL);
}

TEST(BucketPadTransformTest, UnsignedIntTypesSupport) {
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      BucketPadTransform::Create(/*target_dimensions=*/{3}, /*axis=*/1));

  // Uint32
  xla_serving::HostBuffer input_u32(xla::ifrt::DType(xla::ifrt::DType::kU32),
                            xla::ifrt::Shape({1, 2}));
  uint32_t* u32_ptr = static_cast<uint32_t*>(input_u32.base());
  u32_ptr[0] = 42;
  u32_ptr[1] = 99;
  ASSERT_OK_AND_ASSIGN(auto out_u32, (*transform)(input_u32));
  const uint32_t* out_u32_ptr = static_cast<const uint32_t*>(out_u32.base());
  EXPECT_EQ(out_u32_ptr[0], 42);
  EXPECT_EQ(out_u32_ptr[1], 99);
  EXPECT_EQ(out_u32_ptr[2], 0);

  // Uint64
  xla_serving::HostBuffer input_u64(xla::ifrt::DType(xla::ifrt::DType::kU64),
                            xla::ifrt::Shape({1, 2}));
  uint64_t* u64_ptr = static_cast<uint64_t*>(input_u64.base());
  u64_ptr[0] = 123456789012345ULL;
  u64_ptr[1] = 987654321098765ULL;
  ASSERT_OK_AND_ASSIGN(auto out_u64, (*transform)(input_u64));
  const uint64_t* out_u64_ptr = static_cast<const uint64_t*>(out_u64.base());
  EXPECT_EQ(out_u64_ptr[0], 123456789012345ULL);
  EXPECT_EQ(out_u64_ptr[1], 987654321098765ULL);
  EXPECT_EQ(out_u64_ptr[2], 0ULL);
}

TEST(BucketPadTransformTest, Int8AndUint8Support) {
  utils::DefaultValue pad_val;
  pad_val.set_int_value(7);
  ASSERT_OK_AND_ASSIGN(auto transform,
                       BucketPadTransform::Create(/*target_dimensions=*/{3},
                                                  /*axis=*/1, pad_val));

  // Int8
  xla_serving::HostBuffer input_s8(xla::ifrt::DType(xla::ifrt::DType::kS8),
                           xla::ifrt::Shape({1, 1}));
  int8_t* s8_ptr = static_cast<int8_t*>(input_s8.base());
  s8_ptr[0] = -12;
  ASSERT_OK_AND_ASSIGN(auto out_s8, (*transform)(input_s8));
  const int8_t* out_s8_ptr = static_cast<const int8_t*>(out_s8.base());
  EXPECT_EQ(out_s8_ptr[0], -12);
  EXPECT_EQ(out_s8_ptr[1], 7);
  EXPECT_EQ(out_s8_ptr[2], 7);

  // Uint8
  xla_serving::HostBuffer input_u8(xla::ifrt::DType(xla::ifrt::DType::kU8),
                           xla::ifrt::Shape({1, 1}));
  uint8_t* u8_ptr = static_cast<uint8_t*>(input_u8.base());
  u8_ptr[0] = 200;
  ASSERT_OK_AND_ASSIGN(auto out_u8, (*transform)(input_u8));
  const uint8_t* out_u8_ptr = static_cast<const uint8_t*>(out_u8.base());
  EXPECT_EQ(out_u8_ptr[0], 200);
  EXPECT_EQ(out_u8_ptr[1], 7);
  EXPECT_EQ(out_u8_ptr[2], 7);
}

TEST(BucketPadTransformTest, BoolPredSupport) {
  utils::DefaultValue pad_val;
  pad_val.set_bool_value(true);
  ASSERT_OK_AND_ASSIGN(auto transform,
                       BucketPadTransform::Create(/*target_dimensions=*/{3},
                                                  /*axis=*/1, pad_val));

  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kPred),
                        xla::ifrt::Shape({1, 1}));
  bool* in_ptr = static_cast<bool*>(input.base());
  in_ptr[0] = false;

  ASSERT_OK_AND_ASSIGN(auto output, (*transform)(input));
  const bool* out_ptr = static_cast<const bool*>(output.base());
  EXPECT_EQ(out_ptr[0], false);
  EXPECT_EQ(out_ptr[1], true);
  EXPECT_EQ(out_ptr[2], true);
}

}  // namespace
}  // namespace xla_serving::data::composite::transforms
