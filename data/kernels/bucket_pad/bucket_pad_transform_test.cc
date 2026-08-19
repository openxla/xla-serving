// Copyright 2024 The Neptune Authors.
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

#include "data/kernels/bucket_pad/bucket_pad_transform.h"  // from @neptune

#include <cstdint>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/bucket_pad/bucket_pad_transform.pb.h"  // from @neptune
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/utils/default_value.pb.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace neptune::data::composite::transforms {
namespace {

using ::absl_testing::StatusIs;

TEST(BucketPadTransformTest, Basic2DPadSuccess) {
  // Buckets: [4, 8, 16]. Input shape: [2, 3] -> Expect: [2, 4]
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      BucketPadTransform::Create(/*target_dimensions=*/{4, 8, 16}, /*axis=*/1));

  neptune::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kF32),
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

  neptune::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kS32),
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

  neptune::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kF32),
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

  neptune::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kF32),
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

  neptune::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::kS32),
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
}

}  // namespace
}  // namespace neptune::data::composite::transforms
