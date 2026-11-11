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

#include "runtime/kernel/batching/pad_utils.h"  // from @xla_serving

#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

template <typename T>
class PadUtilsTypedTest : public ::testing::Test {};

using InputTypes = ::testing::Types<float, double, int32_t, int64_t, uint8_t>;
TYPED_TEST_SUITE(PadUtilsTypedTest, InputTypes);

TYPED_TEST(PadUtilsTypedTest, PadToMaxNonBatchDims) {
  using T = TypeParam;
  Allocator* allocator = DefaultAllocator::GetInstance();
  // Before padding:
  //   Task 1: shape [1, 1, 2, 2] -> 4 elements
  //   Task 2: shape [2, 2, 2, 3] -> 24 elements. Already matches target shape.
  //   Task 3: shape [1, 2, 2, 1] -> 4 elements
  // Common shape: [x, 2, 2, 3]
  HostBuffer task1 =
      AsHostBuffer<T>({1, 2, 3, 4}, xla::ifrt::Shape({1, 1, 2, 2}), allocator);
  HostBuffer task2 =
      AsHostBuffer<T>({5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28},
                      xla::ifrt::Shape({2, 2, 2, 3}), allocator);
  HostBuffer task3 = AsHostBuffer<T>({29, 30, 31, 32},
                                     xla::ifrt::Shape({1, 2, 2, 1}), allocator);
  // After padding, the shapes are:
  //   Task 1 shape: [1, 2, 2, 3] -> 12 elements
  //   Task 2 shape: [2, 2, 2, 3] -> 24 elements. No padding needed.
  //   Task 3 shape: [1, 2, 2, 3] -> 12 elements
  HostBuffer expected_task1 =
      AsHostBuffer<T>({1, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0},
                      xla::ifrt::Shape({1, 2, 2, 3}), allocator);
  HostBuffer expected_task2 =
      AsHostBuffer<T>({5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28},
                      xla::ifrt::Shape({2, 2, 2, 3}), allocator);
  HostBuffer expected_task3 =
      AsHostBuffer<T>({29, 0, 0, 30, 0, 0, 31, 0, 0, 32, 0, 0},
                      xla::ifrt::Shape({1, 2, 2, 3}), allocator);
  std::vector<HostBuffer> inputs = {task1, task2, task3};

  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          PadToCommonNonBatchDims(inputs, allocator));

  ASSERT_EQ(outputs.size(), 3);
  EXPECT_THAT(outputs[0], HostBufferEq(expected_task1));
  EXPECT_THAT(outputs[1], HostBufferEq(expected_task2));
  EXPECT_THAT(outputs[2], HostBufferEq(expected_task3));
  EXPECT_NE(outputs[0].base(), task1.base());
  // Task 2 is not padded. The output is a shallow copy.
  EXPECT_EQ(outputs[1].base(), task2.base());
  EXPECT_NE(outputs[2].base(), task3.base());
}

TEST(PadUtilsTest, EmptyInputs) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  std::vector<HostBuffer> inputs = {};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          PadToCommonNonBatchDims(inputs, allocator));

  EXPECT_EQ(outputs.size(), 0);
}

TEST(PadUtilsTest, NullAllocator) {
  HostBuffer task1 =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({1, 2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          PadToCommonNonBatchDims({task1}, nullptr));

  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].base(), task1.base());
}

TEST(PadUtilsTest, DtypeMismatchError) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer task1 =
      AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2}), allocator);
  HostBuffer task2 =
      AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({1, 2}), allocator);

  std::vector<HostBuffer> inputs = {task1, task2};
  EXPECT_THAT(PadToCommonNonBatchDims(inputs, allocator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dtype mismatch across inputs: input 0 has "
                                 "S32 but input 1 has F32")));
}

TEST(PadUtilsTest, RankMismatchError) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer task1 =
      AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({2}), allocator);
  HostBuffer task2 =
      AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2}), allocator);

  std::vector<HostBuffer> inputs = {task1, task2};
  EXPECT_THAT(PadToCommonNonBatchDims(inputs, allocator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Rank mismatch across inputs: input 0 has "
                                 "rank 1 but input 1 has rank 2")));
}

TEST(PadUtilsTest, InputNonBatchDimsAlreadyMatch) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer task1 = AsHostBuffer<int32_t>(
      {1, 2, 3, 4}, xla::ifrt::Shape({1, 2, 2}), allocator);
  HostBuffer task2 = AsHostBuffer<int32_t>(
      {5, 6, 7, 8, 9, 10, 11, 12}, xla::ifrt::Shape({2, 2, 2}), allocator);

  std::vector<HostBuffer> inputs = {task1, task2};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          PadToCommonNonBatchDims(inputs, allocator));

  ASSERT_EQ(outputs.size(), 2);
  // Left untouched, without allocating.
  EXPECT_EQ(outputs[0].base(), task1.base());
  EXPECT_EQ(outputs[1].base(), task2.base());
}

TEST(PadUtilsTest, Rank1PassesThrough) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer task1 =
      AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({3}), allocator);
  HostBuffer task2 =
      AsHostBuffer<int32_t>({4, 5}, xla::ifrt::Shape({2}), allocator);

  std::vector<HostBuffer> inputs = {task1, task2};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          PadToCommonNonBatchDims(inputs, allocator));

  ASSERT_EQ(outputs.size(), 2);
  // Left untouched, without allocating.
  EXPECT_EQ(outputs[0].base(), task1.base());
  EXPECT_EQ(outputs[1].base(), task2.base());
}

TEST(PadUtilsTest, PadString) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  // Before padding:
  //   Task 1: shape [1, 2, 1] -> 2 elements
  //   Task 2: shape [1, 1, 3] -> 3 elements
  // Common shape: [1, 2, 3]
  HostBuffer task1 = AsHostBuffer<tsl::tstring>(
      {"hello", "world"}, xla::ifrt::Shape({1, 2, 1}), allocator);
  HostBuffer task2 = AsHostBuffer<tsl::tstring>(
      {"a", "b", "c"}, xla::ifrt::Shape({1, 1, 3}), allocator);
  // After padding:
  //   Task 1: shape [1, 2, 3] -> 6 elements
  //   Task 2: shape [1, 2, 3] -> 6 elements
  HostBuffer expected_task1 =
      AsHostBuffer<tsl::tstring>({"hello", "", "", "world", "", ""},
                                 xla::ifrt::Shape({1, 2, 3}), allocator);
  HostBuffer expected_task2 = AsHostBuffer<tsl::tstring>(
      {"a", "b", "c", "", "", ""}, xla::ifrt::Shape({1, 2, 3}), allocator);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          PadToCommonNonBatchDims({task1, task2}, allocator));

  EXPECT_EQ(outputs.size(), 2);
  EXPECT_THAT(outputs[0], HostBufferEq(expected_task1));
  EXPECT_THAT(outputs[1], HostBufferEq(expected_task2));
}

TEST(PadUtilsTest, PadBoolean) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  // Before padding:
  //   Task 1: shape [1, 2, 1] -> 2 elements
  //   Task 2: shape [1, 1, 3] -> 3 elements
  // Common shape: [1, 2, 3]
  HostBuffer task1 =
      AsHostBuffer<bool>({true, false}, xla::ifrt::Shape({1, 2, 1}), allocator);
  HostBuffer task2 = AsHostBuffer<bool>({true, false, true},
                                        xla::ifrt::Shape({1, 1, 3}), allocator);
  // After padding:
  //   Task 1: shape [1, 2, 3] -> 6 elements
  //   Task 2: shape [1, 2, 3] -> 6 elements
  HostBuffer expected_task1 =
      AsHostBuffer<bool>({true, false, false, false, false, false},
                         xla::ifrt::Shape({1, 2, 3}), allocator);
  HostBuffer expected_task2 =
      AsHostBuffer<bool>({true, false, true, false, false, false},
                         xla::ifrt::Shape({1, 2, 3}), allocator);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          PadToCommonNonBatchDims({task1, task2}, allocator));

  EXPECT_EQ(outputs.size(), 2);
  EXPECT_THAT(outputs[0], HostBufferEq(expected_task1));
  EXPECT_THAT(outputs[1], HostBufferEq(expected_task2));
}

TEST(PadUtilsTest, EmptyElementsWithPadding) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  // Input 1 shape [0, 2] has 0 elements but non-batch dimension 2.
  HostBuffer task1 =
      AsHostBuffer<int32_t>({}, xla::ifrt::Shape({0, 2}), allocator);
  // Input 2 shape [1, 3] establishes target non-batch dimension 3.
  HostBuffer task2 =
      AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({1, 3}), allocator);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          PadToCommonNonBatchDims({task1, task2}, allocator));

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].shape(), xla::ifrt::Shape({0, 3}));
  EXPECT_EQ(outputs[0].shape().num_elements(), 0);
  EXPECT_THAT(outputs[1], HostBufferEq(task2));
}

TEST(PadUtilsTest, SingleInputPassesThrough) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer task1 =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}), allocator);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          PadToCommonNonBatchDims({task1}, allocator));

  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].base(), task1.base());
}

TEST(PadUtilsTest, UnsupportedDTypeError) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  const xla::ifrt::DType invalid_dtype(xla::ifrt::DType::Kind::kInvalid);
  char dummy_data[16] = {0};
  HostBuffer task1(dummy_data, invalid_dtype, xla::ifrt::Shape({1, 2}));
  HostBuffer task2(dummy_data, invalid_dtype, xla::ifrt::Shape({1, 3}));

  EXPECT_THAT(
      PadToCommonNonBatchDims({task1, task2}, allocator),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unsupported data type for padding: INVALID")));
}

TEST(PadUtilsTest, MissingByteStridesError) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  int32_t dummy_data[6] = {0};
  HostBuffer task1(dummy_data, xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                   xla::ifrt::Shape({1, 2}), /*byte_strides=*/std::nullopt,
                   /*owner=*/nullptr);
  HostBuffer task2 =
      AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({1, 3}), allocator);

  EXPECT_THAT(
      PadToCommonNonBatchDims({task1, task2}, allocator),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("HostBuffer must have byte strides for padding, got: "
                         "S32[1,2]")));
}

TEST(PadUtilsTest, UninitializedInputError) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer valid_task =
      AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({1, 3}), allocator);

  EXPECT_THAT(PadToCommonNonBatchDims({HostBuffer(), valid_task}, allocator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is uninitialized")));
  EXPECT_THAT(PadToCommonNonBatchDims({valid_task, HostBuffer()}, allocator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is uninitialized")));
}

}  // namespace
}  // namespace xla_serving
