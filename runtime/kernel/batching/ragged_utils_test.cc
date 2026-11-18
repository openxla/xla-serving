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

#include "runtime/kernel/batching/ragged_utils.h"  // from @xla_serving

#include <cstdint>
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
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(RaggedUtilsTest, ConcatRowSplitsInt32) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits1 = AsHostBuffer<int32_t>({0, 2, 3}, xla::ifrt::Shape({3}));
  HostBuffer splits2 = AsHostBuffer<int32_t>({0, 1, 3}, xla::ifrt::Shape({3}));

  std::vector<HostBuffer> tasks_splits = {splits1, splits2};
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer concatenated,
      ConcatRowSplits(tasks_splits, /*padding_amount=*/0, allocator));

  std::vector<int32_t> expected = {0, 2, 3, 4, 6};
  EXPECT_THAT(concatenated, HostBufferEq(AsHostBuffer<int32_t>(
                                expected, xla::ifrt::Shape({5}))));
}

TEST(RaggedUtilsTest, ConcatRowSplitsInt64WithPadding) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits1 = AsHostBuffer<int64_t>({0, 2, 3}, xla::ifrt::Shape({3}));
  HostBuffer splits2 = AsHostBuffer<int64_t>({0, 1, 3}, xla::ifrt::Shape({3}));

  std::vector<HostBuffer> tasks_splits = {splits1, splits2};
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer concatenated,
      ConcatRowSplits(tasks_splits, /*padding_amount=*/4, allocator));

  std::vector<int64_t> expected = {0, 2, 3, 4, 6, 6, 6, 6, 6};
  EXPECT_THAT(concatenated, HostBufferEq(AsHostBuffer<int64_t>(
                                expected, xla::ifrt::Shape({9}))));
}

TEST(RaggedUtilsTest, ConcatRowSplitsSingleTaskNoPadding) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits = AsHostBuffer<int32_t>({0, 5, 10}, xla::ifrt::Shape({3}));

  std::vector<HostBuffer> tasks_splits = {splits};
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer concatenated,
      ConcatRowSplits(tasks_splits, /*padding_amount=*/0, allocator));

  std::vector<int32_t> expected = {0, 5, 10};
  EXPECT_THAT(concatenated, HostBufferEq(AsHostBuffer<int32_t>(
                                expected, xla::ifrt::Shape({3}))));
}

TEST(RaggedUtilsTest, ConcatRowSplitsSingleTaskWithPadding) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits = AsHostBuffer<int64_t>({0, 3}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> tasks_splits = {splits};
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer concatenated,
      ConcatRowSplits(tasks_splits, /*padding_amount=*/3, allocator));

  std::vector<int64_t> expected = {0, 3, 3, 3, 3};
  EXPECT_THAT(concatenated, HostBufferEq(AsHostBuffer<int64_t>(
                                expected, xla::ifrt::Shape({5}))));
}

TEST(RaggedUtilsTest, ConcatRowSplitsZeroRowsTask) {
  // Corner case: concatenating tasks with 0 rows (splits shape {1}, value [0])
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits1 = AsHostBuffer<int32_t>({0}, xla::ifrt::Shape({1}));
  HostBuffer splits2 = AsHostBuffer<int32_t>({0, 3}, xla::ifrt::Shape({2}));
  HostBuffer splits3 = AsHostBuffer<int32_t>({0}, xla::ifrt::Shape({1}));
  HostBuffer splits4 = AsHostBuffer<int32_t>({0, 2}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> tasks_splits = {splits1, splits2, splits3, splits4};
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer concatenated,
      ConcatRowSplits(tasks_splits, /*padding_amount=*/0, allocator));

  // splits1 & splits3 contribute 0 rows (0 extra elements in splits).
  // Total rows = 0 + 1 + 0 + 1 = 2 rows -> 3 split elements: [0, 3, 5].
  std::vector<int32_t> expected = {0, 3, 5};
  EXPECT_THAT(concatenated, HostBufferEq(AsHostBuffer<int32_t>(
                                expected, xla::ifrt::Shape({3}))));
}

TEST(RaggedUtilsTest, ConcatRowSplitsEmptyRaggedRows) {
  // Corner case: tasks containing rows with 0 elements (e.g., [0, 0, 0] and [0,
  // 0])
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits1 = AsHostBuffer<int64_t>({0, 0, 0}, xla::ifrt::Shape({3}));
  HostBuffer splits2 = AsHostBuffer<int64_t>({0, 0}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> tasks_splits = {splits1, splits2};
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer concatenated,
      ConcatRowSplits(tasks_splits, /*padding_amount=*/0, allocator));

  std::vector<int64_t> expected = {0, 0, 0, 0};
  EXPECT_THAT(concatenated, HostBufferEq(AsHostBuffer<int64_t>(
                                expected, xla::ifrt::Shape({4}))));
}

TEST(RaggedUtilsTest, CreateWarmupRowSplitsInt32) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  // Both splits_1 and splits_2 should yield the same warmup splits.
  HostBuffer splits_1 = AsHostBuffer<int32_t>({0, 2, 3}, xla::ifrt::Shape({3}));
  HostBuffer splits_2 = AsHostBuffer<int32_t>({0, 2}, xla::ifrt::Shape({2}));

  for (const auto& splits : {splits_1, splits_2}) {
    TF_ASSERT_OK_AND_ASSIGN(
        HostBuffer warmup_splits,
        CreateWarmupRowSplits(splits, /*warmup_batch_size=*/4, allocator));

    // Row 0 has offset difference 2. Replicated 4 times: [0, 2, 4, 6, 8].
    std::vector<int32_t> expected = {0, 2, 4, 6, 8};
    EXPECT_THAT(warmup_splits, HostBufferEq(AsHostBuffer<int32_t>(
                                  expected, xla::ifrt::Shape({5}))));
    }
}

TEST(RaggedUtilsTest, CreateWarmupRowSplitsInvalidShape) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits = AsHostBuffer<int32_t>({0}, xla::ifrt::Shape({1}));

  EXPECT_THAT(CreateWarmupRowSplits(splits, /*warmup_batch_size=*/4, allocator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("fewer than 2 elements")));
}

TEST(RaggedUtilsTest, CreateWarmupFlatValuesInt32) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits = AsHostBuffer<int32_t>({0, 2, 5}, xla::ifrt::Shape({3}));
  HostBuffer values =
      AsHostBuffer<int32_t>({10, 20, 30, 40, 50}, xla::ifrt::Shape({5}));

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer warmup_values,
      CreateWarmupFlatValues(values, splits, /*warmup_batch_size=*/3,
                             /*thread_pool=*/nullptr, allocator));

  // Row 0 contains the first 2 values [10, 20]. Replicated 3 times: [10, 20,
  // 10, 20, 10, 20].
  std::vector<int32_t> expected = {10, 20, 10, 20, 10, 20};
  EXPECT_THAT(warmup_values, HostBufferEq(AsHostBuffer<int32_t>(
                                 expected, xla::ifrt::Shape({6}))));
}

TEST(RaggedUtilsTest, CreateWarmupFlatValuesWithSingleElementInFirstRow) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits = AsHostBuffer<int32_t>({0, 1, 5}, xla::ifrt::Shape({3}));
  HostBuffer values =
      AsHostBuffer<int32_t>({10, 20, 30, 40, 50}, xla::ifrt::Shape({5}));

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer warmup_values,
      CreateWarmupFlatValues(values, splits, /*warmup_batch_size=*/3,
                             /*thread_pool=*/nullptr, allocator));

  // Row 0 contains the 1 value [10]. Replicated 3 times for
  // warmup_batch_size 3: [10, 10, 10].
  std::vector<int32_t> expected = {10, 10, 10};
  EXPECT_THAT(warmup_values, HostBufferEq(AsHostBuffer<int32_t>(
                                 expected, xla::ifrt::Shape({3}))));
}

TEST(RaggedUtilsTest, CreateWarmupFlatValuesInvalidSplits) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits = AsHostBuffer<int32_t>({0}, xla::ifrt::Shape({1}));
  HostBuffer values = AsHostBuffer<int32_t>({10}, xla::ifrt::Shape({1}));

  EXPECT_THAT(CreateWarmupFlatValues(values, splits, /*warmup_batch_size=*/3,
                                     /*thread_pool=*/nullptr, allocator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("at least 2 elements")));
}

TEST(RaggedUtilsTest, ConcatRowSplitsUnsupportedDType) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer splits = AsHostBuffer<float>({0.0f, 1.0f}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> tasks_splits = {splits};
  EXPECT_THAT(ConcatRowSplits(tasks_splits, /*padding_amount=*/0, allocator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported splits dtype")));
}

TEST(RaggedUtilsTest, ConcatRowSplitsMismatchedDTypes) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer s64_splits =
      AsHostBuffer<int64_t>({0, 2, 3}, xla::ifrt::Shape({3}));
  HostBuffer s32_splits =
      AsHostBuffer<int32_t>({0, 1, 3}, xla::ifrt::Shape({3}));

  EXPECT_THAT(ConcatRowSplits({s64_splits, s32_splits}, /*padding_amount=*/0,
                              allocator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dtypes of all row splits buffers")));
  EXPECT_THAT(
      ConcatRowSplitsTyped<int64_t>({s32_splits}, /*padding_amount=*/0,
                                    allocator),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Dtypes of all row splits buffers")));
  EXPECT_THAT(ConcatRowSplits({HostBuffer(), s64_splits}, /*padding_amount=*/0,
                              allocator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is uninitialized")));
  EXPECT_THAT(ConcatRowSplits({s64_splits, HostBuffer()}, /*padding_amount=*/0,
                              allocator),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is uninitialized")));
  EXPECT_THAT(
      ConcatRowSplitsTyped<int64_t>({HostBuffer()}, /*padding_amount=*/0,
                                    allocator),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("is uninitialized")));
}

TEST(RaggedUtilsTest, ConcatRowSplitsEmptyError) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  std::vector<HostBuffer> empty_splits;
  EXPECT_THAT(
      ConcatRowSplits(empty_splits, 0, allocator),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty splits")));
}

TEST(RaggedUtilsTest, ConcatRowSplitsTypedEmptyErrorInt32) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  std::vector<HostBuffer> empty_splits;
  EXPECT_THAT(
      ConcatRowSplitsTyped<int32_t>(empty_splits, 0, allocator),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty splits")));
}

TEST(RaggedUtilsTest, ConcatRowSplitsTypedEmptyErrorInt64) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  std::vector<HostBuffer> empty_splits;
  EXPECT_THAT(
      ConcatRowSplitsTyped<int64_t>(empty_splits, 0, allocator),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty splits")));
}

TEST(RaggedUtilsTest, CloneAndModifySplitsInt32) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer original =
      AsHostBuffer<int32_t>({0, 2, 3, 5, 8}, xla::ifrt::Shape({5}));

  // Slice from index 1 to 3 (rows 1..3, total 3 rows -> 4 split elements)
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer sub_splits,
      CloneAndModifySplits(original, /*start_idx=*/1, /*end_idx=*/3, allocator));

  // original[1..3] is [2, 3, 5], offset is original[1] = 2
  // modified values: [2-2, 3-2, 5-2] = [0, 1, 3]
  std::vector<int32_t> expected = {0, 1, 3};
  EXPECT_THAT(sub_splits, HostBufferEq(AsHostBuffer<int32_t>(
                              expected, xla::ifrt::Shape({3}))));
}

TEST(RaggedUtilsTest, CloneAndModifySplitsInt64) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer original =
      AsHostBuffer<int64_t>({0, 2, 3, 5, 8}, xla::ifrt::Shape({5}));

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer sub_splits,
      CloneAndModifySplits(original, /*start_idx=*/2, /*end_idx=*/4, allocator));

  // original[2..4] is [3, 5, 8], offset is original[2] = 3
  // modified values: [3-3, 5-3, 8-3] = [0, 2, 5]
  std::vector<int64_t> expected = {0, 2, 5};
  EXPECT_THAT(sub_splits, HostBufferEq(AsHostBuffer<int64_t>(
                              expected, xla::ifrt::Shape({3}))));
}

TEST(RaggedUtilsTest, CloneAndModifySplitsSingleElement) {
  // Corner case: start_idx == end_idx (slicing a single split point -> 0 rows)
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer original =
      AsHostBuffer<int32_t>({0, 2, 3, 5, 8}, xla::ifrt::Shape({5}));

  TF_ASSERT_OK_AND_ASSIGN(HostBuffer sub_splits,
                          CloneAndModifySplits(original, /*start_idx=*/2,
                                               /*end_idx=*/2, allocator));

  std::vector<int32_t> expected = {0};
  EXPECT_THAT(sub_splits, HostBufferEq(AsHostBuffer<int32_t>(
                              expected, xla::ifrt::Shape({1}))));
}

TEST(RaggedUtilsTest, CloneAndModifySplitsFullRange) {
  // Corner case: slicing the full range of original splits
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer original =
      AsHostBuffer<int64_t>({0, 2, 3, 5, 8}, xla::ifrt::Shape({5}));

  TF_ASSERT_OK_AND_ASSIGN(HostBuffer sub_splits,
                          CloneAndModifySplits(original, /*start_idx=*/0,
                                               /*end_idx=*/4, allocator));

  std::vector<int64_t> expected = {0, 2, 3, 5, 8};
  EXPECT_THAT(sub_splits, HostBufferEq(AsHostBuffer<int64_t>(
                              expected, xla::ifrt::Shape({5}))));
}

TEST(RaggedUtilsTest, CloneAndModifySplitsUnsupportedDType) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  HostBuffer original =
      AsHostBuffer<float>({0.0f, 1.0f, 2.0f}, xla::ifrt::Shape({3}));

  EXPECT_THAT(
      CloneAndModifySplits(original, /*start_idx=*/0, /*end_idx=*/2, allocator),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unsupported splits dtype")));
}

}  // namespace
}  // namespace xla_serving
