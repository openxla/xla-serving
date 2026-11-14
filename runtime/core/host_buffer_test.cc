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

#include "runtime/core/host_buffer.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "xla/tsl/platform/test_benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/cleanup/cleanup.h"  // from @abseil-cpp
#include "absl/flags/declare.h"  // from @abseil-cpp
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/grpc/host_buffer.pb.h"  // from @xla_serving
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/platform/bfloat16.h"  // from @tsl
#include "tsl/platform/ml_dtypes.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

ABSL_DECLARE_FLAG(int64_t, xla_serving_host_buffer_max_elements);

namespace xla_serving {

class HostBufferSplitEasyCasesTest : public ::testing::Test {
 protected:
  void SetUp() override { allocator_ = DefaultAllocator::GetInstance(); }

  Allocator* allocator_;

  absl::Status SplitEasyCases(HostBuffer* host_buffer,
                              absl::Span<const int64_t> sizes,
                              std::vector<HostBuffer>& output,
                              bool& easy_cases_done) {
    return host_buffer->SplitEasyCases(allocator_, sizes, output,
                                       easy_cases_done);
  }
};

namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(HostBufferTest, NullPtrAllocator) {
  HostBuffer host_buffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kU32),
                         xla::ifrt::Shape({2, 2}), nullptr);
  EXPECT_EQ(host_buffer.dtype(),
            xla::ifrt::DType(xla::ifrt::DType::Kind::kU32));
  EXPECT_EQ(host_buffer.shape(), xla::ifrt::Shape({2, 2}));
}

// Type parameterized test for testing HostBuffer mutation and storage of data
// type int8_t, int16_t, int32_t, float, double.
template <typename T>
class HostBufferMutationAndStorageTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(HostBufferMutationAndStorageTest);

TYPED_TEST_P(HostBufferMutationAndStorageTest, CanMutate) {
  using T = TypeParam;
  std::vector<T> original_data = {1, 2, 3, 4};
  T mutated_data = static_cast<T>(127);
  std::vector<T> expected_data = {1, mutated_data, 3, 4};

  HostBuffer host_buffer =
      AsHostBuffer<T>(original_data, xla::ifrt::Shape({2, 2}));
  host_buffer.flat<T>()(1) = mutated_data;
  EXPECT_THAT(host_buffer, HostBufferEq(AsHostBuffer<T>(
                               expected_data, xla::ifrt::Shape({2, 2}))));
}

TYPED_TEST_P(HostBufferMutationAndStorageTest, TotalStorageBytes) {
  using T = TypeParam;
  std::vector<T> original_data = {1, 2, 3, 4};
  HostBuffer host_buffer =
      AsHostBuffer<T>(original_data, xla::ifrt::Shape({2, 2}));
  EXPECT_EQ(host_buffer.total_storage_bytes(),
            original_data.size() * sizeof(T));
}

REGISTER_TYPED_TEST_SUITE_P(HostBufferMutationAndStorageTest, CanMutate,
                            TotalStorageBytes);

using HostBufferTestTypes =
    ::testing::Types<int8_t, int16_t, int32_t, float, double>;

INSTANTIATE_TYPED_TEST_SUITE_P(HostBufferMutationAndStorageTests,
                               HostBufferMutationAndStorageTest,
                               HostBufferTestTypes);

TEST(HostBufferTest, BoolCanMutate) {
  HostBuffer host_buffer =
      AsHostBuffer<bool>({true, false, true, false}, xla::ifrt::Shape({2, 2}));
  host_buffer.flat<bool>()(1) = true;
  EXPECT_THAT(host_buffer,
              HostBufferEq(AsHostBuffer<bool>({true, true, true, false},
                                              xla::ifrt::Shape({2, 2}))));
}

TEST(HostBufferTest, Int4CanMutate) {
  HostBuffer host_buffer =
      AsHostBuffer<tsl::int4, int>({1, 2, 3, 1}, xla::ifrt::Shape({2, 2}));

  host_buffer.flat<tsl::int4>()(1) = tsl::int4(3);
  EXPECT_THAT(host_buffer, HostBufferEq(AsHostBuffer<tsl::int4, int>(
                               {1, 3, 3, 1}, xla::ifrt::Shape({2, 2}))));
}

TEST(HostBufferTest, Bfloat16CanMutate) {
  HostBuffer host_buffer = AsHostBuffer<tsl::bfloat16, float>(
      {1.0f, 2.0f, 3.0f, 1.0f}, xla::ifrt::Shape({2, 2}));

  host_buffer.flat<tsl::bfloat16>()(1) = tsl::bfloat16(3.0f);
  EXPECT_THAT(host_buffer,
              HostBufferEq(AsHostBuffer<tsl::bfloat16, float>(
                  {1.0f, 3.0f, 3.0f, 1.0f}, xla::ifrt::Shape({2, 2}))));
}

TEST(HostBufferTest, StringCanMutate) {
  std::string long_str;
  for (int i = 0; i < 1000000; ++i) {
    long_str += "long";
  }
  std::vector<HostBuffer> inputs;
  tsl::tstring str_data[6] = {"aaa", "bbb", "ccc", "ddd", "eee", long_str};
  HostBuffer host_buffer(str_data,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                         xla::ifrt::Shape({2, 3}));
  host_buffer.flat<tsl::tstring>()(1) = long_str;
  EXPECT_EQ(host_buffer.flat<tsl::tstring>()(1), long_str);
}

TEST(HostBufferTest, TotalStorageBytesForInt4) {
  HostBuffer host_buffer =
      AsHostBuffer<tsl::int4, int>({1, 2}, xla::ifrt::Shape({2}));
  ASSERT_EQ(host_buffer.total_storage_bytes(), 2);
}

TEST(HostBufferTest, TotalStorageBytesForBfloat16) {
  HostBuffer host_buffer =
      AsHostBuffer<tsl::bfloat16, float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  // sizeof(tsl::bfloat16) is 2.
  EXPECT_EQ(host_buffer.total_storage_bytes(), 4);
}

TEST(HostBufferTest, DebugStringOfEmpty) {
  HostBuffer host_buffer;
  EXPECT_EQ(host_buffer.DebugString(), "Empty host buffer");
}

TEST(HostBufferTest, DebugStringOfNumericDtype) {
  HostBuffer host_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  EXPECT_EQ(host_buffer.DebugString(), "S32[2,2][\n  [1, 2],\n  [3, 4]\n]");
}

TEST(HostBufferTest, DebugStringOfStringDtype) {
  tsl::tstring str_data[2] = {"hello", "world"};
  HostBuffer host_buffer(str_data,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                         xla::ifrt::Shape({2}));
  EXPECT_EQ(host_buffer.DebugString(), "STRING[2]['hello', 'world']");
}

TEST(HostBufferTest, DebugStringOfBfloat16Dtype) {
  tsl::bfloat16 data[2] = {tsl::bfloat16(1.5f), tsl::bfloat16(2.5f)};
  HostBuffer host_buffer(data, xla::ifrt::DType(xla::ifrt::DType::Kind::kBF16),
                         xla::ifrt::Shape({2}));
  EXPECT_EQ(host_buffer.DebugString(), "BF16[2][1.5, 2.5]");
}

TEST(HostBufferTest, Copy) {
  HostBuffer original =
      AsHostBuffer<float>({1.0, 2.0, 3.0}, xla::ifrt::Shape({3}));
  HostBuffer copy = original;
  EXPECT_THAT(copy, HostBufferEq(original));
}

TEST(HostBufferTest, StringCopy) {
  tsl::tstring str_data[3] = {"aaa", "bbb", "ccc"};
  HostBuffer original(str_data,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                         xla::ifrt::Shape({3}));
  HostBuffer copy = original;
  EXPECT_THAT(copy, HostBufferEq(original));
}

TEST(HostBufferTest, CopyIsAliveAfterOriginalBufferIsDestroyed) {
  HostBuffer copy;
  {
    HostBuffer original =
        AsHostBuffer<bool>({true, false, true}, xla::ifrt::Shape({3}));
    copy = original;
  }
  EXPECT_THAT(copy, HostBufferEq(AsHostBuffer<bool>({true, false, true},
                                                    xla::ifrt::Shape({3}))));
}

TEST(HostBufferTest, StringCopyIsAliveAfterOriginalBufferIsDestroyed) {
  HostBuffer copy;
  tsl::tstring str_data[3] = {"aaa", "bbb", "ccc"};
  {
    HostBuffer original(str_data,
      xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({3}));
    copy = original;
  }
  HostBuffer expected(str_data,
    xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
    xla::ifrt::Shape({3}));
  EXPECT_THAT(copy, HostBufferEq(expected));
}

TEST(HostBufferTest, CopyAlsoMutates) {
  HostBuffer original =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  HostBuffer copy = original;
  EXPECT_THAT(copy, HostBufferEq(original));

  original.flat<int32_t>()(1) = 200;
  // Verify that the copied buffer is also mutated after the mutation of the
  // original buffer.
  for (const HostBuffer& buffer : {original, copy}) {
    EXPECT_THAT(buffer, HostBufferEq(AsHostBuffer<int32_t>(
                            {1, 200, 3, 4}, xla::ifrt::Shape({2, 2}))));
  }

  copy.flat<int32_t>()(3) = 201;
  // Verify that the original buffer is also mutated after the mutation of the
  // copied buffer.
  for (const HostBuffer& buffer : {original, copy}) {
    EXPECT_THAT(buffer, HostBufferEq(AsHostBuffer<int32_t>(
                            {1, 200, 3, 201}, xla::ifrt::Shape({2, 2}))));
  }
}

TEST(HostBufferTest, StringCopyAlsoMutates) {
  HostBuffer original = AsHostBuffer<tsl::tstring, std::string>(
      {"aaa", "bbb", "ccc", "ddd"}, xla::ifrt::Shape({2, 2}));
  HostBuffer copy = original;
  EXPECT_THAT(copy, HostBufferEq(original));

  original.flat<tsl::tstring>()(1) = "changed";
  // Verify that the copied buffer is also mutated after the mutation of the
  // original buffer.
  for (const HostBuffer& buffer : {original, copy}) {
    EXPECT_THAT(buffer, HostBufferEq(AsHostBuffer<tsl::tstring, std::string>(
                            {"aaa", "changed", "ccc", "ddd"},
                            xla::ifrt::Shape({2, 2}))));
  }

  copy.flat<tsl::tstring>()(3) = "new_changed";
  // Verify that the original buffer is also mutated after the mutation of the
  // copied buffer.
  for (const HostBuffer& buffer : {original, copy}) {
    EXPECT_THAT(buffer, HostBufferEq(AsHostBuffer<tsl::tstring, std::string>(
                            {"aaa", "changed", "ccc", "new_changed"},
                            xla::ifrt::Shape({2, 2}))));
  }
}

TEST(HostBufferTest, Move) {
  HostBuffer original =
      AsHostBuffer<float>({1.0, 2.0, 3.0}, xla::ifrt::Shape({3}));
  int original_ref_count = original.ref_count();
  HostBuffer move = std::move(original);
  EXPECT_THAT(move, HostBufferEq(AsHostBuffer<float>({1.0, 2.0, 3.0},
                                                     xla::ifrt::Shape({3}))));
  // Verify that the original buffer does not hold a valid host array after
  // move.
  EXPECT_EQ(move.ref_count(), original_ref_count);
}

TEST(HostBufferTest, StringMove) {
  HostBuffer original = AsHostBuffer<tsl::tstring, std::string>(
      {"aaa", "bbb", "ccc"}, xla::ifrt::Shape({3}));
  int original_ref_count = original.ref_count();
  HostBuffer move = std::move(original);
  EXPECT_THAT(move, HostBufferEq(AsHostBuffer<tsl::tstring, std::string>(
                        {"aaa", "bbb", "ccc"}, xla::ifrt::Shape({3}))));
  // Verify that the original buffer does not hold a valid host array after
  // move.
  EXPECT_EQ(move.ref_count(), original_ref_count);
}

TEST(HostBufferTest, MoveIsAliveAfterOriginalBufferIsDestroyed) {
  HostBuffer move;
  {
    HostBuffer original =
        AsHostBuffer<double>({1.0, 2.0, 3.0}, xla::ifrt::Shape({1, 3, 1}));
    move = std::move(original);
  }
  EXPECT_THAT(move, HostBufferEq(AsHostBuffer<double>(
                        {1.0, 2.0, 3.0}, xla::ifrt::Shape({1, 3, 1}))));
}

class HostBufferManipulationTest : public ::testing::Test {
 protected:
  void SetUp() override { allocator_ = DefaultAllocator::GetInstance(); }

  Allocator* allocator_;
};

TEST_F(HostBufferManipulationTest, SliceStringBuffer) {
  std::string long_str;
  for (int i = 0; i < 1000000; ++i) {
    long_str += "long";
  }
  std::vector<HostBuffer> inputs;
  tsl::tstring str_data[6] = {"aaa", long_str, "ccc", "ddd", "eee", long_str};
  HostBuffer host_buffer(str_data,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                         xla::ifrt::Shape({2, 3}));

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer first,
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/1, allocator_));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer second,
      host_buffer.SliceAlongFirstDimension(/*start=*/1, /*end=*/2, allocator_));

  tsl::tstring expected_first_strings[3] = {"aaa", long_str, "ccc"};
  tsl::tstring expected_second_strings[3] = {"ddd", "eee", long_str};
  HostBuffer expected_first(expected_first_strings,
                            xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                            xla::ifrt::Shape({1, 3}));
  HostBuffer expected_second(expected_second_strings,
                             xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                             xla::ifrt::Shape({1, 3}));
  EXPECT_THAT(first, HostBufferEq(expected_first));
  EXPECT_THAT(second, HostBufferEq(expected_second));
}

TEST_F(HostBufferManipulationTest, Slice2DBuffer) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 4}));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer sliced_host_buffer,
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/1, allocator_));
  EXPECT_THAT(sliced_host_buffer, HostBufferEq(AsHostBuffer<int32_t>(
                                      {1, 2, 3, 4}, xla::ifrt::Shape({1, 4}))));
}

TEST_F(HostBufferManipulationTest, Slice3DBufferForOneRow) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer sliced_host_buffer,
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/1, allocator_));
  EXPECT_THAT(sliced_host_buffer,
              HostBufferEq(AsHostBuffer<int32_t>({1, 2, 3, 4},
                                                 xla::ifrt::Shape({1, 2, 2}))));
}

TEST_F(HostBufferManipulationTest, Slice3DBufferForMultipleRwos) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer sliced_host_buffer,
      host_buffer.SliceAlongFirstDimension(/*start=*/1, /*end=*/3, allocator_));
  EXPECT_THAT(sliced_host_buffer,
              HostBufferEq(AsHostBuffer<int32_t>({5, 6, 7, 8, 9, 10, 11, 12},
                                                 xla::ifrt::Shape({2, 2, 2}))));
}

TEST_F(HostBufferManipulationTest,
       SlicedBufferLivesAfterOriginalBufferIsDestroyed) {
  HostBuffer sliced_host_buffer;
  {
    HostBuffer host_buffer = AsHostBuffer<int32_t>(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        xla::ifrt::Shape({4, 4}));
    TF_ASSERT_OK_AND_ASSIGN(sliced_host_buffer,
                            host_buffer.SliceAlongFirstDimension(
                                /*start=*/0, /*end=*/1, allocator_));
  }

  EXPECT_THAT(sliced_host_buffer, HostBufferEq(AsHostBuffer<int32_t>(
                                      {1, 2, 3, 4}, xla::ifrt::Shape({1, 4}))));
}

TEST_F(HostBufferManipulationTest, SliceFailedOnScalarBuffer) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(1);

  auto sliced_host_buffer =
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/1, allocator_);

  EXPECT_THAT(sliced_host_buffer, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, SliceFailedStartLessThanZero) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 4}));

  auto sliced_host_buffer =
      host_buffer.SliceAlongFirstDimension(/*start=*/-1, /*end=*/1, allocator_);

  EXPECT_THAT(sliced_host_buffer, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, SliceFailedEndLessThanZero) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 4}));

  auto sliced_host_buffer =
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/-1, allocator_);

  EXPECT_THAT(sliced_host_buffer, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, SliceFailedStartLargerThanEnd) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 4}));

  auto sliced_host_buffer =
      host_buffer.SliceAlongFirstDimension(/*start=*/1, /*end=*/0, allocator_);

  EXPECT_THAT(sliced_host_buffer, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, SliceDoesNotCopy) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8},
      xla::ifrt::Shape({4, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer sliced_host_buffer,
      host_buffer.SliceAlongFirstDimension(/*start=*/1, /*end=*/3, allocator_));
  EXPECT_THAT(sliced_host_buffer,
              HostBufferEq(AsHostBuffer<int32_t>({3, 4, 5, 6},
                                                 xla::ifrt::Shape({2, 2}))));

  // Modify the slice.
  sliced_host_buffer.unaligned_flat<int32_t>()(0) = 100;

  // Check if the original buffer is modified.
  EXPECT_EQ(host_buffer.unaligned_flat<int32_t>()(2), 100);
  EXPECT_THAT(sliced_host_buffer,
              HostBufferEq(AsHostBuffer<int32_t>({100, 4, 5, 6},
                                                 xla::ifrt::Shape({2, 2}))));
  EXPECT_THAT(host_buffer, HostBufferEq(AsHostBuffer<int32_t>(
                               {1, 2, 100, 4, 5, 6, 7, 8},
                               xla::ifrt::Shape({4, 2}))));
}

TEST_F(HostBufferManipulationTest, SliceEmptyBuffer) {
  HostBuffer host_buffer(/*base=*/nullptr,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kU32),
                         xla::ifrt::Shape({0}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto sliced_host_buffer,
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/0, allocator_));

  EXPECT_THAT(
      sliced_host_buffer,
      HostBufferEq(HostBuffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kU32),
                              xla::ifrt::Shape({0}), allocator_)));
}

TEST_F(HostBufferManipulationTest, StringConcatAndSplit) {
  std::string long_str;
  for (int i = 0; i < 100000; ++i) {
    long_str += "long";
  }
  tsl::tstring str_data0[2] = {"aaa", long_str};
  tsl::tstring str_data1[4] = {"ccc", "ddd", long_str, "fff"};
  tsl::tstring str_data2[4] = {"ggg", long_str, "iii", "jjjj"};
  tsl::tstring str_data3[10] = {"aaa", long_str, "ccc",    "ddd", long_str,
                                "fff", "ggg",    long_str, "iii", "jjjj"};
  HostBuffer host_buffer0(str_data0,
                          xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                          xla::ifrt::Shape({1, 2}));
  HostBuffer host_buffer1(str_data1,
                          xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                          xla::ifrt::Shape({2, 2}));
  HostBuffer host_buffer2(str_data2,
                          xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                          xla::ifrt::Shape({2, 2}));
  HostBuffer expected_host_buffer(
      str_data3, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({5, 2}));
  std::vector<HostBuffer> inputs;
  inputs.push_back(host_buffer0);
  inputs.push_back(host_buffer1);
  inputs.push_back(host_buffer2);
  TF_ASSERT_OK_AND_ASSIGN(auto concat_output,
                          Concat(nullptr, inputs, allocator_));
  EXPECT_THAT(concat_output, HostBufferEq(expected_host_buffer));
  TF_ASSERT_OK_AND_ASSIGN(auto outputs, concat_output.SplitAlongFirstDimension(
                                            {1, 2, 2}, allocator_));
  EXPECT_THAT(outputs, ElementsAre(HostBufferEq(host_buffer0),
                                   HostBufferEq(host_buffer1),
                                   HostBufferEq(host_buffer2)));
}

TEST_F(HostBufferManipulationTest, SplitAlongFirstDimensionTrivial) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({4, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<HostBuffer> split_host_buffers,
      host_buffer.SplitAlongFirstDimension({4}, allocator_));

  EXPECT_THAT(split_host_buffers, ElementsAre(HostBufferEq(host_buffer)));
}

TEST_F(HostBufferManipulationTest, SplitAlongFirstDimension) {
  std::vector<HostBuffer> split_host_buffers;
  {
    HostBuffer host_buffer = AsHostBuffer<int32_t>(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        xla::ifrt::Shape({4, 4}));
    TF_ASSERT_OK_AND_ASSIGN(
        split_host_buffers,
        host_buffer.SplitAlongFirstDimension({3, 1}, allocator_));
  }

  EXPECT_THAT(split_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                              xla::ifrt::Shape({3, 4}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {13, 14, 15, 16}, xla::ifrt::Shape({1, 4})))));
}

TEST_F(HostBufferManipulationTest, SplitSmallTypeAlongFirstDimension) {
  std::vector<HostBuffer> split_host_buffers;
  {
    HostBuffer host_buffer = AsHostBuffer<tsl::int4, int>(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        xla::ifrt::Shape({4, 4}));
    TF_ASSERT_OK_AND_ASSIGN(
        split_host_buffers,
        host_buffer.SplitAlongFirstDimension({3, 1}, allocator_));
  }

  for (const HostBuffer& split_host_buffer : split_host_buffers) {
    ASSERT_EQ(split_host_buffer.dtype(),
              xla::ifrt::DType(xla::ifrt::DType::Kind::kS4));
  }
  EXPECT_THAT(split_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<tsl::int4, int>(
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                              xla::ifrt::Shape({3, 4}))),
                          HostBufferEq(AsHostBuffer<tsl::int4, int>(
                              {12, 13, 14, 15}, xla::ifrt::Shape({1, 4})))));
}

TEST_F(HostBufferManipulationTest, Reshape) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 4}));
  auto reshaped_tensor =
      host_buffer.shaped<int32_t, 2>((std::vector<int64_t>{2, 8}));
  TF_ASSERT_OK(reshaped_tensor);
  EXPECT_EQ(reshaped_tensor->dimension(0), 2);
  EXPECT_EQ(reshaped_tensor->dimension(1), 8);
}

TEST_F(HostBufferManipulationTest, ConcatEmptyBuffers) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<int32_t>({}, xla::ifrt::Shape({2, 0})));
  inputs.push_back(AsHostBuffer<int32_t>({}, xla::ifrt::Shape({4, 0})));
  inputs.push_back(AsHostBuffer<int32_t>({}, xla::ifrt::Shape({6, 0})));
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));

  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                          {}, xla::ifrt::Shape({12, 0}))));
}

TEST_F(HostBufferManipulationTest, ConcatSingleBufferReturnsSameBuffer) {
  HostBuffer original_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  std::vector<HostBuffer> inputs;
  inputs.push_back(original_buffer);
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));

  EXPECT_THAT(output, HostBufferEq(original_buffer));
  EXPECT_EQ(output.base(), original_buffer.base());
}

TEST_F(HostBufferManipulationTest, ConcatSingleDimensionBuffers) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({2})));
  inputs.push_back(AsHostBuffer<int32_t>({3, 4, 5, 6}, xla::ifrt::Shape({4})));
  inputs.push_back(
      AsHostBuffer<int32_t>({7, 8, 9, 10, 11, 12}, xla::ifrt::Shape({6})));
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));

  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                          xla::ifrt::Shape({12}))));
}

TEST_F(HostBufferManipulationTest, ConcatWithZeroBatchDimension) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<int32_t>({}, xla::ifrt::Shape({0, 2, 2})));
  inputs.push_back(
      AsHostBuffer<int32_t>({3, 4, 5, 6}, xla::ifrt::Shape({1, 2, 2})));
  inputs.push_back(AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                         xla::ifrt::Shape({2, 2, 2})));
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));

  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                          {3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8},
                          xla::ifrt::Shape({3, 2, 2}))));
}

TEST_F(HostBufferManipulationTest, Concat) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(
      AsHostBuffer<int32_t>({3, 4, 5, 6}, xla::ifrt::Shape({1, 2, 2})));
  inputs.push_back(AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                         xla::ifrt::Shape({2, 2, 2})));
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));

  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                          {3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8},
                          xla::ifrt::Shape({3, 2, 2}))));
}

TEST_F(HostBufferManipulationTest, ConcatLargeBuffer) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "concat_test", 5);
  auto gen_data = [](absl::Span<const int64_t> shape,
                     std::vector<int32_t>* data) -> HostBuffer {
    xla::ifrt::Shape input_shape(shape);
    data->resize(input_shape.num_elements());
    std::iota(data->begin(), data->end(), 0);
    HostBuffer host_buffer = AsHostBuffer<int32_t>(*data, input_shape);
    return host_buffer;
  };
  std::vector<int32_t> data1, data2, data3;
  std::vector<HostBuffer> inputs;
  std::vector<int32_t> output_data;
  int batch_size = 0;
  for (int i = 0; i < 3; ++i) {
    std::vector<int32_t> data;
    inputs.push_back(gen_data({i, 7777, 9999}, &data));
    batch_size += i;
    output_data.insert(output_data.end(), data.begin(), data.end());
  }

  xla::ifrt::Shape output_shape({batch_size, 7777, 9999});

  TF_ASSERT_OK_AND_ASSIGN(auto output,
                          Concat(&thread_pool, inputs, allocator_));
  EXPECT_THAT(output,
              HostBufferEq(AsHostBuffer<int32_t>(output_data, output_shape)));
}

TEST_F(HostBufferManipulationTest, ConcatLargeBufferWithSmallType) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "concat_test", 5);
  auto gen_data = [](absl::Span<const int64_t> shape,
                     std::vector<int32_t>* data) -> HostBuffer {
    xla::ifrt::Shape input_shape(shape);
    data->resize(input_shape.num_elements());
    std::iota(data->begin(), data->end(), 0);
    HostBuffer host_buffer =
        AsHostBuffer<tsl::int4, int32_t>(*data, input_shape);
    return host_buffer;
  };
  std::vector<int32_t> data1, data2, data3;
  std::vector<HostBuffer> inputs;
  std::vector<int32_t> output_data;
  int batch_size = 0;
  for (int i = 0; i < 3; ++i) {
    std::vector<int32_t> data;
    inputs.push_back(gen_data({i, 7777, 9999}, &data));
    batch_size += i;
    output_data.insert(output_data.end(), data.begin(), data.end());
  }

  xla::ifrt::Shape output_shape({batch_size, 7777, 9999});

  TF_ASSERT_OK_AND_ASSIGN(auto output,
                          Concat(&thread_pool, inputs, allocator_));
  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<tsl::int4, int32_t>(
                          output_data, output_shape)));
}

TEST_F(HostBufferManipulationTest, ConcatLargeBufferWithFloatType) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "concat_test", 5);
  auto gen_data = [](absl::Span<const int64_t> shape,
                     std::vector<float>* data) -> HostBuffer {
    xla::ifrt::Shape input_shape(shape);
    data->resize(input_shape.num_elements());
    std::iota(data->begin(), data->end(), 0);
    HostBuffer host_buffer = AsHostBuffer<float>(*data, input_shape);
    return host_buffer;
  };
  std::vector<float> data1, data2, data3;
  std::vector<HostBuffer> inputs;
  std::vector<float> output_data;
  int batch_size = 0;
  for (int i = 0; i < 3; ++i) {
    std::vector<float> data;
    inputs.push_back(gen_data({i, 777, 999}, &data));
    batch_size += i;
    output_data.insert(output_data.end(), data.begin(), data.end());
  }

  xla::ifrt::Shape output_shape({batch_size, 777, 999});

  TF_ASSERT_OK_AND_ASSIGN(auto output,
                          Concat(&thread_pool, inputs, allocator_));
  EXPECT_THAT(output,
              HostBufferEq(AsHostBuffer<float>(output_data, output_shape)));
}

void BM_Concat(benchmark::State& state) {
  std::vector<HostBuffer> inputs;
  auto gen_buffer = [](absl::Span<const int64_t> shape) -> HostBuffer {
    xla::ifrt::Shape input_shape(shape);
    std::vector<int32_t> data;
    data.resize(input_shape.num_elements());
    std::iota(data.begin(), data.end(), 0);
    HostBuffer host_buffer = AsHostBuffer<int32_t>(data, input_shape);
    return host_buffer;
  };
  for (int i = 0; i < 3; ++i) {
    inputs.push_back(gen_buffer({i, 777, 999}));
  }
  for (auto s : state) {
    CHECK_OK(Concat(nullptr, inputs, DefaultAllocator::GetInstance()));
  }
}

BENCHMARK(BM_Concat);

void BM_ConcatParallel(benchmark::State& state) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "concat_test", 5);
  std::vector<HostBuffer> inputs;
  auto gen_buffer = [](absl::Span<const int64_t> shape) -> HostBuffer {
    xla::ifrt::Shape input_shape(shape);
    std::vector<int32_t> data;
    data.resize(input_shape.num_elements());
    std::iota(data.begin(), data.end(), 0);
    HostBuffer host_buffer = AsHostBuffer<int32_t>(data, input_shape);
    return host_buffer;
  };
  for (int i = 0; i < 3; ++i) {
    inputs.push_back(gen_buffer({i, 777, 999}));
  }
  for (auto s : state) {
    CHECK_OK(Concat(&thread_pool, inputs, DefaultAllocator::GetInstance()));
  }
}

BENCHMARK(BM_ConcatParallel);

TEST_F(HostBufferManipulationTest, ConcatSmallType) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<tsl::int4, int32_t>(
      {3, 4, 5, 6}, xla::ifrt::Shape({1, 2, 2})));
  inputs.push_back(AsHostBuffer<tsl::int4, int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({2, 2, 2})));
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));
  EXPECT_THAT(output.dtype(), xla::ifrt::DType(xla::ifrt::DType::Kind::kS4));
  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<tsl::int4, int32_t>(
                          {3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8},
                          xla::ifrt::Shape({3, 2, 2}))));
}

TEST_F(HostBufferManipulationTest, ConcatString) {
  tsl::tstring str_data1[6] = {"aaa", "bbb", "ccc", "ddd", "eee", "fff"};
  tsl::tstring str_data2[4] = {"ggg", "hhh", "iii", "jjjj"};
  tsl::tstring str_data3[10] = {"aaa", "bbb", "ccc", "ddd", "eee",
                                "fff", "ggg", "hhh", "iii", "jjjj"};
  HostBuffer host_buffer1(str_data1,
                          xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                          xla::ifrt::Shape({3, 2}));
  HostBuffer host_buffer2(str_data2,
                          xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                          xla::ifrt::Shape({2, 2}));
  HostBuffer expected_host_buffer(
      str_data3, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({5, 2}));
  std::vector<HostBuffer> inputs;
  inputs.push_back(host_buffer1);
  inputs.push_back(host_buffer2);
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));
  EXPECT_THAT(output, HostBufferEq(expected_host_buffer));
}

TEST_F(HostBufferManipulationTest, ConcatFailedOnEmptyInput) {
  EXPECT_THAT(Concat(nullptr, {}, allocator_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, ConcatFailedOnDifferentRankInputs) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})));
  inputs.push_back(AsHostBuffer<int32_t>({3, 4, 5, 6}, xla::ifrt::Shape({4})));
  inputs.push_back(AsHostBuffer<int32_t>({7, 8, 9, 10, 11, 12},
                                         xla::ifrt::Shape({1, 2, 3})));
  EXPECT_THAT(Concat(nullptr, inputs, allocator_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, ConcatFailedOnDifferentDimensionsInputs) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})));
  inputs.push_back(
      AsHostBuffer<int32_t>({3, 4, 5, 6}, xla::ifrt::Shape({1, 4})));
  EXPECT_THAT(Concat(nullptr, inputs, allocator_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferSplitEasyCasesTest, SplitEasyCasesSizeOne) {
  HostBuffer host_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({4, 2}));
  std::vector<HostBuffer> output;
  bool easy_cases_done = false;
  ASSERT_OK(SplitEasyCases(&host_buffer, {4}, output, easy_cases_done));
  EXPECT_TRUE(easy_cases_done);
  EXPECT_THAT(output, ElementsAre(HostBufferEq(host_buffer)));
  EXPECT_EQ(output[0].raw_data(),
            host_buffer.raw_data());
}

TEST_F(HostBufferSplitEasyCasesTest, SplitEasyCasesAligned) {
  // Shape {4, 16} with int32_t. Inner dim size is 16 * sizeof(int32_t) = 64
  // bytes. This is typically aligned with EIGEN_MAX_ALIGN_BYTES.
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
       33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
       49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64},
      xla::ifrt::Shape({4, 16}));
  std::vector<HostBuffer> output;
  bool easy_cases_done = false;
  ASSERT_OK(SplitEasyCases(&host_buffer, {1, 2, 1}, output, easy_cases_done));
  EXPECT_TRUE(easy_cases_done);

  HostBuffer expected1 = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({1, 16}));
  HostBuffer expected2 = AsHostBuffer<int32_t>(
      {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
       33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
      xla::ifrt::Shape({2, 16}));
  HostBuffer expected3 = AsHostBuffer<int32_t>(
      {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64},
      xla::ifrt::Shape({1, 16}));

  EXPECT_THAT(output,
              ElementsAre(HostBufferEq(expected1), HostBufferEq(expected2),
                          HostBufferEq(expected3)));
  EXPECT_EQ(output[0].base(),
            host_buffer.base());
  EXPECT_EQ(output[1].base(),
            static_cast<const int32_t*>(host_buffer.base()) +
                16);
  EXPECT_EQ(output[2].base(),
            static_cast<const int32_t*>(host_buffer.base()) +
                (16 * 3));
}

TEST_F(HostBufferSplitEasyCasesTest, SplitEasyCasesAlignedForString) {
  // Shape {4, 8} with each element being a string of length 4.
  tsl::tstring str_data[32] = {
      "aaaa", "bbbb",  "cccc",  "dddd", "eeee", "ffff",  "gggg", "hhhh",
      "iiii", "jjjjj", "kkkkk", "llll", "mmmm", "nnnn",  "oooo", "pppp",
      "qqqq", "rrrr",  "ssss",  "tttt", "uuuu", "vvvvv", "wwww", "xxxx",
      "yyyy", "zzzz",  "aaaa",  "bbbb", "cccc", "dddd",  "eeee", "ffff",
  };
  HostBuffer host_buffer(str_data,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                         xla::ifrt::Shape({4, 8}));

  std::vector<int64_t> split_sizes = {1, 2, 1};
  std::vector<HostBuffer> output;
  bool easy_cases_done = false;
  ASSERT_OK(SplitEasyCases(&host_buffer, split_sizes, output, easy_cases_done));

  EXPECT_TRUE(easy_cases_done);
  tsl::tstring expected_str_1[8] = {
      "aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh",
  };
  HostBuffer expected_output_1(
      expected_str_1, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({1, 8}));
  tsl::tstring expected_str_2[16] = {
      "iiii", "jjjjj", "kkkkk", "llll", "mmmm", "nnnn",  "oooo", "pppp",
      "qqqq", "rrrr",  "ssss",  "tttt", "uuuu", "vvvvv", "wwww", "xxxx",
  };
  HostBuffer expected_output_2(
      expected_str_2, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({2, 8}));
  tsl::tstring expected_str_3[8] = {
      "yyyy", "zzzz", "aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff",
  };
  HostBuffer expected_output_3(
      expected_str_3, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({1, 8}));
  EXPECT_THAT(output, ElementsAre(HostBufferEq(expected_output_1),
                                  HostBufferEq(expected_output_2),
                                  HostBufferEq(expected_output_3)));
}

TEST_F(HostBufferSplitEasyCasesTest, SplitEasyCasesAlignedSubByteTypesU4) {
  // Shape {4, EIGEN_MAX_ALIGN_BYTES} with kU4.
  // Inner dim size is EIGEN_MAX_ALIGN_BYTES which should always be aligned.
  std::vector<int> u4_data;
  // NOLINTNEXTLINE(misc-include-cleaner)
  for (int i = 0; i < 4 * EIGEN_MAX_ALIGN_BYTES; ++i) {
    u4_data.push_back(i % 16);
  }

  // Create a host buffer with second row EIGEN_MAX_ALIGN_BYTES to ensure
  // sub-byte types are handled correctly.
  HostBuffer host_buffer = AsHostBuffer<tsl::uint4, int>(
      u4_data, xla::ifrt::Shape({4, EIGEN_MAX_ALIGN_BYTES}));

  std::vector<HostBuffer> output;
  bool easy_cases_done = false;
  ASSERT_OK(SplitEasyCases(&host_buffer, {1, 2, 1}, output, easy_cases_done));
  EXPECT_TRUE(easy_cases_done);

  // Expected splits are created from slices of the original u2_data.
  HostBuffer expected1 = AsHostBuffer<tsl::uint4, int>(
      std::vector<int>(u4_data.begin(),
                       u4_data.begin() + EIGEN_MAX_ALIGN_BYTES),
      xla::ifrt::Shape({1, EIGEN_MAX_ALIGN_BYTES}));
  HostBuffer expected2 = AsHostBuffer<tsl::uint4, int>(
      std::vector<int>(u4_data.begin() + EIGEN_MAX_ALIGN_BYTES,
                       u4_data.begin() + (EIGEN_MAX_ALIGN_BYTES * 3)),
      xla::ifrt::Shape({2, EIGEN_MAX_ALIGN_BYTES}));
  HostBuffer expected3 = AsHostBuffer<tsl::uint4, int>(
      std::vector<int>(u4_data.begin() + (EIGEN_MAX_ALIGN_BYTES * 3),
                       u4_data.begin() + (EIGEN_MAX_ALIGN_BYTES * 4)),
      xla::ifrt::Shape({1, EIGEN_MAX_ALIGN_BYTES}));

  EXPECT_THAT(output,
              ElementsAre(HostBufferEq(expected1), HostBufferEq(expected2),
                          HostBufferEq(expected3)));
  EXPECT_EQ(output[0].base(),
            host_buffer.base());
  EXPECT_EQ(output[1].base(),
            static_cast<const char*>(host_buffer.base()) +
                EIGEN_MAX_ALIGN_BYTES);
  EXPECT_EQ(output[2].base(),
            static_cast<const char*>(host_buffer.base()) +
                (EIGEN_MAX_ALIGN_BYTES * 3));
}

TEST_F(HostBufferSplitEasyCasesTest, SplitEasyCasesNotEasy) {
  // Shape {3, 3} with int32_t. Inner dim size is 3 * 4 = 12 bytes.
  // This is not aligned to EIGEN_MAX_ALIGN_BYTES (typically 16 or 32).
  HostBuffer host_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                 xla::ifrt::Shape({3, 3}));
  std::vector<HostBuffer> output;
  bool easy_cases_done = false;
  ASSERT_OK(SplitEasyCases(&host_buffer, {1, 2}, output, easy_cases_done));
  EXPECT_FALSE(easy_cases_done);
  EXPECT_TRUE(output.empty());
}

TEST(HostBufferProtoTest, ToAndFromProtoNumeric) {
  HostBuffer original_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({2, 4}));

  HostBufferProto proto =
      HostBuffer::ToHostBufferProto(std::move(original_buffer));

  // Verify the contents of the intermediate proto.
  EXPECT_EQ(proto.dtype().kind(), xla::ifrt::DType::Kind::kS32);
  EXPECT_THAT(proto.shape().dims(), ElementsAre(2, 4));
  std::vector<int32_t> expected_data = {1, 2, 3, 4, 5, 6, 7, 8};
  absl::string_view expected_data_bytes(
      reinterpret_cast<const char*>(expected_data.data()),
      expected_data.size() * sizeof(int32_t));
  EXPECT_EQ(proto.raw_data(), expected_data_bytes);

  TF_ASSERT_OK_AND_ASSIGN(HostBuffer new_buffer, HostBuffer::FromProto(proto));

  HostBuffer expected_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({2, 4}));

  EXPECT_THAT(new_buffer, HostBufferEq(expected_buffer));
  EXPECT_EQ(new_buffer.dtype(), expected_buffer.dtype());
  EXPECT_EQ(new_buffer.shape(), expected_buffer.shape());
}

TEST(HostBufferProtoTest, ToAndFromProtoString) {
  tsl::tstring str_data[4] = {"hello", "world", "test", "string"};
  HostBuffer original_buffer(str_data,
                             xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                             xla::ifrt::Shape({2, 2}));

  HostBufferProto proto =
      HostBuffer::ToHostBufferProto(std::move(original_buffer));

  // Verify the contents of the intermediate proto.
  EXPECT_EQ(proto.dtype().kind(), xla::ifrt::DType::Kind::kString);
  EXPECT_THAT(proto.shape().dims(), ElementsAre(2, 2));
  EXPECT_THAT(proto.string_data().elements(),
              ElementsAre("hello", "world", "test", "string"));

  TF_ASSERT_OK_AND_ASSIGN(HostBuffer new_buffer, HostBuffer::FromProto(proto));

  HostBuffer expected_buffer(str_data,
                             xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                             xla::ifrt::Shape({2, 2}));

  EXPECT_THAT(new_buffer, HostBufferEq(expected_buffer));
  EXPECT_EQ(new_buffer.dtype(), expected_buffer.dtype());
  EXPECT_EQ(new_buffer.shape(), expected_buffer.shape());
}

TEST(HostBufferProtoTest, ToAndFromProtoNonUtf8String) {
  // "\xFF\xFF" is invalid UTF-8.
  tsl::tstring str_data[4] = {"hello", "world", "\xFF\xFF", "string"};
  HostBuffer original_buffer(str_data,
                             xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                             xla::ifrt::Shape({2, 2}));

  HostBufferProto proto =
      HostBuffer::ToHostBufferProto(std::move(original_buffer));

  std::string serialized;
  ASSERT_TRUE(proto.SerializeToString(&serialized));

  HostBufferProto parsed_proto;
  ASSERT_TRUE(parsed_proto.ParseFromString(serialized));

  EXPECT_EQ(parsed_proto.dtype().kind(), xla::ifrt::DType::Kind::kString);
  EXPECT_THAT(parsed_proto.shape().dims(), ElementsAre(2, 2));
  EXPECT_THAT(parsed_proto.string_data().elements(),
              ElementsAre("hello", "world", "\xFF\xFF", "string"));

  TF_ASSERT_OK_AND_ASSIGN(HostBuffer new_buffer,
                          HostBuffer::FromProto(parsed_proto));

  HostBuffer expected_buffer(str_data,
                             xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                             xla::ifrt::Shape({2, 2}));

  EXPECT_THAT(new_buffer, HostBufferEq(expected_buffer));
}

TEST(HostBufferProtoTest, FromProtoNonEmptyShapeWithEmptyDataReturnsError) {
  HostBufferProto proto;
  *proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kS32).ToProto();
  proto.mutable_shape()->add_dims(2);
  proto.mutable_shape()->add_dims(2);

  // No raw_data or string_data is set.
  EXPECT_THAT(HostBuffer::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostBufferProtoTest, FromProtoShapeOverflowReturnsError) {
  HostBufferProto proto;
  *proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kS32).ToProto();
  // Multiplied dimensions overflow int64_t
  proto.mutable_shape()->add_dims(2000000000LL);  // 2*10^9
  proto.mutable_shape()->add_dims(5000000000LL);  // 5*10^9
  // Product is 10^19, which is larger than INT64_MAX (approx 9.22 * 10^18).

  EXPECT_THAT(HostBuffer::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostBufferProtoTest, FromProtoByteSizeOverflowReturnsError) {
  HostBufferProto proto;
  *proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kString).ToProto();
  // 24 * num_elements overflows.
  // Let's choose num_elements = 2^62.
  proto.mutable_shape()->add_dims(4611686018427387904LL);  // 2^62

  EXPECT_THAT(HostBuffer::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostBufferProtoTest, FromProtoShapeTooLargeReturnsError) {
  HostBufferProto proto;
  *proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kS32).ToProto();
  // 1 << 32 + 1 elements.
  proto.mutable_shape()->add_dims((int64_t{1} << 32) + 1);

  EXPECT_THAT(HostBuffer::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostBufferProtoTest, FromProtoShapeTooLargeWithCustomLimit) {
  // Set limit to a small value for testing.
  absl::SetFlag(&FLAGS_xla_serving_host_buffer_max_elements, 10);
  absl::Cleanup reset_flag = [] {
    absl::SetFlag(&FLAGS_xla_serving_host_buffer_max_elements,
                  int64_t{1} << 32);
  };

  HostBufferProto proto;
  *proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kS32).ToProto();
  proto.mutable_shape()->add_dims(11);

  EXPECT_THAT(HostBuffer::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostBufferProtoTest, FromProtoByteSizeOverflowForStringReturnsError) {
  // Set limit to a large value to allow shape to pass limit check.
  absl::SetFlag(&FLAGS_xla_serving_host_buffer_max_elements, int64_t{1} << 62);
  absl::Cleanup reset_flag = [] {
    absl::SetFlag(&FLAGS_xla_serving_host_buffer_max_elements,
                  int64_t{1} << 32);
  };

  HostBufferProto proto;
  *proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kString).ToProto();
  // 24 * num_elements overflows.
  // Let's choose num_elements = 2^62.
  proto.mutable_shape()->add_dims(4611686018427387904LL);  // 2^62

  EXPECT_THAT(HostBuffer::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

class RecordingAllocator : public Allocator {
 public:
  explicit RecordingAllocator(bool fail_allocations = false)
      : fail_allocations_(fail_allocations) {}

  void* Allocate(size_t size, size_t alignment) override {
    ++allocation_count_;
    last_requested_bytes_ = size;
    if (fail_allocations_) {
      return nullptr;
    }
    return DefaultAllocator::GetInstance()->Allocate(size, alignment);
  }

  void Free(void* ptr, size_t alignment) override {
    DefaultAllocator::GetInstance()->Free(ptr, alignment);
  }

  int allocation_count() const { return allocation_count_; }
  size_t last_requested_bytes() const { return last_requested_bytes_; }

 private:
  bool fail_allocations_ = false;
  int allocation_count_ = 0;
  size_t last_requested_bytes_ = 0;
};

TEST(HostBufferProtoTest, FromProtoMismatchedRawDataDoesNotAllocate) {
  // Regression test: a tiny proto claiming max_elements (2^32 f32 = 16 GiB)
  // with mismatched raw_data must be rejected before allocating memory.
  HostBufferProto proto;
  *proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kF32).ToProto();
  proto.mutable_shape()->add_dims(int64_t{1} << 32);
  proto.set_raw_data("abcd");

  RecordingAllocator allocator(/*fail_allocations=*/true);
  EXPECT_THAT(HostBuffer::FromProto(proto, &allocator),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(allocator.allocation_count(), 0);
  EXPECT_THAT(HostBuffer::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostBufferProtoTest, FromProtoMismatchedStringDataDoesNotAllocate) {
  // Regression test: a tiny proto claiming max_elements (2^32 tstring = 96 GiB)
  // with mismatched string_data must be rejected before allocating memory.
  HostBufferProto proto;
  *proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kString).ToProto();
  proto.mutable_shape()->add_dims(int64_t{1} << 32);
  proto.mutable_string_data()->add_elements("x");

  RecordingAllocator allocator(/*fail_allocations=*/true);
  EXPECT_THAT(HostBuffer::FromProto(proto, &allocator),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(allocator.allocation_count(), 0);
  EXPECT_THAT(HostBuffer::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostBufferProtoTest, FromProtoMissingPayloadWithMaxShapeDoesNotAllocate) {
  HostBufferProto f32_proto;
  *f32_proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kF32).ToProto();
  f32_proto.mutable_shape()->add_dims(int64_t{1} << 32);

  RecordingAllocator allocator(/*fail_allocations=*/true);
  EXPECT_THAT(HostBuffer::FromProto(f32_proto, &allocator),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(allocator.allocation_count(), 0);

  HostBufferProto str_proto;
  *str_proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kString).ToProto();
  str_proto.mutable_shape()->add_dims(int64_t{1} << 32);

  EXPECT_THAT(HostBuffer::FromProto(str_proto, &allocator),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(allocator.allocation_count(), 0);
}

TEST(HostBufferProtoTest, FromProtoWrongDataVariantDoesNotAllocate) {
  HostBufferProto str_with_raw;
  *str_with_raw.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kString).ToProto();
  str_with_raw.mutable_shape()->add_dims(int64_t{1} << 32);
  str_with_raw.set_raw_data("abcd");

  RecordingAllocator allocator(/*fail_allocations=*/true);
  EXPECT_THAT(HostBuffer::FromProto(str_with_raw, &allocator),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(allocator.allocation_count(), 0);

  HostBufferProto f32_with_strings;
  *f32_with_strings.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kF32).ToProto();
  f32_with_strings.mutable_shape()->add_dims(int64_t{1} << 32);
  f32_with_strings.mutable_string_data()->add_elements("abcd");

  EXPECT_THAT(HostBuffer::FromProto(f32_with_strings, &allocator),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(allocator.allocation_count(), 0);
}

TEST(HostBufferProtoTest, FromProtoUnsupportedDTypeDoesNotAllocate) {
  RecordingAllocator allocator(/*fail_allocations=*/true);

  // Unspecified dtype deserializes to kInvalid in xla::ifrt::DType::FromProto.
  HostBufferProto unspecified_dtype_proto;
  unspecified_dtype_proto.mutable_shape()->add_dims(1);
  unspecified_dtype_proto.set_raw_data("x");
  EXPECT_THAT(HostBuffer::FromProto(unspecified_dtype_proto, &allocator),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(allocator.allocation_count(), 0);

  HostBufferProto token_proto;
  *token_proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kToken).ToProto();
  token_proto.mutable_shape()->add_dims(0);
  EXPECT_THAT(HostBuffer::FromProto(token_proto, &allocator),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(allocator.allocation_count(), 0);
}

TEST(HostBufferProtoTest, ToAndFromProtoZeroElements) {
  HostBuffer empty_f32(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                       xla::ifrt::Shape({0, 4}));
  HostBufferProto f32_proto = HostBuffer::ToHostBufferProto(empty_f32);
  TF_ASSERT_OK_AND_ASSIGN(HostBuffer roundtrip_f32,
                          HostBuffer::FromProto(f32_proto));
  EXPECT_EQ(roundtrip_f32.dtype(), empty_f32.dtype());
  EXPECT_EQ(roundtrip_f32.shape(), empty_f32.shape());
  EXPECT_TRUE(roundtrip_f32.raw_data().empty());

  HostBuffer empty_str(xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                       xla::ifrt::Shape({0, 2}));
  HostBufferProto str_proto = HostBuffer::ToHostBufferProto(empty_str);
  TF_ASSERT_OK_AND_ASSIGN(HostBuffer roundtrip_str,
                          HostBuffer::FromProto(str_proto));
  EXPECT_EQ(roundtrip_str.dtype(), empty_str.dtype());
  EXPECT_EQ(roundtrip_str.shape(), empty_str.shape());
}

TEST(HostBufferTest, CreateValidatesShapesAndAllocationFailures) {
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer valid,
      HostBuffer::Create(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                         xla::ifrt::Shape({2, 3})));
  EXPECT_EQ(valid.shape(), xla::ifrt::Shape({2, 3}));
  EXPECT_EQ(valid.total_storage_bytes(), 24);

  EXPECT_THAT(HostBuffer::Create(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                                 xla::ifrt::Shape({2000000LL, 5000000LL})),
              StatusIs(absl::StatusCode::kInvalidArgument));

  RecordingAllocator failing_allocator(/*fail_allocations=*/true);
  EXPECT_THAT(HostBuffer::Create(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                                 xla::ifrt::Shape({4}), &failing_allocator),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(failing_allocator.allocation_count(), 1);
  EXPECT_EQ(failing_allocator.last_requested_bytes(), 16);

  HostBufferProto valid_proto = HostBuffer::ToHostBufferProto(valid);
  EXPECT_THAT(HostBuffer::FromProto(valid_proto, &failing_allocator),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(HostBufferTest, ConcatAndSplitRejectInvalidOrTooLargeSizes) {
  HostBuffer a(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
               xla::ifrt::Shape({6}));
  HostBuffer b(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
               xla::ifrt::Shape({6}));

  // Lower max_elements so Concat's output shape ({12}) exceeds the limit.
  {
    absl::SetFlag(&FLAGS_xla_serving_host_buffer_max_elements, 10);
    absl::Cleanup reset_flag = [] {
      absl::SetFlag(&FLAGS_xla_serving_host_buffer_max_elements,
                    int64_t{1} << 32);
    };
    EXPECT_THAT(Concat(/*threadpool=*/nullptr, {a, b}),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Negative and overflowing split sizes must return InvalidArgumentError.
  EXPECT_THAT(a.SplitAlongFirstDimension({-1, 7}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      a.SplitAlongFirstDimension({2, std::numeric_limits<int64_t>::max()}),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostBufferTest, ShapedRejectsNegativeAndOverflowingDimensions) {
  HostBuffer host_buffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({4, 4}));
  const HostBuffer& const_host_buffer = host_buffer;

  // Negative dimensions with an even count whose product matches 16 must fail.
  EXPECT_THAT((host_buffer.shaped<int32_t, 2>({-2, -8})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("negative dimension")));
  EXPECT_THAT((const_host_buffer.shaped<int32_t, 2>({-2, -8})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("negative dimension")));

  // Dimensions whose product overflows int64_t must fail cleanly.
  EXPECT_THAT((host_buffer.shaped<int32_t, 2>({2000000000LL, 5000000000LL})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Shape overflow")));
  EXPECT_THAT(
      (const_host_buffer.shaped<int32_t, 2>({2000000000LL, 5000000000LL})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Shape overflow")));
}

TEST(HostBufferTest, Reshape) {
  HostBuffer orig(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                  xla::ifrt::Shape({6}));

  // Reshape from {6} -> {2, 3}
  xla::ifrt::Shape shape_2_3({2, 3});
  absl::StatusOr<HostBuffer> reshaped = orig.Reshape(shape_2_3);
  ASSERT_TRUE(reshaped.ok());
  EXPECT_EQ(reshaped->shape(), shape_2_3);
  EXPECT_EQ(reshaped->base(), orig.base());

  // Reshape back from {2, 3} -> {6}
  xla::ifrt::Shape shape_6({6});
  absl::StatusOr<HostBuffer> reshaped_back = reshaped->Reshape(shape_6);
  ASSERT_TRUE(reshaped_back.ok());
  EXPECT_EQ(reshaped_back->shape(), shape_6);

  // Test size mismatch failure
  xla::ifrt::Shape mismatch_shape({5});
  EXPECT_FALSE(orig.Reshape(mismatch_shape).ok());

  // Test uninitialized buffer failure
  HostBuffer empty;
  EXPECT_FALSE(empty.Reshape(shape_2_3).ok());
}

TEST(HostBufferTest, ReshapeLifetimeAndOwnership) {
  HostBuffer reshaped;
  xla::ifrt::Shape target_shape({2, 3});
  {
    HostBuffer orig(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                    xla::ifrt::Shape({6}));
    // Initialize buffer with test values.
    orig.flat<float>().setValues({0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f});

    absl::StatusOr<HostBuffer> status_or_reshaped = orig.Reshape(target_shape);
    ASSERT_TRUE(status_or_reshaped.ok());
    reshaped = *std::move(status_or_reshaped);
    // orig goes out of scope and is destroyed here.
  }

  EXPECT_EQ(reshaped.shape(), target_shape);
  TTypes<float, 2>::Tensor matrix = reshaped.tensor<float, 2>();
  EXPECT_EQ(matrix(0, 0), 0.0f);
  EXPECT_EQ(matrix(0, 1), 10.0f);
  EXPECT_EQ(matrix(1, 2), 50.0f);
}

TEST(HostBufferTest, Bitcast) {
  // 4 elements of uint8_t = 4 bytes
  HostBuffer orig(xla::ifrt::DType(xla::ifrt::DType::Kind::kU8),
                  xla::ifrt::Shape({4}));

  // Bitcast 4 x uint8 (4 bytes) -> 1 x uint32 (4 bytes)
  xla::ifrt::DType u32_dtype(xla::ifrt::DType::Kind::kU32);
  xla::ifrt::Shape target_shape({1});
  absl::StatusOr<HostBuffer> bitcasted = orig.Bitcast(u32_dtype, target_shape);
  ASSERT_TRUE(bitcasted.ok());
  EXPECT_EQ(bitcasted->dtype(), u32_dtype);
  EXPECT_EQ(bitcasted->shape(), target_shape);
  EXPECT_EQ(bitcasted->base(), orig.base());

  // Test byte size mismatch (4 bytes -> 8 bytes)
  xla::ifrt::Shape mismatch_shape({2});
  EXPECT_FALSE(orig.Bitcast(u32_dtype, mismatch_shape).ok());

  // Test dtype without fixed byte size failure (e.g. kString)
  xla::ifrt::DType string_dtype(xla::ifrt::DType::Kind::kString);
  EXPECT_FALSE(orig.Bitcast(string_dtype, target_shape).ok());

  // Test uninitialized buffer failure
  HostBuffer empty;
  EXPECT_FALSE(empty.Bitcast(u32_dtype, target_shape).ok());
}

TEST(HostBufferTest, BitcastLifetimeAndOwnership) {
  HostBuffer bitcasted;
  xla::ifrt::Shape target_shape({2, 3});
  xla::ifrt::DType u32_dtype(xla::ifrt::DType::Kind::kU32);
  {
    HostBuffer orig(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                    xla::ifrt::Shape({6}));
    // Initialize buffer with test values.
    orig.flat<int32_t>().setValues({0, 10, 20, 30, 40, 50});

    absl::StatusOr<HostBuffer> status_or_bitcast =
        orig.Bitcast(u32_dtype, target_shape);
    ASSERT_TRUE(status_or_bitcast.ok());
    bitcasted = *std::move(status_or_bitcast);
    // orig goes out of scope and is destroyed here.
  }

  EXPECT_EQ(bitcasted.dtype(), u32_dtype);
  EXPECT_EQ(bitcasted.shape(), target_shape);
  TTypes<uint32_t, 2>::Tensor matrix = bitcasted.tensor<uint32_t, 2>();
  EXPECT_EQ(matrix(0, 0), 0);
  EXPECT_EQ(matrix(0, 1), 10);
  EXPECT_EQ(matrix(1, 2), 50);
}

TEST(HostBufferTest, BitcastSubByteDTypeReturnsError) {
  HostBuffer orig(xla::ifrt::DType(xla::ifrt::DType::Kind::kU8),
                  xla::ifrt::Shape({4}));

  // Target sub-byte type (kS4, 4-bit) returns error.
  xla::ifrt::DType s4_dtype(xla::ifrt::DType::Kind::kS4);
  EXPECT_FALSE(orig.Bitcast(s4_dtype, xla::ifrt::Shape({4})).ok());

  // Source sub-byte type (kS4) returns error.
  HostBuffer s4_buffer = AsHostBuffer<tsl::int4, int32_t>(
      {1, 2, 3, 4}, xla::ifrt::Shape({4}));
  EXPECT_FALSE(s4_buffer
                   .Bitcast(xla::ifrt::DType(xla::ifrt::DType::Kind::kU8),
                            xla::ifrt::Shape({4}))
                   .ok());
}

}  // namespace
}  // namespace xla_serving
