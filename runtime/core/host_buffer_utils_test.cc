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

#include "runtime/core/host_buffer_utils.h"  // from @xla_serving

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/log_severity.h"  // from @abseil-cpp
#include "absl/log/scoped_mock_log.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "tsl/platform/bfloat16.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {
namespace {

using ::absl::ScopedMockLog;
using ::testing::_;
using ::testing::AtMost;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;

TEST(CreateDefaultStridesTest, TokenDTypeReturnsNullopt) {
  xla::ifrt::DType token_dtype(xla::ifrt::DType::Kind::kToken);

  ScopedMockLog log;
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("DType TOKEN has no bit size.")))
      .Times(AtMost(1));
  log.StartCapturingLogs();

  EXPECT_THAT(CreateDefaultStrides(/*dims=*/{2, 3}, /*dtype=*/token_dtype),
              Eq(std::nullopt));
}

TEST(CreateDefaultStridesTest, StringDType) {
  xla::ifrt::DType string_dtype(xla::ifrt::DType::Kind::kString);
  EXPECT_THAT(
      CreateDefaultStrides(/*dims=*/{2, 3}, /*dtype=*/string_dtype).value(),
      ElementsAre(72, 24));
}

TEST(CreateDefaultStridesTest, S32DType) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kS32);
  EXPECT_THAT(CreateDefaultStrides(/*dims=*/{2, 3, 4}, /*dtype=*/dtype).value(),
              ElementsAre(48, 16, 4));
}

TEST(CreateDefaultStridesTest, Bf16DType) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kBF16);
  EXPECT_THAT(CreateDefaultStrides(/*dims=*/{2, 2}, /*dtype=*/dtype).value(),
              ElementsAre(4, 2));
}

TEST(CreateDefaultStridesTest, F16DType) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kF16);
  EXPECT_THAT(CreateDefaultStrides(/*dims=*/{2, 2}, /*dtype=*/dtype).value(),
              ElementsAre(4, 2));
}

TEST(AppendRecursiveTest, ScalarS32) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kS32);
  int32_t data = 10;
  std::string out;
  AppendRecursive(&out, dtype, &data, 0, {}, {});
  EXPECT_THAT(out, Eq("(10)"));
}

TEST(AppendRecursiveTest, 1DF16) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kF16);
  Eigen::half data[] = {Eigen::half(1.5f), Eigen::half(2.5f)};
  std::string out;
  AppendRecursive(&out, dtype, data, 0, {2}, {2});
  EXPECT_THAT(out, Eq("[1.5, 2.5]"));
}

TEST(AppendRecursiveTest, 1DBf16) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kBF16);
  tsl::bfloat16 data[] = {tsl::bfloat16(1.5f), tsl::bfloat16(2.5f)};
  std::string out;
  AppendRecursive(&out, dtype, data, 0, {2}, {2});
  EXPECT_THAT(out, Eq("[1.5, 2.5]"));
}

TEST(AppendRecursiveTest, 2DS32) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kS32);
  int32_t data[] = {1, 2, 3, 4, 5, 6};
  std::string out;
  AppendRecursive(&out, dtype, data, 0, {12, 4}, {2, 3});
  EXPECT_THAT(out, Eq("[\n  [1, 2, 3],\n  [4, 5, 6]\n]"));
}

TEST(AppendRecursiveTest, 1DString) {
  xla::ifrt::DType dtype(xla::ifrt::DType::Kind::kString);
  tsl::tstring data[] = {"hello", "world"};
  std::string out;
  AppendRecursive(&out, dtype, data, 0, {24}, {2});
  EXPECT_THAT(out, Eq("['hello', 'world']"));
}

TEST(AppendRecursiveTest, 1DS32Truncated) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kS32);
  int32_t data[] = {1, 2, 3, 4, 5};
  std::string out;
  AppendRecursive(&out, dtype, data, 0, {4}, {5}, /*max_entries=*/2);
  EXPECT_THAT(out, Eq("[1, 2, ..., 4, 5]"));
}

TEST(AppendRecursiveTest, 2DS32SummarizeZero) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kS32);
  int32_t data[] = {1, 2, 3, 4, 5, 6};
  std::string out;
  AppendRecursive(&out, dtype, data, 0, {12, 4}, {2, 3}, /*max_entries=*/0);
  EXPECT_THAT(out, Eq("[\n  ...\n]"));
}

TEST(AppendRecursiveTest, 2DS32Truncated) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kS32);
  int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::string out;
  AppendRecursive(&out, dtype, data, 0, {16, 4}, {3, 4}, /*max_entries=*/1);
  EXPECT_THAT(out, Eq("[\n  [1, ..., 4],\n  ...,\n  [9, ..., 12]\n]"));
}

TEST(CreateDefaultStridesTest, FailsWithHugeDims) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kS32);
  std::vector<int64_t> dims = {1000000000, 1000000000, 1000000000};
  EXPECT_THAT(CreateDefaultStrides(dims, dtype), Eq(std::nullopt));
}

TEST(SummarizeBufferTest, ScalarS32) {
  int32_t val = 42;
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {}, &val, -1),
      Eq("(42)"));
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {}, &val, 0),
      Eq("(42)"));
}

TEST(SummarizeBufferTest, ScalarString) {
  tsl::tstring val = "hello";
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                              {}, &val, -1),
              Eq("('hello')"));
}

TEST(SummarizeBufferTest, ScalarBool) {
  bool val_true = true;
  bool val_false = false;
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kPred), {},
                              &val_true, -1),
              Eq("(1)"));
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kPred), {},
                              &val_false, -1),
              Eq("(0)"));
}

TEST(SummarizeBufferTest, ScalarFloat) {
  float val = 3.5f;
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kF32), {}, &val, -1),
      Eq("(3.5)"));
}

TEST(SummarizeBufferTest, ScalarF16) {
  Eigen::half val = Eigen::half(3.5f);
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kF16), {}, &val, -1),
      Eq("(3.5)"));
}

TEST(SummarizeBufferTest, OneDS32) {
  int32_t data[] = {1, 2, 3, 4, 5};
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {5}, data, -1),
      Eq("[1, 2, 3, 4, 5]"));
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {5}, data, 10),
      Eq("[1, 2, 3, 4, 5]"));
}

TEST(SummarizeBufferTest, OneDS32Truncated) {
  int32_t data[] = {1, 2, 3, 4, 5};
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {5}, data, 2),
      Eq("[1, 2, ..., 4, 5]"));
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {5}, data, 1),
      Eq("[1, ..., 5]"));
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {5}, data, 0),
      Eq("[...]"));
}

TEST(SummarizeBufferTest, OneDStrided) {
  int32_t data[] = {1, 999, 2, 999, 3};
  int64_t strides[] = {8};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32),
                              /*shape=*/{3}, data, /*max_entries=*/-1,
                              /*byte_strides=*/strides),
              Eq("[1, 2, 3]"));
}

TEST(SummarizeBufferTest, OneDStridedTruncated) {
  int32_t data[] = {1, 999, 2, 999, 3, 999, 4, 999, 5};
  int64_t strides[] = {8};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32),
                              /*shape=*/{5}, data, /*max_entries=*/2,
                              /*byte_strides=*/strides),
              Eq("[1, 2, ..., 4, 5]"));
}

TEST(SummarizeBufferTest, TwoDStrided) {
  int32_t data[] = {1, 2, 999, 3, 4, 999};
  int64_t strides[] = {12, 4};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32),
                              /*shape=*/{2, 2}, data, /*max_entries=*/-1,
                              /*byte_strides=*/strides),
              Eq("[\n  [1, 2],\n  [3, 4]\n]"));
}

TEST(SummarizeBufferTest, OneDFloatAndBf16) {
  float float_data[] = {1.0f, 2.5f};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kF32), {2},
                              float_data, -1),
              Eq("[1, 2.5]"));
  tsl::bfloat16 bf16_data[] = {tsl::bfloat16(1.5f), tsl::bfloat16(2.5f)};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kBF16), {2},
                              bf16_data, -1),
              Eq("[1.5, 2.5]"));
}

TEST(SummarizeBufferTest, OneDF16) {
  Eigen::half f16_data[] = {Eigen::half(1.5f), Eigen::half(2.5f)};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kF16), {2},
                              f16_data, -1),
              Eq("[1.5, 2.5]"));
}

TEST(SummarizeBufferTest, OneDString) {
  tsl::tstring data[] = {"hello", "world"};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                              {2}, data, -1),
              Eq("['hello', 'world']"));
}

TEST(SummarizeBufferTest, OneDStringTruncated) {
  tsl::tstring data[] = {"one", "two", "three", "four", "five"};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                              {5}, data, 1),
              Eq("['one', ..., 'five']"));
}

TEST(SummarizeBufferTest, OneDBool) {
  bool data[] = {false, true, true, false, true};
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kPred), {5}, data, -1),
      Eq("[0, 1, 1, 0, 1]"));
}

TEST(SummarizeBufferTest, TwoDS32) {
  int32_t data[] = {1, 2, 3, 4, 5, 6};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {2, 3},
                              data, -1),
              Eq("[\n  [1, 2, 3],\n  [4, 5, 6]\n]"));
}

TEST(SummarizeBufferTest, TwoDS32Truncated) {
  int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {3, 4},
                              data, 1),
              Eq("[\n  [1, ..., 4],\n  ...,\n  [9, ..., 12]\n]"));
}

TEST(SummarizeBufferTest, TwoDS32SummarizeZero) {
  int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {3, 4},
                              data, 0),
              Eq("[\n  ...\n]"));
}

TEST(SummarizeBufferTest, ThreeDS32) {
  int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32),
                              {2, 2, 2}, data, -1),
              Eq("[\n  [\n    [1, 2],\n    [3, 4]\n  ],\n  [\n    [5, 6],\n    "
                 "[7, 8]\n  ]\n]"));
}

TEST(SummarizeBufferTest, EmptyTensor) {
  // An empty tensor is shown as `[]` regardless of the value of `max_entries`.
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {0},
                              nullptr, -1),
              Eq("[]"));
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kF64), {0},
                              nullptr, 1),
              Eq("[]"));
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kString), {0},
                              nullptr, 0),
              Eq("[]"));
}

TEST(SummarizeBufferTest, UninitializedTensor) {
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {},
                              nullptr, -1),
              Eq("Uninitialized Buffer of 1 elements of type S32"));
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {}, nullptr, 0),
      Eq("Uninitialized Buffer of 1 elements of type S32"));
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32), {3},
                              nullptr, 1),
              Eq("Uninitialized Buffer of 3 elements of type S32"));
  EXPECT_THAT(
      SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32),
                      {1000000000, 1000000000, 1000000000}, nullptr, -1),
      Eq("Overflowing dimensions in uninitialized Buffer of type S32: "
         "[1000000000,1000000000,1000000000]"));
}

TEST(SummarizeBufferTest, WithoutDefaultStrides) {
  int32_t val = 1;
  EXPECT_THAT(SummarizeBuffer(xla::ifrt::DType(xla::ifrt::DType::kS32),
                              {1000000000, 1000000000, 1000000000}, &val, -1),
              Eq("Failed to create default strides for Buffer of type S32"));
}

TEST(SummarizeBufferTest, TokenDType) {
  xla::ifrt::DType token_dtype(xla::ifrt::DType::Kind::kToken);
  // Unsupported dtype results in a message in the output HostBuffer.
  EXPECT_THAT(SummarizeBuffer(token_dtype, {2}, nullptr, -1),
              Eq("Wrong dtype for Buffer: TOKEN"));
}

}  // namespace
}  // namespace xla_serving
