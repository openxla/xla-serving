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

#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving

#include <cmath>
#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "tsl/platform/ml_dtypes.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {
namespace {

using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::Not;
using ::testing::Matcher;

TEST(HostBufferMatcherTest, BasicPod) {
  std::vector<HostBuffer> expected;
  xla::ifrt::Shape shape({1});
  const int16_t in1 = 100;
  expected.push_back(AsHostBuffer<int16_t>({in1}, shape));
  const int16_t in2 = 16;
  expected.push_back(AsHostBuffer<int16_t>({in2}, shape));

  EXPECT_THAT(expected,
              ElementsAre(HostBufferEq(AsHostBuffer<int16_t>({in1}, shape)),
                          HostBufferEq(AsHostBuffer<int16_t>({in2}, shape))));
}

TEST(HostBufferMatcherTest, MismatchIntShallFail) {
  std::vector<HostBuffer> expected;
  xla::ifrt::Shape shape({1});
  const int16_t in1 = 100;
  const int16_t in2 = 16;

  ::testing::StringMatchResultListener listener;
  EXPECT_FALSE(
      Matcher<const HostBuffer&>(
          HostBufferEq(AsHostBuffer<int16_t>({in2}, shape)))
          .MatchAndExplain(AsHostBuffer<int16_t>({in1}, shape), &listener));
}

TEST(HostBufferMatcherTest, BasicPodArray) {
  std::vector<HostBuffer> expected;
  xla::ifrt::Shape shape({2, 2});
  const std::vector<int16_t> in1 = {100, 200, 300, 400};
  expected.push_back(AsHostBuffer<int16_t>(in1, shape));
  const std::vector<int16_t> in2 = {16, 32, 64, 128};
  expected.push_back(AsHostBuffer<int16_t>(in2, shape));

  EXPECT_THAT(expected,
              ElementsAre(HostBufferEq(AsHostBuffer<int16_t>(in1, shape)),
                          HostBufferEq(AsHostBuffer<int16_t>(in2, shape))));
}

TEST(HostBufferMatcherTest, Basic4Bit) {
  HostBuffer in1(xla::ifrt::DType(xla::ifrt::DType::Kind::kS4),
                 xla::ifrt::Shape({1}));
  in1.flat<tsl::int4>()(0) = static_cast<tsl::int4>(7);

  HostBuffer in2(xla::ifrt::DType(xla::ifrt::DType::Kind::kU4),
                 xla::ifrt::Shape({1}));
  in2.flat<tsl::uint4>()(0) = static_cast<tsl::uint4>(15);

  std::vector<HostBuffer> expected;
  expected.push_back(in1);
  expected.push_back(in2);

  EXPECT_THAT(expected, ElementsAre(HostBufferEq(in1), HostBufferEq(in2)));
}

TEST(HostBufferMatcherTest, MismatchStrings) {
  tsl::tstring str_data1[3] = {"aaa", "bbb", "ccc"};
  HostBuffer in1(str_data1, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                 xla::ifrt::Shape({3}));
  tsl::tstring str_data2[3] = {"ccc", "dddd", "eeee"};
  HostBuffer in2(str_data2, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                 xla::ifrt::Shape({3}));
  EXPECT_THAT(in1, Not(HostBufferEq(in2)));
}

TEST(HostBufferMatcherTest, BasicString) {
  tsl::tstring str_data[3] = {"aaa", "bbb", "ccc"};
  HostBuffer in1(str_data, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                 xla::ifrt::Shape({3}));
  HostBuffer in2(str_data, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                 xla::ifrt::Shape({3}));
  EXPECT_THAT(in1, HostBufferEq(in2));
}

TEST(HostBufferMatcherTest, StringArrays) {
  std::vector<HostBuffer> expected;
  tsl::tstring str_data1[3] = {"aaa", "bbb", "ccc"};
  tsl::tstring str_data2[3] = {"ccc", "dddd", "eeee"};
  xla::ifrt::Shape shape({3});
  expected.push_back(HostBuffer(
      str_data1, xla::ifrt::DType(xla::ifrt::DType::Kind::kString), shape));
  expected.push_back(HostBuffer(
      str_data2, xla::ifrt::DType(xla::ifrt::DType::Kind::kString), shape));
  EXPECT_THAT(
      expected,
      ElementsAre(
          HostBufferEq(HostBuffer(
              str_data1, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
              shape)),
          HostBufferEq(HostBuffer(
              str_data2, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
              shape))));
}

TEST(HostBufferMatcherTest, FloatComparisonUsesTolerance) {
  // Two floats that are *nearly* equal.
  float f1(1);
  float f2 = std::nextafter(f1, f1 + 1);

  // Direct equality checks should fail, but use of the specialized `FloatEq`
  // should succeed since this matcher applies ULP-based comparison.
  ASSERT_NE(f1, f2);
  ASSERT_THAT(f1, FloatEq(f2));

  EXPECT_THAT(AsHostBuffer<float>({f1}, xla::ifrt::Shape({1})),
              HostBufferEq(AsHostBuffer<float>({f2}, xla::ifrt::Shape({1}))));
}

TEST(HostBufferMatcherTest, DoubleComparisonUsesTolerance) {
  // Two doubles that are *nearly* equal.
  double d1(1);
  double d2 = std::nextafter(d1, d1 + 1);

  // Direct equality checks should fail, but use of the specialized `DoubleEq`
  // should succeed since this matcher applies ULP-based comparison.
  ASSERT_NE(d1, d2);
  ASSERT_THAT(d1, DoubleEq(d2));

  EXPECT_THAT(AsHostBuffer<double>({d1}, xla::ifrt::Shape({1})),
              HostBufferEq(AsHostBuffer<double>({d2}, xla::ifrt::Shape({1}))));
}

TEST(HostBufferMatcherTest, MismatchFloatShallFail) {
  xla::ifrt::Shape shape({1});
  float f1(1);
  float f2(2);

  ::testing::StringMatchResultListener listener;
  EXPECT_FALSE(
      Matcher<const HostBuffer&>(
          HostBufferEq(AsHostBuffer<float>({f1}, shape)))
          .MatchAndExplain(AsHostBuffer<float>({f2}, shape), &listener));
}

}  // namespace
}  // namespace xla_serving
