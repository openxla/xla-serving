/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2026 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "data/kernels/util/shape_utils.h"  // from @xla_serving

#include <cstdint>
#include <limits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::kernel {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::xla::ifrt::Shape;

TEST(ShapeUtilsTest, Predicates) {
  const Shape scalar({});
  const Shape vector({3});
  const Shape matrix({2, 3});
  const Shape square_matrix({3, 3});
  const Shape tensor3d({2, 3, 4});

  EXPECT_TRUE(ShapeUtils::IsScalar(scalar));
  EXPECT_FALSE(ShapeUtils::IsScalar(vector));
  EXPECT_FALSE(ShapeUtils::IsScalar(matrix));

  EXPECT_TRUE(ShapeUtils::IsVector(vector));
  EXPECT_FALSE(ShapeUtils::IsVector(scalar));
  EXPECT_FALSE(ShapeUtils::IsVector(matrix));

  EXPECT_FALSE(ShapeUtils::IsVectorOrHigher(scalar));
  EXPECT_TRUE(ShapeUtils::IsVectorOrHigher(vector));
  EXPECT_TRUE(ShapeUtils::IsVectorOrHigher(matrix));
  EXPECT_TRUE(ShapeUtils::IsVectorOrHigher(tensor3d));

  EXPECT_TRUE(ShapeUtils::IsMatrix(matrix));
  EXPECT_TRUE(ShapeUtils::IsMatrix(square_matrix));
  EXPECT_FALSE(ShapeUtils::IsMatrix(scalar));
  EXPECT_FALSE(ShapeUtils::IsMatrix(vector));
  EXPECT_FALSE(ShapeUtils::IsMatrix(tensor3d));

  EXPECT_TRUE(ShapeUtils::IsSquareMatrix(square_matrix));
  EXPECT_FALSE(ShapeUtils::IsSquareMatrix(matrix));
  EXPECT_FALSE(ShapeUtils::IsSquareMatrix(vector));
  EXPECT_FALSE(ShapeUtils::IsSquareMatrix(scalar));

  EXPECT_FALSE(ShapeUtils::IsMatrixOrHigher(scalar));
  EXPECT_FALSE(ShapeUtils::IsMatrixOrHigher(vector));
  EXPECT_TRUE(ShapeUtils::IsMatrixOrHigher(matrix));
  EXPECT_TRUE(ShapeUtils::IsMatrixOrHigher(square_matrix));
  EXPECT_TRUE(ShapeUtils::IsMatrixOrHigher(tensor3d));
}

TEST(ShapeUtilsTest, MakeShape) {
  EXPECT_THAT(ShapeUtils::MakeShape(std::vector<int32_t>{2, 3}),
              IsOkAndHolds(Shape({2, 3})));
  EXPECT_THAT(ShapeUtils::MakeShape(std::vector<int64_t>{4, 5, 6}),
              IsOkAndHolds(Shape({4, 5, 6})));
  EXPECT_THAT(ShapeUtils::MakeShape(std::vector<int64_t>{7, 8}),
              IsOkAndHolds(Shape({7, 8})));
  EXPECT_THAT(ShapeUtils::MakeShape(std::vector<int64_t>{1, 2, 3}),
              IsOkAndHolds(Shape({1, 2, 3})));

  const int32_t dims_s32[] = {2, 3};
  EXPECT_THAT(ShapeUtils::MakeShape(dims_s32, 2), IsOkAndHolds(Shape({2, 3})));

  const int64_t dims_s64[] = {4, 5, 6};
  EXPECT_THAT(ShapeUtils::MakeShape(dims_s64, 3),
              IsOkAndHolds(Shape({4, 5, 6})));

  // Negative number of dimensions
  int32_t dummy = 1;
  EXPECT_THAT(ShapeUtils::MakeShape(&dummy, -1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Negative number of dimensions")));

  // Negative dimension size
  EXPECT_THAT(ShapeUtils::MakeShape(std::vector<int64_t>{2, -1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dimension sizes must be non-negative")));

  // Overflow
  constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
  EXPECT_THAT(ShapeUtils::MakeShape(std::vector<int64_t>{kInt64Max, 2}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("would have more than 2**56 - 1 elements")));
}

TEST(ShapeUtilsTest, Large) {
  // Ensure the supported shape bound matches `xla::ifrt::Shape`'s 128-bit
  // element-size limit (`INT64_MAX / 128` = `2**56 - 1`).
  int64_t one = 1;
  int64_t max = std::numeric_limits<int64_t>::max() / 128;
  EXPECT_THAT(ShapeUtils::MakeShape(absl::MakeConstSpan({max})),
              IsOkAndHolds(Shape({max})));
  EXPECT_THAT(ShapeUtils::MakeShape(absl::MakeConstSpan({one, max})),
              IsOkAndHolds(Shape({one, max})));
  EXPECT_THAT(ShapeUtils::MakeShape(absl::MakeConstSpan({max, one})),
              IsOkAndHolds(Shape({max, 1})));
  EXPECT_THAT(ShapeUtils::MakeShape(absl::MakeConstSpan({one << 55})),
              IsOkAndHolds(Shape({one << 55})));
  EXPECT_THAT(
      ShapeUtils::MakeShape(absl::MakeConstSpan({one << 20, one << 35})),
      IsOkAndHolds(Shape({one << 20, one << 35})));
  EXPECT_THAT(ShapeUtils::MakeShape(
                  absl::MakeConstSpan({500, 500, 500, 500, 500, 500})),
              IsOkAndHolds(Shape({500, 500, 500, 500, 500, 500})));
}

TEST(ShapeUtilsTest, Overflow) {
  int64_t one = 1;
  int64_t max = std::numeric_limits<int64_t>::max() / 128;
  std::vector<std::vector<int64_t>> overflows = {
      {max + 1},
      {1 << 30, 1 << 30, 1 << 30},
      {1 << 5, (one << 60) + 1},
  };
  for (const auto& overflow : overflows) {
    EXPECT_THAT(ShapeUtils::MakeShape(overflow),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("would have more than 2**56 - 1 elements")));
  }
}

TEST(ShapeUtilsTest, StartsWith) {
  EXPECT_TRUE(ShapeUtils::StartsWith(Shape({}), Shape({})));
  EXPECT_TRUE(ShapeUtils::StartsWith(Shape({2, 3}), Shape({})));
  EXPECT_TRUE(ShapeUtils::StartsWith(Shape({2, 3}), Shape({2})));
  EXPECT_TRUE(ShapeUtils::StartsWith(Shape({2, 3}), Shape({2, 3})));
  EXPECT_TRUE(ShapeUtils::StartsWith(Shape({2, 3, 4}), Shape({2, 3})));
  EXPECT_FALSE(ShapeUtils::StartsWith(Shape({2, 3}), Shape({3})));
  EXPECT_FALSE(ShapeUtils::StartsWith(Shape({2, 3}), Shape({2, 4})));
  EXPECT_FALSE(ShapeUtils::StartsWith(Shape({2, 3}), Shape({2, 3, 4})));
  EXPECT_FALSE(ShapeUtils::StartsWith(Shape({2, 3, 4}), Shape({3, 4})));
}

TEST(ShapeUtilsTest, EndsWith) {
  EXPECT_TRUE(ShapeUtils::EndsWith(Shape({}), Shape({})));
  EXPECT_TRUE(ShapeUtils::EndsWith(Shape({2, 3}), Shape({})));
  EXPECT_TRUE(ShapeUtils::EndsWith(Shape({2, 3}), Shape({3})));
  EXPECT_TRUE(ShapeUtils::EndsWith(Shape({2, 3}), Shape({2, 3})));
  EXPECT_TRUE(ShapeUtils::EndsWith(Shape({2, 3, 4}), Shape({3, 4})));
  EXPECT_FALSE(ShapeUtils::EndsWith(Shape({2, 3}), Shape({2})));
  EXPECT_FALSE(ShapeUtils::EndsWith(Shape({2, 3}), Shape({2, 4})));
  EXPECT_FALSE(ShapeUtils::EndsWith(Shape({2, 3}), Shape({2, 3, 4})));
  EXPECT_FALSE(ShapeUtils::EndsWith(Shape({2, 3, 4}), Shape({2, 3})));
}

TEST(ShapeUtilsTest, ShapeListString) {
  EXPECT_EQ(ShapeUtils::ShapeListString({}), "[]");
  EXPECT_EQ(ShapeUtils::ShapeListString({Shape({}), Shape({})}), "[[], []]");
  EXPECT_EQ(ShapeUtils::ShapeListString({Shape({2, 3}), Shape({4, 5, 6})}),
            "[[2,3], [4,5,6]]");
}

TEST(ShapeUtilsTest, NumElements) {
  EXPECT_THAT(ShapeUtils::NumElements({}), IsOkAndHolds(1));
  EXPECT_THAT(ShapeUtils::NumElements({1}), IsOkAndHolds(1));
  EXPECT_THAT(ShapeUtils::NumElements({2, 3, 4}), IsOkAndHolds(24));

  constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
  EXPECT_THAT(ShapeUtils::NumElements({kInt64Max, kInt64Max}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("product would overflow int64")));

  // Test zero dimension with overflow sizes
  EXPECT_THAT(ShapeUtils::NumElements({0, kInt64Max}), IsOkAndHolds(0));
  EXPECT_THAT(ShapeUtils::NumElements({kInt64Max, 0}), IsOkAndHolds(0));

  // Test zero dimension with negative dimensions in
  // ShapeUtils::NumElements should fail
  EXPECT_THAT(ShapeUtils::NumElements({0, -1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid dimension size -1")));
  EXPECT_THAT(ShapeUtils::NumElements({-1, 0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid dimension size -1")));
}

}  // namespace
}  // namespace xla_serving::data::kernel
