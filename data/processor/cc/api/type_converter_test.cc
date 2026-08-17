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

#include "data/processor/cc/api/type_converter.h"  // from @neptune

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/api/value.h"  // from @neptune
#include "data/test_utils/status_macros.h"  // from @neptune

namespace neptune::data {
namespace composite {
namespace api {
namespace {

struct TestStruct {
  int value;
};

class IntToTestStructConverter : public TypeConverter<int, TestStruct> {
 public:
  absl::StatusOr<TestStruct> Convert(const int& input) const override {
    return TestStruct(input);
  }
};

TEST(TypedTransformTest, MovableValueTransformSucceeds) {
  IntToTestStructConverter converter;

  ASSERT_OK_AND_ASSIGN(auto producer_result,
                       converter.ConvertUntyped(Value(100)));
  EXPECT_EQ(producer_result.Get<TestStruct>().value, 100);
}

class ErrorConverter : public TypeConverter<int, TestStruct> {
 public:
  absl::StatusOr<TestStruct> Convert(const int& input) const override {
    return absl::InvalidArgumentError("Error");
  }
};

TEST(TypedTransformTest, ErrorTransform) {
  ErrorConverter converter;

  auto producer_result = converter.ConvertUntyped(Value(100));
  ASSERT_FALSE(producer_result.ok());
  EXPECT_EQ(producer_result.status().message(), "Error");
}

}  // namespace
}  // namespace api
}  // namespace composite
}  // namespace neptune::data
