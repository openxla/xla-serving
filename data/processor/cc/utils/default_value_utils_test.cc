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

#include "data/processor/cc/utils/default_value_utils.h"  // from @xla_serving

#include <cstdint>
#include <string>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/utils/default_value.pb.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving

namespace neptune::data {
namespace composite {
namespace utils {
namespace {

TEST(DefaultValueUtilsTest, GetDefaultValue) {
  EXPECT_EQ(GetDefaultValue<int32_t>(), 0);
  EXPECT_EQ(GetDefaultValue<int64_t>(), 0);
  EXPECT_FLOAT_EQ(GetDefaultValue<float>(), 0.0f);
  EXPECT_DOUBLE_EQ(GetDefaultValue<double>(), 0.0);
  EXPECT_EQ(GetDefaultValue<bool>(), false);
  EXPECT_EQ(GetDefaultValue<std::string>(), "");
}

TEST(DefaultValueUtilsTest, UnpackDefaultValueSuccess) {
  DefaultValue proto;
  proto.set_int_value(42);
  ASSERT_OK_AND_ASSIGN(auto val, UnpackDefaultValue<int32_t>(proto));
  EXPECT_EQ(val, 42);

  proto.set_int64_value(42000000000LL);
  ASSERT_OK_AND_ASSIGN(auto val64, UnpackDefaultValue<int64_t>(proto));
  EXPECT_EQ(val64, 42000000000LL);

  proto.set_float_value(4.2f);
  ASSERT_OK_AND_ASSIGN(auto float_val, UnpackDefaultValue<float>(proto));
  EXPECT_FLOAT_EQ(float_val, 4.2f);

  proto.set_double_value(4.2);
  ASSERT_OK_AND_ASSIGN(auto double_val, UnpackDefaultValue<double>(proto));
  EXPECT_DOUBLE_EQ(double_val, 4.2);

  proto.set_bool_value(true);
  ASSERT_OK_AND_ASSIGN(auto bool_val, UnpackDefaultValue<bool>(proto));
  EXPECT_EQ(bool_val, true);

  proto.set_string_value("hello");
  ASSERT_OK_AND_ASSIGN(auto string_val, UnpackDefaultValue<std::string>(proto));
  EXPECT_EQ(string_val, "hello");
}

TEST(DefaultValueUtilsTest, UnpackDefaultValueFailure) {
  DefaultValue proto;
  proto.set_int_value(42);
  auto val = UnpackDefaultValue<float>(proto);
  EXPECT_FALSE(val.ok());
  EXPECT_EQ(val.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(DefaultValueUtilsTest, UnpackAnySuccess) {
  DefaultValue proto;
  proto.set_int_value(42);
  google::protobuf::Any any;
  any.PackFrom(proto);

  ASSERT_OK_AND_ASSIGN(auto val, UnpackAny<int32_t>(any));
  EXPECT_EQ(val, 42);
}

TEST(DefaultValueUtilsTest, UnpackAnyFailure) {
  DefaultValue proto;
  proto.set_int_value(42);
  google::protobuf::Any any;
  any.PackFrom(proto);

  auto val = UnpackAny<float>(any);
  EXPECT_FALSE(val.ok());

  google::protobuf::Any empty_any;
  auto empty = UnpackAny<int32_t>(empty_any);
  EXPECT_FALSE(empty.ok());
}

TEST(DefaultValueUtilsTest, GetValue) {
  google::protobuf::Any any;
  int32_t val = -1;

  // Empty Any should give default value
  auto status = GetValue<int32_t>(any, val);
  ASSERT_OK(status);
  EXPECT_EQ(val, 0);

  // Non-empty Any with correct type
  DefaultValue proto;
  proto.set_int_value(42);
  any.PackFrom(proto);
  status = GetValue<int32_t>(any, val);
  ASSERT_OK(status);
  EXPECT_EQ(val, 42);

  // Non-empty Any with incorrect type
  float float_val = -1.0f;
  status = GetValue<float>(any, float_val);
  EXPECT_FALSE(status.ok());
}

template <typename T>
struct PackToAnyTestHelper;

template <>
struct PackToAnyTestHelper<int32_t> {
  static int32_t TestValue() { return 42; }
  static void AssertValue(const DefaultValue& proto, int32_t expected) {
    EXPECT_EQ(proto.int_value(), expected);
  }
};

template <>
struct PackToAnyTestHelper<int64_t> {
  static int64_t TestValue() { return 42000000000LL; }
  static void AssertValue(const DefaultValue& proto, int64_t expected) {
    EXPECT_EQ(proto.int64_value(), expected);
  }
};

template <>
struct PackToAnyTestHelper<float> {
  static float TestValue() { return 4.2f; }
  static void AssertValue(const DefaultValue& proto, float expected) {
    EXPECT_FLOAT_EQ(proto.float_value(), expected);
  }
};

template <>
struct PackToAnyTestHelper<double> {
  static double TestValue() { return 4.2; }
  static void AssertValue(const DefaultValue& proto, double expected) {
    EXPECT_DOUBLE_EQ(proto.double_value(), expected);
  }
};

template <>
struct PackToAnyTestHelper<bool> {
  static bool TestValue() { return true; }
  static void AssertValue(const DefaultValue& proto, bool expected) {
    EXPECT_EQ(proto.bool_value(), expected);
  }
};

template <>
struct PackToAnyTestHelper<std::string> {
  static std::string TestValue() { return "hello"; }
  static void AssertValue(const DefaultValue& proto,
                          const std::string& expected) {
    EXPECT_EQ(proto.string_value(), expected);
  }
};

template <typename T>
class PackToAnySuccessTest : public ::testing::Test {};

using SupportedTypes =
    ::testing::Types<int32_t, int64_t, float, double, bool, std::string>;
TYPED_TEST_SUITE(PackToAnySuccessTest, SupportedTypes);

TYPED_TEST(PackToAnySuccessTest, PacksCorrectly) {
  using T = TypeParam;
  T val = PackToAnyTestHelper<T>::TestValue();
  ASSERT_OK_AND_ASSIGN(auto any, PackToAny<T>(val));
  DefaultValue proto;
  ASSERT_TRUE(any.UnpackTo(&proto));
  PackToAnyTestHelper<T>::AssertValue(proto, val);
}

TEST(DefaultValueUtilsTest, PackToAnyUnimplementedFailure) {
  auto any = PackToAny<std::vector<int32_t>>({});
  EXPECT_FALSE(any.ok());
}

}  // namespace
}  // namespace utils
}  // namespace composite
}  // namespace neptune::data
