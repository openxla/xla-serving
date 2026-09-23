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

#include "data/processor/cc/core/type_converter_registry.h"  // from @xla_serving

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/api/type_converter.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {
namespace core {

class TypeConverterRegistryTest : public ::testing::Test {
 protected:
  TypeConverterRegistry registry_;
};

namespace {

struct TestTypeA {};
struct TestTypeB {};

class ConverterAB : public api::TypeConverter<TestTypeA, TestTypeB> {
 public:
  absl::StatusOr<TestTypeB> Convert(const TestTypeA& input) const override {
    return TestTypeB();
  }
};

TEST_F(TypeConverterRegistryTest, RegisterDuplicateRegistrationFails) {
  // Register first time (should succeed)
  ASSERT_TRUE(
      registry_.RegisterConverter(std::make_unique<ConverterAB>()).ok());

  // Register second time (should fail)
  auto register_duplicate =
      registry_.RegisterConverter(std::make_unique<ConverterAB>());
  EXPECT_FALSE(register_duplicate.ok());
  EXPECT_EQ(register_duplicate.code(), absl::StatusCode::kAlreadyExists);
}

TEST_F(TypeConverterRegistryTest, LookupSucceeds) {
  ASSERT_TRUE(
      registry_.RegisterConverter(std::make_unique<ConverterAB>()).ok());
  ASSERT_OK_AND_ASSIGN(auto converter, registry_.LookupConverter(
                                           api::CreateTypeInfo<TestTypeA>(),
                                           api::CreateTypeInfo<TestTypeB>()));
  EXPECT_NE(converter, nullptr);
}

TEST_F(TypeConverterRegistryTest, LookupFails) {
  ASSERT_TRUE(
      registry_.RegisterConverter(std::make_unique<ConverterAB>()).ok());
  auto converter = registry_.LookupConverter(api::CreateTypeInfo<TestTypeB>(),
                                             api::CreateTypeInfo<TestTypeA>());
  EXPECT_FALSE(converter.ok());
  EXPECT_EQ(converter.status().code(), absl::StatusCode::kNotFound);
}

}  // namespace

}  // namespace core
}  // namespace composite
}  // namespace xla_serving::data
