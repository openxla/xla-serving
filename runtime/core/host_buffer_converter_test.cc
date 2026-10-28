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

#include "runtime/core/host_buffer_converter.h"  // from @xla_serving

#include <memory>
#include <optional>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;

// Host value types that no real converter knows about.
struct FakeTensor {
  float value;
};
struct OtherFakeTensor {
  float value;
};

absl::StatusOr<HostBuffer> ConvertFakeTensor(const FakeTensor& tensor) {
  return AsHostBuffer<float>({tensor.value}, xla::ifrt::Shape({1}));
}

absl::StatusOr<HostBuffer> ConvertOtherFakeTensor(
    const OtherFakeTensor& tensor) {
  return AsHostBuffer<float>({tensor.value * 10}, xla::ifrt::Shape({1}));
}

void ExpectConvertsTo(const absl::StatusOr<std::optional<HostBuffer>>& result,
                      float expected) {
  ASSERT_THAT(result, IsOk());
  ASSERT_TRUE(result->has_value());
  EXPECT_THAT(**result, HostBufferEq(AsHostBuffer<float>(
                            {expected}, xla::ifrt::Shape({1}))));
}

TEST(HostBufferConverterTest, ConvertsRegisteredType) {
  HostBufferConverterRegistry registry;
  ASSERT_THAT(registry.Register<FakeTensor>("fake", ConvertFakeTensor), IsOk());
  ExpectConvertsTo(registry.Convert(Value(FakeTensor{2.0f})), 2.0f);
  std::shared_ptr<const HostBufferConverter> converter =
      registry.Find(Value(FakeTensor{2.0f}));
  ASSERT_THAT(converter, NotNull());
  EXPECT_THAT(converter->name(), Eq("fake"));
}

TEST(HostBufferConverterTest, UnregisteredTypeHasNoConverter) {
  HostBufferConverterRegistry registry;
  ASSERT_THAT(registry.Register<FakeTensor>("fake", ConvertFakeTensor), IsOk());
  EXPECT_THAT(registry.Convert(Value(OtherFakeTensor{1.0f})),
              IsOkAndHolds(Eq(std::nullopt)));
  EXPECT_THAT(registry.Find(Value(OtherFakeTensor{1.0f})), IsNull());
}

TEST(HostBufferConverterTest, EmptyRegistryHasNoConverter) {
  HostBufferConverterRegistry registry;
  EXPECT_THAT(registry.Convert(Value(FakeTensor{1.0f})),
              IsOkAndHolds(Eq(std::nullopt)));
  EXPECT_THAT(registry.Find(Value()), IsNull());
}

TEST(HostBufferConverterTest, ConverterErrorIsPropagatedWithItsName) {
  HostBufferConverterRegistry registry;
  ASSERT_THAT(registry.Register<FakeTensor>(
                  "broken",
                  [](const FakeTensor&) -> absl::StatusOr<HostBuffer> {
                    return absl::InvalidArgumentError("bad tensor");
                  }),
              IsOk());
  EXPECT_THAT(registry.Convert(Value(FakeTensor{1.0f})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("'broken'"), HasSubstr("bad tensor"))));
}

TEST(HostBufferConverterTest, DuplicateTypeIsRejectedNamingBoth) {
  HostBufferConverterRegistry registry;
  ASSERT_THAT(registry.Register<FakeTensor>("first", ConvertFakeTensor),
              IsOk());
  EXPECT_THAT(
      registry.Register<FakeTensor>("second", ConvertFakeTensor),
      StatusIs(absl::StatusCode::kAlreadyExists,
               Eq("A HostBuffer converter for this type is already "
                  "registered ('first'); refusing to register 'second'.")));
  // The first registration is kept.
  EXPECT_THAT(registry.Find(Value(FakeTensor{1.0f}))->name(), Eq("first"));
}

TEST(HostBufferConverterTest, DifferentTypesCoexist) {
  HostBufferConverterRegistry registry;
  ASSERT_THAT(registry.Register<FakeTensor>("fake", ConvertFakeTensor), IsOk());
  ASSERT_THAT(
      registry.Register<OtherFakeTensor>("other", ConvertOtherFakeTensor),
      IsOk());
  ExpectConvertsTo(registry.Convert(Value(FakeTensor{2.0f})), 2.0f);
  ExpectConvertsTo(registry.Convert(Value(OtherFakeTensor{2.0f})), 20.0f);
}

TEST(HostBufferConverterTest, UnregisterRemovesOnlyThatType) {
  HostBufferConverterRegistry registry;
  ASSERT_THAT(registry.Register<FakeTensor>("fake", ConvertFakeTensor), IsOk());
  ASSERT_THAT(
      registry.Register<OtherFakeTensor>("other", ConvertOtherFakeTensor),
      IsOk());
  EXPECT_TRUE(registry.Unregister<FakeTensor>());
  EXPECT_FALSE(registry.Unregister<FakeTensor>());
  EXPECT_THAT(registry.Find(Value(FakeTensor{1.0f})), IsNull());
  EXPECT_THAT(registry.Find(Value(OtherFakeTensor{1.0f})), NotNull());
}

TEST(HostBufferConverterTest, ScopedRegistrationRemovesOnlyItsOwnEntry) {
  HostBufferConverterRegistry registry;
  ASSERT_THAT(
      registry.Register<OtherFakeTensor>("other", ConvertOtherFakeTensor),
      IsOk());
  {
    absl::StatusOr<ScopedHostBufferConverterRegistration> scoped =
        ScopedHostBufferConverterRegistration::Create<FakeTensor>(
            "scoped", ConvertFakeTensor, registry);
    ASSERT_THAT(scoped, IsOk());
    ExpectConvertsTo(registry.Convert(Value(FakeTensor{3.0f})), 3.0f);
    // A second scoped registration for the same type is rejected, and its
    // failure does not remove the first one.
    EXPECT_THAT(ScopedHostBufferConverterRegistration::Create<FakeTensor>(
                    "again", ConvertFakeTensor, registry),
                StatusIs(absl::StatusCode::kAlreadyExists));
    EXPECT_THAT(registry.Find(Value(FakeTensor{3.0f})), NotNull());
  }
  EXPECT_THAT(registry.Find(Value(FakeTensor{3.0f})), IsNull());
  EXPECT_THAT(registry.Find(Value(OtherFakeTensor{3.0f})), NotNull());
  // The type can be registered again afterwards.
  EXPECT_THAT(registry.Register<FakeTensor>("later", ConvertFakeTensor),
              IsOk());
}

TEST(HostBufferConverterTest, ScopedRegistrationUsesTheGlobalRegistry) {
  {
    absl::StatusOr<ScopedHostBufferConverterRegistration> scoped =
        ScopedHostBufferConverterRegistration::Create<FakeTensor>(
            "scoped", ConvertFakeTensor);
    ASSERT_THAT(scoped, IsOk());
    ExpectConvertsTo(ConvertToHostBuffer(Value(FakeTensor{4.0f})), 4.0f);
    ASSERT_THAT(FindHostBufferConverter(Value(FakeTensor{4.0f})), NotNull());
  }
  EXPECT_THAT(ConvertToHostBuffer(Value(FakeTensor{4.0f})),
              IsOkAndHolds(Eq(std::nullopt)));
}

}  // namespace
}  // namespace xla_serving
