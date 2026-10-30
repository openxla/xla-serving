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

#include <atomic>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/cord.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Eq;
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

// Expects `converter` to be non-null and to convert `value` to a 1-element
// float `HostBuffer` holding `expected`.
void ExpectConvertsTo(
    const std::shared_ptr<const HostBufferConverter>& converter,
    const Value& value, float expected) {
  ASSERT_THAT(converter, NotNull());
  absl::StatusOr<HostBuffer> result = converter->Convert(value);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(*result, HostBufferEq(AsHostBuffer<float>(
                           {expected}, xla::ifrt::Shape({1}))));
}

TEST(HostBufferConverterTest, ConvertsRegisteredType) {
  HostBufferConverterRegistry registry;
  ASSERT_THAT(registry.Register<FakeTensor>("fake", ConvertFakeTensor), IsOk());
  std::shared_ptr<const HostBufferConverter> converter =
      registry.Find(Value(FakeTensor{2.0f}));
  ExpectConvertsTo(converter, Value(FakeTensor{2.0f}), 2.0f);
  EXPECT_THAT(converter->name(), Eq("fake"));
}

TEST(HostBufferConverterTest, UnregisteredTypeHasNoConverter) {
  HostBufferConverterRegistry registry;
  ASSERT_THAT(registry.Register<FakeTensor>("fake", ConvertFakeTensor), IsOk());
  EXPECT_THAT(registry.Find(Value(OtherFakeTensor{1.0f})), IsNull());
}

TEST(HostBufferConverterTest, EmptyRegistryHasNoConverter) {
  HostBufferConverterRegistry registry;
  EXPECT_THAT(registry.Find(Value(FakeTensor{1.0f})), IsNull());
  EXPECT_THAT(registry.Find(Value()), IsNull());
}

// The converter's error is returned as is; callers (e.g. the StableHLO runner)
// add their own context.
TEST(HostBufferConverterTest, ConverterErrorIsReturnedUnchanged) {
  absl::Status error = absl::ResourceExhaustedError("out of memory");
  error.SetPayload("type.googleapis.com/test.Detail", absl::Cord("detail"));
  HostBufferConverterRegistry registry;
  ASSERT_THAT(registry.Register<FakeTensor>(
                  "exhausted",
                  [error](const FakeTensor&) -> absl::StatusOr<HostBuffer> {
                    return error;
                  }),
              IsOk());
  std::shared_ptr<const HostBufferConverter> converter =
      registry.Find(Value(FakeTensor{1.0f}));
  ASSERT_THAT(converter, NotNull());
  EXPECT_THAT(converter->Convert(Value(FakeTensor{1.0f})).status(), Eq(error));
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
  ExpectConvertsTo(registry.Find(Value(FakeTensor{2.0f})),
                   Value(FakeTensor{2.0f}), 2.0f);
  ExpectConvertsTo(registry.Find(Value(OtherFakeTensor{2.0f})),
                   Value(OtherFakeTensor{2.0f}), 20.0f);
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
    ExpectConvertsTo(registry.Find(Value(FakeTensor{3.0f})),
                     Value(FakeTensor{3.0f}), 3.0f);
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

TEST(HostBufferConverterTest,
     ConcurrentLookupsWhileRegisteringAndUnregistering) {
  HostBufferConverterRegistry registry;
  ASSERT_THAT(registry.Register<FakeTensor>("fake", ConvertFakeTensor), IsOk());
  std::atomic<bool> done = false;
  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&] {
      while (!done.load()) {
        // `FakeTensor`'s converter is never removed.
        ExpectConvertsTo(registry.Find(Value(FakeTensor{2.0f})),
                         Value(FakeTensor{2.0f}), 2.0f);
        // `OtherFakeTensor`'s converter comes and goes; when found, it works.
        if (std::shared_ptr<const HostBufferConverter> other =
                registry.Find(Value(OtherFakeTensor{2.0f}));
            other != nullptr) {
          ExpectConvertsTo(other, Value(OtherFakeTensor{2.0f}), 20.0f);
        }
      }
    });
  }
  for (int i = 0; i < 200; ++i) {
    absl::StatusOr<ScopedHostBufferConverterRegistration> scoped =
        ScopedHostBufferConverterRegistration::Create<OtherFakeTensor>(
            "other", ConvertOtherFakeTensor, registry);
    ASSERT_THAT(scoped, IsOk());
  }
  done = true;
  for (std::thread& reader : readers) reader.join();
  EXPECT_THAT(registry.Find(Value(OtherFakeTensor{1.0f})), IsNull());
  EXPECT_THAT(registry.Find(Value(FakeTensor{1.0f})), NotNull());
}

TEST(HostBufferConverterTest, ScopedRegistrationUsesTheGlobalRegistry) {
  {
    absl::StatusOr<ScopedHostBufferConverterRegistration> scoped =
        ScopedHostBufferConverterRegistration::Create<FakeTensor>(
            "scoped", ConvertFakeTensor);
    ASSERT_THAT(scoped, IsOk());
    ExpectConvertsTo(FindHostBufferConverter(Value(FakeTensor{4.0f})),
                     Value(FakeTensor{4.0f}), 4.0f);
  }
  EXPECT_THAT(FindHostBufferConverter(Value(FakeTensor{4.0f})), IsNull());
}

}  // namespace
}  // namespace xla_serving
