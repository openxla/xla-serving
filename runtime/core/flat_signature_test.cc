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

#include "runtime/core/flat_signature.h"  // from @xla_serving

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/core/value_matcher.h"  // from @xla_serving
#include "runtime/core/value_utils.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

// Creates a simple flat signature. `names` is the list of names in
// `MetaArgSpec`s.
FlatSignature CreateSimpleFlatSignature(
    std::vector<std::optional<std::string>> names) {
  ArgSpec arg_spec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kS32),
                   .shape = {1},
                   .sharding = xla::HloSharding::Replicate()};
  std::vector<MetaArgSpec> meta_specs;
  meta_specs.reserve(names.size());
  for (int i = 0; i < names.size(); ++i) {
    meta_specs.push_back(MetaArgSpec{.spec = arg_spec, .name = names[i]});
  }
  return FlatSignature(std::move(meta_specs));
}

TEST(MetaArgSpecTest, DebugString) {
  MetaArgSpec meta_spec{
      .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                      .shape = {1, 2},
                      .sharding = xla::HloSharding::Replicate()},
      .name = "x",
      .default_value =
          Value(AsHostBuffer<float>({1.0, 2.0}, xla::ifrt::Shape({1, 2}))),
  };
  EXPECT_THAT(
      meta_spec.DebugString(),
      AllOf(HasSubstr(meta_spec.spec.DebugString()), HasSubstr("name: x"),
            HasSubstr("has_default_value: true")));
}

TEST(MetaArgSpecTest, EqualExceptDefaultValue) {
  MetaArgSpec meta_spec{
      .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kS32),
                      .shape = {3},
                      .sharding = xla::HloSharding::Replicate()},
      .name = "y",
      .default_value =
          Value(AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({3}))),
  };

  MetaArgSpec meta_spec_copy = meta_spec;
  EXPECT_TRUE(meta_spec.EqualExceptDefaultValue(meta_spec_copy));

  // Change `spec`.
  meta_spec_copy.spec.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32);
  EXPECT_FALSE(meta_spec.EqualExceptDefaultValue(meta_spec_copy));

  // Change `name`.
  meta_spec_copy.name = "z";
  EXPECT_FALSE(meta_spec.EqualExceptDefaultValue(meta_spec_copy));

  // Remove `default_value`.
  meta_spec_copy = meta_spec;
  meta_spec_copy.default_value.Reset();
  EXPECT_FALSE(meta_spec.EqualExceptDefaultValue(meta_spec_copy));
}

TEST(FlatSignatureTest, DebugString) {
  FlatSignature flat_signature = CreateSimpleFlatSignature({"x", "y"});
  EXPECT_THAT(flat_signature.DebugString(),
              AllOf(HasSubstr("#0: MetaArgSpec: "),
                    HasSubstr(flat_signature.specs()[0].spec.DebugString()),
                    HasSubstr("#1: MetaArgSpec: "),
                    HasSubstr(flat_signature.specs()[1].spec.DebugString())));
}

TEST(FlatSignatureTest, GetAllNames) {
  FlatSignature flat_signature = CreateSimpleFlatSignature({"x", "y", "z"});
  EXPECT_THAT(flat_signature.GetAllNames(),
              IsOkAndHolds(ElementsAre("x", "y", "z")));
}

TEST(FlatSignatureTest, GetAllNamesMissingName) {
  FlatSignature flat_signature =
      CreateSimpleFlatSignature({"x", "z", std::nullopt});
  EXPECT_THAT(flat_signature.GetAllNames(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Not all specs have names")));
}

TEST(FlatSignatureTest, AddDefaultValueWithName) {
  FlatSignature flat_signature({MetaArgSpec{
      .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kS32),
                      .shape = {3},
                      .sharding = xla::HloSharding::Replicate()},
      .name = "s",
      // No default value.
  }});
  HostBuffer host_buffer =
      AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({3}));
  TF_ASSERT_OK(flat_signature.AddDefaultValue("s", Value(host_buffer)));
  EXPECT_THAT(flat_signature.specs().front().default_value,
              ValueOfType<HostBuffer>(HostBufferEq(host_buffer)));
}

TEST(FlatSignatureTest, AddDefaultValueWithIndex) {
  FlatSignature flat_signature({MetaArgSpec{
      .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kS32),
                      .shape = {2, 2},
                      .sharding = xla::HloSharding::Replicate()},
      // No default value.
  }});
  HostBuffer host_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  TF_ASSERT_OK(flat_signature.AddDefaultValue(0, Value(host_buffer)));
  EXPECT_THAT(flat_signature.specs().front().default_value,
              ValueOfType<HostBuffer>(HostBufferEq(host_buffer)));
}

TEST(FlatSignatureTest, AddDefaultValueWithWrongName) {
  FlatSignature flat_signature({MetaArgSpec{
      .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kS32),
                      .shape = {2, 2},
                      .sharding = xla::HloSharding::Replicate()},
      // No default value.
  }});
  HostBuffer host_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  EXPECT_THAT(
      flat_signature.AddDefaultValue("x", Value(host_buffer)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("not found")));
}

TEST(FlatSignatureTest, AddDefaultValueAlreadyHasDefaultValue) {
  HostBuffer host_buffer = AsHostBuffer<float>({1, 2}, xla::ifrt::Shape({2}));
  FlatSignature flat_signature({MetaArgSpec{
      .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                      .shape = {2},
                      .sharding = xla::HloSharding::Replicate()},
      .default_value = Value(std::move(host_buffer)),
  }});

  EXPECT_THAT(flat_signature.AddDefaultValue(
                  0, Value(AsHostBuffer<float>({3, 2}, xla::ifrt::Shape({2})))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("already has a default value")));
}

TEST(FlatSignatureTest, FillArgsWithPositionalArgs) {
  FlatSignature flat_signature = CreateSimpleFlatSignature(
      std::vector<std::optional<std::string>>{3, std::nullopt});
  int num_args = flat_signature.specs().size();
  std::vector<HostBuffer> host_buffers;
  host_buffers.reserve(num_args);
  for (int i = 0; i < num_args; ++i) {
    host_buffers.push_back(AsHostBuffer<int32_t>({i}, xla::ifrt::Shape({1})));
  }

  std::vector<Value> flattened_args =
      internal::PackAsValues(std::move(host_buffers));
  TF_ASSERT_OK(flat_signature.FillArgs(flattened_args, {}));

  ASSERT_EQ(flattened_args.size(), num_args);
  for (int i = 0; i < num_args; ++i) {
    EXPECT_THAT(flattened_args[i],
                ValueOfType<HostBuffer>(HostBufferEq(
                    AsHostBuffer<int32_t>({i}, xla::ifrt::Shape({1})))));
  }
}

TEST(FlatSignatureTest, FillArgsWithKwargs) {
  absl::flat_hash_map<std::string, HostBuffer> kwargs = {
      {"a", AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}))},
      {"b", AsHostBuffer<int32_t>({2}, xla::ifrt::Shape({1}))},
      {"c", AsHostBuffer<int32_t>({3}, xla::ifrt::Shape({1}))},
  };
  std::vector<Value> flattened_args;
  TF_ASSERT_OK(
      CreateSimpleFlatSignature({"c", "b", "a"})
          .FillArgs(flattened_args, internal::PackAsValues(std::move(kwargs))));

  // Note the order has changed.
  EXPECT_THAT(
      flattened_args,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(
                      AsHostBuffer<int32_t>({3}, xla::ifrt::Shape({1})))),
                  ValueOfType<HostBuffer>(HostBufferEq(
                      AsHostBuffer<int32_t>({2}, xla::ifrt::Shape({1})))),
                  ValueOfType<HostBuffer>(HostBufferEq(
                      AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}))))));
}

TEST(FlatSignatureTest, FillArgsWithPositionalArgsAndKwargs) {
  FlatSignature flat_signature = CreateSimpleFlatSignature({"c", "b", "a"});
  std::vector<HostBuffer> positional_args{
      AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1})),
      AsHostBuffer<int32_t>({2}, xla::ifrt::Shape({1}))};
  absl::flat_hash_map<std::string, Value> kwargs = {
      {"a", Value(AsHostBuffer<int32_t>({3}, xla::ifrt::Shape({1})))},
  };

  std::vector<Value> flattened_args =
      internal::PackAsValues(std::move(positional_args));
  TF_ASSERT_OK(flat_signature.FillArgs(
      flattened_args, internal::PackAsValues(std::move(kwargs))));
  EXPECT_THAT(
      flattened_args,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(
                      AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1})))),
                  ValueOfType<HostBuffer>(HostBufferEq(
                      AsHostBuffer<int32_t>({2}, xla::ifrt::Shape({1})))),
                  ValueOfType<HostBuffer>(HostBufferEq(
                      AsHostBuffer<int32_t>({3}, xla::ifrt::Shape({1}))))));
}

TEST(FlatSignatureTest, FillArgsConflictAssignments) {
  std::vector<HostBuffer> positional_args{
      AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}))};
  absl::flat_hash_map<std::string, Value> kwargs = {
      {"a", Value(AsHostBuffer<int32_t>({3}, xla::ifrt::Shape({1})))},
  };

  std::vector<Value> flattened_args =
      internal::PackAsValues(std::move(positional_args));
  EXPECT_THAT(CreateSimpleFlatSignature({"a"}).FillArgs(
                  flattened_args, internal::PackAsValues(std::move(kwargs))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is already set by positional args.")));
}

TEST(FlatSignatureTest, FillArgsOnlyKwargsMissingName) {
  absl::flat_hash_map<std::string, HostBuffer> kwargs = {
      {"a", AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}))}};
  std::vector<Value> flattened_args;
  EXPECT_THAT(
      CreateSimpleFlatSignature({"a", std::nullopt})
          .FillArgs(flattened_args, internal::PackAsValues(std::move(kwargs))),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Not all specs have names")));
}

TEST(FlatSignatureTest, FillArgsMissingValue) {
  std::vector<HostBuffer> positional_args{
      AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}))};
  absl::flat_hash_map<std::string, Value> kwargs = {
      {"a", Value(AsHostBuffer<int32_t>({3}, xla::ifrt::Shape({1})))},
  };

  std::vector<Value> flattened_args =
      internal::PackAsValues(std::move(positional_args));
  EXPECT_THAT(
      CreateSimpleFlatSignature({"c", "b", "a"})
          .FillArgs(flattened_args, internal::PackAsValues(std::move(kwargs))),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("is not set and does not have a default value")));
}

TEST(FlatSignatureTest, FillArgsWithDefaultValues) {
  FlatSignature flat_signature = CreateSimpleFlatSignature({"c", "b", "a"});
  TF_ASSERT_OK(flat_signature.AddDefaultValue(
      1, Value(AsHostBuffer<int32_t>({2}, xla::ifrt::Shape({1})))));
  std::vector<HostBuffer> positional_args{
      AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}))};
  absl::flat_hash_map<std::string, Value> kwargs = {
      {"a", Value(AsHostBuffer<int32_t>({3}, xla::ifrt::Shape({1})))},
  };

  std::vector<Value> flattened_args =
      internal::PackAsValues(std::move(positional_args));
  TF_ASSERT_OK(flat_signature.FillArgs(
      flattened_args, internal::PackAsValues(std::move(kwargs))));

  EXPECT_THAT(
      flattened_args,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(
                      AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1})))),
                  ValueOfType<HostBuffer>(HostBufferEq(
                      AsHostBuffer<int32_t>({2}, xla::ifrt::Shape({1})))),
                  ValueOfType<HostBuffer>(HostBufferEq(
                      AsHostBuffer<int32_t>({3}, xla::ifrt::Shape({1}))))));
}

TEST(FlatSignatureTest, FillArgsStrictMatchConflictAssignments) {
  std::vector<HostBuffer> positional_args{
      AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}))};
  absl::flat_hash_map<std::string, Value> kwargs = {
      {"a", Value(AsHostBuffer<int32_t>({3}, xla::ifrt::Shape({1})))},
  };

  std::vector<Value> flattened_args =
      internal::PackAsValues(std::move(positional_args));
  EXPECT_THAT(CreateSimpleFlatSignature({"a"}).FillArgs(
                  flattened_args, internal::PackAsValues(std::move(kwargs))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is already set by positional args")));
}

TEST(FlatSignatureTest, FillArgsNonStrictMatchUnknownKwargs) {
  FlatSignature flat_signature = CreateSimpleFlatSignature({"a"});
  TF_ASSERT_OK(flat_signature.AddDefaultValue(
      0, Value(AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1})))));
  std::vector<Value> flattened_args;
  absl::flat_hash_map<std::string, Value> kwargs = {
      {"x", Value(AsHostBuffer<int32_t>({2}, xla::ifrt::Shape({1})))},
  };
  TF_ASSERT_OK(flat_signature.FillArgs(
      flattened_args, internal::PackAsValues(std::move(kwargs))));
  EXPECT_THAT(flattened_args,
              ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(
                  AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}))))));
}

TEST(FlatSignatureTest, UnflattenToMap) {
  FlatSignature flat_signature = CreateSimpleFlatSignature({"c", "b", "a"});
  std::vector<HostBuffer> host_buffers;
  host_buffers.reserve(flat_signature.specs().size());
  for (int i = 0; i < flat_signature.specs().size(); ++i) {
    host_buffers.push_back(AsHostBuffer<int32_t>({i}, xla::ifrt::Shape({1})));
  }
  std::vector<Value> values = internal::PackAsValues(std::move(host_buffers));
  TF_ASSERT_OK_AND_ASSIGN(auto unflattened_args,
                          flat_signature.UnflattenToMap(std::move(values)));
  EXPECT_THAT(
      unflattened_args,
      UnorderedElementsAre(
          Pair("a", ValueOfType<HostBuffer>(HostBufferEq(
                        AsHostBuffer<int32_t>({2}, xla::ifrt::Shape({1}))))),
          Pair("b", ValueOfType<HostBuffer>(HostBufferEq(
                        AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}))))),
          Pair("c", ValueOfType<HostBuffer>(HostBufferEq(
                        AsHostBuffer<int32_t>({0}, xla::ifrt::Shape({1})))))));
}

TEST(FlatSignatureTest, UnflattenToMapMissingName) {
  FlatSignature flat_signature = CreateSimpleFlatSignature({"a", std::nullopt});

  std::vector<HostBuffer> host_buffers;
  host_buffers.reserve(flat_signature.specs().size());
  for (int i = 0; i < flat_signature.specs().size(); ++i) {
    host_buffers.push_back(AsHostBuffer<int32_t>({i}, xla::ifrt::Shape({1})));
  }
  EXPECT_THAT(
      flat_signature.UnflattenToMap(
          internal::PackAsValues(std::move(host_buffers))),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("no names")));
}

TEST(FlatSignatureTest, ResolvePlacementsNamedSparseSuccess) {
  FlatSignature flat_signature = CreateSimpleFlatSignature({"a", "b", "c"});
  absl::flat_hash_map<std::string, Placement> named = {
      {"b", Placement::kDevice},
  };
  std::vector<Placement> resolved = flat_signature.ResolvePlacements(named);
  EXPECT_THAT(resolved, ElementsAre(Placement::kHost, Placement::kDevice,
                                    Placement::kHost));
}

TEST(FlatSignatureTest, ResolvePlacementsEmptyDefaultsToEmpty) {
  FlatSignature flat_signature = CreateSimpleFlatSignature({"a", "b"});
  std::vector<Placement> resolved = flat_signature.ResolvePlacements({});
  EXPECT_TRUE(resolved.empty());
}

TEST(FlatSignatureTest, ResolvePlacementsIgnoresUnknownTensorNames) {
  FlatSignature flat_signature = CreateSimpleFlatSignature({"a", "b"});
  absl::flat_hash_map<std::string, Placement> named = {
      {"b", Placement::kDevice},
      {"unknown_tensor", Placement::kDevice},
  };
  std::vector<Placement> resolved = flat_signature.ResolvePlacements(named);
  EXPECT_THAT(resolved, ElementsAre(Placement::kHost, Placement::kDevice));
}

TEST(FlatSignatureTest, ResolvePlacementsManifestDefaultFallback) {
  FlatSignature flat_signature({
      MetaArgSpec{
          .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = {2},
                          .sharding = xla::HloSharding::Replicate(),
                          .placement = Placement::kDevice},
          .name = "a",
      },
      MetaArgSpec{
          .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = {2},
                          .sharding = xla::HloSharding::Replicate(),
                          .placement = Placement::kHost},
          .name = "b",
      },
  });

  absl::flat_hash_map<std::string, Placement> named = {
      {"b", Placement::kDevice},
  };
  std::vector<Placement> resolved = flat_signature.ResolvePlacements(named);
  EXPECT_THAT(resolved, ElementsAre(Placement::kDevice, Placement::kDevice));
}

TEST(FlatSignatureTest, ResolvePlacementsUnnamedSignatureBehavior) {
  FlatSignature unnamed_signature({
      MetaArgSpec{
          .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = {2},
                          .sharding = xla::HloSharding::Replicate()},
          .name = std::nullopt,
      },
  });
  // Empty placement map succeeds on unnamed signature
  std::vector<Placement> resolved = unnamed_signature.ResolvePlacements({});
  EXPECT_TRUE(resolved.empty());
  // Non-empty placement map ignores unknown names on unnamed signature
  absl::flat_hash_map<std::string, Placement> named = {
      {"a", Placement::kDevice}};
  resolved = unnamed_signature.ResolvePlacements(named);
  EXPECT_THAT(resolved, ElementsAre(Placement::kHost));
}

}  // namespace
}  // namespace xla_serving
