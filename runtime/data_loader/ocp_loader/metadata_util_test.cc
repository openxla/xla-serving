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

#include "runtime/data_loader/ocp_loader/metadata_util.h"  // from @xla_serving

#include <string>
#include <vector>

#include "xla/tsl/platform/resource_loader.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

constexpr const char* kCheckpointPath =
    "runtime/data_loader/ocp_loader/testdata/"
    "checkpoint_with_ocp_metadata";
constexpr const char* kMetadataFileName = "_METADATA";
constexpr const char* kMetadataSkipDeserializeFileName =
    "_METADATA_SKIP_DESERIALIZE";

absl::StatusOr<nlohmann::ordered_json> GetOcpMetadata() {
  std::string _metadata;
  CHECK_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(
          tsl::GetDataDependencyFilepath(kCheckpointPath),
          kMetadataFileName),
      &_metadata));
  return nlohmann::ordered_json::parse(_metadata);
}

absl::StatusOr<nlohmann::ordered_json> GetOcpMetadataWithSkipDeserialization() {
  std::string _metadata;
  CHECK_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(
          tsl::GetDataDependencyFilepath(kCheckpointPath),
          kMetadataSkipDeserializeFileName),
      &_metadata));
  return nlohmann::ordered_json::parse(_metadata);
}

TEST(ParamNamesTest, GetParamNames) {
  TF_ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json metadata, GetOcpMetadata());
  TF_ASSERT_OK_AND_ASSIGN(auto result, GetParamNames(metadata));

  EXPECT_THAT(
      result,
      ElementsAre("my_param.a.b", "my_param.a/b/c", "my_param.0.a/b",
                  "my_param.0.c/d", "my_param", ".my_param", "Dense_0.bias",
                  "Dense_0.kernel", "Dense_1.bias", "Dense_1.kernel"));
}

TEST(ParamNamesTest, GetParamNamesWithSkipDeserialize) {
  TF_ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json metadata,
                       GetOcpMetadataWithSkipDeserialization());
  TF_ASSERT_OK_AND_ASSIGN(auto result, GetParamNames(metadata));

  EXPECT_THAT(result,
              ElementsAre("my_param.a/b/c", "my_param.0.a/b", "my_param.0.c/d",
                          "my_param", ".my_param", "Dense_0.bias",
                          "Dense_0.kernel", "Dense_1.bias"));
}

TEST(ParamNamesTest, GetParamNamesForSubTree) {
  TF_ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json metadata, GetOcpMetadata());

  // Test case: empty sub_tree, get all params.
  std::vector<std::string> sub_tree;
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> param_params,
                       GetParamNamesForSubTree(metadata, sub_tree));
  EXPECT_THAT(
      param_params,
      ElementsAre("my_param.a.b", "my_param.a/b/c", "my_param.0.a/b",
                  "my_param.0.c/d", "my_param", ".my_param", "Dense_0.bias",
                  "Dense_0.kernel", "Dense_1.bias", "Dense_1.kernel"));

  // Test case: sub_tree for "Dense_0"
  sub_tree = {"Dense_0"};
  TF_ASSERT_OK_AND_ASSIGN(param_params,
                       GetParamNamesForSubTree(metadata, sub_tree));
  EXPECT_THAT(param_params, ElementsAre("Dense_0.bias", "Dense_0.kernel"));

  // Test case: sub_tree for ["my_param", "0"]
  sub_tree = {"my_param", "0"};
  TF_ASSERT_OK_AND_ASSIGN(param_params,
                       GetParamNamesForSubTree(metadata, sub_tree));
  EXPECT_THAT(param_params, ElementsAre("my_param.0.a/b", "my_param.0.c/d"));

  // Test case: sub_tree for "my_param"
  sub_tree = {"my_param"};
  TF_ASSERT_OK_AND_ASSIGN(param_params,
                       GetParamNamesForSubTree(metadata, sub_tree));
  EXPECT_THAT(param_params,
              ElementsAre("my_param.a.b", "my_param.a/b/c", "my_param.0.a/b",
                          "my_param.0.c/d", "my_param"));

  // Test case: sub_tree for "Dense", which is invalid and should return error.
  sub_tree = {"Dense"};
  EXPECT_THAT(GetParamNamesForSubTree(metadata, sub_tree),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("No valid param name")));

  // Test case: sub_tree for {"Dense_0", "b"}, which is invalid and should
  // return error.
  sub_tree = {"Dense_0", "b"};
  EXPECT_THAT(GetParamNamesForSubTree(metadata, sub_tree),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("No valid param name")));

  // Test case: sub_tree for {"Dense_0.bias"}, which is invalid and should
  // return error.
  sub_tree = {"Dense_0.bias"};
  EXPECT_THAT(GetParamNamesForSubTree(metadata, sub_tree),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("No valid param name")));
}

TEST(ParamNamesTest, GetAllWeightNamesFromCheckpoint) {
  TF_ASSERT_OK_AND_ASSIGN(
      const auto weight_names,
      GetAllWeightNamesFromCheckpoint(
          tsl::GetDataDependencyFilepath(kCheckpointPath)));
  EXPECT_THAT(
      weight_names,
      ElementsAre("my_param.a.b", "my_param.a/b/c", "my_param.0.a/b",
                  "my_param.0.c/d", "my_param", ".my_param", "Dense_0.bias",
                  "Dense_0.kernel", "Dense_1.bias", "Dense_1.kernel"));
}

}  // namespace
}  // namespace xla_serving
