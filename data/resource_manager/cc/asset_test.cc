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

#include "data/resource_manager/cc/asset.h"  // from @xla_serving

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "xla/tsl/platform/env.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace xla_serving::data {
namespace {

using ::absl_testing::StatusIs;

TEST(AssetTest, LoadContent) {
  std::string path = tsl::io::JoinPath(::testing::TempDir(), "my_asset");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), path, "test content"));
  Asset asset = {.name = "test", .path = path};
  EXPECT_OK(asset.LoadContent());
  EXPECT_EQ(asset.content, "test content");
}

TEST(AssetTest, LoadContentNotFound) {
  Asset asset = {.name = "test", .path = "/no/such/file"};
  EXPECT_THAT(asset.LoadContent(), StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xla_serving::data
