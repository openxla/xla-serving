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

#include "data/resource_manager/cc/asset_manager.h"  // from @xla_serving

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving

namespace xla_serving::data {
namespace {

using ::absl_testing::StatusIs;

TEST(AssetManagerTest, GetInstance) {
  AssetManager& instance1 = AssetManager::GetInstance();
  AssetManager& instance2 = AssetManager::GetInstance();
  EXPECT_EQ(&instance1, &instance2);
}

TEST(AssetManagerTest, AddAsset) {
  AssetManager& instance = AssetManager::GetInstance();
  EXPECT_OK(instance.AddAsset("asset1", "/path/to/asset1"));
  ASSERT_OK_AND_ASSIGN(const Asset* asset, instance.GetAsset("asset1"));
  EXPECT_EQ(asset->name, "asset1");
  EXPECT_EQ(asset->path, "/path/to/asset1");
}

TEST(AssetManagerTest, AddAssetAlreadyExists) {
  AssetManager& instance = AssetManager::GetInstance();
  EXPECT_OK(instance.AddAsset("asset2", "/path/to/asset2"));
  EXPECT_THAT(instance.AddAsset("asset2", "/path/to/asset2"),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(AssetManagerTest, GetAssetNotFound) {
  AssetManager& instance = AssetManager::GetInstance();
  EXPECT_THAT(instance.GetAsset("asset3"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(AssetManagerTest, AddAndGetDifferentResources) {
  AssetManager& instance = AssetManager::GetInstance();
  EXPECT_OK(instance.AddAsset("assetx", "path/to/asset/x"));
  EXPECT_OK(instance.AddAsset("assety", "path/to/asset/y"));
  ASSERT_OK_AND_ASSIGN(const Asset* assetx, instance.GetAsset("assetx"));
  ASSERT_OK_AND_ASSIGN(const Asset* assety, instance.GetAsset("assety"));
  EXPECT_NE(assetx, assety);
}

}  // namespace
}  // namespace xla_serving::data
