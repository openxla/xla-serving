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

#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/cpu_embedding_engine.h"  // from @xla_serving
#include "data/kernels/cpu_embedding/cpu_embedding.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "data/resource_manager/cc/resource_manager.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving
#include "xla/tsl/platform/env.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace xla_serving::data {
namespace composite {
namespace {

using transforms::cpu_embedding::CpuEmbeddingManifest;

TEST(CpuEmbeddingEngineResourceTest, Lifecycle) {
  CpuEmbeddingEngineResource resource("my_engine", {},
                                      "CpuEmbeddingEngineResource");

  EXPECT_EQ(resource.name(), "my_engine");
  EXPECT_NE(resource.Get(), nullptr);
}

TEST(CpuEmbeddingEngineResourceTest, RegistrationAndRetrieval) {
  ResourceManager manager;

  auto resource = std::make_shared<CpuEmbeddingEngineResource>(
      "cpu_engine", std::vector<Asset>{}, "CpuEmbeddingEngineResource");

  EXPECT_OK(manager.AddResource(resource));

  ASSERT_OK_AND_ASSIGN(auto retrieved_base,
                       manager.GetResource(resource->id()));

  auto retrieved_resource =
      std::dynamic_pointer_cast<CpuEmbeddingEngineResource>(retrieved_base);
  ASSERT_NE(retrieved_resource, nullptr);
  EXPECT_NE(retrieved_resource->Get(), nullptr);
}

TEST(CpuEmbeddingEngineResourceTest, LoadFromAssets) {
  // Setup CpuEmbeddingManifest proto representing a 2x3 embedding table.
  CpuEmbeddingManifest manifest;
  auto* table = manifest.add_tables();
  table->set_name("table1");
  table->set_vocab_size(2);
  table->set_embedding_dim(3);
  table->set_combine_mode(CpuEmbeddingManifest::EmbeddingTable::MEAN);

  auto* feature = manifest.add_feature_specs();
  feature->set_name("table1");
  feature->set_table("table1");

  std::string temp_dir = ::testing::TempDir();
  std::string manifest_path = tsl::io::JoinPath(temp_dir, "manifest.textproto");
  std::string table_path = tsl::io::JoinPath(temp_dir, "table1.bin");

  ASSERT_OK(tsl::WriteTextProto(tsl::Env::Default(), manifest_path, manifest));

  // Setup raw table data bytes (flat float array, no JCE header).
  // flat body: float matrix [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] (24 bytes)
  std::vector<float> float_table_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::string raw_table_data(
      reinterpret_cast<const char*>(float_table_data.data()),
      float_table_data.size() * sizeof(float));

  ASSERT_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), table_path, raw_table_data));

  // Wrap as Assets and initialize Resource
  Asset manifest_asset{
      .name = std::string(CpuEmbeddingEngineResource::kManifestAssetName),
      .path = manifest_path};

  Asset table_asset{.name = "table1", .path = table_path};

  CpuEmbeddingEngineResource resource(
      std::vector<Asset>{manifest_asset, table_asset});

  // Trigger native Restoration loader
  EXPECT_OK(resource.Load());

  // Verify cached features properties are extracted and served.
  ASSERT_OK_AND_ASSIGN(auto dim, resource.GetEmbeddingDim("table1"));
  EXPECT_EQ(dim, 3);

  // Verify lookup serving query executes successfully on top of restored
  // assets!
  ASSERT_NE(resource.Get(), nullptr);

  std::vector<int64_t> ids = {0, 0, 1};
  std::vector<int32_t> row_pointers = {0, 1, 3};
  std::vector<float> weights = {1.0f, 0.5f, 0.5f};

  std::vector<float> output_vec;
  auto allocator = [&](size_t size) -> absl::Span<float> {
    output_vec.resize(size);
    return absl::MakeSpan(output_vec);
  };

  // Perform dynamic lookup query on JCE engine using logical feature "table1"
  ASSERT_OK(resource.Get()->Execute<float>("table1", ids, row_pointers, weights,
                                           std::move(allocator)));

  // Expected output (BatchSize=2, EmbeddingDim=3):
  // Row 0: 1.0 * [1.0, 2.0, 3.0] = [1.0, 2.0, 3.0]
  // Row 1: 0.5 * [1.0, 2.0, 3.0] + 0.5 * [4.0, 5.0, 6.0] = [2.5, 3.5, 4.5]
  EXPECT_THAT(output_vec, testing::ElementsAre(
                              testing::FloatEq(1.0f), testing::FloatEq(2.0f),
                              testing::FloatEq(3.0f), testing::FloatEq(2.5f),
                              testing::FloatEq(3.5f), testing::FloatEq(4.5f)));
}

}  // namespace
}  // namespace composite
}  // namespace xla_serving::data
