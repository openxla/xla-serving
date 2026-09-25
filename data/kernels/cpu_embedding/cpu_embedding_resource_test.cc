// Copyright 2026 Google LLC
// Copyright 2026 Arm Ltd. and affiliates
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
#include "absl/status/status.h"  // from @abseil-cpp
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

CpuEmbeddingManifest MakeSumManifest(int64_t vocab_size, int64_t dimension) {
  CpuEmbeddingManifest manifest;
  auto* table = manifest.add_tables();
  table->set_name("table1");
  table->set_vocab_size(vocab_size);
  table->set_embedding_dim(dimension);
  table->set_combine_mode(CpuEmbeddingManifest::EmbeddingTable::SUM);
  auto* feature = manifest.add_feature_specs();
  feature->set_name("feature1");
  feature->set_table("table1");
  return manifest;
}

std::string RawBytes(absl::Span<const float> values) {
  return std::string(reinterpret_cast<const char*>(values.data()),
                     values.size() * sizeof(float));
}

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
  // Verify path-backed table contents remain unmaterialized after loading.
  ASSERT_EQ(resource.assets().size(), 2);
  EXPECT_TRUE(resource.assets()[1].content.empty());

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

class MappedCpuEmbeddingResourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    manifest_path_ =
        tsl::io::JoinPath(::testing::TempDir(), "mapped_manifest.textproto");
    table_path_ = tsl::io::JoinPath(::testing::TempDir(), "mapped_table.bin");
    ASSERT_OK(WriteManifest());
    ASSERT_OK(WriteTable(table_data_));
  }

  absl::Status WriteManifest() {
    return tsl::WriteTextProto(tsl::Env::Default(), manifest_path_, manifest_);
  }

  absl::Status WriteTable(absl::Span<const float> values) {
    return tsl::WriteStringToFile(tsl::Env::Default(), table_path_,
                                  RawBytes(values));
  }

  std::vector<Asset> Assets(std::string content = {}) const {
    return {
        {.name = std::string(CpuEmbeddingEngineResource::kManifestAssetName),
         .path = manifest_path_},
        {.name = "table1", .path = table_path_, .content = std::move(content)},
    };
  }

  static absl::Status Execute(CpuEmbeddingEngineResource& resource,
                              std::vector<float>* output) {
    const std::vector<int64_t> ids = {0, 1};
    const std::vector<int32_t> rows = {0, 2};
    const std::vector<float> weights = {1.0f, 1.0f};
    auto allocator = [output](size_t size) -> absl::Span<float> {
      output->resize(size);
      return absl::MakeSpan(*output);
    };
    return resource.Get()
        ->Execute<float>("feature1", ids, rows, weights, std::move(allocator))
        .status();
  }

  CpuEmbeddingManifest manifest_ = MakeSumManifest(2, 3);
  std::vector<float> table_data_ = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::string manifest_path_;
  std::string table_path_;
};

// Verify inline and mapped tables produce identical outputs.
TEST_F(MappedCpuEmbeddingResourceTest, InlineAndMappedTablesMatch) {
  CpuEmbeddingEngineResource mapped(Assets());
  CpuEmbeddingEngineResource inline_resource(Assets(RawBytes(table_data_)));
  ASSERT_OK(mapped.Load());
  ASSERT_OK(inline_resource.Load());

  std::vector<float> mapped_output;
  std::vector<float> inline_output;
  ASSERT_OK(Execute(mapped, &mapped_output));
  ASSERT_OK(Execute(inline_resource, &inline_output));
  EXPECT_THAT(mapped_output, testing::ElementsAre(5.0f, 7.0f, 9.0f));
  EXPECT_THAT(mapped_output, testing::ContainerEq(inline_output));
  EXPECT_TRUE(mapped.assets()[1].content.empty());
}

// Verify mappings survive path replacement and removal.
TEST_F(MappedCpuEmbeddingResourceTest,
       MappingSurvivesPathReplacementAndRemoval) {
  const std::string replacement_path =
      tsl::io::JoinPath(::testing::TempDir(), "mapped_replacement.bin");
  const std::vector<float> replacement_data(6, 100.0f);
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), replacement_path,
                                   RawBytes(replacement_data)));
  CpuEmbeddingEngineResource resource(Assets());
  ASSERT_OK(resource.Load());

  ASSERT_OK(tsl::Env::Default()->DeleteFile(table_path_));
  ASSERT_OK(tsl::Env::Default()->RenameFile(replacement_path, table_path_));
  std::vector<float> output;
  ASSERT_OK(Execute(resource, &output));
  EXPECT_THAT(output, testing::ElementsAre(5.0f, 7.0f, 9.0f));

  ASSERT_OK(tsl::Env::Default()->DeleteFile(table_path_));
  ASSERT_OK(Execute(resource, &output));
  EXPECT_THAT(output, testing::ElementsAre(5.0f, 7.0f, 9.0f));
}

// Verify mapped-file errors and exact-size validation.
TEST_F(MappedCpuEmbeddingResourceTest, RejectsMissingAndTruncatedTables) {
  ASSERT_OK(WriteTable(absl::MakeConstSpan(table_data_).first(3)));
  CpuEmbeddingEngineResource truncated(Assets());
  absl::Status status = truncated.Load();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), testing::HasSubstr("size mismatch"));
  EXPECT_TRUE(truncated.assets()[1].content.empty());

  ASSERT_OK(tsl::Env::Default()->DeleteFile(table_path_));
  CpuEmbeddingEngineResource missing(Assets());
  status = missing.Load();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Failed to map table asset"));
  EXPECT_THAT(status.message(), testing::HasSubstr("mapped_table.bin"));
}

// Verify failed loads leave the resource retryable.
TEST_F(MappedCpuEmbeddingResourceTest, FailedLoadCanBeRetriedTransactionally) {
  auto* feature = manifest_.add_feature_specs();
  feature->set_name("feature2");
  feature->set_table("missing_table");
  ASSERT_OK(WriteManifest());

  CpuEmbeddingEngineResource resource(Assets());
  EXPECT_FALSE(resource.Load().ok());
  manifest_.mutable_feature_specs(1)->set_table("table1");
  ASSERT_OK(WriteManifest());
  ASSERT_OK(resource.Load());
  EXPECT_OK(resource.GetEmbeddingDim("feature1"));
  EXPECT_OK(resource.GetEmbeddingDim("feature2"));
}

}  // namespace
}  // namespace composite
}  // namespace xla_serving::data
