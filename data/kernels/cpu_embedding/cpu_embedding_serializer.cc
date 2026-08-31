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

#include "data/kernels/cpu_embedding/cpu_embedding_serializer.h"  // from @neptune

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/match.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @neptune
#include "data/resource_manager/cc/asset.h"  // from @neptune
#include "xla/tsl/platform/env.h"  // from @xla

namespace neptune::data {
namespace composite {

absl::StatusOr<std::vector<Asset>> SerializeCpuEmbedding(
    std::string_view dir_path,
    const absl::flat_hash_map<std::string, cpu_embedding::Array>& tables,
    std::string_view manifest_proto_text,
    const absl::flat_hash_map<std::string, std::string>& custom_combiners) {
  std::vector<Asset> assets;
  assets.reserve(1 + tables.size() + custom_combiners.size());

  std::string manifest_path = absl::StrCat(dir_path, "/manifest.textproto");
  RETURN_IF_ERROR(tsl::WriteStringToFile(tsl::Env::Default(), manifest_path,
                                         manifest_proto_text));
  assets.push_back(
      Asset{.name = std::string(CpuEmbeddingEngineResource::kManifestAssetName),
            .path = manifest_path});

  // Serialize each table array to name.bin
  for (const auto& [name, array] : tables) {
    if (absl::StrContains(name, '/') || absl::StrContains(name, '\\') ||
        absl::StrContains(name, "..")) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid table name (path traversal detected): ", name));
    }
    if (array.GetDType() != cpu_embedding::DType::kFloat32) {
      return absl::InvalidArgumentError(
          absl::StrCat("Only FLOAT32 tables are supported for serialization, "
                       "table ",
                       name, " has type ", static_cast<int>(array.GetDType())));
    }
    std::string table_path = absl::StrCat(dir_path, "/", name, ".bin");
    absl::Span<const float> span = array.AsSpan<float>();
    RETURN_IF_ERROR(tsl::WriteStringToFile(
        tsl::Env::Default(), table_path,
        absl::string_view(reinterpret_cast<const char*>(span.data()),
                          span.size() * sizeof(float))));
    assets.push_back(Asset{.name = name, .path = table_path});
  }

  // Serialize custom combiner assets (e.g. .mlirbc, .mlir, .pb, .hlo)
  for (const auto& [asset_name, content] : custom_combiners) {
    if (absl::StrContains(asset_name, '/') ||
        absl::StrContains(asset_name, '\\') ||
        absl::StrContains(asset_name, "..")) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid combiner asset name (path traversal detected): ",
          asset_name));
    }
    std::string asset_path = absl::StrCat(dir_path, "/", asset_name);
    RETURN_IF_ERROR(
        tsl::WriteStringToFile(tsl::Env::Default(), asset_path, content));
    assets.push_back(Asset{.name = asset_name, .path = asset_path});
  }

  return assets;
}

}  // namespace composite
}  // namespace neptune::data
