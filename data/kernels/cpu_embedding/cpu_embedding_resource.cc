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

#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @neptune

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/cpu_embedding/cc/cpu_embedding_engine.h"  // from @neptune
#include "data/cpu_embedding/cc/embedding_table.h"  // from @neptune
#include "data/cpu_embedding/cc/feature_spec.h"  // from @neptune
#include "data/kernels/cpu_embedding/cpu_embedding.pb.h"  // from @neptune
#include "data/kernels/cpu_embedding/safe_int_utils.h"  // from @neptune
#include "data/resource_manager/cc/asset.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "xla/tsl/platform/env.h"  // from @xla

namespace neptune::data {
namespace composite {

namespace {

using transforms::cpu_embedding::CpuEmbeddingManifest;

absl::Status LoadCpuEmbeddingEngine(
    Asset& manifest,
    const absl::flat_hash_map<absl::string_view, Asset* absl_nonnull>&
        table_assets,
    cpu_embedding::CpuEmbeddingEngine* engine,
    absl::flat_hash_map<std::string, int32_t>& feature_to_embedding_dim) {
  if (manifest.content.empty()) {
    RETURN_IF_ERROR(manifest.LoadContent());
  }

  CpuEmbeddingManifest manifest_proto;
  RETURN_IF_ERROR(
      tsl::ReadTextProto(tsl::Env::Default(), manifest.path, &manifest_proto));

  absl::flat_hash_map<std::string, int32_t> table_to_dim;
  for (const auto& table_config : manifest_proto.tables()) {
    auto it = table_assets.find(table_config.name());
    if (it == table_assets.end()) {
      return absl::NotFoundError(
          absl::StrCat("Failed to find table asset: ", table_config.name()));
    }
    Asset* table_asset = it->second;
    if (table_asset->content.empty()) {
      RETURN_IF_ERROR(table_asset->LoadContent());
    }

    int64_t vocab_size = table_config.vocab_size();
    int64_t embedding_dim = table_config.embedding_dim();
    if (vocab_size <= 0 || embedding_dim <= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Table dimensions must be strictly positive. Got vocab_size: ",
          vocab_size, ", embedding_dim: ", embedding_dim));
    }

    table_to_dim[table_config.name()] = table_config.embedding_dim();

    size_t intermediate_bytes = 0;
    if (!utils::SafeMultiply(static_cast<size_t>(vocab_size),
                             static_cast<size_t>(embedding_dim),
                             &intermediate_bytes)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Table dimensions too large, multiplication overflowed: ", vocab_size,
          " * ", embedding_dim));
    }

    size_t expected_bytes = 0;
    if (!utils::SafeMultiply(intermediate_bytes, sizeof(float),
                             &expected_bytes)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Total byte size too large, multiplication overflowed: ",
                       intermediate_bytes, " * ", sizeof(float)));
    }

    absl::string_view bytes(table_asset->content);
    if (bytes.size() != expected_bytes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Table asset size mismatch for ", table_config.name(), ". Expected ",
          expected_bytes, " bytes, but got ", bytes.size()));
    }

    const float* raw_table_data = reinterpret_cast<const float*>(bytes.data());
    cpu_embedding::Shape shape(
        {table_config.vocab_size(), table_config.embedding_dim()});

    auto array = cpu_embedding::Array::Wrap(
        const_cast<float*>(raw_table_data), shape);

    cpu_embedding::EmbeddingTable::CombineMode combine_mode;
    switch (table_config.combine_mode()) {
      case CpuEmbeddingManifest::EmbeddingTable::COMBINE_MODE_UNSPECIFIED:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unspecified combine mode for table: ", table_config.name()));
      case CpuEmbeddingManifest::EmbeddingTable::SUM:
        combine_mode = cpu_embedding::EmbeddingTable::CombineMode::kSum;
        break;
      case CpuEmbeddingManifest::EmbeddingTable::MEAN:
        combine_mode = cpu_embedding::EmbeddingTable::CombineMode::kMean;
        break;
      case CpuEmbeddingManifest::EmbeddingTable::SQRT_N:
        combine_mode = cpu_embedding::EmbeddingTable::CombineMode::kSqrtN;
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unknown combine mode for table: ", table_config.name()));
    }

    auto table_status = cpu_embedding::EmbeddingTable::Create(
        table_config.name(), std::move(array), combine_mode);
    if (!table_status.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to create embedding table: ",
                       table_status.status().message()));
    }
    RETURN_IF_ERROR(engine->AddLocalTable(std::string(table_config.name()),
                                          std::move(table_status).value()));
  }

  for (const auto& feature : manifest_proto.feature_specs()) {
    cpu_embedding::FeatureSpec spec;
    spec.name = feature.name();
    spec.table_name = feature.table();
    spec.max_sequence_length = feature.max_sequence_length();
    RETURN_IF_ERROR(engine->RegisterFeatureSpec(spec));

    auto it = table_to_dim.find(feature.table());
    if (it == table_to_dim.end()) {
      return absl::NotFoundError(absl::StrCat("Failed to find table ",
                                              feature.table(), " for feature ",
                                              feature.name()));
    }
    auto [dim_it, dim_inserted] =
        feature_to_embedding_dim.emplace(feature.name(), it->second);
    if (!dim_inserted) {
      return absl::AlreadyExistsError(
          absl::StrCat("Duplicate feature name in manifest: ", feature.name()));
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status CpuEmbeddingEngineResource::LoadImpl() {
  Asset* manifest_asset = nullptr;
  absl::flat_hash_map<absl::string_view, Asset* absl_nonnull> table_assets;
  for (Asset& asset : assets_) {
    if (asset.name == CpuEmbeddingEngineResource::kManifestAssetName) {
      manifest_asset = &asset;
    } else {
      auto [_, inserted] = table_assets.insert({asset.name, &asset});
      if (!inserted) {
        return absl::InvalidArgumentError(
            absl::StrCat("Duplicate table asset found for ", asset.name));
      }
    }
  }
  if (manifest_asset == nullptr) {
    return absl::NotFoundError(
        "Failed to find manifest asset in JCE Resource.");
  }

  return LoadCpuEmbeddingEngine(*manifest_asset, table_assets, engine_.get(),
                                feature_to_embedding_dim_);
}

absl::StatusOr<int32_t> CpuEmbeddingEngineResource::GetEmbeddingDim(
    absl::string_view feature_name) const {
  auto it = feature_to_embedding_dim_.find(feature_name);
  if (it == feature_to_embedding_dim_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Embedding dimension not found for feature: ", feature_name));
  }
  return it->second;
}

ND_REGISTER_RESOURCE(CpuEmbeddingEngineResource, "CpuEmbeddingEngineResource");

}  // namespace composite
}  // namespace neptune::data
