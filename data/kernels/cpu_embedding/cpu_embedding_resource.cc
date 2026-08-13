#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @neptune

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include "google/protobuf/text_format.h"
#include "xla/tsl/platform/env.h"  // from @xla

namespace neptune::data {
namespace composite {

namespace {

using transforms::cpu_embedding::CpuEmbeddingManifest;

absl::StatusOr<std::unique_ptr<cpu_embedding::CpuEmbeddingEngine>>
LoadCpuEmbeddingEngine(
    Asset& manifest,
    const absl::flat_hash_map<absl::string_view, Asset* absl_nonnull>&
        table_assets,
    absl::flat_hash_map<std::string, int32_t>& feature_to_embedding_dim) {
  if (manifest.content.empty()) {
    RETURN_IF_ERROR(manifest.LoadContent());
  }

  CpuEmbeddingManifest manifest_proto;
  RETURN_IF_ERROR(
      tsl::ReadTextProto(tsl::Env::Default(), manifest.path, &manifest_proto));

  cpu_embedding::CpuEmbeddingEngine::Options options;

  const char* env_max_cpu_slots = std::getenv("JCE_MAX_CPU_SLOTS");
  if (env_max_cpu_slots != nullptr) {
    options.max_cpu_slots = std::atoi(env_max_cpu_slots);
  } else if (manifest_proto.max_cpu_slots() > 0) {
    options.max_cpu_slots = manifest_proto.max_cpu_slots();
  }

  const char* env_block_size = std::getenv("JCE_BLOCK_SIZE");
  if (env_block_size != nullptr) {
    options.block_size = std::atoi(env_block_size);
  } else if (manifest_proto.block_size() > 0) {
    options.block_size = manifest_proto.block_size();
  }

  auto engine = std::make_unique<cpu_embedding::CpuEmbeddingEngine>(options);

  absl::flat_hash_map<std::string, int64_t> table_to_dim;
  int64_t block_size = options.block_size;
  for (const auto& table_config : manifest_proto.tables()) {
    int64_t vocab_size = table_config.vocab_size();
    int64_t embedding_dim = table_config.embedding_dim();
    if (vocab_size <= 0 || embedding_dim <= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Table dimensions must be strictly positive. Got vocab_size: ",
          vocab_size, ", embedding_dim: ", embedding_dim));
    }

    size_t intermediate_bytes = 0;
    if (!utils::SafeMultiply(static_cast<size_t>(vocab_size),
                             static_cast<size_t>(embedding_dim),
                             &intermediate_bytes)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Table dimensions too large, multiplication overflowed: ", vocab_size,
          " * ", embedding_dim));
    }

    size_t element_size =
        table_config.is_quantized() ? sizeof(int8_t) : sizeof(float);
    size_t expected_bytes = 0;
    if (!utils::SafeMultiply(intermediate_bytes, element_size,
                             &expected_bytes)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Total byte size too large, multiplication overflowed: ",
                       intermediate_bytes, " * ", element_size));
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

    absl::StatusOr<std::unique_ptr<cpu_embedding::EmbeddingTable>> table_status;

    if (table_config.is_quantized()) {
      auto it_data = table_assets.find(table_config.name());
      if (it_data == table_assets.end()) {
        return absl::NotFoundError(absl::StrCat(
            "Failed to find quantized data asset: ", table_config.name()));
      }
      Asset* data_asset = it_data->second;
      if (data_asset->content.empty()) {
        RETURN_IF_ERROR(data_asset->LoadContent());
      }
      size_t expected_data_bytes = vocab_size * embedding_dim;
      absl::string_view data_bytes(data_asset->content);
      if (data_bytes.size() != expected_data_bytes) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Data asset size mismatch for ", table_config.name(), ". Expected ",
            expected_data_bytes, " bytes, but got ", data_bytes.size()));
      }
      auto data_arr = cpu_embedding::Array::Wrap(
          const_cast<int8_t*>(
              reinterpret_cast<const int8_t*>(data_bytes.data())),
          cpu_embedding::Shape({vocab_size, embedding_dim}));

      std::string scales_name = absl::StrCat(table_config.name(), ".scales");
      auto it_scales = table_assets.find(scales_name);
      if (it_scales == table_assets.end()) {
        return absl::NotFoundError(
            absl::StrCat("Failed to find scales asset: ", scales_name));
      }
      Asset* scales_asset = it_scales->second;
      if (scales_asset->content.empty()) {
        RETURN_IF_ERROR(scales_asset->LoadContent());
      }
      int64_t num_blocks = (embedding_dim + block_size - 1) / block_size;
      size_t expected_scales_bytes = vocab_size * num_blocks * sizeof(float);
      absl::string_view scales_bytes(scales_asset->content);
      if (scales_bytes.size() != expected_scales_bytes) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Scales asset size mismatch for ", scales_name, ". Expected ",
            expected_scales_bytes, " bytes, but got ", scales_bytes.size()));
      }
      auto scales_arr = cpu_embedding::Array::Wrap(
          const_cast<float*>(
              reinterpret_cast<const float*>(scales_bytes.data())),
          cpu_embedding::Shape({vocab_size, num_blocks}));

      std::string offsets_name = absl::StrCat(table_config.name(), ".offsets");
      auto it_offsets = table_assets.find(offsets_name);
      cpu_embedding::Array offsets_arr;
      if (it_offsets != table_assets.end()) {
        Asset* offsets_asset = it_offsets->second;
        if (offsets_asset->content.empty()) {
          RETURN_IF_ERROR(offsets_asset->LoadContent());
        }
        size_t expected_offsets_bytes = vocab_size * num_blocks * sizeof(float);
        absl::string_view offsets_bytes(offsets_asset->content);
        if (offsets_bytes.size() != expected_offsets_bytes) {
          return absl::InvalidArgumentError(
              absl::StrCat("Offsets asset size mismatch for ", offsets_name,
                           ". Expected ", expected_offsets_bytes,
                           " bytes, but got ", offsets_bytes.size()));
        }
        offsets_arr = cpu_embedding::Array::Wrap(
            const_cast<float*>(
                reinterpret_cast<const float*>(offsets_bytes.data())),
            cpu_embedding::Shape({vocab_size, num_blocks}));
      }

      table_status = cpu_embedding::EmbeddingTable::CreateQuantized(
          table_config.name(), std::move(data_arr), std::move(scales_arr),
          std::move(offsets_arr), block_size, combine_mode);
    } else {
      auto it = table_assets.find(table_config.name());
      if (it == table_assets.end()) {
        return absl::NotFoundError(
            absl::StrCat("Failed to find table asset: ", table_config.name()));
      }
      Asset* table_asset = it->second;
      if (table_asset->content.empty()) {
        RETURN_IF_ERROR(table_asset->LoadContent());
      }
      size_t expected_bytes = vocab_size * embedding_dim * sizeof(float);
      absl::string_view bytes(table_asset->content);
      if (bytes.size() != expected_bytes) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Table asset size mismatch for ", table_config.name(),
            ". Expected ", expected_bytes, " bytes, but got ", bytes.size()));
      }
      auto array = cpu_embedding::Array::Wrap(
          const_cast<float*>(reinterpret_cast<const float*>(bytes.data())),
          cpu_embedding::Shape({vocab_size, embedding_dim}));

      table_status = cpu_embedding::EmbeddingTable::Create(
          table_config.name(), std::move(array), combine_mode);
    }

    if (!table_status.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to create embedding table: ",
                       table_status.status().message()));
    }
    RETURN_IF_ERROR(engine->AddLocalTable(std::string(table_config.name()),
                                          std::move(table_status).value()));
    table_to_dim[table_config.name()] = table_config.embedding_dim();
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

  return engine;
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

  ASSIGN_OR_RETURN(engine_,
                   LoadCpuEmbeddingEngine(*manifest_asset, table_assets,
                                          feature_to_embedding_dim_));
  return absl::OkStatus();
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
