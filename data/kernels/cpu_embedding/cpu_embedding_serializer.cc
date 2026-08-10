#include "data/kernels/cpu_embedding/cpu_embedding_serializer.h"  // from @neptune

#include <fstream>
#include <ios>
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
#include "data/kernels/cpu_embedding/cpu_embedding.pb.h"  // from @neptune
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @neptune
#include "data/resource_manager/cc/asset.h"  // from @neptune
#include "xla/tsl/platform/env.h"  // from @xla

namespace neptune::data {
namespace composite {

absl::StatusOr<std::vector<Asset>> SerializeCpuEmbedding(
    std::string_view dir_path,
    const absl::flat_hash_map<std::string, cpu_embedding::Array>& tables,
    const ::neptune::data::composite::transforms::cpu_embedding::
        CpuEmbeddingManifest& manifest) {
  std::vector<Asset> assets;
  assets.reserve(1 + tables.size());

  // Serialize manifest proto to manifest.textproto
  std::string manifest_path = absl::StrCat(dir_path, "/manifest.textproto");
  RETURN_IF_ERROR(
      tsl::WriteTextProto(tsl::Env::Default(), manifest_path, manifest));
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
    std::ofstream table_file(table_path, std::ios::binary);
    if (!table_file.is_open()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to open table file for writing: ", table_path));
    }
    absl::Span<const float> span = array.AsSpan<float>();
    table_file.write(reinterpret_cast<const char*>(span.data()),
                     span.size() * sizeof(float));
    table_file.close();
    if (!table_file) {
      return absl::InternalError(
          absl::StrCat("Failed to write table file: ", table_path));
    }
    assets.push_back(Asset{.name = name, .path = table_path});
  }

  return assets;
}

}  // namespace composite
}  // namespace neptune::data
