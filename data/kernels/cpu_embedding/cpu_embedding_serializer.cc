#include "data/kernels/cpu_embedding/cpu_embedding_serializer.h"  // from @neptune

#include <cstdint>
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
#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @neptune
#include "data/resource_manager/cc/asset.h"  // from @neptune
#include "google/protobuf/text_format.h"
#include "xla/tsl/platform/env.h"  // from @xla

namespace neptune::data {
namespace composite {

absl::StatusOr<std::vector<Asset>> SerializeCpuEmbedding(
    std::string_view dir_path,
    const absl::flat_hash_map<std::string, cpu_embedding::Array>& tables,
    std::string_view manifest_proto_text) {
  std::vector<Asset> assets;
  assets.reserve(1 + tables.size());

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
    int64_t byte_width = cpu_embedding::ByteWidth(array.GetDType());
    if (byte_width <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported DType for serialization: ",
                       static_cast<int>(array.GetDType())));
    }
    RETURN_IF_ERROR(tsl::WriteStringToFile(
        tsl::Env::Default(), table_path,
        absl::string_view(reinterpret_cast<const char*>(array.Data()),
                          array.GetShape().NumElements() * byte_width)));
    assets.push_back(Asset{.name = name, .path = table_path});
  }

  return assets;
}

}  // namespace composite
}  // namespace neptune::data
