#ifndef THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_SERIALIZER_H_
#define THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_SERIALIZER_H_

#include <string>
#include <vector>

#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/kernels/cpu_embedding/cpu_embedding.pb.h"  // from @neptune
#include "data/resource_manager/cc/asset.h"  // from @neptune
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp

namespace neptune::data {
namespace composite {

absl::StatusOr<std::vector<Asset>> SerializeCpuEmbedding(
    std::string_view dir_path,
    const absl::flat_hash_map<std::string, cpu_embedding::Array>& tables,
    const ::neptune::data::composite::transforms::cpu_embedding::
        CpuEmbeddingManifest& manifest);

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_SERIALIZER_H_
