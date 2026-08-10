#ifndef THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_RESOURCE_H_
#define THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_RESOURCE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/cpu_embedding_engine.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {
namespace composite {

class CpuEmbeddingEngineResource : public Resource {
 public:
  using Resource::Resource;

  static constexpr absl::string_view kManifestAssetName = "__jce_manifest__";

  cpu_embedding::CpuEmbeddingEngine* Get() const { return engine_.get(); }

  absl::StatusOr<int32_t> GetEmbeddingDim(absl::string_view feature_name) const;

  void RegisterEmbeddingDimForTest(absl::string_view feature_name,
                                   int32_t embedding_dim) {
    feature_to_embedding_dim_[std::string(feature_name)] = embedding_dim;
  }

 protected:
  absl::Status LoadImpl() override;

  absl::flat_hash_map<std::string, int32_t> feature_to_embedding_dim_;

 private:
  std::unique_ptr<cpu_embedding::CpuEmbeddingEngine> engine_ =
      std::make_unique<cpu_embedding::CpuEmbeddingEngine>();
};

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_RESOURCE_H_
