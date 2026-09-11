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

#ifndef XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_RESOURCE_H_
#define XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_RESOURCE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/cpu_embedding_engine.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving

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

#endif  // XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_RESOURCE_H_
