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

#ifndef XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_LOOKUP_TRANSFORM_H_
#define XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_LOOKUP_TRANSFORM_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace neptune::data {
namespace composite {

class CpuEmbeddingLookupTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<CpuEmbeddingLookupTransform>> Create(
      std::shared_ptr<CpuEmbeddingEngineResource> resource,
      std::vector<std::string> feature_names, bool dense_lookup = false,
      std::vector<std::string> output_feature_names = {},
      int64_t embedding_dim = 128,
      std::vector<std::vector<int64_t>> output_shapes = {});

  std::vector<std::shared_ptr<neptune::data::Resource>> GetAllResources()
      const override;

  absl::StatusOr<std::vector<xla_serving::HostBuffer>> operator()(
      std::vector<xla_serving::HostBuffer> ids,
      std::vector<xla_serving::HostBuffer> row_pointers,
      std::vector<xla_serving::HostBuffer> weights);

  absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const override;

  absl::StatusOr<TransformConfig> ToProto() const override;

  absl::string_view DebugString() const override {
    return "CpuEmbeddingLookupTransform";
  }

 private:
  CpuEmbeddingLookupTransform(
      std::shared_ptr<CpuEmbeddingEngineResource> resource,
      std::vector<std::string> feature_names,
      std::vector<std::string> output_feature_names, bool dense_lookup,
      int64_t embedding_dim, std::vector<std::vector<int64_t>> output_shapes);

  std::shared_ptr<CpuEmbeddingEngineResource> resource_;
  std::vector<std::string> feature_names_;
  std::vector<std::string> output_feature_names_;
  bool dense_lookup_ = false;
  int64_t embedding_dim_ = 128;
  std::vector<std::vector<int64_t>> output_shapes_;
};

}  // namespace composite
}  // namespace neptune::data

#endif  // XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_LOOKUP_TRANSFORM_H_
