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

#ifndef XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_DENSE_CPU_EMBEDDING_LOOKUP_TRANSFORM_H_
#define XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_DENSE_CPU_EMBEDDING_LOOKUP_TRANSFORM_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @neptune
#include "data/kernels/cpu_embedding/dense_cpu_embedding_lookup.pb.h"  // from @neptune
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune

namespace neptune::data::composite {

// Transform for high-performance dense CPU embedding lookup.
//
// Accepts 2D dense ID tensors of shape [batch_size, sequence_length] (dtype S32
// or S64) for one or more features. Converts non-padding tokens (where ID !=
// pad_value) into CSR representation, executes parallel multi-threaded lookups
// with pooling on CpuEmbeddingEngine, and returns dense float32 tensors of
// shape [batch_size, embedding_dim]. If a sequence consists entirely of
// padding tokens, a single pad_value token is retained to ensure valid CSR
// bounds.
class DenseCpuEmbeddingLookupTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<DenseCpuEmbeddingLookupTransform>>
  Create(std::shared_ptr<CpuEmbeddingEngineResource> resource,
         std::vector<std::string> feature_names, int64_t pad_value = 0,
         std::vector<std::string> output_feature_names = {});

  static absl::StatusOr<std::unique_ptr<DenseCpuEmbeddingLookupTransform>>
  CreateForFeature(std::shared_ptr<CpuEmbeddingEngineResource> resource,
                   std::string feature_name, int64_t pad_value = 0);

  std::vector<std::shared_ptr<neptune::data::Resource>> GetAllResources()
      const override;

  absl::Status Execute(api::ExecuteContext& context) override;
  absl::Span<const api::TypeInfo> InputTypes() const override;
  absl::Span<const api::TypeInfo> OutputTypes() const override;

  absl::StatusOr<std::vector<xla_serving::HostBuffer>> operator()(
      std::vector<xla_serving::HostBuffer> dense_ids_list);

  absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const override;

  absl::StatusOr<TransformConfig> ToProto() const override;

  absl::string_view DebugString() const override { return debug_string_; }

  const std::vector<std::string>& feature_names() const {
    return feature_names_;
  }
  int64_t pad_value() const { return pad_value_; }

 private:
  DenseCpuEmbeddingLookupTransform(
      std::shared_ptr<CpuEmbeddingEngineResource> resource,
      std::vector<std::string> feature_names, int64_t pad_value,
      std::vector<std::string> output_feature_names);

  std::shared_ptr<CpuEmbeddingEngineResource> resource_;
  std::vector<std::string> feature_names_;
  int64_t pad_value_ = 0;
  std::vector<std::string> output_feature_names_;
  std::vector<api::TypeInfo> input_types_;
  std::vector<api::TypeInfo> output_types_;
  std::string debug_string_;
};

}  // namespace neptune::data::composite

#endif  // XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_DENSE_CPU_EMBEDDING_LOOKUP_TRANSFORM_H_
