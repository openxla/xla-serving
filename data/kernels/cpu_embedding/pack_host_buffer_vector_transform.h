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

#ifndef THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_PACK_HOST_BUFFER_VECTOR_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_PACK_HOST_BUFFER_VECTOR_TRANSFORM_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune

namespace neptune {
namespace data {
namespace composite {
namespace transforms {

// Packs N individual HostBuffers into a single std::vector<HostBuffer>.
class PackHostBufferVectorTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<PackHostBufferVectorTransform>> Create(
      int input_count);

  static absl::StatusOr<std::unique_ptr<PackHostBufferVectorTransform>>
  CreateFromTransformConfig(
      const ::neptune::data::composite::TransformConfig& config,
      absl::Span<const std::shared_ptr<neptune::data::Resource>> resources);

  explicit PackHostBufferVectorTransform(int input_count);

  absl::Status Execute(api::ExecuteContext& context) override;

  absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const override;

  absl::Span<const api::TypeInfo> InputTypes() const override {
    return input_types_;
  }

  absl::Span<const api::TypeInfo> OutputTypes() const override {
    return output_types_;
  }

  absl::StatusOr<::neptune::data::composite::TransformConfig> ToProto()
      const override;

  std::string DebugName() const override;

 private:
  int input_count_;
  std::vector<api::TypeInfo> input_types_;
  std::vector<api::TypeInfo> output_types_;
};

}  // namespace transforms
}  // namespace composite
}  // namespace data
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_PACK_HOST_BUFFER_VECTOR_TRANSFORM_H_
