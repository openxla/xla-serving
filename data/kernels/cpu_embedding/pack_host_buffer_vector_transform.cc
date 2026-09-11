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

#include "data/kernels/cpu_embedding/pack_host_buffer_vector_transform.h"  // from @neptune

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/cpu_embedding/cpu_embedding.pb.h"  // from @neptune
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune

namespace neptune {
namespace data {
namespace composite {
namespace transforms {

absl::StatusOr<std::unique_ptr<PackHostBufferVectorTransform>>
PackHostBufferVectorTransform::Create(int input_count) {
  if (input_count < 0) {
    return absl::InvalidArgumentError("input_count must be non-negative");
  }
  return std::make_unique<PackHostBufferVectorTransform>(input_count);
}

PackHostBufferVectorTransform::PackHostBufferVectorTransform(int input_count)
    : input_count_(input_count) {
  input_types_.resize(input_count_, api::CreateTypeInfo<xla_serving::HostBuffer>());
  output_types_.push_back(api::CreateTypeInfo<std::vector<xla_serving::HostBuffer>>());
}

absl::Status PackHostBufferVectorTransform::Execute(
    api::ExecuteContext& context) {
  std::vector<xla_serving::HostBuffer> packed;
  packed.reserve(context.inputs.size());
  for (const auto& val : context.inputs) {
    packed.push_back(val.Get<xla_serving::HostBuffer>());
  }
  context.outputs.push_back(api::Value(std::move(packed)));
  return absl::OkStatus();
}

absl::Status PackHostBufferVectorTransform::PropagateSignatureImpl(
    api::ShapeInferenceContext& context) const {
  std::vector<api::HostBufferSpec> specs;
  specs.reserve(context.inputs.size());
  for (const auto& in_val : context.inputs) {
    if (in_val.IsSpec()) {
      specs.push_back(in_val.spec());
    } else {
      return absl::InvalidArgumentError(
          "PackHostBufferVectorTransform expected HostBufferSpec input.");
    }
  }
  context.outputs.push_back(api::ShapeInferenceValue(std::move(specs)));
  return absl::OkStatus();
}

absl::StatusOr<::neptune::data::composite::TransformConfig>
PackHostBufferVectorTransform::ToProto() const {
  ::neptune::data::composite::TransformConfig config;
  config.set_op_name("PackHostBufferVectorTransform");
  ::neptune::data::composite::transforms::cpu_embedding::
      PackHostBufferVectorArgs args;

  args.set_input_count(input_count_);
  config.mutable_detail()->PackFrom(args);
  return config;
}

std::string PackHostBufferVectorTransform::DebugName() const {
  return "PackHostBufferVectorTransform";
}

absl::StatusOr<std::unique_ptr<PackHostBufferVectorTransform>>
PackHostBufferVectorTransform::CreateFromTransformConfig(
    const ::neptune::data::composite::TransformConfig& config,
    absl::Span<const std::shared_ptr<neptune::data::Resource>> resources) {
  ::neptune::data::composite::transforms::cpu_embedding::
      PackHostBufferVectorArgs args;

  if (!config.detail().UnpackTo(&args)) {
    return absl::InvalidArgumentError(
        "Failed to unpack PackHostBufferVectorArgs from config.");
  }
  return Create(args.input_count());
}

ND_REGISTER_TRANSFORM(PackHostBufferVectorTransform,
                      "PackHostBufferVectorTransform")
    .SetDeserializer(&PackHostBufferVectorTransform::CreateFromTransformConfig);

}  // namespace transforms
}  // namespace composite
}  // namespace data
}  // namespace neptune
