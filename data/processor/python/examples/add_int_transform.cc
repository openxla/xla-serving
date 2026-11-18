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

#include "data/processor/python/examples/add_int_transform.h"  // from @xla_serving

#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/api/shape.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/core/transform_registry.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {

absl::StatusOr<std::unique_ptr<AddIntTransform>> AddIntTransform::Create() {
  return std::make_unique<AddIntTransform>();
}

int AddIntTransform::operator()(int a, int b) { return a + b; }

absl::StatusOr<TransformConfig> AddIntTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("AddIntTransform");
  return config;
}

ND_REGISTER_TRANSFORM(AddIntTransform, "AddIntTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<xla_serving::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return std::make_unique<AddIntTransform>();
        });

absl::Status AddIntTransform::PropagateSignatureImpl(
    api::ShapeInferenceContext& context) const {
  if (context.inputs.size() != 2) {
    return absl::InvalidArgumentError("AddIntTransform expects 2 inputs");
  }
  if (!context.inputs[0].IsSpec() || !context.inputs[1].IsSpec()) {
    return absl::InvalidArgumentError(
        "AddIntTransform expects single spec inputs");
  }
  const auto& spec0 = context.inputs[0].spec();
  const auto& spec1 = context.inputs[1].spec();
  absl::StatusOr<api::PartialShape> merged_shape =
      spec0.shape.MergeWith(spec1.shape);
  if (!merged_shape.ok()) {
    return absl::InvalidArgumentError("Inputs must have the same shape");
  }
  if (spec0.dtype != spec1.dtype) {
    return absl::InvalidArgumentError("Inputs must have the same dtype");
  }
  context.outputs.push_back(
      api::HostBufferSpec{spec0.dtype, *std::move(merged_shape)});
  return absl::OkStatus();
}

}  // namespace composite
}  // namespace xla_serving::data
