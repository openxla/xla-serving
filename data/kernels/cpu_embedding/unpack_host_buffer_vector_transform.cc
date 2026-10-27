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

#include "data/kernels/cpu_embedding/unpack_host_buffer_vector_transform.h"  // from @xla_serving

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/cpu_embedding/cpu_embedding.pb.h"  // from @xla_serving
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace xla_serving {
namespace data {
namespace composite {
namespace transforms {

absl::StatusOr<std::unique_ptr<UnpackHostBufferVectorTransform>>
UnpackHostBufferVectorTransform::Create(int output_count) {
  if (output_count <= 0) {
    return absl::InvalidArgumentError("output_count must be greater than 0");
  }
  return std::make_unique<UnpackHostBufferVectorTransform>(output_count);
}

UnpackHostBufferVectorTransform::UnpackHostBufferVectorTransform(
    int output_count)
    : output_count_(output_count) {
  input_types_.push_back(api::CreateTypeInfo<std::vector<xla_serving::HostBuffer>>());
  output_types_.resize(output_count_, api::CreateTypeInfo<xla_serving::HostBuffer>());
}

absl::Status UnpackHostBufferVectorTransform::Execute(
    api::ExecuteContext& context) {
  const std::vector<xla_serving::HostBuffer>& packed =
      context.inputs[0].Get<std::vector<xla_serving::HostBuffer>>();

  if (packed.size() != output_count_) {
    return absl::InternalError(absl::StrCat(
        "Execute error: UnpackHostBufferVectorTransform got a vector "
        "of size ",
        packed.size(), " but expected size ", output_count_));
  }

  for (const auto& val : packed) {
    context.outputs.push_back(api::Value(val));
  }
  return absl::OkStatus();
}

absl::Status UnpackHostBufferVectorTransform::PropagateSignatureImpl(
    api::ShapeInferenceContext& context) const {
  if (context.inputs.empty() || !context.inputs[0].IsSpecList()) {
    return absl::InvalidArgumentError(
        "UnpackHostBufferVectorTransform expected SpecList input.");
  }
  for (const auto& spec : context.inputs[0].spec_list()) {
    context.outputs.push_back(api::ShapeInferenceValue(spec));
  }
  return absl::OkStatus();
}

absl::StatusOr<::xla_serving::data::composite::TransformConfig>
UnpackHostBufferVectorTransform::ToProto() const {
  ::xla_serving::data::composite::TransformConfig config;
  config.set_op_name("UnpackHostBufferVectorTransform");
  ::xla_serving::data::composite::transforms::cpu_embedding::
      UnpackHostBufferVectorArgs args;

  args.set_output_count(output_count_);
  config.mutable_detail()->PackFrom(args);
  return config;
}

std::string UnpackHostBufferVectorTransform::DebugName() const {
  return "UnpackHostBufferVectorTransform";
}

ND_REGISTER_TRANSFORM(UnpackHostBufferVectorTransform,
                      "UnpackHostBufferVectorTransform")
    .SetDeserializerWithDetail<::xla_serving::data::composite::transforms::
                                   cpu_embedding::UnpackHostBufferVectorArgs>(
        [](const ::xla_serving::data::composite::transforms::cpu_embedding::
               UnpackHostBufferVectorArgs& args) {
          return UnpackHostBufferVectorTransform::Create(args.output_count());
        });

}  // namespace transforms
}  // namespace composite
}  // namespace data
}  // namespace xla_serving
