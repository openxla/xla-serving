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

#include "data/processor/cc/core/transform.h"  // from @neptune

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/cc/api/value.h"  // from @neptune
#include "data/processor/cc/core/transform_registry.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune

namespace neptune::data {
namespace composite {

absl::StatusOr<std::unique_ptr<TransformBase>> TransformBase::FromProto(
    const TransformConfig& config,
    absl::Span<const std::shared_ptr<neptune::data::Resource>> resources) {
  return core::TransformRegistry::Instance().DeserializeTransform(config,
                                                                  resources);
}

absl::Status TransformBase::Execute(api::ExecuteContext& context) {
  ASSIGN_OR_RETURN(
      const core::TransformFactoryHolder* absl_nonnull holder,
      core::TransformRegistry::Instance().GetTransformFactoryHolder(
          typeid(*this)));
  if (!holder->HasExecutor()) {
    return absl::UnimplementedError(
        "Executor not registered for this transform. Did you forget to call "
        "SetExecutor() during registration, or did you forget to override "
        "TransformBase::Execute()?");
  }
  return holder->Execute(this, context);
}

absl::Span<const api::TypeInfo> TransformBase::InputTypes() const {
  absl::StatusOr<const core::TransformFactoryHolder* absl_nonnull> holder =
      core::TransformRegistry::Instance().GetTransformFactoryHolder(
          typeid(*this));
  if (!holder.ok()) {
    return {};
  }
  return (*holder)->InputTypes();
}

absl::Span<const api::TypeInfo> TransformBase::OutputTypes() const {
  absl::StatusOr<const core::TransformFactoryHolder* absl_nonnull> holder =
      core::TransformRegistry::Instance().GetTransformFactoryHolder(
          typeid(*this));
  if (!holder.ok()) {
    return {};
  }
  return (*holder)->OutputTypes();
}

absl::Status TransformBase::PropagateSignature(
    api::ShapeInferenceContext& context) const {
  ASSIGN_OR_RETURN(
      const core::TransformFactoryHolder* absl_nonnull holder,
      core::TransformRegistry::Instance().GetTransformFactoryHolder(
          typeid(*this)));
  RETURN_IF_ERROR(holder->ValidateInputs(context.inputs));
  return PropagateSignatureImpl(context);
}

// By default we will propagate unknown if a Transform does not provide
// a mechanism to propagate shapes.
absl::Status TransformBase::PropagateSignatureImpl(
    api::ShapeInferenceContext& context) const {
  absl::Span<const api::TypeInfo> output_types = OutputTypes();

  context.outputs.reserve(output_types.size());
  for (size_t i = 0; i < output_types.size(); ++i) {
    context.outputs.push_back(api::ShapeInferenceValue());
  }
  return absl::OkStatus();
}

}  // namespace composite
}  // namespace neptune::data
