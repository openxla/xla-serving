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

#include "data/processor/cc/core/transform.h"  // from @xla_serving

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/shape.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/core/transform_registry.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla

namespace xla_serving::data {
namespace composite {
namespace {

// Returns InvalidArgument unless `value` is nullopt, or is a valid trace-time
// value for `spec_value`: see api::PartialValue. `kind` is "input" or "output".
absl::Status CheckPartialValue(absl::string_view transform_name,
                               absl::string_view kind, size_t index,
                               const api::ShapeInferenceValue& spec_value,
                               const std::optional<api::PartialValue>& value) {
  if (!value.has_value()) return absl::OkStatus();
  auto error = [&](absl::string_view problem) {
    return absl::InvalidArgumentError(
        absl::StrCat(transform_name, ": ", kind, " ", index, " has value ",
                     value->DebugString(), ", but ", problem));
  };
  if (!spec_value.IsSpec()) {
    return error("it is not a single HostBuffer spec");
  }
  const api::HostBufferSpec& spec = spec_value.spec();
  const bool is_s32 = spec.dtype.kind() == xla::ifrt::DType::kS32;
  if (!is_s32 && spec.dtype.kind() != xla::ifrt::DType::kS64) {
    return error(
        absl::StrCat("its dtype is ", spec.dtype, ", not int32 or int64"));
  }
  if (!spec.shape.IsFullyKnown()) {
    return error(absl::StrCat("its shape ", spec.shape.DebugString(),
                              " is not fully known"));
  }
  const int64_t num_elements = *spec.shape.num_elements().Value();
  if (value->size() != num_elements) {
    return error(absl::StrCat("its shape ", spec.shape.DebugString(), " has ",
                              num_elements, " elements"));
  }
  if (num_elements > api::PartialValue::kMaxElements) {
    return error(absl::StrCat("it has more than ",
                              api::PartialValue::kMaxElements, " elements"));
  }
  if (is_s32) {
    for (int64_t i = 0; i < value->size(); ++i) {
      std::optional<int64_t> element = value->element(i);
      if (element.has_value() &&
          (*element < std::numeric_limits<int32_t>::min() ||
           *element > std::numeric_limits<int32_t>::max())) {
        return error(
            absl::StrCat("element ", i, " does not fit its dtype int32"));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<TransformBase>> TransformBase::FromProto(
    const TransformConfig& config,
    absl::Span<const std::shared_ptr<xla_serving::data::Resource>> resources) {
  return core::TransformRegistry::Instance().DeserializeTransform(config,
                                                                  resources);
}

absl::StatusOr<const core::TransformFactoryHolder* absl_nonnull>
TransformBase::GetTransformFactoryHolder() const {
  const core::TransformFactoryHolder* holder =
      holder_.load(std::memory_order_acquire);
  if (holder != nullptr) {
    return holder;
  }
  // TransformRegistry::GetTransformFactoryHolder() returns the same stable
  // pointer for a given transform type on every call, so caching it in an
  // atomic member avoids repeated registry mutex acquisitions on subsequent
  // calls.
  ASSIGN_OR_RETURN(
      holder, core::TransformRegistry::Instance().GetTransformFactoryHolder(
                  typeid(*this)));
  holder_.store(holder, std::memory_order_release);
  return holder;
}

absl::Status TransformBase::Execute(api::ExecuteContext& context) {
  ASSIGN_OR_RETURN(const core::TransformFactoryHolder* absl_nonnull holder,
                   GetTransformFactoryHolder());
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
      GetTransformFactoryHolder();
  if (!holder.ok()) {
    return {};
  }
  return (*holder)->InputTypes();
}

absl::Span<const api::TypeInfo> TransformBase::OutputTypes() const {
  absl::StatusOr<const core::TransformFactoryHolder* absl_nonnull> holder =
      GetTransformFactoryHolder();
  if (!holder.ok()) {
    return {};
  }
  return (*holder)->OutputTypes();
}

absl::Status TransformBase::PropagateSignature(
    api::ShapeInferenceContext& context) const {
  ASSIGN_OR_RETURN(const core::TransformFactoryHolder* absl_nonnull holder,
                   GetTransformFactoryHolder());
  RETURN_IF_ERROR(holder->ValidateInputs(context.inputs));
  if (!context.input_partial_values.empty() &&
      context.input_partial_values.size() != context.inputs.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        holder->name(), ": input_partial_values has ",
        context.input_partial_values.size(), " entries, but there are ",
        context.inputs.size(), " inputs"));
  }
  for (size_t i = 0; i < context.input_partial_values.size(); ++i) {
    RETURN_IF_ERROR(CheckPartialValue(holder->name(), "input", i,
                                      context.inputs[i],
                                      context.input_partial_values[i]));
  }
  // Shape rules share helpers, such as PartialShape::GetStatic, whose errors
  // don't say which transform failed.
  if (absl::Status status = PropagateSignatureImpl(context); !status.ok()) {
    return absl::Status(status.code(),
                        absl::StrCat(holder->name(), ": ", status.message()));
  }
  if (!context.output_partial_values.empty() &&
      context.output_partial_values.size() != context.outputs.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        holder->name(), ": output_partial_values has ",
        context.output_partial_values.size(), " entries, but there are ",
        context.outputs.size(), " outputs"));
  }
  for (size_t i = 0; i < context.output_partial_values.size(); ++i) {
    RETURN_IF_ERROR(CheckPartialValue(holder->name(), "output", i,
                                      context.outputs[i],
                                      context.output_partial_values[i]));
  }
  return absl::OkStatus();
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
}  // namespace xla_serving::data
