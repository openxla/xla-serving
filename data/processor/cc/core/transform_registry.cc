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

#include "data/processor/cc/core/transform_registry.h"  // from @neptune

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/cc/api/value.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace neptune::data {
namespace composite {
namespace core {

TransformRegistry& TransformRegistry::Instance() {
  static absl::NoDestructor<TransformRegistry> instance;
  return *instance;
}

absl::Status TransformRegistry::RegisterTransformFactory(
    absl::string_view name, TransformFactoryHolder::TransformFactory factory) {
  absl::MutexLock lock(transform_registry_mu_);
  std::unique_ptr<TransformFactoryHolder>& existing_creator = library_[name];
  if (existing_creator != nullptr) {
    return absl::AlreadyExistsError(
        absl::StrFormat("A Transform factory is already registered for key: "
                        " %s",
                        name));
  }
  existing_creator =
      std::make_unique<TransformFactoryHolder>(std::move(factory));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<TransformBase>>
TransformRegistry::CreateTransform(absl::string_view name,
                                   const std::vector<api::Value>& args) {
  TransformFactoryHolder* holder = nullptr;
  {
    absl::MutexLock lock(transform_registry_mu_);
    auto it = library_.find(name);
    if (it == library_.end()) {
      return absl::NotFoundError(absl::StrFormat(
          "Transform '%s' not found. Please ensure the relevant transform "
          "library has been linked, and the transform factory "
          "has been registered by calling RegisterGlobalTransform or "
          "RegisterGlobalTransformFactory.",
          name));
    }
    holder = it->second.get();
  }
  auto transform = holder->CreateTransform(args);
  if (transform != nullptr) {
    transform->SetDefaultDebugString(name);
  }
  return transform;
}

absl::StatusOr<std::unique_ptr<TransformBase>>
TransformRegistry::DeserializeTransform(
    const TransformConfig& config,
    absl::Span<const std::shared_ptr<neptune::data::Resource>> resources) {
  TransformFactoryHolder* holder = nullptr;
  {
    absl::MutexLock lock(transform_registry_mu_);
    auto it = library_.find(config.op_name());
    if (it == library_.end()) {
      return absl::NotFoundError(absl::StrFormat(
          "Transform deserializer for '%s' not found. Please ensure the "
          "relevant transform library has been linked, and the transform "
          "deserializer has been registered by calling "
          "RegisterTransformDeserializer.",
          config.op_name()));
    }
    holder = it->second.get();
  }
  ASSIGN_OR_RETURN(auto transform,
                   holder->DeserializeTransform(config, resources));
  if (transform != nullptr) {
    transform->SetDefaultDebugString(config.op_name());
  }
  return transform;
}

absl::Status TransformRegistry::RegisterTransformDeserializer(
    absl::string_view name,
    TransformFactoryHolder::TransformDeserializer deserializer) {
  absl::MutexLock lock(transform_registry_mu_);
  auto it = library_.find(name);
  if (it == library_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Transform factory for '%s' not found. Please ensure the relevant "
        "transform library has been linked, and the transform factory "
        "has been registered by calling RegisterGlobalTransform or "
        "RegisterGlobalTransformFactory.",
        name));
  }
  if (it->second->HasDeserializer()) {
    return absl::AlreadyExistsError(absl::StrFormat(
        "A Transform deserializer is already registered for key: "
        " %s",
        name));
  }
  it->second->SetDeserializer(std::move(deserializer));
  return absl::OkStatus();
}

absl::Status TransformRegistry::RegisterTransformExecutor(
    absl::string_view name, std::vector<api::TypeInfo> input_types,
    std::vector<api::TypeInfo> output_types,
    TransformFactoryHolder::ExecutorFunc executor) {
  absl::MutexLock lock(transform_registry_mu_);
  auto it = library_.find(name);
  if (it == library_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Transform factory for '%s' not found. Please ensure the relevant "
        "transform library has been linked, and the transform factory "
        "has been registered by calling ND_REGISTER_TRANSFORM.",
        name));
  }
  if (it->second->HasExecutor()) {
    return absl::AlreadyExistsError(absl::StrFormat(
        "A Transform executor is already registered for key: %s", name));
  }
  it->second->SetExecutorInfo(std::move(input_types), std::move(output_types),
                              std::move(executor));
  return absl::OkStatus();
}

void TransformRegistry::RegisterTransformType(absl::string_view name,
                                              std::type_index type_index) {
  absl::MutexLock lock(transform_registry_mu_);
  type_to_name_[type_index] = std::string(name);
}

absl::StatusOr<const TransformFactoryHolder* absl_nonnull>
TransformRegistry::GetTransformFactoryHolder(std::type_index type_index) {
  absl::MutexLock lock(transform_registry_mu_);
  auto name_it = type_to_name_.find(type_index);
  if (name_it == type_to_name_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Transform '%s' not found. Please ensure the relevant "
                        "transform library has been linked, and the transform "
                        "has been registered by calling ND_REGISTER_TRANSFORM.",
                        type_index.name()));
  }
  auto it = library_.find(name_it->second);
  if (it == library_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Transform '%s' not found. Please ensure the relevant "
                        "transform library has been linked, and the transform "
                        "has been registered by calling ND_REGISTER_TRANSFORM.",
                        name_it->second));
  }
  return it->second.get();
}

absl::Status TransformFactoryHolder::Execute(TransformBase* base,
                                             api::ExecuteContext& ctx) const {
  DCHECK(HasExecutor());
  RETURN_IF_ERROR(ValidateInputs(ctx.inputs));
  return executor_(base, ctx);
}

namespace {

xla::ifrt::DType GetDType(const neptune::HostBuffer& hb) { return hb.dtype(); }
xla::ifrt::DType GetDType(const api::HostBufferSpec& spec) {
  return spec.dtype;
}

const xla::ifrt::Shape& GetShape(const neptune::HostBuffer& hb) {
  return hb.shape();
}
const xla::ifrt::Shape& GetShape(const api::HostBufferSpec& spec) {
  return spec.shape;
}

template <typename Item>
absl::Status ValidateSingleItem(const Item& item, size_t input_idx,
                                const InputConstraint& constraint) {
  xla::ifrt::DType dtype = GetDType(item);
  const xla::ifrt::Shape& shape = GetShape(item);

  if (!constraint.allowed_dtypes.empty()) {
    bool allowed = false;
    for (auto allowed_dtype : constraint.allowed_dtypes) {
      if (dtype.kind() == allowed_dtype) {
        allowed = true;
        break;
      }
    }
    if (!allowed) {
      std::ostringstream oss;
      oss << dtype;
      return absl::InvalidArgumentError(absl::StrCat(
          "Input ", input_idx, " has unsupported dtype: ", oss.str()));
    }
  }
  if (constraint.expected_shape.has_value()) {
    const auto& expected_shape = *constraint.expected_shape;
    if (shape.dims().size() != expected_shape.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input ", input_idx, " has invalid shape. Rank mismatch: ",
          shape.dims().size(), " vs ", expected_shape.size()));
    }
    for (size_t d = 0; d < expected_shape.size(); ++d) {
      if (expected_shape[d].has_value() &&
          shape.dims()[d] != *expected_shape[d]) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Input ", input_idx, " has invalid shape at dim ", d, ": ",
            shape.dims()[d], ". Expected: ", *expected_shape[d]));
      }
    }
  }
  return absl::OkStatus();
}

template <typename V>
struct ValueTraits;

template <>
struct ValueTraits<api::Value> {
  static bool IsList(const api::Value& value) {
    return value.IsType<std::vector<neptune::HostBuffer>>();
  }
  static absl::Span<const neptune::HostBuffer> GetList(const api::Value& value) {
    return value.Get<std::vector<neptune::HostBuffer>>();
  }
  static bool IsSingle(const api::Value& value) {
    return value.IsType<neptune::HostBuffer>();
  }
  static const neptune::HostBuffer& GetSingle(const api::Value& value) {
    return value.Get<neptune::HostBuffer>();
  }
};

template <>
struct ValueTraits<api::ShapeInferenceValue> {
  static bool IsList(const api::ShapeInferenceValue& value) {
    return value.IsSpecList();
  }
  static absl::Span<const api::HostBufferSpec> GetList(
      const api::ShapeInferenceValue& value) {
    return value.spec_list();
  }
  static bool IsSingle(const api::ShapeInferenceValue& value) {
    return value.IsSpec();
  }
  static const api::HostBufferSpec& GetSingle(
      const api::ShapeInferenceValue& value) {
    return value.spec();
  }
};

template <typename V>
absl::Status ValidateInputSpan(absl::Span<const V> inputs,
                               absl::Span<const InputConstraint> constraints) {
  using Traits = ValueTraits<V>;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i >= constraints.size()) break;
    const auto& constraint = constraints[i];
    const auto& value = inputs[i];

    if (constraint.is_list) {
      if (Traits::IsList(value)) {
        auto vec = Traits::GetList(value);
        for (size_t j = 0; j < vec.size(); ++j) {
          auto status = ValidateSingleItem(vec[j], i, constraint);
          if (!status.ok()) {
            return absl::Status(
                status.code(),
                absl::StrCat("In list argument ", i, " at index ", j, ": ",
                             status.message()));
          }
        }
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Input ", i, " expected to be a list of HostBuffers"));
      }
    } else {
      if (Traits::IsSingle(value)) {
        RETURN_IF_ERROR(
            ValidateSingleItem(Traits::GetSingle(value), i, constraint));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status TransformFactoryHolder::ValidateInputs(
    absl::Span<const api::Value> inputs) const {
  return ValidateInputSpan(inputs, input_constraints_);
}

absl::Status TransformFactoryHolder::ValidateInputs(
    absl::Span<const api::ShapeInferenceValue> inputs) const {
  return ValidateInputSpan(inputs, input_constraints_);
}

absl::Status TransformRegistry::SetInputConstraints(
    absl::string_view name, std::vector<InputConstraint> constraints) {
  absl::MutexLock lock(transform_registry_mu_);
  auto it = library_.find(name);
  if (it == library_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Transform factory for '%s' not found.", name));
  }
  it->second->SetInputConstraints(std::move(constraints));
  return absl::OkStatus();
}

}  // namespace core
}  // namespace composite
}  // namespace neptune::data
