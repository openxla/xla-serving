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
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
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
    absl::string_view name, std::type_index type_index,
    TransformFactoryHolder::TransformFactory factory) {
  absl::MutexLock lock(transform_registry_mu_);
  std::unique_ptr<TransformFactoryHolder>& existing_creator =
      type_to_holder_[type_index];
  if (existing_creator != nullptr) {
    return absl::AlreadyExistsError(
        absl::StrFormat("A Transform factory is already registered for key: "
                        " %s",
                        name));
  }
  existing_creator =
      std::make_unique<TransformFactoryHolder>(std::move(factory));
  library_[std::string(name)].push_back(existing_creator.get());
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<TransformBase>>
TransformRegistry::CreateTransform(absl::string_view name,
                                   const std::vector<api::Value>& args,
                                   Device device) {
  TransformFactoryHolder* holder = nullptr;
  {
    absl::MutexLock lock(transform_registry_mu_);
    auto it = library_.find(name);
    if (it == library_.end() || it->second.empty()) {
      return absl::NotFoundError(absl::StrFormat(
          "Transform '%s' not found. Please ensure the relevant transform "
          "library has been linked, and the transform factory "
          "has been registered by calling RegisterGlobalTransform or "
          "RegisterGlobalTransformFactory.",
          name));
    }
    // Multiple holders: wrap in DynamicDispatchTransform
    if (it->second.size() > 1) {
      return DynamicDispatchTransform::Create(std::string(name), args, device);
    }
    holder = it->second[0];
  }

  if (holder->device() != device) {
    return absl::NotFoundError(
        absl::StrFormat("No kernel registered for '%s' on device %s.", name,
                        device == Device::kGpu ? "GPU" : "CPU"));
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
    if (it == library_.end() || it->second.empty()) {
      return absl::NotFoundError(absl::StrFormat(
          "Transform deserializer for '%s' not found. Please ensure the "
          "relevant transform library has been linked, and the transform "
          "deserializer has been registered by calling "
          "RegisterTransformDeserializer.",
          config.op_name()));
    }
    // Multiple holders: wrap in DynamicDispatchTransform
    if (it->second.size() > 1) {
      std::vector<std::shared_ptr<neptune::data::Resource>> res_vec(
          resources.begin(), resources.end());
      Device device = FromDeviceProto(config.device());
      return DynamicDispatchTransform::Create(config, std::move(res_vec),
                                              device);
    }
    holder = it->second[0];
  }

  // Exactly 1 holder: deserialize directly.
  ASSIGN_OR_RETURN(auto transform,
                   holder->DeserializeTransform(config, resources));
  if (transform != nullptr) {
    transform->SetDefaultDebugString(config.op_name());
  }
  return transform;
}

absl::Status TransformRegistry::RegisterTransformDeserializer(
    absl::string_view name, std::type_index type_index,
    TransformFactoryHolder::TransformDeserializer deserializer) {
  absl::MutexLock lock(transform_registry_mu_);
  auto it = type_to_holder_.find(type_index);
  if (it == type_to_holder_.end()) {
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
    absl::string_view name, std::type_index type_index,
    std::vector<api::TypeInfo> input_types,
    std::vector<api::TypeInfo> output_types,
    TransformFactoryHolder::ExecutorFunc executor) {
  absl::MutexLock lock(transform_registry_mu_);
  auto it = type_to_holder_.find(type_index);
  if (it == type_to_holder_.end()) {
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

absl::StatusOr<const TransformFactoryHolder* absl_nonnull>
TransformRegistry::GetTransformFactoryHolder(std::type_index type_index) {
  absl::MutexLock lock(transform_registry_mu_);
  auto it = type_to_holder_.find(type_index);
  if (it != type_to_holder_.end()) {
    return it->second.get();
  }
  return absl::NotFoundError(
      absl::StrFormat("Transform '%s' not found. Please ensure the relevant "
                      "transform library has been linked, and the transform "
                      "has been registered by calling ND_REGISTER_TRANSFORM.",
                      type_index.name()));
}

absl::StatusOr<const TransformFactoryHolder* absl_nonnull>
TransformRegistry::GetAnyTransformFactoryHolder(absl::string_view name,
                                                Device device) const {
  absl::MutexLock lock(transform_registry_mu_);
  auto it = library_.find(name);
  if (it == library_.end() || it->second.empty()) {
    return absl::NotFoundError(
        absl::StrFormat("Transform '%s' not found.", name));
  }
  for (TransformFactoryHolder* holder : it->second) {
    if (holder->device() == device) {
      return holder;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("No kernel registered for '%s' on device %s.", name,
                      device == Device::kGpu ? "GPU" : "CPU"));
  return it->second[0];
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
    absl::string_view name, std::type_index type_index,
    std::vector<InputConstraint> constraints) {
  absl::MutexLock lock(transform_registry_mu_);
  auto it = type_to_holder_.find(type_index);
  if (it == type_to_holder_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Transform factory for '%s' not found.", name));
  }

  it->second->SetInputConstraints(std::move(constraints));
  return absl::OkStatus();
}

absl::Status TransformRegistry::SetDevice(absl::string_view name,
                                          std::type_index type_index,
                                          Device device) {
  absl::MutexLock lock(transform_registry_mu_);
  auto it = type_to_holder_.find(type_index);
  if (it == type_to_holder_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Transform factory for type '%s' under '%s' not found.",
                        type_index.name(), name));
  }
  it->second->SetDevice(device);
  return absl::OkStatus();
}

template <typename V>
absl::StatusOr<const TransformFactoryHolder* absl_nonnull>
FindMatchingHolderImpl(
    const absl::flat_hash_map<std::string,
                              std::vector<TransformFactoryHolder*>>& library,
    absl::string_view name, absl::Span<const V> inputs, Device device) {
  auto it = library.find(name);
  if (it == library.end() || it->second.empty()) {
    return absl::NotFoundError(
        absl::StrFormat("Transform '%s' not found.", name));
  }
  std::vector<TransformFactoryHolder*> matching_device_holders;
  for (TransformFactoryHolder* holder : it->second) {
    if (holder->device() == device) {
      matching_device_holders.push_back(holder);
    }
  }
  if (matching_device_holders.empty()) {
    return absl::NotFoundError(
        absl::StrFormat("No kernel registered for '%s' on device %s.", name,
                        device == Device::kGpu ? "GPU" : "CPU"));
  }
  if (matching_device_holders.size() == 1) {
    return matching_device_holders[0];
  }
  for (TransformFactoryHolder* holder : matching_device_holders) {
    if (holder->ValidateInputs(inputs).ok()) {
      return holder;
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "No matching kernel found for '%s' on device %s with the provided "
      "inputs.",
      name, device == Device::kGpu ? "GPU" : "CPU"));
}

absl::StatusOr<const TransformFactoryHolder* absl_nonnull>
TransformRegistry::FindMatchingHolder(absl::string_view name,
                                      absl::Span<const api::Value> inputs,
                                      Device device) const {
  absl::MutexLock lock(
      const_cast<TransformRegistry*>(this)->transform_registry_mu_);
  return FindMatchingHolderImpl(library_, name, inputs, device);
}

absl::StatusOr<const TransformFactoryHolder* absl_nonnull>
TransformRegistry::FindMatchingHolder(
    absl::string_view name, absl::Span<const api::ShapeInferenceValue> inputs,
    Device device) const {
  absl::MutexLock lock(
      const_cast<TransformRegistry*>(this)->transform_registry_mu_);
  return FindMatchingHolderImpl(library_, name, inputs, device);
}

// The kernel is resolved in the first call to PropagateSignature or Execute.
// Subsequent calls have to be done with the same type. Users will receive a
// msg indicating input validation failed if this transform is re-used with
// different input types.
absl::Status DynamicDispatchTransform::Execute(api::ExecuteContext& context) {
  ASSIGN_OR_RETURN(auto kernel, GetOrCreateKernel(context.inputs));
  return kernel->Execute(context);
}

absl::Status DynamicDispatchTransform::PropagateSignature(
    api::ShapeInferenceContext& context) const {
  ASSIGN_OR_RETURN(auto kernel, GetOrCreateKernel(context.inputs));
  return kernel->PropagateSignature(context);
}

absl::StatusOr<TransformConfig> DynamicDispatchTransform::ToProto() const {
  TransformConfig config;
  if (config_.has_value()) {
    config = *config_;
  } else {
    absl::MutexLock lock(active_kernel_mu_);
    if (active_kernel_ != nullptr) {
      ASSIGN_OR_RETURN(config, active_kernel_->ToProto());
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot serialize transform: ", op_name_,
          " before resolving it to a specific kernel. Please ensure "
          "the op has been traced or executed."));
    }
  }
  config.set_op_name(op_name_);
  config.set_device(ToDeviceProto(device_));
  return config;
}

absl::Span<const api::TypeInfo> DynamicDispatchTransform::InputTypes() const {
  {
    absl::MutexLock lock(active_kernel_mu_);
    if (active_kernel_ != nullptr) {
      return active_kernel_->InputTypes();
    }
  }
  auto holder = TransformRegistry::Instance().GetAnyTransformFactoryHolder(
      op_name_, device_);
  if (holder.ok()) return (*holder)->InputTypes();
  return {};
}

absl::Span<const api::TypeInfo> DynamicDispatchTransform::OutputTypes() const {
  {
    absl::MutexLock lock(active_kernel_mu_);
    if (active_kernel_ != nullptr) {
      return active_kernel_->OutputTypes();
    }
  }
  auto holder = TransformRegistry::Instance().GetAnyTransformFactoryHolder(
      op_name_, device_);
  if (holder.ok()) return (*holder)->OutputTypes();
  return {};
}

}  // namespace core
}  // namespace composite
}  // namespace neptune::data
