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

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_CORE_TRANSFORM_REGISTRY_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_CORE_TRANSFORM_REGISTRY_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla

namespace xla_serving::data {
namespace composite {
namespace core {

enum class Device {
  kCpu = 0,
  kGpu = 1,
};

inline DeviceProto ToDeviceProto(Device device) {
  switch (device) {
    case Device::kCpu:
      return DeviceProto::DEVICE_CPU;
    case Device::kGpu:
      return DeviceProto::DEVICE_GPU;
  }
  return DeviceProto::DEVICE_UNSPECIFIED;
}

inline Device FromDeviceProto(DeviceProto proto) {
  switch (proto) {
    case DeviceProto::DEVICE_GPU:
      return Device::kGpu;
    case DeviceProto::DEVICE_CPU:
    case DeviceProto::DEVICE_UNSPECIFIED:
    default:
      return Device::kCpu;
  }
}

struct InputConstraint {
  std::vector<::xla::ifrt::DType::Kind> allowed_dtypes;
  std::optional<std::vector<std::optional<int64_t>>> expected_shape =
      std::nullopt;
  bool is_list = false;
};

template <typename... Args>
std::vector<api::TypeInfo> GetTypeInfos() {
  return {api::CreateTypeInfo<std::decay_t<Args>>()...};
}

template <typename F, typename Tuple>
auto Invoke(F&& f, absl::Span<const api::Value> args) {
  auto invoke = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::invoke(
        std::forward<F>(f),
        args[I].Get<std::decay_t<std::tuple_element_t<I, Tuple>>>()...);
  };
  return invoke(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

namespace internal {

template <typename T>
struct ExtractStatusOr {
  using type = T;
  static constexpr bool is_status = false;
  static constexpr bool is_status_or = false;
};

template <>
struct ExtractStatusOr<absl::Status> {
  using type = void;
  static constexpr bool is_status = true;
  static constexpr bool is_status_or = false;
};

template <typename T>
struct ExtractStatusOr<absl::StatusOr<T>> {
  using type = T;
  static constexpr bool is_status = false;
  static constexpr bool is_status_or = true;
};

}  // namespace internal

template <typename ReturnType, typename... Args>
struct FunctionTraitsBase {
  using args_tuple = std::tuple<std::decay_t<Args>...>;
  using return_type = ReturnType;
  static std::vector<api::TypeInfo> InputTypes() {
    return GetTypeInfos<Args...>();
  }
  static std::vector<api::TypeInfo> OutputTypes() {
    using actual_return_type =
        typename internal::ExtractStatusOr<ReturnType>::type;
    if constexpr (std::is_void_v<actual_return_type>) {
      return {};
    } else {
      return GetTypeInfos<actual_return_type>();
    }
  }
};

template <typename T>
struct FunctionTraits;

// For free functions
template <typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (*)(Args...)>
    : FunctionTraitsBase<ReturnType, Args...> {};

// Pointer to member function
template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...)>
    : FunctionTraitsBase<ReturnType, Args...> {
  using class_type = ClassType;
};

// Pointer to const member function
template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...) const>
    : FunctionTraitsBase<ReturnType, Args...> {
  using class_type = ClassType;
};

// Holds factory, deserializer, and executor metadata for a transform type.
// A TransformFactoryHolder instance must outlive all transforms created or
// deserialized by it. However, callers should prefer using TransformRegistry
// (which manages holder lifetimes automatically) instead of creating and
// managing TransformFactoryHolder instances manually.
class TransformFactoryHolder {
 public:
  using TransformFactory = absl::AnyInvocable<std::unique_ptr<TransformBase>(
      const std::vector<api::Value>&) const>;
  using TransformDeserializer =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<TransformBase>>(
          const TransformConfig&,
          absl::Span<const std::shared_ptr<xla_serving::data::Resource>>) const>;

  using ExecutorFunc = absl::AnyInvocable<absl::Status(
      TransformBase*, api::ExecuteContext&) const>;

  explicit TransformFactoryHolder(TransformFactory factory)
      : factory_(std::move(factory)) {}

  std::unique_ptr<TransformBase> CreateTransform(
      const std::vector<api::Value>& args) const {
    std::unique_ptr<TransformBase> transform = factory_(args);
    if (transform != nullptr) {
      transform->SetTransformFactoryHolder(this);
    }
    return transform;
  }

  void SetDeserializer(TransformDeserializer deserializer) {
    deserializer_ = std::move(deserializer);
  }

  bool HasDeserializer() const { return deserializer_ != nullptr; }

  absl::StatusOr<std::unique_ptr<TransformBase>> DeserializeTransform(
      const TransformConfig& config,
      absl::Span<const std::shared_ptr<xla_serving::data::Resource>> resources = {})
      const {
    if (deserializer_ == nullptr) {
      return absl::UnimplementedError(
          "Deserializer not registered for transform.");
    }
    ASSIGN_OR_RETURN(std::unique_ptr<TransformBase> transform,
                     deserializer_(config, resources));
    if (transform != nullptr) {
      transform->SetTransformFactoryHolder(this);
    }
    return transform;
  }

  void SetExecutorInfo(std::vector<api::TypeInfo> input_types,
                       std::vector<api::TypeInfo> output_types,
                       ExecutorFunc executor) {
    input_types_ = std::move(input_types);
    output_types_ = std::move(output_types);
    executor_ = std::move(executor);
  }

  bool HasExecutor() const { return executor_ != nullptr; }

  absl::Status Execute(TransformBase* base, api::ExecuteContext& ctx) const;

  absl::Span<const api::TypeInfo> InputTypes() const { return input_types_; }
  absl::Span<const api::TypeInfo> OutputTypes() const { return output_types_; }

  void SetInputConstraints(std::vector<InputConstraint> constraints) {
    input_constraints_ = std::move(constraints);
  }
  absl::Span<const InputConstraint> InputConstraints() const {
    return input_constraints_;
  }

  void SetDevice(Device device) { device_ = device; }
  Device device() const { return device_; }

  absl::Status ValidateInputs(absl::Span<const api::Value> inputs) const;
  absl::Status ValidateInputs(
      absl::Span<const api::ShapeInferenceValue> inputs) const;

 private:
  absl_nonnull TransformFactory factory_;
  absl_nullable TransformDeserializer deserializer_;
  absl_nullable ExecutorFunc executor_;
  std::vector<api::TypeInfo> input_types_;
  std::vector<api::TypeInfo> output_types_;
  std::vector<InputConstraint> input_constraints_;
  Device device_ = Device::kCpu;
};

class TransformRegistry {
 public:
  static TransformRegistry& Instance();

  // Get the creator from the library and invoke it to create a Transform.
  absl::StatusOr<std::unique_ptr<TransformBase>> CreateTransform(
      absl::string_view name, const std::vector<api::Value>& args = {},
      Device device = Device::kCpu) ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

  // Register a transform factory for a given name and type_index.
  // Returns an error if the (name, type_index) has already been registered.
  absl::Status RegisterTransformFactory(
      absl::string_view name, std::type_index type_index,
      TransformFactoryHolder::TransformFactory factory)
      ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

  // Deserialize a transform from a configuration proto.
  absl::StatusOr<std::unique_ptr<TransformBase>> DeserializeTransform(
      const TransformConfig& config,
      absl::Span<const std::shared_ptr<xla_serving::data::Resource>> resources = {})
      ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

  // Register a transform deserializer for a given name and type_index.
  absl::Status RegisterTransformDeserializer(
      absl::string_view name, std::type_index type_index,
      TransformFactoryHolder::TransformDeserializer deserializer)
      ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

  absl::Status RegisterTransformExecutor(
      absl::string_view name, std::type_index type_index,
      std::vector<api::TypeInfo> input_types,
      std::vector<api::TypeInfo> output_types,
      TransformFactoryHolder::ExecutorFunc executor)
      ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

  absl::Status SetInputConstraints(absl::string_view name,
                                   std::type_index type_index,
                                   std::vector<InputConstraint> constraints)
      ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

  absl::Status SetDevice(absl::string_view name, std::type_index type_index,
                         Device device)
      ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

  // Returns the TransformFactoryHolder for the given name.
  absl::StatusOr<const TransformFactoryHolder* absl_nonnull>
  GetTransformFactoryHolder(std::type_index type_index)
      ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

  absl::StatusOr<const TransformFactoryHolder* absl_nonnull>
  GetAnyTransformFactoryHolder(absl::string_view name,
                               Device device = Device::kCpu) const
      ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

  absl::StatusOr<const TransformFactoryHolder* absl_nonnull> FindMatchingHolder(
      absl::string_view name, absl::Span<const api::Value> inputs,
      Device device = Device::kCpu) const
      ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

  absl::StatusOr<const TransformFactoryHolder* absl_nonnull> FindMatchingHolder(
      absl::string_view name, absl::Span<const api::ShapeInferenceValue> inputs,
      Device device = Device::kCpu) const
      ABSL_LOCKS_EXCLUDED(transform_registry_mu_);

 private:
  mutable absl::Mutex transform_registry_mu_;
  absl::flat_hash_map<std::type_index,
                      absl_nonnull std::unique_ptr<TransformFactoryHolder>>
      type_to_holder_ ABSL_GUARDED_BY(transform_registry_mu_);
  absl::flat_hash_map<std::string, std::vector<TransformFactoryHolder*>>
      library_ ABSL_GUARDED_BY(transform_registry_mu_);
};

// All ops registering multiple kernels must:
// - Use HostBuffers as the input and output types.
// - Use the same set of resources.
class DynamicDispatchTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<DynamicDispatchTransform>> Create(
      std::string op_name, std::vector<api::Value> args = {},
      Device device = Device::kCpu) {
    ASSIGN_OR_RETURN(auto holder,
                     TransformRegistry::Instance().GetAnyTransformFactoryHolder(
                         op_name, device));
    std::vector<std::shared_ptr<xla_serving::data::Resource>> resources;
    if (!args.empty()) {
      auto kernel = holder->CreateTransform(args);
      if (kernel == nullptr) {
        return absl::InternalError(absl::StrFormat(
            "Failed to instantiate kernel for '%s' with the provided args.",
            op_name));
      }
      resources = kernel->GetAllResources();
    }
    return absl::WrapUnique(new DynamicDispatchTransform(
        std::move(op_name), std::move(args), std::move(resources), device));
  }

  static absl::StatusOr<std::unique_ptr<DynamicDispatchTransform>> Create(
      TransformConfig config,
      std::vector<std::shared_ptr<xla_serving::data::Resource>> resources = {},
      Device device = Device::kCpu) {
    return absl::WrapUnique(new DynamicDispatchTransform(
        std::move(config), std::move(resources), device));
  }

  absl::Status Execute(api::ExecuteContext& context) override;

  absl::string_view DebugString() const override {
    absl::MutexLock lock(active_kernel_mu_);
    if (active_kernel_ != nullptr) {
      return active_kernel_->DebugString();
    }
    return op_name_;
  }

  absl::StatusOr<TransformConfig> ToProto() const override;

  // Must always be HostBuffer for kernels using dynamic dispatch.
  absl::Span<const api::TypeInfo> InputTypes() const override;
  absl::Span<const api::TypeInfo> OutputTypes() const override;

  std::vector<std::shared_ptr<xla_serving::data::Resource>> GetAllResources()
      const override {
    return resources_;
  }

  Device device() const { return device_; }
  void SetDevice(Device device) { device_ = device; }

  absl::Status PropagateSignature(
      api::ShapeInferenceContext& context) const override;

 private:
  explicit DynamicDispatchTransform(
      std::string op_name, std::vector<api::Value> args,
      std::vector<std::shared_ptr<xla_serving::data::Resource>> resources,
      Device device)
      : op_name_(std::move(op_name)),
        args_(std::move(args)),
        resources_(std::move(resources)),
        device_(device) {}

  explicit DynamicDispatchTransform(
      TransformConfig config,
      std::vector<std::shared_ptr<xla_serving::data::Resource>> resources,
      Device device)
      : op_name_(config.op_name()),
        config_(std::move(config)),
        resources_(std::move(resources)),
        device_(device) {}

  template <typename InputType>
  absl::StatusOr<TransformBase* absl_nonnull> GetOrCreateKernel(
      absl::Span<InputType> inputs) const {
    absl::MutexLock lock(active_kernel_mu_);
    if (active_kernel_ != nullptr) return active_kernel_.get();

    // Create the kernel based on the target input types.
    ASSIGN_OR_RETURN(auto holder,
                     TransformRegistry::Instance().FindMatchingHolder(
                         op_name_, inputs, device_));
    if (config_.has_value()) {
      // Graph loading case - loading a fully traced function.
      ASSIGN_OR_RETURN(auto kernel,
                       holder->DeserializeTransform(*config_, resources_));
      if (kernel != nullptr) kernel->SetDefaultDebugString(op_name_);
      active_kernel_ = std::move(kernel);
    } else {
      // This occurs in one of the two following scenarios:
      // Graph tracing case - creating the graph for the first time.
      // Eager execution case - single op is instantiated and executed.
      auto kernel = holder->CreateTransform(args_);
      if (kernel != nullptr) kernel->SetDefaultDebugString(op_name_);
      active_kernel_ = std::move(kernel);
    }

    // This should never happen and is indicative of a bigger problem if we
    // see this error msg.
    if (active_kernel_ == nullptr) {
      return absl::InternalError(
          absl::StrCat("Failed to instantiate kernel for ", op_name_));
    }
    return active_kernel_.get();
  }

  std::string op_name_;
  std::vector<api::Value> args_;
  std::optional<TransformConfig> config_;
  std::vector<std::shared_ptr<xla_serving::data::Resource>> resources_;
  Device device_ = Device::kCpu;
  mutable absl::Mutex active_kernel_mu_;
  mutable std::unique_ptr<TransformBase> active_kernel_
      ABSL_GUARDED_BY(active_kernel_mu_);
};

template <typename T>
class TransformRegistrar {
 public:
  explicit TransformRegistrar(std::string name)
      : name_(std::move(name)), type_index_(typeid(T)) {
    SetFactory(&T::Create);
    // TODO: Remove the requires and make it a compile time error
    // to not specify the `operator()`.
    if constexpr (requires { &T::operator(); }) {
      SetExecutor(&T::operator());
    }
  }

  // Register a deserializer for the transform. This is used to create the
  // transform from the TransformConfig proto. If not provided a serialized
  // transform cannot be created from the Registry. Users must provide this for
  // the transform to be compatible with serving.
  // TODO: Consider providing a default deserializer for simple ops
  // that do not rely on details.
  TransformRegistrar& SetDeserializer(
      TransformFactoryHolder::TransformDeserializer deserializer) {
    CHECK_OK(TransformRegistry::Instance().RegisterTransformDeserializer(
        name_, type_index_, std::move(deserializer)));
    return *this;
  }

  TransformRegistrar& SetInputConstraints(
      std::vector<InputConstraint> constraints) {
    CHECK_OK(TransformRegistry::Instance().SetInputConstraints(
        name_, type_index_, std::move(constraints)));
    return *this;
  }

  TransformRegistrar& SetDevice(Device device) {
    CHECK_OK(
        TransformRegistry::Instance().SetDevice(name_, type_index_, device));
    return *this;
  }

 private:
  template <typename Func>
  TransformRegistrar& SetExecutor(Func func) {
    using Traits = FunctionTraits<Func>;

    auto executor_runner = [func](TransformBase* base,
                                  api::ExecuteContext& ctx) -> absl::Status {
      auto bound_f = [&]() {
        if constexpr (std::is_member_function_pointer_v<Func>) {
          return std::bind_front(
              func, static_cast<typename Traits::class_type*>(base));
        } else {
          return func;
        }
      }();

      if (ctx.inputs.size() != std::tuple_size_v<typename Traits::args_tuple>) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected ", std::tuple_size_v<typename Traits::args_tuple>,
            " input(s) when calling Execute, but got ", ctx.inputs.size(),
            " instead."));
      }

      using ReturnT = typename Traits::return_type;
      using Extracted = internal::ExtractStatusOr<ReturnT>;

      if constexpr (std::is_void_v<ReturnT>) {
        Invoke<decltype(bound_f), typename Traits::args_tuple>(
            std::move(bound_f), ctx.inputs);
        return absl::OkStatus();
      } else if constexpr (Extracted::is_status) {
        return Invoke<decltype(bound_f), typename Traits::args_tuple>(
            std::move(bound_f), ctx.inputs);
      } else if constexpr (Extracted::is_status_or) {
        auto result = Invoke<decltype(bound_f), typename Traits::args_tuple>(
            std::move(bound_f), ctx.inputs);
        if (!result.ok()) return result.status();
        api::Value v;
        v.Set(std::move(*result));
        ctx.outputs.push_back(std::move(v));
        return absl::OkStatus();
      } else {
        auto result = Invoke<decltype(bound_f), typename Traits::args_tuple>(
            std::move(bound_f), ctx.inputs);
        api::Value v;
        v.Set(std::move(result));
        ctx.outputs.push_back(std::move(v));
        return absl::OkStatus();
      }
    };

    CHECK_OK(TransformRegistry::Instance().RegisterTransformExecutor(
        name_, type_index_, Traits::InputTypes(), Traits::OutputTypes(),
        std::move(executor_runner)));
    return *this;
  }

  // Takes a function ptr and automatically deduces input types and decomposes
  // it into std::vector<api::Value>. The only caller should be the
  // TransformRegistrar ctor through a call to the registering class's Create
  // method -> `SetFactory(&T::Create)`.
  template <typename ReturnType, typename... Args>
  TransformRegistrar& SetFactory(ReturnType (*factory)(Args...)) {
    static_assert(
        requires(ReturnType r) {
          r.status();
          *std::move(r);
        },
        "Transform class must implement a static 'Create' method. "
        "E.g., static absl::StatusOr<std::unique_ptr<TransformBase>> "
        "Create(...)");
    auto wrapper = [factory](absl::Span<const api::Value> args)
        -> std::unique_ptr<TransformBase> {
      CHECK_EQ(args.size(), sizeof...(Args))
          << "Expected " << sizeof...(Args) << " arguments, but got "
          << args.size();
      auto invoke = [&, args]<std::size_t... I>(std::index_sequence<I...>) {
        auto result = factory(std::move(
            const_cast<api::Value&>(args[I]).Get<std::decay_t<Args>>())...);
        CHECK_OK(result);
        return *std::move(result);
      };
      return invoke(std::index_sequence_for<Args...>{});
    };
    CHECK_OK(TransformRegistry::Instance().RegisterTransformFactory(
        name_, type_index_, std::move(wrapper)));
    return *this;
  }

  std::string name_;
  std::type_index type_index_;
};

}  // namespace core
}  // namespace composite
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_CORE_TRANSFORM_REGISTRY_H_
