#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_TRANSFORM_PYBIND_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_TRANSFORM_PYBIND_H_

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "include/pybind11/pybind11.h"  // from @pybind11_bazel
#include "include/pybind11/stl.h"  // from @pybind11_bazel
#include "include/pybind11/trampoline_self_life_support.h"  // from @pybind11_bazel
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf

namespace neptune::data {
namespace composite {
namespace pybind_internal {

template <typename T>
struct IsStatusOr : std::false_type {};

template <typename T>
struct IsStatusOr<absl::StatusOr<T>> : std::true_type {};

template <typename T, typename = void>
struct HasOperator : std::false_type {};

template <typename T>
struct HasOperator<T, std::void_t<decltype(&T::operator())>> : std::true_type {
};

// This is very similar to `SetFactoryInternal` from TransformRegistry.
// However, this version has dependencies on pybind11 and is only used for
// Transforms where the user has explicitly added a `transform_pybind_library`
// rule to request auto-generated PyBindings. To avoid unnecessarily pulling in
// pybind11 dependencies in to TransformRegistry we reimplement
// SetFactoryInternal here.
template <typename T, typename ReturnType, typename... Args>
void SetFactoryInternal(pybind11::module& m, const std::string& name,
                        ReturnType (*factory)(Args...)) {
  static_assert(IsStatusOr<ReturnType>::value,
                "Create method must return an absl::StatusOr");
  auto cls = pybind11::classh<T, neptune::data::composite::TransformBase>(
      m, name.c_str());
  cls.def(pybind11::init([factory](Args... args) {
            auto result = factory(std::forward<Args>(args)...);
            if (!result.ok()) {
              throw std::runtime_error(std::string(result.status().message()));
            }
            return dynamic_cast<T*>(result.value().release());
          }),
          ((void)sizeof(Args), pybind11::arg().none(true))...);
  cls.def_property_readonly("name", [name](const T&) { return name; });
  cls.def(
      "execute",
      [](T& self, api::ExecuteContext& context) {
        absl::Status status = self.Execute(context);
        if (!status.ok()) {
          throw std::runtime_error(std::string(status.message()));
        }
      },
      "Internal api: Execute the transform with an explicitly provided "
      "api::ExecuteContext.");
  if constexpr (HasOperator<T>::value) {
    cls.def("execute", &T::operator(), "Execute the transform directly.");
  }
}

template <typename T>
void SetFactory(pybind11::module& m, const std::string& name) {
  SetFactoryInternal<T>(m, name, &T::Create);
}

}  // namespace pybind_internal

// Trampoline class for TransformBase to allow overriding virtual methods in
// Python. This is only used to provide a base class for python-defined
// transforms for testing. User defined transforms will have an
// auto-generated pybind.
// We manually implement the trampoline instead of using
// PYBIND11_OVERRIDE_PURE because:
//  - Execute: the macro silently copies reference arguments (like
//             ExecuteContext&), which prevents Python modifications from
//             propagating back to C++. We force reference passing by manually
//             casting with return_value_policy::reference.
//  - InputTypes/OutputTypes/DebugString: The macro returns non-owning
//                                        dangling references because Python
//                                        will garbage collect the underlying
//                                        objects within the same function call.
class PyTransform : public TransformBase,
                    public pybind11::trampoline_self_life_support {
 public:
  using TransformBase::TransformBase;

  absl::Status Execute(api::ExecuteContext& context) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload = pybind11::get_overload(
        static_cast<const TransformBase*>(this), "Execute");
    if (overload) {
      auto py_context =
          pybind11::cast(context, pybind11::return_value_policy::reference);
      auto o = overload(py_context);
      return o.cast<absl::Status>();
    }
    pybind11::pybind11_fail(
        "Tried to call pure virtual function \"TransformBase::Execute\"");
  }

  absl::Span<const api::TypeInfo> InputTypes() const override {
    if (input_types_initialized_.load(std::memory_order_acquire)) {
      return input_types_cache_;
    }
    pybind11::gil_scoped_acquire gil;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!input_types_initialized_.load(std::memory_order_relaxed)) {
      pybind11::function overload = pybind11::get_overload(
          static_cast<const TransformBase*>(this), "InputTypes");
      if (overload) {
        auto o = overload();
        input_types_cache_ = o.cast<std::vector<api::TypeInfo>>();
        input_types_initialized_.store(true, std::memory_order_release);
      } else {
        pybind11::pybind11_fail(
            "Tried to call pure virtual function "
            "\"TransformBase::InputTypes\"");
      }
    }
    return input_types_cache_;
  }

  absl::Span<const api::TypeInfo> OutputTypes() const override {
    if (output_types_initialized_.load(std::memory_order_acquire)) {
      return output_types_cache_;
    }
    pybind11::gil_scoped_acquire gil;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!output_types_initialized_.load(std::memory_order_relaxed)) {
      pybind11::function overload = pybind11::get_overload(
          static_cast<const TransformBase*>(this), "OutputTypes");
      if (overload) {
        auto o = overload();
        output_types_cache_ = o.cast<std::vector<api::TypeInfo>>();
        output_types_initialized_.store(true, std::memory_order_release);
      } else {
        pybind11::pybind11_fail(
            "Tried to call pure virtual function "
            "\"TransformBase::OutputTypes\"");
      }
    }
    return output_types_cache_;
  }

  absl::string_view DebugString() const override {
    if (debug_string_initialized_.load(std::memory_order_acquire)) {
      return debug_string_cache_;
    }
    pybind11::gil_scoped_acquire gil;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!debug_string_initialized_.load(std::memory_order_relaxed)) {
      pybind11::function overload = pybind11::get_overload(
          static_cast<const TransformBase*>(this), "DebugString");
      if (overload) {
        auto o = overload();
        debug_string_cache_ = o.cast<std::string>();
        debug_string_initialized_.store(true, std::memory_order_release);
      } else {
        pybind11::pybind11_fail(
            "Tried to call pure virtual function "
            "\"TransformBase::DebugString\"");
      }
    }
    return debug_string_cache_;
  }

  absl::StatusOr<TransformConfig> ToProto() const override {
    PYBIND11_OVERRIDE_PURE(absl::StatusOr<TransformConfig>, TransformBase,
                           ToProto);
  }

 private:
  mutable std::mutex mutex_;
  mutable std::atomic<bool> input_types_initialized_{false};
  mutable std::atomic<bool> output_types_initialized_{false};
  mutable std::atomic<bool> debug_string_initialized_{false};
  mutable std::string debug_string_cache_;
  mutable std::vector<api::TypeInfo> input_types_cache_;
  mutable std::vector<api::TypeInfo> output_types_cache_;
};

// Register a specific class in the pybind11 module.
// Note: This needs to be instantiated per-class.
template <typename T>
void RegisterPybindTransformExt(pybind11::module& m, const std::string& name) {
  pybind_internal::SetFactory<T>(m, name);
}

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_TRANSFORM_PYBIND_H_
