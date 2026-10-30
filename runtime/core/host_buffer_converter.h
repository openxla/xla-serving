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

#ifndef XLA_SERVING_RUNTIME_CORE_HOST_BUFFER_CONVERTER_H_
#define XLA_SERVING_RUNTIME_CORE_HOST_BUFFER_CONVERTER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving

namespace xla_serving {

// Converts a host value of type `T` (e.g. `tensorflow::Tensor`) to a
// `HostBuffer`.
template <typename T>
using HostBufferConverterFn =
    absl::AnyInvocable<absl::StatusOr<HostBuffer>(const T&) const>;

// A converter registered for one `Value` type.
class HostBufferConverter {
 public:
  // The name given at registration, for error messages.
  absl::string_view name() const { return name_; }

  // Converts `value`, which must hold the converter's type.
  absl::StatusOr<HostBuffer> Convert(const Value& value) const {
    return convert_(value);
  }

 private:
  friend class HostBufferConverterRegistry;

  using TypeKey = const value_internal::TypeInfo*;
  using Matcher = bool (*)(const Value&);
  using ErasedFn =
      absl::AnyInvocable<absl::StatusOr<HostBuffer>(const Value&) const>;

  HostBufferConverter(std::string name, TypeKey type, Matcher matches,
                      ErasedFn convert)
      : name_(std::move(name)),
        type_(type),
        matches_(matches),
        convert_(std::move(convert)) {}

  std::string name_;
  TypeKey type_;
  Matcher matches_;
  ErasedFn convert_;
};

// `HostBufferConverter`s keyed by the `Value` type they convert, at most one
// per type.
//
// This lets kernels that only take `HostBuffer`s (e.g. the StableHLO runner)
// accept host values of framework types without `runtime/core` depending on
// those frameworks: the framework's support library registers a converter for
// its type, typically from a static registrar (see
// `RegisterHostBufferConverterOrDie`).
class HostBufferConverterRegistry {
 public:
  // The process-wide registry that the free functions below use.
  static HostBufferConverterRegistry& Global();

  // An empty registry, e.g. for unit tests.
  HostBufferConverterRegistry() = default;
  HostBufferConverterRegistry(const HostBufferConverterRegistry&) = delete;
  HostBufferConverterRegistry& operator=(const HostBufferConverterRegistry&) =
      delete;

  // Registers `fn` as the converter for values holding a `T`. Returns
  // `AlreadyExists`, naming both converters, if `T` already has one.
  // Thread-safe.
  template <typename T>
  absl::Status Register(absl::string_view name, HostBufferConverterFn<T> fn) {
    return RegisterImpl(
        std::string(name), value_internal::GetTypeInfo<T>(),
        [](const Value& value) { return value.IsType<T>(); },
        [fn = std::move(fn)](const Value& value) -> absl::StatusOr<HostBuffer> {
          return fn(value.Get<T>());
        });
  }

  // Test-only: removes the converter for `T`, if any. Returns whether there was
  // one. Production code registers converters once, from static registrars, and
  // never removes them.
  // TODO: Move this to a test util.
  template <typename T>
  bool Unregister() {
    return UnregisterImpl(value_internal::GetTypeInfo<T>());
  }

  // Returns the converter for `value`'s type, or null if there is none. The
  // converter stays usable even if it is unregistered meanwhile. Thread-safe;
  // it takes a short reader lock only to copy the snapshot pointer, then looks
  // up the converter in the snapshot without the lock.
  std::shared_ptr<const HostBufferConverter> Find(const Value& value) const;

 private:
  absl::Status RegisterImpl(std::string name, HostBufferConverter::TypeKey type,
                            HostBufferConverter::Matcher matches,
                            HostBufferConverter::ErasedFn convert);
  bool UnregisterImpl(HostBufferConverter::TypeKey type);

  using ConverterList = std::vector<std::shared_ptr<const HostBufferConverter>>;

  // The current converters, copied under a short reader lock on
  // `snapshot_mu_`. Readers then look up in this immutable snapshot without
  // the lock.
  std::shared_ptr<const ConverterList> Snapshot() const;
  // Replaces the snapshot. Requires `write_mu_`.
  void Publish(std::shared_ptr<const ConverterList> converters)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(write_mu_);

  // Copy-on-write: writers (`RegisterImpl`, `UnregisterImpl`) serialize on
  // `write_mu_`, copy the list, change the copy and publish it. `snapshot_mu_`
  // only guards the pointer; readers hold it to copy the `shared_ptr`.
  // (`std::atomic<std::shared_ptr>` is not available in all of Neptune's
  // toolchains.)
  absl::Mutex write_mu_;
  mutable absl::Mutex snapshot_mu_ ABSL_ACQUIRED_AFTER(write_mu_);
  std::shared_ptr<const ConverterList> converters_
      ABSL_GUARDED_BY(snapshot_mu_) = std::make_shared<const ConverterList>();
};

// Registers `fn` in the global registry as the converter for values holding a
// `T`. Returns `AlreadyExists` if `T` already has one.
template <typename T>
absl::Status RegisterHostBufferConverter(absl::string_view name,
                                         HostBufferConverterFn<T> fn) {
  return HostBufferConverterRegistry::Global().Register<T>(name, std::move(fn));
}

// For static registrars: like `RegisterHostBufferConverter`, but crashes if
// `T` already has a converter, so that two libraries registering converters
// for the same type are caught when the binary starts rather than one of them
// silently winning depending on static initialization order. Returns true.
template <typename T>
bool RegisterHostBufferConverterOrDie(absl::string_view name,
                                      HostBufferConverterFn<T> fn) {
  if (absl::Status status = RegisterHostBufferConverter<T>(name, std::move(fn));
      !status.ok()) {
    LOG(FATAL) << status;  // Crash OK: conflicting static registrations.
  }
  return true;
}

// Returns the global registry's converter for `value`'s type, or null.
std::shared_ptr<const HostBufferConverter> FindHostBufferConverter(
    const Value& value);

// Test-only: registers a converter in a registry (the global one by default)
// for the lifetime of this object, and then removes exactly that type's entry.
// Lets a test install a converter without leaking it into other tests.
// TODO: Move this to a test util.
class ScopedHostBufferConverterRegistration {
 public:
  // Fails with `AlreadyExists` if `T` already has a converter in `registry`.
  template <typename T>
  static absl::StatusOr<ScopedHostBufferConverterRegistration> Create(
      absl::string_view name, HostBufferConverterFn<T> fn,
      HostBufferConverterRegistry& registry =
          HostBufferConverterRegistry::Global()) {
    if (absl::Status status = registry.Register<T>(name, std::move(fn));
        !status.ok()) {
      return status;
    }
    return ScopedHostBufferConverterRegistration(&registry, &UnregisterType<T>);
  }

  ScopedHostBufferConverterRegistration(
      ScopedHostBufferConverterRegistration&& other)
      : registry_(std::exchange(other.registry_, nullptr)),
        unregister_(other.unregister_) {}
  ScopedHostBufferConverterRegistration& operator=(
      ScopedHostBufferConverterRegistration&&) = delete;
  ~ScopedHostBufferConverterRegistration() {
    if (registry_ != nullptr) unregister_(*registry_);
  }

 private:
  using Unregisterer = void (*)(HostBufferConverterRegistry&);

  template <typename T>
  static void UnregisterType(HostBufferConverterRegistry& registry) {
    registry.Unregister<T>();
  }

  ScopedHostBufferConverterRegistration(HostBufferConverterRegistry* registry,
                                        Unregisterer unregister)
      : registry_(registry), unregister_(unregister) {}

  HostBufferConverterRegistry* registry_;
  Unregisterer unregister_;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_HOST_BUFFER_CONVERTER_H_
