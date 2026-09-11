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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_FACTORY_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_FACTORY_H_

#include <stdbool.h>

#include <memory>
#include <string>

#include "runtime/interpreter/optional_function_handler.h"  // from @neptune
#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp

namespace xla_serving {

// A factory class interface to create FunctionHandler.
//
// The factory class is registered by `RegisterOptionalFunctionHandlerFactory`
// and the handler is retrieved by `GetFunctionHandler`.
// Factory classes can be registered with string keys. All implementations of
// `OptionalFunctionHandlerFactory` should be default constructible.
class OptionalFunctionHandlerFactory {
 public:
  virtual ~OptionalFunctionHandlerFactory() = default;
  virtual std::unique_ptr<OptionalFunctionHandler> GetHandler() = 0;
};

// A static singleton registry class for OptionalFunctionHandlerFactory.
// The registry is composed at link-time.
class OptionalFunctionHandlerFactoryRegistry {
 public:
  static OptionalFunctionHandlerFactoryRegistry& Instance();

  void Register(absl::string_view key, OptionalFunctionHandlerFactory* factory)
      ABSL_LOCKS_EXCLUDED(registry_mu_);

  absl::flat_hash_map<std::string, OptionalFunctionHandlerFactory*>
  GetFactories() const ABSL_LOCKS_EXCLUDED(registry_mu_);

 private:
  friend class absl::NoDestructor<OptionalFunctionHandlerFactoryRegistry>;
  OptionalFunctionHandlerFactoryRegistry() = default;

  mutable absl::Mutex registry_mu_;
  absl::flat_hash_map<std::string, OptionalFunctionHandlerFactory*> factories_
      ABSL_GUARDED_BY(registry_mu_);
};

// Registers the factory class with the given key. This should be called at link
// time.
template <typename Factory>
void RegisterOptionalFunctionHandlerFactory(absl::string_view key) {
  OptionalFunctionHandlerFactoryRegistry::Instance().Register(key,
                                                              new Factory());
}

// A router class which maintains a mapping of processor keys to
// singleton OptionalFunctionHandler instances. Each OptionalFunctionHandler is
// created once and maintained in an in-memory map.
class OptionalFunctionHandlerRouter {
 public:
  OptionalFunctionHandlerRouter() {
    const absl::flat_hash_map<std::string, OptionalFunctionHandlerFactory*>
        factories =
            OptionalFunctionHandlerFactoryRegistry::Instance().GetFactories();

    for (const auto& [key, factory] : factories) {
      handlers_[key] = factory->GetHandler();
    }
  }

  // Returns the handler for the given key. Returns an internal error if the
  // handler is not found.
  absl::StatusOr<OptionalFunctionHandler*> GetHandler(
      absl::string_view key) const ABSL_LOCKS_EXCLUDED(router_mu_);

 private:
  mutable absl::Mutex router_mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<OptionalFunctionHandler>>
      handlers_ ABSL_GUARDED_BY(router_mu_);
};

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_FACTORY_H_
