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

#include "runtime/interpreter/optional_function_handler_factory.h"  // from @neptune

#include <memory>
#include <string>

#include "runtime/interpreter/optional_function_handler.h"  // from @neptune
#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp

namespace xla_serving {

OptionalFunctionHandlerFactoryRegistry&
OptionalFunctionHandlerFactoryRegistry::Instance() {
  static absl::NoDestructor<OptionalFunctionHandlerFactoryRegistry> instance;
  return *instance;
}

void OptionalFunctionHandlerFactoryRegistry::Register(
    absl::string_view key, OptionalFunctionHandlerFactory* factory) {
  absl::MutexLock lock(registry_mu_);
  auto [_, inserted] = factories_.try_emplace(key, factory);
  CHECK(inserted) << "An OptionalFunctionHandlerFactory is already registered "
                     "for key: "
                  << key;
}

absl::flat_hash_map<std::string, OptionalFunctionHandlerFactory*>
OptionalFunctionHandlerFactoryRegistry::GetFactories() const {
  absl::MutexLock lock(registry_mu_);
  return factories_;
}

absl::StatusOr<OptionalFunctionHandler*>
OptionalFunctionHandlerRouter::GetHandler(absl::string_view key) const {
  absl::MutexLock lock(router_mu_);
  if (auto it = handlers_.find(key); it != handlers_.end()) {
    return it->second.get();
  }
  return absl::InternalError(absl::StrCat(
      "No `OptionalFunctionHandler` is found for key: ", key,
      ". This can be because `InitializeHandlers` was not called or the "
      "handler is not registered."));
}

}  // namespace xla_serving
