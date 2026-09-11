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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_PYTHON_IFRT_CLIENT_FACTORY_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_PYTHON_IFRT_CLIENT_FACTORY_H_

#include <memory>
#include <string>

#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "xla/python/ifrt/client.h"  // from @xla

namespace xla_serving {

// Factory function type for creating an IFRT client.
using IfrtClientFactory =
    absl::AnyInvocable<absl::StatusOr<std::shared_ptr<xla::ifrt::Client>>(
        absl::string_view args)>;

// Registers a factory for the given platform name (e.g., "tpu").
void RegisterIfrtClientFactory(std::string platform_name,
                               IfrtClientFactory factory);

// Creates an IFRT client for the given platform.
// Checks if a factory is registered for the platform. If so, invokes it.
// If the factory returns NotFoundError, or if no factory is registered,
// proceeds to create a default PjRt client for the platform.
absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> CreateIfrtClient(
    absl::string_view platform);

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_PYTHON_IFRT_CLIENT_FACTORY_H_
