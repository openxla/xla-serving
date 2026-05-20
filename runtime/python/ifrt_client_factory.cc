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

#include "runtime/python/ifrt_client_factory.h"  // from @neptune

#include <memory>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"  // from @xla
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace neptune {

namespace {

absl::flat_hash_map<std::string, IfrtClientFactory>* GetRegistry() {
  static absl::NoDestructor<absl::flat_hash_map<std::string, IfrtClientFactory>>
      registry;
  return registry.get();
}

}  // namespace

void RegisterIfrtClientFactory(std::string platform_name,
                               IfrtClientFactory factory) {
  (*GetRegistry())[platform_name] = std::move(factory);
}

absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> CreateIfrtClient(
    absl::string_view platform) {
  absl::flat_hash_map<std::string, IfrtClientFactory>* registry = GetRegistry();
  auto it = registry->find(platform);
  if (it != registry->end()) {
    absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> client =
        it->second(platform);
    if (client.ok()) {
      return client;
    }
    if (!absl::IsNotFound(client.status())) {
      return client;
    }
    // If NotFound, fall back to default creation.
  }

  // Default PjRt creation logic.
  if (platform == "cpu") {
    TF_ASSIGN_OR_RETURN(auto pjrt_client,
                        xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
    return xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported platform: ", platform));
}

}  // namespace neptune
