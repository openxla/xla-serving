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

#include "runtime/serving/gcp/server/ifrt_client_creator.h"  // from @xla_serving

#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "runtime/serving/gcp/server/tpu_env_init.h"  // from @xla_serving
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"  // from @xla
#include "xla/pjrt/pjrt_api.h"  // from @xla
#include "xla/pjrt/plugin/plugin_names.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

ABSL_FLAG(std::string, tpu_library_path, "",
          "Path to the TPU PJRT plugin (libtpu.so) to load when the plugin is "
          "not linked into the binary. If empty, falls back to "
          "$TPU_LIBRARY_PATH and then to /usr/lib/libtpu.so and "
          "/lib/libtpu.so (where the serving images install it).");

ABSL_FLAG(bool, tpu_auto_env, true,
          "Before loading the TPU PJRT plugin, derive single-host defaults for "
          "the TPU_* environment variables (topology, worker identity, device "
          "paths, skipping the GCE metadata server when it has no 'tpu-env' "
          "attribute) from the host's PCI devices. Variables that are already "
          "set are never overridden. Has no effect when the plugin is linked "
          "statically.");

namespace xla_serving {
namespace serving {
namespace {

// Makes sure a TPU PJRT plugin is registered and initialized.
//
// In the  build the plugin is linked statically and registered before
// `main()`. In the OSS build nothing is registered at this point, so the
// plugin is `dlopen`ed from `ResolveTpuLibraryPath()` here, after flags and
// logging are set up, so that a missing library turns into an actionable
// error instead of a crash in a static initializer.
absl::Status EnsureTpuPluginReady() {
  if (!pjrt::PjrtApi(kTpuPjrtName).ok()) {
    if (absl::GetFlag(FLAGS_tpu_auto_env)) {
      // Must happen before the plugin is loaded: libtpu reads its environment
      // during initialization.
      ApplyTpuEnvDefaults();
    }
    TF_ASSIGN_OR_RETURN(
        const std::string library_path,
        ResolveTpuLibraryPath(absl::GetFlag(FLAGS_tpu_library_path)));
    LOG(INFO) << "Loading TPU PJRT plugin from " << library_path;
    TF_RETURN_IF_ERROR(
        pjrt::LoadPjrtPlugin(kTpuPjrtName, library_path).status());
  }
  // Same as JAX: initialize the plugin (driver, LIBTPU_INIT_ARGS, TPU lock)
  // and check that its PJRT C API version is compatible with ours. For the
  // statically linked  plugin this is a no-op beyond the version check.
  TF_ASSIGN_OR_RETURN(const bool initialized,
                      pjrt::IsPjrtPluginInitialized(kTpuPjrtName));
  if (!initialized) {
    TF_RETURN_IF_ERROR(pjrt::InitializePjrtPlugin(kTpuPjrtName));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> GetIfrtClient(
    absl::string_view pjrt_platform, const IfrtClientOptions& options) {
  if (pjrt_platform == "tpu") {
    TF_RETURN_IF_ERROR(EnsureTpuPluginReady());
    TF_ASSIGN_OR_RETURN(auto pjrt_client, xla::GetCApiClient(kTpuPjrtName));
    return xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported or uncompiled PJRT platform: ", pjrt_platform));
}

}  // namespace serving
}  // namespace xla_serving
