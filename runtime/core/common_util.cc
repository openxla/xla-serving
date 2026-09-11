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

#include "runtime/core/common_util.h"  // from @neptune

#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "runtime/core/constants.h"  // from @neptune
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace xla_serving {

absl::Status IsOrbaxModelDirectory(absl::string_view path) {
  tsl::Env* env = tsl::Env::Default();
  const std::string model_manifest_version_path =
      tsl::io::JoinPath(path, kNeptuneModelManifestVersionTxt);
  if (env->FileExists(model_manifest_version_path).ok()) {
    return absl::OkStatus();
  }
  const std::string orbax_model_manifest_version_path =
      tsl::io::JoinPath(path, kOrbaxModelManifestVersionTxt);
  return env->FileExists(orbax_model_manifest_version_path);
}

}  // namespace xla_serving
