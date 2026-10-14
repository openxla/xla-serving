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

#include "runtime/core/common_util.h"  // from @xla_serving

#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "runtime/core/constants.h"  // from @xla_serving
#include "xla/tsl/platform/env.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace xla_serving {

absl::Status IsOrbaxModelDirectory(absl::string_view path) {
  tsl::Env* env = tsl::Env::Default();
  return env->FileExists(tsl::io::JoinPath(path, kModelVersionTxt));
}

bool IsBundle(absl::string_view model_path) {
  if (!IsOrbaxModelDirectory(model_path).ok()) {
    return false;
  }
  std::string pb_path =
      tsl::io::JoinPath(model_path, kBundleOrchestrationFileName);
  return tsl::Env::Default()->FileExists(pb_path).ok();
}

}  // namespace xla_serving
