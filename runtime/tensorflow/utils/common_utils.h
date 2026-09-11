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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_COMMON_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_COMMON_UTILS_H_

#include <string>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp

namespace xla_serving {

// Returns true if the model has a TF SavedModel supplement. This can be used to
// determine if the model has TF components. Returns an error if the model path
// is invalid.
absl::StatusOr<bool> HasTfSavedModelSupplement(absl::string_view model_path);

absl::StatusOr<std::string> GetTfSavedModelDirPath(
    absl::string_view model_path);

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_COMMON_UTILS_H_
