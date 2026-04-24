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

#include "runtime/tensorflow/utils/common_utils.h"  // from @neptune

#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "runtime/core/constants.h"  // from @neptune
#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace neptune {
namespace {

// Gets the TF SavedModel supplement from the manifest.
absl::StatusOr<const neptune_model_manifest::UnstructuredData>
GetTfSavedModelSupplement(const neptune_model_manifest::Manifest& manifest) {
  const auto& dict = manifest.supplemental_info();
  auto it = dict.find(kTfSavedModelSupplementalName);
  if (it == dict.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Manifest does not have a global supplemental with name \"",
        kTfSavedModelSupplementalName, "\"."));
  }

  return it->second;
}

}  // namespace

absl::StatusOr<bool> HasTfSavedModelSupplement(absl::string_view model_path) {
  TF_ASSIGN_OR_RETURN(neptune_model_manifest::Manifest manifest,
                      GetManifest(model_path));
  const auto& dict = manifest.supplemental_info();
  return dict.contains(kTfSavedModelSupplementalName);
}

absl::StatusOr<std::string> GetTfSavedModelDirPath(
    absl::string_view model_path) {
  TF_ASSIGN_OR_RETURN(neptune_model_manifest::Manifest manifest,
                      GetManifest(model_path));
  TF_ASSIGN_OR_RETURN(
      const neptune_model_manifest::UnstructuredData tf_saved_model_supplement,
      GetTfSavedModelSupplement(manifest));

  TF_RETURN_IF_ERROR(CheckMimeAndDataType(
      tf_saved_model_supplement, {kTfSavedModelMimeType},
      neptune_model_manifest::UnstructuredData::kFileSystemLocation,
      "TF SavedModel supplement"));

  return GetAbsoluteFileSystemPath(
      model_path, tf_saved_model_supplement.file_system_location());
}
}  // namespace neptune
