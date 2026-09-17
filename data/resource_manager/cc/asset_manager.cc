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

#include "data/resource_manager/cc/asset_manager.h"  // from @xla_serving

#include <memory>
#include <string>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @xla_serving

namespace xla_serving::data {

AssetManager& AssetManager::GetInstance() {
  static absl::NoDestructor<AssetManager> instance;
  return *instance;
}

absl::Status AssetManager::AddAsset(absl::string_view name,
                                    absl::string_view path) {
  absl::MutexLock lock(mu_);
  std::unique_ptr<Asset>& asset = assets_[name];
  if (asset != nullptr) {
    return absl::AlreadyExistsError(
        absl::StrCat("Asset with name ", name, " already exists."));
  }
  asset = std::make_unique<Asset>(
      Asset{.name = std::string(name), .path = std::string(path)});
  return absl::OkStatus();
}

absl::StatusOr<const Asset*> AssetManager::GetAsset(absl::string_view name) {
  absl::MutexLock lock(mu_);
  auto it = assets_.find(name);
  if (it == assets_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Asset with name ", name, " not found."));
  }
  return it->second.get();
}

}  // namespace xla_serving::data
