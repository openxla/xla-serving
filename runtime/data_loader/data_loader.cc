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

#include "runtime/data_loader/data_loader.h"  // from @neptune

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "neptune_model/protos/manifest.pb.h"  // from @neptune

namespace neptune {

absl::Status DataLoaderFactoryRegistry::RegisterDataLoaderFactory(
    neptune_model_manifest::LoaderType loader_type,
    DataLoaderFactory loader_factory) {
  absl::MutexLock lock(mu_);
  if (!type_to_factory_.try_emplace(loader_type, std::move(loader_factory))
           .second) {
    return absl::AlreadyExistsError(
        absl::StrCat("Data loader factory with name ",
                     neptune_model_manifest::LoaderType_Name(loader_type),
                     " already exists."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<DataLoader>>
DataLoaderFactoryRegistry::CreateDataLoader(
    neptune_model_manifest::LoaderType loader_type,
    DataLoader::CreateOptions create_options) const {
  absl::MutexLock lock(mu_);
  const auto it = type_to_factory_.find(loader_type);
  if (it == type_to_factory_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Data loader with name ",
        neptune_model_manifest::LoaderType_Name(loader_type), " not found."));
  }
  return it->second(std::move(create_options));
}

}  // namespace neptune
