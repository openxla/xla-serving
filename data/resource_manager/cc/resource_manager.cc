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

#include "data/resource_manager/cc/resource_manager.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "data/resource_manager/cc/resource_internal.h"  // from @xla_serving
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @xla_serving

namespace xla_serving::data {

namespace {

bool HasDirectoryTraversal(absl::string_view path) {
  size_t pos = 0;
  while (pos < path.size()) {
    size_t next_separator = path.find_first_of("/\\", pos);
    absl::string_view segment = path.substr(pos, next_separator - pos);
    if (segment == "..") {
      return true;
    }
    if (next_separator == absl::string_view::npos) {
      break;
    }
    pos = next_separator + 1;
  }
  return false;
}

bool IsSafePath(absl::string_view path) { return !HasDirectoryTraversal(path); }

}  // namespace

absl::StatusOr<std::unique_ptr<ResourceManager>> ResourceManager::Create(
    std::vector<std::shared_ptr<Resource>> resources) {
  auto resource_manager = std::make_unique<ResourceManager>();
  for (auto& resource : resources) {
    RETURN_IF_ERROR(resource_manager->AddResource(std::move(resource)));
  }
  return resource_manager;
}

absl::Status ResourceManager::AddResourceInternal(
    std::shared_ptr<Resource> resource, bool return_error_on_duplicate_id) {
  if (resource == nullptr) {
    return absl::InvalidArgumentError("Cannot add nullptr as resource.");
  }
  absl::MutexLock lock(mu_);
  std::shared_ptr<Resource>& r = resources_[resource->id()];
  if (r != nullptr) {
    if (return_error_on_duplicate_id) {
      return absl::AlreadyExistsError(absl::StrCat(
          "Failed to add resource (id: ", resource->id(),
          " name: ", resource->name(), ") because resource (id: ", r->id(),
          " name: ", r->name(), " already exists."));
    }
    // Silently ignore the new resource.
    return absl::OkStatus();
  }
  r = std::move(resource);
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<Resource>> ResourceManager::GetResource(
    uint64_t id) const {
  absl::MutexLock lock(mu_);
  auto it = resources_.find(id);
  if (it == resources_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Resource with id ", id, " not found."));
  }
  if (it->second == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Resource with id ", id,
                     " in an invalid state: nullptr inside ResourceManager."));
  }
  return it->second;
}

std::vector<Asset> ResourceManager::GetAllAssets() const {
  absl::MutexLock lock(mu_);
  std::vector<Asset> all_assets;
  for (const auto& [key, resource] : resources_) {
    for (const auto& asset : resource->assets()) {
      all_assets.push_back(asset);
    }
  }
  return all_assets;
}

std::vector<std::shared_ptr<Resource>> ResourceManager::GetAllResources()
    const {
  absl::MutexLock lock(mu_);
  std::vector<std::shared_ptr<Resource>> resources;
  resources.reserve(resources_.size());
  for (const auto& [name, resource] : resources_) {
    resources.push_back(resource);
  }
  return resources;
}

absl::StatusOr<std::unique_ptr<ResourceManager>> ResourceManager::FromProto(
    const ResourceManagerProto& proto) {
  auto manager = std::make_unique<ResourceManager>();
  internal::ResourceFactory& resource_factory =
      internal::ResourceFactory::GetInstance();
  for (const auto& resource_proto : proto.resources()) {
    std::vector<Asset> assets;
    assets.reserve(resource_proto.assets_size());
    for (const auto& asset_proto : resource_proto.assets()) {
      if (!IsSafePath(asset_proto.path())) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid asset path (directory "
                         "traversal is not allowed): ",
                         asset_proto.path()));
      }
      assets.push_back(
          Asset{.name = std::string(asset_proto.name()),
                .path = std::string(asset_proto.path()),
                .content = std::string(asset_proto.content())});
    }
    uint64_t id = resource_proto.id();
    if (id == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "ResourceProto missing ID for resource: ", resource_proto.name()));
    }
    ASSIGN_OR_RETURN(
        auto resource,
        resource_factory.Create(resource_proto.class_name(), id,
                                std::string(resource_proto.name()),
                                std::move(assets)));
    // ResourceManager should have deduplicated resource IDs during
    // serialization, so we should throw an error if we detect duplicates.
    RETURN_IF_ERROR(manager->AddResource(std::move(resource)));
  }
  return manager;
}

ResourceManagerProto ResourceManager::ToProto() const {
  absl::MutexLock lock(mu_);
  ResourceManagerProto proto;
  for (const auto& [id, resource] : resources_) {
    *proto.add_resources() = resource->ToProto();
  }
  return proto;
}

}  // namespace xla_serving::data
