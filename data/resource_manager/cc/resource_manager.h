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

#ifndef XLA_SERVING_DATA_RESOURCE_MANAGER_CC_RESOURCE_MANAGER_H_
#define XLA_SERVING_DATA_RESOURCE_MANAGER_CC_RESOURCE_MANAGER_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @xla_serving

namespace xla_serving::data {

class ResourceManager {
 public:
  ResourceManager() = default;

  ResourceManager(const ResourceManager&) = delete;
  ResourceManager& operator=(const ResourceManager&) = delete;

  // Creates a new ResourceManager with the given resources. If a resource
  // with the same ID already exists, returns an error.
  static absl::StatusOr<std::unique_ptr<ResourceManager>> Create(
      std::vector<std::shared_ptr<Resource>> resources);

  // Adds a new resource. If the resource already exists, returns without
  // doing anything.
  absl::Status AddResourceIgnoringDuplicates(
      std::shared_ptr<Resource> resource) {
    return AddResourceInternal(std::move(resource),
                               /*return_error_on_duplicate_id=*/false);
  }
  // Adds a new resource, returning an error if a resource with the same ID
  // already exists.
  absl::Status AddResource(std::shared_ptr<Resource> resource) {
    return AddResourceInternal(std::move(resource),
                               /*return_error_on_duplicate_id=*/true);
  }
  absl::StatusOr<std::shared_ptr<Resource>> GetResource(uint64_t id) const;
  std::vector<Asset> GetAllAssets() const;
  std::vector<std::shared_ptr<Resource>> GetAllResources() const;

  static absl::StatusOr<std::unique_ptr<ResourceManager>> FromProto(
      const ResourceManagerProto& proto);

  ResourceManagerProto ToProto() const;

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<uint64_t, std::shared_ptr<Resource>> resources_
      ABSL_GUARDED_BY(mu_);

  absl::Status AddResourceInternal(std::shared_ptr<Resource> resource,
                                   bool return_error_on_duplicate_id);
};

// A helper class to interact with a collection of ResourceManagers. This class
// does not own the underlying data so it should only be used when the owning
// collection is alive.
class ResourceManagersView {
 public:
  explicit ResourceManagersView(
      absl::Span<const absl_nonnull std::unique_ptr<ResourceManager>>
          resource_managers) {
    for (const auto& resource_manager : resource_managers) {
      for (const auto& resource : resource_manager->GetAllResources()) {
        resources_map_[resource->id()] = resource.get();
      }
    }
  }

  std::vector<const Resource* absl_nonnull> GetAllResources() const {
    std::vector<const Resource* absl_nonnull> resources;
    resources.reserve(resources_map_.size());
    for (const auto& [id, resource] : resources_map_) {
      resources.push_back(resource);
    }
    return resources;
  }

  absl::StatusOr<const Resource* absl_nonnull> GetResource(uint64_t id) const {
    auto it = resources_map_.find(id);
    if (it == resources_map_.end()) {
      return absl::NotFoundError(absl::StrCat("Resource not found: ", id));
    }
    return it->second;
  }

 private:
  absl::flat_hash_map<uint64_t, const Resource* absl_nonnull> resources_map_;
};

}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_RESOURCE_MANAGER_CC_RESOURCE_MANAGER_H_
