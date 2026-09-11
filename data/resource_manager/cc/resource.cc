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

#include "data/resource_manager/cc/resource.h"  // from @xla_serving

#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/substitute.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "data/resource_manager/cc/resource_internal.h"  // from @xla_serving
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @xla_serving

namespace neptune::data {

namespace {

// Generates a machine-local unique ID for resources.
uint64_t GenerateResourceId() {
  // For the most part, we can expect models and data pipelines to be defined
  // within a single process; however, it's common for multiprocessing to be
  // used in Python, so we'd like to avoid collisions across processes if
  // possible in case somebody decides to try to create Resources in another
  // process and pass them back to the main process. Resource ID is a uint64_t
  // structured as follows (from least significant to most significant bits):
  //  - Bits [0-41]: 42-bit counter, allowing for 4 trillion IDs per process.
  //  - Bits [42-63]: 22-bit process id (pid)
  // 22 bits is the upper bound for Linux PIDs barring some extreme edge cases,
  // so it should be safe to assume this is unique within a single machine.
  static std::atomic<uint64_t> counter(1);
  // Upcasting from pid_t (typically int) to uint64_t should be safe because
  // Linux PIDs are not negative.
  static const uint64_t pid = getpid();
  // We start the counter at 1 because 0 is reserved for unset resource IDs in
  // the proto.
  uint64_t count = counter.fetch_add(1, std::memory_order_relaxed);
  return (pid << 42) | (count & ((1ULL << 42) - 1));
}

}  // namespace

namespace internal {

// ResourceTypeRegistry implementation *****************************************
ResourceTypeRegistry& ResourceTypeRegistry::GetInstance() {
  static absl::NoDestructor<ResourceTypeRegistry> instance;
  return *instance;
}

absl::Status ResourceTypeRegistry::Register(std::type_index type_index,
                                            const std::string& class_name) {
  absl::MutexLock lock(mu_);
  auto [it, inserted] =
      type_to_name_.insert({std::move(type_index), class_name});
  if (!inserted) {
    return absl::AlreadyExistsError(absl::Substitute(
        "Factory with class name $0 already registered.", class_name));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> ResourceTypeRegistry::GetClassName(
    std::type_index type_index) {
  absl::MutexLock lock(mu_);
  auto it = type_to_name_.find(type_index);
  if (it == type_to_name_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Could not find Resource Type with type_index: ", type_index.name()));
  }
  return it->second;
}

// ResourceFactory implementation **********************************************
ResourceFactory& ResourceFactory::GetInstance() {
  static absl::NoDestructor<ResourceFactory> instance;
  return *instance;
}

absl::Status ResourceFactory::Register(const std::string& class_name,
                                       FactoryFn factory) {
  absl::MutexLock lock(mu_);
  auto [it, inserted] = factories_.insert({class_name, std::move(factory)});
  if (!inserted) {
    return absl::AlreadyExistsError(absl::Substitute(
        "Factory with class name $0 already registered.", class_name));
  }
  return absl::OkStatus();
}

absl::StatusOr<absl_nonnull std::shared_ptr<Resource>> ResourceFactory::Create(
    const std::string& class_name, uint64_t id, std::string name,
    std::vector<Asset> assets) {
  absl::MutexLock lock(mu_);
  auto it = factories_.find(class_name);
  if (it == factories_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Could not find factory function for resource: ", class_name));
  }
  return it->second(id, std::move(name), std::move(assets), class_name);
}

}  // namespace internal

// Resource Implementation *****************************************************
Resource::Resource() : Resource(GenerateResourceId()) {}

Resource::Resource(std::vector<Asset> assets)
    : Resource(GenerateResourceId(), std::move(assets)) {}

Resource::Resource(std::string name, std::vector<Asset> assets,
                   std::string class_name)
    : Resource(GenerateResourceId(), std::move(name), std::move(assets),
               std::move(class_name)) {}

Resource::Resource(uint64_t id, std::string name, std::vector<Asset> assets,
                   std::string class_name)
    : id_(id),
      name_(std::move(name)),
      class_name_(std::move(class_name)),
      assets_(std::move(assets)) {}

Resource::Resource(uint64_t id) : Resource(id, absl::StrCat(id), {}, "") {}

Resource::Resource(uint64_t id, std::vector<Asset> assets)
    : Resource(id, absl::StrCat(id), std::move(assets), "") {}

uint64_t Resource::id() const { return id_; }

const std::string& Resource::name() const { return name_; }

const std::vector<Asset>& Resource::assets() const { return assets_; }

const std::string& Resource::class_name() const {
  absl::MutexLock lock(mu_);
  if (class_name_.empty()) {
    absl::StatusOr<std::string> class_name =
        internal::ResourceTypeRegistry::GetInstance().GetClassName(
            std::type_index(typeid(*this)));
    if (class_name.ok() && !class_name->empty()) {
      class_name_ = std::move(*class_name);
    }
  }
  return class_name_;
}

absl::Status Resource::Load() {
  absl::MutexLock lock(mu_);
  if (!loaded_) {
    RETURN_IF_ERROR(LoadImpl());
    loaded_ = true;
  }
  return absl::OkStatus();
}

ResourceProto Resource::ToProto() const {
  ResourceProto resource_proto;
  resource_proto.set_id(id());
  resource_proto.set_name(name());
  resource_proto.set_class_name(class_name());
  for (const auto& asset : assets()) {
    AssetProto* asset_proto = resource_proto.add_assets();
    asset_proto->set_name(asset.name);
    if (!asset.path.empty()) {
      asset_proto->set_path(asset.path);
    } else if (!asset.content.empty()) {
      asset_proto->set_content(asset.content);
    }
  }
  return resource_proto;
}

}  // namespace neptune::data
