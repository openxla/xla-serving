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

#ifndef XLA_SERVING_DATA_RESOURCE_MANAGER_CC_RESOURCE_INTERNAL_H_
#define XLA_SERVING_DATA_RESOURCE_MANAGER_CC_RESOURCE_INTERNAL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @neptune

namespace neptune::data {

class Resource;

namespace internal {

class ResourceTypeRegistry {
 public:
  static ResourceTypeRegistry& GetInstance();

  absl::Status Register(std::type_index type_index,
                        const std::string& class_name);

  absl::StatusOr<std::string> GetClassName(std::type_index type_index);

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<std::type_index, std::string> type_to_name_
      ABSL_GUARDED_BY(mu_);
};

class ResourceFactory {
 public:
  using FactoryFn = absl::AnyInvocable<std::shared_ptr<Resource>(
      uint64_t id, std::string name, std::vector<Asset> assets,
      std::string class_name)>;

  static ResourceFactory& GetInstance();

  absl::Status Register(const std::string& class_name, FactoryFn factory);

  absl::StatusOr<absl_nonnull std::shared_ptr<Resource>> Create(
      const std::string& class_name, uint64_t id, std::string name,
      std::vector<Asset> assets);

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, FactoryFn> factories_ ABSL_GUARDED_BY(mu_);
};

}  // namespace internal
}  // namespace neptune::data

#endif  // XLA_SERVING_DATA_RESOURCE_MANAGER_CC_RESOURCE_INTERNAL_H_
