#ifndef THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_RESOURCE_INTERNAL_H_
#define THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_RESOURCE_INTERNAL_H_

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

#endif  // THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_RESOURCE_INTERNAL_H_
