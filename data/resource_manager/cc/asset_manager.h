#ifndef THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_ASSET_MANAGER_H_
#define THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_ASSET_MANAGER_H_

#include <memory>
#include <string>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @neptune

namespace neptune::data {

class AssetManager {
 public:
  AssetManager(const AssetManager&) = delete;
  AssetManager& operator=(const AssetManager&) = delete;
  static AssetManager& GetInstance();
  absl::Status AddAsset(absl::string_view name, absl::string_view path);
  absl::StatusOr<const Asset*> GetAsset(absl::string_view name);

 private:
  AssetManager() = default;
  friend absl::NoDestructor<AssetManager>;
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<Asset>> assets_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_ASSET_MANAGER_H_
