#include "data/resource_manager/cc/asset_manager.h"  // from @neptune

#include <memory>
#include <string>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @neptune

namespace neptune::data {

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

}  // namespace neptune::data
