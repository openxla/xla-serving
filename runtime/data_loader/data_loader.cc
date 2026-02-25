#include "third_party/neptune/runtime/data_loader/data_loader.h"

#include <memory>
#include <utility>

#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/functional/any_invocable.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.proto.h"

namespace jsv {

absl::Status DataLoaderFactoryRegistry::RegisterDataLoaderFactory(
    orbax_model_manifest::LoaderType loader_type,
    DataLoaderFactory loader_factory) {
  absl::MutexLock lock(mu_);
  if (!type_to_factory_.try_emplace(loader_type, std::move(loader_factory))
           .second) {
    return absl::AlreadyExistsError(
        absl::StrCat("Data loader factory with name ",
                     orbax_model_manifest::LoaderType_Name(loader_type),
                     " already exists."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<DataLoader>>
DataLoaderFactoryRegistry::CreateDataLoader(
    orbax_model_manifest::LoaderType loader_type,
    DataLoader::CreateOptions create_options) const {
  absl::MutexLock lock(mu_);
  const auto it = type_to_factory_.find(loader_type);
  if (it == type_to_factory_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Data loader with name ",
        orbax_model_manifest::LoaderType_Name(loader_type), " not found."));
  }
  return it->second(std::move(create_options));
}

}  // namespace jsv
