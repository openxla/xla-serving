#include <memory>
#include <utility>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "runtime/data_loader/data_loader.h"  // from @neptune
#include "runtime/data_loader/default_data_loader.h"  // from @neptune

namespace neptune {
namespace {

bool kUnused = []() {
  const auto s =
      DataLoaderFactoryRegistry::GetGlobalRegistry().RegisterDataLoaderFactory(
          neptune_model_manifest::LoaderType::LOADER_TYPE_DEFAULT,
          [](DataLoader::CreateOptions create_options)
              -> absl::StatusOr<std::unique_ptr<DataLoader>> {
            return DefaultDataLoader::Create(std::move(create_options));
          });
  LOG_IF(FATAL, !s.ok()) << s;
  return true;
}();

}  // namespace
}  // namespace neptune
