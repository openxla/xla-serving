#include <memory>
#include <utility>

#include "third_party/neptune/data_loader/data_loader.h"
#include "third_party/neptune/data_loader/default_data_loader.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.proto.h"

namespace jsv {
namespace {

bool kUnused = []() {
  const auto s =
      DataLoaderFactoryRegistry::GetGlobalRegistry().RegisterDataLoaderFactory(
          orbax_model_manifest::LoaderType::LOADER_TYPE_DEFAULT,
          [](DataLoader::CreateOptions create_options)
              -> absl::StatusOr<std::unique_ptr<DataLoader>> {
            return DefaultDataLoader::Create(std::move(create_options));
          });
  LOG_IF(FATAL, !s.ok()) << s;
  return true;
}();

}  // namespace
}  // namespace jsv
