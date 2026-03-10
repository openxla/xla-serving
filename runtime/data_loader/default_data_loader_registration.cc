#include <memory>
#include <utility>

#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/data_loader/data_loader.h"  // from @neptune
#include "runtime/data_loader/default_data_loader.h"  // from @neptune
#include "third_party/py/orbax/experimental/model/core/protos/manifest.pb.h"

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
