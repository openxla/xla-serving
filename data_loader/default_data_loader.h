#ifndef THIRD_PARTY_NEPTUNE_DATA_LOADER_DEFAULT_DATA_LOADER_H_
#define THIRD_PARTY_NEPTUNE_DATA_LOADER_DEFAULT_DATA_LOADER_H_

#include <memory>
#include <optional>

#include "third_party/neptune/data_loader/data_loader.h"
#include "learning/infra/mira/experimental/orbax_model/interpreter/manifest_utils.h"
#include "third_party/absl/base/nullability.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.proto.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"

namespace jsv {

// TODO: Revamp the checkpoint-related tests.
// Default data loader (i.e. model params loader) for loading params (e.g.
// weights) from orbax checkpoint.
class DefaultDataLoader : public DataLoader {
 public:
  static absl::StatusOr<std::unique_ptr<DefaultDataLoader>> Create(
      CreateOptions create_options);

  absl::StatusOr<LoadResult> Load(
      const orbax_model_manifest::ExternalValue& value_to_load,
      std::optional<absl::string_view> model_path,
      const AssignedDevices* absl_nullable assigned_devices,
      std::optional<const orbax_model_manifest::Manifest*> manifest,
      std::optional<absl::Span<const orbax_model_manifest::Function* const>>
          stablehlo_funcs) override;

 private:
  explicit DefaultDataLoader(CreateOptions create_options);

  absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client_;
  // Thread pool for loading checkpoints in the background. Not owned.
  // TODO(b406294940): remove this once implemented: NOLINTNEXTLINE
  tsl::thread::ThreadPool* absl_nullable checkpoint_loader_thread_pool_;
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_DATA_LOADER_DEFAULT_DATA_LOADER_H_
