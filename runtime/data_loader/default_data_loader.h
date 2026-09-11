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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_DEFAULT_DATA_LOADER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_DEFAULT_DATA_LOADER_H_

#include <memory>
#include <optional>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "runtime/data_loader/data_loader.h"  // from @neptune
#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

// TODO: Revamp the checkpoint-related tests.
// Default data loader (i.e. model params loader) for loading params (e.g.
// weights) from orbax checkpoint.
class DefaultDataLoader : public DataLoader {
 public:
  static absl::StatusOr<std::unique_ptr<DefaultDataLoader>> Create(
      CreateOptions create_options);

  absl::StatusOr<LoadResult> Load(
      const neptune_model_manifest::ExternalValue& value_to_load,
      std::optional<absl::string_view> model_path,
      const AssignedDevices* absl_nullable assigned_devices,
      std::optional<const neptune_model_manifest::Manifest*> manifest,
      std::optional<absl::Span<const neptune_model_manifest::Function* const>>
          stablehlo_funcs) override;

 private:
  explicit DefaultDataLoader(CreateOptions create_options);

  absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client_;
  // Thread pool for loading checkpoints in the background. Not owned.
  // TODO(b406294940): remove this once implemented: NOLINTNEXTLINE
  tsl::thread::ThreadPool* absl_nullable checkpoint_loader_thread_pool_;
};

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_DEFAULT_DATA_LOADER_H_
