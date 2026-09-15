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

#ifndef XLA_SERVING_RUNTIME_DATA_LOADER_DATA_LOADER_H_
#define XLA_SERVING_RUNTIME_DATA_LOADER_DATA_LOADER_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @xla_serving
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

// Abstract class for loading data (e.g. params from checkpoint).
class DataLoader {
 public:
  // Options for creating a `DataLoader` instance.
  struct CreateOptions {
    // TODO: Rename to avoid "IFRT".
    IfrtLoadedVariableRegistry* absl_nonnull
        ifrt_loaded_variable_registry;  // Not owned.
    std::optional<absl_nonnull std::shared_ptr<xla::ifrt::Client>> ifrt_client;
    // Providing a thread pool enables loading checkpoints in the background.
    // A null pointer disables background checkpoint loading. Not owned.
    tsl::thread::ThreadPool* absl_nullable checkpoint_loader_thread_pool;
  };

  // TODO: Evaluate how to better structure the return value.
  struct LoadResult {
    // The names of data (e.g. model param name) that are found, including both
    // loaded and unloaded ones.
    std::vector<std::string> data_names;
  };

  explicit DataLoader(CreateOptions create_options)
      : ifrt_loaded_variable_registry_(
            create_options.ifrt_loaded_variable_registry) {}

  virtual ~DataLoader() = default;

  // Loads the data (e.g. model params) from the `value_to_load`. Loaded data
  // will be registered with `ifrt_loaded_variable_registry_`. Only used weights
  // will be loaded. If `assigned_devices` is null, the data will be loaded to
  // all devices.
  // TODO: Unify the type style of all the optional args (`const
  // T* absl_nullable` preferred).
  virtual absl::StatusOr<LoadResult> Load(
      const xla_saved_model_manifest::ExternalValue& value_to_load,
      std::optional<absl::string_view> model_path,
      const AssignedDevices* absl_nullable assigned_devices,
      std::optional<const xla_saved_model_manifest::Manifest*> manifest,
      std::optional<absl::Span<const xla_saved_model_manifest::Function* const>>
          stable_hlo_func) = 0;

 protected:
  IfrtLoadedVariableRegistry* absl_nonnull
      ifrt_loaded_variable_registry_;  // Not owned.
};

using DataLoaderFactory =
    absl::AnyInvocable<absl::StatusOr<std::unique_ptr<DataLoader>>(
        DataLoader::CreateOptions) const>;

class DataLoaderFactoryRegistry {
 public:
  // Registers a data loader factory for the given `loader_type`.
  // Returns error if the `loader_type` is already registered.
  absl::Status RegisterDataLoaderFactory(
      xla_saved_model_manifest::LoaderType loader_type,
      DataLoaderFactory data_loader_factory);

  // Creates a `DataLoader` instance for the given `loader_type` with
  // `create_options`. Returns error if the `loader_type` is not registered.
  absl::StatusOr<std::unique_ptr<DataLoader>> CreateDataLoader(
      xla_saved_model_manifest::LoaderType loader_type,
      DataLoader::CreateOptions create_options) const;

  static DataLoaderFactoryRegistry& GetGlobalRegistry() {
    static absl::NoDestructor<DataLoaderFactoryRegistry> registry;
    return *registry;
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<xla_saved_model_manifest::LoaderType, DataLoaderFactory>
      type_to_factory_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_DATA_LOADER_DATA_LOADER_H_
