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

#ifndef XLA_SERVING_RUNTIME_CORE_STABLEHLO_RUNNER_BASE_H_
#define XLA_SERVING_RUNTIME_CORE_STABLEHLO_RUNNER_BASE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/hash/hash.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/interpreter/context.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @xla_serving
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

// Metadata capturing model IO signatures and variable bindings, decoupled from
// MLIR/LLVM loaded modules.
struct ModelIoMetadata {
  std::vector<std::optional<ArgSpec>> weight_specs;
  std::vector<ArgSpec> output_specs;
  int64_t num_user_inputs = 0;
  int64_t num_outputs = 0;
  bool has_streaming = false;

  static absl::StatusOr<ModelIoMetadata> CreateFromStableHlo(
      const StableHlo& stablehlo, int num_weights);
};

struct StableHloRunnerOptions {
  // IFRT Client backend (e.g. TPU, GPU) that the executables are bound to.
  absl_nonnull std::shared_ptr<xla::ifrt::Client> client;
  // Thread pool used for host-side dispatch and asynchronous compilation
  // tasks.
  tsl::thread::ThreadPool* absl_nonnull thread_pool;
  // Mapping registry to retrieve physical weight buffers by variable name.
  IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry;
  // Ordered list of original weight variable names defined by the model.
  std::vector<std::string> weights;
  // Topology and shard mapping assignment for the physical execution devices.
  absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices;
  // Optional ID for host callback dispatch in streaming execution.
  std::optional<int64_t> callback_id = std::nullopt;
};

class StableHloRunnerBase {
 public:
  virtual ~StableHloRunnerBase() = default;

  StableHloRunnerBase(StableHloRunnerBase&& other) = default;
  StableHloRunnerBase& operator=(StableHloRunnerBase&& other) = default;
  StableHloRunnerBase(const StableHloRunnerBase& other) = delete;
  StableHloRunnerBase& operator=(const StableHloRunnerBase& other) = delete;

  int GetNumInputs() const { return model_io_metadata_.num_user_inputs; }
  int GetNumOutputs() const { return model_io_metadata_.num_outputs; }
  bool has_streaming() const { return model_io_metadata_.has_streaming; }

  std::optional<int64_t> streaming_host_callback_id() const {
    return streaming_host_callback_id_;
  }

  struct ExecuteOptions {
    std::vector<Placement> placements;

    static ExecuteOptions AllDevice(int num_outputs) {
      return ExecuteOptions{.placements = std::vector<Placement>(
                                num_outputs, Placement::kDevice)};
    }

    static ExecuteOptions AllHost(int num_outputs) {
      return ExecuteOptions{
          .placements = std::vector<Placement>(num_outputs, Placement::kHost)};
    }
  };

  absl::StatusOr<std::vector<xla_serving::Value>> Execute(
      absl::Span<const xla_serving::Value> inputs, ExecuteOptions options,
      const RunContext& run_context);

  absl::StatusOr<std::vector<xla_serving::Value>> Execute(
      absl::Span<const xla_serving::Value> inputs, const RunContext& run_context) {
    return Execute(inputs, ExecuteOptions{}, run_context);
  }

  int num_executables() const {
    absl::MutexLock lock(mutex_);
    return executable_bundles_.size();
  }

  virtual absl::Status MarkFullyInitialized();

 protected:
  // Forward declarations and inner types for executable caching and per-device
  // resources
  struct CachedExecutableBundle {
    using ShardingList = std::vector<xla::ifrt::ShardingRef>;
    using PortableShardingMap =
        absl::flat_hash_map<xla::ifrt::DeviceId, ShardingList>;

    std::vector<int> module_kept_var_indices;
    int input_offset = 0;
    StableHloIOSpec original_io_spec;
    StableHloIOSpec io_spec;
    xla::ifrt::LoadedExecutableRef ifrt_executable;
    std::string program_fingerprint_with_timestamp;
    std::variant<std::monostate, ShardingList, PortableShardingMap>
        arg_shardings;
    std::variant<std::monostate, ShardingList, PortableShardingMap>
        output_shardings;
    std::optional<int64_t> callback_id;

    CachedExecutableBundle() = default;
    CachedExecutableBundle(CachedExecutableBundle&& other) = default;
    CachedExecutableBundle& operator=(CachedExecutableBundle&& other) = default;
    CachedExecutableBundle(const CachedExecutableBundle& other) = delete;
    CachedExecutableBundle& operator=(const CachedExecutableBundle& other) =
        delete;

    absl::StatusOr<absl::Span<const xla::ifrt::ShardingRef>> GetOutputShardings(
        xla::ifrt::DeviceId device_id) const {
      if (const auto* portable_map =
              std::get_if<PortableShardingMap>(&output_shardings)) {
        auto it = portable_map->find(device_id);
        if (it != portable_map->end()) {
          return absl::MakeConstSpan(it->second);
        }
        return absl::InternalError(absl::StrCat(
            "No portable single device output sharding found for device ",
            device_id.value()));
      } else if (const auto* sharding_list =
                     std::get_if<ShardingList>(&output_shardings)) {
        return absl::MakeConstSpan(*sharding_list);
      }
      return absl::InternalError("Output shardings are uninitialized.");
    }
  };

  using SharedCachedExecutableBundle = std::shared_ptr<CachedExecutableBundle>;

  class DataConverter {
   public:
    DataConverter(
        absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client,
        xla::ifrt::DeviceListRef device_list,
        tsl::thread::ThreadPool* absl_nonnull thread_pool,
        std::unique_ptr<H2DTransferExecutor> h2d_transfer_executor = nullptr);

    absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ConvertInput(
        absl::Span<const HostBuffer> inputs,
        const CachedExecutableBundle& executable_bundle,
        absl::Span<const int> stablehlo_indices) const;

    absl::StatusOr<std::vector<xla_serving::Value>> ConvertOutput(
        absl::Span<const xla::ifrt::ArrayRef> stablehlo_outputs,
        const CachedExecutableBundle& executable_bundle,
        absl::Span<const Placement> placements);

    xla::ifrt::DeviceListRef device_list() const { return device_list_; }

   private:
    absl::StatusOr<absl::Span<const xla::ifrt::ShardingRef>>
    ValidateAndResolveShardings(absl::Span<const HostBuffer> inputs,
                                const CachedExecutableBundle& executable_bundle,
                                absl::Span<const int> stablehlo_indices,
                                absl::InlinedVector<xla::ifrt::ShardingRef, 16>&
                                    mixed_shardings_storage) const;

    absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ExecuteDefaultTransfer(
        absl::Span<const HostBuffer> inputs,
        absl::Span<const xla::ifrt::ShardingRef> shardings_span) const;

    absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client_;
    const xla::ifrt::DeviceListRef device_list_;
    tsl::thread::ThreadPool* const absl_nonnull thread_pool_;
    const std::unique_ptr<H2DTransferExecutor> h2d_transfer_executor_;
    tsl::monitoring::CounterCell* const arena_cell_;
    tsl::monitoring::CounterCell* const default_cell_;
  };

  struct ResourcesPerDeviceList {
    ResourcesPerDeviceList(
        absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client,
        xla::ifrt::DeviceListRef device_list,
        tsl::thread::ThreadPool* absl_nonnull thread_pool,
        std::unique_ptr<H2DTransferExecutor> h2d_transfer_executor)
        : data_converter(std::make_unique<DataConverter>(
              std::move(ifrt_client), device_list, thread_pool,
              std::move(h2d_transfer_executor))) {}

    std::unique_ptr<DataConverter> data_converter;
    absl::StatusOr<std::vector<std::optional<xla::ifrt::ArrayRef>>>
        weight_arrays;
  };

  struct Key;
  struct KeyView {
    KeyView(const Key& key);  // NOLINT
    explicit KeyView(absl::Span<const xla::ifrt::Shape> input_shapes)
        : input_shapes(input_shapes) {}

    template <typename H>
    friend H AbslHashValue(H h, const KeyView& key) {
      for (const auto& shape : key.input_shapes) {
        h = H::combine(std::move(h), shape.dims());
      }
      return h;
    }

    friend bool operator==(const KeyView& x, const KeyView& y) {
      return x.input_shapes == y.input_shapes;
    }

    absl::Span<const xla::ifrt::Shape> input_shapes;
  };

  struct Key {
    explicit Key(absl::Span<const xla::ifrt::Shape> input_shapes)
        : input_shapes(input_shapes.begin(), input_shapes.end()) {}

    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      return AbslHashValue(std::move(h), KeyView(key.input_shapes));
    }

    friend bool operator==(const Key& x, const Key& y) {
      return KeyView(x.input_shapes) == KeyView(y.input_shapes);
    }

    std::string DebugString() const {
      return "Key: " + absl::StrJoin(input_shapes, ", ");
    }

    std::vector<xla::ifrt::Shape> input_shapes;
  };

  struct KeyHash {
    using is_transparent = void;
    size_t operator()(const KeyView& key) const { return absl::HashOf(key); }
  };

  struct KeyEq {
    using is_transparent = void;
    bool operator()(const KeyView& a, const KeyView& b) const { return a == b; }
  };

  StableHloRunnerBase(
      ModelIoMetadata model_io_metadata,
      absl_nonnull std::shared_ptr<xla::ifrt::Client> client,
      IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry,
      std::vector<std::string> weights,
      xla::ifrt::DeviceListRef assigned_device_list,
      absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices,
      ServingCoreSelector* absl_nullable serving_core_selector = nullptr,
      std::unique_ptr<H2DTransferExecutorFactory>
          h2d_transfer_executor_factory = nullptr,
      std::optional<int64_t> callback_id = std::nullopt);

  virtual tsl::Future<SharedCachedExecutableBundle> LookUpOrCreateExecutable(
      absl::Span<const xla::ifrt::Shape> input_shapes,
      absl::Span<const xla::ifrt::DType> input_dtypes) = 0;

  absl::Status Init(tsl::thread::ThreadPool* absl_nonnull thread_pool);
  absl::Status CreateResourcesPerDeviceList(
      xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* thread_pool);
  bool IsPortableExecutable() const;
  absl::StatusOr<std::vector<std::optional<xla::ifrt::ArrayRef>>> LookupWeights(
      std::optional<size_t> device_id = std::nullopt);

  const ModelIoMetadata model_io_metadata_;
  const xla::ifrt::DeviceListRef assigned_device_list_;
  absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client_;
  IfrtLoadedVariableRegistry& ifrt_loaded_variable_registry_;
  std::vector<std::string> weights_;
  absl::once_flag weights_lookup_once_;
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<Key, tsl::Future<SharedCachedExecutableBundle>, KeyHash,
                      KeyEq>
      executable_bundles_ ABSL_GUARDED_BY(mutex_);
  bool fully_initialized_ ABSL_GUARDED_BY(mutex_) = false;
  static std::atomic<uint32_t> launch_id_;
  absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices_;
  ServingCoreSelector* absl_nullable serving_core_selector_;
  std::unique_ptr<H2DTransferExecutorFactory> h2d_transfer_executor_factory_;
  std::optional<int64_t> streaming_host_callback_id_;
  absl::flat_hash_map<xla::ifrt::DeviceListRef,
                      std::unique_ptr<ResourcesPerDeviceList>>
      device_list_to_resources_map_;

  friend class StableHloRunnerTestPeer;
};

inline StableHloRunnerBase::KeyView::KeyView(const Key& key)
    : input_shapes(key.input_shapes) {}

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_STABLEHLO_RUNNER_BASE_H_
