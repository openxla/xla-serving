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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_RUNNER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_RUNNER_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/call_once.h"  // from @abseil-cpp
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/hash/hash.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @neptune
#include "runtime/core/loaded_stablehlo.h"  // from @neptune
#include "runtime/core/stablehlo.h"  // from @neptune
#include "runtime/core/stablehlo_optimizer.h"  // from @neptune
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/interpreter/context.h"  // from @neptune
#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @neptune
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @neptune
#include "google/protobuf/message.h"
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
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
#include "xla/xla_data.pb.h"  // from @xla

namespace neptune {

class StableHloRunner {
 public:
  static absl::StatusOr<std::unique_ptr<StableHloRunner>> Create(
      absl_nonnull std::shared_ptr<StableHlo> stablehlo,
      std::optional<xla::CompileOptionsProto> compile_options,
      absl_nonnull std::shared_ptr<xla::ifrt::Client> client,
      IfrtPersistentCompilationCache* absl_nullable
          persistent_compilation_cache,
      IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry,
      std::vector<std::string> weights,
      tsl::protobuf::Message* absl_nullable compilation_environment_proto,
      tsl::thread::ThreadPool* absl_nonnull thread_pool,
      StablehloOptimizer* absl_nullable optimizer,
      absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices,
      ServingCoreSelector* absl_nullable serving_core_selector,
      std::unique_ptr<H2DTransferExecutorFactory>
          h2d_transfer_executor_factory = nullptr,
      std::optional<int64_t> callback_id = std::nullopt);

  // Movable but not copyable.
  StableHloRunner(StableHloRunner&& other) = default;
  StableHloRunner& operator=(StableHloRunner&& other) = default;
  StableHloRunner(const StableHloRunner& other) = delete;
  StableHloRunner& operator=(const StableHloRunner& other) = delete;

  int GetNumInputs() const {
    absl::MutexLock lock(mutex_);
    return loaded_stablehlo_->num_user_inputs();
  }

  int GetNumOutputs() const {
    absl::MutexLock lock(mutex_);
    return loaded_stablehlo_->num_outputs();
  }

  struct ExecuteOptions {
    bool require_d2h = true;
  };

  bool has_streaming() const { return loaded_stablehlo_->has_streaming(); }

  std::optional<int64_t> streaming_host_callback_id() const {
    return streaming_host_callback_id_;
  }

  // Executes the computation.
  // Only user provided input arguments are passed to this function. The weights
  // (as passed to `Create`) will be looked in `IfrtLoadedVariableRegistry` from
  // the variable name. These will be passed as the first arguments to the
  // computation. The inputs and outputs are wrapped in `neptune::Value` containers.
  // If options.require_d2h is true, each returned value
  // will wrap a `HostBuffer`. If this option is false, it returns
  // a `neptune::DeviceBuffer`.
  absl::StatusOr<std::vector<neptune::Value>> Execute(
      absl::Span<const neptune::Value> inputs, ExecuteOptions options,
      const RunContext& run_context);

  // Executes the computation using the default ExecuteOptions.
  absl::StatusOr<std::vector<neptune::Value>> Execute(
      absl::Span<const neptune::Value> inputs, const RunContext& run_context) {
    return Execute(inputs, ExecuteOptions{}, run_context);
  }

  // Returns the number of executables in the cache.
  int num_executables() const {
    absl::MutexLock lock(mutex_);
    return executable_bundles_.size();
  }

  // Marks the runner to fully initialized. After this is called, the
  // runner will not allow new accelerator compilation.
  absl::Status MarkFullyInitialized();

 private:
  // Creates and initializes the `ResourcesPerDeviceList` for the given
  // `device_list`. This structure encapsulates the `DataConverter` (which
  // internally holds the `H2DTransferExecutor`) and the loaded weight arrays
  // specific to this device list, inserting the result into
  // `device_list_to_resources_map_`.
  absl::Status CreateResourcesPerDeviceList(
      xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* thread_pool);

  // Container for lazily-evaluated compilation artifacts and execution
  // metadata.
  struct CachedExecutableBundle {
    using ShardingList = std::vector<xla::ifrt::ShardingRef>;
    using PortableShardingMap =
        absl::flat_hash_map<xla::ifrt::DeviceId, ShardingList>;

    // The indices of model inputs that need to be passed to the main function
    // of the MLIR module.
    std::vector<int> module_kept_var_indices;
    // The offset of user input index in the main function of the MLIR module.
    // Unused weights are excluded.
    int input_offset;
    // IOSpec from the StableHLO.
    StableHloIOSpec io_spec;
    xla::ifrt::LoadedExecutableRef ifrt_executable;

    // The canonical program fingerprint of the `ifrt_executable` and the
    // timestamp in microseconds. The timestamp is added for uniqueness (e.g.,
    // in case the same executable has multiple loaded instances on the same
    // server) and as a fallback when the canonical program fingerprint is not
    // available.
    std::string program_fingerprint_with_timestamp;
    // Cached IFRT shardings for the model inputs.
    // To enforce mutual exclusivity between fixed and portable execution modes,
    // this field holds one of three states:
    // 1. std::monostate: Uninitialized default state.
    // 2. ShardingList: Used when the executable is not portable (i.e., when
    //    targeting a specific fixed device assignment).
    // 3. PortableShardingMap: Populated when portable execution is used.
    //    Currently only single device sharding is supported. To avoid repeated
    //    `malloc` calls on the inference hot path, this map pre-allocates and
    //    stores fully expanded vectors of ShardingRefs per device ID, allowing
    //    zero-allocation pointer slices (O(1) subspans) to be passed directly
    //    to the batched H2DTransferExecutor.
    std::variant<std::monostate, ShardingList, PortableShardingMap>
        arg_shardings;
    // Cached IFRT shardings for the model outputs.
    // To enforce mutual exclusivity between fixed and portable execution modes,
    // this field holds one of three states:
    // 1. std::monostate: Uninitialized default state.
    // 2. ShardingList: Used when the executable is not portable (i.e., when
    //    targeting a specific fixed device assignment).
    // 3. PortableShardingMap: Populated when portable execution is used.
    //    Currently only single device sharding is supported. To avoid repeated
    //    `malloc` calls on the inference hot path, this map pre-allocates and
    //    stores fully expanded vectors of ShardingRefs per device ID, allowing
    //    zero-allocation pointer slices (O(1) subspans) to be passed directly
    //    when resolving output buffer layouts and wrapping device arrays.
    std::variant<std::monostate, ShardingList, PortableShardingMap>
        output_shardings;
    std::optional<int64_t> callback_id;

    CachedExecutableBundle() = default;
    // Move only
    CachedExecutableBundle(CachedExecutableBundle&& other) = default;
    CachedExecutableBundle& operator=(CachedExecutableBundle&& other) = default;
    CachedExecutableBundle(const CachedExecutableBundle& other) = delete;
    CachedExecutableBundle& operator=(const CachedExecutableBundle& other) =
        delete;

    // Helper method to resolve and retrieve the output sharding layout.
    // If the executable is portable, it looks up the pre-populated sharding
    // for the specified target device ID. Otherwise, it simply returns the
    // universally cached output shardings.
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
  // A helper class to convert the inputs and outputs between
  // `HostBuffer` and `xla::ifrt::Array`.
  class DataConverter {
   public:
    DataConverter(absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client,
                  xla::ifrt::DeviceListRef device_list,
                  tsl::thread::ThreadPool* absl_nonnull thread_pool,
                  std::unique_ptr<H2DTransferExecutor> h2d_transfer_executor);

    const xla::ifrt::DeviceListRef& device_list() const { return device_list_; }

    // Converts the inputs from `HostBuffer` to `xla::ifrt::Array`.
    // Args:
    // - `inputs` are the inputs from the user excluding weights.
    // - `executable_bundle` is the executable bundle for this execution.
    // - `input_offset` is the offset of the inputs in the StableHloIOSpec.
    absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ConvertInput(
        absl::Span<const HostBuffer> inputs,
        const CachedExecutableBundle& executable_bundle,
        absl::Span<const int> stablehlo_indices) const;

    // Converts the outputs from `xla::ifrt::Array` to `HostBuffer`.
    absl::StatusOr<std::vector<HostBuffer>> ConvertOutput(
        absl::Span<const xla::ifrt::ArrayRef> stablehlo_outputs,
        const CachedExecutableBundle& executable_bundle);

   private:
    // Validates input data types and shapes against the model specification
    // and resolves the corresponding physical sharding references. Uses the
    // provided `mixed_shardings_storage` to safely hold non-contiguous
    // references without heap allocation, returning an O(1) pointer subspan
    // when possible.
    absl::StatusOr<absl::Span<const xla::ifrt::ShardingRef>>
    ValidateAndResolveShardings(absl::Span<const HostBuffer> inputs,
                                const CachedExecutableBundle& executable_bundle,
                                absl::Span<const int> stablehlo_indices,
                                absl::InlinedVector<xla::ifrt::ShardingRef, 16>&
                                    mixed_shardings_storage) const;

    // Performs standard host-to-device transfers for inputs using the default
    // IFRT MakeArrayFromHostBuffer path.
    absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ExecuteDefaultTransfer(
        absl::Span<const HostBuffer> inputs,
        absl::Span<const xla::ifrt::ShardingRef> shardings_span) const;

    absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client_;
    xla::ifrt::DeviceListRef device_list_;
    tsl::thread::ThreadPool* absl_nonnull thread_pool_;  // NOT OWNED.
    std::unique_ptr<H2DTransferExecutor> h2d_transfer_executor_;

    tsl::monitoring::CounterCell* const arena_cell_;
    tsl::monitoring::CounterCell* const default_cell_;
  };

  // Resources such as `DataConverter` and weight arrays that are organized
  // per device list.
  struct ResourcesPerDeviceList {
    ResourcesPerDeviceList(
        std::shared_ptr<xla::ifrt::Client> ifrt_client,
        xla::ifrt::DeviceListRef device_list,
        tsl::thread::ThreadPool* thread_pool,
        std::unique_ptr<H2DTransferExecutor> h2d_transfer_executor)
        : data_converter(std::make_unique<DataConverter>(
              ifrt_client, device_list, thread_pool,
              std::move(h2d_transfer_executor))) {}

    std::unique_ptr<DataConverter> data_converter;
    // IFRT array for weights. They are looked up from the registry.
    // When IFRT array is set to std::nullopt, it means the weight is unused
    // (i.e. does not needed to be passed to the StableHLO for execution). The
    // weights are looked up only once across all the executions. Lookup status
    // is also stored along with the weights and reused for all the executions.
    absl::StatusOr<std::vector<std::optional<xla::ifrt::ArrayRef>>>
        weight_arrays;
  };

  // In memory cache key that owns the input shapes.
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

  // In memory cache key that doesn't own the input shapes.
  struct KeyView {
    KeyView(const Key& key) : input_shapes(key.input_shapes) {}  // NOLINT
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

  // Key hash for heterogeneous lookups by key in absl containers.
  struct KeyHash {
    using is_transparent = void;
    size_t operator()(const KeyView& key) const { return absl::HashOf(key); }
  };

  // Key equality for heterogeneous lookups by key in absl containers.
  struct KeyEq {
    using is_transparent = void;
    bool operator()(const KeyView& a, const KeyView& b) const { return a == b; }
  };

  // 'weights' are expected to be in the same order as the variable arguments in
  // the computation. It is also assumed that variables will always be at the
  // beginning of the argument list
  StableHloRunner(
      absl_nonnull std::unique_ptr<LoadedStableHlo> loaded_stablehlo,
      std::optional<xla::CompileOptionsProto> compile_options,
      absl_nonnull std::shared_ptr<xla::ifrt::Client> client,
      IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry,
      std::vector<std::string> weights,
      xla::ifrt::DeviceListRef assigned_device_list,
      tsl::protobuf::Message* compilation_environment_proto,
      IfrtPersistentCompilationCache* absl_nullable
          persistent_compilation_cache,
      StablehloOptimizer* absl_nullable optimizer,
      absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices,
      ServingCoreSelector* absl_nullable serving_core_selector,
      std::unique_ptr<H2DTransferExecutorFactory> h2d_transfer_executor_factory,
      std::optional<int64_t> callback_id);

  using SharedCachedExecutableBundle = std::shared_ptr<CachedExecutableBundle>;

  // Looks up the executable bundle in the cache, uses the shapes of the
  // input (excluding the weights) to determine which executable to use. If not
  // found, creates a new executable bundle and inserts it into the cache.
  // Args
  // - `input_shapes` and `input_dtypes` are the shapes and dtypes of the
  // inputs (excluding the weights) irrespective of whether they are used by the
  // module "main" function.
  tsl::Future<SharedCachedExecutableBundle> LookUpOrCreateExecutable(
      absl::Span<const xla::ifrt::Shape> input_shapes,
      absl::Span<const xla::ifrt::DType> input_dtypes);

  // Creates a new executable bundle synchronously.
  // `input_shapes` and `input_dtypes` are the shapes and dtypes of the inputs
  // (excluding the weights) irrespective of whether they are used by the
  // module "main" function.
  absl::StatusOr<StableHloRunner::SharedCachedExecutableBundle>
  CreateExecutableSynchronously(
      absl::Span<const xla::ifrt::Shape> input_shapes,
      absl::Span<const xla::ifrt::DType> input_dtypes,
      std::unique_ptr<LoadedStableHlo::ModuleBundle> module_bundle,
      const StablehloOptimizer::Args& args);

  // Initializes the runner. This is called right after the constructor.
  absl::Status Init(tsl::thread::ThreadPool* absl_nonnull thread_pool);

  // Looks up the weights from the registry and returns the IFRT arrays for
  // them. When a weight is set to std::nullopt, it means the weight is unused
  // (i.e. does not needed to be passed to the StableHLO for execution). If
  // `device_id` is not set, it will use the assigned devices; otherwise, it
  // will use the device with the given ID.
  absl::StatusOr<std::vector<std::optional<xla::ifrt::ArrayRef>>> LookupWeights(
      std::optional<size_t> device_id = std::nullopt);

  // Returns true if the executable is portable.
  bool IsPortableExecutable() const;

  absl_nonnull std::unique_ptr<LoadedStableHlo> loaded_stablehlo_;

  std::optional<xla::CompileOptionsProto> compile_options_;

  const xla::ifrt::DeviceListRef assigned_device_list_;

  absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client_;

  // The persistent compilation cache is a global cache and is not owned by
  // this StableHloRunner. When it is nullptr, the persistent compilation cache
  // is disabled at ifrt serving level.
  IfrtPersistentCompilationCache* absl_nullable persistent_compilation_cache_;

  IfrtLoadedVariableRegistry& ifrt_loaded_variable_registry_;  // NOT OWNED.

  std::vector<std::string> weights_;  // a list of weight names.

  // Used to make sure we only lookup weights once.
  absl::once_flag weights_lookup_once_;

  tsl::protobuf::Message* absl_nullable
      compilation_environment_proto_;  // NOT OWNED.

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<Key, tsl::Future<SharedCachedExecutableBundle>, KeyHash,
                      KeyEq>
      executable_bundles_ ABSL_GUARDED_BY(mutex_);
  bool fully_initialized_ ABSL_GUARDED_BY(mutex_) = false;

  // Monotonically increasing launch ID that we pass to the executable.
  // TODO: Remove this launch_id once request_id is always
  // propagated through Interpreter::Run.
  static std::atomic<uint32_t> launch_id_;

  // If nullptr, no optimization will be enabled.
  StablehloOptimizer* absl_nullable optimizer_;

  absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices_;

  ServingCoreSelector* absl_nullable serving_core_selector_;

  std::unique_ptr<H2DTransferExecutorFactory> h2d_transfer_executor_factory_;
  std::optional<int64_t> streaming_host_callback_id_;

  absl::flat_hash_map<xla::ifrt::DeviceListRef,
                      std::unique_ptr<ResourcesPerDeviceList>>
      device_list_to_resources_map_;

  friend class StableHloRunnerTestPeer;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_RUNNER_H_
