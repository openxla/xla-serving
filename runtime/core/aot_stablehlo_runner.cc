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

#include "runtime/core/aot_stablehlo_runner.h"  // from @xla_serving

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/clock.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/ifrt_device_utils.h"  // from @xla_serving
#include "runtime/core/loaded_stablehlo.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/executable_build_utils.h"  // from @xla_serving
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {

namespace {}  // namespace

absl::StatusOr<std::unique_ptr<AotStableHloRunner>> AotStableHloRunner::Create(
    StableHloRunnerAotOptions options) {
  if (options.executables.empty()) {
    return absl::InvalidArgumentError("No AOT executables provided.");
  }
  if (!options.stablehlo) {
    return absl::InvalidArgumentError("No StableHlo provided in AOT options.");
  }
  if (options.executables.size() != options.stablehlo->specs().size()) {
    return absl::InvalidArgumentError(
        "Mismatched array sizes in AOT options: executables and stablehlo "
        "specs must have the same size.");
  }

  int num_weights = options.base_options.weights.size();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LoadedStableHlo> loaded_stablehlo,
      LoadedStableHlo::CreateForAot(options.stablehlo, num_weights));

  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::Device*> assigned_ifrt_devices,
      GetAssignedIfrtDevices(
          *options.base_options.client,
          options.base_options.assigned_devices->num_replicas,
          options.base_options.assigned_devices->num_cores_per_replica,
          options.base_options.assigned_devices->device_assignment));
  TF_ASSIGN_OR_RETURN(
      xla::ifrt::DeviceListRef assigned_device_list,
      options.base_options.client->MakeDeviceList(assigned_ifrt_devices));

  auto runner = absl::WrapUnique(new AotStableHloRunner(
      std::move(loaded_stablehlo), /*compile_options=*/std::nullopt,
      options.base_options.client,
      options.base_options.ifrt_loaded_variable_registry,
      std::move(options.base_options.weights), assigned_device_list,
      /*compilation_environment_proto=*/nullptr,
      /*persistent_compilation_cache=*/nullptr, /*optimizer=*/nullptr,
      options.base_options.assigned_devices,
      /*serving_core_selector=*/nullptr,
      /*h2d_transfer_executor_factory=*/nullptr,
      options.base_options.callback_id));

  runner->aot_memory_regions_ = std::move(options.aot_memory_regions);

  TF_RETURN_IF_ERROR(runner->Init(options.base_options.thread_pool));
  TF_RETURN_IF_ERROR(runner->PopulateAotExecutableCache(std::move(options)));

  {
    absl::MutexLock lock(runner->mutex_);
    runner->fully_initialized_ = true;
  }

  return runner;
}

absl::Status AotStableHloRunner::PopulateAotExecutableCache(
    StableHloRunnerAotOptions options) {
  int num_weights = loaded_stablehlo_->weight_specs().size();

  absl::MutexLock lock(mutex_);
  for (size_t i = 0; i < options.executables.size(); ++i) {
    SharedCachedExecutableBundle executable_bundle =
        std::make_shared<CachedExecutableBundle>();
    const StableHlo::Spec& spec = options.stablehlo->specs()[i];
    executable_bundle->module_kept_var_indices = spec.mlir_module_kept_var_idx;

    executable_bundle->input_offset =
        std::count_if(executable_bundle->module_kept_var_indices.begin(),
                      executable_bundle->module_kept_var_indices.end(),
                      [num_weights](int idx) { return idx < num_weights; });

    executable_bundle->io_spec.inputs.reserve(
        executable_bundle->module_kept_var_indices.size());
    for (int idx : executable_bundle->module_kept_var_indices) {
      executable_bundle->io_spec.inputs.push_back(spec.inputs[idx]);
    }
    executable_bundle->io_spec.outputs = spec.outputs;
    executable_bundle->original_io_spec.inputs =
        executable_bundle->io_spec.inputs;
    executable_bundle->original_io_spec.outputs = spec.outputs;
    executable_bundle->ifrt_executable = std::move(options.executables[i]);
    executable_bundle->program_fingerprint_with_timestamp =
        GetProgramFingerprintWithTimestamp(executable_bundle->ifrt_executable,
                                           absl::Now());

    std::vector<xla::ifrt::Shape> user_input_shapes;
    // The cache key must span every user input the caller supplies at runtime,
    // including inputs DCE pruned from the module. Slice the unpruned
    // `spec.inputs`; `io_spec.inputs` holds only kept inputs and is indexed by
    // `input_offset`, which counts kept weights rather than all weights.
    user_input_shapes.reserve(spec.inputs.size() - num_weights);
    for (size_t j = num_weights; j < spec.inputs.size(); ++j) {
      user_input_shapes.push_back(spec.inputs[j].GetIfrtShape());
    }

    // IFRT requires an explicit device mapping when executing unbound
    // (portable) AOT artifacts. Bound executables already contain their device
    // assignment.
    if (!executable_bundle->ifrt_executable->devices().has_value()) {
      CachedExecutableBundle::PortableShardingMap arg_map;
      CachedExecutableBundle::PortableShardingMap output_map;
      for (xla::ifrt::Device* device :
           options.base_options.client->addressable_devices()) {
        xla::ifrt::ShardingRef sharding =
            xla::ifrt::SingleDeviceSharding::Create(device,
                                                    xla::ifrt::MemoryKind());
        arg_map[device->Id()] = std::vector<xla::ifrt::ShardingRef>(
            executable_bundle->module_kept_var_indices.size(), sharding);
        output_map[device->Id()] = std::vector<xla::ifrt::ShardingRef>(
            executable_bundle->io_spec.outputs.size(), sharding);
      }
      executable_bundle->arg_shardings = std::move(arg_map);
      executable_bundle->output_shardings = std::move(output_map);
    } else {
      CachedExecutableBundle::ShardingList arg_list;
      arg_list.reserve(executable_bundle->io_spec.inputs.size());
      for (const auto& arg_spec : executable_bundle->io_spec.inputs) {
        TF_ASSIGN_OR_RETURN(auto ifrt_sharding,
                            xla_serving::sharding_internal::ToIfrtSharding(
                                *options.base_options.client, arg_spec.sharding,
                                assigned_device_list_));
        arg_list.push_back(std::move(ifrt_sharding));
      }
      executable_bundle->arg_shardings = std::move(arg_list);

      CachedExecutableBundle::ShardingList output_list;
      output_list.reserve(executable_bundle->io_spec.outputs.size());
      for (const auto& out_spec : executable_bundle->io_spec.outputs) {
        TF_ASSIGN_OR_RETURN(auto ifrt_sharding,
                            xla_serving::sharding_internal::ToIfrtSharding(
                                *options.base_options.client, out_spec.sharding,
                                assigned_device_list_));
        output_list.push_back(std::move(ifrt_sharding));
      }
      executable_bundle->output_shardings = std::move(output_list);
    }

    tsl::Promise<SharedCachedExecutableBundle> promise;
    tsl::Future<SharedCachedExecutableBundle> future;
    std::tie(promise, future) =
        tsl::MakePromise<SharedCachedExecutableBundle>();
    promise.Set(std::move(executable_bundle));

    executable_bundles_.emplace(Key(user_input_shapes), std::move(future));
  }
  return absl::OkStatus();
}

tsl::Future<StableHloRunner::SharedCachedExecutableBundle>
AotStableHloRunner::LookUpOrCreateExecutable(
    absl::Span<const xla::ifrt::Shape> input_shapes,
    absl::Span<const xla::ifrt::DType> input_dtypes) {
  absl::MutexLock lock(mutex_);
  const auto it = executable_bundles_.find(KeyView(input_shapes));
  if (ABSL_PREDICT_TRUE(it != executable_bundles_.end())) {
    return it->second;
  }

  return tsl::Future<SharedCachedExecutableBundle>(
      absl::FailedPreconditionError("AOT cache miss: unseen input shape."));
}

}  // namespace xla_serving
