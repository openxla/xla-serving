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

#include "runtime/core/stablehlo_runner.h"  // from @xla_serving

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/call_once.h"  // from @abseil-cpp
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/optimization.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/log/vlog_is_on.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/clock.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "runtime/core/device_buffer.h"  // from @xla_serving
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/ifrt_device_utils.h"  // from @xla_serving
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/core/loaded_stablehlo.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_optimizer.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/core/streaming/stream.h"  // from @xla_serving
#include "runtime/core/streaming/utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/interpreter/context.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/executable_build_utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @xla_serving
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @xla_serving
#include "runtime/libs/mlir/dump_mlir_util.h"  // from @xla_serving
#include "google/protobuf/message.h"
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/layout.h"  // from @xla
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/array_spec.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/host_callback.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/refine_polymorphic_shapes.h"  // from @xla
#include "xla/service/computation_placer.h"  // from @xla
#include "xla/status_macros.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/framework/serving_device_selector.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace xla_serving {

std::atomic<uint32_t> StableHloRunner::launch_id_(1);

namespace {

// Returns the device assignment from the given IFRT devices list.
absl::StatusOr<xla::DeviceAssignment> GetRuntimeXlaDeviceAssignment(
    const xla::ifrt::DeviceListRef& devices, int num_replicas,
    int num_cores_per_replica) {
  const int num_devices = num_replicas * num_cores_per_replica;
  if (devices->size() != num_devices) {
    return absl::FailedPreconditionError(
        absl::StrCat("Device assignment has ", devices->size(),
                     " devices, but expected ", num_devices));
  }
  xla::DeviceAssignment da(num_replicas, num_cores_per_replica);
  int device_index = 0;
  for (int replica_idx = 0; replica_idx < num_replicas; replica_idx++) {
    for (int core_idx = 0; core_idx < num_cores_per_replica;
         core_idx++, device_index++) {
      da(replica_idx, core_idx) =
          devices->devices()[device_index]->Id().value();
    }
  }
  return da;
}



auto* input_transfer_counter =
    ::tsl::monitoring::Counter<1>::New("/jsv/input_transfer/dispatched",
                                       "Tracks how many transfers were routed "
                                       "to the Arena vs. Default IFRT path.",
                                       "transfer_path");

void AnnotateConstantWeights(mlir::ModuleOp module, int input_offset) {
  if (auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main")) {
    mlir::Builder builder(main_func.getContext());
    mlir::StringAttr frontend_attr_name =
        builder.getStringAttr("mhlo.frontend_attributes");
    mlir::StringAttr is_constant_name = builder.getStringAttr("is_constant");
    mlir::BoolAttr true_attr = builder.getBoolAttr(true);

    for (int i = 0; i < input_offset; ++i) {
      mlir::NamedAttrList attrs;
      if (auto existing_attrs =
              main_func.getArgAttrOfType<mlir::DictionaryAttr>(
                  i, frontend_attr_name)) {
        attrs = mlir::NamedAttrList(existing_attrs);
      }
      attrs.set(is_constant_name, true_attr);
      main_func.setArgAttr(i, frontend_attr_name,
                           attrs.getDictionary(builder.getContext()));
    }
  }
}

absl::Status ValidateDeviceBufferInput(const xla::ifrt::Array& original_input,
                                       xla::ifrt::DType target_dtype,
                                       const xla::ifrt::Shape& target_shape,
                                       int idx_in_stablehlo_spec) {
  if (original_input.dtype() != target_dtype) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input DType mismatch for device buffer at index ",
                     idx_in_stablehlo_spec, ". Expected ", target_dtype,
                     ", but got ", original_input.dtype()));
  }
  if (original_input.shape() != target_shape) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Zero-copy D2D execution requires the input "
        "shape to exactly match the optimized model's target shape. "
        "Expected ",
        target_shape, ", but got ", original_input.shape(), ". ",
        "To fix this, you must pre-reshape the device array to match the "
        "optimized signature. Otherwise, provide a HostBuffer to allow "
        "StableHloRunner to perform the reshape via host memory."));
  }
  return absl::OkStatus();
}

absl::Status ValidateOutputArray(const xla::ifrt::Array& array_for_copy,
                                 xla::ifrt::DType expected_dtype,
                                 const xla::ifrt::Shape& expected_ifrt_shape,
                                 int output_index) {
  if (array_for_copy.dtype() != expected_dtype) {
    return absl::InternalError(absl::StrCat(
        "Output DType mismatch at index ", output_index, ". Expected ",
        expected_dtype, ", but got ", array_for_copy.dtype()));
  }
  if (array_for_copy.shape() != expected_ifrt_shape) {
    return absl::InternalError(absl::StrCat(
        "Output shape mismatch at index ", output_index, ". Expected ",
        expected_ifrt_shape, ", but got ", array_for_copy.shape()));
  }
  return absl::OkStatus();
}

}  // namespace

StableHloRunner::DataConverter::DataConverter(
    absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client,
    xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool,
    std::unique_ptr<H2DTransferExecutor> h2d_transfer_executor)
    : ifrt_client_(std::move(ifrt_client)),
      device_list_(std::move(device_list)),
      thread_pool_(thread_pool),
      h2d_transfer_executor_(std::move(h2d_transfer_executor)),
      arena_cell_(input_transfer_counter->GetCell("arena")),
      default_cell_(input_transfer_counter->GetCell("default")) {}

absl::StatusOr<absl::Span<const xla::ifrt::ShardingRef>>
StableHloRunner::DataConverter::ValidateAndResolveShardings(
    absl::Span<const HostBuffer> inputs,
    const CachedExecutableBundle& executable_bundle,
    absl::Span<const int> stablehlo_indices,
    absl::InlinedVector<xla::ifrt::ShardingRef, 16>& mixed_shardings_storage)
    const {
  if (stablehlo_indices.size() != inputs.size()) {
    return absl::InvalidArgumentError(
        "Mismatched inputs and stablehlo_indices size.");
  }

  // Validate data types and shapes before passing to executor to prevent
  // buffer over-reads during memcpy.
  for (int i = 0; i < inputs.size(); ++i) {
    int idx_in_stablehlo_spec = stablehlo_indices[i];
    if (idx_in_stablehlo_spec >= executable_bundle.io_spec.inputs.size()) {
      return absl::InvalidArgumentError("Index out of bounds.");
    }
    const ArgSpec& input_spec =
        executable_bundle.io_spec.inputs[idx_in_stablehlo_spec];
    if (inputs[i].dtype() != input_spec.dtype) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input dtype mismatch. Expected: ", input_spec.dtype,
                       ", but got: ", inputs[i].dtype()));
    }
    TF_RETURN_IF_ERROR(InputShapeEqual(input_spec.shape, inputs[i].shape()));
  }

  absl::Span<const xla::ifrt::ShardingRef> all_shardings;
  if (const auto* portable_map = std::get_if<
          StableHloRunner::CachedExecutableBundle::PortableShardingMap>(
          &executable_bundle.arg_shardings)) {
    TF_RET_CHECK(!device_list_->devices().empty())
        << "Device list cannot be empty";
    auto it = portable_map->find(device_list_->devices()[0]->Id());
    if (it == portable_map->end()) {
      return absl::FailedPreconditionError(
          absl::StrCat("No portable single device sharding found for device ",
                       device_list_->devices()[0]->Id()));
    }
    all_shardings = absl::MakeConstSpan(it->second);
  } else if (const auto* sharding_list = std::get_if<
                 StableHloRunner::CachedExecutableBundle::ShardingList>(
                 &executable_bundle.arg_shardings)) {
    all_shardings = absl::MakeConstSpan(*sharding_list);
  } else {
    return absl::InternalError("Arg shardings are uninitialized.");
  }

  // Check if requested indices are strictly contiguous (The Fast Path)
  bool is_contiguous = true;
  for (int i = 0; i < stablehlo_indices.size(); ++i) {
    if (stablehlo_indices[i] != executable_bundle.input_offset + i) {
      is_contiguous = false;
      break;
    }
  }

  if (is_contiguous) {
    if (executable_bundle.input_offset + stablehlo_indices.size() >
        all_shardings.size()) {
      return absl::InternalError(
          "IFRT Executable arg shardings size mismatch.");
    }
    // O(1) Zero-allocation, zero-refcount-bump fast path
    return all_shardings.subspan(executable_bundle.input_offset,
                                 stablehlo_indices.size());
  }

  // Fallback for mixed inputs, pushing into the caller-owned InlinedVector
  mixed_shardings_storage.reserve(stablehlo_indices.size());
  for (int idx : stablehlo_indices) {
    if (idx >= all_shardings.size()) {
      return absl::InternalError(
          "IFRT Executable arg shardings size mismatch.");
    }
    mixed_shardings_storage.push_back(all_shardings[idx]);
  }
  return absl::MakeConstSpan(mixed_shardings_storage);
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
StableHloRunner::DataConverter::ExecuteDefaultTransfer(
    absl::Span<const HostBuffer> inputs,
    absl::Span<const xla::ifrt::ShardingRef> shardings_span) const {
  std::vector<xla::ifrt::ArrayRef> converted_inputs;
  converted_inputs.reserve(inputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    const HostBuffer& input_data = inputs[i];
    VLOG(2) << "Converting input of shape " << input_data.shape();

    xla::ifrt::ArrayRef array;
    TF_ASSIGN_OR_RETURN(array, sharding_internal::MakeArrayFromHostBuffer(
                                   *ifrt_client_, input_data, device_list_,
                                   shardings_span[i], *thread_pool_));

    converted_inputs.push_back(std::move(array));
  }
  return converted_inputs;
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
StableHloRunner::DataConverter::ConvertInput(
    absl::Span<const HostBuffer> inputs,
    const CachedExecutableBundle& executable_bundle,
    absl::Span<const int> stablehlo_indices) const {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(h2d_transfer_executor_ != nullptr
                                            ? "ArenaTransferFromHostToDevice"
                                            : "TransferFromHostToDevice",
                                        {});
  });

  // Storage must live for the duration of ConvertInput to prevent dangling span
  // pointers
  absl::InlinedVector<xla::ifrt::ShardingRef, 16> mixed_shardings_storage;
  TF_ASSIGN_OR_RETURN(
      absl::Span<const xla::ifrt::ShardingRef> shardings_span,
      ValidateAndResolveShardings(inputs, executable_bundle, stablehlo_indices,
                                  mixed_shardings_storage));

  if (h2d_transfer_executor_ != nullptr) {
    arena_cell_->IncrementBy(1);
    return h2d_transfer_executor_->ExecuteTransfer(inputs, shardings_span);
  }

  default_cell_->IncrementBy(1);
  return ExecuteDefaultTransfer(inputs, shardings_span);
}

absl::StatusOr<std::vector<xla_serving::Value>>
StableHloRunner::DataConverter::ConvertOutput(
    absl::Span<const xla::ifrt::ArrayRef> stablehlo_outputs,
    const CachedExecutableBundle& executable_bundle,
    absl::Span<const Placement> placements) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode("TransferFromDeviceToHost", {});
  });
  if (placements.size() != stablehlo_outputs.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatched placements size (", placements.size(),
                     ") and outputs size (", stablehlo_outputs.size(), ")."));
  }
  std::vector<std::optional<tsl::Future<HostBuffer>>> output_futures(
      stablehlo_outputs.size());

  TF_RET_CHECK(!device_list_->devices().empty())
      << "Device list cannot be empty";
  TF_ASSIGN_OR_RETURN(
      absl::Span<const xla::ifrt::ShardingRef> output_shardings,
      executable_bundle.GetOutputShardings(device_list_->devices()[0]->Id()));

  TF_RET_CHECK(output_shardings.size() == stablehlo_outputs.size())
      << "Output shardings count (" << output_shardings.size()
      << ") does not match outputs count (" << stablehlo_outputs.size() << ")";
  TF_RET_CHECK(executable_bundle.io_spec.outputs.size() ==
               stablehlo_outputs.size())
      << "IO spec outputs count (" << executable_bundle.io_spec.outputs.size()
      << ") does not match outputs count (" << stablehlo_outputs.size() << ")";
  TF_RET_CHECK(executable_bundle.original_io_spec.outputs.size() ==
               stablehlo_outputs.size())
      << "Original IO spec outputs count ("
      << executable_bundle.original_io_spec.outputs.size()
      << ") does not match outputs count (" << stablehlo_outputs.size() << ")";

  for (int i = 0; i < stablehlo_outputs.size(); ++i) {
    const xla::ifrt::ArrayRef& array_for_copy = stablehlo_outputs[i];

    if (VLOG_IS_ON(2)) {
      TF_RETURN_IF_ERROR(ValidateOutputArray(
          *array_for_copy, executable_bundle.io_spec.outputs[i].dtype,
          executable_bundle.io_spec.outputs[i].GetIfrtShape(), i));
    }

    if (placements[i] == Placement::kHost) {
      // IFRT's return does not contain sufficient information; so we use
      // sharding spec from metadata.
      VLOG(2) << "Output sharding: " << array_for_copy->sharding();
      output_futures[i] = sharding_internal::MakeHostBufferFromArray(
          *ifrt_client_, *array_for_copy, output_shardings[i], device_list_,
          *thread_pool_);
    }
  }

  std::vector<xla_serving::Value> converted_outputs;
  converted_outputs.reserve(stablehlo_outputs.size());
  for (int i = 0; i < stablehlo_outputs.size(); ++i) {
    if (placements[i] == Placement::kHost) {
      TF_ASSIGN_OR_RETURN(auto host_buffer, output_futures[i]->Await());
      const xla::ifrt::Shape expected_ifrt_shape =
          executable_bundle.original_io_spec.outputs[i].GetIfrtShape();
      if (host_buffer.shape() != expected_ifrt_shape) {
        if (host_buffer.shape().num_elements() !=
            expected_ifrt_shape.num_elements()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Output shape mismatch. Expected: ", expected_ifrt_shape,
              ", but got: ", host_buffer.shape()));
        }
        HostBuffer reshaped_buffer(host_buffer.base(), host_buffer.dtype(),
                                   expected_ifrt_shape, host_buffer.owner());
        converted_outputs.push_back(xla_serving::Value(std::move(reshaped_buffer)));
      } else {
        converted_outputs.push_back(xla_serving::Value(std::move(host_buffer)));
      }
    } else {
      converted_outputs.push_back(
          xla_serving::Value(xla_serving::DeviceBuffer{.ifrt_client = ifrt_client_,
                                       .ifrt_array = stablehlo_outputs[i],
                                       .sharding = output_shardings[i],
                                       .device_list = device_list_}));
    }
  }
  return converted_outputs;
}

StableHloRunner::StableHloRunner(
    absl_nonnull std::unique_ptr<LoadedStableHlo> loaded_stablehlo,
    std::optional<xla::CompileOptionsProto> compile_options,
    absl_nonnull std::shared_ptr<xla::ifrt::Client> client,
    IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry,
    std::vector<std::string> weights,
    xla::ifrt::DeviceListRef assigned_device_list,
    tsl::protobuf::Message* compilation_environment_proto,
    IfrtPersistentCompilationCache* absl_nullable persistent_compilation_cache,
    StablehloOptimizer* absl_nullable optimizer,
    absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices,
    ServingCoreSelector* absl_nullable serving_core_selector,
    std::unique_ptr<H2DTransferExecutorFactory> h2d_transfer_executor_factory,
    std::optional<int64_t> callback_id)
    : loaded_stablehlo_(std::move(loaded_stablehlo)),
      compile_options_(std::move(compile_options)),
      assigned_device_list_(std::move(assigned_device_list)),
      ifrt_client_(std::move(client)),
      persistent_compilation_cache_(persistent_compilation_cache),
      ifrt_loaded_variable_registry_(*ifrt_loaded_variable_registry),
      weights_(std::move(weights)),
      compilation_environment_proto_(compilation_environment_proto),
      optimizer_(optimizer),
      assigned_devices_(std::move(assigned_devices)),
      serving_core_selector_(serving_core_selector),
      h2d_transfer_executor_factory_(std::move(h2d_transfer_executor_factory)) {
  if (loaded_stablehlo_->has_streaming()) {
    DCHECK(callback_id.has_value());
    streaming_host_callback_id_ = callback_id;
  }
}

absl::StatusOr<std::unique_ptr<StableHloRunner>> StableHloRunner::Create(
    absl_nonnull std::shared_ptr<StableHlo> stablehlo,
    std::optional<xla::CompileOptionsProto> compile_options,
    absl_nonnull std::shared_ptr<xla::ifrt::Client> client,
    IfrtPersistentCompilationCache* absl_nullable persistent_compilation_cache,
    IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry,
    std::vector<std::string> weights,
    tsl::protobuf::Message* absl_nullable compilation_environment_proto,
    tsl::thread::ThreadPool* absl_nonnull thread_pool,
    StablehloOptimizer* absl_nullable optimizer,
    absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices,
    ServingCoreSelector* absl_nullable serving_core_selector,
    std::unique_ptr<H2DTransferExecutorFactory> h2d_transfer_executor_factory,
    std::optional<int64_t> callback_id) {
  if (!stablehlo) {
    return absl::InvalidArgumentError(
        "`StableHlo` should be non-null when creating `StableHloRunner`");
  }
  if (client == nullptr) {
    return absl::InvalidArgumentError(
        "`ifrt::Client` client should be non-null when creating "
        "`StableHloRunner`");
  }
  if (ifrt_loaded_variable_registry == nullptr) {
    return absl::InvalidArgumentError(
        "`IfrtLoadedVariableRegistry` should be non-null when creating "
        "`StableHloRunner`");
  }
  if (thread_pool == nullptr) {
    return absl::InvalidArgumentError(
        "Thread pool is null when creating `StableHloRunner`");
  }
  if (assigned_devices == nullptr) {
    return absl::InvalidArgumentError(
        "Assigned devices is null when creating `StableHloRunner`");
  }

  // Load StableHlo.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LoadedStableHlo> loaded_stablehlo,
      LoadedStableHlo::Create(std::move(stablehlo), weights.size(),
                              client->platform_name(), compile_options));

  if (loaded_stablehlo->has_streaming() && !callback_id.has_value()) {
    return absl::InvalidArgumentError(
        "Callback ID must be provided when creating StableHloRunner for a "
        "streaming model.");
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::Device*> assigned_ifrt_devices,
      GetAssignedIfrtDevices(*client, assigned_devices->num_replicas,
                             assigned_devices->num_cores_per_replica,
                             assigned_devices->device_assignment));
  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef assigned_device_list,
                      client->MakeDeviceList(assigned_ifrt_devices));

  auto runner = absl::WrapUnique(new StableHloRunner(
      std::move(loaded_stablehlo), std::move(compile_options),
      std::move(client), ifrt_loaded_variable_registry, std::move(weights),
      assigned_device_list, compilation_environment_proto,
      persistent_compilation_cache, optimizer, assigned_devices,
      serving_core_selector, std::move(h2d_transfer_executor_factory),
      callback_id));
  TF_RETURN_IF_ERROR(runner->Init(thread_pool));
  return runner;
}

absl::Status StableHloRunner::Init(
    tsl::thread::ThreadPool* absl_nonnull thread_pool) {
  if (IsPortableExecutable()) {
    // Create resources for all devices.
    for (size_t i = 0; i < ifrt_client_->addressable_device_count(); ++i) {
      TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                          ifrt_client_->LookupDevice(xla::ifrt::DeviceId(i)));
      TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                          ifrt_client_->MakeDeviceList({device}));
      TF_RETURN_IF_ERROR(
          CreateResourcesPerDeviceList(device_list, thread_pool));
    }
  } else {
    TF_RETURN_IF_ERROR(
        CreateResourcesPerDeviceList(assigned_device_list_, thread_pool));
  }
  return absl::OkStatus();
}

absl::StatusOr<StableHloRunner::SharedCachedExecutableBundle>
StableHloRunner::CreateExecutableSynchronously(
    absl::Span<const xla::ifrt::Shape> input_shapes,
    absl::Span<const xla::ifrt::DType> input_dtypes,
    std::unique_ptr<LoadedStableHlo::ModuleBundle> module_bundle,
    const StablehloOptimizer::Args& args, bool is_streaming) {
  // Refine the dynamic shapes in the module.
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      RefineShapesIfNeeded(
          module_bundle->module.get(), loaded_stablehlo_->weight_specs(),
          input_shapes, input_dtypes, module_bundle->module_kept_var_indices),
      "Failed to refine shapes.");
  TF_RETURN_IF_ERROR(xla::ValidateStaticShapes(module_bundle->module.get()));

  // Annotate parameters tied to constant weights with an attribute.
  // All parameters before input_offset are considered constant weights.
  AnnotateConstantWeights(*module_bundle->module, module_bundle->input_offset);

  if (VLOG_IS_ON(1)) {
    DumpMlirOpToFile("stablehlo_module_after_refinement",
                     module_bundle->module.get());
  }

  if (is_streaming) {
    // Differentiate if multi-request batching is used
    // (!run_context.request_ids.empty()). When batching is active, a 1D shape
    // {batch_size} (dims().size() == 1) was appended to input_shapes for taking
    // step_id of different requests as inputs; for unbatched single request, a
    // 0D scalar shape {} (dims().size() == 0) was appended. This is only valid
    // for streaming models.
    bool is_batched_in_streaming =
        !input_shapes.empty() && input_shapes.back().dims().size() == 1;
    TF_RETURN_IF_ERROR(PropagateStreamingStepId(module_bundle->module.get(),
                                                is_batched_in_streaming));

    if (VLOG_IS_ON(1)) {
      DumpMlirOpToFile("stablehlo_module_after_streaming_step_id_propagation",
                       module_bundle->module.get());
    }
  }

  // If we have a streaming_host_callback_id_, try to set it in the module.
  // SetStreamCallbackId will return std::nullopt if the module doesn't actual
  // contain any streaming custom calls (even if the runner thought it might).
  // We use this returned callback_id to decide whether to treat this specific
  // compiled executable as streaming (e.g. adding extra request_id input).
  std::optional<int64_t> callback_id;
  if (streaming_host_callback_id_.has_value()) {
    TF_ASSIGN_OR_RETURN(callback_id,
                        SetStreamCallbackId(module_bundle->module.get(),
                                            *streaming_host_callback_id_));
  }

  TF_ASSIGN_OR_RETURN(
      StableHloIOSpec original_io_spec,
      GetStableHloIOSpec(*module_bundle->module, compile_options_.has_value()
                                                     ? &compile_options_.value()
                                                     : nullptr));

  if (optimizer_ != nullptr) {
    TF_RETURN_IF_ERROR(optimizer_->Optimize(args, module_bundle->module.get()));
  }

  if (VLOG_IS_ON(1)) {
    DumpMlirOpToFile("stablehlo_module_after_optimization",
                     module_bundle->module.get());
  }

  TF_ASSIGN_OR_RETURN(
      StableHloIOSpec io_spec,
      GetStableHloIOSpec(*module_bundle->module, compile_options_.has_value()
                                                     ? &compile_options_.value()
                                                     : nullptr));

  // Output placement metadata is propagated from the loaded StableHLO specs so
  // that output conversion (ConvertOutput) can honor manifest-declared
  // placements (Device vs. Host). Input placement does not need to be set here
  // because inputs carry their physical placement intrinsically via their
  // runtime type (DeviceBuffer vs. HostBuffer) during execution.
  for (size_t i = 0; i < io_spec.outputs.size(); ++i) {
    if (i < loaded_stablehlo_->output_specs().size()) {
      io_spec.outputs[i].placement =
          loaded_stablehlo_->output_specs()[i].placement;
      original_io_spec.outputs[i].placement =
          loaded_stablehlo_->output_specs()[i].placement;
    }
  }

  BuildIfrtExecutableOptions options;

  if (compile_options_.has_value()) {
    options.compile_options = &*compile_options_;
  }
  // TODO: Remove these overrides once the compile options are
  // fully supported.
  options.num_replicas = assigned_devices_->num_replicas;
  options.num_cores_per_replica = assigned_devices_->num_cores_per_replica;
  if (IsPortableExecutable()) {
    options.compile_portable_executable = true;
  } else {
    TF_ASSIGN_OR_RETURN(
        xla::DeviceAssignment da,
        GetRuntimeXlaDeviceAssignment(
            assigned_device_list_, assigned_devices_->num_replicas,
            assigned_devices_->num_cores_per_replica));
    VLOG(2) << "Device assignment :" << da.ToString();
    options.device_assignment = std::move(da);
  }

  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutableRef ifrt_executable,
      BuildIfrtExecutable(module_bundle->module.get(), options, ifrt_client_,
                          persistent_compilation_cache_));

  SharedCachedExecutableBundle executable_bundle =
      std::make_shared<CachedExecutableBundle>();
  executable_bundle->module_kept_var_indices =
      module_bundle->module_kept_var_indices;
  executable_bundle->input_offset = module_bundle->input_offset;
  executable_bundle->original_io_spec = original_io_spec;
  executable_bundle->io_spec = io_spec;

  executable_bundle->ifrt_executable = std::move(ifrt_executable);
  executable_bundle->program_fingerprint_with_timestamp =
      GetProgramFingerprintWithTimestamp(executable_bundle->ifrt_executable,
                                         absl::Now());

  executable_bundle->callback_id = callback_id;

  // With streaming, there will be an addtional input for the request id.
  if (callback_id.has_value()) {
    executable_bundle->module_kept_var_indices.push_back(io_spec.inputs.size() -
                                                         1);
  }

  return executable_bundle;
}

tsl::Future<StableHloRunner::SharedCachedExecutableBundle>
StableHloRunner::LookUpOrCreateExecutable(
    absl::Span<const xla::ifrt::Shape> input_shapes,
    absl::Span<const xla::ifrt::DType> input_dtypes) {
  tsl::Promise<SharedCachedExecutableBundle> promise;
  tsl::Future<SharedCachedExecutableBundle> future;
  std::unique_ptr<LoadedStableHlo::ModuleBundle> module_bundle;

  {
    absl::MutexLock lock(mutex_);

    // Cache hit.
    const auto it = executable_bundles_.find(KeyView(input_shapes));
    if (ABSL_PREDICT_TRUE(it != executable_bundles_.end())) {
      return it->second;
    }

    // Cache miss.

    std::tie(promise, future) =
        tsl::MakePromise<SharedCachedExecutableBundle>();
    if (fully_initialized_) {
      promise.Set(absl::FailedPreconditionError(
          "Got new input shapes requiring a new executable, but "
          "can't create it after full initialization. Please warm "
          "up the model with all the input shapes that are possible during "
          "serving. (See go/tf-model-warmup for more details.)"));
      return future;
    }

    executable_bundles_.emplace(Key(input_shapes), future);

    // Look up the MLIR module bundle for the given input shapes, since
    // there might be multiple MLIR modules with different input shapes as
    // the result of concretizing a polymorphic StableHLO to a set of concrete
    // StableHLOs.
    absl::StatusOr<std::unique_ptr<LoadedStableHlo::ModuleBundle>> bundle =
        loaded_stablehlo_->CloneModuleBundle(input_shapes);
    if (!bundle.ok()) {
      promise.Set(bundle.status());
      return future;
    }
    module_bundle = std::move(*bundle);
  }

  LOG(INFO) << "Cache missed. Building executable";
  if (VLOG_IS_ON(1)) {
    DumpMlirOpToFile("stablehlo_module_before_compilation",
                     module_bundle->module.get());
  }

  StablehloOptimizer::Args args;
  args.variable_indices.resize(module_bundle->input_offset);
  std::iota(args.variable_indices.begin(), args.variable_indices.end(), 0);
  if (compile_options_.has_value()) {
    args.compile_options = &*compile_options_;
  }
  absl::StatusOr<SharedCachedExecutableBundle> executable_bundle =
      CreateExecutableSynchronously(input_shapes, input_dtypes,
                                    std::move(module_bundle), args,
                                    has_streaming());
  if (!executable_bundle.ok()) {
    LOG(ERROR) << "Failed to create executable: "
               << executable_bundle.status().message();
  } else {
    if (IsPortableExecutable()) {
      CachedExecutableBundle::PortableShardingMap arg_map;
      CachedExecutableBundle::PortableShardingMap output_map;
      for (xla::ifrt::Device* device : ifrt_client_->addressable_devices()) {
        xla::ifrt::ShardingRef sharding =
            xla::ifrt::SingleDeviceSharding::Create(device,
                                                    xla::ifrt::MemoryKind());
        arg_map[device->Id()] = std::vector<xla::ifrt::ShardingRef>(
            (*executable_bundle)->io_spec.inputs.size(), sharding);
        output_map[device->Id()] = std::vector<xla::ifrt::ShardingRef>(
            (*executable_bundle)->io_spec.outputs.size(), sharding);
      }
      (*executable_bundle)->arg_shardings = std::move(arg_map);
      (*executable_bundle)->output_shardings = std::move(output_map);
    } else {
      CachedExecutableBundle::ShardingList arg_list;
      arg_list.reserve((*executable_bundle)->io_spec.inputs.size());
      for (const auto& arg_spec : (*executable_bundle)->io_spec.inputs) {
        TF_ASSIGN_OR_RETURN(
            auto ifrt_sharding,
            sharding_internal::ToIfrtSharding(*ifrt_client_, arg_spec.sharding,
                                              assigned_device_list_));
        arg_list.push_back(std::move(ifrt_sharding));
      }
      (*executable_bundle)->arg_shardings = std::move(arg_list);

      CachedExecutableBundle::ShardingList output_list;
      output_list.reserve((*executable_bundle)->io_spec.outputs.size());
      for (const auto& out_spec : (*executable_bundle)->io_spec.outputs) {
        TF_ASSIGN_OR_RETURN(
            auto ifrt_sharding,
            sharding_internal::ToIfrtSharding(*ifrt_client_, out_spec.sharding,
                                              assigned_device_list_));
        output_list.push_back(std::move(ifrt_sharding));
      }
      (*executable_bundle)->output_shardings = std::move(output_list);
    }
  }

  promise.Set(std::move(executable_bundle));
  return future;
}

bool StableHloRunner::IsPortableExecutable() const {
  return serving_core_selector_ != nullptr &&
         assigned_devices_->num_replicas == 1 &&
         assigned_devices_->num_cores_per_replica == 1;
}

absl::StatusOr<std::vector<std::optional<xla::ifrt::ArrayRef>>>
StableHloRunner::LookupWeights(std::optional<size_t> device_id) {
  std::vector<int> device_ids;
  if (device_id.has_value()) {
    device_ids.push_back(*device_id);
  } else {
    device_ids.reserve(assigned_device_list_->size());
    for (const auto& device : assigned_device_list_->devices()) {
      device_ids.push_back(device->Id().value());
    }
  }

  const int num_weights = weights_.size();

  // Ifrt Arrays for the weights. When an element is set to std::nullopt, it
  // means the weight is unused (i.e. does not needed to be passed to the
  // StableHLO for execution).
  std::vector<std::optional<xla::ifrt::ArrayRef>> arrays(num_weights,
                                                         std::nullopt);

  // Get weight specs. The weight spec will be nullopt if the weight is unused
  // by the StableHLO.
  const std::vector<std::optional<ArgSpec>>& weights_specs =
      loaded_stablehlo_->weight_specs();
  if (weights_specs.size() != num_weights) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected ", weights_.size(), " weight specs, but got ",
                     weights_specs.size(), " weight specs."));
  }

  // Build IFRT arrays for the weights. The array will be nullopt if the weight
  // is unused.
  for (int i = 0; i < num_weights; i++) {
    if (!weights_specs[i].has_value()) {
      // Unused weights. They should never be needed when executing StableHLO.
      continue;
    }

    const ArgSpec& weight_spec = *weights_specs[i];
    const auto& weight_name = weights_[i];
    std::optional<xla::Layout> layout = std::nullopt;
    if (weight_spec.layout.has_value()) {
      layout = *weight_spec.layout;
    }
    IfrtLoadedVariableRegistry::Key key{
        .device_ids = device_ids,
        .input_name = weight_name,
        .hlo_sharding = device_id.has_value()
                            ? xla::HloSharding::SingleDevice(*device_id)
                            : weight_spec.sharding,
        .layout = layout,
    };
    TF_ASSIGN_OR_RETURN(auto loaded_weight,
                        ifrt_loaded_variable_registry_.GetLoadedVariable(key));
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::ArrayRef loaded_weight_ifrt_array,
        loaded_weight.value.Get<tsl::Future<xla::ifrt::ArrayRef>>().Await());
    // Check the shape and DType of loaded weight matches the weight spec.
    // Make sure the shape and dtype of IFRT array for the weight matches the
    // the spec of the weight.
    if (loaded_weight_ifrt_array->shape() != ToIfrtShape(weight_spec.shape)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "StableHlo expected weight #", i, " (", weights_[i],
          ")"
          " to have shape: ",
          ToIfrtShape(weight_spec.shape), ", but got ",
          loaded_weight_ifrt_array->shape(), " from the actual weight."));
    }
    if (loaded_weight_ifrt_array->dtype() != weight_spec.dtype) {
      return absl::FailedPreconditionError(absl::StrCat(
          "StableHlo expected weight #", i, " '", weights_[i],
          "' to have dtype: ", weight_spec.dtype, ", but got ",
          loaded_weight_ifrt_array->dtype(), " from the actual weight."));
    }
    TF_RETURN_IF_ERROR(
        InputShapeEqual(weight_spec.shape, loaded_weight_ifrt_array->shape()));

    arrays[i] = std::move(loaded_weight_ifrt_array);
  }
  return arrays;
}

absl::Status StableHloRunner::SetNumExecutionStreams(
    int num_execution_streams) {
  if (num_execution_streams < 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_execution_streams must be >= 1, got ",
                     num_execution_streams));
  }
  num_execution_streams_ = num_execution_streams;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<xla_serving::Value>> StableHloRunner::Execute(
    absl::Span<const xla_serving::Value> inputs, ExecuteOptions options,
    const RunContext& run_context) {
  bool has_streaming = loaded_stablehlo_->has_streaming();
  int64_t expected_num_inputs = loaded_stablehlo_->num_user_inputs();
  const uint32_t launch_seq = launch_id_.fetch_add(1);
  int32_t xla_launch_id = static_cast<int32_t>(launch_seq);
  // Round-robin executions across streams. IFRT orders executions only within
  // a stream, so batches on different streams can execute concurrently. The
  // ID is surfaced in the trace so concurrency can be verified from a profile.
  // Modulo on the unsigned counter keeps the ID non-negative after wraparound.
  const int64_t execution_stream_id = static_cast<int64_t>(
      launch_seq % static_cast<uint32_t>(num_execution_streams_));
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(
        "StableHloRunner::Execute",
        {{"execution_stream_id", execution_stream_id}});
  });

  VLOG(2) << "Start StableHloRunner execution on stream "
          << execution_stream_id;

  if (inputs.size() != expected_num_inputs &&
      (!has_streaming || inputs.size() != expected_num_inputs + 1)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected ", expected_num_inputs, " user inputs, but got ",
                     inputs.size()));
  }

  // TODO: Consider moving this to `Init()`. Maybe also remove the
  // `call_once` then.
  absl::call_once(weights_lookup_once_, [&]() {
    if (IsPortableExecutable()) {
      for (auto& [device_list, resources] : device_list_to_resources_map_) {
        resources->weight_arrays =
            LookupWeights(device_list->devices()[0]->Id().value());
      }
    } else {
      device_list_to_resources_map_[assigned_device_list_]->weight_arrays =
          LookupWeights();
    }
  });

  std::vector<xla::ifrt::Shape> input_shapes;
  input_shapes.reserve(inputs.size() + 1);
  std::vector<xla::ifrt::DType> input_dtypes;
  input_dtypes.reserve(inputs.size() + 1);

  // Gather shapes before appending streaming IDs to match cache signature.
  for (const auto& input : inputs) {
    if (input.IsType<HostBuffer>()) {
      const auto& host_buf = input.Get<HostBuffer>();
      input_shapes.push_back(host_buf.shape());
      input_dtypes.push_back(host_buf.dtype());
    } else if (input.IsType<xla_serving::DeviceBuffer>()) {
      const auto& device_buf = input.Get<xla_serving::DeviceBuffer>();
      input_shapes.push_back(device_buf.ifrt_array->shape());
      input_dtypes.push_back(device_buf.ifrt_array->dtype());
    } else {
      return absl::InvalidArgumentError(
          "StableHloRunner inputs must be HostBuffer or xla_serving::DeviceBuffer.");
    }
  }
  // If the model is streaming and the caller did not provide the request ID
  // input, we need to append the shape and dtype of the request ID input
  // for the cache lookup.
  if (has_streaming && inputs.size() == expected_num_inputs) {
    if (!run_context.request_ids.empty()) {
      input_shapes.push_back(xla::ifrt::Shape(
          {static_cast<int64_t>(run_context.request_ids.size())}));
    } else {
      input_shapes.push_back(xla::ifrt::Shape({}));
    }
    input_dtypes.push_back(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32));
  }

  TF_ASSIGN_OR_RETURN(
      SharedCachedExecutableBundle executable_bundle,
      LookUpOrCreateExecutable(input_shapes, input_dtypes).Await());

  // Find the resources per the device(s) to use. If the executable is portable,
  // use the core selector to reserve a device first.
  tsl::DeviceReservation device_reservation(/*device_index=*/-1,
                                            /*selector=*/nullptr);
  xla::ifrt::DeviceListRef device_list = assigned_device_list_;
  if (IsPortableExecutable()) {
    device_reservation = serving_core_selector_->ReserveDevice(
        executable_bundle->program_fingerprint_with_timestamp);
    const int device_id = device_reservation.device_index();
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::Device * device,
        ifrt_client_->LookupDevice(xla::ifrt::DeviceId(device_id)));
    TF_ASSIGN_OR_RETURN(device_list, ifrt_client_->MakeDeviceList({device}));
  }
  const auto resources_it = device_list_to_resources_map_.find(device_list);
  if (resources_it == device_list_to_resources_map_.end()) {
    return absl::InternalError(
        absl::StrCat("Device list resources not initialized for device list: ",
                     *device_list));
  }
  TF_ASSIGN_OR_RETURN(const auto& weights_array,
                      resources_it->second->weight_arrays);
  DataConverter* data_converter = resources_it->second->data_converter.get();

  std::vector<xla_serving::Value> unpacked_inputs(inputs.begin(), inputs.end());

  // `used_input_indices` is used to track the index of used inputs in
  // stablehlo spec because some inputs may be presented in the model signature
  // but not used by StableHlo.
  std::vector<int> used_input_indices;
  used_input_indices.reserve(executable_bundle->module_kept_var_indices.size());
  for (int idx : executable_bundle->module_kept_var_indices) {
    if (idx >= weights_array.size()) {
      used_input_indices.push_back(idx - weights_array.size());
    }
  }

  std::vector<HostBuffer> host_buffers_to_convert;
  std::vector<int> host_buffers_stablehlo_indices;
  host_buffers_to_convert.reserve(used_input_indices.size());
  host_buffers_stablehlo_indices.reserve(used_input_indices.size());

  std::vector<std::optional<xla::ifrt::ArrayRef>> final_used_inputs(
      used_input_indices.size(), std::nullopt);

  for (int i = 0; i < used_input_indices.size(); ++i) {
    int idx_in_stablehlo_spec = i + executable_bundle->input_offset;
    int original_input_idx = used_input_indices[i];
    auto& original_input_value = unpacked_inputs[original_input_idx];
    if (original_input_value.IsType<xla_serving::DeviceBuffer>()) {
      auto& device_buf = original_input_value.Get<xla_serving::DeviceBuffer>();
      xla::ifrt::ArrayRef original_input = std::move(device_buf.ifrt_array);

      TF_RETURN_IF_ERROR(ValidateDeviceBufferInput(
          *original_input,
          executable_bundle->io_spec.inputs[idx_in_stablehlo_spec].dtype,
          executable_bundle->io_spec.inputs[idx_in_stablehlo_spec]
              .GetIfrtShape(),
          idx_in_stablehlo_spec));
      final_used_inputs[i] = std::move(original_input);
    } else if (original_input_value.IsType<HostBuffer>()) {
      auto& original_input = original_input_value.Get<HostBuffer>();

      const xla::ifrt::Shape target_shape =
          executable_bundle->io_spec.inputs[idx_in_stablehlo_spec]
              .GetIfrtShape();

      if (original_input.shape() == target_shape) {
        // Shapes match, use the buffer directly.
        host_buffers_to_convert.push_back(std::move(original_input));
      } else {
        // Shapes differ (e.g., due to optimization). Perform a zero-copy
        // reshape to match the model's expected signature.
        HostBuffer reshaped_buffer(original_input.base(),
                                   original_input.dtype(), target_shape,
                                   original_input.owner());
        host_buffers_to_convert.push_back(std::move(reshaped_buffer));
      }
      host_buffers_stablehlo_indices.push_back(idx_in_stablehlo_spec);
    }
  }

  std::vector<xla::ifrt::ArrayRef> converted_inputs;
  if (!host_buffers_to_convert.empty()) {
    TF_ASSIGN_OR_RETURN(converted_inputs,
                        data_converter->ConvertInput(
                            host_buffers_to_convert, *executable_bundle,
                            host_buffers_stablehlo_indices));
  }

  int host_buf_idx = 0;
  for (int i = 0; i < used_input_indices.size(); ++i) {
    if (!final_used_inputs[i].has_value()) {
      final_used_inputs[i] = std::move(converted_inputs[host_buf_idx++]);
    }
  }

  std::vector<xla::ifrt::ArrayRef> inputs_with_weights;
  inputs_with_weights.reserve(
      executable_bundle->module_kept_var_indices.size());
  for (int idx : executable_bundle->module_kept_var_indices) {
    if (idx < weights_array.size()) {
      const auto& weight_array = weights_array[idx];
      if (!weight_array.has_value()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Weight #", idx,
            " is required by the StableHLO program but was not loaded. This "
            "usually indicates the weight was incorrectly treated as unused, "
            "thus not being loaded."));
      }
      inputs_with_weights.push_back(*weight_array);
    }
  }
  // `final_used_inputs` only contains used inputs, so we just insert them
  // at the end of the list.
  for (auto& arr : final_used_inputs) {
    inputs_with_weights.push_back(std::move(*arr));
  }

  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutable::ExecuteResult execution_result,
      executable_bundle->ifrt_executable->Execute(
          absl::MakeSpan(inputs_with_weights),
          /*options=*/
          {
              .launch_id = xla_launch_id,
              .fill_status = true,
              .execution_stream_id = execution_stream_id,
          },
          // IFRT requires an explicit device mapping when executing portable
          //  or unbound executables. If the executable already has
          // baked-in devices (bound), we pass std::nullopt to defer to the
          // binary's ground truth.
          /*devices=*/
          !executable_bundle->ifrt_executable->devices().has_value()
              ? std::make_optional(data_converter->device_list())
              : std::nullopt));
  TF_RETURN_IF_ERROR(execution_result.status.Await());

  std::vector<Placement> placements;
  // Branch 1: Explicit runtime placement overrides.
  // When the caller explicitly supplies `options.placements`, it takes
  // precedence over model manifest declarations.
  if (!options.placements.empty()) {
    if (options.placements.size() != execution_result.outputs.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected ", execution_result.outputs.size(),
                       " placements, but got ", options.placements.size()));
    }
    placements = std::move(options.placements);
  } else {
    // Branch 2: Manifest-declared placement with default host fallback.
    // When runtime placement overrides are absent:
    // 1. Defer to the model manifest placement if explicitly annotated in
    // `io_spec.outputs[i]`.
    // 2. Otherwise, fall back to Placement::kHost (D2H transfer).
    placements.reserve(execution_result.outputs.size());
    for (size_t i = 0; i < execution_result.outputs.size(); ++i) {
      if (i < executable_bundle->io_spec.outputs.size() &&
          executable_bundle->io_spec.outputs[i].placement.has_value()) {
        placements.push_back(*executable_bundle->io_spec.outputs[i].placement);
      } else {
        placements.push_back(Placement::kHost);
      }
    }
  }

  return data_converter->ConvertOutput(absl::MakeSpan(execution_result.outputs),
                                       *executable_bundle, placements);
}

absl::Status StableHloRunner::MarkFullyInitialized() {
  absl::MutexLock lock(mutex_);
  fully_initialized_ = true;
  // Release the module as it is no longer needed after fully initialized
  // because no compilation will be allowed. A module can be large, e.g.
  // containing a large const.
  loaded_stablehlo_->ReleaseAllModules();
  return absl::OkStatus();
}

absl::Status StableHloRunner::CreateResourcesPerDeviceList(
    xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* thread_pool) {
  std::unique_ptr<H2DTransferExecutor> h2d_transfer_executor;
  if (h2d_transfer_executor_factory_ != nullptr) {
    TF_ASSIGN_OR_RETURN(h2d_transfer_executor,
                        h2d_transfer_executor_factory_->Create(
                            ifrt_client_.get(), device_list, thread_pool));
  }
  device_list_to_resources_map_[device_list] =
      std::make_unique<ResourcesPerDeviceList>(
          ifrt_client_, device_list, thread_pool,
          std::move(h2d_transfer_executor));
  return absl::OkStatus();
}

}  // namespace xla_serving
