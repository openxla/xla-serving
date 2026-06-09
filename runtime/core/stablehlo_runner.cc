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

#include "runtime/core/stablehlo_runner.h"  // from @neptune

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"  // from @abseil-cpp
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/optimization.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
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
#include "runtime/core/device_buffer.h"  // from @neptune
#include "runtime/core/h2d_transfer_executor.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/ifrt_device_utils.h"  // from @neptune
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @neptune
#include "runtime/core/loaded_stablehlo.h"  // from @neptune
#include "runtime/core/sharding_utils.h"  // from @neptune
#include "runtime/core/stablehlo.h"  // from @neptune
#include "runtime/core/stablehlo_optimizer.h"  // from @neptune
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "runtime/core/streaming/stream.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/core/value_utils.h"  // from @neptune
#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "runtime/libs/ifrt/executable_build_utils.h"  // from @neptune
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @neptune
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @neptune
#include "runtime/libs/mlir/dump_mlir_util.h"  // from @neptune
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
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/python/refine_polymorphic_shapes.h"  // from @xla
#include "xla/service/computation_placer.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/framework/serving_device_selector.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace neptune {
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

absl::StatusOr<xla::ifrt::ShardingRef> ToIfrtSharding(
    xla::ifrt::Client& ifrt_client, const xla::HloSharding& hlo_sharding,
    const xla::ifrt::DeviceListRef& device_list) {
  if (!hlo_sharding.IsTiled() && !hlo_sharding.IsReplicatedOrSingleDevice()) {
    return absl::UnimplementedError(absl::StrCat(
        "Only support MAXIMAL, OTHER or REPLICATED, but got sharding : ",
        hlo_sharding.ToString()));
  }

  if (device_list->size() == 1) {
    return xla::ifrt::SingleDeviceSharding::Create(
        device_list->devices().front(), xla::ifrt::MemoryKind());
  }

  if (hlo_sharding.IsSingleDevice()) {
    VLOG(1) << "Single device fast path for Maximal tiled tensor";
    xla::ifrt::Device* device;
    int unique_device_id = hlo_sharding.GetUniqueDevice();
    TF_ASSIGN_OR_RETURN(device, ifrt_client.LookupDevice(
                                    xla::ifrt::DeviceId(unique_device_id)));
    return xla::ifrt::SingleDeviceSharding::Create(device,
                                                   xla::ifrt::MemoryKind());
  }

  return xla::ifrt::HloSharding::Create(device_list, xla::ifrt::MemoryKind(),
                                        hlo_sharding);
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
    const CachedExecutableBundle& executable_bundle, int input_offset) const {
  if (input_offset + inputs.size() > executable_bundle.io_spec.inputs.size()) {
    return absl::InvalidArgumentError("Too many inputs provided.");
  }

  // Validate data types and shapes before passing to executor to prevent
  // buffer over-reads during memcpy.
  for (int i = 0; i < inputs.size(); ++i) {
    const ArgSpec& input_spec =
        executable_bundle.io_spec.inputs[input_offset + i];
    if (inputs[i].dtype() != input_spec.dtype) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input dtype mismatch. Expected: ", input_spec.dtype,
                       ", but got: ", inputs[i].dtype()));
    }
    TF_RETURN_IF_ERROR(InputShapeEqual(input_spec.shape, inputs[i].shape()));
  }

  if (!executable_bundle.portable_single_device_shardings.empty()) {
    DCHECK(!device_list_->devices().empty());
    auto it = executable_bundle.portable_single_device_shardings.find(
        device_list_->devices()[0]->Id());
    if (it == executable_bundle.portable_single_device_shardings.end()) {
      return absl::FailedPreconditionError(
          absl::StrCat("No portable single device sharding found for device ",
                       device_list_->devices()[0]->Id()));
    }
    return absl::MakeConstSpan(it->second).subspan(input_offset, inputs.size());
  }

  if (input_offset + inputs.size() >
      executable_bundle.arg_ifrt_shardings.size()) {
    return absl::InternalError("IFRT Executable arg shardings size mismatch.");
  }
  return absl::MakeConstSpan(executable_bundle.arg_ifrt_shardings)
      .subspan(input_offset, inputs.size());
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
    const CachedExecutableBundle& executable_bundle, int input_offset) const {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(h2d_transfer_executor_ != nullptr
                                            ? "ArenaTransferFromHostToDevice"
                                            : "TransferFromHostToDevice",
                                        {});
  });

  TF_ASSIGN_OR_RETURN(
      absl::Span<const xla::ifrt::ShardingRef> shardings_span,
      ValidateAndResolveShardings(inputs, executable_bundle, input_offset));

  if (h2d_transfer_executor_ != nullptr) {
    arena_cell_->IncrementBy(1);
    return h2d_transfer_executor_->ExecuteTransfer(inputs, shardings_span);
  }

  default_cell_->IncrementBy(1);
  return ExecuteDefaultTransfer(inputs, shardings_span);
}

absl::StatusOr<std::vector<HostBuffer>>
StableHloRunner::DataConverter::ConvertOutput(
    absl::Span<const xla::ifrt::ArrayRef> stablehlo_outputs,
    const StableHloIOSpec& io_spec) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode("TransferFromDeviceToHost", {});
  });
  std::vector<tsl::Future<HostBuffer>> output_futures;
  output_futures.reserve(stablehlo_outputs.size());

  for (int i = 0; i < stablehlo_outputs.size(); ++i) {
    const xla::ifrt::ArrayRef& array_for_copy = stablehlo_outputs[i];
    // IFRT's return does not contain sufficient information; so we use
    // sharding spec from metadata.
    VLOG(2) << "Output sharding: " << array_for_copy->sharding();
    output_futures.push_back(sharding_internal::MakeHostBufferFromArray(
        *ifrt_client_, *array_for_copy, io_spec.outputs[i].sharding,
        device_list_, *thread_pool_));
  }

  std::vector<HostBuffer> converted_outputs;
  converted_outputs.reserve(stablehlo_outputs.size());
  for (tsl::Future<HostBuffer>& output_future : output_futures) {
    TF_ASSIGN_OR_RETURN(auto host_buffer, output_future.Await());
    converted_outputs.push_back(std::move(host_buffer));
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
    std::function<void(absl::flat_hash_map<std::string, neptune::HostBuffer>)>
        streamed_output_callback)
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
      h2d_transfer_executor_factory_(std::move(h2d_transfer_executor_factory)),
      streamed_output_callback_(std::move(streamed_output_callback)) {}

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
    std::function<void(absl::flat_hash_map<std::string, neptune::HostBuffer>)>
        streamed_output_callback) {
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
      std::move(streamed_output_callback)));
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
    const StablehloOptimizer::Args& args) {
  // Refine the dynamic shapes in the module.
  // Use any device list's weight arrays, because the weight specs are the
  // same across all device lists.
  // TODO: Pass weight specs instead of weight arrays to
  // `RefineShapesIfNeeded()`.
  if (device_list_to_resources_map_.empty()) {
    return absl::FailedPreconditionError(
        "No device list resources initialized.");
  }
  TF_RETURN_IF_ERROR(
      device_list_to_resources_map_.begin()->second->weight_arrays.status());
  std::vector<std::optional<xla::ifrt::ArrayRef>>* weight_arrays =
      &device_list_to_resources_map_.begin()->second->weight_arrays.value();
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      RefineShapesIfNeeded(module_bundle->module.get(), *weight_arrays,
                           input_shapes, input_dtypes,
                           module_bundle->module_kept_var_idx),
      "Failed to refine shapes.");
  TF_RETURN_IF_ERROR(xla::ValidateStaticShapes(module_bundle->module.get()));

  TF_ASSIGN_OR_RETURN(
      StableHloIOSpec io_spec,
      GetStableHloIOSpec(*module_bundle->module, compile_options_.has_value()
                                                     ? &compile_options_.value()
                                                     : nullptr));

  // Annotate parameters tied to constant weights with an attribute.
  // All parameters before input_offset are considered constant weights.
  AnnotateConstantWeights(*module_bundle->module, module_bundle->input_offset);

  if (VLOG_IS_ON(1)) {
    DumpMlirOpToFile("stablehlo_module_after_refinement",
                     module_bundle->module.get());
  }

  TF_ASSIGN_OR_RETURN(
      std::optional<int64_t> callback_id,
      neptune::CreateStreamCallbackId(module_bundle->module.get()));

  if (VLOG_IS_ON(1)) {
    neptune::DumpMlirOpToFile(
        "stablehlo_module_after_populate_stream_host_callback",
        module_bundle->module.get());
  }

  if (optimizer_ != nullptr) {
    TF_RETURN_IF_ERROR(optimizer_->Optimize(args, module_bundle->module.get()));
  }

  if (VLOG_IS_ON(1)) {
    DumpMlirOpToFile("stablehlo_module_after_optimization",
                     module_bundle->module.get());
  }

  TF_ASSIGN_OR_RETURN(
      io_spec,
      GetStableHloIOSpec(*module_bundle->module, compile_options_.has_value()
                                                     ? &compile_options_.value()
                                                     : nullptr));

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
  executable_bundle->module_kept_var_idx =
      std::vector<int>(module_bundle->module_kept_var_idx.begin(),
                       module_bundle->module_kept_var_idx.end());
  executable_bundle->input_offset = module_bundle->input_offset;
  executable_bundle->io_spec = io_spec;

  executable_bundle->ifrt_executable = std::move(ifrt_executable);
  executable_bundle->program_fingerprint_with_timestamp =
      GetProgramFingerprintWithTimestamp(executable_bundle->ifrt_executable,
                                         absl::Now());

  executable_bundle->callback_id = callback_id;

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
                                    std::move(module_bundle), args);
  if (!executable_bundle.ok()) {
    LOG(ERROR) << "Failed to create executable: "
               << executable_bundle.status().message();
  } else {
    if (IsPortableExecutable()) {
      for (xla::ifrt::Device* device : ifrt_client_->addressable_devices()) {
        xla::ifrt::ShardingRef sharding =
            xla::ifrt::SingleDeviceSharding::Create(device,
                                                    xla::ifrt::MemoryKind());
        (*executable_bundle)->portable_single_device_shardings[device->Id()] =
            std::vector<xla::ifrt::ShardingRef>(
                (*executable_bundle)->io_spec.inputs.size(), sharding);
      }
    } else {
      (*executable_bundle)
          ->arg_ifrt_shardings.reserve(
              (*executable_bundle)->io_spec.inputs.size());
      for (const auto& arg_spec : (*executable_bundle)->io_spec.inputs) {
        TF_ASSIGN_OR_RETURN(auto ifrt_sharding,
                            ToIfrtSharding(*ifrt_client_, arg_spec.sharding,
                                           assigned_device_list_));
        (*executable_bundle)
            ->arg_ifrt_shardings.push_back(std::move(ifrt_sharding));
      }
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

absl::StatusOr<std::vector<neptune::Value>> StableHloRunner::Execute(
    absl::Span<const neptune::Value> inputs, ExecuteOptions options) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode("StableHloRunner::Execute", {});
  });

  VLOG(2) << "Start StableHloRunner execution";

  if (inputs.size() != loaded_stablehlo_->num_user_inputs()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected ", loaded_stablehlo_->num_user_inputs(),
                     " user inputs, but got ", inputs.size()));
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

  std::vector<HostBuffer> host_buffers;
  std::vector<neptune::DeviceBuffer> device_buffers;
  if (options.require_h2d) {
    TF_ASSIGN_OR_RETURN(host_buffers,
                        neptune::internal::UnpackFromValues<HostBuffer>(inputs));
  } else {
    TF_ASSIGN_OR_RETURN(
        device_buffers,
        neptune::internal::UnpackFromValues<neptune::DeviceBuffer>(inputs));
  }

  // Lookup the executable bundle in the cache.
  std::vector<xla::ifrt::Shape> input_shapes;
  std::vector<xla::ifrt::DType> input_dtypes;
  if (options.require_h2d) {
    input_shapes.reserve(host_buffers.size());
    input_dtypes.reserve(host_buffers.size());
    for (const auto& input : host_buffers) {
      input_shapes.push_back(input.shape());
      input_dtypes.push_back(input.dtype());
    }
  } else {
    input_shapes.reserve(device_buffers.size());
    input_dtypes.reserve(device_buffers.size());
    for (const auto& input : device_buffers) {
      input_shapes.push_back(input.ifrt_array->shape());
      input_dtypes.push_back(input.ifrt_array->dtype());
    }
  }
  TF_ASSIGN_OR_RETURN(
      SharedCachedExecutableBundle executable_bundle,
      LookUpOrCreateExecutable(input_shapes, input_dtypes).Await());

  int64_t current_request_id = launch_id_.fetch_add(1);

  // The runtime assumes the streaming implementation will be a sync call so
  // that once the Execute() returns, the streaming is done.
  std::optional<neptune::ScopedStreamCallback> scoped_stream_callback;
  if (executable_bundle->callback_id.has_value() &&
      streamed_output_callback_ != nullptr) {
    // The lifetime of `StableHloRunner` will outlive all the outstanding
    // streaming requests, so it is safe to capture `this` to access the
    // `streamed_output_callback_`.
    auto wrapped_callback =
        [this](absl::flat_hash_map<std::string, neptune::HostBuffer> host_buffers) {
          if (this->streamed_output_callback_) {
            this->streamed_output_callback_(std::move(host_buffers));
          }
        };

    // TODO: Propagate the model name to the stream callback
    // registry for recording latency metrics.
    TF_ASSIGN_OR_RETURN(auto* stream_callback_registry,
                        neptune::GetGlobalStreamCallbackRegistry());
    TF_ASSIGN_OR_RETURN(scoped_stream_callback,
                        stream_callback_registry->Register(
                            /*model_name=*/"", *executable_bundle->callback_id,
                            current_request_id, std::move(wrapped_callback)));
  }

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

  // `used_input_indices` is used to track the index of used inputs in
  // stablehlo spec because some inputs may be presented in the model signature
  // but not used by StableHlo.
  std::vector<int> used_input_indices;
  for (int idx : executable_bundle->module_kept_var_idx) {
    if (idx >= weights_array.size()) {
      used_input_indices.push_back(idx - weights_array.size());
    }
  }

  std::vector<HostBuffer> reshaped_inputs;
  std::vector<xla::ifrt::ArrayRef> converted_device_inputs;
  if (options.require_h2d) {
    reshaped_inputs.reserve(used_input_indices.size());
    for (int i = 0; i < used_input_indices.size(); ++i) {
      int idx_in_stablehlo_spec = i + executable_bundle->input_offset;
      int original_input_idx = used_input_indices[i];
      auto& original_input = host_buffers[original_input_idx];

      // If the optimizer is not set, the input shape won't be changed. If the
      // input is already rank 1 or is a scalar, we can copy the input without
      // doing any reshaping.
      if (optimizer_ == nullptr || original_input.shape().dims().size() <= 1) {
        reshaped_inputs.push_back(std::move(original_input));
        continue;
      }

      // R1 optimization is applied, the shape expected by StableHlo is changed
      // to rank 1, so here we need to reshape the input to rank 1.
      xla::ifrt::Shape shape = ToIfrtShape(
          executable_bundle->io_spec.inputs[idx_in_stablehlo_spec].shape);

      // The original HostBuffer in `inputs` will be alive during the execution,
      // so there is no need to give the `reshaped_buffer` an owner.
      HostBuffer reshaped_buffer(original_input.base(), original_input.dtype(),
                                 shape);
      reshaped_inputs.push_back(std::move(reshaped_buffer));
    }
  } else {
    converted_device_inputs.reserve(used_input_indices.size());
    for (int i = 0; i < used_input_indices.size(); ++i) {
      int idx_in_stablehlo_spec = i + executable_bundle->input_offset;
      int original_input_idx = used_input_indices[i];
      xla::ifrt::ArrayRef original_input =
          device_buffers[original_input_idx].ifrt_array;

      xla::ifrt::Shape target_shape = ToIfrtShape(
          executable_bundle->io_spec.inputs[idx_in_stablehlo_spec].shape);

      if (original_input->shape() != target_shape) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Zero-copy D2D execution (require_h2d=false) requires the input "
            "shape to exactly match the optimized model's target shape. "
            "Expected ",
            target_shape, ", but got ", original_input->shape(), ". "));
      }
      converted_device_inputs.push_back(std::move(original_input));
    }
  }

  // TODO: Consider validating that the inputs and outputs match
  // the model IO spec before converting.
  std::vector<xla::ifrt::ArrayRef> converted_inputs;
  if (options.require_h2d) {
    TF_ASSIGN_OR_RETURN(
        converted_inputs,
        data_converter->ConvertInput(reshaped_inputs, *executable_bundle,
                                     executable_bundle->input_offset));
  } else {
    converted_inputs = std::move(converted_device_inputs);
  }

  std::vector<xla::ifrt::ArrayRef> inputs_with_weights;
  inputs_with_weights.reserve(executable_bundle->module_kept_var_idx.size());
  for (int idx : executable_bundle->module_kept_var_idx) {
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
  // `converted_inputs` only contains used inputs, so we just insert them
  // at the end of the list.
  inputs_with_weights.insert(inputs_with_weights.end(),
                             std::make_move_iterator(converted_inputs.begin()),
                             std::make_move_iterator(converted_inputs.end()));
  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutable::ExecuteResult execution_result,
      executable_bundle->ifrt_executable->Execute(
          absl::MakeSpan(inputs_with_weights),
          /*options=*/
          {
              .launch_id = static_cast<int32_t>(current_request_id),
              .fill_status = true,
          },
          /*devices=*/
          IsPortableExecutable()
              ? std::make_optional(data_converter->device_list())
              : std::nullopt));
  TF_RETURN_IF_ERROR(execution_result.status.Await());

  if (options.require_d2h) {
    TF_ASSIGN_OR_RETURN(
        std::vector<HostBuffer> host_outputs,
        data_converter->ConvertOutput(absl::MakeSpan(execution_result.outputs),
                                      executable_bundle->io_spec));
    return neptune::internal::PackAsValues(std::move(host_outputs));
  } else {
    std::vector<neptune::DeviceBuffer> device_outputs;
    device_outputs.reserve(execution_result.outputs.size());
    for (int i = 0; i < execution_result.outputs.size(); ++i) {
      // TODO: Precompute and cache the `xla::ifrt::HloSharding`
      // (or `xla::ifrt::ShardingRef`) directly in the `CachedExecutableBundle`.
      // Currently, copying `xla::HloSharding` by value per-request and
      // dynamically recreating the `ifrt_sharding` during D2H transfers brings
      // unnecessary CPU overhead.
      device_outputs.push_back(neptune::DeviceBuffer{
          .ifrt_client = ifrt_client_,
          .ifrt_array = std::move(execution_result.outputs[i]),
          .sharding = executable_bundle->io_spec.outputs[i].sharding,
          .device_list = data_converter->device_list()});
    }
    return neptune::internal::PackAsValues(std::move(device_outputs));
  }
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
                            ifrt_client_, device_list, thread_pool));
  }
  device_list_to_resources_map_[device_list] =
      std::make_unique<ResourcesPerDeviceList>(
          ifrt_client_, device_list, thread_pool,
          std::move(h2d_transfer_executor));
  return absl::OkStatus();
}

}  // namespace neptune
