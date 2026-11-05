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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/optimization.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/log/vlog_is_on.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/clock.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"  // from @llvm-project
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/ifrt_device_utils.h"  // from @xla_serving
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/core/loaded_stablehlo.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_optimizer.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner_base.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/core/streaming/stream.h"  // from @xla_serving
#include "runtime/core/streaming/utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/executable_build_utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @xla_serving
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @xla_serving
#include "runtime/libs/mlir/dump_mlir_util.h"  // from @xla_serving
#include "google/protobuf/message.h"
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/refine_polymorphic_shapes.h"  // from @xla
#include "xla/service/device_assignment.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

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

StableHloRunner::StableHloRunner(
    absl_nonnull std::unique_ptr<LoadedStableHlo> loaded_stablehlo,
    std::vector<std::optional<xla::ifrt::Shape>> user_input_static_shapes,
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
    : StableHloRunnerBase(
          ModelIoMetadata{
              .weight_specs = loaded_stablehlo->weight_specs(),
              .output_specs = loaded_stablehlo->output_specs(),
              .num_user_inputs = loaded_stablehlo->num_user_inputs(),
              .num_outputs = loaded_stablehlo->num_outputs(),
              .has_streaming = loaded_stablehlo->has_streaming(),
              .user_input_static_shapes = std::move(user_input_static_shapes),
          },
          client, ifrt_loaded_variable_registry, std::move(weights),
          std::move(assigned_device_list), std::move(assigned_devices),
          serving_core_selector, std::move(h2d_transfer_executor_factory),
          callback_id),
      loaded_stablehlo_(std::move(loaded_stablehlo)),
      compile_options_(std::move(compile_options)),
      persistent_compilation_cache_(persistent_compilation_cache),
      compilation_environment_proto_(compilation_environment_proto),
      optimizer_(optimizer) {}

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
  if (thread_pool == nullptr) {
    return absl::InvalidArgumentError(
        "`tsl::thread::ThreadPool` should be non-null when creating "
        "`StableHloRunner`");
  }
  if (ifrt_loaded_variable_registry == nullptr) {
    return absl::InvalidArgumentError(
        "`IfrtLoadedVariableRegistry` should be non-null when creating "
        "`StableHloRunner`");
  }
  if (assigned_devices == nullptr) {
    return absl::InvalidArgumentError(
        "`AssignedDevices` should be non-null when creating "
        "`StableHloRunner`");
  }

  int num_weights = weights.size();
  // Load StableHlo.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LoadedStableHlo> loaded_stablehlo,
      LoadedStableHlo::Create(stablehlo, num_weights, client->platform_name(),
                              compile_options));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::optional<xla::ifrt::Shape>> user_input_static_shapes,
      ModelIoMetadata::GetUserInputStaticShapes(
          *stablehlo, num_weights, loaded_stablehlo->num_user_inputs()));

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
      std::move(loaded_stablehlo), std::move(user_input_static_shapes),
      std::move(compile_options), client, ifrt_loaded_variable_registry,
      std::move(weights), assigned_device_list, compilation_environment_proto,
      persistent_compilation_cache, optimizer, assigned_devices,
      serving_core_selector, std::move(h2d_transfer_executor_factory),
      callback_id));
  TF_RETURN_IF_ERROR(runner->Init(thread_pool));
  return runner;
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
  // SetStreamCallbackId will return std::nullopt if the module doesn't actually
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

  absl::flat_hash_map<int, std::vector<int64_t>> host_transpose_permutations;
  if (optimizer_ != nullptr) {
    TF_RETURN_IF_ERROR(optimizer_->Optimize(args, module_bundle->module.get()));
  }
  if (mlir::func::FuncOp main_func =
          module_bundle->module->lookupSymbol<mlir::func::FuncOp>("main")) {
    for (int i = 0, e = main_func.getNumArguments(); i < e; ++i) {
      mlir::Attribute attr = main_func.removeArgAttr(i, kHostTransposeAttr);
      if (!attr) {
        continue;
      }
      auto permutation = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(attr);
      if (!permutation) {
        return absl::InternalError(
            absl::StrCat(kHostTransposeAttr, " on argument ", i,
                         " must be a DenseI64ArrayAttr"));
      }
      host_transpose_permutations[i].assign(permutation.asArrayRef().begin(),
                                            permutation.asArrayRef().end());
    }
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
  executable_bundle->host_transpose_permutations =
      std::move(host_transpose_permutations);

  executable_bundle->ifrt_executable = std::move(ifrt_executable);
  executable_bundle->program_fingerprint_with_timestamp =
      GetProgramFingerprintWithTimestamp(executable_bundle->ifrt_executable,
                                         absl::Now());

  executable_bundle->callback_id = callback_id;

  // With streaming, there will be an additional input for the request id.
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
          "serving."));
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

absl::Status StableHloRunner::MarkFullyInitialized() {
  absl::MutexLock lock(mutex_);
  fully_initialized_ = true;
  // Release the module as it is no longer needed after fully initialized
  // because no compilation will be allowed. A module can be large, e.g.
  // containing a large const.
  loaded_stablehlo_->ReleaseAllModules();
  return absl::OkStatus();
}

}  // namespace xla_serving
