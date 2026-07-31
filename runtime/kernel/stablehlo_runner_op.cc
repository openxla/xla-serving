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

#include "runtime/kernel/stablehlo_runner_op.h"  // from @neptune

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/aot_stablehlo_runner.h"  // from @neptune
#include "runtime/core/flat_signature.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @neptune
#include "runtime/core/stablehlo.h"  // from @neptune
#include "runtime/core/stablehlo_optimizer.h"  // from @neptune
#include "runtime/core/stablehlo_runner.h"  // from @neptune
#include "runtime/core/streaming/utils.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_constants.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @neptune
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @neptune
#include "google/protobuf/message.h"
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/tsl/platform/file_system.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace neptune {

namespace {

absl::StatusOr<std::unique_ptr<StableHloRunner>> CreateAotRunner(
    const KernelBuilderContext& context,
    std::vector<xla::ifrt::LoadedExecutableRef> loaded_executables,
    std::shared_ptr<StableHlo> stablehlo,
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    tsl::thread::ThreadPool* thread_pool,
    IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
    std::vector<std::string> weights,
    std::shared_ptr<const AssignedDevices> assigned_devices,
    std::optional<int64_t> callback_id) {
  std::vector<std::shared_ptr<tsl::ReadOnlyMemoryRegion>> aot_memory_regions;
  auto aot_memory_regions_config =
      context
          .GetConfig<std::vector<std::shared_ptr<tsl::ReadOnlyMemoryRegion>>>(
              kIfrtAotMemoryRegions);
  if (aot_memory_regions_config.ok()) {
    aot_memory_regions = std::move(*aot_memory_regions_config);
  } else if (!absl::IsNotFound(aot_memory_regions_config.status())) {
    return aot_memory_regions_config.status();
  }

  StableHloRunnerAotOptions aot_options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool,
              .ifrt_loaded_variable_registry = ifrt_loaded_variable_registry,
              .weights = std::move(weights),
              .assigned_devices = assigned_devices,
              .callback_id = callback_id,
          },
      .executables = std::move(loaded_executables),
      .stablehlo = std::move(stablehlo),
      .aot_memory_regions = std::move(aot_memory_regions),
  };

  return AotStableHloRunner::Create(std::move(aot_options));
}

absl::StatusOr<std::unique_ptr<StableHloRunner>> CreateJitRunner(
    const KernelBuilderContext& context, std::shared_ptr<StableHlo> stablehlo,
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    tsl::thread::ThreadPool* thread_pool,
    IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
    std::vector<std::string> weights,
    std::shared_ptr<const AssignedDevices> assigned_devices,
    ServingCoreSelector* serving_core_selector,
    std::optional<int64_t> callback_id) {
  TF_ASSIGN_OR_RETURN(
      IfrtPersistentCompilationCache * persistent_compilation_cache,
      context.GetConfig<IfrtPersistentCompilationCache*>(
          kPersistentCompilationCache));
  TF_ASSIGN_OR_RETURN(
      tsl::protobuf::Message * compilation_environment_proto,
      context.GetConfig<tsl::protobuf::Message*>(kCompilationEnvProto));

  absl::StatusOr<xla::CompileOptionsProto> compile_options =
      context.GetConfig<xla::CompileOptionsProto>(kXlaCompileOptions);
  // If the compile options is not found, we will use the default compile
  // options.
  if (!compile_options.ok() && !absl::IsNotFound(compile_options.status())) {
    return compile_options.status();
  }

  absl::StatusOr<StablehloOptimizer*> optimizer =
      context.GetConfig<StablehloOptimizer*>(kStablehloOptimizer);
  // If StablehloOptimizer is not set in the kernel builder context, we will
  // not use the optimizer.
  if (absl::IsNotFound(optimizer.status())) {
    optimizer = nullptr;
  }

  return StableHloRunner::Create(
      std::move(stablehlo),
      compile_options.ok() ? std::make_optional(*compile_options)
                           : std::nullopt,
      ifrt_client, persistent_compilation_cache, ifrt_loaded_variable_registry,
      std::move(weights), compilation_environment_proto, thread_pool,
      *optimizer, assigned_devices, serving_core_selector,
      /*h2d_transfer_executor_factory=*/nullptr, callback_id);
}

}  // namespace

absl::StatusOr<std::unique_ptr<Kernel>> StableHloRunnerOp::Create(
    const KernelBuilderContext& context) {
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<StableHlo> stablehlo,
      context.GetConfig<std::shared_ptr<StableHlo>>(kStableHlo));
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<xla::ifrt::Client> ifrt_client,
      context.GetConfig<std::shared_ptr<xla::ifrt::Client>>(kIfrtClient));
  TF_ASSIGN_OR_RETURN(
      tsl::thread::ThreadPool * thread_pool,
      context.GetConfig<tsl::thread::ThreadPool*>(kIfrtWorkerThreadPool));
  TF_ASSIGN_OR_RETURN(
      IfrtLoadedVariableRegistry * ifrt_loaded_variable_registry,
      context.GetConfig<IfrtLoadedVariableRegistry*>(
          kIfrtLoadedVariableRegistry));
  TF_ASSIGN_OR_RETURN(std::vector<std::string> weights,
                      context.GetConfig<std::vector<std::string>>(kWeights));
  TF_ASSIGN_OR_RETURN(std::shared_ptr<const AssignedDevices> assigned_devices,
                      context.GetConfig<std::shared_ptr<const AssignedDevices>>(
                          kAssignedDevices));
  absl::StatusOr<FlatSignature> input_sig_status =
      context.GetConfig<FlatSignature>(kFunctionFlatInputSignature);
  std::unique_ptr<FlatSignature> input_sig;
  if (input_sig_status.ok()) {
    input_sig = std::make_unique<FlatSignature>(*input_sig_status);
  } else if (!absl::IsNotFound(input_sig_status.status())) {
    return input_sig_status.status();
  }
  absl::StatusOr<FlatSignature> output_sig_status =
      context.GetConfig<FlatSignature>(kFunctionFlatOutputSignature);
  std::unique_ptr<FlatSignature> output_sig;
  if (output_sig_status.ok()) {
    output_sig = std::make_unique<FlatSignature>(*output_sig_status);
  } else if (!absl::IsNotFound(output_sig_status.status())) {
    return output_sig_status.status();
  }
  absl::StatusOr<ServingCoreSelector*> serving_core_selector_status =
      context.GetConfig<ServingCoreSelector*>(kServingCoreSelector);
  ServingCoreSelector* serving_core_selector = nullptr;
  if (serving_core_selector_status.ok()) {
    serving_core_selector = *serving_core_selector_status;
  } else if (!absl::IsNotFound(serving_core_selector_status.status())) {
    return serving_core_selector_status.status();
  }

  absl::StatusOr<int64_t> callback_id_status =
      context.GetConfig<int64_t>(kStableCallbackId);
  std::optional<int64_t> callback_id;
  if (callback_id_status.ok()) {
    callback_id = *callback_id_status;
  } else if (!absl::IsNotFound(callback_id_status.status())) {
    return callback_id_status.status();
  }

  std::unique_ptr<StableHloRunner> stablehlo_runner;

  auto loaded_executables =
      context.GetConfig<std::vector<xla::ifrt::LoadedExecutableRef>>(
          kIfrtLoadedExecutables);
  if (loaded_executables.ok()) {
    // AOT Execution Path
    TF_ASSIGN_OR_RETURN(
        stablehlo_runner,
        CreateAotRunner(context, std::move(*loaded_executables), stablehlo,
                        ifrt_client, thread_pool, ifrt_loaded_variable_registry,
                        std::move(weights), assigned_devices, callback_id));
  } else if (absl::IsNotFound(loaded_executables.status())) {
    // JIT Fallback Execution Path
    TF_ASSIGN_OR_RETURN(
        stablehlo_runner,
        CreateJitRunner(context, stablehlo, ifrt_client, thread_pool,
                        ifrt_loaded_variable_registry, std::move(weights),
                        assigned_devices, serving_core_selector, callback_id));
  } else {
    return loaded_executables.status();
  }

  return absl::WrapUnique(new StableHloRunnerOp(std::move(stablehlo_runner),
                                                std::move(input_sig),
                                                std::move(output_sig)));
}

std::string StableHloRunnerOp::Name() const { return "StableHloRunnerOp"; }

absl::Status StableHloRunnerOp::RunInternal(ExecuteContext& context) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(
        "StableHloRunnerOp::Run",
        {{"model_id", absl::StrCat(context.metadata.model_name, ":",
                                   context.metadata.model_version)}});
  });
  absl::Span<const Value> inputs_span = context.inputs;
  std::vector<Value> inputs_with_stream_step_id;
  if (stablehlo_runner_->has_streaming()) {
    int32_t request_id = context.run_context.request_id;
    if (inputs_span.size() == stablehlo_runner_->GetNumInputs()) {
      absl::StatusOr<neptune::HostBuffer> request_ids_buffer_status;
      if (!context.run_context.request_ids.empty()) {
        request_ids_buffer_status =
            neptune::CreateRequestIdsBuffer(context.run_context.request_ids);
      } else {
        request_ids_buffer_status =
            neptune::CreateScalarRequestIdBuffer(request_id);
      }
      TF_ASSIGN_OR_RETURN(neptune::HostBuffer request_ids_buffer,
                          request_ids_buffer_status);
      inputs_with_stream_step_id.assign(inputs_span.begin(), inputs_span.end());
      inputs_with_stream_step_id.push_back(
          Value(std::move(request_ids_buffer)));
      inputs_span = inputs_with_stream_step_id;
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<Value> output_values,
      stablehlo_runner_->Execute(inputs_span, context.run_context));
  context.outputs = std::move(output_values);

  return absl::OkStatus();
}

}  // namespace neptune
