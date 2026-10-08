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

#include "runtime/kernel/stablehlo_runner_op.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/flat_signature.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner_base.h"  // from @xla_serving
#include "runtime/core/streaming/utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_constants.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace xla_serving {

absl::Status StableHloRunnerFactoryRegistry::Register(
    StableHloRunnerMode mode, StableHloRunnerFactory factory) {
  absl::MutexLock lock(mu_);
  auto [it, inserted] = factories_.try_emplace(mode, std::move(factory));
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("StableHLO runner factory already registered for mode ",
                     mode == StableHloRunnerMode::kAot ? "kAot" : "kJit"));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<StableHloRunnerBase>>
StableHloRunnerFactoryRegistry::Create(
    StableHloRunnerMode mode, const KernelBuilderContext& context,
    std::shared_ptr<StableHlo> stablehlo,
    StableHloRunnerOptions base_options) const {
  absl::MutexLock lock(mu_);
  auto it = factories_.find(mode);
  if (it == factories_.end()) {
    const char* target_hint = mode == StableHloRunnerMode::kAot
                                  ? "':aot_execution'"
                                  : "':jit_compilation'";
    return absl::NotFoundError(absl::StrCat(
        "No StableHLO runner factory registered for mode ",
        mode == StableHloRunnerMode::kAot ? "kAot" : "kJit",
        ". Did you forget to link the ", target_hint, " build target?"));
  }
  return it->second(context, std::move(stablehlo), std::move(base_options));
}

namespace {

// Takes, out of the caller's name-keyed aliased buffers, those naming this
// component's outputs, in the order the executable expects: the order of
// `output_signature`. The executable's aliases are compiled against argument
// positions, so the names have to be resolved before the arrays reach it.
//
// Returns an empty vector when none of this component's outputs are named, in
// which case the runner allocates the output memory itself. The interpreter has
// already checked that every name in the map belongs to exactly one component
// that uses input aliasing, so entries left behind belong to other components.
absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ClaimInputAliasedOutputArrays(
    const FlatSignature* output_signature,
    absl::flat_hash_map<std::string, xla::ifrt::ArrayRef>& by_name) {
  if (by_name.empty() || output_signature == nullptr ||
      !output_signature->IsAllSpecWithName()) {
    return std::vector<xla::ifrt::ArrayRef>();
  }
  const std::vector<MetaArgSpec>& specs = output_signature->specs();

  std::vector<xla::ifrt::ArrayRef> claimed(specs.size());
  int num_claimed = 0;
  for (int i = 0; i < specs.size(); ++i) {
    absl::flat_hash_map<std::string, xla::ifrt::ArrayRef>::iterator it =
        by_name.find(*specs[i].name);
    if (it != by_name.end()) {
      claimed[i] = std::move(it->second);
      by_name.erase(it);
      ++num_claimed;
    }
  }
  if (num_claimed == 0) {
    return std::vector<xla::ifrt::ArrayRef>();
  }

  // Input aliasing is all-or-nothing: rather than quietly allocating the
  // outputs the caller left out, which would put an allocation back on a path
  // the caller is trying to keep free of them, reject the request.
  for (int i = 0; i < specs.size(); ++i) {
    if (claimed[i] == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "No input-aliased output buffer was supplied for output \"",
          *specs[i].name,
          "\", although buffers were supplied for this component's other "
          "outputs. Every output of a component must be given a buffer, or "
          "none."));
    }
  }
  return claimed;
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
  absl::StatusOr<int64_t> callback_id_status =
      context.GetConfig<int64_t>(kStableCallbackId);
  std::optional<int64_t> callback_id;
  if (callback_id_status.ok()) {
    callback_id = *callback_id_status;
  } else if (!absl::IsNotFound(callback_id_status.status())) {
    return callback_id_status.status();
  }

  StableHloRunnerOptions base_options{
      .client = std::move(ifrt_client),
      .thread_pool = thread_pool,
      .ifrt_loaded_variable_registry = ifrt_loaded_variable_registry,
      .weights = std::move(weights),
      .assigned_devices = std::move(assigned_devices),
      .callback_id = callback_id,
  };

  StableHloRunnerMode mode;
  auto loaded_executables =
      context.GetConfig<std::vector<xla::ifrt::LoadedExecutableRef>>(
          kIfrtLoadedExecutables);
  if (loaded_executables.ok()) {
    // AOT Execution Path
    mode = StableHloRunnerMode::kAot;
  } else if (absl::IsNotFound(loaded_executables.status())) {
    // JIT Fallback Execution Path
    mode = StableHloRunnerMode::kJit;
  } else {
    return loaded_executables.status();
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<StableHloRunnerBase> stablehlo_runner,
      StableHloRunnerFactoryRegistry::GetGlobalRegistry().Create(
          mode, context, std::move(stablehlo), std::move(base_options)));

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
      absl::StatusOr<HostBuffer> request_ids_buffer_status;
      if (!context.run_context.request_ids.empty()) {
        request_ids_buffer_status =
            CreateRequestIdsBuffer(context.run_context.request_ids);
      } else {
        request_ids_buffer_status = CreateScalarRequestIdBuffer(request_id);
      }
      TF_ASSIGN_OR_RETURN(HostBuffer request_ids_buffer,
                          request_ids_buffer_status);
      inputs_with_stream_step_id.assign(inputs_span.begin(), inputs_span.end());
      inputs_with_stream_step_id.push_back(
          Value(std::move(request_ids_buffer)));
      inputs_span = inputs_with_stream_step_id;
    }
  }

  StableHloRunnerBase::ExecuteOptions execute_options;
  execute_options.placements = context.run_context.placements;
  // Take only the buffers naming this component's outputs. Moving them out of
  // our own copy of the run context leaves the other kernels' copies
  // untouched, and makes it impossible to hand the same arrays to a second
  // execution.
  TF_ASSIGN_OR_RETURN(
      execute_options.output_arrays,
      ClaimInputAliasedOutputArrays(
          output_signature(), context.run_context.input_aliased_output_arrays));
  TF_ASSIGN_OR_RETURN(
      std::vector<Value> output_values,
      stablehlo_runner_->Execute(inputs_span, std::move(execute_options),
                                 context.run_context));
  context.outputs = std::move(output_values);

  return absl::OkStatus();
}

}  // namespace xla_serving
