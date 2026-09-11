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

#include "runtime/interpreter/bundle_interpreter.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/bundle_orchestration.pb.h"  // from @xla_serving
#include "runtime/core/common_util.h"  // from @xla_serving
#include "runtime/core/flat_signature.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/interpreter/interpreter.h"  // from @xla_serving
#include "runtime/interpreter/interpreter_create_options.h"  // from @xla_serving
#include "runtime/interpreter/run_options.h"  // from @xla_serving
#include "runtime/kernel/kernel_io_data_type.h"  // from @xla_serving
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace xla_serving {

namespace {

absl::StatusOr<neptune_bundle_orchestration::BundlePipelines>
ParseBundleOrchestration(absl::string_view bundle_path) {
  neptune_bundle_orchestration::BundlePipelines proto;
  std::string pb_path =
      tsl::io::JoinPath(bundle_path, "bundle_orchestration.pb");
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), pb_path, &proto));
  return proto;
}

}  // namespace

bool BundleInterpreter::IsBundle(absl::string_view model_path) {
  if (!IsOrbaxModelDirectory(model_path).ok()) {
    return false;
  }
  std::string pb_path =
      tsl::io::JoinPath(model_path, "bundle_orchestration.pb");
  return tsl::Env::Default()->FileExists(pb_path).ok();
}

BundleInterpreter::BundleInterpreter(
    std::unique_ptr<InterpreterCreateOptions> global_options)
    : Interpreter(std::move(global_options)) {}

absl::StatusOr<std::unique_ptr<BundleInterpreter>> BundleInterpreter::Create(
    std::unique_ptr<InterpreterCreateOptions> global_options) {
  if (global_options == nullptr) {
    return absl::InvalidArgumentError("global_options must not be null.");
  }
  std::string bundle_path = global_options->model_path;
  auto interpreter =
      std::make_unique<BundleInterpreter>(std::move(global_options));
  interpreter->bundle_path_ = bundle_path;
  TF_RETURN_IF_ERROR(interpreter->Load());
  return interpreter;
}

absl::Status BundleInterpreter::Load() {
  // 1. Parse master bundle orchestration
  TF_ASSIGN_OR_RETURN(master_orchestration_,
                      ParseBundleOrchestration(bundle_path_));

  if (create_options() == nullptr) {
    return absl::InvalidArgumentError(
        "BundleInterpreter::Load() failed: create_options() is null.");
  }

  // 2. Initialize sub-interpreters for ModelExecutionComponents
  for (const auto& [pipeline_name, pipeline] :
       master_orchestration_.pipelines()) {
    for (const auto& model_comp : pipeline.components()) {
      auto& interpreter_ptr = interpreters_[model_comp.model_name()];
      if (interpreter_ptr == nullptr) {
        TF_ASSIGN_OR_RETURN(interpreter_ptr,
                            CreateSubInterpreter(model_comp.model_name()));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Value>> BundleInterpreter::Run(
    absl::string_view pipeline_name, absl::Span<const Value> positional_inputs,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs,
    RunOptions run_options) {
  return absl::UnimplementedError(
      "Run is not supported for BundleInterpreter. Use RunWithMapReturn "
      "instead.");
}

tsl::Future<std::vector<Value>> BundleInterpreter::RunAsync(
    absl::string_view pipeline_name, absl::Span<const Value> positional_inputs,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs,
    RunOptions run_options) {
  return tsl::Future<std::vector<Value>>(absl::UnimplementedError(
      "RunAsync is not supported for BundleInterpreter. Use "
      "RunAsyncWithMapReturn instead."));
}

std::vector<std::string> BundleInterpreter::GetPipelineNames() const {
  std::vector<std::string> names;
  names.reserve(master_orchestration_.pipelines().size());
  for (const auto& [name, _] : master_orchestration_.pipelines()) {
    names.push_back(name);
  }
  return names;
}

absl::StatusOr<FunctionFlatSignature> BundleInterpreter::GetPipelineSignature(
    absl::string_view pipeline_name) const {
  // Get input signature from first component
  TF_ASSIGN_OR_RETURN(const auto* first_comp, GetFirstComponent(pipeline_name));
  auto first_interp_it = interpreters_.find(first_comp->model_name());
  if (first_interp_it == interpreters_.end() ||
      first_interp_it->second == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Sub-interpreter not loaded for model: ", first_comp->model_name()));
  }
  TF_ASSIGN_OR_RETURN(FunctionFlatSignature first_sig,
                      first_interp_it->second->GetPipelineSignature(
                          first_comp->pipeline_name()));

  // Get output signature from last component
  TF_ASSIGN_OR_RETURN(const auto* last_comp, GetLastComponent(pipeline_name));
  auto last_interp_it = interpreters_.find(last_comp->model_name());
  if (last_interp_it == interpreters_.end() ||
      last_interp_it->second == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Sub-interpreter not loaded for model: ", last_comp->model_name()));
  }
  TF_ASSIGN_OR_RETURN(
      FunctionFlatSignature last_sig,
      last_interp_it->second->GetPipelineSignature(last_comp->pipeline_name()));

  return FunctionFlatSignature{
      .input_signature = std::move(first_sig.input_signature),
      .output_signature = std::move(last_sig.output_signature),
  };
}

absl::StatusOr<neptune_model_orchestration::Pipeline>
BundleInterpreter::GetPipelineProto(absl::string_view pipeline_name) const {
  return absl::UnimplementedError(
      "GetPipelineProto is not implemented for BundleInterpreter.");
}

absl::StatusOr<KernelIODataType> BundleInterpreter::GetModelInputDataType(
    absl::string_view pipeline_name) const {
  // Find the first component
  TF_ASSIGN_OR_RETURN(const auto* first_comp, GetFirstComponent(pipeline_name));
  auto interp_it = interpreters_.find(first_comp->model_name());
  if (interp_it == interpreters_.end() || interp_it->second == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Sub-interpreter not loaded for model: ", first_comp->model_name()));
  }
  return interp_it->second->GetModelInputDataType(first_comp->pipeline_name());
}

absl::StatusOr<KernelIODataType> BundleInterpreter::GetModelOutputDataType(
    absl::string_view pipeline_name) const {
  // Find the last component
  TF_ASSIGN_OR_RETURN(const auto* last_comp, GetLastComponent(pipeline_name));
  auto interp_it = interpreters_.find(last_comp->model_name());
  if (interp_it == interpreters_.end() || interp_it->second == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Sub-interpreter not loaded for model: ", last_comp->model_name()));
  }
  return interp_it->second->GetModelOutputDataType(last_comp->pipeline_name());
}

absl::StatusOr<absl::flat_hash_map<std::string, Value>>
BundleInterpreter::RunWithMapReturn(
    absl::string_view pipeline_name, absl::Span<const Value> positional_inputs,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs,
    RunOptions run_options) {
  TF_ASSIGN_OR_RETURN(FunctionFlatSignature signature,
                      GetPipelineSignature(pipeline_name));
  if (!signature.output_signature.IsAllSpecWithName()) {
    return absl::FailedPreconditionError(
        "Not all output specs have names so map return is not supported for "
        "this pipeline. Change the pipeline's signature to have names for "
        "all outputs, e.g., use dict-structured outputs.");
  }

  TF_ASSIGN_OR_RETURN(const auto* pipeline, GetPipeline(pipeline_name));

  absl::flat_hash_map<std::string, Value> accumulated_values = keyword_inputs;

  for (const auto& model_comp : pipeline->components()) {
    auto interp_it = interpreters_.find(model_comp.model_name());
    if (interp_it == interpreters_.end() || interp_it->second == nullptr) {
      return absl::InternalError(absl::StrCat(
          "Sub-interpreter not loaded for model: ", model_comp.model_name()));
    }
    auto* interpreter = interp_it->second.get();

    int64_t repeat_count = model_comp.repeated_times();
    if (repeat_count <= 0) {
      repeat_count = 1;
    }

    for (int64_t i = 0; i < repeat_count; ++i) {
      // Get MapReturn output and save them in map
      absl::flat_hash_map<std::string, Value> component_outputs;
      TF_ASSIGN_OR_RETURN(component_outputs,
                          interpreter->RunWithMapReturn(
                              model_comp.pipeline_name(), positional_inputs,
                              accumulated_values, run_options));

      for (auto&& [key, val] : component_outputs) {
        accumulated_values.insert_or_assign(key, std::move(val));
      }

      // Positional inputs are only used in the first component in the pipeline.
      // After the first run, the outputs go into accumulated_values and we
      // no longer need any positional inputs. By clearing the positional
      // inputs we prevent them from being incorrectly passed to subsequent
      // components.
      positional_inputs = {};
    }
  }

  // Filter outputs to match the bundle's output signature.
  absl::flat_hash_map<std::string, Value> final_outputs;
  for (const auto& spec : signature.output_signature.specs()) {
    if (!spec.name.has_value()) {
      return absl::InternalError("Output spec has no name");
    }
    const std::string& name = *spec.name;
    auto it = accumulated_values.find(name);
    if (it == accumulated_values.end()) {
      return absl::InternalError(absl::StrCat(
          "Output with name \"", name, "\" not found in accumulated values."));
    }
    final_outputs.emplace(name, std::move(it->second));
  }

  return final_outputs;
}

tsl::Future<absl::flat_hash_map<std::string, Value>>
BundleInterpreter::RunAsyncWithMapReturn(
    absl::string_view pipeline_name, absl::Span<const Value> positional_inputs,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs,
    RunOptions run_options) {
  auto [promise, future] =
      tsl::MakePromise<absl::flat_hash_map<std::string, Value>>();

  if (create_options() == nullptr ||
      create_options()->execute_worker_thread_pool == nullptr) {
    promise.Set(absl::FailedPreconditionError(
        "BundleInterpreter::RunAsyncWithMapReturn() requires "
        "execute_worker_thread_pool to be set in global_options."));
    return future;
  }

  create_options()->execute_worker_thread_pool->Schedule(
      [this, pipeline_name = std::string(pipeline_name),
       positional_inputs_copy = std::vector<Value>(positional_inputs.begin(),
                                                   positional_inputs.end()),
       keyword_inputs_copy = keyword_inputs, run_options,
       promise = std::move(promise).ToShared()]() mutable {
        absl::StatusOr<absl::flat_hash_map<std::string, Value>> result =
            this->RunWithMapReturn(pipeline_name, positional_inputs_copy,
                                   keyword_inputs_copy, run_options);
        if (result.ok()) {
          promise->Set(std::move(*result));
        } else {
          promise->Set(result.status());
        }
      });

  return future;
}

absl::Status BundleInterpreter::MarkFullyInitialized() {
  for (auto& [_, interpreter] : interpreters_) {
    if (interpreter != nullptr) {
      TF_RETURN_IF_ERROR(interpreter->MarkFullyInitialized());
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Interpreter>>
BundleInterpreter::CreateSubInterpreter(absl::string_view model_name) const {
  auto sub_options = std::make_unique<InterpreterCreateOptions>();
  sub_options->ifrt_client = create_options()->ifrt_client;
  sub_options->host_ifrt_client = create_options()->host_ifrt_client;
  sub_options->execute_worker_thread_pool =
      create_options()->execute_worker_thread_pool;
  sub_options->computation_thread_pool =
      create_options()->computation_thread_pool;
  sub_options->allocator = create_options()->allocator;
  sub_options->platform_name = create_options()->platform_name;

  sub_options->model_path = tsl::io::JoinPath(bundle_path_, model_name);

  return Interpreter::Create(std::move(sub_options));
}

absl::StatusOr<const neptune_bundle_orchestration::BundlePipeline*>
BundleInterpreter::GetPipeline(absl::string_view pipeline_name) const {
  auto pipe_it =
      master_orchestration_.pipelines().find(std::string(pipeline_name));
  if (pipe_it == master_orchestration_.pipelines().end()) {
    return absl::NotFoundError(
        absl::StrCat("Pipeline not found in bundle: ", pipeline_name));
  }
  return &pipe_it->second;
}

absl::StatusOr<const neptune_bundle_orchestration::ModelExecutionComponent*>
BundleInterpreter::GetFirstComponent(absl::string_view pipeline_name) const {
  TF_ASSIGN_OR_RETURN(const auto* pipeline, GetPipeline(pipeline_name));
  if (pipeline->components().empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Pipeline ", pipeline_name, " has no components."));
  }
  return &pipeline->components(0);
}

absl::StatusOr<const neptune_bundle_orchestration::ModelExecutionComponent*>
BundleInterpreter::GetLastComponent(absl::string_view pipeline_name) const {
  TF_ASSIGN_OR_RETURN(const auto* pipeline, GetPipeline(pipeline_name));
  if (pipeline->components().empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Pipeline ", pipeline_name, " has no components."));
  }
  return &pipeline->components(pipeline->components_size() - 1);
}

}  // namespace xla_serving
