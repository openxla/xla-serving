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

#include "runtime/interpreter/bundle_interpreter.h"  // from @neptune

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/bundle_orchestration.pb.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/interpreter/interpreter.h"  // from @neptune
#include "runtime/interpreter/interpreter_create_options.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace neptune {

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

absl::StatusOr<std::unique_ptr<BundleInterpreter>> BundleInterpreter::Create(
    std::unique_ptr<InterpreterCreateOptions> global_options) {
  if (global_options == nullptr) {
    return absl::InvalidArgumentError("global_options must not be null.");
  }
  auto interpreter = std::make_unique<BundleInterpreter>();
  interpreter->bundle_path_ = global_options->model_path;
  interpreter->global_options_ = std::move(global_options);
  TF_RETURN_IF_ERROR(interpreter->Load());
  return interpreter;
}

absl::Status BundleInterpreter::Load() {
  // 1. Parse master bundle orchestration
  TF_ASSIGN_OR_RETURN(master_orchestration_,
                      ParseBundleOrchestration(bundle_path_));

  if (global_options_ == nullptr) {
    return absl::InvalidArgumentError(
        "BundleInterpreter::Load() failed: global_options_ is null.");
  }

  // 2. Initialize sub-interpreters for ModelExecutionComponents
  for (const auto& [pipeline_name, pipeline] :
       master_orchestration_.pipelines()) {
    for (const auto& model_comp : pipeline.components()) {
      if (interpreters_.contains(model_comp.model_name())) continue;

      auto sub_options = std::make_unique<InterpreterCreateOptions>();
      // Inherit global resources
      sub_options->ifrt_client = global_options_->ifrt_client;
      sub_options->host_ifrt_client = global_options_->host_ifrt_client;
      sub_options->execute_worker_thread_pool =
          global_options_->execute_worker_thread_pool;
      sub_options->computation_thread_pool =
          global_options_->computation_thread_pool;
      sub_options->allocator = global_options_->allocator;
      sub_options->platform_name = global_options_->platform_name;

      sub_options->model_path =
          tsl::io::JoinPath(bundle_path_, model_comp.model_name());

      TF_ASSIGN_OR_RETURN(auto interpreter,
                          Interpreter::Create(std::move(sub_options)));
      interpreters_[model_comp.model_name()] = std::move(interpreter);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Value>> BundleInterpreter::Run(
    absl::string_view pipeline_name, absl::Span<const Value> inputs,
    RunOptions run_options) {
  std::vector<Value> current_values(inputs.begin(), inputs.end());

  auto pipe_it =
      master_orchestration_.pipelines().find(std::string(pipeline_name));
  if (pipe_it == master_orchestration_.pipelines().end()) {
    return absl::NotFoundError(
        absl::StrCat("Pipeline not found in bundle: ", pipeline_name));
  }
  const auto& pipeline = pipe_it->second;

  // Execute components in sequential order
  for (const auto& model_comp : pipeline.components()) {
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
      // Execute the sub-model, passing inputs from the previous component
      TF_ASSIGN_OR_RETURN(current_values,
                          interpreter->Run(model_comp.pipeline_name(),
                                           current_values, {}, run_options));
    }
  }

  return current_values;
}

tsl::Future<std::vector<Value>> BundleInterpreter::RunAsync(
    absl::string_view pipeline_name, absl::Span<const Value> inputs,
    RunOptions run_options) {
  auto [promise, future] = tsl::MakePromise<std::vector<Value>>();

  if (global_options_ == nullptr ||
      global_options_->execute_worker_thread_pool == nullptr) {
    promise.Set(absl::FailedPreconditionError(
        "BundleInterpreter::RunAsync() requires execute_worker_thread_pool "
        "to be set in global_options."));
    return future;
  }

  global_options_->execute_worker_thread_pool->Schedule(
      [this, pipeline_name = std::string(pipeline_name),
       inputs_copy = std::vector<Value>(inputs.begin(), inputs.end()),
       run_options, promise = std::move(promise).ToShared()]() mutable {
        absl::StatusOr<std::vector<Value>> result =
            this->Run(pipeline_name, inputs_copy, run_options);
        if (result.ok()) {
          promise->Set(std::move(*result));
        } else {
          promise->Set(result.status());
        }
      });

  return future;
}

}  // namespace neptune
