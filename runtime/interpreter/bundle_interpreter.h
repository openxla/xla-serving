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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_BUNDLE_INTERPRETER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_BUNDLE_INTERPRETER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/bundle_orchestration.pb.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/interpreter/interpreter.h"  // from @neptune
#include "runtime/interpreter/interpreter_create_options.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune
#include "xla/tsl/concurrency/future.h"  // from @xla

namespace xla_serving {

// BundleInterpreter manages a bundle of models and executes them sequentially
// as defined in a BundleOrchestration proto.
class BundleInterpreter : public Interpreter {
 public:
  explicit BundleInterpreter(
      std::unique_ptr<InterpreterCreateOptions> global_options);

  // Checks if the given path points to a bundle.
  static bool IsBundle(absl::string_view model_path);

  // Creates a BundleInterpreter instance and loads the bundle configuration.
  static absl::StatusOr<std::unique_ptr<BundleInterpreter>> Create(
      std::unique_ptr<InterpreterCreateOptions> global_options);

  // Runs the specified pipeline synchronously.
  absl::StatusOr<std::vector<Value>> Run(
      absl::string_view pipeline_name,
      absl::Span<const Value> positional_inputs,
      const absl::flat_hash_map<std::string, Value>& keyword_inputs = {},
      RunOptions run_options = RunOptions()) override;

  // Runs the specified pipeline asynchronously.
  tsl::Future<std::vector<Value>> RunAsync(
      absl::string_view pipeline_name,
      absl::Span<const Value> positional_inputs,
      const absl::flat_hash_map<std::string, Value>& keyword_inputs = {},
      RunOptions run_options = RunOptions()) override;

  // All the methods to be overridden from Interpreter:
  std::vector<std::string> GetPipelineNames() const override;

  absl::StatusOr<FunctionFlatSignature> GetPipelineSignature(
      absl::string_view pipeline_name) const override;

  absl::StatusOr<neptune_model_orchestration::Pipeline> GetPipelineProto(
      absl::string_view pipeline_name) const override;

  absl::StatusOr<KernelIODataType> GetModelInputDataType(
      absl::string_view pipeline_name) const override;

  absl::StatusOr<KernelIODataType> GetModelOutputDataType(
      absl::string_view pipeline_name) const override;

  absl::StatusOr<absl::flat_hash_map<std::string, Value>> RunWithMapReturn(
      absl::string_view pipeline_name,
      absl::Span<const Value> positional_inputs,
      const absl::flat_hash_map<std::string, Value>& keyword_inputs,
      RunOptions run_options = RunOptions()) override;

  tsl::Future<absl::flat_hash_map<std::string, Value>> RunAsyncWithMapReturn(
      absl::string_view pipeline_name,
      absl::Span<const Value> positional_inputs,
      const absl::flat_hash_map<std::string, Value>& keyword_inputs,
      RunOptions run_options = RunOptions()) override;

  absl::Status MarkFullyInitialized() override;

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<Interpreter>> interpreters_;
  neptune_bundle_orchestration::BundlePipelines master_orchestration_;
  std::string bundle_path_;
  absl::Status Load();
  absl::StatusOr<std::unique_ptr<Interpreter>> CreateSubInterpreter(
      absl::string_view model_name) const;
  absl::StatusOr<const neptune_bundle_orchestration::BundlePipeline*>
  GetPipeline(absl::string_view pipeline_name) const;
  absl::StatusOr<const neptune_bundle_orchestration::ModelExecutionComponent*>
  GetFirstComponent(absl::string_view pipeline_name) const;
  absl::StatusOr<const neptune_bundle_orchestration::ModelExecutionComponent*>
  GetLastComponent(absl::string_view pipeline_name) const;
};

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_BUNDLE_INTERPRETER_H_
