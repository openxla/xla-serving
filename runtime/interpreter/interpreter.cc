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

#include "runtime/interpreter/interpreter.h"  // from @xla_serving

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/random/random.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/cord.h"  // from @abseil-cpp
#include "absl/strings/match.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @xla_serving
#include "protos/orchestration.pb.h"  // from @xla_serving
#include "protos/type.pb.h"  // from @xla_serving
#include "runtime/core/batching/batch_scheduler.h"  // from @xla_serving
#include "runtime/core/constants.h"  // from @xla_serving
#include "runtime/core/flat_signature.h"  // from @xla_serving
#include "runtime/core/ifrt_device_utils.h"  // from @xla_serving
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/core/streaming/stream.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/data_loader/data_loader.h"  // from @xla_serving
#include "runtime/interpreter/bundle_interpreter.h"  // from @xla_serving
#include "runtime/interpreter/context.h"  // from @xla_serving
#include "runtime/interpreter/interpreter_create_options.h"  // from @xla_serving
#include "runtime/interpreter/linear_graph_execute.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/interpreter/optional_function_handler.h"  // from @xla_serving
#include "runtime/interpreter/optional_function_handler_factory.h"  // from @xla_serving
#include "runtime/interpreter/run_options.h"  // from @xla_serving
#include "runtime/kernel/batching/batch_options_utils.h"  // from @xla_serving
#include "runtime/kernel/batching/batch_resource_base.h"  // from @xla_serving
#include "runtime/kernel/batching/batch_runner_op.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_constants.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "runtime/kernel/kernel_io_data_type.h"  // from @xla_serving
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/serdes.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_compiler.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter_gauge.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/file_system.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/platform/blocking_counter.h"  // from @tsl
#include "tsl/platform/context.h"  // from @tsl
#include "tsl/platform/path.h"  // from @tsl

ABSL_FLAG(bool, jsv_enable_parallel_loading_pipelines, true,
          "When set to true, load the pipelines in parallel.");

namespace xla_serving {

using xla_saved_model_orchestration::BatchOptions;

namespace {

auto* model_loading_count = tsl::monitoring::Counter<3>::New(
    "/orbax/serving/model_loading_count",
    "Counts the number model loading attempts, broken down by model "
    "name, version and status code.",
    "model_name", "model_version", "status");

// Protects `model_loaded_count` from concurrent access.
auto* model_loaded_count = tsl::monitoring::CounterGauge<2>::New(
    "/orbax/serving/model_loaded_count",
    "Number of currently loaded models, broken down by model name and version.",
    "model_name", "model_version");

auto* request_count = tsl::monitoring::Counter<4>::New(
    "/orbax/serving/request_count",
    "Counts the number requests, broken down by model name, version, sync_mode "
    "and status code.",
    "model_name", "model_version", "sync_mode", "status");

using KernelSubGraph = std::vector<std::unique_ptr<Kernel>>;

struct BucketAotData {
  xla::ifrt::LoadedExecutableRef executable;
  StableHlo::Spec spec;
};

absl::Status ValidateStableHlo(
    absl::Span<const xla_saved_model_manifest::Function* absl_nonnull const>
        funcs) {
  for (auto* func : funcs) {
    // TODO: mapping from function name to kernel
    if (!func) {
      return absl::InvalidArgumentError("Function should not be null.");
    }
    if (!func->has_body()) {
      return absl::InvalidArgumentError("Function does not have body.");
    }
    if (!func->body().has_stable_hlo_body()) {
      return absl::InvalidArgumentError(
          "Function body must have `StableHloFunctionBody`");
    }
    // TODO b/401364010: Verify all have the same number of inputs and outputs.
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<tsl::ReadOnlyMemoryRegion>> MapAotMemoryRegion(
    absl::string_view model_path,
    const xla_saved_model_manifest::CompiledExecutableBody& compiled_exec) {
  if (!compiled_exec.has_aot_executable()) {
    return absl::InvalidArgumentError(
        "CompiledExecutableBody missing aot_executable.");
  }
  const auto& aot_exec = compiled_exec.aot_executable();
  if (!aot_exec.has_file_system_location()) {
    return absl::InvalidArgumentError(
        "aot_executable missing file_system_location.");
  }
  const auto& location = aot_exec.file_system_location();
  TF_ASSIGN_OR_RETURN(std::string file_path,
                      GetAbsoluteFileSystemPath(model_path, location));

  LOG(INFO) << "Mapping AOT executable file: " << file_path;

  std::unique_ptr<tsl::ReadOnlyMemoryRegion> memory_region;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(
      file_path, &memory_region));

  return std::shared_ptr<tsl::ReadOnlyMemoryRegion>(std::move(memory_region));
}

absl::StatusOr<const xla_saved_model_manifest::CompiledExecutableBody*>
SelectCompiledExecutable(const xla_saved_model_manifest::Function& func,
                         absl::string_view target_topology_label) {
  if (target_topology_label.empty()) {
    return absl::InvalidArgumentError(
        "target_topology_label is strictly required for AOT execution "
        "routing.");
  }
  for (const auto& compiled_exec :
       func.body().stable_hlo_body().compiled_executables()) {
    if (target_topology_label == compiled_exec.target_topology_label()) {
      return &compiled_exec;
    }
  }

  return absl::NotFoundError(
      absl::StrCat("No matching AOT executable found for topology: ",
                   target_topology_label));
}

absl::Status VerifyExecutableCompatibility(
    xla::ifrt::Client& ifrt_client,
    const xla_saved_model_manifest::CompiledExecutableBody& compiled_exec,
    xla::ifrt::DeviceListRef ifrt_device_list,
    xla::ifrt::Serialized* serialized_executable_version) {
  // An empty executable_version indicates that the exporter did not provide
  // executable ABI version metadata (e.g., from backends where versioning is
  // optional or unpopulated). Skip fast-path version compatibility checks.
  if (compiled_exec.executable_version().empty()) {
    LOG_FIRST_N(WARNING, 1)
        << "AOT executable_version is missing in the manifest. "
           "Skipping fast-path version compatibility checks.";
    return absl::OkStatus();
  }
  if (!serialized_executable_version->ParseFromString(
          compiled_exec.executable_version())) {
    return absl::InvalidArgumentError(
        "Failed to parse executable_version proto.");
  }
  auto deserialized_version =
      xla::ifrt::Deserialize<xla::ifrt::ExecutableVersion>(
          *serialized_executable_version,
          std::make_unique<xla::ifrt::DeserializeOptions>());
  if (!deserialized_version.ok()) {
    return deserialized_version.status();
  }
  absl::Status compat_status =
      ifrt_client.GetDefaultCompiler()->IsExecutableVersionCompatible(
          **deserialized_version, ifrt_device_list);
  if (absl::IsUnimplemented(compat_status)) {
    VLOG(2) << "IFRT client skipped version compatibility checks "
               "(unimplemented).";
    return absl::OkStatus();
  }
  return compat_status;
}

bool ShouldShloRunOnHost(
    absl::Span<const xla_saved_model_manifest::Function* absl_nonnull const>
        funcs,
    absl::string_view platform_name) {
  if (absl::EqualsIgnoreCase(platform_name, "cpu")) {
    return true;
  }
  for (auto* func : funcs) {
    const google::protobuf::RepeatedPtrField<std::string>& platforms =
        func->body().stable_hlo_body().lowering_platforms();
    if (platforms.empty()) {
      return false;
    }
    for (const auto& platform : platforms) {
      if (platform == "tpu" || platform == "cuda" || platform == "rocm") {
        return false;
      }
    }
  }
  return true;
}

absl::StatusOr<std::string> GetKeyFromProcessorMimeType(
    const xla_saved_model_manifest::Function& func) {
  if (func.body().other().has_mime_type()) {
    return std::string(func.body().other().mime_type());
  }
  return absl::InvalidArgumentError(
      "Optional function handler does not have mime type needed for function "
      "routing.");
}

void CreateAuxiliaryComponents(
    xla_saved_model_orchestration::Pipeline& pipeline) {
  for (const auto& pre_processor_name : pipeline.pre_processor_names()) {
    xla_saved_model_orchestration::Component* component =
        pipeline.add_components();
    component->set_function_name(pre_processor_name);
    component->set_role(xla_saved_model_orchestration::ROLE_PREPROCESSOR);
  }
  for (const auto& model_function : pipeline.model_functions()) {
    xla_saved_model_orchestration::Component* component =
        pipeline.add_components();
    component->set_function_name(model_function.model_function_name());
    component->set_role(xla_saved_model_orchestration::ROLE_MODEL);
    if (model_function.has_weights_name()) {
      component->set_weights_name(model_function.weights_name());
    }
  }
  for (const auto& post_processor_name : pipeline.post_processor_names()) {
    xla_saved_model_orchestration::Component* component =
        pipeline.add_components();
    component->set_function_name(post_processor_name);
    component->set_role(xla_saved_model_orchestration::ROLE_POSTPROCESSOR);
  }
}

// Build a BatchKernel from a list of kernels to be batched.

absl::StatusOr<std::unique_ptr<Kernel>> BuildBatchKernel(
    std::vector<std::unique_ptr<Kernel>> batched_kernels,
    const xla_saved_model_orchestration::Pipeline& pipeline,
    const InterpreterCreateOptions* create_options,
    tsl::thread::ThreadPool* thread_pool) {
  LOG(INFO) << "Building the batch kernel.";
  bool has_streaming = false;
  const BatchOptions& batch_options = pipeline.batch_options();
  for (const auto& kernel : batched_kernels) {
    if (kernel->has_streaming()) {
      has_streaming = true;
      break;
    }
  }
  if (has_streaming) {
    LOG(INFO) << "Streaming detected in batched kernels. "
              << "Forcing disable_large_batch_splitting to true.";
  }

  std::vector<MetaArgSpec> input_signature;
  if (!batched_kernels.empty() &&
      batched_kernels.front()->input_signature() != nullptr) {
    input_signature = batched_kernels.front()->input_signature()->specs();
  }
  std::vector<MetaArgSpec> output_signature;
  if (!batched_kernels.empty() &&
      batched_kernels.back()->output_signature() != nullptr) {
    output_signature = batched_kernels.back()->output_signature()->specs();
  }
  BatchOptions::BatchPaddingPolicy batch_padding_policy =
      batch_options.batch_padding_policy();
  TF_ASSIGN_OR_RETURN(std::string batch_padding_policy_string,
                      GetBatchPaddingPolicyString(batch_padding_policy));
  TF_ASSIGN_OR_RETURN(
      MixedPriorityBatchingPolicy mixed_priority_batching_policy,
      GetMixedPriorityBatchingPolicy(
          batch_options.mixed_priority_batching_policy()));
  TF_ASSIGN_OR_RETURN(
      PriorityBatchingPolicy priority_batching_policy,
      GetPriorityBatchingPolicy(batch_options.priority_batching_policy()));

  BatchRunnerOpOptions batch_runner_op_options = {
      .input_signature = std::move(input_signature),
      .output_signature = std::move(output_signature),

      // Allow overriding  `num_batch_threads` and `max_enqueued_batches` from
      // the create options.
      .num_batch_threads = std::max(batch_options.num_batch_threads(),
                                    create_options->min_num_batch_threads),
      .max_enqueued_batches =
          std::max(batch_options.max_enqueued_batches(),
                   create_options->min_max_enqueued_batches),
      .max_batch_size = batch_options.max_batch_size(),
      .batch_timeout_micros = batch_options.batch_timeout_micros(),
      .allowed_batch_sizes =
          std::vector<int32_t>{batch_options.allowed_batch_sizes().begin(),
                               batch_options.allowed_batch_sizes().end()},
      .enable_large_batch_splitting =
          !batch_options.disable_large_batch_splitting() && !has_streaming,
      .batch_padding_policy = std::move(batch_padding_policy_string),
      .mixed_priority_batching_policy = mixed_priority_batching_policy,
      .priority_batching_policy = priority_batching_policy,
      .allocator = create_options->allocator,
      .ragged_inputs =
          [&]() {
            std::vector<RaggedInputSpec> result;
            if (pipeline.has_metadata_for_batching()) {
              const auto& metadata = pipeline.metadata_for_batching();
              result.reserve(metadata.ragged_inputs_size());
              for (const auto& ragged_input : metadata.ragged_inputs()) {
                result.push_back({
                    .flat_values_index = ragged_input.flat_values_index(),
                    .row_splits_index = ragged_input.row_splits_index(),
                });
              }
            }
            return result;
          }(),
  };
  if (batch_options.has_low_priority_batch_options()) {
    const BatchOptions::LowPriorityBatchOptions& low_priority_batch_options =
        batch_options.low_priority_batch_options();
    batch_runner_op_options.low_priority_max_batch_size =
        low_priority_batch_options.max_batch_size();
    batch_runner_op_options.low_priority_batch_timeout_micros =
        low_priority_batch_options.batch_timeout_micros();
    batch_runner_op_options.low_priority_max_enqueued_batches =
        low_priority_batch_options.max_enqueued_batches();
    batch_runner_op_options.low_priority_allowed_batch_sizes =
        std::vector<int32_t>{
            low_priority_batch_options.allowed_batch_sizes().begin(),
            low_priority_batch_options.allowed_batch_sizes().end()};
  }
  if (batch_options.has_priority_aware_batching_options()) {
    TF_ASSIGN_OR_RETURN(
        batch_runner_op_options.priority_aware_batching_options,
        GetPriorityAwareBatchingOptions(
            batch_options.priority_aware_batching_options()));
  }

  return CreateBatchRunnerOp(std::move(batched_kernels),
                             batch_runner_op_options, thread_pool);
}

// TODO: Merge this function with
// GetStableHloFunctionsByObjectName in manifest_utils.cc.
absl::StatusOr<const xla_saved_model_manifest::Function*> GetFunctionByObjectName(
    const absl::flat_hash_map<std::string,
                              const xla_saved_model_manifest::TopLevelObject*>&
        manifest_objects,
    absl::string_view function_object_name) {
  if (!manifest_objects.contains(function_object_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to find function object with name: \"",
                     function_object_name, "\""));
  }
  const xla_saved_model_manifest::TopLevelObject* function_obj =
      manifest_objects.at(function_object_name);

  if (!function_obj->has_function() ||
      !function_obj->function().has_signature()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Function object: ", function_object_name,
                     " has no function or the function has no signature."));
  }
  return &function_obj->function();
}

// Holds batching configuration for different components of a pipeline.
// The values are derived from `BatchOptions::BatchComponent` in the pipeline
// configuration:
// - `batch_preprocessing` is true if batch_component is
//   `PRE_PROCESSOR_AND_MODEL_FUNCTION` or `WHOLE_PIPELINE`.
// - `batch_model` is true if batch_component is `MODEL_FUNCTION`,
//   `PRE_PROCESSOR_AND_MODEL_FUNCTION`, `MODEL_FUNCTION_AND_POST_PROCESSOR`,
//   or `WHOLE_PIPELINE`.
// - `batch_postprocessing` is true if batch_component is
//   `MODEL_FUNCTION_AND_POST_PROCESSOR` or `WHOLE_PIPELINE`.
struct BatchingConfig {
  // Whether batching is enabled for the preprocessing component.
  bool batch_preprocessing;
  // Whether batching is enabled for the model component.
  bool batch_model;
  // Whether batching is enabled for the postprocessing component.
  bool batch_postprocessing;
};

BatchingConfig GetBatchingConfig(
    bool enable_batching,
    const xla_saved_model_orchestration::Pipeline& pipeline) {
  BatchOptions::BatchComponent batch_component =
      BatchOptions::BATCH_COMPONENT_UNSPECIFIED;
  if (enable_batching && pipeline.has_batch_options()) {
    const BatchOptions& batch_options = pipeline.batch_options();
    LOG(INFO) << "Batch options: " << batch_options;
    batch_component = batch_options.batch_component();
  }

  bool batch_preprocessing =
      batch_component == BatchOptions::PRE_PROCESSOR_AND_MODEL_FUNCTION ||
      batch_component == BatchOptions::WHOLE_PIPELINE;
  bool batch_model =
      batch_component == BatchOptions::MODEL_FUNCTION ||
      batch_component == BatchOptions::PRE_PROCESSOR_AND_MODEL_FUNCTION ||
      batch_component == BatchOptions::MODEL_FUNCTION_AND_POST_PROCESSOR ||
      batch_component == BatchOptions::WHOLE_PIPELINE;
  bool batch_postprocessing =
      batch_component == BatchOptions::MODEL_FUNCTION_AND_POST_PROCESSOR ||
      batch_component == BatchOptions::WHOLE_PIPELINE;
  return {batch_preprocessing, batch_model, batch_postprocessing};
}

// Normalizes the pipeline in place.
// 1. If the pipeline relies on legacy fields (e.g., pre_processor_names), it
//    populates the `components` field using them via CreateAuxiliaryComponents.
// 2. Returns an error if the pipeline has no components even after attempts to
//    normalize.
// 3. Sets the `use_batching` flag for each component based on its role and the
//    batching configuration.
absl::Status NormalizePipeline(xla_saved_model_orchestration::Pipeline& pipeline,
                       bool enable_batching) {
  if (pipeline.components().empty()) {
    CreateAuxiliaryComponents(pipeline);
  }

  if (pipeline.components().empty()) {
    return absl::InvalidArgumentError(
        "Pipeline does not contain any component.");
  }

  BatchingConfig batching_config = GetBatchingConfig(enable_batching, pipeline);
  for (auto& component : *pipeline.mutable_components()) {
    bool use_batching = false;
    switch (component.role()) {
      case xla_saved_model_orchestration::ROLE_PREPROCESSOR:
        use_batching = batching_config.batch_preprocessing;
        break;
      case xla_saved_model_orchestration::ROLE_MODEL:
        use_batching = batching_config.batch_model;
        break;
      case xla_saved_model_orchestration::ROLE_POSTPROCESSOR:
        use_batching = batching_config.batch_postprocessing;
        break;
      default:
        break;
    }
    if (use_batching) {
      component.set_use_batching(true);
    }
  }
  return absl::OkStatus();
}
// Check the connection between kernels in the graph. If kernel `b` follows
// kernel `a` in the graph, then
// - the number of inputs of kernel `b` must match the number of outputs of
// kernel `a`, and
// - the input data type of kernel `b` must match the output data type of kernel
// `a`.
absl::Status ValidateGraphConnections(
    absl::Span<const std::unique_ptr<Kernel>> graph) {
  for (int i = 1; i < graph.size(); ++i) {
    const std::unique_ptr<Kernel>& kernel = graph[i];
    const std::unique_ptr<Kernel>& prev_kernel = graph[i - 1];
    if (kernel->GetNumInputs() != prev_kernel->GetNumOutputs()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Kernel ", i, " has ", kernel->GetNumInputs(),
          " inputs, which does not match the number of outputs of the "
          "previous kernel ",
          i - 1, " which has ", prev_kernel->GetNumOutputs(), " outputs."));
    }
    if (kernel->GetIODataTypes().input !=
        prev_kernel->GetIODataTypes().output) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Kernel ", i, " has input data type ", kernel->GetIODataTypes().input,
          ", which does not match the output data type of the previous kernel ",
          i - 1, " which has output data type ",
          prev_kernel->GetIODataTypes().output, "."));
    }
  }
  return absl::OkStatus();
}

// Validates the output flat signature meets the requirements for map return.
absl::Status ValidateMapReturnIsAllowed(
    const FlatSignature& output_flat_signature) {
  if (!output_flat_signature.IsAllSpecWithName()) {
    return absl::FailedPreconditionError(
        "Not all output specs have names so map return is not supported for "
        "this pipeline. Change the pipeline's signature to have names for all "
        "outputs, e.g., use dict-structured outputs.");
  }
  return absl::OkStatus();
}

// Loads the input signature from the given pipeline.
absl::StatusOr<FlatSignature> LoadPipelineInputSignature(
    const xla_saved_model_orchestration::Pipeline& pipeline) {
  if (pipeline.has_named_signature() &&
      pipeline.named_signature().has_inputs()) {
    // Get the flat signature from the `NamedSignature` in orchestration.
    return GetFlatSignatureFromNamedSignature(
        pipeline.named_signature().inputs());
  }
  // Otherwise, get the flat signature from the `Signature` in orchestration
  // without tensor names.
  return GetInputFlatSignature(pipeline.signature().input());
}

// Loads the output signature from the given pipeline.
absl::StatusOr<FlatSignature> LoadPipelineOutputSignature(
    const xla_saved_model_orchestration::Pipeline& pipeline) {
  if (pipeline.has_named_signature() &&
      pipeline.named_signature().has_outputs()) {
    // Get the flat signature from the `NamedSignature` in orchestration.
    return GetFlatSignatureFromNamedSignature(
        pipeline.named_signature().outputs());
  }
  // Otherwise, get the flat signature from the `Signature` in orchestration
  // without tensor names.
  return GetOutputFlatSignature(pipeline.signature().output());
}

absl::StatusOr<std::variant<std::string, absl::string_view>>
GetStableHloMlirString(
    absl::string_view model_path,
    const xla_saved_model_manifest::StableHloFunctionBody& stable_hlo_body) {
  switch (stable_hlo_body.stable_hlo().data_case()) {
    case xla_saved_model_manifest::UnstructuredData::kInlinedBytes:
      return std::variant<std::string, absl::string_view>(
          absl::string_view(stable_hlo_body.stable_hlo().inlined_bytes()));
    case xla_saved_model_manifest::UnstructuredData::kFileSystemLocation: {
      // The file system location should contain a relative path with respect to
      // the `model_path`.
      std::string mlir_path = tsl::io::JoinPath(
          model_path,
          stable_hlo_body.stable_hlo().file_system_location().string_path());
      std::string mlir_module_string;
      absl::Status status = tsl::ReadFileToString(
          tsl::Env::Default(), mlir_path, &mlir_module_string);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to read StableHlo MLIR module from file: "
                   << status;
        return status;
      }
      return std::variant<std::string, absl::string_view>(
          std::move(mlir_module_string));
    }
    default:
      return absl::InvalidArgumentError(
          "StableHlo function body does not contain inlined bytes or file "
          "system location.");
  }
}

// Builds an in-memory representation `StableHlo` of StableHLOs from
// proto representations in `Function`s.
absl::StatusOr<std::unique_ptr<StableHlo>> BuildStableHloFromFunctions(
    absl::string_view model_path,
    absl::Span<const xla_saved_model_manifest::Function* const>
        stable_hlo_funcs) {
  std::vector<StableHlo::Spec> stablehlo_specs;
  stablehlo_specs.reserve(stable_hlo_funcs.size());
  for (const auto& func : stable_hlo_funcs) {
    TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> input_specs, GetInputSpec(*func));
    TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> output_specs,
                        GetOutputSpec(*func));
    TF_RETURN_IF_ERROR(CheckMimeAndDataType(
        func->body().stable_hlo_body().stable_hlo(), {kStableHloMimeType},
        {xla_saved_model_manifest::UnstructuredData::kInlinedBytes,
         xla_saved_model_manifest::UnstructuredData::kFileSystemLocation},
        "StableHlo MLIR module"));
    std::variant<std::string, absl::string_view> mlir_module;
    TF_ASSIGN_OR_RETURN(
        mlir_module,
        GetStableHloMlirString(model_path, func->body().stable_hlo_body()));
    stablehlo_specs.push_back(StableHlo::Spec{
        .inputs = std::move(input_specs),
        .outputs = std::move(output_specs),
        .mlir_module = std::move(mlir_module),
        .mlir_module_kept_var_idx =
            {func->body().stable_hlo_body().module_kept_var_idx().begin(),
             func->body().stable_hlo_body().module_kept_var_idx().end()},
        .target_platforms =
            {func->body().stable_hlo_body().lowering_platforms().begin(),
             func->body().stable_hlo_body().lowering_platforms().end()},
    });
  }
  return StableHlo::Create(stablehlo_specs);
}

// Loads the signature from the given pipeline.
absl::StatusOr<FunctionFlatSignature> LoadPipelineSignature(
    const xla_saved_model_orchestration::Pipeline& pipeline) {
  TF_ASSIGN_OR_RETURN(FlatSignature input_flat_signature,
                      LoadPipelineInputSignature(pipeline));
  if (input_flat_signature.IsAllSpecWithName()) {
    TF_ASSIGN_OR_RETURN(std::vector<absl::string_view> input_names,
                        input_flat_signature.GetAllNames());
    LOG(INFO) << "Input flat signature created. Interpreter will allow `Run` "
                 "with input names: "
              << absl::StrJoin(input_names, ",");
  } else {
    LOG(INFO) << "Input flat signature created but has no names. Interpreter "
                 "will not allow `Run` with input names.";
  }
  TF_ASSIGN_OR_RETURN(FlatSignature output_flat_signature,
                      LoadPipelineOutputSignature(pipeline));
  if (output_flat_signature.IsAllSpecWithName()) {
    TF_ASSIGN_OR_RETURN(std::vector<absl::string_view> output_names,
                        output_flat_signature.GetAllNames());
    LOG(INFO) << "Output flat signature created. Interpreter will allow "
                 "`RunWithMapReturn` with output names: "
              << absl::StrJoin(output_names, ",");
  } else {
    LOG(INFO) << "Output flat signature created but has no names. Interpreter "
                 "will not allow `RunWithMapReturn` with output names.";
  }
  return FunctionFlatSignature{
      .input_signature = std::move(input_flat_signature),
      .output_signature = std::move(output_flat_signature),
  };
}

// Creates a StableHloRunnerOp kernel for executing the given StableHLO
// function(s). When there are multiple StableHLO functions, they are
// concretized StableHLOs for a single polymorphic function.
absl::StatusOr<std::unique_ptr<Kernel>> CreateStableHloRunnerOp(
    absl::Span<const xla_saved_model_manifest::Function* const> stable_hlo_funcs,
    const xla_saved_model_manifest::Manifest& manifest,
    const std::vector<std::string>& weight_names,
    const xla_saved_model_orchestration::NamedTensorTypeList* named_inputs,
    const xla_saved_model_orchestration::NamedTensorTypeList* named_outputs,
    xla::CompileOptionsProto* absl_nullable compile_options,
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    const InterpreterCreateOptions& create_options,
    tsl::thread::ThreadPool* computation_thread_pool,
    IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
    std::optional<int64_t> callback_id) {
  KernelBuilderContext builder;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<StableHlo> stablehlo,
      BuildStableHloFromFunctions(create_options.model_path, stable_hlo_funcs));
  TF_RETURN_IF_ERROR(builder.SetConfig(
      kStableHlo, std::shared_ptr<StableHlo>(std::move(stablehlo))));
  if (named_inputs != nullptr) {
    TF_ASSIGN_OR_RETURN(FlatSignature input_flat_signature,
                        GetFlatSignatureFromNamedSignature(*named_inputs));
    TF_RETURN_IF_ERROR(builder.SetConfig(kFunctionFlatInputSignature,
                                         std::move(input_flat_signature)));
  }
  if (named_outputs != nullptr) {
    TF_ASSIGN_OR_RETURN(FlatSignature output_flat_signature,
                        GetFlatSignatureFromNamedSignature(*named_outputs));
    TF_RETURN_IF_ERROR(builder.SetConfig(kFunctionFlatOutputSignature,
                                         std::move(output_flat_signature)));
  }
  TF_RETURN_IF_ERROR(builder.SetConfig(kIfrtClient, ifrt_client));
  TF_RETURN_IF_ERROR(
      builder.SetConfig(kPersistentCompilationCache,
                        create_options.persistent_compilation_cache.get()));
  TF_RETURN_IF_ERROR(
      builder.SetConfig(kIfrtWorkerThreadPool, computation_thread_pool));
  TF_RETURN_IF_ERROR(builder.SetConfig(kIfrtLoadedVariableRegistry,
                                       ifrt_loaded_variable_registry));
  TF_RETURN_IF_ERROR(builder.SetConfig(kWeights, weight_names));
  TF_RETURN_IF_ERROR(
      builder.SetConfig(kCompilationEnvProto, create_options.compilation_env));
  if (compile_options != nullptr) {
    TF_RETURN_IF_ERROR(builder.SetConfig(kXlaCompileOptions, *compile_options));
  }
  TF_RETURN_IF_ERROR(builder.SetConfig(kServingCoreSelector,
                                       create_options.serving_core_selector));
  TF_RETURN_IF_ERROR(
      builder.SetConfig(kStablehloOptimizer, create_options.optimizer.get()));

  // TODO b/401364010: It is fine to use the first function to infer the
  // assigned devices. We will rely on the `manifest` to get the assigned
  // devices, rather than using `func` to infer the assigned devices. Remove
  // `func` from the input argument list once assigned devices are available in
  // the manifest.
  TF_ASSIGN_OR_RETURN(
      AssignedDevices assigned_devices,
      GetAssignedDevices(manifest, *stable_hlo_funcs[0], compile_options,
                         create_options.ignore_device_assignment_in_manifest));
  TF_RETURN_IF_ERROR(builder.SetConfig(
      kAssignedDevices,
      std::make_shared<const AssignedDevices>(std::move(assigned_devices))));
  if (callback_id.has_value()) {
    TF_RETURN_IF_ERROR(builder.SetConfig(kStableCallbackId, *callback_id));
  }

  return GetGlobalKernelRegistry().Create("StableHloRunnerOp", builder);
}

struct NamedSignature {
  const xla_saved_model_orchestration::NamedTensorTypeList* inputs = nullptr;
  const xla_saved_model_orchestration::NamedTensorTypeList* outputs = nullptr;
};

NamedSignature GetNamedSignatureOfComponent(
    const xla_saved_model_orchestration::Component& component) {
  if (component.has_named_signature()) {
    return {.inputs = &component.named_signature().inputs(),
            .outputs = &component.named_signature().outputs()};
  }
  return {};
}

absl::Status ValidateInterpreterCreateOptions(
    const InterpreterCreateOptions& create_options) {
  if (create_options.model_path.empty()) {
    return absl::InvalidArgumentError("Model path must not be empty.");
  }

  if (create_options.execute_worker_thread_pool == nullptr) {
    return absl::InvalidArgumentError(
        "Execute worker thread pool must not be null.");
  }
  if (create_options.ifrt_client == nullptr) {
    return absl::InvalidArgumentError("Ifrt client must not be null.");
  }
  if (create_options.computation_thread_pool == nullptr) {
    return absl::InvalidArgumentError(
        "Computation thread pool must not be null.");
  }
  return absl::OkStatus();
}

// Modular execution bucket loader
absl::StatusOr<BucketAotData> LoadAotExecutableForBucket(
    const xla_saved_model_manifest::Function& func,
    absl::string_view target_topology_label, absl::string_view component_name,
    absl::string_view model_path, const AssignedDevices& assigned_devices,
    xla::ifrt::Client* ifrt_client,
    std::vector<std::shared_ptr<tsl::ReadOnlyMemoryRegion>>& memory_regions) {
  TF_ASSIGN_OR_RETURN(const auto* compiled_exec,
                      SelectCompiledExecutable(func, target_topology_label));

  VLOG(1) << "Loading AOT compiled executable for " << component_name;

  TF_ASSIGN_OR_RETURN(std::shared_ptr<tsl::ReadOnlyMemoryRegion> memory_region,
                      MapAotMemoryRegion(model_path, *compiled_exec));
  absl::string_view serialized_executable(
      static_cast<const char*>(memory_region->data()), memory_region->length());
  memory_regions.push_back(memory_region);

  std::vector<int> aot_kept_var_idx(compiled_exec->aot_kept_var_idx().begin(),
                                    compiled_exec->aot_kept_var_idx().end());

  TF_ASSIGN_OR_RETURN(auto input_specs, GetInputSpec(func));
  TF_ASSIGN_OR_RETURN(auto output_specs, GetOutputSpec(func));

  TF_RETURN_IF_ERROR(
      ValidateCompiledExecutable(*compiled_exec, input_specs.size()));

  StableHlo::Spec spec;
  spec.inputs = std::move(input_specs);
  spec.outputs = std::move(output_specs);
  spec.mlir_module_kept_var_idx = std::move(aot_kept_var_idx);
  spec.target_platforms.reserve(
      func.body().stable_hlo_body().lowering_platforms().size());
  for (const auto& platform :
       func.body().stable_hlo_body().lowering_platforms()) {
    spec.target_platforms.push_back(platform);
  }

  TF_ASSIGN_OR_RETURN(
      auto assigned_ifrt_devices,
      GetAssignedIfrtDevices(*ifrt_client, assigned_devices.num_replicas,
                             assigned_devices.num_cores_per_replica,
                             assigned_devices.device_assignment));

  TF_ASSIGN_OR_RETURN(auto ifrt_device_list,
                      ifrt_client->MakeDeviceList(assigned_ifrt_devices));

  xla::ifrt::Serialized serialized_executable_version;
  absl::Status compat_status = VerifyExecutableCompatibility(
      *ifrt_client, *compiled_exec, ifrt_device_list,
      &serialized_executable_version);
  if (!compat_status.ok()) {
    LOG(WARNING) << "AOT executable version is not compatible for "
                 << component_name << ": " << compat_status.message();
    return compat_status;
  }

  auto deserialize_options =
      std::make_unique<xla::ifrt::XlaDeserializeExecutableOptions>(
          /*compile_options=*/std::nullopt, ifrt_device_list);

  absl::Cord serialized_executable_cord = absl::MakeCordFromExternal(
      serialized_executable, [memory_region](absl::string_view) {
        // Captures memory_region (shared_ptr) to keep the mmap alive zero-copy.
      });
  auto loaded_executable_future =
      ifrt_client->GetDefaultCompiler()->DeserializeLoadedExecutable(
          serialized_executable_cord, std::move(deserialize_options));

  auto executable_ref = loaded_executable_future.Await();
  if (!executable_ref.ok()) return executable_ref.status();

  if (!(*executable_ref)->devices().has_value()) {
    int total_devices =
        assigned_devices.num_replicas * assigned_devices.num_cores_per_replica;
    if (total_devices > 1) {
      return absl::InvalidArgumentError(
          "Invalid AOT payload: The XLA compiler strictly requires "
          "multi-device executables to be bound to a specific device "
          "assignment, but the loaded artifact is portable.");
    }
  }
  return BucketAotData{std::move(*executable_ref), std::move(spec)};
}

struct BatchIndices {
  int start;
  int end;
};

// Finds the start and end indices of consecutive batched components in a
// pipeline. Returns an error if batched components are non-consecutive.
absl::StatusOr<std::optional<BatchIndices>> FindBatchIndices(
    const google::protobuf::RepeatedPtrField<xla_saved_model_orchestration::Component>&
        components) {
  std::optional<BatchIndices> indices;
  bool batch_completed = false;
  for (int i = 0; i < components.size(); ++i) {
    const auto& component = components[i];
    if (component.use_batching()) {
      // If we already completed a batch segment and find another batched
      // component, it means they are non-consecutive.
      if (batch_completed) {
        return absl::InvalidArgumentError(
            "Non-consecutive batched components are not supported.");
      }
      // Mark the start of the batched segment if not already set.
      if (!indices.has_value()) {
        indices = BatchIndices{.start = i, .end = i};
      } else {
        indices->end = i;
      }
    } else {
      // If we are on a non-batched component but have started a batch segment,
      // mark it as completed.
      if (indices.has_value()) {
        batch_completed = true;
      }
    }
  }
  return indices;
}
}  // namespace

absl::StatusOr<std::unique_ptr<Interpreter>> Interpreter::Create(
    std::unique_ptr<InterpreterCreateOptions> create_options) {
  if (create_options == nullptr) {
    return absl::InvalidArgumentError("create_options must not be null.");
  }
  TF_RETURN_IF_ERROR(ValidateInterpreterCreateOptions(*create_options));

  // Call Bundle Interpreter if it's a bundle
  if (BundleInterpreter::IsBundle(create_options->model_path)) {
    return BundleInterpreter::Create(std::move(create_options));
  }
  std::unique_ptr<Interpreter> interpreter =
      absl::WrapUnique(new Interpreter(std::move(create_options)));
  // TODO: make the load method static so it can be called before
  // the constructor and simplify the constructor.
  absl::Status status = interpreter->Load();
  model_loading_count
      ->GetCell(interpreter->create_options_->model_name,
                absl::StrCat(interpreter->create_options_->model_version),
                absl::StatusCodeToString(status.code()))
      ->IncrementBy(1);
  TF_RETURN_IF_ERROR(status);

  model_loaded_count
      ->GetCell(interpreter->create_options_->model_name,
                absl::StrCat(interpreter->create_options_->model_version))
      ->Increment();

  return interpreter;
}

Interpreter::Interpreter(
    std::unique_ptr<InterpreterCreateOptions> create_options)
    : create_options_(std::move(create_options)) {
  absl::BitGen gen;
  callback_id_ = absl::Uniform(gen, 0LL, std::numeric_limits<int64_t>::max());
}

std::vector<std::string> Interpreter::GetPipelineNames() const {
  std::vector<std::string> pipeline_names;
  pipeline_names.reserve(manifest_.objects_size());
  for (const auto& [pipeline_name, unused] : graph_bundles_) {
    pipeline_names.push_back(pipeline_name);
  }
  return pipeline_names;
}

absl::StatusOr<FunctionFlatSignature> Interpreter::GetPipelineSignature(
    absl::string_view pipeline_name) const {
  TF_ASSIGN_OR_RETURN(const GraphBundle* graph_bundle,
                      GetGraphBundle(pipeline_name));
  return graph_bundle->signature;
}

absl::StatusOr<xla_saved_model_orchestration::Pipeline>
Interpreter::GetPipelineProto(absl::string_view pipeline_name) const {
  TF_ASSIGN_OR_RETURN(const GraphBundle* graph_bundle,
                      GetGraphBundle(pipeline_name));
  return graph_bundle->pipeline;
}

absl::Status Interpreter::Load() {
  TF_ASSIGN_OR_RETURN(manifest_, GetManifest(create_options_->model_path));

  for (const auto& object : manifest_.objects()) {
    manifest_objects_[object.first] = &object.second;
  }

  TF_ASSIGN_OR_RETURN(
      xla_saved_model_orchestration::Pipelines pipelines,
      GetOexOrchestration(create_options_->model_path, manifest_));

  if (pipelines.name_to_pipeline().empty()) {
    return absl::InvalidArgumentError(
        "Model does not contain any pipeline. At least one pipeline is "
        "required.");
  }
  LOG(INFO) << "Found " << pipelines.name_to_pipeline_size()
            << " pipelines from Orbax model.";
  for (auto& [pipeline_name, pipeline] :
       *pipelines.mutable_name_to_pipeline()) {
    auto [it_graph_bundle, inserted] =
        graph_bundles_.try_emplace(pipeline_name);
    if (!inserted) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Pipeline with name \"", pipeline_name, "\" already exists."));
    }
    // Always load the signature as this is a cheap operation. This allows the
    // user to get the signature before loading the graph.
    LOG(INFO) << "Loading signature from pipeline: " << pipeline_name;
    TF_ASSIGN_OR_RETURN(FunctionFlatSignature signature,
                        LoadPipelineSignature(pipeline));
    TF_RETURN_IF_ERROR(
        NormalizePipeline(pipeline, create_options_->enable_batching));
    it_graph_bundle->second = std::make_unique<GraphBundle>(
        pipeline_name, std::move(pipeline), std::move(signature));
    GraphBundle& graph_bundle = *it_graph_bundle->second;
    graph_bundle.pipeline_name = pipeline_name;
    graph_bundle.keyword_arguments_execution =
        graph_bundle.pipeline.keyword_arguments_based();
  }
  if (create_options_->lazy_loading_pipelines) {
    return absl::OkStatus();
  }
  return LoadGraphs();
}

absl::Status Interpreter::LoadGraphs() {
  bool parallel_loading =
      absl::GetFlag(FLAGS_jsv_enable_parallel_loading_pipelines);
  LOG(INFO) << "Loading graphs from pipelines";
  if (parallel_loading) {
    static absl::NoDestructor<tsl::thread::ThreadPool> shared_loading_pool(
        tsl::Env::Default(), "JsvGraphLoaderPool", 8);
    tsl::BlockingCounter counter(graph_bundles_.size());
    for (auto& [pipeline_name, graph_bundle] : graph_bundles_) {
      LOG(INFO) << "Scheduling graph load from pipeline: " << pipeline_name;

      tsl::Context context(tsl::ContextKind::kThread);
      GraphBundle* bundle = graph_bundle.get();
      // Load all graphs eagerly and concurrently.
      shared_loading_pool->Schedule(
          [this, bundle,
           ctx = std::make_shared<tsl::Context>(std::move(context)),
           &counter]() mutable {
            {
              absl::MutexLock l(bundle->graph_mutex);
              tsl::WithContext wc(*ctx);
              bundle->graph.emplace(LoadGraph(bundle->pipeline));
            }
            counter.DecrementCount();
          });
    }
    counter.Wait();
    // Check the status of all graphs and return error if any graph fails to
    // load.
    for (auto& [pipeline_name, graph_bundle] : graph_bundles_) {
      absl::MutexLock l(graph_bundle->graph_mutex);
      TF_RETURN_IF_ERROR(graph_bundle->graph->status());
    }
  } else {
    // Sequentially load all graphs.
    for (auto& [pipeline_name, graph_bundle] : graph_bundles_) {
      absl::MutexLock l(graph_bundle->graph_mutex);
      TF_ASSIGN_OR_RETURN(graph_bundle->graph,
                          LoadGraph(graph_bundle->pipeline));
    }
  }
  LOG(INFO) << "Done loading graphs from pipelines";
  return absl::OkStatus();
}

// TODO: Support partial loading of per-function data.
absl::StatusOr<std::vector<std::string>> Interpreter::GetOrLoadWeightNames(
    absl::string_view weights_name,
    absl::Span<const xla_saved_model_manifest::Function* const> stablehlo_funcs,
    const AssignedDevices& assigned_devices,
    std::shared_ptr<xla::ifrt::Client> ifrt_client) {
  if (weights_name.empty()) {
    return std::vector<std::string>{};
  }
  absl::MutexLock l(weight_sets_mutex_);
  // It is possible that multiple pipelines share the set of weights. So it is
  // okay if `inserted` is true.
  auto [iter, inserted] = weight_sets_.try_emplace(weights_name);

  // We should always try to load the weights, since only used weights by
  // the provided `stablehlo_funcs` will be loaded, and `stablehlo_funcs`
  // across pipelines may need different weights in the same set.
  TF_ASSIGN_OR_RETURN(DataLoader::LoadResult load_result,
                      LoadWeights(weights_name, stablehlo_funcs,
                                  assigned_devices, ifrt_client));
  if (inserted) {
    LOG(INFO) << "Adding weight set \"" << iter->first << "\" with "
              << load_result.data_names.size() << " weights.";
    // `loaded_weight_names` is used below; we should make a copy
    iter->second = load_result.data_names;
  }
  return load_result.data_names;
}

absl::StatusOr<std::unique_ptr<Kernel>> Interpreter::LoadStableHloRunnerOp(
    absl::string_view model_function_name, absl::string_view weights_name,
    const xla_saved_model_orchestration::NamedTensorTypeList* named_inputs,
    const xla_saved_model_orchestration::NamedTensorTypeList* named_outputs,
    const std::string& platform_name) {
  TF_ASSIGN_OR_RETURN(
      std::vector<const xla_saved_model_manifest::Function*> stablehlo_funcs,
      GetStableHloFunctionsByObjectName(manifest_objects_,
                                        model_function_name));

  TF_RETURN_IF_ERROR(ValidateStableHlo(stablehlo_funcs));

  TF_ASSIGN_OR_RETURN(std::optional<xla::CompileOptionsProto> compile_options,
                      GetCompileOptions(stablehlo_funcs, platform_name));

  // This should never happen since we have a check earlier to ensure there is
  // at least one StableHLO function. Still keep this check here out of extra
  // caution.
  if (stablehlo_funcs.empty()) {
    return absl::InvalidArgumentError(
        "No StableHLO functions are found. At least one StableHLO function is "
        "required.");
  }
  // TODO: Remove this once the device assignment is passed from
  // the manifest.
  // When there are multiple StableHLO functions (i.e., concretized StableHLOs
  // for a polymorphic JAX function), we can use any of them to infer the
  // the device assignment, which is done based upon the input sharding that
  // should be the same across all concretized StableHLOs.
  TF_ASSIGN_OR_RETURN(
      AssignedDevices assigned_devices,
      GetAssignedDevices(
          manifest_, *stablehlo_funcs[0],
          compile_options.has_value() ? &*compile_options : nullptr,
          create_options_->ignore_device_assignment_in_manifest));

  std::shared_ptr<xla::ifrt::Client> ifrt_client =
      platform_name == "cpu" ? (create_options_->host_ifrt_client != nullptr
                                    ? create_options_->host_ifrt_client
                                    : create_options_->ifrt_client)
                             : create_options_->ifrt_client;

  TF_ASSIGN_OR_RETURN(std::vector<std::string> loaded_weight_names,
                      GetOrLoadWeightNames(weights_name, stablehlo_funcs,
                                           assigned_devices, ifrt_client));

  LOG(INFO) << "Adding StableHlo function: " << model_function_name;
  return CreateStableHloRunnerOp(
      stablehlo_funcs, manifest_, loaded_weight_names, named_inputs,
      named_outputs, compile_options.has_value() ? &*compile_options : nullptr,
      ifrt_client, *create_options_, create_options_->computation_thread_pool,
      &ifrt_loaded_variable_registry_, callback_id_);
}

absl::StatusOr<Interpreter::AotLoadResult> Interpreter::LoadAotExecutable(
    const xla_saved_model_orchestration::Component& component,
    absl::Span<const xla_saved_model_manifest::Function* const> stablehlo_funcs,
    absl::string_view target_topology_label) {
  if (stablehlo_funcs.empty()) {
    return absl::InternalError("stablehlo_funcs must not be empty.");
  }

  AotLoadResult result;

  // 1. Initial validation
  TF_ASSIGN_OR_RETURN(
      std::optional<xla::CompileOptionsProto> compile_options,
      GetCompileOptions(stablehlo_funcs, create_options_->platform_name));

  if (!compile_options.has_value()) {
    result.last_deserialization_error = absl::FailedPreconditionError(
        "AOT compilation options not found in manifest.");
    return result;
  }

  auto assigned_devices = GetAssignedDevices(
      manifest_, *stablehlo_funcs[0],
      compile_options.has_value() ? &*compile_options : nullptr,
      create_options_->ignore_device_assignment_in_manifest);
  if (!assigned_devices.ok()) {
    result.last_deserialization_error = assigned_devices.status();
    return result;
  }
  result.shared_assigned_devices =
      std::make_shared<const AssignedDevices>(*std::move(assigned_devices));

  // 2. Load Weights outside the bucket loop
  // TODO: Support multiple weight sets in AOT mode.
  TF_ASSIGN_OR_RETURN(
      result.loaded_weight_names,
      GetOrLoadWeightNames(component.weights_name(), stablehlo_funcs,
                           *result.shared_assigned_devices,
                           create_options_->ifrt_client));

  result.executables.reserve(stablehlo_funcs.size());
  result.aot_memory_regions.reserve(stablehlo_funcs.size());
  std::vector<StableHlo::Spec> aot_specs;
  aot_specs.reserve(stablehlo_funcs.size());

  // 3. Safely load executables per bucket
  for (const auto* func : stablehlo_funcs) {
    auto bucket_result = LoadAotExecutableForBucket(
        *func, target_topology_label, component.function_name(),
        create_options_->model_path, *result.shared_assigned_devices,
        create_options_->ifrt_client.get(), result.aot_memory_regions);

    if (!bucket_result.ok()) {
      result.last_deserialization_error = bucket_result.status();
      break;
    }
    result.executables.push_back(std::move(bucket_result->executable));
    aot_specs.push_back(std::move(bucket_result->spec));
  }

  // 4. Final validation block
  // TODO: Update 1:1 validation when exporting multiple JAX
  // modules as a single OBM model.
  if (result.executables.size() == stablehlo_funcs.size()) {
    auto aot_stablehlo = StableHlo::Create(std::move(aot_specs));
    if (!aot_stablehlo.ok()) {
      result.last_deserialization_error = aot_stablehlo.status();
    } else {
      result.aot_match_found = true;
      result.aot_stablehlo = *std::move(aot_stablehlo);
    }
  }

  return result;
}

absl::StatusOr<std::unique_ptr<Kernel>> Interpreter::LoadAotRunnerOp(
    AotLoadResult aot_result,
    const xla_saved_model_orchestration::NamedTensorTypeList* named_inputs,
    const xla_saved_model_orchestration::NamedTensorTypeList* named_outputs) {
  KernelBuilderContext builder;
  TF_RETURN_IF_ERROR(builder.SetConfig(kStableHlo, aot_result.aot_stablehlo));
  TF_RETURN_IF_ERROR(builder.SetConfig(kIfrtLoadedExecutables,
                                       std::move(aot_result.executables)));
  TF_RETURN_IF_ERROR(builder.SetConfig(
      kIfrtAotMemoryRegions, std::move(aot_result.aot_memory_regions)));

  TF_RETURN_IF_ERROR(
      builder.SetConfig(kIfrtClient, create_options_->ifrt_client));
  TF_RETURN_IF_ERROR(builder.SetConfig(
      kIfrtWorkerThreadPool, create_options_->computation_thread_pool));
  TF_RETURN_IF_ERROR(builder.SetConfig(kIfrtLoadedVariableRegistry,
                                       &ifrt_loaded_variable_registry_));
  TF_RETURN_IF_ERROR(
      builder.SetConfig(kWeights, aot_result.loaded_weight_names));
  TF_RETURN_IF_ERROR(
      builder.SetConfig(kAssignedDevices, aot_result.shared_assigned_devices));

  if (named_inputs != nullptr) {
    TF_ASSIGN_OR_RETURN(FlatSignature input_flat_signature,
                        GetFlatSignatureFromNamedSignature(*named_inputs));
    TF_RETURN_IF_ERROR(
        builder.SetConfig(kFunctionFlatInputSignature, input_flat_signature));
  }
  if (named_outputs != nullptr) {
    TF_ASSIGN_OR_RETURN(FlatSignature output_flat_signature,
                        GetFlatSignatureFromNamedSignature(*named_outputs));
    TF_RETURN_IF_ERROR(
        builder.SetConfig(kFunctionFlatOutputSignature, output_flat_signature));
  }

  return GetGlobalKernelRegistry().Create("StableHloRunnerOp", builder);
}

absl::StatusOr<std::unique_ptr<Kernel>> Interpreter::LoadModelRunnerOp(
    const xla_saved_model_orchestration::Component& component,
    const xla_saved_model_orchestration::NamedTensorTypeList* named_inputs,
    const xla_saved_model_orchestration::NamedTensorTypeList* named_outputs) {
  TF_ASSIGN_OR_RETURN(
      std::vector<const xla_saved_model_manifest::Function*> stablehlo_funcs,
      GetStableHloFunctionsByObjectName(manifest_objects_,
                                        component.function_name()));

  if (stablehlo_funcs.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "No StableHLO functions found for ", component.function_name()));
  }

  bool has_any_aot = false;
  for (const auto* func : stablehlo_funcs) {
    if (!func->body().stable_hlo_body().compiled_executables().empty()) {
      has_any_aot = true;
      break;
    }
  }

  if (has_any_aot) {
    TF_ASSIGN_OR_RETURN(
        AotLoadResult aot_result,
        LoadAotExecutable(component, stablehlo_funcs,
                          create_options_->target_topology_label));

    if (aot_result.aot_match_found) {
      return LoadAotRunnerOp(std::move(aot_result), named_inputs,
                             named_outputs);
    }

    if (create_options_->allow_jit_fallback_for_aot) {
      LOG(WARNING) << "AOT executable loading failed; falling back to JIT "
                      "compilation. Reason: "
                   << aot_result.last_deserialization_error;
    } else {
      return aot_result.last_deserialization_error.ok()
                 ? absl::FailedPreconditionError(
                       "AOT executable missing or Topology mismatch")
                 : aot_result.last_deserialization_error;
    }
  }

  return LoadStableHloRunnerOp(component.function_name(),
                               component.weights_name(), named_inputs,
                               named_outputs, create_options_->platform_name);
}

absl::StatusOr<std::unique_ptr<Kernel>> Interpreter::LoadOnHostFunction(
    const xla_saved_model_orchestration::Component& component,
    int component_index, int total_components, bool keyword_arguments_based) {
  LOG(INFO) << "Adding function: " << component.function_name();
  TF_ASSIGN_OR_RETURN(
      const xla_saved_model_manifest::Function* func,
      GetFunctionByObjectName(manifest_objects_, component.function_name()));

  NamedSignature named_signature = keyword_arguments_based
                                       ? GetNamedSignatureOfComponent(component)
                                       : NamedSignature{};

  if (func->body().has_stable_hlo_body()) {
    if (!ShouldShloRunOnHost({func}, create_options_->platform_name)) {
      std::string error_msg =
          absl::StrCat("StableHLO function ", component.function_name(),
                       " is not supported to run on host.");
      return absl::InvalidArgumentError(error_msg);
    }
    return LoadStableHloRunnerOp(
        component.function_name(), component.weights_name(),
        named_signature.inputs, named_signature.outputs,
        /*platform_name=*/"cpu");
  } else {
    // Handle non-StableHLO functions.
    TF_ASSIGN_OR_RETURN(std::string key, GetKeyFromProcessorMimeType(*func));
    TF_ASSIGN_OR_RETURN(OptionalFunctionHandler * handler,
                        optional_function_handler_router_.GetHandler(key));

    // By default, processor kernels convert inputs and outputs to/from
    // HostBuffer.
    // For an unbatched first component, we skip HostBuffer -> Tensor conversion
    // so clients can pass native framework tensors directly.
    bool pre_data_conversion = true;
    if (component_index == 0) {
      pre_data_conversion = component.use_batching();
    }
    // For an unbatched last component , we skip Tensor -> HostBuffer conversion
    // so the pipeline returns native framework tensors directly to clients.
    bool post_data_conversion = true;
    if (component_index == total_components - 1) {
      post_data_conversion = component.use_batching();
    }

    return handler->HandleFunction(create_options_->model_path, *func,
                                   *create_options_, pre_data_conversion,
                                   post_data_conversion, named_signature.inputs,
                                   named_signature.outputs);
  }
}


absl::StatusOr<KernelGraph> Interpreter::LoadGraphFromComponentProto(
    const xla_saved_model_orchestration::Pipeline& pipeline) {
  if (pipeline.components().empty()) {
    return absl::InvalidArgumentError(
        "Pipeline does not contain any component.");
  }

  const google::protobuf::RepeatedPtrField<xla_saved_model_orchestration::Component>&
      components = pipeline.components();

  bool keyword_arguments_based = pipeline.keyword_arguments_based();

  // Trace the batch indices of the components in the pipeline.
  TF_ASSIGN_OR_RETURN(std::optional<BatchIndices> batch_indices,
                      FindBatchIndices(components));

  std::vector<std::unique_ptr<Kernel>> loaded_kernels(components.size());
  // Explicitly start loading the on-device stablehlo runner op first to
  // accelerate the loading process.
  for (int component_idx = 0; component_idx < components.size();
       ++component_idx) {
    const auto& component = components[component_idx];
    if (component.role() == xla_saved_model_orchestration::ROLE_MODEL) {
      NamedSignature named_signature =
          keyword_arguments_based ? GetNamedSignatureOfComponent(component)
                                  : NamedSignature{};
      TF_ASSIGN_OR_RETURN(loaded_kernels[component_idx],
                          LoadModelRunnerOp(component, named_signature.inputs,
                                            named_signature.outputs));
    }
  }

  // Load host functions (processors, CPU SHLOs).
  int total_components = components.size();
  for (int component_idx = 0; component_idx < total_components;
       ++component_idx) {
    const auto& component = components[component_idx];

    if (component.role() != xla_saved_model_orchestration::ROLE_MODEL) {
      TF_ASSIGN_OR_RETURN(
          loaded_kernels[component_idx],
          LoadOnHostFunction(component, component_idx, total_components,
                             keyword_arguments_based));
    }
  }

  // Assemble final pipeline.
  KernelGraph graph;
  int component_idx = 0;
  while (component_idx < total_components) {
    if (batch_indices.has_value() && component_idx == batch_indices->start) {
      std::vector<std::unique_ptr<Kernel>> batch_segment;
      batch_segment.reserve(batch_indices->end - batch_indices->start + 1);
      for (int j = batch_indices->start; j <= batch_indices->end; ++j) {
        batch_segment.push_back(std::move(loaded_kernels[j]));
      }

      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<Kernel> batched_kernel,
          BuildBatchKernel(std::move(batch_segment), pipeline,
                           create_options_.get(),
                           create_options_->computation_thread_pool));
      graph.push_back(std::move(batched_kernel));
      component_idx = batch_indices->end;  // Skip batched components
    } else {
      graph.push_back(std::move(loaded_kernels[component_idx]));
    }
    ++component_idx;
  }

  if (graph.empty()) {
    return absl::InvalidArgumentError("The loaded graph is empty.");
  }
  // Validate graph connections by matching output size of kernel[i] to input
  // size of kernel[i+1]. This check is skipped for keyword-based graphs
  // because dependencies are resolved by name, and the number of inputs and
  // outputs between adjacent kernels may differ.
  if (!pipeline.keyword_arguments_based()) {
    TF_RETURN_IF_ERROR(ValidateGraphConnections(graph));
  }
  LOG(INFO) << "Finished loading the graph.";
  return graph;
}

// TODO: Add metric to monitor the model that use the legacy
// fields and remove this function once all models are migrated to the new
// components.
absl::StatusOr<KernelGraph> Interpreter::LoadGraph(
    const xla_saved_model_orchestration::Pipeline& pipeline) {
  return LoadGraphFromComponentProto(pipeline);
}

absl::StatusOr<std::vector<Value>> Interpreter::ConvertToVectorInputs(
    const GraphBundle& graph_bundle, absl::Span<const Value> positional_inputs,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs) {
  if (positional_inputs.empty() &&
      !graph_bundle.signature.input_signature.IsAllSpecWithName()) {
    return absl::FailedPreconditionError(
        "Not all input specs have names so cannot run this pipeline with a "
        "map. Change the pipeline's signature to have names for all inputs, "
        "e.g., use keyword-only arguments or dict-structured arguments.");
  }
  std::vector<Value> inputs(positional_inputs.begin(), positional_inputs.end());
  TF_RETURN_IF_ERROR(
      graph_bundle.signature.input_signature.FillArgs(inputs, keyword_inputs));
  return inputs;
}

absl::StatusOr<Interpreter::GraphBundle* absl_nonnull>
Interpreter::GetGraphBundle(absl::string_view pipeline_name) {
  absl::StatusOr<const GraphBundle*> graph_bundle =
      std::as_const(*this).GetGraphBundle(pipeline_name);
  TF_RETURN_IF_ERROR(graph_bundle.status());
  return const_cast<Interpreter::GraphBundle*>(*graph_bundle);
}

absl::StatusOr<const Interpreter::GraphBundle* absl_nonnull>
Interpreter::GetGraphBundle(absl::string_view pipeline_name) const {
  auto it = graph_bundles_.find(pipeline_name);
  if (it == graph_bundles_.end()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Pipeline name \"", pipeline_name,
                     "\" not found. Available pipelines: ",
                     absl::StrJoin(GetPipelineNames(), ", ")));
  }
  return it->second.get();
}

absl::StatusOr<DataLoader::LoadResult> Interpreter::LoadWeights(
    absl::string_view weight_set_name,
    absl::Span<const xla_saved_model_manifest::Function* const> stablehlo_funcs,
    const AssignedDevices& assigned_devices,
    std::shared_ptr<xla::ifrt::Client> ifrt_client) {
  const auto it = manifest_objects_.find(weight_set_name);
  if (it == manifest_objects_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Object key for weight names ", weight_set_name,
                     " not found in manifest."));
  }
  const xla_saved_model_manifest::Value& value = it->second->value();
  if (!value.has_external()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Weights name ", weight_set_name, " is not an external value."));
  }
  const xla_saved_model_manifest::ExternalValue& external = value.external();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<DataLoader> data_loader,
      DataLoaderFactoryRegistry::GetGlobalRegistry().CreateDataLoader(
          external.loader_type(),
          DataLoader::CreateOptions{
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry_,
              .ifrt_client = ifrt_client,
              .checkpoint_loader_thread_pool =
                  create_options_ != nullptr
                      ? create_options_->checkpoint_loader_thread_pool
                      : nullptr}));
  const bool is_portable_executable =
      create_options_ != nullptr &&
      create_options_->serving_core_selector != nullptr &&
      assigned_devices.num_replicas == 1 &&
      assigned_devices.num_cores_per_replica == 1;
  return data_loader->Load(external, create_options_->model_path,
                           is_portable_executable ? nullptr : &assigned_devices,
                           &manifest_, stablehlo_funcs);
}

absl::StatusOr<const KernelGraph* absl_nonnull> Interpreter::GetOrCreateGraph(
    GraphBundle& graph_bundle) {
  absl::MutexLock lock(graph_bundle.graph_mutex);
  if (!graph_bundle.graph.has_value()) {
    graph_bundle.graph.emplace(LoadGraph(graph_bundle.pipeline));
  }
  if (graph_bundle.graph->ok()) {
    return &(**graph_bundle.graph);
  }
  return graph_bundle.graph->status();
}

absl::StatusOr<std::vector<Value>> Interpreter::Run(
    absl::string_view pipeline_name, absl::Span<const Value> positional_inputs,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs,
    RunOptions run_options) {
  run_options.request_id = GenerateNextRequestId();
  VLOG(2) << "Interpreter::Run: pipeline_name=" << pipeline_name
          << ", model_name=" << create_options_->model_name;

  std::optional<ScopedStreamCallback> scoped_stream_callback;
  if (run_options.streamed_output_callback != nullptr &&
      callback_id_.has_value()) {
    TF_ASSIGN_OR_RETURN(StreamCallbackRegistry * stream_callback_registry,
                        GetGlobalStreamCallbackRegistry());
    TF_ASSIGN_OR_RETURN(
        scoped_stream_callback,
        stream_callback_registry->Register(
            create_options_->model_name, *callback_id_, run_options.request_id,
            run_options.streamed_output_callback));
  }

  auto result = [this, pipeline_name, &positional_inputs, &keyword_inputs,
                 run_options]() -> absl::StatusOr<std::vector<Value>> {
    TF_ASSIGN_OR_RETURN(GraphBundle * graph_bundle,
                        GetGraphBundle(pipeline_name));
    TF_ASSIGN_OR_RETURN(const KernelGraph* graph,
                        GetOrCreateGraph(*graph_bundle));
    std::vector<Placement> resolved_placements =
        graph_bundle->signature.output_signature.ResolvePlacements(
            run_options.output_placement_map);
    RunContext run_context = RunContext::Create(
        run_options, std::move(resolved_placements), callback_id_);

    if (graph_bundle->keyword_arguments_execution) {
      absl::flat_hash_map<std::string, Value> keyword_outputs;
      TF_ASSIGN_OR_RETURN(
          keyword_outputs,
          ExecuteLinearGraphWithKeywordArgs(std::move(run_context),
                                            keyword_inputs, *graph,
                                            create_options_->model_name,
                                            create_options_->model_version));
      TF_ASSIGN_OR_RETURN(
          std::vector<absl::string_view> pipeline_output_names,
          graph_bundle->signature.output_signature.GetAllNames());
      std::vector<Value> outputs;
      outputs.reserve(pipeline_output_names.size());
      for (const auto& name : pipeline_output_names) {
        auto it = keyword_outputs.find(name);
        if (it == keyword_outputs.end()) {
          return absl::InternalError(absl::StrCat(
              "Output with name \"", name, "\" not found in keyword outputs."));
        }
        outputs.push_back(std::move(it->second));
      }
      return outputs;
    }
    if (keyword_inputs.empty()) {
      return ExecuteLinearGraph(std::move(run_context), positional_inputs,
                                *graph, create_options_->model_name,
                                create_options_->model_version);
    }
    TF_ASSIGN_OR_RETURN(std::vector<Value> inputs,
                        ConvertToVectorInputs(*graph_bundle, positional_inputs,
                                              keyword_inputs));
    return ExecuteLinearGraph(std::move(run_context), inputs, *graph,
                              create_options_->model_name,
                              create_options_->model_version);
  }();

  request_count
      ->GetCell(create_options_->model_name,
                absl::StrCat(create_options_->model_version), "sync",
                absl::StatusCodeToString(result.status().code()))
      ->IncrementBy(1);
  return result;
}

absl::StatusOr<absl::flat_hash_map<std::string, Value>>
Interpreter::RunWithMapReturn(
    absl::string_view pipeline_name, absl::Span<const Value> positional_inputs,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs,
    RunOptions run_options) {
  TF_ASSIGN_OR_RETURN(GraphBundle * graph_bundle,
                      GetGraphBundle(pipeline_name));
  TF_RETURN_IF_ERROR(
      ValidateMapReturnIsAllowed(graph_bundle->signature.output_signature));
  TF_ASSIGN_OR_RETURN(
      std::vector<Value> outputs,
      Run(pipeline_name, positional_inputs, keyword_inputs, run_options));
  return graph_bundle->signature.output_signature.UnflattenToMap(
      std::move(outputs));
}

tsl::Future<std::vector<Value>> Interpreter::RunAsync(
    absl::string_view pipeline_name, absl::Span<const Value> positional_inputs,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs,
    RunOptions run_options) {
  run_options.request_id = GenerateNextRequestId();
  VLOG(2) << "Interpreter::RunAsync: pipeline_name=" << pipeline_name
          << ", model_name=" << create_options_->model_name;

  std::shared_ptr<ScopedStreamCallback> scoped_stream_callback;
  if (run_options.streamed_output_callback != nullptr &&
      callback_id_.has_value()) {
    absl::StatusOr<StreamCallbackRegistry*> stream_callback_registry =
        GetGlobalStreamCallbackRegistry();
    if (!stream_callback_registry.ok()) {
      return tsl::Future<std::vector<Value>>(stream_callback_registry.status());
    }
    absl::StatusOr<ScopedStreamCallback> registered_callback =
        (*stream_callback_registry)
            ->Register(create_options_->model_name, *callback_id_,
                       run_options.request_id,
                       run_options.streamed_output_callback);
    if (!registered_callback.ok()) {
      return tsl::Future<std::vector<Value>>(registered_callback.status());
    }
    scoped_stream_callback =
        std::make_shared<ScopedStreamCallback>(std::move(*registered_callback));
  }

  auto future = [this, pipeline_name, &positional_inputs, &keyword_inputs,
                 run_options]() {
    // TODO: type check inputs and outputs.
    absl::StatusOr<GraphBundle*> graph_bundle = GetGraphBundle(pipeline_name);
    if (!graph_bundle.ok()) {
      return tsl::Future<std::vector<Value>>(graph_bundle.status());
    }

    absl::StatusOr<const KernelGraph*> graph = GetOrCreateGraph(**graph_bundle);
    if (!graph.ok()) {
      return tsl::Future<std::vector<Value>>(graph.status());
    }

    std::vector<Placement> resolved_placements =
        (*graph_bundle)
            ->signature.output_signature.ResolvePlacements(
                run_options.output_placement_map);
    RunContext run_context = RunContext::Create(
        run_options, std::move(resolved_placements), callback_id_);

    if (keyword_inputs.empty()) {
      return AsyncExecuteLinearGraph(
          std::move(run_context), positional_inputs, **graph,
          create_options_->execute_worker_thread_pool,
          create_options_->model_name, create_options_->model_version);
    }
    absl::StatusOr<std::vector<Value>> inputs = ConvertToVectorInputs(
        **graph_bundle, positional_inputs, keyword_inputs);
    if (!inputs.ok()) {
      return tsl::Future<std::vector<Value>>(inputs.status());
    }
    return AsyncExecuteLinearGraph(std::move(run_context), *inputs, **graph,
                                   create_options_->execute_worker_thread_pool,
                                   create_options_->model_name,
                                   create_options_->model_version);
  }();

  future.OnReady([model_name = create_options_->model_name,
                  model_version = create_options_->model_version,
                  scoped_stream_callback](
                     const absl::StatusOr<std::vector<Value>>& result) {
    request_count
        ->GetCell(model_name, absl::StrCat(model_version), "async",
                  absl::StatusCodeToString(result.status().code()))
        ->IncrementBy(1);
  });

  return future;
}

tsl::Future<absl::flat_hash_map<std::string, Value>>
Interpreter::RunAsyncWithMapReturn(
    absl::string_view pipeline_name, absl::Span<const Value> positional_inputs,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs,
    RunOptions run_options) {
  absl::StatusOr<GraphBundle*> graph_bundle = GetGraphBundle(pipeline_name);
  if (!graph_bundle.ok()) {
    return tsl::Future<absl::flat_hash_map<std::string, Value>>(
        graph_bundle.status());
  }
  absl::Status allow_map_return =
      ValidateMapReturnIsAllowed((*graph_bundle)->signature.output_signature);
  if (!allow_map_return.ok()) {
    return tsl::Future<absl::flat_hash_map<std::string, Value>>(
        allow_map_return);
  }
  tsl::Future<std::vector<Value>> outputs =
      RunAsync(pipeline_name, positional_inputs, keyword_inputs, run_options);
  auto [promise, map_output_future] =
      tsl::MakePromise<absl::flat_hash_map<std::string, Value>>();
  outputs.OnReady(
      [output_flat_signature = (*graph_bundle)->signature.output_signature,
       output_promise = std::move(promise)](
          absl::StatusOr<std::vector<Value>> vector_outputs) mutable -> void {
        if (!vector_outputs.ok()) {
          output_promise.Set(vector_outputs.status());
          return;
        }
        output_promise.Set(
            output_flat_signature.UnflattenToMap(*std::move(vector_outputs)));
        return;
      });
  return map_output_future;
}

absl::StatusOr<KernelIODataType> Interpreter::GetModelInputDataType(
    absl::string_view pipeline_name) const {
  TF_ASSIGN_OR_RETURN(const GraphBundle* graph_bundle,
                      GetGraphBundle(pipeline_name));
  // When lazy loading is enabled, the graph may not be loaded yet. For this
  // simple query, instead of loading the graph and getting the data type from
  // the first kernel, we just infer the data type.
  const xla_saved_model_orchestration::Pipeline& pipeline =
      graph_bundle->pipeline;
  const auto first_pipeline_component = pipeline.components().begin();
  if (first_pipeline_component->role() ==
      xla_saved_model_orchestration::ROLE_MODEL) {
    return KernelIODataType::kHostBuffer;
  }
  const std::string& first_component_name =
      first_pipeline_component->function_name();

  TF_ASSIGN_OR_RETURN(
      const xla_saved_model_manifest::Function* func,
      GetFunctionByObjectName(manifest_objects_, first_component_name));
  // If the first pipeline component has a StableHLO body, the corresponding
  // kernel will take HostBuffer as input.
  if (func->body().has_stable_hlo_body()) {
    return KernelIODataType::kHostBuffer;
  }

  TF_ASSIGN_OR_RETURN(std::string key, GetKeyFromProcessorMimeType(*func));
  TF_ASSIGN_OR_RETURN(OptionalFunctionHandler * handler,
                      optional_function_handler_router_.GetHandler(key));
  bool pre_data_conversion = first_pipeline_component->use_batching();
  return handler->GetInputDataType(pre_data_conversion);
}

absl::StatusOr<KernelIODataType> Interpreter::GetModelOutputDataType(
    absl::string_view pipeline_name) const {
  TF_ASSIGN_OR_RETURN(const GraphBundle* graph_bundle,
                      GetGraphBundle(pipeline_name));
  // When lazy loading is enabled, the graph may not be loaded yet. For this
  // simple query, instead of loading the graph and getting the data type from
  // the last kernel, we just infer the data type.
  const xla_saved_model_orchestration::Pipeline& pipeline =
      graph_bundle->pipeline;
  const auto last_pipeline_component = pipeline.components().rbegin();
  if (last_pipeline_component->role() ==
      xla_saved_model_orchestration::ROLE_MODEL) {
    return KernelIODataType::kHostBuffer;
  }
  // If the last pipeline component has a StableHLO body, the corresponding
  // kernel will return HostBuffer.
  TF_ASSIGN_OR_RETURN(
      const xla_saved_model_manifest::Function* func,
      GetFunctionByObjectName(manifest_objects_,
                              last_pipeline_component->function_name()));
  if (func->body().has_stable_hlo_body()) {
    return KernelIODataType::kHostBuffer;
  }

  TF_ASSIGN_OR_RETURN(std::string key, GetKeyFromProcessorMimeType(*func));
  TF_ASSIGN_OR_RETURN(OptionalFunctionHandler * handler,
                      optional_function_handler_router_.GetHandler(key));
  bool post_data_conversion = last_pipeline_component->use_batching();

  return handler->GetOutputDataType(post_data_conversion);
}

absl::Status Interpreter::MarkFullyInitialized() {
  absl::Status unloaded_pipeline_status = absl::FailedPreconditionError(
      "The pipeline is in the model but not loaded during warmup (check if "
      "lazy loading is enabled). Cannot load new pipeline after full "
      "initialization.");
  for (auto& [unused, graph_bundle] : graph_bundles_) {
    absl::MutexLock lock(graph_bundle->graph_mutex);
    if (graph_bundle->graph.has_value()) {
      if (!graph_bundle->graph->ok()) {
        continue;
      }
      for (auto& kernel : **graph_bundle->graph) {
        TF_RETURN_IF_ERROR(kernel->MarkFullyInitialized());
      }
    } else {
      graph_bundle->graph.emplace(unloaded_pipeline_status);
    }
  }
  return absl::OkStatus();
}

int32_t Interpreter::GenerateNextRequestId() {
  int32_t next_id = request_id_counter_.fetch_add(1);
  // -1 is the sentinel value for unassigned request ID. Skip it if the
  // counter wraps around.
  if (next_id == -1) {
    next_id = request_id_counter_.fetch_add(1);
  }
  return next_id;
}

}  // namespace xla_serving
