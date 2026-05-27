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

#include "runtime/interpreter/interpreter.h"  // from @neptune

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "protos/orchestration.pb.h"  // from @neptune
#include "protos/type.pb.h"  // from @neptune
#include "runtime/core/batching/batch_scheduler.h"  // from @neptune
#include "runtime/core/constants.h"  // from @neptune
#include "runtime/core/flat_signature.h"  // from @neptune
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @neptune
#include "runtime/core/stablehlo.h"  // from @neptune
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/data_loader/data_loader.h"  // from @neptune
#include "runtime/interpreter/context.h"  // from @neptune
#include "runtime/interpreter/interpreter_create_options.h"  // from @neptune
#include "runtime/interpreter/linear_graph_execute.h"  // from @neptune
#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "runtime/interpreter/optional_function_handler.h"  // from @neptune
#include "runtime/interpreter/optional_function_handler_factory.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune
#include "runtime/kernel/batching/batch_options_utils.h"  // from @neptune
#include "runtime/kernel/batching/batch_runner_op.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_constants.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter_gauge.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/platform/context.h"  // from @tsl
#include "tsl/platform/path.h"  // from @tsl
#include "tsl/profiler/lib/traceme.h"  // from @tsl

ABSL_FLAG(bool, jsv_enable_parallel_loading_pipelines, true,
          "When set to true, load the pipelines in parallel.");

namespace neptune {

using neptune_model_orchestration::BatchOptions;

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

absl::Status ValidateStableHlo(
    absl::Span<const neptune_model_manifest::Function* absl_nonnull const>
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

bool ShouldShloRunOnHost(
    absl::Span<const neptune_model_manifest::Function* absl_nonnull const>
        funcs,
    absl::string_view platform_name) {
  if (platform_name == "cpu") {
    return true;
  }
  for (auto* func : funcs) {
    const auto& platforms = func->body().stable_hlo_body().lowering_platforms();
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
    const neptune_model_manifest::Function& func) {
  if (func.body().other().has_mime_type()) {
    return std::string(func.body().other().mime_type());
  }
  return absl::InvalidArgumentError(
      "Optional function handler does not have mime type needed for function "
      "routing.");
}

google::protobuf::RepeatedPtrField<neptune_model_orchestration::Component>
CreateAuxiliaryComponents(
    const neptune_model_orchestration::Pipeline& pipeline) {
  google::protobuf::RepeatedPtrField<neptune_model_orchestration::Component> components;
  for (const auto& pre_processor_name : pipeline.pre_processor_names()) {
    auto* component = components.Add();
    component->set_function_name(pre_processor_name);
    component->set_role(neptune_model_orchestration::ROLE_PREPROCESSOR);
  }
  for (const auto& model_function : pipeline.model_functions()) {
    auto* component = components.Add();
    component->set_function_name(model_function.model_function_name());
    component->set_role(neptune_model_orchestration::ROLE_MODEL);
    if (model_function.has_weights_name()) {
      component->set_weights_name(model_function.weights_name());
    }
  }
  for (const auto& post_processor_name : pipeline.post_processor_names()) {
    auto* component = components.Add();
    component->set_function_name(post_processor_name);
    component->set_role(neptune_model_orchestration::ROLE_POSTPROCESSOR);
  }
  return components;
}

// Creates batched kernels.
absl::StatusOr<std::vector<std::unique_ptr<Kernel>>> CreateBatchedKernels(
    std::vector<KernelSubGraph> pre_processor_sub_graphs,
    std::unique_ptr<Kernel> stablehlo_runner_op,
    std::vector<KernelSubGraph> post_processor_sub_graphs) {
  std::vector<std::unique_ptr<Kernel>> batched_kernels;
  for (auto& sub_graph : pre_processor_sub_graphs) {
    batched_kernels.insert(batched_kernels.end(),
                           std::make_move_iterator(sub_graph.begin()),
                           std::make_move_iterator(sub_graph.end()));
  }

  batched_kernels.push_back(std::move(stablehlo_runner_op));
  for (auto& sub_graph : post_processor_sub_graphs) {
    batched_kernels.insert(batched_kernels.end(),
                           std::make_move_iterator(sub_graph.begin()),
                           std::make_move_iterator(sub_graph.end()));
  }
  return batched_kernels;
}

// Build a BatchKernel from a list of kernels to be batched.

absl::StatusOr<std::unique_ptr<Kernel>> BuildBatchKernel(
    std::vector<std::unique_ptr<Kernel>> batched_kernels,
    const BatchOptions& batch_options,
    const InterpreterCreateOptions* create_options,
    tsl::thread::ThreadPool* thread_pool) {
  LOG(INFO) << "Building the batch kernel.";
  BatchOptions::BatchPaddingPolicy batch_padding_policy =
      batch_options.batch_padding_policy();
  TF_ASSIGN_OR_RETURN(std::string batch_padding_policy_string,
                      GetBatchPaddingPolicyString(batch_padding_policy));
  TF_ASSIGN_OR_RETURN(
      MixedPriorityBatchingPolicy mixed_priority_batching_policy,
      GetMixedPriorityBatchingPolicy(
          batch_options.mixed_priority_batching_policy()));

  BatchRunnerOpOptions batch_runner_op_options = {
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
          !batch_options.disable_large_batch_splitting(),
      .batch_padding_policy = std::move(batch_padding_policy_string),
      .mixed_priority_batching_policy = mixed_priority_batching_policy,
      .allocator = create_options->allocator,
  };
  if (batch_options.has_low_priority_batch_options()) {
    auto low_priority_batch_options =
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

  return CreateBatchRunnerOp(std::move(batched_kernels),
                             batch_runner_op_options, thread_pool);
}

absl::StatusOr<const neptune_model_manifest::Function*> GetFunctionByObjectName(
    const absl::flat_hash_map<std::string,
                              const neptune_model_manifest::TopLevelObject*>&
        manifest_objects,
    absl::string_view function_object_name) {
  if (!manifest_objects.contains(function_object_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to find function object with name: \"",
                     function_object_name, "\""));
  }
  const neptune_model_manifest::TopLevelObject* function_obj =
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
    const InterpreterCreateOptions& create_options,
    const neptune_model_orchestration::Pipeline& pipeline) {
  BatchOptions::BatchComponent batch_component =
      BatchOptions::BATCH_COMPONENT_UNSPECIFIED;
  if (create_options.enable_batching && pipeline.has_batch_options()) {
    const auto& batch_options = pipeline.batch_options();
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

// Inserts kernels to the graph.
void InsertKernels(std::vector<KernelSubGraph> pre_processor_sub_graphs,
                   std::unique_ptr<Kernel> stablehlo_or_batch_runner_op,
                   std::vector<KernelSubGraph> post_processor_sub_graphs,
                   std::vector<std::unique_ptr<Kernel>>& graph) {
  for (auto& sub_graph : pre_processor_sub_graphs) {
    graph.insert(graph.end(), std::make_move_iterator(sub_graph.begin()),
                 std::make_move_iterator(sub_graph.end()));
  }

  if (stablehlo_or_batch_runner_op != nullptr) {
    graph.push_back(std::move(stablehlo_or_batch_runner_op));
  }

  for (auto& post_processor_sub_graph : post_processor_sub_graphs) {
    graph.insert(graph.end(),
                 std::make_move_iterator(post_processor_sub_graph.begin()),
                 std::make_move_iterator(post_processor_sub_graph.end()));
  }
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
    const auto& kernel = graph[i];
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
    const neptune_model_orchestration::Pipeline& pipeline) {
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
    const neptune_model_orchestration::Pipeline& pipeline) {
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
    const neptune_model_manifest::StableHloFunctionBody& stable_hlo_body) {
  switch (stable_hlo_body.stable_hlo().data_case()) {
    case neptune_model_manifest::UnstructuredData::kInlinedBytes:
      return std::variant<std::string, absl::string_view>(
          absl::string_view(stable_hlo_body.stable_hlo().inlined_bytes()));
    case neptune_model_manifest::UnstructuredData::kFileSystemLocation: {
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
    absl::Span<const neptune_model_manifest::Function* const>
        stable_hlo_funcs) {
  std::vector<StableHlo::Spec> stablehlo_specs;
  stablehlo_specs.reserve(stable_hlo_funcs.size());
  for (const auto& func : stable_hlo_funcs) {
    TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> input_specs, GetInputSpec(*func));
    TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> output_specs,
                        GetOutputSpec(*func));
    TF_RETURN_IF_ERROR(CheckMimeAndDataType(
        func->body().stable_hlo_body().stable_hlo(), {kStableHloMimeType},
        {neptune_model_manifest::UnstructuredData::kInlinedBytes,
         neptune_model_manifest::UnstructuredData::kFileSystemLocation},
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
    const neptune_model_orchestration::Pipeline& pipeline) {
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
    absl::Span<const neptune_model_manifest::Function* const> stable_hlo_funcs,
    const neptune_model_manifest::Manifest& manifest,
    const std::vector<std::string>& weight_names,
    const neptune_model_orchestration::NamedTensorTypeList* named_inputs,
    const neptune_model_orchestration::NamedTensorTypeList* named_outputs,
    xla::CompileOptionsProto* absl_nullable compile_options,
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    const InterpreterCreateOptions& create_options,
    tsl::thread::ThreadPool* computation_thread_pool,
    IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry) {
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

  if (create_options.streamed_output_callback != nullptr) {
    TF_RETURN_IF_ERROR(builder.SetConfig(
        kStreamedOutputCallback, create_options.streamed_output_callback));
  }
  return GetGlobalKernelRegistry().Create("StableHloRunnerOp", builder);
}

struct NamedSignature {
  const neptune_model_orchestration::NamedTensorTypeList* inputs = nullptr;
  const neptune_model_orchestration::NamedTensorTypeList* outputs = nullptr;
};

NamedSignature GetNamedSignatureOfComponent(
    const neptune_model_orchestration::Component& component) {
  if (component.has_named_signature()) {
    return {.inputs = &component.named_signature().inputs(),
            .outputs = &component.named_signature().outputs()};
  }
  return {};
}

// LINT.IfChange(InterpreterCreateOptionsValidation)
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
// LINT.ThenChange(//depot//third_party/neptune/runtime/interpreter/interpreter_create_options.h:InterpreterCreateOptions)
}  // namespace

absl::StatusOr<std::unique_ptr<Interpreter>> Interpreter::Create(
    std::unique_ptr<InterpreterCreateOptions> create_options) {
  if (create_options == nullptr) {
    return absl::InvalidArgumentError("create_options must not be null.");
  }
  TF_RETURN_IF_ERROR(ValidateInterpreterCreateOptions(*create_options));
  auto interpreter =
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

absl::StatusOr<neptune_model_orchestration::Pipeline>
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
      neptune_model_orchestration::Pipelines pipelines,
      GetOexOrchestration(create_options_->model_path, manifest_));

  if (pipelines.name_to_pipeline().empty()) {
    return absl::InvalidArgumentError(
        "Model does not contain any pipeline. At least one pipeline is "
        "required.");
  }
  LOG(INFO) << "Found " << pipelines.name_to_pipeline_size()
            << " pipelines from Orbax model.";
  for (const auto& [pipeline_name, pipeline] : pipelines.name_to_pipeline()) {
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
    it_graph_bundle->second = std::make_unique<GraphBundle>(
        pipeline_name, pipeline, std::move(signature));
    GraphBundle& graph_bundle = *it_graph_bundle->second;
    graph_bundle.pipeline_name = pipeline_name;
    graph_bundle.keyword_arguments_execution =
        pipeline.keyword_arguments_based();
  }
  if (create_options_->lazy_loading_pipelines) {
    return absl::OkStatus();
  }
  bool parallel_loading =
      absl::GetFlag(FLAGS_jsv_enable_parallel_loading_pipelines);
  LOG(INFO) << "Loading graphs from pipelines";
  if (parallel_loading) {
    // If lazy loading is not enabled, load the graph eagerly.
    std::vector<std::unique_ptr<tsl::Thread>> graph_loading_threads;
    for (auto& [pipeline_name, graph_bundle] : graph_bundles_) {
      LOG(INFO) << "Loading graph from pipeline: " << pipeline_name;
      tsl::Context context(tsl::ContextKind::kThread);
      // Load all graphs eagerly and concurrently.
      graph_loading_threads.emplace_back(tsl::Env::Default()->StartThread(
          tsl::ThreadOptions(),
          absl::StrCat("graph_loading_thread_", pipeline_name),
          [this, &graph_bundle, context = std::move(context)]() {
            absl::MutexLock l(graph_bundle->graph_mutex);
            tsl::WithContext wc(context);
            graph_bundle->graph.emplace(LoadGraph(graph_bundle->pipeline));
          }));
    }
    graph_loading_threads.clear();  // The threads will be joined here.
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

absl::StatusOr<std::unique_ptr<Kernel>> Interpreter::LoadStableHloRunnerOp(
    absl::string_view model_function_name, absl::string_view weights_name,
    const neptune_model_orchestration::NamedTensorTypeList* named_inputs,
    const neptune_model_orchestration::NamedTensorTypeList* named_outputs,
    const std::string& platform_name) {
  TF_ASSIGN_OR_RETURN(
      std::vector<const neptune_model_manifest::Function*> stablehlo_funcs,
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
          compile_options.has_value() ? &compile_options.value() : nullptr,
          create_options_->ignore_device_assignment_in_manifest));

  std::vector<std::string> loaded_weight_names;
  // TODO: Support partial loading of per-function data.
  if (!weights_name.empty()) {
    absl::MutexLock l(weight_sets_mutex_);
    // It is possible that multiple pipelines share the set of weights. So it is
    // okay if `inserted` is true.
    auto [iter, inserted] = weight_sets_.try_emplace(weights_name);
    // A set of weights is uniquely identified by the `weights_name`. A given
    // set of weights can be shared by multiple pipelines. At the moment, we
    // only allow one set of weights. It is not clear if we will have a case
    // where we need multiple sets of weights.
    if (weight_sets_.size() > 1) {
      return absl::UnimplementedError(
          "Model has more than one weights name. Only a single shared "
          "weights_name is supported at the moment.");
    }
    // We should always try to load the weights, since only used weights by
    // the provided `stablehlo_funcs` will be loaded, and `stablehlo_funcs`
    // across pipelines may need different weights in the same set.
    TF_ASSIGN_OR_RETURN(
        DataLoader::LoadResult load_result,
        LoadWeights(weights_name, stablehlo_funcs, assigned_devices));
    loaded_weight_names = load_result.data_names;
    if (inserted) {
      LOG(INFO) << "Adding weight set \"" << iter->first << "\" with "
                << loaded_weight_names.size() << " weights.";
      // `loaded_weight_names` is used below; we should make a copy
      iter->second = loaded_weight_names;
    }
  }
  LOG(INFO) << "Adding StableHlo function: " << model_function_name;
  std::shared_ptr<xla::ifrt::Client> ifrt_client =
      platform_name == "cpu" ? (create_options_->host_ifrt_client != nullptr
                                    ? create_options_->host_ifrt_client
                                    : create_options_->ifrt_client)
                             : create_options_->ifrt_client;
  return CreateStableHloRunnerOp(
      stablehlo_funcs, manifest_, loaded_weight_names, named_inputs,
      named_outputs,
      compile_options.has_value() ? &compile_options.value() : nullptr,
      ifrt_client, *create_options_, create_options_->computation_thread_pool,
      &ifrt_loaded_variable_registry_);
}

absl::StatusOr<KernelGraph> Interpreter::LoadGraphFromComponentProto(
    const neptune_model_orchestration::Pipeline& pipeline,
    const google::protobuf::RepeatedPtrField<
        neptune_model_orchestration::Component>* absl_nullable
        auxiliary_components) {
  if (auxiliary_components == nullptr && pipeline.components().empty()) {
    return absl::InvalidArgumentError(
        "Pipeline does not contain any component.");
  }
  const google::protobuf::RepeatedPtrField<neptune_model_orchestration::Component>&
      components = auxiliary_components != nullptr ? *auxiliary_components
                                                   : pipeline.components();

  BatchingConfig batching_config =
      GetBatchingConfig(*create_options_, pipeline);

  bool keyword_arguments_based = pipeline.keyword_arguments_based();

  KernelGraph graph;
  std::vector<std::unique_ptr<Kernel>> batched_kernels;
  std::vector<KernelSubGraph> preprocessors;
  std::vector<KernelSubGraph> postprocessors;
  std::unique_ptr<Kernel> stablehlo_runner_op;

  // Explicitly start loading the stablehlo runner op first to accelerate the
  // loading process.
  for (const auto& component : components) {
    // TODO: use the nmed_signature to initialize the flat
    // signatures for the kernel runner op.
    NamedSignature named_signature =
        keyword_arguments_based ? GetNamedSignatureOfComponent(component)
                                : NamedSignature{};
    if (component.role() == neptune_model_orchestration::ROLE_MODEL) {
      if (stablehlo_runner_op != nullptr) {
        return absl::InvalidArgumentError(
            "A pipeline can only support one JAX function.");
      }
      TF_ASSIGN_OR_RETURN(
          stablehlo_runner_op,
          LoadStableHloRunnerOp(component.function_name(),
                                component.weights_name(),
                                named_signature.inputs, named_signature.outputs,
                                create_options_->platform_name));
    }
  }
  // Load data processors.
  int component_index = -1;
  for (const auto& component : components) {
    component_index++;
    if (component.role() == neptune_model_orchestration::ROLE_MODEL) {
      // The stablehlo runner op has been loaded explicitly at the beginning.
      // Skip it here.
      continue;
    }
    TF_ASSIGN_OR_RETURN(
        const neptune_model_manifest::Function* func,
        GetFunctionByObjectName(manifest_objects_, component.function_name()));
    LOG(INFO) << "Adding function: " << component.function_name();
    NamedSignature named_signature =
        keyword_arguments_based ? GetNamedSignatureOfComponent(component)
                                : NamedSignature{};
    std::vector<std::unique_ptr<Kernel>> sub_graph;
    if (func->body().has_stable_hlo_body()) {
      bool running_on_host =
          ShouldShloRunOnHost({func}, create_options_->platform_name);
      std::string platform_name =
          running_on_host ? "cpu" : create_options_->platform_name;
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<Kernel> shlo_kernel,
          LoadStableHloRunnerOp(
              component.function_name(), component.weights_name(),
              named_signature.inputs, named_signature.outputs, platform_name));
      sub_graph.push_back(std::move(shlo_kernel));
    } else {
      TF_ASSIGN_OR_RETURN(std::string key, GetKeyFromProcessorMimeType(*func));
      TF_ASSIGN_OR_RETURN(OptionalFunctionHandler * handler,
                          optional_function_handler_router_.GetHandler(key));

      bool pre_data_conversion = true;
      if (component_index == 0 &&
          component.role() == neptune_model_orchestration::ROLE_PREPROCESSOR) {
        pre_data_conversion = batching_config.batch_preprocessing;
      }
      bool post_data_conversion = true;
      if (component_index == components.size() - 1 &&
          component.role() == neptune_model_orchestration::ROLE_POSTPROCESSOR) {
        post_data_conversion = batching_config.batch_postprocessing;
      }
      TF_ASSIGN_OR_RETURN(
          sub_graph, handler->HandleFunction(
                         create_options_->model_path, *func, *create_options_,
                         pre_data_conversion, post_data_conversion,
                         named_signature.inputs, named_signature.outputs));
    }
    if (component.role() == neptune_model_orchestration::ROLE_PREPROCESSOR) {
      preprocessors.push_back(std::move(sub_graph));
    } else if (component.role() ==
               neptune_model_orchestration::ROLE_POSTPROCESSOR) {
      postprocessors.push_back(std::move(sub_graph));
    }
  }
  if (batching_config.batch_preprocessing) {
    for (auto& sub_graph : preprocessors) {
      batched_kernels.insert(batched_kernels.end(),
                             std::make_move_iterator(sub_graph.begin()),
                             std::make_move_iterator(sub_graph.end()));
    }
    preprocessors.clear();
  }
  if (batching_config.batch_model) {
    batched_kernels.push_back(std::move(stablehlo_runner_op));
    stablehlo_runner_op.reset();
  }
  if (batching_config.batch_postprocessing) {
    for (auto& sub_graph : postprocessors) {
      batched_kernels.insert(batched_kernels.end(),
                             std::make_move_iterator(sub_graph.begin()),
                             std::make_move_iterator(sub_graph.end()));
    }
    postprocessors.clear();
  }
  if (batched_kernels.empty()) {
    InsertKernels(std::move(preprocessors), std::move(stablehlo_runner_op),
                  std::move(postprocessors), graph);
  } else {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<Kernel> stablehlo_or_batch_runner_op,
        BuildBatchKernel(std::move(batched_kernels), pipeline.batch_options(),
                         create_options_.get(),
                         create_options_->computation_thread_pool));
    InsertKernels(std::move(preprocessors),
                  std::move(stablehlo_or_batch_runner_op),
                  std::move(postprocessors), graph);
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

// TODO: refactor this method into smaller ones.
// TODO: Add metric to monitor the model that use the legacy
// fields and remove this function once all models are migrated to the new
// components.
absl::StatusOr<KernelGraph> Interpreter::LoadGraph(
    const neptune_model_orchestration::Pipeline& pipeline) {
  // When there are components in the pipeline and the pipeline is keyword
  // arguments based, we load the graph from the component proto for now as the
  // LoadGraphFromComponentProto method is not fully implemented yet.
  if (!pipeline.components().empty()) {
    return LoadGraphFromComponentProto(pipeline);
  }
  // If there are no components in the pipeline, create them from legacy fields
  // and then call LoadGraphFromComponentProto.
  google::protobuf::RepeatedPtrField<neptune_model_orchestration::Component> components =
      CreateAuxiliaryComponents(pipeline);
  return LoadGraphFromComponentProto(pipeline, &components);
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
    absl::Span<const neptune_model_manifest::Function* const> stablehlo_funcs,
    const AssignedDevices& assigned_devices) {
  const auto it = manifest_objects_.find(weight_set_name);
  if (it == manifest_objects_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Object key for weight names ", weight_set_name,
                     " not found in manifest."));
  }
  const auto& value = it->second->value();
  if (!value.has_external()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Weights name ", weight_set_name, " is not an external value."));
  }
  const auto& external = value.external();
  TF_ASSIGN_OR_RETURN(
      auto data_loader,
      DataLoaderFactoryRegistry::GetGlobalRegistry().CreateDataLoader(
          external.loader_type(),
          DataLoader::CreateOptions{
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry_,
              .ifrt_client = create_options_->ifrt_client,
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
    const RunOptions run_options) {
  VLOG(2) << "Interpreter::Run: pipeline_name=" << pipeline_name
          << ", model_name=" << create_options_->model_name;
  auto result = [this, pipeline_name, &positional_inputs, &keyword_inputs,
                 &run_options]() -> absl::StatusOr<std::vector<Value>> {
    TF_ASSIGN_OR_RETURN(GraphBundle * graph_bundle,
                        GetGraphBundle(pipeline_name));
    TF_ASSIGN_OR_RETURN(const KernelGraph* graph,
                        GetOrCreateGraph(*graph_bundle));
    if (graph_bundle->keyword_arguments_execution) {
      absl::flat_hash_map<std::string, Value> keyword_outputs;
      TF_ASSIGN_OR_RETURN(
          keyword_outputs,
          ExecuteLinearGraphWithKeywordArgs(
              RunContext::Create(run_options), keyword_inputs, *graph,
              create_options_->model_name, create_options_->model_version));
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
      return ExecuteLinearGraph(
          RunContext::Create(run_options), positional_inputs, *graph,
          create_options_->model_name, create_options_->model_version);
    }
    TF_ASSIGN_OR_RETURN(std::vector<Value> inputs,
                        ConvertToVectorInputs(*graph_bundle, positional_inputs,
                                              keyword_inputs));
    return ExecuteLinearGraph(RunContext::Create(run_options), inputs, *graph,
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
    const RunOptions run_options) {
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
    const RunOptions run_options) {
  VLOG(2) << "Interpreter::RunAsync: pipeline_name=" << pipeline_name
          << ", model_name=" << create_options_->model_name;
  auto future = [this, pipeline_name, &positional_inputs, &keyword_inputs,
                 &run_options]() {
    // TODO: type check inputs and outputs.
    absl::StatusOr<GraphBundle*> graph_bundle = GetGraphBundle(pipeline_name);
    if (!graph_bundle.ok()) {
      return tsl::Future<std::vector<Value>>(graph_bundle.status());
    }

    absl::StatusOr<const KernelGraph*> graph = GetOrCreateGraph(**graph_bundle);
    if (!graph.ok()) {
      return tsl::Future<std::vector<Value>>(graph.status());
    }

    if (keyword_inputs.empty()) {
      return AsyncExecuteLinearGraph(
          RunContext::Create(run_options), positional_inputs, **graph,
          create_options_->execute_worker_thread_pool,
          create_options_->model_name, create_options_->model_version);
    }
    auto inputs = ConvertToVectorInputs(**graph_bundle, positional_inputs,
                                        keyword_inputs);
    if (!inputs.ok()) {
      return tsl::Future<std::vector<Value>>(inputs.status());
    }
    return AsyncExecuteLinearGraph(
        RunContext::Create(run_options), *inputs, **graph,
        create_options_->execute_worker_thread_pool,
        create_options_->model_name, create_options_->model_version);
  }();

  future.OnReady([model_name = create_options_->model_name,
                  model_version = create_options_->model_version](
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
    const RunOptions run_options) {
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
      [&output_flat_signature = (*graph_bundle)->signature.output_signature,
       output_promise = std::move(promise)](
          absl::StatusOr<std::vector<Value>> vector_outputs) mutable -> void {
        if (!vector_outputs.ok()) {
          output_promise.Set(vector_outputs.status());
          return;
        }
        output_promise.Set(
            output_flat_signature.UnflattenToMap(*vector_outputs));
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
  auto& pipeline = graph_bundle->pipeline;

  absl::string_view pre_processor_name;
  for (const auto& component : pipeline.components()) {
    if (component.role() == neptune_model_orchestration::ROLE_PREPROCESSOR) {
      pre_processor_name = component.function_name();
      break;
    }
  }
  if (pre_processor_name.empty()) {
    // Fall back to the legacy field.
    if (!pipeline.pre_processor_names().empty()) {
      pre_processor_name = pipeline.pre_processor_names(0);
    } else {
      return KernelIODataType::kHostBuffer;
    }
  }

  bool pre_processor_in_batching =
      create_options_->enable_batching && pipeline.has_batch_options() &&
      (pipeline.batch_options().batch_component() ==
           BatchOptions::WHOLE_PIPELINE ||
       pipeline.batch_options().batch_component() ==
           BatchOptions::PRE_PROCESSOR_AND_MODEL_FUNCTION);

  TF_ASSIGN_OR_RETURN(
      const neptune_model_manifest::Function* func,
      GetFunctionByObjectName(manifest_objects_, pre_processor_name));
  if (func->body().has_stable_hlo_body()) {
    return KernelIODataType::kHostBuffer;
  }

  TF_ASSIGN_OR_RETURN(std::string key, GetKeyFromProcessorMimeType(*func));
  TF_ASSIGN_OR_RETURN(OptionalFunctionHandler * handler,
                      optional_function_handler_router_.GetHandler(key));

  return handler->GetInputDataType(pre_processor_in_batching);
}

absl::StatusOr<KernelIODataType> Interpreter::GetModelOutputDataType(
    absl::string_view pipeline_name) const {
  TF_ASSIGN_OR_RETURN(const GraphBundle* graph_bundle,
                      GetGraphBundle(pipeline_name));
  // When lazy loading is enabled, the graph may not be loaded yet. For this
  // simple query, instead of loading the graph and getting the data type from
  // the last kernel, we just infer the data type.
  auto& pipeline = graph_bundle->pipeline;
  if (pipeline.post_processor_names().empty()) {
    return KernelIODataType::kHostBuffer;
  }

  bool post_processor_in_batching =
      create_options_->enable_batching && pipeline.has_batch_options() &&
      (pipeline.batch_options().batch_component() ==
           BatchOptions::WHOLE_PIPELINE ||
       pipeline.batch_options().batch_component() ==
           BatchOptions::MODEL_FUNCTION_AND_POST_PROCESSOR);

  absl::string_view post_processor_name =
      pipeline.post_processor_names(pipeline.post_processor_names_size() - 1);
  TF_ASSIGN_OR_RETURN(
      const neptune_model_manifest::Function* func,
      GetFunctionByObjectName(manifest_objects_, post_processor_name));
  if (func->body().has_stable_hlo_body()) {
    return KernelIODataType::kHostBuffer;
  }

  TF_ASSIGN_OR_RETURN(std::string key, GetKeyFromProcessorMimeType(*func));
  TF_ASSIGN_OR_RETURN(OptionalFunctionHandler * handler,
                      optional_function_handler_router_.GetHandler(key));

  return handler->GetOutputDataType(post_processor_in_batching);
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

}  // namespace neptune
