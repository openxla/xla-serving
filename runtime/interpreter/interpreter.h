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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_INTERPRETER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_INTERPRETER_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "protos/orchestration.pb.h"  // from @neptune
#include "runtime/core/flat_signature.h"  // from @neptune
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/data_loader/data_loader.h"  // from @neptune
#include "runtime/interpreter/interpreter_create_options.h"  // from @neptune
#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "runtime/interpreter/optional_function_handler_factory.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/tsl/concurrency/future.h"  // from @xla

namespace neptune {

using KernelGraph = std::vector<std::unique_ptr<Kernel>>;

class InterpreterPeer;

// The entry point to run the Orbax model.
class Interpreter {
 public:
  // Loads the model from the given `model_path` and creates an interpreter.
  //
  // Args:
  //   create_options: The options to create the model.
  static absl::StatusOr<std::unique_ptr<Interpreter>> Create(
      std::unique_ptr<InterpreterCreateOptions> create_options);

  // Returns the names of the pipelines in the model.
  std::vector<std::string> GetPipelineNames() const;

  // Returns the signature of the given pipeline, or an error if the pipeline
  // name is invalid.
  // Note: each flat signature returned by this method either contains names for
  // all tensors (when `named_signature` is available) or does not contain
  // names for any tensors (when `named_signature` is not available).
  absl::StatusOr<FunctionFlatSignature> GetPipelineSignature(
      absl::string_view pipeline_name) const;

  // Returns the pipeline proto for the given pipeline name.
  absl::StatusOr<neptune_model_orchestration::Pipeline> GetPipelineProto(
      absl::string_view pipeline_name) const;

  // Runs a pipeline in the model with the given inputs. `pipeline_name` is the
  // name of the pipeline to run. `positional_inputs` is a list of positional
  // inputs. `keyword_inputs` is a map of keyword inputs. They will be merged
  // together and passed to the model. `keyword_inputs` can only be used for
  // input arguments that have names available.
  //
  // The returned `Value`s are ordered based on the tree flattening order of the
  // model's output signature. For example, if the model's output signature is a
  // tuple of tensors, the returned `Value`s follow the same order as the tuple.
  // If the model's output signature is a dict of tensors, the returned `Value`s
  // are ordered by the keys of the dict.
  //
  // This method runs in the caller's thread instead of the internal worker
  // pool. This is primarily for (advanced) users who manages their own
  // threadpool and want to run the model in the same thread as the caller.
  absl::StatusOr<std::vector<Value>> Run(
      absl::string_view pipeline_name,
      absl::Span<const Value> positional_inputs,
      const absl::flat_hash_map<std::string, Value>& keyword_inputs = {},
      RunOptions run_options = RunOptions());

  // Same as `Run`, but returns a map of output values. Unlike `Run`, this
  // version can only be used when the model's output names are available.
  absl::StatusOr<absl::flat_hash_map<std::string, Value>> RunWithMapReturn(
      absl::string_view pipeline_name,
      absl::Span<const Value> positional_inputs,
      const absl::flat_hash_map<std::string, Value>& keyword_inputs,
      RunOptions run_options = RunOptions());

  // Runs a pipeline in the model asynchronously in the internal worker pool.
  // See `Run` for the definition of `pipeline_name`, `positional_inputs`, and
  // `keyword_inputs`. A user may choose to use this API if they needs to
  // support extremely high QPS and cannot afford to have a large number of
  // outstanding threads running the model synchronously. The input values are
  // assumed to be either reference counted or simply copyable such that copy of
  // `Value` can keep the input values alive.
  tsl::Future<std::vector<Value>> RunAsync(
      absl::string_view pipeline_name,
      absl::Span<const Value> positional_inputs,
      const absl::flat_hash_map<std::string, Value>& keyword_inputs = {},
      RunOptions run_options = RunOptions());

  // Same as `RunAsync`, but returns a map of output values. Unlike `RunAsync`,
  // this version can only be used when the model's output names are available.
  tsl::Future<absl::flat_hash_map<std::string, Value>> RunAsyncWithMapReturn(
      absl::string_view pipeline_name,
      absl::Span<const Value> positional_inputs,
      const absl::flat_hash_map<std::string, Value>& keyword_inputs,
      RunOptions run_options = RunOptions());

  // TODO: Allow the user to specify the data type.
  // Returns the accepted data type of the model input. This is determined by
  // the first computation function.
  absl::StatusOr<KernelIODataType> GetModelInputDataType(
      absl::string_view pipeline_name) const;

  // TODO: Allow the user to specify the data type.
  // Returns the data type of the model output. This is determined by the last
  // computation function.
  absl::StatusOr<KernelIODataType> GetModelOutputDataType(
      absl::string_view pipeline_name) const;

  // Marks the interpreter to fully initialized. All kernels in the graph will
  // be fully initialized after this is called. After this is called,
  // time-consuming tasks not suitable for the production environment will be
  // disallowed.
  absl::Status MarkFullyInitialized();

  std::string GetModelName() const { return create_options_->model_name; }

 private:
  // TODO: Remove this peer class once JSV fully supports
  // graph-based orchestration and tests can use the public API.
  friend class InterpreterPeer;
  explicit Interpreter(std::unique_ptr<InterpreterCreateOptions> create_options)
      : create_options_(std::move(create_options)) {}

  // Loads the model. This will load all the pipelines in the manifest unless
  // lazy loading is enabled.
  absl::Status Load();

  struct GraphBundle {
    // TODO: Remove this constructor once we upgrate oss neptune to
    // C++ 20.
    GraphBundle(std::string name, neptune_model_orchestration::Pipeline p,
                FunctionFlatSignature sig)
        : pipeline_name(std::move(name)),
          pipeline(std::move(p)),
          signature(std::move(sig)) {}

    // For logging and debugging.
    std::string pipeline_name;
    neptune_model_orchestration::Pipeline pipeline;
    FunctionFlatSignature signature;
    mutable absl::Mutex graph_mutex;
    // nullopt if the graph is not loaded yet.
    std::optional<absl::StatusOr<KernelGraph>> graph
        ABSL_GUARDED_BY(graph_mutex);
    // Whether the graph is executed based on keyword arguments.
    bool keyword_arguments_execution = false;
  };

  // Returns the graph bundle of the given pipeline.
  absl::StatusOr<GraphBundle* absl_nonnull> GetGraphBundle(
      absl::string_view pipeline_name);

  // Allows const access to the graph bundle of the given pipeline. This is
  // useful for methods that need to access the graph bundle but do not need to
  // modify it.
  absl::StatusOr<const GraphBundle* absl_nonnull> GetGraphBundle(
      absl::string_view pipeline_name) const;

  // Returns the graph of the given graph bundle. If the graph is not loaded
  // yet, it will be loaded.
  absl::StatusOr<const KernelGraph* absl_nonnull> GetOrCreateGraph(
      GraphBundle& graph_bundle);

  // Loads the graph from the given pipeline asynchronously. The graph will be
  // stored in the graph bundle.
  void LoadGraphAsync(GraphBundle& graph_bundle)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(graph_bundle.graph_mutex);

  // Loads the graph from the given pipeline.
  absl::StatusOr<KernelGraph> LoadGraph(
      const neptune_model_orchestration::Pipeline& pipeline);

  // Loads the graph from the given pipeline using the `components` from the
  // pipeline proto. If no components are provided in the pipeline proto, we
  // will create an auxiliary components from the legacy fields and load the
  // graph based on the auxiliary components.
  absl::StatusOr<KernelGraph> LoadGraphFromComponentProto(
      const neptune_model_orchestration::Pipeline& pipeline,
      const google::protobuf::RepeatedPtrField<
          neptune_model_orchestration::Component>* absl_nullable
          auxiliary_components = nullptr);

  // Loads a StableHloRunnerOp kernel based on the model function name and
  // weights name.
  // TODO: Remove the `platform_name` parameter once JSV supports
  // running processors on TPU. Currently, when models are exported using older
  // binaries without specifying a platform name, existing models still assume
  // lowering to CPU. However, JSV can execute this StableHLO on TPU even when
  // targeting the CPU platform. If a customer intends to run on TPU,
  // `StableHloRunnerOp` is called directly with an IFRT client that supports
  // TPU. Without the `platform_name` parameter, the new `host_ifrt_client`
  // would incorrectly route execution to CPU, which fails for TPU workloads.
  absl::StatusOr<std::unique_ptr<Kernel>> LoadStableHloRunnerOp(
      absl::string_view model_function_name, absl::string_view weights_name,
      const neptune_model_orchestration::NamedTensorTypeList* named_inputs =
          nullptr,
      const neptune_model_orchestration::NamedTensorTypeList* named_outputs =
          nullptr,
      const std::string& platform_name = "");

  // Loads the weights from the model.
  // Args:
  // - weight_set_name: the name for a set of weights.
  // - stablehlo_funcs: the StableHLO function(s) to load the weights for.
  // - assigned_devices: the assigned devices for the weights.
  absl::StatusOr<DataLoader::LoadResult> LoadWeights(
      absl::string_view weight_set_name,
      absl::Span<const neptune_model_manifest::Function* const> stablehlo_funcs,
      const AssignedDevices& assigned_devices,
      std::shared_ptr<xla::ifrt::Client> ifrt_client);

  // Converts the inputs to a vector of `Value`s by merging the positional and
  // keyword inputs.
  static absl::StatusOr<std::vector<Value>> ConvertToVectorInputs(
      const GraphBundle& graph_bundle,
      absl::Span<const Value> positional_inputs,
      const absl::flat_hash_map<std::string, Value>& keyword_inputs);

  neptune_model_manifest::Manifest manifest_;
  absl::flat_hash_map<std::string,
                      const neptune_model_manifest::TopLevelObject*>
      manifest_objects_;
  // Routes optional functions to the correct handler based on function
  // type. OptionalFunctionHandlers for each key are unique per
  // Interpreter instance.
  OptionalFunctionHandlerRouter optional_function_handler_router_;
  // WARNING: Destruction order matters here. `create_options_` contains the
  // `ifrt_client`, which must outlive the `PageableIfrtArray`s held inside
  // `ifrt_loaded_variable_registry_`. Therefore, `create_options_` must be
  // declared before `ifrt_loaded_variable_registry_` so it is destroyed after
  // it.
  std::unique_ptr<InterpreterCreateOptions> create_options_;
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry_;

  // TODO: Maybe put `weight_sets_` into
  // `ifrt_loaded_variable_registry_` to simplify usage.
  //
  // Maps a weight set name (key from the manifest's top-level object map) to a
  // list of weight names. Typically, a weight set corresponds to the weights of
  // a JAX model. JAX models across different pipelines may share the same set
  // of weights. In the future, we might have multiple weight sets in the
  // manifest, and each pipeline containing a JAX model has its own independent
  // set of weights.
  mutable absl::Mutex weight_sets_mutex_;
  absl::flat_hash_map<std::string, std::vector<std::string>> weight_sets_
      ABSL_GUARDED_BY(weight_sets_mutex_);
  // The key is the pipeline name in the manifest. The value type needs to be
  // pointer because `GraphBundle` is not copyable nor movable.
  absl::flat_hash_map<std::string, std::unique_ptr<GraphBundle>> graph_bundles_;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_INTERPRETER_H_
