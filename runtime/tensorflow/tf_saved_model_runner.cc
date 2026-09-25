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

#include "runtime/tensorflow/tf_saved_model_runner.h"  // from @xla_serving

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "tensorflow/cc/saved_model/tag_constants.h"  // from @org_tensorflow
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tensorflow/core/framework/function.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/graph.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow
#include "tensorflow/core/protobuf/meta_graph.pb.h"  // from @org_tensorflow
#include "tensorflow/core/tfrt/runtime/runtime.h"  // from @org_tensorflow
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"  // from @org_tensorflow
#include "tensorflow/core/tfrt/saved_model/saved_model.h"  // from @org_tensorflow
#include "include/tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace xla_serving {
namespace {

// Validates that the given `tensor`'s dtype and shape matches the
// `tensor_spec`.
absl::Status ValidateTensorSpec(
    const tensorflow::tfrt_stub::TensorSpec& tensor_spec,
    const tensorflow::Tensor& tensor) {
  if (tensor_spec.dtype != tensorflow::DT_INVALID &&
      tensor.dtype() != tensor_spec.dtype) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor's dtype %s does not match the expected dtype %s.",
        tensorflow::DataTypeString(tensor.dtype()),
        tensorflow::DataTypeString(tensor_spec.dtype)));
  }
  if (!tensor_spec.shape.IsCompatibleWith(tensor.shape())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor's shape %s does not match the expected shape %s.",
        tensor.shape().DebugString(), tensor_spec.shape.DebugString()));
  }
  return absl::OkStatus();
}

// Returns the names of the given `inputs` that are not in `input_names`.
absl::flat_hash_set<std::string> FindInvalidInputNames(
    const absl::flat_hash_map<std::string, tensorflow::Tensor>& inputs,
    absl::Span<const std::string> input_names) {
  absl::flat_hash_set<std::string> given_input_names;
  given_input_names.reserve(inputs.size());
  for (const auto& [name, _] : inputs) {
    given_input_names.insert(name);
  }
  for (absl::string_view input_name : input_names) {
    given_input_names.erase(input_name);
  }
  return given_input_names;
}

}  // namespace

absl::StatusOr<std::shared_ptr<TfSavedModelRunner>> TfSavedModelRunner::Create(
    absl::string_view saved_model_dir,
    std::optional<tensorflow::tfrt_stub::SavedModel::Options> load_options,
    WorkQueueFactory work_queue_factory) {
  if (saved_model_dir.empty()) {
    return absl::InvalidArgumentError("saved_model_dir is empty");
  }

  LOG(INFO) << "Using TFRT to load TF SavedModel from directory: "
            << saved_model_dir;
  std::unordered_set<std::string> tags = {tensorflow::kSavedModelTagServe};
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<tensorflow::tfrt_stub::SavedModel>
          tfrt_loaded_saved_model,
      tensorflow::tfrt_stub::SavedModelImpl::LoadSavedModel(
          load_options.has_value() ? *load_options : GetDefaultLoadOptions(),
          saved_model_dir, tags));

  if (tfrt_loaded_saved_model == nullptr) {
    return absl::InternalError("Failed to load TF SavedModel. Got nullptr.");
  }
  return std::shared_ptr<TfSavedModelRunner>(new TfSavedModelRunner(
      std::move(tfrt_loaded_saved_model), std::move(work_queue_factory)));
}

tensorflow::tfrt_stub::SavedModel::Options
TfSavedModelRunner::GetDefaultLoadOptions() {
  static tensorflow::tfrt_stub::Runtime* default_runtime =
      tensorflow::tfrt_stub::Runtime::Create(
          tensorflow::tfrt_stub::WrapDefaultWorkQueue(
              tfrt::CreateMultiThreadedWorkQueue(8, 8)))
          .release();
  // TODO: consider setting other options to make the default
  // options more suitable for the use case.
  return tensorflow::tfrt_stub::SavedModel::Options(default_runtime);
}

bool TfSavedModelRunner::ContainsFunction(
    absl::string_view function_name) const {
  return tfrt_loaded_saved_model_->GetMetaGraphDef().signature_def().contains(
      function_name);
}

std::vector<std::string> TfSavedModelRunner::FunctionNames() const {
  return tfrt_loaded_saved_model_->GetFunctionNames();
}

int TfSavedModelRunner::GetNumInputs(absl::string_view function_name) const {
  return tfrt_loaded_saved_model_->GetMetaGraphDef()
      .signature_def()
      .at(function_name)
      .inputs_size();
}

int TfSavedModelRunner::GetNumOutputs(absl::string_view function_name) const {
  return tfrt_loaded_saved_model_->GetMetaGraphDef()
      .signature_def()
      .at(function_name)
      .outputs_size();
}

tensorflow::tfrt_stub::FunctionMetadata TfSavedModelRunner::GetFunctionMetadata(
    absl::string_view function_name) const {
  std::optional<tensorflow::tfrt_stub::FunctionMetadata> metadata =
      tfrt_loaded_saved_model_->GetFunctionMetadata(function_name);
  CHECK(metadata.has_value()) << "Function not found: " << function_name;
  return *metadata;
}

absl::Status TfSavedModelRunner::GetSortedOutputNamesInSignatureDef(
    absl::string_view function_name,
    absl::Span<const std::string> output_names,
    std::vector<std::string>& sorted_output_names,
    std::vector<int>& output_original_indices) const {
  const tensorflow::SignatureDef& signature_def =
      tfrt_loaded_saved_model_->GetMetaGraphDef().signature_def().at(
          function_name);
  std::vector<std::string> output_names_in_signature_def;
  output_names_in_signature_def.reserve(output_names.size());
  for (const auto& output_name : output_names) {
    auto it = signature_def.outputs().find(output_name);
    if (it == signature_def.outputs().end()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid output tensor name \"%s\". It is not found "
                          "in the signature def.",
                          output_name));
    }
    output_names_in_signature_def.emplace_back(it->second.name());
  }
  tensorflow::tfrt_stub::CreateSortedNamesAndOriginalIndices(
      output_names_in_signature_def, sorted_output_names,
      output_original_indices);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<tensorflow::Tensor>> TfSavedModelRunner::Execute(
    tensorflow::tfrt_stub::SavedModel::RunOptions run_options,
    absl::string_view function_name, const NameToTensorMap& inputs,
    absl::Span<const std::string> sorted_output_tensor_names,
    absl::Span<const int> output_original_indices) {
  tensorflow::tfrt_stub::FunctionMetadata function_metadata =
      GetFunctionMetadata(function_name);
  int num_used_inputs = 0;
  for (int i = 0; i < GetNumInputs(function_name); ++i) {
    const std::string& input_name = GetInputNames(function_name)[i];
    if (auto it = inputs.find(input_name); it == inputs.end()) {
      if (!function_metadata.GetDefaultInputs().contains(input_name)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Required input tensor name \"%s\" is not found in "
                            "the inputs, and the function does not have a "
                            "default value for it.",
                            input_name));
      }
    } else {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          ValidateTensorSpec(function_metadata.GetInputSpecs()[i], it->second),
          absl::StrCat(
              "Input tensor with name \"", input_name,
              "\" does not match the TensorSpec in the function signature."));
      num_used_inputs++;
    }
  }

  // If a work queue factory is provided, run the TF function on the
  // per-request work queue obtained from it. `work_queue` must outlive the
  // graph execution below.
  std::unique_ptr<tensorflow::tfrt_stub::WorkQueueInterface> work_queue;
  if (work_queue_factory_) {
    work_queue = work_queue_factory_();
    if (work_queue != nullptr) {
      run_options.work_queue = work_queue.get();
    }
  }

  // Since we have looped through all required input names, if any input name
  // is not used, it means that the input name is invalid.
  if (num_used_inputs < inputs.size()) {
    absl::flat_hash_set<std::string> invalid_input_names =
        FindInvalidInputNames(inputs, function_metadata.GetInputNames());
    CHECK_EQ(invalid_input_names.size(), inputs.size() - num_used_inputs);

    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid input tensor names found in the inputs: ",
        absl::StrJoin(invalid_input_names, ","), ". Allowable input names: ",
        absl::StrJoin(GetInputNames(function_name), ",")));
  }

  const tensorflow::SignatureDef& signature_def =
      tfrt_loaded_saved_model_->GetMetaGraphDef().signature_def().at(
          function_name);
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_with_names;
  inputs_with_names.reserve(inputs.size());
  for (auto& [name, tensor] : inputs) {
    inputs_with_names.push_back(std::make_pair(
        signature_def.inputs().at(name).name(), std::move(tensor)));
  }

  // Handle inputs.
  std::vector<std::string> input_names;
  input_names.reserve(inputs.size());
  for (const auto& p : inputs_with_names) input_names.push_back(p.first);
  std::vector<std::string> sorted_input_names;
  std::vector<int> input_original_indices;
  tensorflow::tfrt_stub::CreateSortedNamesAndOriginalIndices(
      input_names, sorted_input_names, input_original_indices);

  // We also need to create sorted input dtypes as they are needed for the
  // RunWithSortedInputsOutputs.
  std::vector<tensorflow::DataType> sorted_input_dtypes;
  sorted_input_dtypes.reserve(inputs.size());
  for (int original_index : input_original_indices) {
    sorted_input_dtypes.push_back(
        inputs_with_names.at(original_index).second.dtype());
  }

  std::vector<tensorflow::Tensor> outputs;
  auto status =
      tfrt_loaded_saved_model_->graph_executor().RunWithSortedInputsOutputs(
          run_options, function_name, inputs_with_names, sorted_input_names,
          sorted_input_dtypes, sorted_output_tensor_names,
          /*sorted_target_node_names=*/{}, input_original_indices,
          output_original_indices, &outputs);
  if (absl::IsDeadlineExceeded(status)) {
    return absl::CancelledError(
        absl::StrCat("Deadline exceeded in function '", function_name,
                     "' in TF SavedModel: ", status.message()));
  }
  TF_RETURN_IF_ERROR(status);
  return outputs;
}

}  // namespace xla_serving
