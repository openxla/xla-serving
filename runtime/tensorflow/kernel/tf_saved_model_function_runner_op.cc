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

#include "runtime/tensorflow/kernel/tf_saved_model_function_runner_op.h"  // from @neptune

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/flat_signature.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/core/value_utils.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "runtime/tensorflow/kernel/kernel_builder_constants.h"  // from @neptune
#include "runtime/tensorflow/tf_saved_model_runner.h"  // from @neptune
#include "runtime/tensorflow/utils/tf_to_host_buffer.h"  // from @neptune
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tsl/profiler/lib/traceme.h"  // from @tsl
#include "tsl/profiler/lib/traceme_encode.h"  // from @tsl

namespace neptune {

namespace {

// Returns true if the two vectors have the same elements in any order.
bool HaveTheSameUnorderedElements(const std::vector<std::string>& a,
                                  const std::vector<std::string>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  std::vector<std::string> a_sorted = a;
  std::sort(a_sorted.begin(), a_sorted.end());
  std::vector<std::string> b_sorted = b;
  std::sort(b_sorted.begin(), b_sorted.end());
  return a_sorted == b_sorted;
}

// Checks that the ordered input and output names match the input and output
// names in the TF SavedModel. The order of the names does not matter. For
// example,
//
//   input_names = ["a", "b", "c"]
//   output_names = ["d", "e", "f"]
//
// is valid if the input names in the TF SavedModel are ["a", "c", "b"] and the
// output names in the TF SavedModel are ["f", "e", "d"].
absl::Status CheckOrderedInputOutputNames(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const TfSavedModelRunner& tf_saved_model_runner,
    absl::string_view function_name) {
  VLOG(3) << "Ordered input names: " << absl::StrJoin(input_names, ", ");
  const std::vector<std::string>& saved_model_input_names =
      tf_saved_model_runner.GetInputNames(function_name);
  if (!HaveTheSameUnorderedElements(input_names, saved_model_input_names)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Ordered input names (%s) do not match the input names in the TF "
        "SavedModel (%s)",
        absl::StrJoin(input_names, ", "),
        absl::StrJoin(saved_model_input_names, ", ")));
  }

  VLOG(3) << "Ordered output names: " << absl::StrJoin(output_names, ", ");
  const std::vector<std::string>& saved_model_output_names =
      tf_saved_model_runner.GetOutputNames(function_name);
  if (!HaveTheSameUnorderedElements(output_names, saved_model_output_names)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Ordered output names (%s) do not match the output names in the TF "
        "SavedModel (%s)",
        absl::StrJoin(output_names, ", "),
        absl::StrJoin(saved_model_output_names, ", ")));
  }

  return absl::OkStatus();
}

}  // namespace

TfSavedModelFunctionRunnerOp::TfSavedModelFunctionRunnerOp(
    std::shared_ptr<TfSavedModelRunner> tf_saved_model_runner,
    std::string function_name, std::vector<std::string> input_names,
    std::vector<std::string> sorted_output_names,
    std::vector<int> output_original_indices,
    std::unique_ptr<FlatSignature> input_signature,
    std::unique_ptr<FlatSignature> output_signature, bool pre_data_conversion,
    bool post_data_conversion)
    : tf_saved_model_runner_(tf_saved_model_runner),
      function_name_(std::move(function_name)),
      input_names_(std::move(input_names)),
      sorted_output_names_(std::move(sorted_output_names)),
      output_original_indices_(std::move(output_original_indices)),
      input_signature_(std::move(input_signature)),
      output_signature_(std::move(output_signature)),
      pre_data_conversion_(pre_data_conversion),
      post_data_conversion_(post_data_conversion) {}

absl::StatusOr<std::unique_ptr<Kernel>> TfSavedModelFunctionRunnerOp::Create(
    const KernelBuilderContext& context) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<TfSavedModelRunner> tf_saved_model_runner,
                      context.GetConfig<std::shared_ptr<TfSavedModelRunner>>(
                          kTfSavedModelRunner));
  if (tf_saved_model_runner == nullptr) {
    return absl::InternalError("TfSavedModelRunner is nullptr");
  }
  TF_ASSIGN_OR_RETURN(std::string function_name,
                      context.GetConfig<std::string>(kTfFunctionSignatureName));
  if (function_name.empty()) {
    return absl::InvalidArgumentError("Function name is empty");
  }
  if (!tf_saved_model_runner->ContainsFunction(function_name)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Function name `%s` not found in the TF SavedModel MetaGraphDef. "
        "Available functions: %s",
        function_name,
        absl::StrJoin(tf_saved_model_runner->FunctionNames(), ", ")));
  }

  TF_ASSIGN_OR_RETURN(auto input_names,
                      context.GetConfig<std::vector<std::string>>(
                          kTfFunctionOrderedInputNames));

  TF_ASSIGN_OR_RETURN(auto output_names,
                      context.GetConfig<std::vector<std::string>>(
                          kTfFunctionOrderedOutputNames));

  TF_RETURN_IF_ERROR(CheckOrderedInputOutputNames(
      input_names, output_names, *tf_saved_model_runner, function_name));

  absl::StatusOr<FlatSignature> input_sig_status =
      context.GetConfig<FlatSignature>(kTfFunctionFlatInputSignature);
  std::unique_ptr<FlatSignature> input_sig;
  if (input_sig_status.ok()) {
    input_sig = std::make_unique<FlatSignature>(*input_sig_status);
    LOG(INFO) << "Flat input signature in TfSavedModelFunctionRunnerOp: "
              << input_sig->DebugString();
  } else if (!absl::IsNotFound(input_sig_status.status())) {
    return input_sig_status.status();
  }
  absl::StatusOr<FlatSignature> output_sig_status =
      context.GetConfig<FlatSignature>(kTfFunctionFlatOutputSignature);
  std::unique_ptr<FlatSignature> output_sig;
  if (output_sig_status.ok()) {
    output_sig = std::make_unique<FlatSignature>(*output_sig_status);
    LOG(INFO) << "Flat output signature in TfSavedModelFunctionRunnerOp: "
              << output_sig->DebugString();
  } else if (!absl::IsNotFound(output_sig_status.status())) {
    return output_sig_status.status();
  }

  bool pre_data_conversion = false;
  absl::StatusOr<bool> pre_data_conversion_status =
      context.GetConfig<bool>(kTfFunctionPreDataConversion);
  if (pre_data_conversion_status.ok()) {
    pre_data_conversion = *pre_data_conversion_status;
  } else if (!absl::IsNotFound(pre_data_conversion_status.status())) {
    return pre_data_conversion_status.status();
  }

  bool post_data_conversion = false;
  absl::StatusOr<bool> post_data_conversion_status =
      context.GetConfig<bool>(kTfFunctionPostDataConversion);
  if (post_data_conversion_status.ok()) {
    post_data_conversion = *post_data_conversion_status;
  } else if (!absl::IsNotFound(post_data_conversion_status.status())) {
    return post_data_conversion_status.status();
  }

  std::vector<std::string> sorted_output_names_in_signature_def;
  std::vector<int> output_original_indices_in_signature_def;
  TF_RETURN_IF_ERROR(
    tf_saved_model_runner->GetSortedOutputNamesInSignatureDef(
        function_name, output_names,
        sorted_output_names_in_signature_def,
        output_original_indices_in_signature_def));

  return absl::WrapUnique(new TfSavedModelFunctionRunnerOp(
      std::move(tf_saved_model_runner), std::move(function_name),
      std::move(input_names), std::move(sorted_output_names_in_signature_def),
      std::move(output_original_indices_in_signature_def), std::move(input_sig),
      std::move(output_sig), pre_data_conversion, post_data_conversion));
}

std::string TfSavedModelFunctionRunnerOp::Name() const {
  return "TfSavedModelFunctionRunnerOp";
}

absl::Status TfSavedModelFunctionRunnerOp::RunInternal(
    ExecuteContext& context) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(
        "TfSavedModelFunctionRunnerOp::Run",
        {{"model_id", absl::StrCat(context.metadata.model_name, ":",
                                   context.metadata.model_version)}});
  });
  const int num_inputs = GetNumInputs();

  std::vector<Value> inputs_values_storage;
  absl::Span<const Value> inputs_values;
  if (input_signature_) {
    inputs_values_storage.reserve(num_inputs);
    TF_RETURN_IF_ERROR(input_signature_->FillArgs(inputs_values_storage,
                                                  context.named_values));
    if (inputs_values_storage.size() != num_inputs) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TfSavedModelFunctionRunnerOp expects %d inputs after signature "
          "mapping, but got %d",
          num_inputs, inputs_values_storage.size()));
    }
    inputs_values = inputs_values_storage;
  } else {
    if (context.inputs.size() != num_inputs) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TfSavedModelFunctionRunnerOp supports %d inputs, but "
          "got %d",
          num_inputs, context.inputs.size()));
    }
    inputs_values = context.inputs;
  }

  TfSavedModelRunner::NameToTensorMap named_inputs;
  named_inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    // The lists `input_names_` and `inputs_values` always match up in order,
    // even when the original input signature contains a dict, because they are
    // both generated by flattening, with the same flattening algorithm:
    //
    // input_names_ is generated by either
    // tf_concrete_functions_to_obm._generate_names in Python (which uses that
    // flattening algorithm) or `GetNamesFromSignature` in C++ (which also uses
    // that flattening algorithm).
    //
    // `inputs_values` is currently prepared by the user (or extracted via
    // signature), who is advised to use that flattening algorithm. In the
    // future, `inputs_values` will be prepared by a `Flatten` function in C++
    // (which uses that flattening algorithm).
    //
    // TODO: Generate the list `inputs_values` with a flattening
    // function under our control, instead of asking/relying on the user to do
    // the flattening and provide the list `inputs_values`.
    const Value& input_value = inputs_values[i];
    const std::string& input_name = input_names_[i];
    if (pre_data_conversion_) {
      if (!input_value.IsType<HostBuffer>()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Input '%s' (at index %d) must be HostBuffer because "
            "pre_data_conversion is true.",
            input_name, i));
      }
      TF_ASSIGN_OR_RETURN(tensorflow::Tensor tensor,
                          ToTFTensor(input_value.Get<HostBuffer>()));
      named_inputs[input_name] = std::move(tensor);
    } else {
      if (!input_value.IsType<tensorflow::Tensor>()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Input '%s' (at index %d) must be tensorflow::Tensor because "
            "pre_data_conversion is false.",
            input_name, i));
      }
      named_inputs[input_name] = input_value.Get<tensorflow::Tensor>();
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<tensorflow::Tensor> outputs,
      tf_saved_model_runner_->Execute(
          {.deadline = absl::ToChronoTime(
               context.run_context.cancellation_context.deadline)},
          function_name_, named_inputs, sorted_output_names_,
          output_original_indices_));

  std::vector<Value> output_values;

  if (post_data_conversion_) {
    std::vector<HostBuffer> host_buffers;
    host_buffers.reserve(outputs.size());
    for (const auto& tensor : outputs) {
      TF_ASSIGN_OR_RETURN(auto host_buffer, ToHostBuffer(tensor));
      host_buffers.push_back(std::move(host_buffer));
    }
    output_values = internal::PackAsValues(std::move(host_buffers));
  } else {
    output_values = internal::PackAsValues(std::move(outputs));
  }

  if (output_signature_) {
    absl::flat_hash_map<std::string, Value> map_outputs;
    TF_ASSIGN_OR_RETURN(map_outputs, output_signature_->UnflattenToMap(
                                              std::move(output_values)));
    for (auto& [key, value] : map_outputs) {
      if (!context.named_values.insert({key, std::move(value)}).second) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Duplicate output key: ", key, " in TF SavedModel processor"));
      }
    }
  } else {
    context.outputs = std::move(output_values);
  }

  return absl::OkStatus();
}

}  // namespace neptune
