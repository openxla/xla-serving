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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_TF_SAVED_MODEL_RUNNER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_TF_SAVED_MODEL_RUNNER_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "tensorflow/core/framework/function.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/graph.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/tfrt/runtime/runtime.h"  // from @org_tensorflow
#include "tensorflow/core/tfrt/saved_model/saved_model.h"  // from @org_tensorflow

namespace xla_serving {

// Runs a TF SavedModel. This is a thin wrapper around
// `tensorflow::tfrt_stub::SavedModel`.
class TfSavedModelRunner {
 public:
  // A map from name (std::string) to tensor (tensorflow::Tensor).
  //
  // The actual container type is an implementation detail and may change. Use
  // it as a typical unordered map such as via construction `NameToTensorMap m =
  // {{"a", tensor1}, {"b", tensor2}}` and iteration `for (const auto& [name,
  // tensor] : m) { ... }`.
  using NameToTensorMap = absl::flat_hash_map<std::string, tensorflow::Tensor>;

  // Creates a TfSavedModelRunner from the given TF SavedModel directory. Here a
  // shared pointer is used to allow sharing the TF SavedModel across different
  // functions, e.g., pre-processing and post-processing functions.

  // If `load_options` is not provided, a default
  // `tensorflow::tfrt_stub::Runtime` will be created.
  static absl::StatusOr<std::shared_ptr<TfSavedModelRunner>> Create(
      absl::string_view saved_model_dir,
      std::optional<tensorflow::tfrt_stub::SavedModel::Options> load_options =
          std::nullopt);

  // Executes a certain TF function in the TF SavedModel with the given inputs.
  // `inputs` contains pairs of <input_tensor_name, input_tensor>. Each input
  // tensor name must be present in `GetInputNames(function_name)`.
  // `sorted_output_tensor_names` and `output_original_indices` are the sorted
  // output tensor names and their original indices in the signature def of the
  // TF function. Users are expected to call
  // `GetSortedOutputNamesInSignatureDef` to get them.
  absl::StatusOr<std::vector<tensorflow::Tensor>> Execute(
      tensorflow::tfrt_stub::SavedModel::RunOptions run_options,
      absl::string_view function_name, const NameToTensorMap& inputs,
      absl::Span<const std::string> sorted_output_tensor_names,
      absl::Span<const int> output_original_indices);

  // Returns true if the TF SavedModel contains the given function.
  bool ContainsFunction(absl::string_view function_name) const;

  // Returns the names of all functions in the TF SavedModel.
  std::vector<std::string> FunctionNames() const;

  // Returns the number of inputs of the TF function.
  int GetNumInputs(absl::string_view function_name) const;

  // Returns the number of outputs of the TF function.
  int GetNumOutputs(absl::string_view function_name) const;

  // Returns the function metadata of the TF function.
  tensorflow::tfrt_stub::FunctionMetadata GetFunctionMetadata(
      absl::string_view function_name) const;

  // Returns the names of the inputs of the TF function.
  const std::vector<std::string>& GetInputNames(
      absl::string_view function_name) const {
    return GetFunctionMetadata(function_name).GetInputNames();
  }

  // Returns the names of the outputs of the TF function.
  const std::vector<std::string>& GetOutputNames(
      absl::string_view function_name) const {
    return GetFunctionMetadata(function_name).GetOutputNames();
  }

  // Returns the sorted output names and their original indices in the signature
  // def of the TF function.
  absl::Status GetSortedOutputNamesInSignatureDef(
      absl::string_view function_name,
      absl::Span<const std::string> output_names,
      std::vector<std::string>& sorted_output_names,
      std::vector<int>& output_original_indices) const;

 private:
  explicit TfSavedModelRunner(std::unique_ptr<tensorflow::tfrt_stub::SavedModel>
                                  tfrt_loaded_saved_model)
      : tfrt_loaded_saved_model_(std::move(tfrt_loaded_saved_model)) {}

  // Creates default options to load the TF SavedModel.
  static tensorflow::tfrt_stub::SavedModel::Options GetDefaultLoadOptions();

  // Will be used by `tfrt_loaded_saved_model_`.
  std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime_;

  // Used to cache the loaded TF SavedModel so that it can be shared across
  // different functions, e.g., pre-processing and post-processing functions.
  std::unique_ptr<tensorflow::tfrt_stub::SavedModel> tfrt_loaded_saved_model_;
};

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_TF_SAVED_MODEL_RUNNER_H_
