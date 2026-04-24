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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_KERNEL_TF_SAVED_MODEL_FUNCTION_RUNNER_OP_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_KERNEL_TF_SAVED_MODEL_FUNCTION_RUNNER_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/flat_signature.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "runtime/tensorflow/tf_saved_model_runner.h"  // from @neptune

namespace neptune {

// A kernel that runs a TF SavedModel.
class TfSavedModelFunctionRunnerOp : public CancellableKernel {
 public:
  ~TfSavedModelFunctionRunnerOp() override = default;

  static absl::StatusOr<std::unique_ptr<Kernel>> Create(
      const KernelBuilderContext& context);

  std::string Name() const override;

  Kernel::IODataTypes GetIODataTypes() const override {
    return {.input = pre_data_conversion_ ? KernelIODataType::kHostBuffer
                                          : KernelIODataType::kTensor,
            .output = post_data_conversion_ ? KernelIODataType::kHostBuffer
                                            : KernelIODataType::kTensor};
  }

  int GetNumInputs() const override {
    return tf_saved_model_runner_->GetNumInputs(function_name_);
  }

  int GetNumOutputs() const override {
    return tf_saved_model_runner_->GetNumOutputs(function_name_);
  }

 private:
  explicit TfSavedModelFunctionRunnerOp(
      std::shared_ptr<TfSavedModelRunner> tf_saved_model_runner,
      std::string function_name, std::vector<std::string> input_names,
      std::vector<std::string> sorted_output_names,
      std::vector<int> output_original_indices,
      std::unique_ptr<FlatSignature> input_signature,
      std::unique_ptr<FlatSignature> output_signature, bool pre_data_conversion,
      bool post_data_conversion);

  absl::Status RunInternal(ExecuteContext& context) override;

  std::shared_ptr<TfSavedModelRunner> tf_saved_model_runner_;
  std::string function_name_;
  std::vector<std::string> input_names_;
  std::vector<std::string> sorted_output_names_;
  std::vector<int> output_original_indices_;
  // Describes TF SavedModel function inputs. If not null, it indicates that not
  // all values in context are needed for the kernel and `input_signature_`
  // will be used to filter and map `context.named_values` to TF kernel inputs
  // via `FlatSignature::FillArgs`. Otherwise, `context.inputs` are used as TF
  // kernel inputs directly.
  std::unique_ptr<FlatSignature> input_signature_;
  // Describes TF SavedModel function outputs. If not null, maps TF kernel
  // outputs to `context.named_values` via `FlatSignature::UnflattenToMap`.
  // Otherwise, kernel outputs will be assigned to `context.outputs`.
  std::unique_ptr<FlatSignature> output_signature_;
  bool pre_data_conversion_ = false;
  bool post_data_conversion_ = false;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_KERNEL_TF_SAVED_MODEL_FUNCTION_RUNNER_OP_H_
