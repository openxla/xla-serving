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

#ifndef XLA_SERVING_RUNTIME_TENSORFLOW_TF_COMPATIBLE_OPTIONAL_FUNCTION_HANDLER_H_
#define XLA_SERVING_RUNTIME_TENSORFLOW_TF_COMPATIBLE_OPTIONAL_FUNCTION_HANDLER_H_

#include <memory>
#include <vector>

#include "absl/base/call_once.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "protos/orchestration.pb.h"  // from @neptune
#include "runtime/interpreter/interpreter_create_options.h"  // from @neptune
#include "runtime/interpreter/optional_function_handler.h"  // from @neptune
#include "runtime/interpreter/optional_function_handler_factory.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "runtime/tensorflow/tf_compatible_interpreter_create_options.h"  // from @neptune
#include "runtime/tensorflow/tf_saved_model_runner.h"  // from @neptune

namespace xla_serving {

// An implementation of `OptionalFunctionHandler` that can handle TensorFlow
// functions. `pre_data_conversion` and `post_data_conversion` are used to
// determine whether to do data conversion before and after the TF function.
class TfCompatibleOptionalFunctionHandler : public OptionalFunctionHandler {
 public:
  absl::StatusOr<std::unique_ptr<Kernel>> HandleFunction(
      absl::string_view model_path,
      const neptune_model_manifest::Function& function,
      const InterpreterCreateOptions& create_options, bool pre_data_conversion,
      bool post_data_conversion,
      const neptune_model_orchestration::NamedTensorTypeList* named_inputs,
      const neptune_model_orchestration::NamedTensorTypeList* named_outputs)
      override;

  KernelIODataType GetInputDataType(bool pre_data_conversion) const override {
    return pre_data_conversion ? KernelIODataType::kHostBuffer
                               : KernelIODataType::kTensor;
  }

  KernelIODataType GetOutputDataType(bool post_data_conversion) const override {
    return post_data_conversion ? KernelIODataType::kHostBuffer
                                : KernelIODataType::kTensor;
  }

 private:
  // Finds the TF SavedModel directory from the manifest and creates the
  // `TfSavedModelRunner`.
  static absl::StatusOr<std::shared_ptr<TfSavedModelRunner>>
  CreateTfSavedModelRunner(absl::string_view model_path,
                           TfrtOptions tfrt_options);

  // Makes sure that the `tf_saved_model_runner_` is initialized only once.
  absl::once_flag tf_saved_model_runner_init_once_;
  // This can be shared across different functions, e.g., pre-processing and
  // post-processing functions.
  absl::StatusOr<std::shared_ptr<TfSavedModelRunner>> tf_saved_model_runner_;
};

// A factory class to create `TfCompatibleOptionalFunctionHandler`.
class TfCompatibleOptionalFunctionHandlerFactory
    : public OptionalFunctionHandlerFactory {
 public:
  std::unique_ptr<OptionalFunctionHandler> GetHandler() override {
    return std::make_unique<TfCompatibleOptionalFunctionHandler>();
  }
};
}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_TENSORFLOW_TF_COMPATIBLE_OPTIONAL_FUNCTION_HANDLER_H_
