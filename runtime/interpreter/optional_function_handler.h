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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_H_

#include <memory>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "protos/orchestration.pb.h"  // from @neptune
#include "runtime/interpreter/interpreter_create_options.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune

namespace neptune {

// An interface to handle optional (non-StableHLO) functions.
class OptionalFunctionHandler {
 public:
  virtual ~OptionalFunctionHandler() = default;

  // Handles the function by creating a kernel that can be inserted into the
  // current graph. `pre_data_conversion` and `post_data_conversion` is used to
  // determine whether to convert the input and output of the function to
  // HostBuffer.
  virtual absl::StatusOr<std::unique_ptr<Kernel>> HandleFunction(
      absl::string_view model_path,
      const neptune_model_manifest::Function& function,
      const InterpreterCreateOptions& create_options, bool pre_data_conversion,
      bool post_data_conversion,
      const neptune_model_orchestration::NamedTensorTypeList* named_inputs,
      const neptune_model_orchestration::NamedTensorTypeList*
          named_outputs) = 0;

  virtual KernelIODataType GetInputDataType(bool pre_data_conversion) const = 0;
  virtual KernelIODataType GetOutputDataType(
      bool post_data_conversion) const = 0;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_H_
