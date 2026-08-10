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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_NEPTUNE_DATA_COMPOSITE_PROCESSOR_FUNCTION_HANDLER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_NEPTUNE_DATA_COMPOSITE_PROCESSOR_FUNCTION_HANDLER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/processor/cc/core/composite_processor.h"  // from @neptune
#include "protos/manifest.pb.h"  // from @neptune
#include "protos/orchestration.pb.h"  // from @neptune
#include "runtime/interpreter/interpreter_create_options.h"  // from @neptune
#include "runtime/interpreter/optional_function_handler.h"  // from @neptune
#include "runtime/interpreter/optional_function_handler_factory.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune

namespace neptune {
namespace composite {

constexpr char kCompositeProcessorMimeType[] =
    "application/protobuf; "
    "type=neptune.data.composite.CompositeProcessorConfig";

constexpr char kLegacyCompositeProcessorMimeType[] =
    "application/protobuf; type=jax_data.composite.CompositeProcessorConfig";

// An implementation of `OptionalFunctionHandler` that can handle
// neptune::data::composite::CompositeProcessor.
class CompositeProcessorFunctionHandler : public OptionalFunctionHandler {
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
    return KernelIODataType::kHostBuffer;
  }

  KernelIODataType GetOutputDataType(bool post_data_conversion) const override {
    return KernelIODataType::kHostBuffer;
  }
};

// A factory class to create `CompositeProcessorFunctionHandler`.
class CompositeProcessorFunctionHandlerFactory
    : public OptionalFunctionHandlerFactory {
 public:
  std::unique_ptr<OptionalFunctionHandler> GetHandler() override {
    return std::make_unique<CompositeProcessorFunctionHandler>();
  }
};

}  // namespace composite
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_NEPTUNE_DATA_COMPOSITE_PROCESSOR_FUNCTION_HANDLER_H_
