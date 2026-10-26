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

#ifndef XLA_SERVING_RUNTIME_XLA_SERVING_DATA_COMPOSITE_PROCESSOR_FUNCTION_HANDLER_H_
#define XLA_SERVING_RUNTIME_XLA_SERVING_DATA_COMPOSITE_PROCESSOR_FUNCTION_HANDLER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/processor/cc/core/composite_processor.h"  // from @xla_serving
#include "protos/manifest.pb.h"  // from @xla_serving
#include "protos/orchestration.pb.h"  // from @xla_serving
#include "runtime/interpreter/interpreter_create_options.h"  // from @xla_serving
#include "runtime/interpreter/optional_function_handler.h"  // from @xla_serving
#include "runtime/interpreter/optional_function_handler_factory.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/kernel/kernel_io_data_type.h"  // from @xla_serving

namespace xla_serving {
namespace composite {

constexpr char kCompositeProcessorMimeType[] =
    "application/protobuf; "
    "type=xla_serving.data.composite.CompositeProcessorConfig";

constexpr char kLegacyCompositeProcessorMimeType[] =
    "application/protobuf; type=jax_data.composite.CompositeProcessorConfig";

// An implementation of `OptionalFunctionHandler` that can handle
// xla_serving::data::composite::CompositeProcessor.
class CompositeProcessorFunctionHandler : public OptionalFunctionHandler {
 public:
  absl::StatusOr<std::unique_ptr<Kernel>> HandleFunction(
      absl::string_view model_path,
      const xla_saved_model_manifest::Function& function,
      const InterpreterCreateOptions& create_options, bool post_data_conversion,
      const xla_saved_model_orchestration::NamedTensorTypeList* named_inputs,
      const xla_saved_model_orchestration::NamedTensorTypeList* named_outputs)
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
}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_XLA_SERVING_DATA_COMPOSITE_PROCESSOR_FUNCTION_HANDLER_H_
