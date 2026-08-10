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

#include "runtime/neptune_data/composite_processor_op.h"  // from @neptune

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/value.h"  // from @neptune
#include "data/processor/cc/core/composite_processor.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune

namespace neptune {
namespace composite {

CompositeProcessorOp::CompositeProcessorOp(
    absl_nonnull
    std::unique_ptr<neptune::data::composite::core::CompositeProcessor>
        processor)
    : processor_(std::move(processor)) {}

std::string CompositeProcessorOp::Name() const {
  return "CompositeProcessorOp";
}

Kernel::IODataTypes CompositeProcessorOp::GetIODataTypes() const {
  return {.input = KernelIODataType::kHostBuffer,
          .output = KernelIODataType::kHostBuffer};
}

int CompositeProcessorOp::GetNumInputs() const {
  return processor_->GetNumInputs();
}

int CompositeProcessorOp::GetNumOutputs() const {
  return processor_->GetNumOutputs();
}

absl::Status CompositeProcessorOp::RunInternal(ExecuteContext& context) {
  neptune::data::composite::api::ExecuteContext jax_data_context{
      .metadata =
          {
              .model_name = context.metadata.model_name,
              .model_version = context.metadata.model_version,
              .trace_id = context.metadata.trace_id,
          },
  };

  std::vector<neptune::data::composite::api::Value> converted_inputs;
  converted_inputs.reserve(context.inputs.size());
  for (const auto& input : context.inputs) {
    if (!input.IsType<neptune::HostBuffer>()) {
      return absl::InvalidArgumentError(
          "Input to CompositeProcessorOp is not a HostBuffer");
    }
    // Should still be zero-copy as HostBuffer as copying a HostBuffer
    // does not copy the underlying buffer.
    // TODO: Ideally we merge jax_data::composite::api::Value
    //                    back into neptune::Value, as incrementing an atomic
    //                    ref counter is not as cheap as moving the underlying
    //                    HostBuffer.
    converted_inputs.push_back(
        neptune::data::composite::api::Value(input.Get<neptune::HostBuffer>()));
  }

  jax_data_context.inputs = absl::MakeSpan(converted_inputs);

  absl::Status status = processor_->Process(jax_data_context);
  if (!status.ok()) {
    return status;
  }

  for (auto& output : jax_data_context.outputs) {
    if (!output.IsType<neptune::HostBuffer>()) {
      return absl::InternalError(
          "Output of CompositeProcessor is not a HostBuffer");
    }
    context.outputs.push_back(
        neptune::Value(std::move(output.Get<neptune::HostBuffer>())));
  }

  return absl::OkStatus();
}

}  // namespace composite
}  // namespace neptune
