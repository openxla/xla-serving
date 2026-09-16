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

#ifndef XLA_SERVING_RUNTIME_KERNEL_AOT_STABLEHLO_RUNNER_OP_H_
#define XLA_SERVING_RUNTIME_KERNEL_AOT_STABLEHLO_RUNNER_OP_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/aot_stablehlo_runner.h"  // from @xla_serving
#include "runtime/core/flat_signature.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "runtime/kernel/kernel_io_data_type.h"  // from @xla_serving

namespace xla_serving {

// A kernel that runs an AOT compiled StableHlo computation.
// It takes a list of `xla_serving::HostBuffer` as inputs and outputs.
// It also supports loading variables from an `xla_serving::IfrtLoadedVariableRegistry`.
class AotStableHloRunnerOp : public CancellableKernel {
 public:
  static absl::StatusOr<std::unique_ptr<Kernel>> Create(
      const KernelBuilderContext& context);

  ~AotStableHloRunnerOp() override = default;

  std::string Name() const override;

  Kernel::IODataTypes GetIODataTypes() const override {
    return {.input = KernelIODataType::kHostBuffer,
            .output = KernelIODataType::kHostBuffer};
  }

  int GetNumInputs() const override { return runner_->GetNumInputs(); }

  int GetNumOutputs() const override { return runner_->GetNumOutputs(); }

  absl::Status MarkFullyInitialized() override {
    return runner_->MarkFullyInitialized();
  }

  bool has_streaming() const override { return runner_->has_streaming(); }

  std::optional<int64_t> streaming_host_callback_id() const {
    return runner_->streaming_host_callback_id();
  }

 private:
  explicit AotStableHloRunnerOp(std::unique_ptr<AotStableHloRunner> runner,
                                std::unique_ptr<FlatSignature> input_signature,
                                std::unique_ptr<FlatSignature> output_signature)
      : CancellableKernel(std::move(input_signature),
                          std::move(output_signature)),
        runner_(std::move(runner)) {}

  absl::Status RunInternal(ExecuteContext& context) override;

  std::unique_ptr<AotStableHloRunner> runner_;
};

absl::Status RegisterAotStableHloRunnerOp(KernelRegistry& registry);

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_KERNEL_AOT_STABLEHLO_RUNNER_OP_H_
