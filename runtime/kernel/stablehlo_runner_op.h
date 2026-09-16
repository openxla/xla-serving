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

#ifndef XLA_SERVING_RUNTIME_KERNEL_STABLEHLO_RUNNER_OP_H_
#define XLA_SERVING_RUNTIME_KERNEL_STABLEHLO_RUNNER_OP_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/flat_signature.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner_base.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "runtime/kernel/kernel_io_data_type.h"  // from @xla_serving

namespace xla_serving {

// A kernel that runs a StableHlo computation.
// It takes a list of `xla_serving::HostBuffer` as inputs and outputs.
// It also supports loading variables from an `xla_serving::IfrtLoadedVariableRegistry`.
// The variable names are provided as a list of strings. The variable names are
// expected to be in the same order as the variable arguments in the
// computation. It is also assumed that variables will always be at the
// beginning of the argument list. The input arguments will follow the
// variables.
class StableHloRunnerOp : public CancellableKernel {
 public:
  static absl::StatusOr<std::unique_ptr<Kernel>> Create(
      const KernelBuilderContext& context);

  ~StableHloRunnerOp() override = default;

  std::string Name() const override;

  Kernel::IODataTypes GetIODataTypes() const override {
    return {.input = KernelIODataType::kHostBuffer,
            .output = KernelIODataType::kHostBuffer};
  }

  int GetNumInputs() const override {
    return stablehlo_runner_->GetNumInputs();
  }

  int GetNumOutputs() const override {
    return stablehlo_runner_->GetNumOutputs();
  }

  absl::Status MarkFullyInitialized() override {
    return stablehlo_runner_->MarkFullyInitialized();
  }

  bool has_streaming() const override {
    return stablehlo_runner_->has_streaming();
  }

  std::optional<int64_t> streaming_host_callback_id() const {
    return stablehlo_runner_->streaming_host_callback_id();
  }

 private:
  explicit StableHloRunnerOp(
      std::unique_ptr<StableHloRunnerBase> stablehlo_runner,
      std::unique_ptr<FlatSignature> input_signature,
      std::unique_ptr<FlatSignature> output_signature)
      : CancellableKernel(std::move(input_signature),
                          std::move(output_signature)),
        stablehlo_runner_(std::move(stablehlo_runner)) {}

  absl::Status RunInternal(ExecuteContext& context) override;

  std::unique_ptr<StableHloRunnerBase> stablehlo_runner_;

  friend class StableHloRunnerTestPeer;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_KERNEL_STABLEHLO_RUNNER_OP_H_
