#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_STABLEHLO_RUNNER_OP_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_STABLEHLO_RUNNER_OP_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/flat_signature.h"  // from @neptune
#include "runtime/core/stablehlo_runner.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune

namespace neptune {

// A kernel that runs a StableHlo computation.
// It takes a list of `jsv::HostBuffer` as inputs and outputs.
// It also supports loading variables from an `jsv::IfrtLoadedVariableRegistry`.
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

 private:
  explicit StableHloRunnerOp(std::unique_ptr<StableHloRunner> stablehlo_runner,
                             std::unique_ptr<FlatSignature> input_signature,
                             std::unique_ptr<FlatSignature> output_signature)
      : stablehlo_runner_(std::move(stablehlo_runner)),
        input_signature_(std::move(input_signature)),
        output_signature_(std::move(output_signature)) {}

  absl::Status RunInternal(ExecuteContext& context) override;

  std::unique_ptr<StableHloRunner> stablehlo_runner_;
  std::unique_ptr<FlatSignature> input_signature_;
  std::unique_ptr<FlatSignature> output_signature_;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_STABLEHLO_RUNNER_OP_H_
