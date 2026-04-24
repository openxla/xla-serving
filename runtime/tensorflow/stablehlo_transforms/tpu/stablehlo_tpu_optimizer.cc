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

#include "runtime/tensorflow/stablehlo_transforms/tpu/stablehlo_tpu_optimizer.h"  // from @neptune

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "runtime/core/stablehlo_optimizer.h"  // from @neptune
#include "runtime/tensorflow/stablehlo_transforms/tpu/tpu_input_r1_optimization.h"  // from @neptune
#include "xla/mlir/utils/error_util.h"  // from @xla

namespace neptune {

absl::Status StablehloTpuOptimizer::Optimize(
    const StablehloOptimizer::Args& args, mlir::ModuleOp module) {
  mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());

  mlir::PassManager pm(module.getContext());
  pm.addPass(CreateTpuInputR1OptimizationPass(args.variable_indices,
                                              args.compile_options));
  if (mlir::failed(pm.run(module))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to optimize StableHLO module: ",
                     diag_handler.ConsumeStatus().message()));
  }
  return absl::OkStatus();
}

}  // namespace neptune
