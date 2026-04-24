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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_STABLEHLO_TRANSFORMS_TPU_STABLEHLO_TPU_OPTIMIZER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_STABLEHLO_TRANSFORMS_TPU_STABLEHLO_TPU_OPTIMIZER_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "runtime/core/stablehlo_optimizer.h"  // from @neptune

namespace neptune {

class StablehloTpuOptimizer : public StablehloOptimizer {
 public:
  absl::Status Optimize(const StablehloOptimizer::Args& args,
                        mlir::ModuleOp module) override;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_STABLEHLO_TRANSFORMS_TPU_STABLEHLO_TPU_OPTIMIZER_H_
