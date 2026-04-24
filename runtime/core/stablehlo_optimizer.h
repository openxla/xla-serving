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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_OPTIMIZER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_OPTIMIZER_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla

namespace neptune {

// An interface for optimizing StableHLO models. All implementation must be
// thread-safe and stateless.
class StablehloOptimizer {
 public:
  struct Args {
    // The indices of the arguments in the module that are variables.
    std::vector<int64_t> variable_indices;
    xla::CompileOptionsProto* compile_options = nullptr;
  };
  virtual ~StablehloOptimizer() = default;
  // Optimizes the StableHLO model. The input `module` must not have dynamic
  // shapes.
  virtual absl::Status Optimize(const Args& args, mlir::ModuleOp module);
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_OPTIMIZER_H_
