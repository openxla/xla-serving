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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_STABLEHLO_LOADER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_STABLEHLO_LOADER_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project

namespace neptune {

class StableHloLoader {
 public:
  StableHloLoader() = default;

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Load(
      absl::string_view stablehlo_str);
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> LoadFromFile(
      absl::string_view mlir_path);

 private:
  // TODO: Pass in thread pool to avoid creating a new pool on
  // demand.
  mlir::MLIRContext context_{mlir::MLIRContext::Threading::ENABLED};
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_STABLEHLO_LOADER_H_
