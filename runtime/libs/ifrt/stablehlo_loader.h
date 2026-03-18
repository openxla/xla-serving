#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_STABLEHLO_LOADER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_STABLEHLO_LOADER_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project

namespace jsv {

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

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_STABLEHLO_LOADER_H_
