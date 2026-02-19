#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_STABLEHLO_LOADER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_STABLEHLO_LOADER_H_

#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"

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
