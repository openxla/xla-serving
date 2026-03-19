#include "runtime/libs/ifrt/stablehlo_loader.h"  // from @neptune

#include <string>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/include/mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/pjrt/mlir_to_hlo.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace neptune {
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> StableHloLoader::Load(
    absl::string_view stablehlo_str) {
  // Load a superset of dialects; we should check at serialization time that
  // we only include allowable dialects.
  context_.loadDialect<mlir::func::FuncDialect>();
  context_.loadDialect<mlir::stablehlo::StablehloDialect>();
  context_.loadDialect<mlir::chlo::ChloDialect>();
  // Register dialect extensions.
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::registerAllDialects(registry);

  context_.appendDialectRegistry(registry);

  TF_ASSIGN_OR_RETURN(auto module,
                      xla::ParseMlirModuleString(stablehlo_str, context_));

  context_.loadAllAvailableDialects();
  return module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> StableHloLoader::LoadFromFile(
    absl::string_view mlir_path) {
  std::string stablehlo_str;
  CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), std::string(mlir_path),
                                   &stablehlo_str));

  return Load(stablehlo_str);
}

}  // namespace neptune
