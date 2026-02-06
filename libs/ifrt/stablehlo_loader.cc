#include "third_party/neptune/libs/ifrt/stablehlo_loader.h"

#include <string>

#include "file/base/helpers.h"
#include "file/base/options.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Shape/IR/Shape.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinAttributes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinDialect.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/DialectRegistry.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/InitAllDialects.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Pass/PassManager.h"
#include "third_party/stablehlo/stablehlo/dialect/ChloOps.h"
#include "third_party/stablehlo/stablehlo/dialect/Register.h"
#include "third_party/stablehlo/stablehlo/dialect/StablehloOps.h"
#include "third_party/tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"

namespace jsv {
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
  TF_ASSIGN_OR_RETURN(std::string stablehlo_str,
                      file::GetContents(mlir_path, file::Defaults()));

  return Load(stablehlo_str);
}

}  // namespace jsv
