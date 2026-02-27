#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/LogicalResult.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Shape/IR/Shape.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/DialectRegistry.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/InitAllDialects.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"
#include "third_party/stablehlo/stablehlo/dialect/Register.h"

int main(int argc, char **argv) {
  // Register dialect extensions.
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::registerAllDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "JSV StableHLO pass driver\n", registry));
}
