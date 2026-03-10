#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo

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
