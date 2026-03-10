#include "runtime/core/stablehlo_optimizer.h"  // from @neptune

#include "absl/status/status.h"  // from @com_google_absl
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace jsv {

absl::Status StablehloOptimizer::Optimize(const Args& args,
                                          mlir::ModuleOp module) {
  // By default, do nothing.
  return absl::OkStatus();
}

}  // namespace jsv
