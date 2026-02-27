#include "third_party/neptune/runtime/core/stablehlo_optimizer.h"

#include "third_party/absl/status/status.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"

namespace jsv {

absl::Status StablehloOptimizer::Optimize(const Args& args,
                                          mlir::ModuleOp module) {
  // By default, do nothing.
  return absl::OkStatus();
}

}  // namespace jsv
