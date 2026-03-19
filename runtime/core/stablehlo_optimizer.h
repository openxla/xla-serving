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
