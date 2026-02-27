#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_OPTIMIZER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_OPTIMIZER_H_

#include <cstdint>
#include <vector>

#include "third_party/absl/status/status.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/tensorflow/compiler/xla/pjrt/proto/compile_options.proto.h"

namespace jsv {

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

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_OPTIMIZER_H_
