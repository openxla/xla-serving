#include "third_party/neptune/runtime/kernel/builtin_kernel.h"

#include "third_party/absl/status/status.h"
#include "third_party/neptune/runtime/kernel/batching/batch_runner_op.h"
#include "third_party/neptune/runtime/kernel/identity.h"
#include "third_party/neptune/runtime/kernel/kernel.h"
#include "third_party/neptune/runtime/kernel/stablehlo_runner_op.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/errors.h"

namespace jsv {

absl::Status RegisterBuiltinKernels(KernelRegistry& registry) {
  TF_RETURN_IF_ERROR(registry.Register("Identity", IdentityKernel::Create));
  TF_RETURN_IF_ERROR(
      registry.Register("StableHloRunnerOp", StableHloRunnerOp::Create));
  TF_RETURN_IF_ERROR(registry.Register("BatchRunnerOp", BatchRunnerOp::Create));
  return absl::OkStatus();
}

}  // namespace jsv
