#include "runtime/kernel/builtin_kernel.h"  // from @neptune

#include "absl/status/status.h"  // from @abseil-cpp
#include "runtime/kernel/batching/batch_runner_op.h"  // from @neptune
#include "runtime/kernel/identity.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/stablehlo_runner_op.h"  // from @neptune
#include "xla/tsl/platform/errors.h"  // from @xla

namespace neptune {

absl::Status RegisterBuiltinKernels(KernelRegistry& registry) {
  TF_RETURN_IF_ERROR(registry.Register("Identity", IdentityKernel::Create));
  TF_RETURN_IF_ERROR(
      registry.Register("StableHloRunnerOp", StableHloRunnerOp::Create));
  TF_RETURN_IF_ERROR(registry.Register("BatchRunnerOp", BatchRunnerOp::Create));
  return absl::OkStatus();
}

}  // namespace neptune
