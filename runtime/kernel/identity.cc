#include "runtime/kernel/identity.h"  // from @neptune

#include <memory>
#include <string>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_constants.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace jsv {

absl::StatusOr<std::unique_ptr<Kernel>> IdentityKernel::Create(
    const KernelBuilderContext& context) {
  TF_ASSIGN_OR_RETURN(int num_host_buffers,
                      context.GetConfig<int>(kNumKernelInputs));
  if (num_host_buffers <= 0) {
    return absl::InvalidArgumentError(
        "Number of host buffers must be positive.");
  }
  return absl::WrapUnique(new IdentityKernel(num_host_buffers));
}

std::string IdentityKernel::Name() const { return "identity"; }

absl::Status IdentityKernel::RunInternal(ExecuteContext& context) {
  if (context.inputs.size() != GetNumInputs()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of inputs (", context.inputs.size(),
        " does not match expected number of inputs (", GetNumInputs(), ")"));
  }

  context.outputs.assign(context.inputs.begin(), context.inputs.end());
  return absl::OkStatus();
}

}  // namespace jsv
