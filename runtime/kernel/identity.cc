#include "third_party/neptune/runtime/kernel/identity.h"

#include <memory>
#include <string>

#include "third_party/absl/memory/memory.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/neptune/runtime/kernel/kernel.h"
#include "third_party/neptune/runtime/kernel/kernel_builder_constants.h"
#include "third_party/neptune/runtime/kernel/kernel_builder_context.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"

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
