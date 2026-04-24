// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/kernel/identity.h"  // from @neptune

#include <memory>
#include <string>

#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_constants.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace neptune {

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

}  // namespace neptune
