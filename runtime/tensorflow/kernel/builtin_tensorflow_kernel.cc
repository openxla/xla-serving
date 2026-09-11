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

#include "runtime/tensorflow/kernel/builtin_tensorflow_kernel.h"  // from @xla_serving

#include "absl/status/status.h"  // from @abseil-cpp
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/tensorflow/kernel/tf_saved_model_function_runner_op.h"  // from @xla_serving
#include "xla/tsl/platform/errors.h"  // from @xla

namespace xla_serving {
absl::Status RegisterBuiltinTensorflowKernels(KernelRegistry& registry) {
  TF_RETURN_IF_ERROR(registry.Register("TfSavedModelFunctionRunnerOp",
                                       TfSavedModelFunctionRunnerOp::Create));
  return absl::OkStatus();
}
}  // namespace xla_serving
