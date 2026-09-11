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

#ifndef XLA_SERVING_RUNTIME_KERNEL_BUILTIN_KERNEL_H_
#define XLA_SERVING_RUNTIME_KERNEL_BUILTIN_KERNEL_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "runtime/kernel/kernel.h"  // from @neptune

namespace xla_serving {

absl::Status RegisterBuiltinKernels(KernelRegistry& registry);
}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_KERNEL_BUILTIN_KERNEL_H_
