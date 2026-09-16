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

#include "runtime/kernel/kernel.h"  // from @xla_serving

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/flat_signature.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {

absl::Status CancellableKernel::Run(ExecuteContext& context) {
  if (context.run_context.cancellation_context.IsCancelled()) {
    return absl::CancelledError(
        absl::StrCat("Kernel '", Name(), "' is cancelled."));
  }

  std::vector<Value> flat_inputs;
  bool inputs_flattened = false;
  if (context.inputs.empty() && input_signature_ != nullptr) {
    TF_RETURN_IF_ERROR(
        input_signature_->FillArgs(flat_inputs, context.named_values));
    context.inputs = flat_inputs;
    inputs_flattened = true;
  }

  absl::Status status = RunInternal(context);

  if (inputs_flattened) {
    context.inputs = absl::Span<const Value>();
  }

  if (!status.ok()) return status;

  if (inputs_flattened && output_signature_ != nullptr) {
    TF_ASSIGN_OR_RETURN(auto map,
                        output_signature_->UnflattenToMap(context.outputs));
    for (auto& [key, value] : map) {
      if (!context.named_values.insert({key, std::move(value)}).second) {
        return absl::InvalidArgumentError(
            absl::StrCat("Duplicate output key: ", key));
      }
    }
    context.outputs.clear();
  }

  return absl::OkStatus();
}

tsl::Future<> CancellableAsyncKernel::RunAsync(ExecuteContext& context) {
  if (context.run_context.cancellation_context.IsCancelled()) {
    return tsl::Future<void>(absl::CancelledError(
        absl::StrCat("Async kernel '", Name(), "' is cancelled.")));
  }

  std::vector<Value> flat_inputs;
  bool inputs_flattened = false;
  if (context.inputs.empty() && input_signature_ != nullptr) {
    auto status = input_signature_->FillArgs(flat_inputs, context.named_values);
    if (!status.ok()) return tsl::Future<void>(status);
    context.inputs = flat_inputs;
    inputs_flattened = true;
  }

  tsl::Future<> future = RunAsyncInternal(context);

  return future.Map([&context, this, inputs_flattened,
                     keep_alive = std::move(flat_inputs)]() -> absl::Status {
    if (inputs_flattened) {
      context.inputs = absl::Span<const Value>();
    }
    if (inputs_flattened && output_signature_ != nullptr) {
      auto map = output_signature_->UnflattenToMap(context.outputs);
      if (!map.ok()) return map.status();
      for (auto& [key, value] : *map) {
        if (!context.named_values.insert({key, std::move(value)}).second) {
          return absl::InvalidArgumentError(
              absl::StrCat("Duplicate output key: ", key));
        }
      }
      context.outputs.clear();
    }
    return absl::OkStatus();
  });
}

KernelRegistry& GetGlobalKernelRegistry() {
  static absl::NoDestructor<KernelRegistry> global_kernel_registry;
  return *global_kernel_registry;
}

absl::Status KernelRegistry::Register(
    absl::string_view name,
    absl::AnyInvocable<absl::StatusOr<std::unique_ptr<Kernel>>(
        const KernelBuilderContext& context)>
        builder) {
  absl::MutexLock lock(mutex_);
  if (!kernel_builders_.insert({std::string(name), std::move(builder)})
           .second) {
    return absl::AlreadyExistsError(
        absl::StrCat("Kernel ", name, " already registered."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> KernelRegistry::Create(
    absl::string_view name, const KernelBuilderContext& context) {
  absl::MutexLock lock(mutex_);
  auto it = kernel_builders_.find(name);
  if (it == kernel_builders_.end()) {
    std::string hint;
    if (name == "StableHloRunnerOp" || name == "AotStableHloRunnerOp") {
      hint =
          " Did you forget to link "
          "//third_party/neptune/runtime/interpreter:aot_execution or "
          "//third_party/neptune/runtime/interpreter:jit_compilation?";
    }
    return absl::NotFoundError(
        absl::StrCat("Kernel ", name, " not found.", hint));
  }
  return it->second(context);
}

}  // namespace xla_serving
