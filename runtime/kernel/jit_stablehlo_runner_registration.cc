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

#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_optimizer.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner_base.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_constants.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "runtime/kernel/stablehlo_runner_op.h"  // from @xla_serving
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @xla_serving
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @xla_serving
#include "google/protobuf/message.h"
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

// Factory function that creates a JIT `StableHloRunner` instance.
absl::StatusOr<std::unique_ptr<StableHloRunnerBase>> CreateJitRunner(
    const KernelBuilderContext& context, std::shared_ptr<StableHlo> stablehlo,
    StableHloRunnerOptions base_options) {
  absl::StatusOr<ServingCoreSelector*> serving_core_selector_status =
      context.GetConfig<ServingCoreSelector*>(kServingCoreSelector);
  ServingCoreSelector* serving_core_selector = nullptr;
  if (serving_core_selector_status.ok()) {
    serving_core_selector = *serving_core_selector_status;
  } else if (!absl::IsNotFound(serving_core_selector_status.status())) {
    return serving_core_selector_status.status();
  }

  TF_ASSIGN_OR_RETURN(
      IfrtPersistentCompilationCache * persistent_compilation_cache,
      context.GetConfig<IfrtPersistentCompilationCache*>(
          kPersistentCompilationCache));
  TF_ASSIGN_OR_RETURN(
      tsl::protobuf::Message * compilation_environment_proto,
      context.GetConfig<tsl::protobuf::Message*>(kCompilationEnvProto));

  absl::StatusOr<xla::CompileOptionsProto> compile_options =
      context.GetConfig<xla::CompileOptionsProto>(kXlaCompileOptions);
  // If the compile options is not found, we will use the default compile
  // options.
  if (!compile_options.ok() && !absl::IsNotFound(compile_options.status())) {
    return compile_options.status();
  }

  absl::StatusOr<StablehloOptimizer*> optimizer =
      context.GetConfig<StablehloOptimizer*>(kStablehloOptimizer);
  // If StablehloOptimizer is not set in the kernel builder context, we will
  // not use the optimizer.
  if (absl::IsNotFound(optimizer.status())) {
    optimizer = nullptr;
  } else if (!optimizer.ok()) {
    return optimizer.status();
  }

  return StableHloRunner::Create(
      std::move(stablehlo),
      compile_options.ok() ? std::make_optional(*compile_options)
                           : std::nullopt,
      std::move(base_options.client), persistent_compilation_cache,
      base_options.ifrt_loaded_variable_registry,
      std::move(base_options.weights), compilation_environment_proto,
      base_options.thread_pool, *optimizer,
      std::move(base_options.assigned_devices), serving_core_selector,
      /*h2d_transfer_executor_factory=*/nullptr, base_options.callback_id);
}

// Registers the JIT `StableHloRunner` factory at link time.
static const bool kRegisterJitRunner = []() {
  CHECK_OK(StableHloRunnerFactoryRegistry::GetGlobalRegistry().Register(
      StableHloRunnerMode::kJit, CreateJitRunner));
  return true;
}();

}  // namespace
}  // namespace xla_serving
