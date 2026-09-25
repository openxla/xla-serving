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
#include <utility>
#include <vector>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/aot_stablehlo_runner.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner_base.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_constants.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "runtime/kernel/stablehlo_runner_op.h"  // from @xla_serving
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/tsl/platform/file_system.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

// Factory function that creates an AOT `AotStableHloRunner` instance.
absl::StatusOr<std::unique_ptr<StableHloRunnerBase>> CreateAotRunner(
    const KernelBuilderContext& context, std::shared_ptr<StableHlo> stablehlo,
    StableHloRunnerOptions base_options) {
  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::LoadedExecutableRef> loaded_executables,
      context.GetConfig<std::vector<xla::ifrt::LoadedExecutableRef>>(
          kIfrtLoadedExecutables));

  std::vector<std::shared_ptr<tsl::ReadOnlyMemoryRegion>> aot_memory_regions;
  auto aot_memory_regions_config =
      context
          .GetConfig<std::vector<std::shared_ptr<tsl::ReadOnlyMemoryRegion>>>(
              kIfrtAotMemoryRegions);
  if (aot_memory_regions_config.ok()) {
    aot_memory_regions = std::move(*aot_memory_regions_config);
  } else if (!absl::IsNotFound(aot_memory_regions_config.status())) {
    return aot_memory_regions_config.status();
  }

  StableHloRunnerAotOptions aot_options{
      .base_options = std::move(base_options),
      .executables = std::move(loaded_executables),
      .stablehlo = std::move(stablehlo),
      .aot_memory_regions = std::move(aot_memory_regions),
  };
  return AotStableHloRunner::Create(std::move(aot_options));
}

// Registers the AOT `AotStableHloRunner` factory at link time.
static const bool kRegisterAotRunner = []() {
  CHECK_OK(StableHloRunnerFactoryRegistry::GetGlobalRegistry().Register(
      StableHloRunnerMode::kAot, CreateAotRunner));
  return true;
}();

}  // namespace
}  // namespace xla_serving
