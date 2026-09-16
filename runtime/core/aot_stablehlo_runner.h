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

#ifndef XLA_SERVING_RUNTIME_CORE_AOT_STABLEHLO_RUNNER_H_
#define XLA_SERVING_RUNTIME_CORE_AOT_STABLEHLO_RUNNER_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner_base.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/file_system.h"  // from @xla

namespace xla_serving {

// Configuration options for creating a StableHloRunner from pre-compiled AOT
// executables.
// Represents the initialization options for an AOT-loaded StableHloRunner.
// `executables`, `module_kept_var_indices`, and `io_specs` are parallel
// arrays representing one entry per shape bucket (if dynamic shape bucketings
// are used). `weights` represents the names of all weights before any Dead
// Code Elimination (DCE) is applied. `aot_memory_regions` holds pre-loaded AOT
// memory buffers whose lifetime is tied to this runner.
struct StableHloRunnerAotOptions {
  StableHloRunnerOptions base_options;
  // Array of pre-compiled executables corresponding to distinct shape buckets
  // for a single platform (parallel to io_specs and module_kept_var_indices).
  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  // Original StableHlo capturing all specs
  std::shared_ptr<StableHlo> stablehlo;
  // Backing memory mapped regions.
  std::vector<std::shared_ptr<tsl::ReadOnlyMemoryRegion>> aot_memory_regions;
};

class AotStableHloRunner : public StableHloRunnerBase {
 public:
  // Statically constructs a StableHloRunner entirely driven by ahead-of-time
  // (AOT) loaded modules and weights, decoupling it from dynamic MLIR compiler
  // initialization when targeting underlying IFRT primitives.
  static absl::StatusOr<std::unique_ptr<AotStableHloRunner>> Create(
      StableHloRunnerAotOptions options);

 private:
  // Inherit constructor access via friendship
  using StableHloRunnerBase::StableHloRunnerBase;

  tsl::Future<SharedCachedExecutableBundle> LookUpOrCreateExecutable(
      absl::Span<const xla::ifrt::Shape> input_shapes,
      absl::Span<const xla::ifrt::DType> input_dtypes) override;

  // Bootstraps the IFRT execution cache with pre-compiled AOT executable
  // bundles and directly synthesizes IFRT ShardingRefs statically, bypassing
  // MLIR compilation completely during hot-path execution.
  absl::Status PopulateAotExecutableCache(StableHloRunnerAotOptions options);

  // Keep memory regions for AOT executables alive.
  std::vector<std::shared_ptr<tsl::ReadOnlyMemoryRegion>> aot_memory_regions_;

  friend class StableHloRunnerTestPeer;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_AOT_STABLEHLO_RUNNER_H_
