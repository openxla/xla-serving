#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_AOT_STABLEHLO_RUNNER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_AOT_STABLEHLO_RUNNER_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/stablehlo.h"  // from @neptune
#include "runtime/core/stablehlo_runner.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/file_system.h"  // from @xla

namespace neptune {

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

class AotStableHloRunner : public StableHloRunner {
 public:
  // Statically constructs a StableHloRunner entirely driven by ahead-of-time
  // (AOT) loaded modules and weights, decoupling it from dynamic MLIR compiler
  // initialization when targeting underlying IFRT primitives.
  static absl::StatusOr<std::unique_ptr<AotStableHloRunner>> Create(
      StableHloRunnerAotOptions options);

 private:
  // Inherit constructor access via friendship
  using StableHloRunner::StableHloRunner;

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

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_AOT_STABLEHLO_RUNNER_H_
