#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_EXECUTABLE_BUILD_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_EXECUTABLE_BUILD_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @neptune
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/service/compilation_environments.h"  // from @xla
#include "xla/service/computation_placer.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace neptune {

constexpr absl::string_view kMissingProgramFingerprint =
    "MISSING_PROGRAM_FINGERPRINT";
constexpr absl::string_view kProgramFingerprintAndTimestampDelimiter = "@";

struct BuildIfrtExecutableOptions {
  std::optional<xla::CompilationEnvironments> compilation_environments;
  int64_t num_replicas;
  int64_t num_cores_per_replica;
  // If false, `device_assignment` must be set.
  bool compile_portable_executable = false;
  std::optional<xla::DeviceAssignment> device_assignment;
  xla::CompileOptionsProto* compile_options = nullptr;
};

absl::StatusOr<xla::ifrt::LoadedExecutableRef> BuildIfrtExecutable(
    mlir::ModuleOp stablehlo_module, const BuildIfrtExecutableOptions& options,
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    IfrtPersistentCompilationCache* persistent_compilation_cache);

// Returns a string in the format of "{fingerprint}@{timestamp}" where the
// fingerprint is the canonical program fingerprint of the `ifrt_executable` and
// the timestamp is `now` in microseconds. If the canonical program fingerprint
// is not available, the string will be
// "MISSING_PROGRAM_FINGERPRINT@{timestamp}".
std::string GetProgramFingerprintWithTimestamp(
    absl_nonnull xla::ifrt::LoadedExecutableRef ifrt_executable,
    absl::Time now);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_EXECUTABLE_BUILD_UTILS_H_
