#ifndef THIRD_PARTY_NEPTUNE_IFRT_EXECUTABLE_BUILD_UTILS_H_
#define THIRD_PARTY_NEPTUNE_IFRT_EXECUTABLE_BUILD_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "third_party/absl/base/nullability.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/time/time.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/neptune/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"
#include "third_party/tensorflow/compiler/xla/pjrt/proto/compile_options.proto.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/executable.h"
#include "third_party/tensorflow/compiler/xla/service/compilation_environments.h"
#include "third_party/tensorflow/compiler/xla/service/computation_placer.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

namespace jsv {

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

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_IFRT_EXECUTABLE_BUILD_UTILS_H_
