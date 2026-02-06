#include "third_party/neptune/libs/ifrt/executable_build_utils.h"

#include <memory>
#include <string>
#include <utility>

#include "third_party/absl/base/nullability.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/time/time.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/neptune/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device_list.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/executable.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/hlo/hlo_program.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/program.h"
#include "third_party/tensorflow/compiler/xla/python/pjrt_ifrt/xla_compiler.h"
#include "third_party/tensorflow/compiler/xla/service/compilation_environments.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

namespace jsv {

namespace {

xla::CompileOptions GetDefaultCompileOptions(
    const BuildIfrtExecutableOptions& options) {
  xla::CompileOptions xla_compile_options;
  if (options.compilation_environments) {
    *xla_compile_options.executable_build_options.mutable_comp_envs() =
        *options.compilation_environments;
  }
  xla_compile_options.executable_build_options.set_num_replicas(
      options.num_replicas);
  xla_compile_options.executable_build_options.set_num_partitions(
      options.num_cores_per_replica);
  xla_compile_options.executable_build_options.set_use_spmd_partitioning(true);
  xla_compile_options.parameter_is_tupled_arguments = false;
  xla_compile_options.compile_portable_executable =
      options.compile_portable_executable;
  if (xla_compile_options.compile_portable_executable) {
    xla_compile_options.executable_build_options.clear_device_assignment();
  } else {
    xla_compile_options.executable_build_options.set_device_assignment(
        *options.device_assignment);
  }
  return xla_compile_options;
}

}  // namespace

absl::StatusOr<xla::ifrt::LoadedExecutableRef> BuildIfrtExecutable(
    mlir::ModuleOp stablehlo_module, const BuildIfrtExecutableOptions& options,
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    IfrtPersistentCompilationCache* persistent_compilation_cache) {
  xla::CompileOptions xla_compile_options;
  if (options.compile_options == nullptr) {
    xla_compile_options = GetDefaultCompileOptions(options);
  } else {
// Some CompilationEnvironment protos used by libtpu are not compiled into the
// binary, so we need to clear them when compiling on GCE.
// TODO: Support compilation environment protos on GCE.
#ifdef JSV_ON_GCE
    if (options.compile_options->has_executable_build_options() &&
        options.compile_options->executable_build_options().has_comp_envs()) {
      options.compile_options->mutable_executable_build_options()
          ->clear_comp_envs();
    }
#endif
    TF_ASSIGN_OR_RETURN(xla_compile_options, xla::CompileOptions::FromProto(
                                                 *options.compile_options));
    // TODO: Remove these overrides once the compile options are
    // fully supported.
    xla_compile_options.executable_build_options.set_num_replicas(
        options.num_replicas);
    xla_compile_options.executable_build_options.set_num_partitions(
        options.num_cores_per_replica);
    xla_compile_options.executable_build_options.set_use_spmd_partitioning(
        true);
    xla_compile_options.parameter_is_tupled_arguments = false;
    if (options.device_assignment.has_value()) {
      xla_compile_options.executable_build_options.set_device_assignment(
          *options.device_assignment);
    } else {
      xla_compile_options.executable_build_options.clear_device_assignment();
      xla_compile_options.compile_portable_executable = true;
    }
  }
  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef devices,
                      xla::ifrt::GetDeviceListFromXlaCompileOptions(
                          ifrt_client.get(), xla_compile_options));

  auto hlo_program = std::make_unique<xla::ifrt::HloProgram>(stablehlo_module);

  auto execute_compile_and_load =
      [&](std::unique_ptr<xla::ifrt::Program> program,
          std::unique_ptr<xla::ifrt::CompileOptions> options)
      -> absl::StatusOr<xla::ifrt::LoadedExecutableRef> {
    return ifrt_client->GetDefaultCompiler()
        ->CompileAndLoad(std::move(program), std::move(options))
        .Await();
  };

  if (persistent_compilation_cache == nullptr) {
    return execute_compile_and_load(
        std::move(hlo_program), std::make_unique<xla::ifrt::XlaCompileOptions>(
                                    xla_compile_options, devices));
  }

  return persistent_compilation_cache->LookupLoadedExecutableOrCreate(
      std::move(hlo_program), devices, xla_compile_options, ifrt_client.get(),
      std::move(execute_compile_and_load));
}

std::string GetProgramFingerprintWithTimestamp(
    absl_nonnull xla::ifrt::LoadedExecutableRef ifrt_executable,
    absl::Time now) {
  std::string program_fingerprint = std::string(kMissingProgramFingerprint);
  const auto now_timestamp = absl::ToUnixMicros(now);
  if (const auto canonical_program_fingerprint = ifrt_executable->Fingerprint();
      canonical_program_fingerprint.ok() &&
      canonical_program_fingerprint->has_value()) {
    program_fingerprint = canonical_program_fingerprint->value();
  }
  return absl::StrCat(program_fingerprint,
                      kProgramFingerprintAndTimestampDelimiter, now_timestamp);
}

}  // namespace jsv
