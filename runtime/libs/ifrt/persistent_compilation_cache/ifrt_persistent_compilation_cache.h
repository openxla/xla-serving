#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_PERSISTENT_COMPILATION_CACHE_IFRT_PERSISTENT_COMPILATION_CACHE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_PERSISTENT_COMPILATION_CACHE_IFRT_PERSISTENT_COMPILATION_CACHE_H_

#include <memory>

#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "xla/pjrt/pjrt_executable.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/hlo/hlo_program.h"  // from @xla
#include "xla/python/ifrt/program.h"  // from @xla

namespace jsv {

class IfrtPersistentCompilationCache {
 public:
  IfrtPersistentCompilationCache() = default;
  virtual ~IfrtPersistentCompilationCache() = default;

  // The implementation of this API should be thread-safe. It generates a key
  // for looking up the executable in the persistent cache and it will return
  // the LoadedExecutable if hits cache. Otherwise, it will call the `value_fn`
  // to generate and return the LoadedExecutable.
  virtual absl::StatusOr<xla::ifrt::LoadedExecutableRef>
  LookupLoadedExecutableOrCreate(
      std::unique_ptr<xla::ifrt::HloProgram> hlo_program,
      xla::ifrt::DeviceListRef device_list,
      const xla::CompileOptions& xla_compile_options,
      xla::ifrt::Client* client,
      absl::AnyInvocable<absl::StatusOr<xla::ifrt::LoadedExecutableRef>(
          std::unique_ptr<xla::ifrt::Program> program,
          std::unique_ptr<xla::ifrt::CompileOptions> options)>
          value_fn);
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_PERSISTENT_COMPILATION_CACHE_IFRT_PERSISTENT_COMPILATION_CACHE_H_
