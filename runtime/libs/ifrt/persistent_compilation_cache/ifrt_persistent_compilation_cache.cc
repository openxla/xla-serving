#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @neptune

#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "xla/pjrt/pjrt_executable.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/hlo/hlo_program.h"  // from @xla
#include "xla/python/ifrt/program.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_compiler.h"  // from @xla

namespace jsv {

absl::StatusOr<xla::ifrt::LoadedExecutableRef>
IfrtPersistentCompilationCache::LookupLoadedExecutableOrCreate(
    std::unique_ptr<xla::ifrt::HloProgram> hlo_program,
    xla::ifrt::DeviceListRef device_list,
    const xla::CompileOptions& xla_compile_options,
    xla::ifrt::Client* client,
    absl::AnyInvocable<absl::StatusOr<xla::ifrt::LoadedExecutableRef>(
        std::unique_ptr<xla::ifrt::Program> program,
        std::unique_ptr<xla::ifrt::CompileOptions> options)>
        value_fn) {
  // No persistent cache implemented, compile directly.
  auto ifrt_xla_compile_options =
      std::make_unique<xla::ifrt::XlaCompileOptions>(xla_compile_options,
                                                     std::move(device_list));
  return value_fn(std::move(hlo_program), std::move(ifrt_xla_compile_options));
}

}  // namespace jsv
