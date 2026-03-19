#include "runtime/kernel/kernel.h"  // from @neptune

#include <memory>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "xla/pjrt/pjrt_future.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla

namespace neptune {

absl::Status CancellableKernel::Run(ExecuteContext& context) {
  if (context.run_context.cancellation_context.IsCancelled()) {
    return absl::CancelledError(
        absl::StrCat("Kernel '", Name(), "' is cancelled."));
  }
  return RunInternal(context);
}

tsl::Future<> CancellableAsyncKernel::RunAsync(ExecuteContext& context) {
  // Because the kernel asynchronous execution is defined in the implementation,
  // we can only cancel the kernel at the point this kernel is launched, which
  // is likely before any async work has happened. Ideally, we also check for
  // cancellation upon actually executing the asynchronous work.
  if (context.run_context.cancellation_context.IsCancelled()) {
    return tsl::Future<void>(absl::CancelledError(
        absl::StrCat("Async kernel '", Name(), "' is cancelled.")));
  }
  return RunAsyncInternal(context);
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
    return absl::NotFoundError(absl::StrCat("Kernel ", name, " not found."));
  }
  return it->second(context);
}

}  // namespace neptune
