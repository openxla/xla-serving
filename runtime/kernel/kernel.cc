#include "third_party/neptune/runtime/kernel/kernel.h"

#include <memory>
#include <string>
#include <utility>

#include "third_party/absl/base/no_destructor.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/functional/any_invocable.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/neptune/runtime/kernel/kernel_builder_context.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"

namespace jsv {

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

}  // namespace jsv
