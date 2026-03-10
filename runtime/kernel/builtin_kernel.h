#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BUILTIN_KERNEL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BUILTIN_KERNEL_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/kernel/kernel.h"  // from @neptune

namespace jsv {

absl::Status RegisterBuiltinKernels(KernelRegistry& registry);
}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BUILTIN_KERNEL_H_
