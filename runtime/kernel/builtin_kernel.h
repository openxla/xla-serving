#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BUILTIN_KERNEL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BUILTIN_KERNEL_H_

#include "third_party/absl/status/status.h"
#include "third_party/neptune/runtime/kernel/kernel.h"

namespace jsv {

absl::Status RegisterBuiltinKernels(KernelRegistry& registry);
}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BUILTIN_KERNEL_H_
