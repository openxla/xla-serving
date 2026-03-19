#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BUILTIN_KERNEL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BUILTIN_KERNEL_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "runtime/kernel/kernel.h"  // from @neptune

namespace neptune {

absl::Status RegisterBuiltinKernels(KernelRegistry& registry);
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BUILTIN_KERNEL_H_
