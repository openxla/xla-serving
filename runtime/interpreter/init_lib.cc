#include "absl/log/check.h"  // from @abseil-cpp
#include "runtime/kernel/builtin_kernel.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune

namespace neptune {
namespace {

// TODO: Find a better way to register the kernels
void RegisterBuiltinKernelsOrDie() {
  CHECK_OK(RegisterBuiltinKernels(GetGlobalKernelRegistry()));
}

// Register the builtin kernels at link time.
static const bool kUnused = []() {
  RegisterBuiltinKernelsOrDie();
  return true;
}();

}  // namespace

}  // namespace neptune
