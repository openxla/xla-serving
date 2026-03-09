#include "third_party/absl/log/check.h"
#include "third_party/neptune/runtime/kernel/builtin_kernel.h"
#include "third_party/neptune/runtime/kernel/kernel.h"

namespace jsv {
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

}  // namespace jsv
