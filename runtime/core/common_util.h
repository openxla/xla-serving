#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_UTIL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_UTIL_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
namespace jsv {

absl::Status IsOrbaxModelDirectory(absl::string_view path);

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_UTIL_H_
