#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_UTIL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_UTIL_H_

#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/string_view.h"
namespace jsv {

absl::Status IsOrbaxModelDirectory(absl::string_view path);

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_UTIL_H_
