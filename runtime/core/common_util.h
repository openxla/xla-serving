#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_UTIL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_UTIL_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
namespace jsv {

absl::Status IsOrbaxModelDirectory(absl::string_view path);

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_UTIL_H_
