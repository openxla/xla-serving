#include "data/resource_manager/cc/asset.h"  // from @neptune

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "xla/tsl/platform/env.h"  // from @xla

namespace neptune::data {

absl::Status Asset::LoadContent() {
  return tsl::ReadFileToString(tsl::Env::Default(), path, &content);
}

}  // namespace neptune::data
