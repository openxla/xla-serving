#include "runtime/core/common_util.h"  // from @neptune

#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "runtime/core/constants.h"  // from @neptune
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace jsv {

absl::Status IsOrbaxModelDirectory(absl::string_view path) {
  const std::string orbax_model_manifest_version_path =
      tsl::io::JoinPath(path, kOrbaxModelManifestVersionTxt);
  tsl::Env* env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(env->FileExists(orbax_model_manifest_version_path));
  return absl::OkStatus();
}

}  // namespace jsv
