#include "third_party/neptune/runtime/core/common_util.h"

#include <string>

#include "file/base/path.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/neptune/runtime/core/constants.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/errors.h"

namespace jsv {

absl::Status IsOrbaxModelDirectory(absl::string_view path) {
  const std::string orbax_model_manifest_version_path =
      ::file::JoinPath(path, kOrbaxModelManifestVersionTxt);
  tsl::Env* env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(env->FileExists(orbax_model_manifest_version_path));
  return absl::OkStatus();
}

}  // namespace jsv
