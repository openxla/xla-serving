// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdlib>
#include <string>
#include <vector>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/strings/match.h"  // from @abseil-cpp
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/tensorflow/kernel/builtin_tensorflow_kernel.h"  // from @neptune
#include "tensorflow/core/platform/env.h"  // from @org_tensorflow
#include "tensorflow/core/platform/path.h"  // from @org_tensorflow

namespace neptune {
namespace {

// Loads external TF custom op libraries (.so) from
// TF_CUSTOM_OP_LIBRARY_PATH at startup to enable custom TF ops in OSS.
void LoadTFCustomOpsFromEnv() {
  const char* path_env = std::getenv("TF_CUSTOM_OP_LIBRARY_PATH");
  if (!path_env) return;

  tensorflow::Env* env = tensorflow::Env::Default();
  if (!env->FileExists(path_env).ok()) {
    LOG(WARNING) << "TF_CUSTOM_OP_LIBRARY_PATH does not exist: " << path_env;
    return;
  }

  auto load_so = [env](const std::string& so_path) {
    if (absl::EndsWith(so_path, ".so")) {
      void* handle = nullptr;
      absl::Status status = env->LoadDynamicLibrary(so_path.c_str(), &handle);
      if (status.ok()) {
        LOG(INFO) << "Loaded TF custom op library: " << so_path;
      } else {
        LOG(WARNING) << "Failed to load TF custom op library " << so_path
                     << ": " << status.ToString();
      }
    }
  };

  if (!env->IsDirectory(path_env).ok()) {
    load_so(path_env);
    return;
  }

  std::vector<std::string> children;
  absl::Status status = env->GetChildren(path_env, &children);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to list directory " << path_env << ": "
                 << status.ToString();
    return;
  }
  for (const auto& child : children) {
    load_so(tensorflow::io::JoinPath(path_env, child));
  }
}

// TODO: Find a better way to register the kernels
void RegisterBuiltinTensorflowKernelsOrDie() {
  CHECK_OK(RegisterBuiltinTensorflowKernels(GetGlobalKernelRegistry()));
}

// Register the builtin kernels at link time.
static const bool kUnused = []() {
  RegisterBuiltinTensorflowKernelsOrDie();
  LoadTFCustomOpsFromEnv();
  return true;
}();

}  // namespace
}  // namespace neptune
