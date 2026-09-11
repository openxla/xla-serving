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

#include <dlfcn.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/match.h"  // from @abseil-cpp
#include "absl/strings/str_split.h"  // from @abseil-cpp
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/tensorflow/kernel/builtin_tensorflow_kernel.h"  // from @xla_serving
#include "tensorflow/core/platform/env.h"  // from @org_tensorflow
#include "tensorflow/core/platform/path.h"  // from @org_tensorflow

namespace xla_serving {
namespace {

void LoadSingleSo(const std::string& so_path) {
  // Note: We deliberately use direct dlopen with RTLD_GLOBAL (rather than
  // env->LoadDynamicLibrary which defaults to RTLD_LOCAL) so that symbols
  // exported by custom op libraries (e.g. TF Text, SentencePiece, C++ kernels)
  // are visible in the global process symbol table. This allows cross-library
  // symbol resolution and registration across interdependent shared
  // libraries.
  if (absl::EndsWith(so_path, ".so")) {
    void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle != nullptr) {
      LOG(INFO) << "Loaded TF custom op library via dlopen: " << so_path;
    } else {
      const char* err = dlerror();
      LOG(WARNING) << "Failed to dlopen TF custom op library " << so_path
                   << ": " << (err ? err : "unknown error");
    }
  }
}

void LoadSoFromPath(tensorflow::Env* env, const std::string& current_path) {
  if (!env->FileExists(current_path).ok()) {
    LOG(WARNING) << "TF_CUSTOM_OP_LIBRARY_PATH entry does not exist: "
                 << current_path;
    return;
  }
  if (!env->IsDirectory(current_path).ok()) {
    LoadSingleSo(current_path);
    return;
  }
  std::vector<std::string> children;
  absl::Status status = env->GetChildren(current_path, &children);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to list directory " << current_path << ": "
                 << status.ToString();
    return;
  }
  for (const auto& child : children) {
    std::string child_path = tensorflow::io::JoinPath(current_path, child);
    if (env->IsDirectory(child_path).ok()) {
      LoadSoFromPath(env, child_path);
    } else {
      LoadSingleSo(child_path);
    }
  }
}

// Loads external TF custom op libraries (.so) from
// TF_CUSTOM_OP_LIBRARY_PATH at startup to enable custom TF ops in OSS.
void LoadTFCustomOpsFromEnv() {
  const char* path_env = std::getenv("TF_CUSTOM_OP_LIBRARY_PATH");
  if (!path_env) return;

  tensorflow::Env* env = tensorflow::Env::Default();
  std::vector<std::string> paths =
      absl::StrSplit(path_env, ':', absl::SkipEmpty());
  for (const auto& raw_path : paths) {
    LoadSoFromPath(env, raw_path);
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
}  // namespace xla_serving
