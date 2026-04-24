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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_PERSISTENT_COMPILATION_CACHE_IFRT_PERSISTENT_COMPILATION_CACHE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_PERSISTENT_COMPILATION_CACHE_IFRT_PERSISTENT_COMPILATION_CACHE_H_

#include <memory>

#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "xla/pjrt/pjrt_executable.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/hlo/hlo_program.h"  // from @xla
#include "xla/python/ifrt/program.h"  // from @xla

namespace neptune {

class IfrtPersistentCompilationCache {
 public:
  IfrtPersistentCompilationCache() = default;
  virtual ~IfrtPersistentCompilationCache() = default;

  // The implementation of this API should be thread-safe. It generates a key
  // for looking up the executable in the persistent cache and it will return
  // the LoadedExecutable if hits cache. Otherwise, it will call the `value_fn`
  // to generate and return the LoadedExecutable.
  virtual absl::StatusOr<xla::ifrt::LoadedExecutableRef>
  LookupLoadedExecutableOrCreate(
      std::unique_ptr<xla::ifrt::HloProgram> hlo_program,
      xla::ifrt::DeviceListRef device_list,
      const xla::CompileOptions& xla_compile_options,
      xla::ifrt::Client* client,
      absl::AnyInvocable<absl::StatusOr<xla::ifrt::LoadedExecutableRef>(
          std::unique_ptr<xla::ifrt::Program> program,
          std::unique_ptr<xla::ifrt::CompileOptions> options)>
          value_fn);
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_PERSISTENT_COMPILATION_CACHE_IFRT_PERSISTENT_COMPILATION_CACHE_H_
