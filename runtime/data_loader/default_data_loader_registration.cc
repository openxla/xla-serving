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

#include <memory>
#include <utility>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @xla_serving
#include "runtime/data_loader/data_loader.h"  // from @xla_serving
#include "runtime/data_loader/default_data_loader.h"  // from @xla_serving

namespace xla_serving {
namespace {

bool kUnused = []() {
  const auto s =
      DataLoaderFactoryRegistry::GetGlobalRegistry().RegisterDataLoaderFactory(
          xla_saved_model_manifest::LoaderType::LOADER_TYPE_DEFAULT,
          [](DataLoader::CreateOptions create_options)
              -> absl::StatusOr<std::unique_ptr<DataLoader>> {
            return DefaultDataLoader::Create(std::move(create_options));
          });
  LOG_IF(FATAL, !s.ok()) << s;
  return true;
}();

}  // namespace
}  // namespace xla_serving
