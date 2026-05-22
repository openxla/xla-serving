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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_IFRT_CLIENT_CREATOR_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_IFRT_CLIENT_CREATOR_H_

#include <memory>
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "xla/python/ifrt/client.h"  // from @xla

namespace neptune {
namespace serving {

absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> GetIfrtClient(
    absl::string_view pjrt_platform);

}  // namespace serving
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_IFRT_CLIENT_CREATOR_H_
