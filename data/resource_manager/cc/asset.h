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

#ifndef THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_ASSET_H_
#define THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_ASSET_H_

#include <string>

#include "absl/status/status.h"  // from @abseil-cpp

namespace neptune::data {

struct Asset {
  const std::string name;
  const std::string path;

  // Primarily for pathways support when assets are not available on each
  // worker. We can remove if this is deemed unnecessary.
  std::string content = "";
  absl::Status LoadContent();

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Asset& asset) {
    absl::Format(&sink, "Asset(name=%s, path=%s, content_size=%zu)", asset.name,
                 asset.path, asset.content.size());
  }
};

}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_ASSET_H_
