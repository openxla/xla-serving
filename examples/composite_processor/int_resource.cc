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

#include "examples/composite_processor/int_resource.h"  // from @xla_serving

#include <fstream>
#include <sstream>
#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/numbers.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "data/resource_manager/cc/resource.h"  // from @xla_serving

namespace neptune_examples {
namespace composite {

absl::Status IntResource::LoadImpl() {
  if (assets().empty() || assets()[0].path.empty()) {
    return absl::InvalidArgumentError(
        "IntResource requires an asset with a non-empty path.");
  }
  const std::string& path = assets()[0].path;

  std::ifstream file(path);
  if (!file) {
    return absl::InvalidArgumentError(
        absl::StrCat("No asset found at specified path: ", path));
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  std::string content = ss.str();

  if (!absl::SimpleAtoi(content, &value_)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not parse asset content as integer: ", content));
  }
  return absl::OkStatus();
}

ND_REGISTER_RESOURCE(IntResource, "ExamplesIntResource");

}  // namespace composite
}  // namespace neptune_examples
