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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_METADATA_UTIL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_METADATA_UTIL_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json

namespace xla_serving {

/**
 * Extracts *all* parameter names from a Orbax Checkpoint JSON metadata.
 *
 * This function parses the input JSON, extracts the "tree_metadata" and
 * "key_metadata" fields,  converts each key in the metadata to a parameter name
 * by joining the keys with '.'
 */
absl::StatusOr<std::vector<std::string>> GetParamNames(
    const nlohmann::ordered_json& metadata_json);

/**
 * Extracts parameter names from a Orbax Checkpoint JSON metadata, filtered by
 * PyTree sub_tree traversal path.
 *
 * The `sub_tree` here represents a PyTree top-down path (e.g. `{"foo",
 * "bar"}` means the sub-tree `/foo/bar` from the root). All params that are
 * descendants of the sub_tree are returned.
 *
 * The ordering of the returned param names is the same as the order of keys in
 * the input nlohmann::ordered_json metadata. Here the nlohmann::ordered_json is
 * used to preserve the order of keys in the _METADATA json file.
 */
absl::StatusOr<std::vector<std::string>> GetParamNamesForSubTree(
    const nlohmann::ordered_json& metadata_json,
    std::vector<std::string>& sub_tree);

/**
 * Reads the "_METADATA" file from the given checkpoint path, parses it as JSON,
 * and extracts all weight names.
 *
 * This function currently assumes that the _METADATA file only contains weights
 * relevant for inference. Future improvements may involve filtering based on an
 * OCP view proto to exclude unrelated keys like optimizer states.
 */
absl::StatusOr<std::vector<std::string>> GetAllWeightNamesFromCheckpoint(
    absl::string_view checkpoint_path);

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_METADATA_UTIL_H_
