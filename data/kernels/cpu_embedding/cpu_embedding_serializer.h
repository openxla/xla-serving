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

#ifndef THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_SERIALIZER_H_
#define THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_SERIALIZER_H_

#include <string>
#include <vector>

#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/resource_manager/cc/asset.h"  // from @neptune
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp

namespace neptune::data {
namespace composite {

absl::StatusOr<std::vector<Asset>> SerializeCpuEmbedding(
    std::string_view dir_path,
    const absl::flat_hash_map<std::string, cpu_embedding::Array>& tables,
    std::string_view manifest_proto_text,
    const absl::flat_hash_map<std::string, std::string>& custom_combiners = {});

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_SERIALIZER_H_
