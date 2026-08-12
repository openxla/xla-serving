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

#ifndef THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_FEATURE_SPEC_H_
#define THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_FEATURE_SPEC_H_

#include <cstdint>
#include <string>

#include "absl/strings/str_format.h"  // from @abseil-cpp

namespace neptune::data::cpu_embedding {

struct FeatureSpec {
  std::string name;
  std::string table_name;
  int32_t max_sequence_length = 0;
};

template <typename Sink>
void AbslStringify(Sink& sink, const FeatureSpec& spec) {
  absl::Format(&sink, "{name: %s, table_name: %s, max_sequence_length: %d}",
               spec.name, spec.table_name, spec.max_sequence_length);
}

}  // namespace neptune::data::cpu_embedding

#endif  // THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_FEATURE_SPEC_H_
