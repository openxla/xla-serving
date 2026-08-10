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
