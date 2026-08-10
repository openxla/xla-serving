#ifndef THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_FEATURE_BATCH_CSR_INPUT_H_
#define THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_FEATURE_BATCH_CSR_INPUT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp

namespace neptune::data::cpu_embedding {

// This class represents a batch of samples for a single feature. It is expected
// to be read-only after construction.
class FeatureBatchCsrInput {
 public:
  FeatureBatchCsrInput() = default;
  // Factory method. Validation is optional.
  static absl::StatusOr<FeatureBatchCsrInput> Create(
      std::string name, std::vector<int64_t> embedding_ids,
      std::vector<int32_t> row_pointers, std::vector<float> weights);

  size_t NumSamples() const { return row_pointers_.size() - 1; }

  absl::Span<const int64_t> GetSampleIds(size_t sample_index) const;
  absl::Span<const float> GetSampleWeights(size_t sample_index) const;

  absl::string_view Name() const { return name_; }
  absl::Span<const int64_t> EmbeddingIds() const { return embedding_ids_; }
  absl::Span<const int32_t> RowPointers() const { return row_pointers_; }
  absl::Span<const float> Weights() const { return weights_; }

 private:
  FeatureBatchCsrInput(std::string name, std::vector<int64_t> embedding_ids,
                       std::vector<int32_t> row_pointers,
                       std::vector<float> weights)
      : name_(std::move(name)),
        embedding_ids_(std::move(embedding_ids)),
        row_pointers_(std::move(row_pointers)),
        weights_(std::move(weights)) {}

  std::string name_;
  // Global embedding ids.
  std::vector<int64_t> embedding_ids_;
  // Row pointers defining the sample splits;
  std::vector<int32_t> row_pointers_;
  // Weights for each embedding id.
  std::vector<float> weights_;
};

}  // namespace neptune::data::cpu_embedding

#endif  // THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_FEATURE_BATCH_CSR_INPUT_H_
