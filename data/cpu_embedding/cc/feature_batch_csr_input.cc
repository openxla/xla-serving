#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @neptune

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp

namespace neptune::data::cpu_embedding {
namespace {

absl::Status Validate(absl::Span<const int64_t> embedding_ids,
                      absl::Span<const int32_t> row_pointers,
                      absl::Span<const float> weights) {
  if (row_pointers.size() < 2) {
    return absl::InvalidArgumentError(
        "Row pointers must have at least 2 elements.");
  }
  if (row_pointers.front() != 0 ||
      embedding_ids.size() != row_pointers.back()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Embedding ids size (%d) does not match last element "
                        "of row pointers (%d)",
                        embedding_ids.size(), row_pointers.back()));
  }
  if (!std::is_sorted(row_pointers.begin(), row_pointers.end())) {
    return absl::InvalidArgumentError("Row pointers are not sorted.");
  }

  if (!weights.empty() && weights.size() != row_pointers.back()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Weights size (%d) does not match expected size (%d)",
                        weights.size(), row_pointers.back()));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<FeatureBatchCsrInput> FeatureBatchCsrInput::Create(
    std::string name, std::vector<int64_t> embedding_ids,
    std::vector<int32_t> row_pointers, std::vector<float> weights) {
  RETURN_IF_ERROR(Validate(embedding_ids, row_pointers, weights));
  return FeatureBatchCsrInput(std::move(name), std::move(embedding_ids),
                              std::move(row_pointers), std::move(weights));
}

absl::Span<const int64_t> FeatureBatchCsrInput::GetSampleIds(
    size_t sample_index) const {
  return absl::MakeConstSpan(embedding_ids_)
      .subspan(row_pointers_[sample_index],
               row_pointers_[sample_index + 1] - row_pointers_[sample_index]);
}

absl::Span<const float> FeatureBatchCsrInput::GetSampleWeights(
    size_t sample_index) const {
  if (weights_.empty()) {
    return absl::Span<const float>();
  }
  return absl::MakeConstSpan(weights_).subspan(
      row_pointers_[sample_index],
      row_pointers_[sample_index + 1] - row_pointers_[sample_index]);
}

}  // namespace neptune::data::cpu_embedding
